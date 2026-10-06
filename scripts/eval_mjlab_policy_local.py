"""Evaluate an mjlab-trained policy on the CPU env (default) or on mjlab.

The basic-suite mjlab tasks are twins of the CPU envs with the same ``env_id``
(same observation vector, action mapping and control timing), so a checkpoint
from ``scripts/train_mjlab.py`` drives the CPU env unchanged.

Examples::

    # CPU rollouts (plain gymnasium env), latest checkpoint of a run
    python scripts/eval_mjlab_policy.py myoElbowPose1D6MRandom-v0 \\
        --checkpoint logs/rsl_rl/myo_elbow_pose/2026-09-23_12-23-29

    # Same policy on the mjlab twin, and a CPU video
    python scripts/eval_mjlab_policy.py myoElbowPose1D6MRandom-v0 --checkpoint RUN --backend mjlab
    python scripts/eval_mjlab_policy.py myoElbowPose1D6MRandom-v0 --checkpoint RUN --video out.mp4

    # Parallel envs side by side in one video (same flags on --backend cpu / mjlab)
    python scripts/eval_mjlab_policy.py myoElbowPose1D6MRandom-v0 --checkpoint RUN \\
        --num-cols 4 --num-rows 2 --episodes-per-env 3 --video grid.mp4 \\
        --width 1280 --height 720

Every run prints the success rate (share of episodes whose final step is
"solved"; CPU: ``info["solved"]``, mjlab: the ``Episode_Metrics/success``
metric that every task twin logs during training too).

    # Video from a model camera (name or id; -1 = free camera)
    python scripts/eval_mjlab_policy.py myoElbowPose1D6MRandom-v0 --checkpoint RUN \\
        --video out.mp4 --camera side_view --width 1280 --height 720
"""

from __future__ import annotations

import copy
import functools
import hashlib
import inspect
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import tyro


@dataclass(frozen=True)
class EvalConfig:
    env_id: tyro.conf.Positional[str]
    """Task id (shared by the CPU env and its mjlab twin)."""
    checkpoint: Path
    """``model_*.pt`` file, or a run directory (latest checkpoint is used)."""
    backend: Literal["cpu", "mjlab"] = "cpu"
    """Roll out in the CPU gymnasium env or in the mjlab env."""
    episodes: int = 10
    """Without ``--num-cols``/``--num-rows``: CPU runs this many episodes on one
    env, mjlab runs this many parallel envs (square-ish grid, one episode each)."""
    episodes_per_env: int = 1
    """Episodes recorded (and scored) for every parallel env when a grid is set
    (or on mjlab); envs reset between episodes and the video runs until all
    have finished."""
    logo: bool = False
    """Video: fade the MyoSuite logo in, big and centred, over the last
    ``LOGO_FADE_SECONDS`` seconds. Default: no logo."""
    success_hold: float = 0.0
    """With ``--end-on-success``: keep each agent running this many seconds after its
    episode succeeded (the success is scored at once), then reset it. Default 0: reset
    immediately. Policies that solve within a few steps otherwise show their targets
    jumping many times per second, which looks like fast-forward."""
    end_on_success: bool = False
    """End every episode the moment its success/solved criterion is met (the env
    then resets and starts its next episode), so envs switch episodes
    asynchronously. Default: episodes run until they time out or terminate.
    With ``--camera-sweep`` and ``--video`` the episode count is not capped: envs keep
    starting new episodes until the sweep has been rendered completely (so more than
    ``--episodes-per-env`` are scored; ``--after-last-episode`` then has no effect)."""
    after_last_episode: Literal["run", "freeze", "stop"] = "run"
    """What happens to an agent that has finished all its episodes while others are
    still going: ``run`` keeps acting (unscored), ``freeze`` keeps it still in its
    final pose (video only on mjlab; CPU envs are simply not stepped again), ``stop``
    ends the whole evaluation as soon as the first agent has finished."""
    first_policy: int | Literal["last"] | None = None
    """Training iteration (``model_<iter>.pt``) of the policy replaying every env's
    FIRST episode. Episodes in between load stored checkpoints linearly interpolated
    over the list of available checkpoints (by position, not by iteration number)
    up to ``--last-policy`` for the last episode. Default and ``last``: the newest
    checkpoint, so every episode uses it (the previous behaviour)."""
    last_policy: int | Literal["last"] | None = None
    """Training iteration of the policy replaying every env's LAST episode
    (see ``--first-policy``; default and ``last``: the newest checkpoint)."""
    pairs: tuple[str, ...] = ()
    """``ENV_ID=CHECKPOINT`` pairs (checkpoint file or run directory). When given,
    every grid cell (agent) samples one pair uniformly at random (seeded by
    ``--seed``) and runs that env with that run's policies (``--first-policy`` /
    ``--last-policy`` apply within each run). CPU backend: the envs may differ but
    must share one body model (e.g. Sarc/Fati variants); mjlab: any envs, each env id
    gets its cells' envs, drawn together in the one grid. The agents are arranged so
    that cells of one body-model family do not touch, in any direction, as far as their
    numbers allow (the number of remaining touching pairs is printed)."""
    fullbody_ref: str = "hf://amathislab/mm-10m-2"
    """MuscleMimic full-body checkpoint (``hf://owner/repo`` or a local run directory)
    of the ``fullbody`` pair. ``--pairs`` accepts the entry ``fullbody``: such cells
    show the pretrained full-body policy walking (a recorded rollout on its reference
    motion, looped seamlessly and played in place) next to the other agents. It is
    not evaluated (no episodes, no success rate). With it in the grid, every other
    body (arm, torso, leg, hand) is seated where the full body has it (same landmark body:
    sacrum, pelvis, torso or right shoulder), so all share one body height. Needs JAX and the
    ``musclemimic`` extras (``pip install -e '.[musclemimic]'``)."""
    fullbody_motion: str = "KIT/314/walking_medium09_poses"
    """Reference motion of the ``fullbody`` pair."""
    fullbody_yaw: float = 0.0
    """Heading (degrees about the vertical axis) the ``fullbody`` agents walk in place
    with."""
    balance_models: bool = False
    """With ``--pairs``: sample the body-model families (all hand tasks, all torso
    tasks, ...) with equal probability, then a pair within the family, instead of every
    pair with equal probability. A family is a set of envs with the same MuJoCo model
    size (``nq``, ``nbody``, ``nu``)."""
    family_weights: tuple[str, ...] = ()
    """With ``--balance-models``: ``ENV_ID=WEIGHT`` entries setting the relative share of
    a body-model family (``ENV_ID`` is any env of that family, which must be one of the
    ``--pairs``; a family is named once). Families not listed get weight 1, so e.g.
    ``myoTorsoPoseFixed-v0=2`` draws the torso family twice as often as an unlisted one."""
    stochastic: bool = False
    """Sample actions from the policy's Gaussian (the learned std, as during training)
    instead of using the mean action. The training success rate is measured with
    sampled actions; for muscle tasks the mean action can behave very differently
    (even fail completely), so use this to reproduce the logged success rate."""
    show_tendons: bool = False
    """Draw the muscle tendons (adds ~150 geoms per env for the arm model)."""
    floor: bool = True
    """Video: draw a light checker-grid floor with a fading white horizon under all
    envs (independent of ``--show-scene``)."""
    shadows: bool = False
    """Video: draw shadows on the floor. Off by default: MuJoCo's shadow map covers
    only a small region around the camera target, so on large grids the shadows break
    up and the floor shows stippled shadow-acne patches."""
    preview: bool = False
    """Camera debugging: render ONE frame (with ``--camera-sweep`` at the sweep's
    progress ``--preview-at``, default the END pose) of the fresh grid, save it as a PNG next to ``--video`` and exit; no episodes
    are run. Needs ``--video``. Use a small ``--width``/``--height`` while tuning."""
    preview_at: float = 1.0
    """``--preview`` with ``--camera-sweep``: progress of the sweep (0 = start pose,
    1 = end pose) at which the frame is rendered."""
    camera_sweep: bool = False
    """Video: slowly move the camera from a low oblique view up and round to the
    front, revealing more and more of the grid. ``--azimuth``/``--elevation``/
    ``--distance`` set the END of the sweep (unless overridden by ``SWEEP_END``); the
    start is ``SWEEP_START``."""
    show_scene: bool = False
    """Also draw everything but the body and targets (floor/world geoms).
    Default: the body + targets only."""
    seed: int = 0
    """Seed of the first episode (CPU) / the mjlab env."""
    video: Path | None = None
    """Write an MP4 of the rollouts (offscreen rendering); parallel envs
    (``--num-cols`` x ``--num-rows``) share one scene, on both backends."""
    num_cols: int | None = None
    """Parallel envs along the horizontal axis of the grid (both backends).
    Setting ``--num-cols`` and/or ``--num-rows`` runs ``cols * rows`` envs
    (a missing one defaults to 1) instead of the ``--episodes`` default."""
    num_rows: int | None = None
    """Parallel envs along the vertical (depth) axis of the grid (both backends)."""
    env_spacing: float | None = None
    """mjlab video only: distance between neighbouring envs in metres
    (default: 1.36 x the size of the visible robot)."""
    camera: str = "-1"
    """Video camera: a camera name from the model, or an id (``-1`` = free
    camera). Only used with a single env; grids always use the free camera."""
    width: int = 640
    """Video frame width."""
    height: int = 480
    """Video frame height."""
    distance: float | None = None
    """Free camera only: distance to the look-at point (default: MuJoCo's; the
    mjlab grid video scales it to fit the grid)."""
    azimuth: float | None = None
    """Free camera only: azimuth in degrees."""
    elevation: float | None = None
    """Free camera only: elevation in degrees."""
    lookat: tuple[float, float, float] | None = None
    """Free camera only: look-at point in world coordinates (mjlab grid video:
    relative to the grid centre)."""


def _resolve_checkpoint(path: Path) -> Path:
    """A checkpoint file, or the newest ``model_<iter>.pt`` in a run directory."""
    if path.is_file():
        return path
    ckpts = sorted(
        (c for c in path.glob("model_*.pt") if re.fullmatch(r"model_\d+", c.stem)),
        key=lambda c: int(c.stem.split("_")[-1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No model_<iter>.pt in {path}")
    return ckpts[-1]


def _video_camera(
    cfg: EvalConfig, model: mujoco.MjModel
) -> int | str | mujoco.MjvCamera:
    """Camera for ``mujoco.Renderer.update_scene``: model camera or tuned free camera."""
    camera = int(cfg.camera) if cfg.camera.lstrip("-").isdigit() else cfg.camera
    if camera != -1:
        return camera
    free = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, free)
    for attr in ("distance", "azimuth", "elevation", "lookat"):
        value = getattr(cfg, attr)
        if value is not None:
            setattr(free, attr, value)
    return free


def _grid_shape(cfg: EvalConfig) -> tuple[int, int]:
    """``(cols, rows)`` of the parallel-env grid.

    Explicit ``--num-cols``/``--num-rows`` win. Otherwise mjlab runs a
    square-ish grid of ``--episodes`` envs and CPU runs a single env.
    """
    if cfg.num_cols is None and cfg.num_rows is None:
        if cfg.backend == "cpu":
            return 1, 1
        cols = int(np.ceil(np.sqrt(cfg.episodes)))
        return cols, int(np.ceil(cfg.episodes / cols))
    return cfg.num_cols or 1, cfg.num_rows or 1


def _episodes_per_env(cfg: EvalConfig) -> int:
    """CPU without a grid runs ``--episodes`` episodes on its single env."""
    explicit_grid = cfg.num_cols is not None or cfg.num_rows is not None
    if cfg.backend == "cpu" and not explicit_grid:
        return cfg.episodes
    return cfg.episodes_per_env


_HIDDEN_GROUP = 5

# Appearance of the target markers (sites named ``*_target``) in videos. Edit
# here; None keeps the model's value. Other sites are not drawn.
TARGET_SITE_RGBA: tuple[float, float, float, float] | None = (0.15, 0.15, 1.0, 0.9)
# Radius: a number, a ``(low, high)`` range (each grid cell samples its own radius
# uniformly, once per video, seeded by ``--seed``; visuals only), or None.
TARGET_SITE_RADIUS: float | tuple[float, float] | None = (0.02, 0.08)
# Every target marker also gets a marker on the body site that has to reach it
# (``wrist`` for ``wrist_target``, ``IFtip`` for ``IFtip_target``) in its own colour,
# so the tracking error is visible.
TRACKER_SITE_RGBA: tuple[float, float, float, float] = (1.0, 0.45, 0.0, 0.9)
# How the target markers are sized (edit here):
#   "threshold": the task's success threshold, so a body site inside its marker counts
#                as solved; ``MARKER_RADIUS_SCALE`` scales it. Falls back to
#                TARGET_SITE_RADIUS if the threshold is unknown.
#   "fixed":     TARGET_SITE_RADIUS for every task, scaled down for hand envs (env ids
#                containing ``Hand``) by ``HAND_MARKER_SCALE``.
# These position markers are for reach tasks. Pose tasks are judged in joint space (the
# norm of all joint errors), so they get joint markers instead, see ``JOINT_MARKER_*``.
# Reach tasks are solved when the distance over all k tip sites is below
# ``REACH_SOLVED_DIST * k`` (``multi_site_reach_reward``).
MARKER_SIZES: Literal["threshold", "fixed"] = "threshold"
# The same choice for hand envs (env ids containing ``Hand``): the threshold-sized
# markers are as large as the fingers there, so they default to the fixed sizes.
HAND_MARKER_SIZES: Literal["threshold", "fixed"] = "fixed"
REACH_SOLVED_DIST = 0.0125
MARKER_RADIUS_SCALE = 1.0  # threshold mode: 1.0 shows exactly the success threshold
HAND_MARKER_SCALE = 0.25  # fixed mode, additionally for hand envs
# Pose tasks: a sphere at the anchor of every joint the task controls, coloured by the
# joint's error |target - angle| (rad) relative to ``pose_thd`` (the task is solved when
# the norm of all joint errors is below it): dark green at 0 to yellow at ``pose_thd``
# (inside the threshold), red above it. The radius is a fraction of the model's size ...
# Velocity-tracking (locomotion) envs draw an arrow of the target planar velocity above
# the root instead of target markers: length = VELOCITY_ARROW_LENGTH * speed (m per m/s).
VELOCITY_ARROW_RGBA = (0.15, 0.15, 1.0, 0.9)
VELOCITY_ARROW_LENGTH = 0.6
VELOCITY_ARROW_WIDTH = 0.03
VELOCITY_ARROW_HEIGHT = 1.0  # base of the arrow above the root (pelvis)
JOINT_MARKER_FRACTION = 0.02  # marker radius as a fraction of the model's size (m) ...
JOINT_MARKER_RANGE = (0.004, 0.03)  # ... limited to this range (m)
JOINT_MARKER_MARGIN = (
    1.25  # ... and at least this much larger than visible joint spheres
)
HAND_JOINT_MARKER_SCALE = 0.5  # joint markers of hand envs (ids containing ``Hand``)
JOINT_INSIDE_RGB = ((0.0, 0.45, 0.1), (1.0, 0.9, 0.1))  # error 0 ... pose_thd
JOINT_OUTSIDE_RGB = (0.95, 0.1, 0.1)  # error > pose_thd


# Camera sweep (``--camera-sweep``). Start: low, oblique, close (``distance_scale``
# multiplies the end distance). End: ``None`` = the regular free camera (front view,
# elevated; ``--azimuth`` / ``--elevation`` / ``--distance`` apply); a number
# overrides it, and ``distance_scale`` multiplies the free camera's distance.
SWEEP_START = {"azimuth": 35.0, "elevation": -6.0, "distance_scale": 0.7}
# SWEEP_END = {"azimuth": None, "elevation": None, "distance_scale": 1.0}
SWEEP_END = {
    "azimuth": None,
    "elevation": -65,
    "distance_scale": 2.0,
}  # default terminal values for myoArmReachRandom-v0 cam: 90, -45, 1.0
DEFAULT_AZIMUTH, DEFAULT_ELEVATION = 90.0, -35.0

# Fog (floor fading into the horizon), as multiples of the camera distance. The
# Renderer bakes in fog colour, distances and extent when it is created. MuJoCo
# scales fog distances (and clip planes) by ``model.stat.extent``, which is huge
# for models that include the scene (arm reach: 25 m), so GridRenderer overrides
# the extent with a camera-based value instead of using the model's.
FOG_RGBA = (1.0, 1.0, 1.0, 1.0)
FOG_START, FOG_END, HORIZON = 1.0, 4.0, 5.0

# Floor look (RGBA): light plate with darker grid lines every FLOOR_CELL metres.
FLOOR_RGBA = (0.94, 0.94, 0.94, 1.0)
FLOOR_LINE_RGBA = (0.45, 0.45, 0.45, 1.0)
FLOOR_CELL = 0.5
POSE_SAMPLES = 200  # random joint poses used to find the lowest point the body reaches


def _target_site_ids(model: mujoco.MjModel) -> list[int]:
    """Ids of the target marker sites (``*_target``, ignoring an entity prefix)."""
    names = [model.site(i).name for i in range(model.nsite)]
    return [i for i, n in enumerate(names) if n.split("/")[-1].endswith("_target")]


def _tracker_site_ids(model: mujoco.MjModel) -> list[int]:
    """Body sites that track a target marker (``wrist`` for ``wrist_target``)."""
    names = (
        model.site(i).name.removesuffix("_target") for i in _target_site_ids(model)
    )
    ids = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in names)
    return [i for i in ids if i >= 0]


def _joint_error_rgba(error: float, pose_thd: float) -> np.ndarray:
    """Dark green -> yellow for errors up to ``pose_thd``, red beyond it."""
    if error > pose_thd:
        color = np.array(JOINT_OUTSIDE_RGB)
    else:
        t = error / pose_thd
        color = (1 - t) * np.array(JOINT_INSIDE_RGB[0]) + t * np.array(
            JOINT_INSIDE_RGB[1]
        )
    return np.append(color, 0.95).astype(np.float32)


def _joint_marker_radii(
    model: mujoco.MjModel, opt: mujoco.MjvOption, base: float
) -> dict[int, float]:
    """Marker radius per joint: *base*, or larger than the visible primitive geoms
    (e.g. the joint spheres of the finger models) of the joint's body so the marker
    is not hidden inside them."""
    visible = np.array(opt.geomgroup, dtype=bool)[model.geom_group]
    primitive = visible & (model.geom_type != mujoco.mjtGeom.mjGEOM_MESH)
    radii = {}
    for joint in range(model.njnt):
        sizes = model.geom_size[
            primitive & (model.geom_bodyid == model.jnt_bodyid[joint]), 0
        ]
        radii[joint] = max(base, JOINT_MARKER_MARGIN * float(sizes.max(initial=0.0)))
    return radii


def _marker_size(
    threshold_radius: float | None, env_id: str
) -> tuple[float | None, float]:
    """``(radius, scale)`` of the target markers for the ``MARKER_SIZES`` mode
    (``HAND_MARKER_SIZES`` for hand envs).

    A ``None`` radius means ``TARGET_SITE_RADIUS``; the marker radius is
    ``radius * scale``.
    """
    mode = HAND_MARKER_SIZES if "Hand" in env_id else MARKER_SIZES
    if mode == "threshold":
        if threshold_radius is None:
            return None, 1.0
        return threshold_radius * MARKER_RADIUS_SCALE, 1.0
    return None, HAND_MARKER_SCALE if "Hand" in env_id else 1.0


def _render_option(
    model: mujoco.MjModel,
    show_scene: bool,
    show_tendons: bool,
    pose_task: bool = False,
    marker_radius: float | None = None,
    marker_scale: float = 1.0,
) -> mujoco.MjvOption:
    """Render option drawing the skeleton and target markers unless *show_scene*.

    World-body geoms and geoms of the static mocap wrapper mjlab attaches
    fixed-base models to (floor, scene mesh, logo), plus non-mesh geoms
    (tendon-wrapping primitives, contact shapes), are moved to a hidden geom
    group, leaving the bone meshes. Sites other than the target markers are
    hidden the same way and the markers are styled with ``TARGET_SITE_*``.
    Tendons follow *show_tendons*. Only visual model metadata is touched;
    physics is unaffected.
    """
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = int(show_tendons)
    if show_scene:
        return opt
    hidden = (model.geom_bodyid == 0) | (model.body_mocapid[model.geom_bodyid] >= 0)
    is_mesh = model.geom_type == mujoco.mjtGeom.mjGEOM_MESH
    if (is_mesh & ~hidden).any():  # bone meshes: hide the primitive helper shapes
        hidden |= ~is_mesh
    # else: primitive-only models (e.g. the finger) are drawn with their primitives
    model.geom_group[hidden] = _HIDDEN_GROUP
    opt.geomgroup[_HIDDEN_GROUP] = 0
    targets = (
        [] if pose_task else _target_site_ids(model)
    )  # pose: joint markers instead
    model.site_group[:] = _HIDDEN_GROUP
    model.site_group[targets] = 0
    opt.sitegroup[:] = 0
    opt.sitegroup[0] = 1
    radius = TARGET_SITE_RADIUS if marker_radius is None else marker_radius
    if isinstance(radius, float | int):
        radius *= marker_scale
    for i in targets:
        if TARGET_SITE_RGBA is not None:
            model.site_rgba[i] = TARGET_SITE_RGBA
        if isinstance(radius, float | int):
            model.site_size[i, 0] = radius
    for i in (
        [] if pose_task else _tracker_site_ids(model)
    ):  # sites that reach the targets
        model.site_group[i] = 0
        model.site_rgba[i] = TRACKER_SITE_RGBA
        if isinstance(radius, float | int):
            model.site_size[i, 0] = radius
    return opt


def _visible_bounds(
    model: mujoco.MjModel, opt: mujoco.MjvOption
) -> tuple[np.ndarray, float, float]:
    """Centre and largest extent of the visible geoms (default pose) and the lowest z.

    The lowest z also covers poses sampled within the joint limits (fixed seed), so a
    body that reaches below its default pose (a curling finger) stays above the floor.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    visible = np.array(opt.geomgroup, dtype=bool)[model.geom_group]
    visible &= model.geom_rgba[:, 3] > 0
    radius = model.geom_rbound[visible, None]
    lo = (data.geom_xpos[visible] - radius).min(axis=0)
    hi = (data.geom_xpos[visible] + radius).max(axis=0)
    lowest = float(lo[2])
    limited = [
        j
        for j in range(model.njnt)
        if model.jnt_limited[j]
        and model.jnt_type[j]
        in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    ]
    rng = np.random.default_rng(0)
    for _ in range(POSE_SAMPLES if limited else 0):
        data.qpos[:] = model.qpos0
        for j in limited:
            data.qpos[model.jnt_qposadr[j]] = rng.uniform(*model.jnt_range[j])
        mujoco.mj_kinematics(model, data)
        lowest = min(
            lowest, float((data.geom_xpos[visible][:, 2] - radius[:, 0]).min())
        )
    return (lo + hi) / 2.0, float((hi - lo).max()), lowest


def _grid_offsets(cols: int, rows: int, spacing: float) -> np.ndarray:
    """World offsets ``(cols * rows, 3)`` of a centred grid, x = column, y = row."""
    n = cols * rows
    idx = np.arange(n)
    xy = np.stack([idx % cols, idx // cols], axis=1).astype(float)
    xy -= (xy.max(axis=0)) / 2.0
    return np.concatenate([xy * spacing, np.zeros((n, 1))], axis=1)


EnvState = tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], dict[int, float]]
"""``(qpos, qvel, {target site id: xyz}, {joint id: |error|})`` of one env; the joint
errors are only given for pose tasks."""


def _add_floor(
    scene: mujoco.MjvScene, z: float, half_extent: float, horizon: float
) -> None:
    """Append a light floor plate with a grid, white sky walls and a shadow light.

    The plate and walls are far beyond the fog end, so they fade into a white
    horizon and sky. Grid lines only cover ``half_extent`` around the origin:
    thin coplanar lines far away z-fight (speckle), and are fogged out anyway.
    """
    eye = np.eye(3).ravel()

    def add(size, pos, rgba) -> None:
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_BOX,
            np.asarray(size, dtype=float),
            np.asarray(pos, dtype=float),
            eye,
            np.asarray(rgba, dtype=np.float32),
        )
        scene.ngeom += 1

    add([horizon, horizon, 0.005], [0, 0, z - 0.005], FLOOR_RGBA)
    # Sky: four solid walls (a hollow dome would be back-face culled from inside).
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        along_x = sy != 0
        add(
            [horizon if along_x else 5.0, 5.0 if along_x else horizon, horizon],
            [sx * horizon, sy * horizon, z + horizon - 1.0],
            FLOOR_RGBA,
        )
    for c in np.arange(-half_extent, half_extent + 1e-6, FLOOR_CELL):
        add([half_extent, 0.004, 0.0015], [0, c, z], FLOOR_LINE_RGBA)
        add([0.004, half_extent, 0.0015], [c, 0, z], FLOOR_LINE_RGBA)
    light = scene.lights[scene.nlight]
    light.headlight = 0
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    light.castshadow = 1
    light.pos[:] = (0.0, 0.0, 6.0)
    light.dir[:] = (0.3, 0.4, -1.0)
    light.diffuse[:] = (0.6, 0.6, 0.6)
    light.specular[:] = (0.1, 0.1, 0.1)
    light.ambient[:] = (0.0, 0.0, 0.0)
    scene.nlight += 1


class GridRenderer:
    """Render parallel envs into one scene, laid out on a grid (either backend).

    Fixed-base robots would overlap (mjlab env origins only move floating
    bases, and CPU envs are separate simulations), so the scene is composed
    here: each env's state is copied into a host ``MjData`` and its geoms are
    translated by the env's grid offset.

    Args:
        model: Host model used for drawing (visual metadata is edited in place).
        state_fn: ``state_fn(env_id) -> EnvState`` for the current step.
        cfg: Video options (size, camera, spacing, what to draw).
        pose_task: Draw joint error markers instead of position markers.
        marker_radius: Radius of the target markers (their task's success threshold).
        pose_thd: Joint error (rad) shown in full red for a pose task.
        cells: Grid cells this renderer draws (default: all); ``state_fn`` gets the
            cell index.
        show_targets: Draw the ``*_target`` markers (and their trackers); off for tasks
            that have no target position (locomotion).
        velocity_fn: ``velocity_fn(cell) -> (vx, vy) | None``, target planar velocity
            drawn as an arrow above the root.
        shift: Translation of this body in the scene, overriding the alignment onto
            ``shared`` (centre and floor).
        shared: Centre, size and floor of a mixed-body grid (see ``SharedView``): the
            camera, spacing and floor follow it and this body is shifted onto it.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        state_fn,
        cfg: EvalConfig,
        total_steps: int = 1,
        pose_task: bool = False,
        marker_radius: float | None = None,
        pose_thd: float | None = None,
        cells: list[int] | None = None,
        shared: SharedView | None = None,
        show_targets: bool = True,
        velocity_fn=None,
        shift: np.ndarray | None = None,
    ) -> None:
        cols, rows = _grid_shape(cfg)
        n_envs = cols * rows
        self._cells = list(range(n_envs)) if cells is None else list(cells)
        self._state_fn = state_fn
        self._shadows = cfg.shadows
        self._model = model
        self._data = mujoco.MjData(model)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, cfg.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, cfg.height)
        marker_radius, marker_scale = _marker_size(marker_radius, cfg.env_id)
        self._pose_thd = pose_thd if pose_task else None
        self._joint_radii: dict[int, float] = {}
        self._opt = _render_option(
            model,
            cfg.show_scene,
            cfg.show_tendons,
            pose_task,
            marker_radius,
            marker_scale,
        )
        if cfg.floor:
            model.vis.rgba.fog[:] = FOG_RGBA
            model.vis.rgba.haze[:] = FOG_RGBA
            model.vis.quality.offsamples = 8  # anti-alias thin far grid lines
        self._pert = mujoco.MjvPerturb()
        center, size, floor_z = _visible_bounds(model, self._opt)
        base_radius = float(np.clip(JOINT_MARKER_FRACTION * size, *JOINT_MARKER_RANGE))
        joint_scale = HAND_JOINT_MARKER_SCALE if "Hand" in cfg.env_id else 1.0
        self._joint_radii = {
            joint: radius * joint_scale
            for joint, radius in _joint_marker_radii(
                model, self._opt, base_radius
            ).items()
        }
        self._shift = np.zeros(3)
        if shared is not None:  # align this body with the grid's common frame
            self._shift = np.array(
                [
                    shared.center[0] - center[0],
                    shared.center[1] - center[1],
                    shared.floor_z - floor_z,
                ]
            )
            center, size, floor_z = shared.center, shared.size, shared.floor_z
        if shift is not None:  # e.g. body-landmark alignment instead of the floor
            self._shift = np.asarray(shift, dtype=float)
        spacing = cfg.env_spacing or 1.36 * size
        self._offsets = _grid_offsets(cols, rows, spacing)
        self._floor_z = floor_z - 0.01 if cfg.floor else None
        self._target_sites = _target_site_ids(model) if not cfg.show_scene else []
        self._radii = None
        self._radius_scale = marker_scale
        self._tracker_sites = (
            _tracker_site_ids(model) if not cfg.show_scene and not pose_task else []
        )
        self._velocity_fn = velocity_fn
        if not show_targets and not cfg.show_scene:  # no target position to show
            hidden = self._target_sites + self._tracker_sites
            model.site_rgba[hidden, 3] = 0.0
            self._target_sites, self._tracker_sites = [], []
        if marker_radius is None and isinstance(
            TARGET_SITE_RADIUS, tuple
        ):  # visuals only
            low, high = TARGET_SITE_RADIUS
            self._radii = np.random.default_rng(cfg.seed).uniform(low, high, n_envs)
        self._total_steps, self._frame = max(total_steps, 1), 0
        self._sweep = None
        self._cam: int | str | mujoco.MjvCamera
        if n_envs == 1 and cfg.camera != "-1":
            self._cam = _video_camera(cfg, model)  # named / numbered model camera
        else:
            self._cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, self._cam)
            grid_size = float(np.ptp(self._offsets, axis=0).max())
            self._cam.lookat[:] = center
            self._cam.distance = (
                cfg.distance
                if cfg.distance is not None
                else 1.6 * size + 1.3 * grid_size
            )
            if cfg.azimuth is not None:
                self._cam.azimuth = cfg.azimuth
            if cfg.elevation is not None:
                self._cam.elevation = cfg.elevation
            self._cam.lookat[:] += cfg.lookat or (0.0, 0.0, 0.0)
            if cfg.camera_sweep:
                end = (
                    self._cam.azimuth
                    if SWEEP_END["azimuth"] is None
                    else SWEEP_END["azimuth"],
                    self._cam.elevation
                    if SWEEP_END["elevation"] is None
                    else SWEEP_END["elevation"],
                    self._cam.distance * SWEEP_END["distance_scale"],
                )
                self._sweep = (SWEEP_START, end)
        if cfg.floor:
            # Fog distances are in units of the extent (MuJoCo multiplies them by it).
            reach = (
                self._cam.distance if isinstance(self._cam, mujoco.MjvCamera) else 4.0
            )
            extent = 1.5 * reach  # visual only: clip planes, shadow clip, fog scale
            model.stat.extent = extent
            model.vis.map.fogstart = FOG_START * reach / extent
            model.vis.map.fogend = FOG_END * reach / extent
            self._horizon = HORIZON * reach  # plate / sky walls, fully fogged
            grid_size = float(np.ptp(self._offsets, axis=0).max())
            half = 0.5 * grid_size + 2.0 * size  # grid lines only near the agents
            self._floor_half = float(np.ceil(half / FLOOR_CELL) * FLOOR_CELL)

        # Create the Renderer last: it bakes in the fog colour, fog distances and
        # ``stat.extent`` set above (later changes to them are ignored).
        # The scene holds every env's geoms (bones, and tendon/site geoms when
        # enabled): size the buffer to the grid instead of MuJoCo's default 10000.
        probe = mujoco.MjvScene(model, maxgeom=10000)
        mujoco.mj_forward(model, self._data)
        mujoco.mjv_addGeoms(
            model,
            self._data,
            self._opt,
            mujoco.MjvPerturb(),
            mujoco.mjtCatBit.mjCAT_ALL.value,
            probe,
        )
        # plate + sky walls + light + grid lines (two directions)
        floor_geoms = (
            2 * (int(2 * self._floor_half / FLOOR_CELL) + 2) + 10 if cfg.floor else 0
        )
        n_drawn = len(self._cells)
        joint_geoms = (64 if pose_task else 1) * n_drawn  # joint markers / arrows
        max_geom = max(
            10000, int(1.5 * probe.ngeom * n_drawn) + 1000 + floor_geoms + joint_geoms
        )
        self._renderer = mujoco.Renderer(
            model, height=cfg.height, width=cfg.width, max_geom=max_geom
        )

    @property
    def sweep_done(self) -> bool:
        """True once the rendered frames have covered the whole camera sweep."""
        return self._frame >= self._total_steps

    def jump_to(self, progress: float) -> None:
        self._frame = progress * max(self._total_steps - 1, 1)

    def _sweep_camera(self) -> None:
        """Ease the free camera from the low oblique start to its end pose."""
        start, (azimuth, elevation, distance) = self._sweep
        t = min(self._frame / max(self._total_steps - 1, 1), 1.0)
        t = t * t * (3.0 - 2.0 * t)  # smoothstep
        self._cam.azimuth = start["azimuth"] + t * (azimuth - start["azimuth"])
        self._cam.elevation = start["elevation"] + t * (elevation - start["elevation"])
        d0 = start["distance_scale"] * distance
        self._cam.distance = d0 + t * (distance - d0)

    def render(self, depth: bool = False):
        """The frame; with ``depth`` also its depth image ``(rgb, depth)``."""
        scene = self._renderer.scene
        if self._sweep is not None:
            self._sweep_camera()
        self._frame += 1
        for k, env_id in enumerate(self._cells):
            offset = self._offsets[env_id] + self._shift
            qpos, qvel, targets, joint_errors = self._state_fn(env_id)
            self._data.qpos[:] = qpos
            self._data.qvel[:] = qvel
            mujoco.mj_forward(self._model, self._data)
            if self._radii is not None:  # this cell's own target radius
                radius = self._radii[env_id] * self._radius_scale
                self._model.site_size[self._target_sites, 0] = radius
                self._model.site_size[self._tracker_sites, 0] = radius
            for site_id, pos in targets.items():
                self._data.site_xpos[site_id] = pos
            if k == 0:
                self._renderer.update_scene(
                    self._data, camera=self._cam, scene_option=self._opt
                )
                first = 0
            else:
                first = scene.ngeom
                mujoco.mjv_addGeoms(
                    self._model,
                    self._data,
                    self._opt,
                    self._pert,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    scene,
                )
            for i in range(first, scene.ngeom):
                scene.geoms[i].pos[:] += offset
            for joint_id, error in joint_errors.items():
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([self._joint_radii[joint_id], 0.0, 0.0]),
                    self._data.xanchor[joint_id] + offset,
                    np.eye(3).ravel(),
                    _joint_error_rgba(error, self._pose_thd),
                )
                scene.ngeom += 1
            velocity = None if self._velocity_fn is None else self._velocity_fn(env_id)
            if velocity is not None and np.linalg.norm(velocity) > 1e-6:
                base = qpos[:3] + offset + (0.0, 0.0, VELOCITY_ARROW_HEIGHT)
                tip = base + VELOCITY_ARROW_LENGTH * np.array([*velocity, 0.0])
                arrow = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    arrow,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).ravel(),
                    np.array(VELOCITY_ARROW_RGBA),
                )
                mujoco.mjv_connector(
                    arrow,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    VELOCITY_ARROW_WIDTH,
                    base,
                    tip,
                )
                scene.ngeom += 1
        if self._floor_z is not None:
            _add_floor(scene, self._floor_z, self._floor_half, self._horizon)
            scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
            scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
            scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(self._shadows)
        if not depth:
            return self._renderer.render()
        self._renderer.enable_depth_rendering()
        depth_image = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        return self._renderer.render(), depth_image

    def close(self) -> None:
        self._renderer.close()


@dataclass(frozen=True)
class SharedView:
    """Frame shared by the bodies of a mixed grid: centre, size, floor height."""

    center: np.ndarray
    size: float
    floor_z: float


def _layer_bounds(
    model: mujoco.MjModel, cfg: EvalConfig, pose_task: bool, marker_radius
) -> tuple[np.ndarray, float, float]:
    """``_visible_bounds`` of a body as ``GridRenderer`` will draw it (on a copy)."""
    radius, scale = _marker_size(marker_radius, cfg.env_id)
    probe = copy.copy(model)
    opt = _render_option(
        probe, cfg.show_scene, cfg.show_tendons, pose_task, radius, scale
    )
    return _visible_bounds(probe, opt)


class MixedGridRenderer:
    """One grid whose cells hold different bodies.

    Every distinct body model is drawn by its own ``GridRenderer`` (all with the same
    camera, cell layout and floor, each only its own cells); the layers are merged
    per pixel by depth into a single frame.
    """

    def __init__(self, layers: list[GridRenderer]) -> None:
        self._layers = layers

    @property
    def sweep_done(self) -> bool:
        return all(layer.sweep_done for layer in self._layers)

    def jump_to(self, progress: float) -> None:
        """Put the camera sweeps at *progress* (0-1) of their way (``--preview``)."""
        for layer in self._layers:
            layer.jump_to(progress)

    def render(self) -> np.ndarray:
        if len(self._layers) == 1:
            return self._layers[0].render()
        rgb, depth = zip(*(layer.render(depth=True) for layer in self._layers))
        nearest = np.argmin(np.stack(depth), axis=0)
        return np.take_along_axis(np.stack(rgb), nearest[None, ..., None], axis=0)[0]

    def close(self) -> None:
        for layer in self._layers:
            layer.close()


class FrozenStates:
    """Wrap a state function so finished agents can be drawn frozen in a pose.

    ``last`` remembers the state of every agent at the previous rendered frame;
    ``freeze(i)`` pins agent ``i`` to it (its pose before an auto-reset), or to
    ``state`` if given (e.g. the current pre-reset state).
    """

    def __init__(self, inner):
        self._inner = inner
        self._last: dict[int, EnvState] = {}
        self._frozen: dict[int, EnvState] = {}

    def __call__(self, env_id: int) -> EnvState:
        if env_id in self._frozen:
            return self._frozen[env_id]
        state = self._inner(env_id)
        self._last[env_id] = state
        return state

    def freeze(self, env_id: int, current: bool = False) -> None:
        self._frozen[env_id] = (
            self._inner(env_id)
            if current or env_id not in self._last
            else self._last[env_id]
        )


def _mjlab_marker_radius(env) -> float | None:
    """Success threshold of an mjlab reach task as a marker radius in metres."""
    cmds = env.command_manager
    if "reach" in cmds.active_terms:
        k = len(cmds.get_term("reach").cfg.tip_sites)
        return REACH_SOLVED_DIST * np.sqrt(k)  # equal error at each of the k sites
    return None


def _cpu_marker_radius(env) -> float | None:
    """Success threshold of a CPU reach env as a marker radius in metres."""
    env = env.unwrapped
    if hasattr(env, "tip_sids") and not hasattr(env, "pose_thd"):  # reach
        return REACH_SOLVED_DIST * np.sqrt(len(env.tip_sids))
    return None


def _cpu_shows_targets(env) -> bool:
    """Pose and reach envs have target markers; locomotion envs do not."""
    env = env.unwrapped
    return any(
        hasattr(env, attr)
        for attr in (
            "target_jnt_value",
            "tip_sids",
            "target_sids",
            "target_reach_range",
        )
    )


def _cpu_target_velocity(env) -> np.ndarray | None:
    """Target planar velocity of a CPU walking env (``None`` if it has none)."""
    env = env.unwrapped
    if hasattr(env, "target_y_vel"):
        return np.array([env.target_x_vel, env.target_y_vel])
    return None


def _mjlab_pose_thd(env) -> float | None:
    """``pose_thd`` (rad) of an mjlab pose task."""
    if "pose" not in env.command_manager.active_terms:
        return None
    return float(env.metrics_manager.cfg["success"].params["pose_thd"])


def _cpu_pose_thd(env) -> float | None:
    """``pose_thd`` (rad) of a CPU pose env."""
    thd = getattr(env.unwrapped, "pose_thd", None)
    return None if thd is None else float(thd)


def _mjlab_state_fn(env):
    """State source of an mjlab env.

    Reach targets come from the reach command (positions). For a pose task the errors
    of the joints of the pose command (target minus current angle) are returned.
    """
    model = env.sim.mj_model
    target_ids = {model.site(i).name.split("/")[-1]: i for i in _target_site_ids(model)}
    cmds = env.command_manager
    joint_of_qadr = {int(model.jnt_qposadr[j]): j for j in range(model.njnt)}
    pose_joints: list[int] = []
    if "pose" in cmds.active_terms:
        term = cmds.get_term("pose")
        qadr = env.scene[term.cfg.entity_name].indexing.joint_q_adr.cpu().numpy()
        # The pose command covers the leading joints (demoted exo joints follow).
        pose_joints = [joint_of_qadr[int(a)] for a in qadr[: len(term.cfg.low)]]

    def state(env_id: int) -> EnvState:
        data = env.sim.data
        targets: dict[int, np.ndarray] = {}
        errors: dict[int, float] = {}
        if "reach" in cmds.active_terms:
            term = cmds.get_term("reach")
            xyz = term.command[env_id].cpu().numpy().reshape(-1, 3)
            for tip, pos in zip(term.cfg.tip_sites, xyz, strict=True):
                if f"{tip}_target" in target_ids:
                    targets[target_ids[f"{tip}_target"]] = pos
        elif pose_joints:
            target = cmds.get_term("pose").command[env_id].cpu().numpy()
            qpos = data.qpos[env_id].cpu().numpy()
            for joint, goal in zip(pose_joints, target, strict=True):
                errors[joint] = abs(float(goal - qpos[model.jnt_qposadr[joint]]))
        return (
            data.qpos[env_id].cpu().numpy(),
            data.qvel[env_id].cpu().numpy(),
            targets,
            errors,
        )

    return state


def _cpu_state_fn(envs):
    """State source of a list of CPU envs (targets are moved in their data).

    For a pose env the errors of the joints of the pose target are returned.
    """
    model = envs[0].unwrapped.model
    target_ids = _target_site_ids(model)
    pose = hasattr(envs[0].unwrapped, "target_jnt_value")

    def state(env_id: int) -> EnvState:
        env = envs[env_id].unwrapped
        data = env.data
        errors: dict[int, float] = {}
        if pose:
            target = np.asarray(env.target_jnt_value, dtype=float)
            for joint in range(model.njnt):
                adr = int(model.jnt_qposadr[joint])
                if adr < len(target):
                    errors[joint] = abs(float(target[adr] - data.qpos[adr]))
        return (
            data.qpos.copy(),
            data.qvel.copy(),
            {i: data.site_xpos[i].copy() for i in target_ids},
            errors,
        )

    return state


def _summary(
    returns: list[float],
    lengths: list[int],
    successes: list[float] | None,
) -> None:
    """Print return/length and the success rate (``n/a`` if the env has none)."""
    print(f"episodes: {len(returns)}")
    print(f"return:   {np.mean(returns):.3f} +- {np.std(returns):.3f}")
    print(f"length:   {np.mean(lengths):.1f}")
    if successes:
        print(f"success:  {100 * np.mean(successes):.1f}% (solved on the final step)")
    else:
        print("success:  n/a (the env reports no 'solved' flag / success metric)")


# MyoSuite logo faded in, big and centred, over the last LOGO_FADE_SECONDS of a video.
LOGO_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs/source/images/MyoSuite 3 Full Color Horizontal wider.png"
)
LOGO_FADE_SECONDS = 5.0
LOGO_WIDTH_FRACTION = 0.7  # logo width relative to the frame width


def _fade_in_logo(frames: list[np.ndarray], fps: int) -> list[np.ndarray]:
    """Overlay the logo with linearly rising opacity (0 -> 1) over the last seconds."""
    from PIL import Image

    height, width = frames[0].shape[:2]
    logo = Image.open(LOGO_PATH).convert("RGBA")
    logo_width = int(width * LOGO_WIDTH_FRACTION)
    logo = logo.resize((logo_width, int(logo.height * logo_width / logo.width)))
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    canvas.paste(logo, ((width - logo.width) // 2, (height - logo.height) // 2))
    overlay = np.asarray(canvas, dtype=np.float32) / 255.0
    rgb, alpha = overlay[..., :3] * 255.0, overlay[..., 3:]
    n_fade = min(len(frames), max(int(round(LOGO_FADE_SECONDS * fps)), 1))
    out = list(frames)
    for k in range(n_fade):
        weight = alpha * ((k + 1) / n_fade)
        i = len(frames) - n_fade + k
        out[i] = (frames[i] * (1.0 - weight) + rgb * weight).astype(np.uint8)
    return out


def _write_preview(cfg: EvalConfig, grid) -> None:
    """``--preview``: one frame of the fresh grid (camera sweep at its end) as a PNG."""
    if grid is None:
        raise SystemExit("--preview needs --video (the PNG goes next to it).")
    import imageio

    grid.jump_to(cfg.preview_at)
    png = cfg.video.with_suffix(".png")
    png.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(png, grid.render())
    print(f"preview:  {png}")


def _write_video(cfg: EvalConfig, frames: list[np.ndarray], step_dt: float) -> None:
    import imageio

    cfg.video.parent.mkdir(parents=True, exist_ok=True)
    fps = int(round(1.0 / step_dt))
    if cfg.logo:
        frames = _fade_in_logo(frames, fps)
    imageio.mimsave(cfg.video, frames, fps=fps)
    print(f"video:    {cfg.video}")


def _episode_checkpoints(cfg: EvalConfig, latest: Path, per_env: int) -> list[Path]:
    """Checkpoint replaying episode ``k`` of every env, ``k = 0 .. per_env - 1``.

    Interpolates over the *list of stored checkpoints* (sorted by iteration), not
    over iteration numbers: ``--first-policy`` / ``--last-policy`` (iterations,
    snapped to a stored checkpoint; default: the iteration of *latest*) give the
    positions of the first and last episode's policy in that list, and episode
    ``k`` takes the stored checkpoint at the linearly interpolated position.
    """
    saved = sorted(
        (int(c.stem.split("_")[-1]), c)
        for c in latest.parent.glob("model_*.pt")
        if re.fullmatch(r"model_\d+", c.stem)
    )
    iterations = [it for it, _ in saved]

    def position(iteration: int | str | None) -> int:
        if iteration is None or iteration == "last":
            iteration = int(latest.stem.split("_")[-1])
        return min(range(len(saved)), key=lambda i: abs(iterations[i] - iteration))

    first, last = position(cfg.first_policy), position(cfg.last_policy)
    return [
        saved[round(first + (last - first) * k / max(per_env - 1, 1))][1]
        for k in range(per_env)
    ]


class PolicySchedule:
    """One policy per episode index (deduplicated); acts per env by its episode.

    Args:
        paths: Checkpoint of episode ``k`` (see :func:`_episode_checkpoints`).
        action_dim: Action dimension of the env.
        to: Optional callable moving a loaded policy (e.g. ``.to(device)``).
    """

    def __init__(self, paths: list[Path], action_dim: int, to=lambda policy: policy):
        from myosuite.utils.rslrl_policy import load_rslrl_policy

        loaded = {p: to(load_rslrl_policy(p, action_dim)) for p in dict.fromkeys(paths)}
        self._policies = [loaded[p] for p in paths]
        self.paths = paths

    def act(self, obs, episode_idx, call):
        """Actions ``call(policy, obs[env])`` with each env's own episode policy."""
        if len(set(map(id, self._policies))) == 1:
            return call(self._policies[0], obs)
        out = None
        for k in sorted({int(i) for i in episode_idx}):
            mask = episode_idx == k
            actions = call(self._policies[min(k, len(self._policies) - 1)], obs[mask])
            if out is None:
                out = (
                    actions.new_zeros((len(obs), actions.shape[-1]))
                    if hasattr(actions, "new_zeros")
                    else np.zeros((len(obs), actions.shape[-1]), dtype=actions.dtype)
                )
            out[mask] = actions
        return out


FULLBODY_ID = "fullbody"
"""Pseudo env id of the replayed MuscleMimic full-body agents in ``--pairs``."""

AgentSpec = tuple[str, Path]
"""``(env id, newest checkpoint of the run)`` of one agent."""


@functools.cache
def _body_model_key(env_id: str) -> tuple[int, int, int]:
    """``(nq, nbody, nu)`` of the env's MuJoCo model: its body-model family."""
    import gymnasium as gym

    if env_id == FULLBODY_ID:
        return (-1, -1, -1)

    import myosuite  # noqa: F401  (registers the CPU envs)

    env = gym.make(env_id)
    model = env.unwrapped.model
    key = (model.nq, model.nbody, model.nu)
    env.close()
    return key


def _family_weights(
    cfg: EvalConfig, pairs: list[AgentSpec], groups: list[list[int]]
) -> np.ndarray:
    """Relative sampling weight of every family in ``groups`` (``--family-weights``)."""
    weights = np.ones(len(groups))
    for item in cfg.family_weights:
        env_id, sep, value = item.partition("=")
        try:
            weight = float(value)
        except ValueError:
            weight = -1.0
        if not sep or weight < 0:
            raise SystemExit(
                f"--family-weights entries look like ENV_ID=WEIGHT (WEIGHT >= 0), got {item!r}"
            )
        matches = [
            g for g, idx in enumerate(groups) if any(pairs[i][0] == env_id for i in idx)
        ]
        if not matches:
            raise SystemExit(
                f"--family-weights: {env_id} is not one of the --pairs envs."
            )
        weights[matches[0]] = weight
    if not weights.any():
        raise SystemExit(
            "--family-weights: at least one family needs a weight above 0."
        )
    return weights


def _neighbours(cols: int, rows: int) -> list[list[int]]:
    """Cells touching each cell (all 8 directions) of the row-major ``cols x rows`` grid."""
    return [
        [
            (r + dr) * cols + c + dc
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr or dc) and 0 <= r + dr < rows and 0 <= c + dc < cols
        ]
        for r in range(rows)
        for c in range(cols)
    ]


def _spread_families(
    choice: np.ndarray, family: np.ndarray, cols: int, rows: int, rng
) -> np.ndarray:
    """Rearrange the pairs of ``choice`` (one per cell) so that cells of one body-model
    family do not touch, in any direction, as far as the family sizes allow.

    Args:
        choice: Pair index of every cell.
        family: Body-model family id of every pair.
        cols: Grid columns.
        rows: Grid rows.
        rng: Random generator (seeded by ``--seed``).

    Returns:
        The same pair indices in a new order (row-major cells).
    """
    n = len(choice)
    near = _neighbours(cols, rows)
    pools: dict[int, list[int]] = {}
    for i in rng.permutation(n):
        pools.setdefault(int(family[choice[i]]), []).append(int(choice[i]))
    placed = np.full(n, -1)
    for cell in range(n):  # greedy: the most frequent family not yet next to this cell
        used = {int(family[placed[k]]) for k in near[cell] if placed[k] >= 0}
        options = [f for f, pool in pools.items() if pool and f not in used]
        options = options or [f for f, pool in pools.items() if pool]
        best = max(options, key=lambda f: (len(pools[f]), rng.random()))
        placed[cell] = pools[best].pop()
    fam = family[placed]

    def clashes(cell: int) -> int:
        return sum(int(fam[k] == fam[cell]) for k in near[cell])

    for _ in range(200 * n):  # refine by swapping cells that clash
        a, b = (int(x) for x in rng.integers(n, size=2))
        if fam[a] == fam[b] or not (clashes(a) or clashes(b)):
            continue
        before = clashes(a) + clashes(b)
        placed[[a, b]], fam[[a, b]] = placed[[b, a]], fam[[b, a]]
        if clashes(a) + clashes(b) > before:  # worse: undo
            placed[[a, b]], fam[[a, b]] = placed[[b, a]], fam[[b, a]]
    touching = sum(clashes(cell) for cell in range(n)) // 2
    print(f"same-family cells touching each other: {touching} pair(s)")
    return placed


def _agent_specs(cfg: EvalConfig, checkpoint: Path, n_envs: int) -> list[AgentSpec]:
    """Env id and run of every agent: the given one, or sampled from ``--pairs``."""
    if not cfg.pairs:
        return [(cfg.env_id, checkpoint)] * n_envs
    pairs: list[AgentSpec] = []
    for item in cfg.pairs:
        env_id, sep, path = item.partition("=")
        if item == FULLBODY_ID:  # replayed full-body policy, no env or run directory
            pairs.append((FULLBODY_ID, Path(FULLBODY_ID)))
            continue
        if not sep or not env_id or not path:
            raise SystemExit(
                f"--pairs entries look like ENV_ID=CHECKPOINT, got {item!r}"
            )
        pairs.append((env_id, _resolve_checkpoint(Path(path))))
    rng = np.random.default_rng(cfg.seed)
    if cfg.family_weights and not cfg.balance_models:
        raise SystemExit("--family-weights needs --balance-models.")
    if cfg.balance_models:
        families: dict[tuple, list[int]] = {}
        for i, (env_id, _) in enumerate(pairs):
            families.setdefault(_body_model_key(env_id), []).append(i)
        groups = list(families.values())
        weights = _family_weights(cfg, pairs, groups)
        picked = rng.choice(len(groups), size=n_envs, p=weights / weights.sum())
        choice = np.array([groups[g][rng.integers(len(groups[g]))] for g in picked])
        for group, share in zip(groups, weights / weights.sum(), strict=True):
            names = ", ".join(pairs[i][0] for i in group)
            print(f"model family sampled with p={share:.3f}: {names}")
    else:
        choice = rng.integers(len(pairs), size=n_envs)
    ids: dict[tuple, int] = {}
    family = np.array([ids.setdefault(_body_model_key(e), len(ids)) for e, _ in pairs])
    cols, rows = _grid_shape(cfg)
    if cols * rows == n_envs:
        choice = _spread_families(choice, family, cols, rows, rng)
    for i, (env_id, ckpt) in enumerate(pairs):
        print(
            f"pair {i}: {env_id}  {ckpt.parent.name}  x{int((choice == i).sum())} agents"
        )
    return [pairs[i] for i in choice]


class AgentPolicies:
    """Per-agent policy schedules; agents sharing an (env, run) share one schedule.

    Args:
        specs: ``(env id, newest checkpoint)`` of every agent.
        cfg: Evaluation options (``--first-policy`` / ``--last-policy``).
        per_env: Episodes per agent.
        action_dims: Action dimension per env id.
        to: Optional callable moving a loaded policy (e.g. ``.to(device)``).
    """

    def __init__(self, specs, cfg, per_env, action_dims, to=lambda policy: policy):
        keys = list(dict.fromkeys(specs))
        self.groups = [
            (
                PolicySchedule(
                    _episode_checkpoints(cfg, ckpt, per_env), action_dims[env_id], to
                ),
                np.array([i for i, spec in enumerate(specs) if spec == (env_id, ckpt)]),
            )
            for env_id, ckpt in keys
        ]

    def act(self, get_obs, episode_idx, call):
        """``[(agent indices, actions)]``; ``get_obs(indices)`` returns their obs."""
        return [
            (idx, schedule.act(get_obs(idx), episode_idx[idx], call))
            for schedule, idx in self.groups
        ]

    def print_policies(self) -> None:
        for schedule, idx in self.groups:
            if len(set(schedule.paths)) > 1:
                names = ", ".join(p.stem.removeprefix("model_") for p in schedule.paths)
                print(f"policies per episode ({len(idx)} agents, iterations): {names}")


def evaluate_cpu(cfg: EvalConfig, checkpoint: Path) -> None:
    """Roll out ``cols x rows`` CPU envs in lockstep, ``episodes_per_env`` each."""
    import gymnasium as gym

    import myosuite  # noqa: F401  (registers the CPU envs)

    cols, rows = _grid_shape(cfg)
    n_envs, per_env = cols * rows, _episodes_per_env(cfg)
    specs = _agent_specs(cfg, checkpoint, n_envs)
    if any(env_id == FULLBODY_ID for env_id, _ in specs):
        raise SystemExit("--pairs fullbody needs --backend mjlab.")
    envs = [gym.make(env_id) for env_id, _ in specs]
    step_dt_cpu = getattr(envs[0].unwrapped, "dt", None) or envs[0].unwrapped._ctrl_dt
    models = {(e.unwrapped.model.nq, e.unwrapped.model.nbody) for e in envs}
    if len(models) > 1:
        raise SystemExit(
            "--pairs: all envs must share one body model to be drawn together."
        )
    agents = AgentPolicies(
        specs,
        cfg,
        per_env,
        {env_id: e.action_space.shape[0] for (env_id, _), e in zip(specs, envs)},
    )
    agents.print_policies()
    grid = (
        GridRenderer(
            envs[0].unwrapped.model,
            _cpu_state_fn(envs),
            cfg,
            per_env * max(e.spec.max_episode_steps or 1000 for e in envs),
            pose_task=hasattr(envs[0].unwrapped, "target_jnt_value"),
            marker_radius=_cpu_marker_radius(envs[0]),
            pose_thd=_cpu_pose_thd(envs[0]),
            show_targets=_cpu_shows_targets(envs[0]),
            velocity_fn=lambda i: _cpu_target_velocity(envs[i]),
        )
        if cfg.video
        else None
    )
    frames: list[np.ndarray] = []
    returns, lengths, successes = [], [], []
    obs = [env.reset(seed=cfg.seed + i * per_env)[0] for i, env in enumerate(envs)]
    if cfg.preview:
        _write_preview(cfg, grid)
        return
    ep_return, ep_length = np.zeros(n_envs), np.zeros(n_envs, dtype=int)
    done_count = np.zeros(n_envs, dtype=int)
    # Camera sweep + early episode ends: keep running episodes until it completes.
    sweeping = cfg.end_on_success and cfg.camera_sweep and grid is not None
    cap = 10**9 if sweeping else per_env
    hold_steps = int(np.ceil(cfg.success_hold / step_dt_cpu))
    hold_left = np.zeros(n_envs, dtype=int)  # steps left of the hold after a success
    while ((done_count < cap) | (hold_left > 0)).any():
        actions: dict[int, np.ndarray] = {}
        for idx, acts in agents.act(
            lambda idx: np.stack([obs[i] for i in idx]),
            done_count,
            lambda pol, o: pol.act(o, cfg.stochastic),
        ):
            actions.update(zip(idx.tolist(), acts))
        for i, env in enumerate(envs):
            if done_count[i] >= cap and hold_left[i] == 0:
                continue
            obs[i], rew, terminated, truncated, info = env.step(actions[i])
            if hold_left[i] > 0:  # holding after a success: only run on
                hold_left[i] = 0 if terminated or truncated else hold_left[i] - 1
                if hold_left[i] == 0 and done_count[i] < cap:
                    seed = cfg.seed + i * per_env + int(done_count[i])
                    obs[i] = env.reset(seed=seed)[0]
                continue
            ep_return[i] += float(rew)
            ep_length[i] += 1
            hit = cfg.end_on_success and bool(info.get("solved", False))
            if terminated or truncated or hit:
                returns.append(ep_return[i])
                lengths.append(int(ep_length[i]))
                if "solved" in info:
                    successes.append(float(bool(info["solved"])))
                ep_return[i], ep_length[i] = 0.0, 0
                done_count[i] += 1
                if hit and not (terminated or truncated) and hold_steps:
                    hold_left[i] = hold_steps
                elif done_count[i] < cap:
                    seed = cfg.seed + i * per_env + int(done_count[i])
                    obs[i] = env.reset(seed=seed)[0]
        if grid is not None:
            frames.append(grid.render())
        if sweeping and grid.sweep_done:
            break
        if cfg.after_last_episode == "stop" and (done_count >= cap).any():
            break
    step_dt = step_dt_cpu
    for env in envs:
        env.close()
    _summary(returns, lengths, successes)
    if grid is not None:
        grid.close()
        _write_video(cfg, frames, step_dt)


class MjlabGroup:
    """Parallel mjlab envs of one env id: own sim (body model), policies, bookkeeping.

    ``--pairs`` with several env ids gives one group per id; groups are stepped in
    lockstep by simulated time and share one video grid (``MixedGridRenderer``).

    Args:
        cfg: Evaluation options.
        env_id: Task of this group.
        specs: ``(env id, newest checkpoint)`` of every agent (env) of the group.
        cells: Grid cell of every agent, in the order of ``specs``.
        per_env: Episodes per agent.
        device: Torch device.
    """

    def __init__(
        self,
        cfg: EvalConfig,
        env_id: str,
        specs,
        cells: list[int],
        per_env: int,
        device,
    ):
        import torch
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg

        self.cfg = replace(cfg, env_id=env_id)
        self.env_id, self.per_env, self.device = env_id, per_env, device
        self.n = len(specs)
        self.cells = cells
        env_cfg = load_env_cfg(env_id, play=True)
        env_cfg.scene.num_envs = self.n
        env_cfg.seed = cfg.seed
        self.env = env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
        self.state_fn = _mjlab_state_fn(env)
        if cfg.video and cfg.after_last_episode == "freeze":
            self.state_fn = FrozenStates(self.state_fn)
        self.agents = AgentPolicies(
            specs,
            self.cfg,
            per_env,
            {env_id: env.action_manager.total_action_dim},
            to=lambda policy: policy.to(device),
        )
        print(f"{env_id}: {self.n} envs")
        self.agents.print_policies()
        self.has_success = "success" in env.metrics_manager.active_terms
        self.success_term = (
            env.metrics_manager.cfg.get("success") if self.has_success else None
        )
        if cfg.end_on_success and self.success_term is None:
            raise SystemExit(
                f"{env_id} defines no success metric for --end-on-success."
            )
        self.show_targets = (
            "reach" in env.command_manager.active_terms
            or "pose" in env.command_manager.active_terms
        )
        self.obs, _ = env.reset()
        self.ep_return = torch.zeros(self.n, device=device)
        self.ep_length = torch.zeros(self.n, dtype=torch.long, device=device)
        self.episodes_done = torch.zeros(self.n, dtype=torch.long, device=device)
        self.returns: list[float] = []
        self.lengths: list[int] = []
        self.successes: list[float] = []
        self.cap = per_env
        self.clock = 0.0
        self.holding = torch.zeros(self.n, dtype=torch.bool, device=device)
        self.hold_left = torch.zeros(self.n, dtype=torch.long, device=device)

    def velocity_target(self, local: int) -> np.ndarray | None:
        """Target planar velocity of agent ``local`` (walking envs), else ``None``."""
        term = self.success_term
        if term is None:
            return None
        defaults = {
            name: p.default
            for name, p in inspect.signature(term.func).parameters.items()
            if p.default is not inspect.Parameter.empty
        }
        params = defaults | dict(term.params)
        if "target_y_vel" in params:
            return np.array([params["target_x_vel"], params["target_y_vel"]])
        if "command_name" in params and "target_speed" in params:  # directional walks
            command = self.env.command_manager.get_command(params["command_name"])
            return params["target_speed"] * command[local].cpu().numpy()
        return None

    @property
    def done_all(self):
        return (self.episodes_done >= self.cap) & ~self.holding

    def step(self) -> None:
        """One env step for every agent, and the episode bookkeeping."""
        import torch

        env, cfg, cap, device = self.env, self.cfg, self.cap, self.device
        actions = torch.zeros(
            self.n, env.action_manager.total_action_dim, device=device
        )
        for idx, acts in self.agents.act(
            lambda idx: self.obs["actor"][torch.as_tensor(idx, device=device)],
            self.episodes_done.cpu().numpy(),
            lambda pol, o: pol.sample(o) if cfg.stochastic else pol(o),
        ):
            actions[torch.as_tensor(idx, device=device)] = acts
        self.obs, rew, terminated, truncated, _ = env.step(actions)
        recording = (self.episodes_done < cap) & ~self.holding
        self.ep_return += rew * recording
        self.ep_length += recording.long()
        finished = (terminated | truncated) & recording
        self.returns += self.ep_return[finished].tolist()
        self.lengths += self.ep_length[finished].tolist()
        if self.has_success:  # final-step value of the standard success metric
            for i in finished.nonzero().flatten().tolist():
                values = dict(env.metrics_manager.get_active_iterable_terms(i))
                self.successes.append(float(values["success"][0]))
        self.ep_return[finished] = 0.0
        self.ep_length[finished] = 0
        self.episodes_done += finished.long()
        if isinstance(self.state_fn, FrozenStates):  # pose before the auto-reset
            last = (finished & (self.episodes_done >= cap)).nonzero().flatten()
            for i in last.tolist():
                self.state_fn.freeze(i)
        hold_steps = int(np.ceil(cfg.success_hold / env.step_dt))
        ended = terminated | truncated
        if hold_steps:  # an env that reset itself while holding is done holding
            expired = self.holding & ended
            if isinstance(self.state_fn, FrozenStates):  # pose before the auto-reset
                for i in (expired & (self.episodes_done >= cap)).nonzero().flatten():
                    self.state_fn.freeze(int(i))
            self.holding &= ~ended
        if cfg.end_on_success:
            term = self.success_term
            hit = (
                (term.func(env, **term.params) > 0.5)
                & (self.episodes_done < cap)
                & ~self.holding
                & ~ended
            )
            if hit.any():
                ids = hit.nonzero().flatten()
                self.returns += self.ep_return[ids].tolist()
                self.lengths += self.ep_length[ids].tolist()
                self.successes += [1.0] * len(ids)
                self.ep_return[ids] = 0.0
                self.ep_length[ids] = 0
                self.episodes_done[ids] += 1
                if hold_steps:  # keep acting for a while, reset afterwards
                    self.holding[ids] = True
                    self.hold_left[ids] = hold_steps
                else:
                    self._reset_finished(ids)
            if hold_steps:
                self.hold_left[self.holding] -= 1
                over = (self.holding & (self.hold_left <= 0)).nonzero().flatten()
                if len(over):
                    self.holding[over] = False
                    self._reset_finished(over)

    def _reset_finished(self, ids) -> None:
        """Freeze agents that ran their last episode (pose now), reset the envs."""
        if isinstance(self.state_fn, FrozenStates):
            for i in ids[self.episodes_done[ids] >= self.cap].tolist():
                self.state_fn.freeze(i, current=True)
        self.obs, _ = self.env.reset(env_ids=ids)


def _quat_yaw(quat: np.ndarray) -> np.ndarray:
    """Heading (rad) of ``(w, x, y, z)`` quaternions ``(..., 4)`` about the vertical axis."""
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rotate_z(quat: np.ndarray, angle: float) -> np.ndarray:
    """``(w, x, y, z)`` quaternions turned by *angle* about the world z axis."""
    half = 0.5 * angle
    qz = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
    out = np.empty_like(quat)
    for i, q in enumerate(quat):
        mujoco.mju_mulQuat(out[i], qz, q)
    return out


def _closed_walking_loop(
    qpos: np.ndarray,
    dt: float,
    yaw: float,
    min_period: float = 0.6,
    max_period: float = 3.0,
    blend: float = 0.25,
) -> np.ndarray:
    """A seamless in-place walking loop ``(T, nq)`` cut from a recorded free-base rollout.

    The loop runs between the two walking frames (root speed above 60% of the fast
    steady speed) whose joint angles, joint velocities and heading match best. The rollout
    is not exactly periodic, so the last *blend* seconds cross-fade into the frames that
    lead up to the loop start: the end then continues into the start. The root travels
    with constant speed over the loop: that drift is removed (a treadmill, so the agent
    stays in its grid cell), the heading is turned to *yaw* (radians). Joints must be
    hinges or slides next to the free root joint.
    """
    root_xy = qpos[:, :2]
    speed = np.linalg.norm(np.gradient(root_xy, dt, axis=0), axis=1)
    speed = np.convolve(speed, np.ones(25) / 25, mode="same")
    heading = _quat_yaw(qpos[:, 3:7])
    joint_speed = np.gradient(qpos[:, 7:], dt, axis=0)
    features = np.concatenate(
        [
            qpos[:, 7:],
            0.05 * joint_speed,
            0.5 * np.stack([np.cos(heading), np.sin(heading)], 1),
        ],
        axis=1,
    )
    fade = int(blend / dt)
    walking = np.flatnonzero(speed > 0.6 * np.percentile(speed, 90))
    walking = walking[walking >= fade]  # frames before the start feed the cross-fade
    lo, hi = int(min_period / dt), int(max_period / dt)
    best, best_cost = None, np.inf
    for i in walking:
        js = walking[(walking >= i + lo) & (walking <= i + hi)]
        if not len(js):
            continue
        cost = np.linalg.norm(features[js] - features[i], axis=1)
        k = int(np.argmin(cost))
        if cost[k] < best_cost:
            best, best_cost = (int(i), int(js[k])), float(cost[k])
    if best is None:
        raise SystemExit("fullbody: the rollout has no steady walking section to loop.")
    i, j = best
    period = j - i
    print(
        f"fullbody loop: frames {i}-{j} ({period * dt:.2f} s), match cost {best_cost:.3f}"
    )
    section = qpos[i - fade : j].copy()  # ``fade`` lead-in frames, then the loop
    drift = (qpos[j, :2] - qpos[i, :2]) / period  # constant-velocity travel per frame
    frames = np.arange(-fade, period)[:, None]
    section[:, :2] -= qpos[i, :2] + drift * frames
    mean_heading = np.arctan2(np.sin(heading[i:j]).mean(), np.cos(heading[i:j]).mean())
    turn = yaw - mean_heading
    c, s_ = np.cos(turn), np.sin(turn)
    section[:, :2] = section[:, :2] @ np.array([[c, s_], [-s_, c]])
    section[:, 3:7] = _rotate_z(section[:, 3:7], turn)
    loop = section[fade:].copy()
    lead_in = section[:fade]
    for m in range(fade):  # cross-fade the tail into the lead-in frames
        w = (m + 1) / fade
        w = w * w * (3.0 - 2.0 * w)
        tail = period - fade + m
        mixed = (1.0 - w) * loop[tail] + w * lead_in[m]
        quat = (1.0 - w) * loop[tail, 3:7] + w * np.sign(
            loop[tail, 3:7] @ lead_in[m, 3:7]
        ) * lead_in[m, 3:7]
        mixed[3:7] = quat / np.linalg.norm(quat)
        loop[tail] = mixed
    return loop


# Bodies present in the full body and in the arm / torso / leg / hand models (right-hand
# side for the arm), in the order they are used to place a model where the full body has it.
BODY_LANDMARKS = ("sacrum", "pelvis", "torso", "clavicle_r", "humerus_r", "scapula_r")


def _body_landmarks(
    model: mujoco.MjModel, data: mujoco.MjData
) -> dict[str, np.ndarray]:
    """World position of every ``BODY_LANDMARKS`` body of the model (entity prefix ignored)."""
    found = {}
    for body in range(model.nbody):
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "").split(
            "/"
        )[-1]
        if name in BODY_LANDMARKS and name not in found:
            found[name] = data.xpos[body].copy()
    return found


def _landmark_shift(
    model: mujoco.MjModel, reference: dict[str, np.ndarray]
) -> np.ndarray | None:
    """Translation putting the model's first shared landmark onto the reference body's.

    All MyoSuite bodies are modelled in one anatomical frame (shoulders at 1.4 m, pelvis at
    0.9 m), so this seats an arm, torso or leg where the full body has it. ``None`` if the
    model has none of the landmarks.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    own = _body_landmarks(model, data)
    for name in BODY_LANDMARKS:
        if name in own and name in reference:
            return reference[name] - own[name]
    return None


class FullbodyReplay:
    """Cells showing the pretrained MuscleMimic full-body policy walking (no episodes).

    The policy is rolled out once on its reference motion (CPU, JAX), a seamless walking
    loop is cut from the recording (``_closed_walking_loop``), and every cell plays that
    loop in place from its own phase. Cached as ``logs/fullbody_replay/*.npz``.

    Args:
        cfg: Evaluation options (``--fullbody-*``).
        cells: Grid cells of the full-body agents.
    """

    def __init__(self, cfg: EvalConfig, cells: list[int]) -> None:
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")  # keep the GPU for the envs
        from myosuite.integrations.musclemimic.fullbody_model import (
            compile_musclemimic_fullbody_mjmodel,
            default_musclemimic_fullbody_config,
        )

        self.cfg = replace(cfg, env_id=FULLBODY_ID)
        self.cells = cells
        self.model, _, _ = compile_musclemimic_fullbody_mjmodel(
            default_musclemimic_fullbody_config()
        )
        self.dt = 5 * float(
            self.model.opt.timestep
        )  # the policy acts every 5 sim steps
        loop = self._load_or_record(cfg)
        self.loop = loop
        self.phase = np.random.default_rng(cfg.seed).integers(
            len(loop), size=len(cells)
        )
        self.clock = 0.0
        self._local = {cell: i for i, cell in enumerate(cells)}
        print(f"{FULLBODY_ID}: {len(cells)} agents replaying {cfg.fullbody_ref}")

    def _load_or_record(self, cfg: EvalConfig) -> np.ndarray:
        key = hashlib.md5(
            f"{cfg.fullbody_ref}|{cfg.fullbody_motion}|{cfg.fullbody_yaw}".encode()
        ).hexdigest()[:10]
        cache = Path("logs/fullbody_replay") / f"{key}.npz"
        if cache.exists():
            return np.load(cache)["loop"]
        rollout = self._rollout(cfg)
        loop = _closed_walking_loop(rollout, self.dt, np.radians(cfg.fullbody_yaw))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, loop=loop)
        return loop

    def _rollout(self, cfg: EvalConfig) -> np.ndarray:
        from myosuite.core.trajectory_io import load_motion_clip, resolve_motion_path
        from myosuite.integrations.musclemimic.fullbody_checkpoint_io import (
            resolve_checkpoint_ref,
        )
        from myosuite.integrations.musclemimic.fullbody_local_policy import (
            FullbodyObsAdapter,
            LocalPolicyRunner,
            load_local_policy_artifacts,
            read_checkpoint_config_metadata,
        )

        model = self.model
        root = resolve_checkpoint_ref(cfg.fullbody_ref).local_path
        motion = resolve_motion_path(cfg.fullbody_motion, env_name="MyoFullBody")
        clip = load_motion_clip(motion, expected_nq=model.nq, expected_nv=model.nv)
        goal_params = (
            read_checkpoint_config_metadata(root)
            .get("experiment", {})
            .get("env_params", {})
            .get("goal_params", {})
        )
        policy = LocalPolicyRunner(
            artifacts=load_local_policy_artifacts(root),
            stochastic=False,
            seed=cfg.seed,
            frame_skip=5,
            obs_adapter=FullbodyObsAdapter(
                model=model, clip=clip, goal_params=goal_params
            ),
        )
        data = mujoco.MjData(model)
        data.qpos[:] = clip.qpos[0]
        if clip.qvel is not None and clip.qvel.shape[0]:
            data.qvel[:] = clip.qvel[0]
        mujoco.mj_forward(model, data)
        frames = int(clip.qpos.shape[0])
        print(
            f"{FULLBODY_ID}: recording {frames} policy steps on {cfg.fullbody_motion} ..."
        )
        out = np.empty((frames, model.nq))
        for i in range(frames):
            policy.step(model, data, policy.action_for(data, clip, i))
            out[i] = data.qpos
        return out

    def landmarks(self) -> dict[str, np.ndarray]:
        """``BODY_LANDMARKS`` positions of the walker in the first frame of its loop."""
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.loop[0]
        mujoco.mj_forward(self.model, data)
        return _body_landmarks(self.model, data)

    def advance(self, dt: float) -> None:
        self.clock += dt

    def state_fn(self, local: int) -> EnvState:
        frame = int(self.clock / self.dt + self.phase[local]) % len(self.loop)
        return self.loop[frame], np.zeros(self.model.nv), {}, {}

    def velocity_target(self, local: int) -> np.ndarray | None:
        return None


def _mixed_renderer(
    cfg: EvalConfig,
    groups: list[MjlabGroup],
    total_frames: int,
    replay: FullbodyReplay | None = None,
) -> MixedGridRenderer:
    """One renderer for the grid; cells of groups with equal body models share a layer."""
    owner = {cell: (g, i) for g in groups for i, cell in enumerate(g.cells)}
    if replay is not None:
        owner |= {cell: (replay, i) for i, cell in enumerate(replay.cells)}

    def state(cell: int) -> EnvState:
        group, local = owner[cell]
        return group.state_fn(local)

    def velocity(cell: int) -> np.ndarray | None:
        group, local = owner[cell]
        return group.velocity_target(local)

    layers: dict[tuple, list[MjlabGroup]] = {}
    for g in groups:
        m = g.env.sim.mj_model
        key = (
            m.nq,
            m.nbody,
            m.nsite,
            m.ngeom,
            "pose" in g.env.command_manager.active_terms,
            _mjlab_marker_radius(g.env),
            _mjlab_pose_thd(g.env),
            g.show_targets,
        )
        layers.setdefault(key, []).append(g)
    bounds = {
        key: _layer_bounds(gs[0].env.sim.mj_model, gs[0].cfg, key[4], key[5])
        for key, gs in layers.items()
    }
    if replay is not None:
        bounds["fullbody"] = _layer_bounds(replay.model, replay.cfg, False, None)
    master = max(bounds.values(), key=lambda b: b[1])
    shared = (
        SharedView(master[0], master[1], master[2])
        if len(layers) + (replay is not None) > 1
        else None
    )
    shifts = {key: None for key in layers}
    if replay is not None:  # seat every body where the full body has it (same heights)
        reference = replay.landmarks()
        shifts = {
            key: _landmark_shift(gs[0].env.sim.mj_model, reference)
            for key, gs in layers.items()
        }
    replay_layers = (
        []
        if replay is None
        else [
            GridRenderer(
                replay.model,
                state,
                replay.cfg,
                total_frames,
                cells=sorted(replay.cells),
                shared=shared,
                show_targets=False,
                shift=np.zeros(3),
            )
        ]
    )
    return MixedGridRenderer(
        replay_layers
        + [
            GridRenderer(
                gs[0].env.sim.mj_model,
                state,
                gs[0].cfg,
                total_frames,
                pose_task=key[4],
                marker_radius=key[5],
                pose_thd=key[6],
                cells=sorted(c for g in gs for c in g.cells),
                shared=shared,
                show_targets=key[7],
                velocity_fn=velocity,
                shift=shifts[key],
            )
            for key, gs in layers.items()
        ]
    )


def evaluate_mjlab(cfg: EvalConfig, checkpoint: Path) -> None:
    """Roll out ``cols x rows`` parallel mjlab envs, ``episodes_per_env`` each.

    ``--pairs`` with several env ids (possibly different bodies) gives every grid cell
    the env of its sampled pair; all cells are drawn in one scene.
    """
    import torch

    import myosuite.envs.myo.backends.mjlab  # noqa: F401  (registers the twins)

    os.environ.setdefault("MUJOCO_GL", "egl")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cols, rows = _grid_shape(cfg)
    n_envs, per_env = cols * rows, _episodes_per_env(cfg)
    by_env: dict[str, list[int]] = {}
    specs = _agent_specs(cfg, checkpoint, n_envs)
    for cell, (env_id, _) in enumerate(specs):
        by_env.setdefault(env_id, []).append(cell)
    fullbody_cells = by_env.pop(FULLBODY_ID, [])
    if not by_env:
        raise SystemExit("--pairs: at least one pair besides fullbody is needed.")
    replay = FullbodyReplay(cfg, fullbody_cells) if fullbody_cells else None
    groups = [
        MjlabGroup(cfg, env_id, [specs[c] for c in cells], cells, per_env, device)
        for env_id, cells in by_env.items()
    ]
    frame_dt = max(g.env.step_dt for g in groups)
    # Camera sweep + early episode ends: keep running episodes until it completes.
    sweeping = cfg.end_on_success and cfg.camera_sweep and bool(cfg.video)
    cap = 10**9 if sweeping else per_env
    for g in groups:
        g.cap = cap
    longest = max(g.env.max_episode_length * g.env.step_dt for g in groups)
    grid = (
        _mixed_renderer(cfg, groups, int(per_env * longest / frame_dt), replay)
        if cfg.video
        else None
    )
    if cfg.preview:
        _write_preview(cfg, grid)
        return
    frames: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(10**9 if sweeping else int(cap * longest / frame_dt) + 2):
            for g in groups:  # advance every group to the same simulated time
                g.clock += frame_dt
                while g.clock >= g.env.step_dt - 1e-9:
                    g.step()
                    g.clock -= g.env.step_dt
            if replay is not None:
                replay.advance(frame_dt)
            if grid is not None:
                frames.append(grid.render())
            if (
                (sweeping and grid.sweep_done)
                or all(bool(g.done_all.all()) for g in groups)
                or (
                    cfg.after_last_episode == "stop"
                    and any(bool(g.done_all.any()) for g in groups)
                )
            ):
                break
    for g in groups:
        g.env.close()
        if len(groups) > 1:
            print(f"\n== {g.env_id}")
            _summary(g.returns, g.lengths, g.successes)
    if len(groups) > 1:
        print("\n== all")
    _summary(
        [r for g in groups for r in g.returns],
        [n for g in groups for n in g.lengths],
        [x for g in groups for x in g.successes],
    )
    if grid is not None:
        grid.close()
        _write_video(cfg, frames, frame_dt)


def main() -> None:
    cfg = tyro.cli(EvalConfig)
    checkpoint = _resolve_checkpoint(cfg.checkpoint)
    print(f"\n\ncheckpoint: {checkpoint}  backend: {cfg.backend}  env: {cfg.env_id}")
    if cfg.backend == "cpu":
        evaluate_cpu(cfg, checkpoint)
    else:
        evaluate_mjlab(cfg, checkpoint)


if __name__ == "__main__":
    main()
