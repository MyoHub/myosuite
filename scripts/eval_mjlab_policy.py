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

import inspect
import os
import re
from dataclasses import dataclass
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
    camera_sweep: bool = False
    """Video: slowly move the camera from a low oblique view up and round to the
    front, revealing more and more of the grid. ``--azimuth``/``--elevation``/
    ``--distance`` set the END of the sweep; the start is ``SWEEP_START``."""
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
TARGET_SITE_RADIUS: float | None = 0.05
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


# Camera sweep start (``--camera-sweep``): low, oblique, close. The end pose is the
# regular free camera (front view, elevated).
SWEEP_START = {"azimuth": 35.0, "elevation": -6.0, "distance_scale": 0.7}
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
    group, leaving the bone meshes; a model without any mesh geoms (the finger) keeps
    its primitive geoms. Sites other than the target markers are
    hidden the same way and the markers are styled with ``TARGET_SITE_*``.
    Tendons follow *show_tendons*. Markers get the radius *marker_radius* (the task's success
    threshold; ``TARGET_SITE_RADIUS`` if None). The body site each
    target belongs to is drawn as a ``TRACKER_SITE_RGBA`` marker of the same size. Only visual model metadata is
    touched; physics is unaffected.
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
    if radius is not None:
        radius *= marker_scale
    for i in targets:
        if TARGET_SITE_RGBA is not None:
            model.site_rgba[i] = TARGET_SITE_RGBA
        if radius is not None:
            model.site_size[i, 0] = radius
    for i in targets:  # the body sites that have to reach the targets
        name = model.site(i).name.removesuffix("_target")
        tracker = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if tracker < 0:
            continue
        model.site_group[tracker] = 0
        model.site_rgba[tracker] = TRACKER_SITE_RGBA
        if radius is not None:
            model.site_size[tracker, 0] = radius
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
        show_targets: Draw the ``*_target`` markers (and their trackers); off for tasks
            that have no target position (locomotion).
        velocity_fn: ``velocity_fn(env_id) -> (vx, vy) | None``, target planar velocity
            drawn as an arrow above the root.
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
        show_targets: bool = True,
        velocity_fn=None,
    ) -> None:
        cols, rows = _grid_shape(cfg)
        n_envs = cols * rows
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
        self._velocity_fn = velocity_fn
        if not show_targets and not cfg.show_scene:  # no target position to show
            model.site_rgba[_target_site_ids(model) + _tracker_site_ids(model), 3] = 0.0
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
        spacing = cfg.env_spacing or 1.36 * size
        self._offsets = _grid_offsets(cols, rows, spacing)
        self._floor_z = floor_z - 0.01 if cfg.floor else None
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
                end = (self._cam.azimuth, self._cam.elevation, self._cam.distance)
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
        joint_geoms = (64 if pose_task else 1) * n_envs  # joint markers / arrows
        max_geom = max(
            10000, int(1.5 * probe.ngeom * n_envs) + 1000 + floor_geoms + joint_geoms
        )
        self._renderer = mujoco.Renderer(
            model, height=cfg.height, width=cfg.width, max_geom=max_geom
        )

    def _sweep_camera(self) -> None:
        """Ease the free camera from the low oblique start to its end pose."""
        start, (azimuth, elevation, distance) = self._sweep
        t = min(self._frame / max(self._total_steps - 1, 1), 1.0)
        t = t * t * (3.0 - 2.0 * t)  # smoothstep
        self._cam.azimuth = start["azimuth"] + t * (azimuth - start["azimuth"])
        self._cam.elevation = start["elevation"] + t * (elevation - start["elevation"])
        d0 = start["distance_scale"] * distance
        self._cam.distance = d0 + t * (distance - d0)

    def render(self) -> np.ndarray:
        scene = self._renderer.scene
        if self._sweep is not None:
            self._sweep_camera()
        self._frame += 1
        for env_id, offset in enumerate(self._offsets):
            qpos, qvel, targets, joint_errors = self._state_fn(env_id)
            self._data.qpos[:] = qpos
            self._data.qvel[:] = qvel
            mujoco.mj_forward(self._model, self._data)
            for site_id, pos in targets.items():
                self._data.site_xpos[site_id] = pos
            if env_id == 0:
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
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()


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


def _mjlab_shows_targets(env) -> bool:
    """Pose and reach envs have target markers; locomotion envs do not."""
    return bool({"pose", "reach"} & set(env.command_manager.active_terms))


def _mjlab_velocity_fn(env):
    """``fn(env_id) -> target planar velocity`` of a walking env, else ``None``.

    Read from the parameters of the env's success metric (the walking tasks' solved
    criterion is a velocity match).
    """
    term = env.metrics_manager.cfg.get("success")
    if term is None:
        return None
    params = {
        name: p.default
        for name, p in inspect.signature(term.func).parameters.items()
        if p.default is not inspect.Parameter.empty
    } | dict(term.params)
    if "target_y_vel" in params:
        velocity = np.array([params["target_x_vel"], params["target_y_vel"]])
        return lambda i: velocity
    if "heading_dir" not in params:
        return None
    if not params.get("randomized"):
        velocity = params["target_speed"] * np.asarray(params["heading_dir"], float)
        return lambda i: velocity
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        _directional_cmd_buffer,
    )

    return lambda i: (
        params["target_speed"] * _directional_cmd_buffer(env)[i].cpu().numpy()
    )


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


def _write_video(cfg: EvalConfig, frames: list[np.ndarray], step_dt: float) -> None:
    import imageio

    cfg.video.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(cfg.video, frames, fps=int(round(1.0 / step_dt)))
    print(f"video:    {cfg.video}")


def _check_stochastic(cfg: EvalConfig, policy) -> None:
    """Fail early if ``--stochastic`` is set but the checkpoint has no action std."""
    if cfg.stochastic and policy.std is None:
        raise SystemExit(
            "--stochastic: the checkpoint has no action std to sample from."
        )


def evaluate_cpu(cfg: EvalConfig, checkpoint: Path) -> None:
    """Roll out ``cols x rows`` CPU envs in lockstep, ``episodes_per_env`` each."""
    import gymnasium as gym

    import myosuite  # noqa: F401  (registers the CPU envs)
    from myosuite.utils.rslrl_policy import load_rslrl_policy

    cols, rows = _grid_shape(cfg)
    n_envs, per_env = cols * rows, _episodes_per_env(cfg)
    envs = [gym.make(cfg.env_id) for _ in range(n_envs)]
    policy = load_rslrl_policy(checkpoint, envs[0].action_space.shape[0])
    _check_stochastic(cfg, policy)
    grid = (
        GridRenderer(
            envs[0].unwrapped.model,
            _cpu_state_fn(envs),
            cfg,
            per_env * (envs[0].spec.max_episode_steps or 1000),
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
    obs = np.stack(
        [env.reset(seed=cfg.seed + i * per_env)[0] for i, env in enumerate(envs)]
    )
    ep_return, ep_length = np.zeros(n_envs), np.zeros(n_envs, dtype=int)
    done_count = np.zeros(n_envs, dtype=int)
    while (done_count < per_env).any():
        actions = policy.act(obs, cfg.stochastic)
        for i, env in enumerate(envs):
            if done_count[i] >= per_env:
                continue
            obs[i], rew, terminated, truncated, info = env.step(actions[i])
            ep_return[i] += float(rew)
            ep_length[i] += 1
            if terminated or truncated:
                returns.append(ep_return[i])
                lengths.append(int(ep_length[i]))
                if "solved" in info:
                    successes.append(float(bool(info["solved"])))
                ep_return[i], ep_length[i] = 0.0, 0
                done_count[i] += 1
                if done_count[i] < per_env:
                    seed = cfg.seed + i * per_env + int(done_count[i])
                    obs[i] = env.reset(seed=seed)[0]
        if grid is not None:
            frames.append(grid.render())
    step_dt = getattr(envs[0].unwrapped, "dt", None) or envs[0].unwrapped._ctrl_dt
    for env in envs:
        env.close()
    _summary(returns, lengths, successes)
    if grid is not None:
        grid.close()
        _write_video(cfg, frames, step_dt)


def evaluate_mjlab(cfg: EvalConfig, checkpoint: Path) -> None:
    """Roll out ``cols x rows`` parallel mjlab envs, ``episodes_per_env`` each."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import myosuite.envs.myo.backends.mjlab  # noqa: F401  (registers the twins)
    from myosuite.utils.rslrl_policy import load_rslrl_policy

    os.environ.setdefault("MUJOCO_GL", "egl")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg(cfg.env_id, play=True)
    cols, rows = _grid_shape(cfg)
    n_envs, per_env = cols * rows, _episodes_per_env(cfg)
    env_cfg.scene.num_envs = n_envs
    env_cfg.seed = cfg.seed
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    grid = (
        GridRenderer(
            env.sim.mj_model,
            _mjlab_state_fn(env),
            cfg,
            per_env * env.max_episode_length,
            pose_task="pose" in env.command_manager.active_terms,
            marker_radius=_mjlab_marker_radius(env),
            pose_thd=_mjlab_pose_thd(env),
            show_targets=_mjlab_shows_targets(env),
            velocity_fn=_mjlab_velocity_fn(env),
        )
        if cfg.video
        else None
    )
    frames: list[np.ndarray] = []
    policy = load_rslrl_policy(checkpoint, env.action_manager.total_action_dim).to(
        device
    )
    _check_stochastic(cfg, policy)
    has_success = "success" in env.metrics_manager.active_terms

    obs, _ = env.reset()
    ep_return = torch.zeros(n_envs, device=device)
    ep_length = torch.zeros(n_envs, dtype=torch.long, device=device)
    episodes_done = torch.zeros(n_envs, dtype=torch.long, device=device)
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    with torch.no_grad():
        for _ in range(per_env * env.max_episode_length):
            act = policy.sample if cfg.stochastic else policy
            obs, rew, terminated, truncated, _ = env.step(act(obs["actor"]))
            if grid is not None:
                frames.append(grid.render())
            recording = episodes_done < per_env
            ep_return += rew * recording
            ep_length += recording.long()
            finished = (terminated | truncated) & recording
            returns += ep_return[finished].tolist()
            lengths += ep_length[finished].tolist()
            if has_success:  # final-step value of the standard success metric
                for i in finished.nonzero().flatten().tolist():
                    values = dict(env.metrics_manager.get_active_iterable_terms(i))
                    successes.append(float(values["success"][0]))
            ep_return[finished] = 0.0
            ep_length[finished] = 0
            episodes_done += finished.long()
            if bool((episodes_done >= per_env).all()):
                break
    step_dt = env.step_dt
    env.close()
    _summary(returns, lengths, successes)
    if grid is not None:
        grid.close()
        _write_video(cfg, frames, step_dt)


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
