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
    show_tendons: bool = False
    """Draw the muscle tendons (adds ~150 geoms per env for the arm model)."""
    floor: bool = True
    """Video: draw a light checker-grid floor with soft shadows and a fading
    white horizon under all envs (independent of ``--show-scene``)."""
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


def _target_site_ids(model: mujoco.MjModel) -> list[int]:
    """Ids of the target marker sites (``*_target``, ignoring an entity prefix)."""
    names = [model.site(i).name for i in range(model.nsite)]
    return [i for i, n in enumerate(names) if n.split("/")[-1].endswith("_target")]


def _render_option(
    model: mujoco.MjModel, show_scene: bool, show_tendons: bool
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
    hidden = (
        (model.geom_bodyid == 0)
        | (model.body_mocapid[model.geom_bodyid] >= 0)
        | (model.geom_type != mujoco.mjtGeom.mjGEOM_MESH)
    )
    model.geom_group[hidden] = _HIDDEN_GROUP
    opt.geomgroup[_HIDDEN_GROUP] = 0
    targets = _target_site_ids(model)
    model.site_group[:] = _HIDDEN_GROUP
    model.site_group[targets] = 0
    opt.sitegroup[:] = 0
    opt.sitegroup[0] = 1
    for i in targets:
        if TARGET_SITE_RGBA is not None:
            model.site_rgba[i] = TARGET_SITE_RGBA
        if TARGET_SITE_RADIUS is not None:
            model.site_size[i, 0] = TARGET_SITE_RADIUS
    return opt


def _visible_bounds(
    model: mujoco.MjModel, opt: mujoco.MjvOption
) -> tuple[np.ndarray, float]:
    """Centre, largest extent and lowest z of the visible geoms (default pose)."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    visible = np.array(opt.geomgroup, dtype=bool)[model.geom_group]
    visible &= model.geom_rgba[:, 3] > 0
    radius = model.geom_rbound[visible, None]
    lo = (data.geom_xpos[visible] - radius).min(axis=0)
    hi = (data.geom_xpos[visible] + radius).max(axis=0)
    return (lo + hi) / 2.0, float((hi - lo).max()), float(lo[2])


def _grid_offsets(cols: int, rows: int, spacing: float) -> np.ndarray:
    """World offsets ``(cols * rows, 3)`` of a centred grid, x = column, y = row."""
    n = cols * rows
    idx = np.arange(n)
    xy = np.stack([idx % cols, idx // cols], axis=1).astype(float)
    xy -= (xy.max(axis=0)) / 2.0
    return np.concatenate([xy * spacing, np.zeros((n, 1))], axis=1)


EnvState = tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]
"""``(qpos, qvel, {target site id: xyz})`` of one env."""


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
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        state_fn,
        cfg: EvalConfig,
        total_steps: int = 1,
    ) -> None:
        cols, rows = _grid_shape(cfg)
        n_envs = cols * rows
        self._state_fn = state_fn
        self._model = model
        self._data = mujoco.MjData(model)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, cfg.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, cfg.height)
        self._opt = _render_option(model, cfg.show_scene, cfg.show_tendons)
        if cfg.floor:
            model.vis.rgba.fog[:] = FOG_RGBA
            model.vis.rgba.haze[:] = FOG_RGBA
            model.vis.quality.offsamples = 8  # anti-alias thin far grid lines
        self._pert = mujoco.MjvPerturb()
        center, size, floor_z = _visible_bounds(model, self._opt)
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
        max_geom = max(10000, int(1.5 * probe.ngeom * n_envs) + 1000 + floor_geoms)
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
            qpos, qvel, targets = self._state_fn(env_id)
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
        if self._floor_z is not None:
            _add_floor(scene, self._floor_z, self._floor_half, self._horizon)
            scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
            scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
            scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()


def _mjlab_state_fn(env):
    """State source of an mjlab env; reach targets come from the reach command."""
    model = env.sim.mj_model
    target_ids = {model.site(i).name.split("/")[-1]: i for i in _target_site_ids(model)}

    def state(env_id: int) -> EnvState:
        data = env.sim.data
        targets: dict[int, np.ndarray] = {}
        cmds = env.command_manager
        if "reach" in cmds.active_terms:  # pose tasks have no target position
            term = cmds.get_term("reach")
            xyz = term.command[env_id].cpu().numpy().reshape(-1, 3)
            for tip, pos in zip(term.cfg.tip_sites, xyz, strict=True):
                if f"{tip}_target" in target_ids:
                    targets[target_ids[f"{tip}_target"]] = pos
        return (
            data.qpos[env_id].cpu().numpy(),
            data.qvel[env_id].cpu().numpy(),
            targets,
        )

    return state


def _cpu_state_fn(envs):
    """State source of a list of CPU envs (targets are moved in their data)."""
    target_ids = _target_site_ids(envs[0].unwrapped.model)

    def state(env_id: int) -> EnvState:
        data = envs[env_id].unwrapped.data
        return (
            data.qpos.copy(),
            data.qvel.copy(),
            {i: data.site_xpos[i].copy() for i in target_ids},
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

    imageio.mimsave(cfg.video, frames, fps=int(round(1.0 / step_dt)))
    print(f"video:    {cfg.video}")


def evaluate_cpu(cfg: EvalConfig, checkpoint: Path) -> None:
    """Roll out ``cols x rows`` CPU envs in lockstep, ``episodes_per_env`` each."""
    import gymnasium as gym

    import myosuite  # noqa: F401  (registers the CPU envs)
    from myosuite.utils.rslrl_policy import load_rslrl_policy

    cols, rows = _grid_shape(cfg)
    n_envs, per_env = cols * rows, _episodes_per_env(cfg)
    envs = [gym.make(cfg.env_id) for _ in range(n_envs)]
    policy = load_rslrl_policy(checkpoint, envs[0].action_space.shape[0])
    grid = (
        GridRenderer(
            envs[0].unwrapped.model,
            _cpu_state_fn(envs),
            cfg,
            per_env * (envs[0].spec.max_episode_steps or 1000),
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
        actions = policy.act(obs)
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
    step_dt = envs[0].unwrapped.dt
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
        )
        if cfg.video
        else None
    )
    frames: list[np.ndarray] = []
    policy = load_rslrl_policy(checkpoint, env.action_manager.total_action_dim).to(
        device
    )
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
            obs, rew, terminated, truncated, _ = env.step(policy(obs["actor"]))
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
    print(f"checkpoint: {checkpoint}  backend: {cfg.backend}  env: {cfg.env_id}")
    if cfg.backend == "cpu":
        evaluate_cpu(cfg, checkpoint)
    else:
        evaluate_mjlab(cfg, checkpoint)


if __name__ == "__main__":
    main()
