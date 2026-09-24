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

    # mjlab backend: all parallel envs side by side in one video
    python scripts/eval_mjlab_policy.py myoElbowPose1D6MRandom-v0 --checkpoint RUN \\
        --backend mjlab --num-cols 4 --num-rows 2 --video grid.mp4 --width 1280 --height 720

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
    """Number of evaluation episodes."""
    seed: int = 0
    """Seed of the first episode (CPU) / the mjlab env."""
    video: Path | None = None
    """Write an MP4 of the rollouts (offscreen rendering). With ``--backend mjlab``
    the parallel envs (``--num-cols`` x ``--num-rows``) are laid out on a grid in
    one shared scene."""
    num_cols: int | None = None
    """mjlab only: parallel envs along the horizontal axis of the video grid.
    Setting ``--num-cols`` and/or ``--num-rows`` runs ``cols * rows`` envs
    (a missing one defaults to 1) instead of ``--episodes`` (square-ish grid)."""
    num_rows: int | None = None
    """mjlab only: parallel envs along the vertical (depth) axis of the grid."""
    env_spacing: float | None = None
    """mjlab video only: distance between neighbouring envs in metres
    (default: 0.8 x the model's extent)."""
    camera: str = "-1"
    """Video camera: a camera name from the model, or an id (``-1`` = free camera).
    The mjlab grid video always uses the free camera."""
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
    if path.is_file():
        return path
    ckpts = sorted(
        path.glob("model_*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1])
    )
    if not ckpts:
        raise FileNotFoundError(f"No model_*.pt in {path}")
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
    """``(cols, rows)`` of the mjlab env grid; square-ish for ``--episodes`` by default."""
    if cfg.num_cols is None and cfg.num_rows is None:
        cols = int(np.ceil(np.sqrt(cfg.episodes)))
        return cols, int(np.ceil(cfg.episodes / cols))
    return cfg.num_cols or 1, cfg.num_rows or 1


def _grid_offsets(cols: int, rows: int, spacing: float) -> np.ndarray:
    """World offsets ``(cols * rows, 3)`` of a centred grid, x = column, y = row."""
    n = cols * rows
    idx = np.arange(n)
    xy = np.stack([idx % cols, idx // cols], axis=1).astype(float)
    xy -= (xy.max(axis=0)) / 2.0
    return np.concatenate([xy * spacing, np.zeros((n, 1))], axis=1)


class GridRenderer:
    """Render every parallel mjlab env into one scene, laid out on a grid.

    mjlab places fixed-base robots on top of each other (env origins only move
    floating bases), so the scene is composed here: each env's state is copied
    into a host ``MjData`` and its geoms are translated by the env's grid offset.

    Args:
        env: An mjlab env (``env.sim.mj_model`` is the host model).
        cfg: Video options (size, camera, spacing).
    """

    def __init__(self, env, cfg: EvalConfig) -> None:
        self._env = env
        self._model = env.sim.mj_model
        self._data = mujoco.MjData(self._model)
        self._model.vis.global_.offwidth = max(
            self._model.vis.global_.offwidth, cfg.width
        )
        self._model.vis.global_.offheight = max(
            self._model.vis.global_.offheight, cfg.height
        )
        self._renderer = mujoco.Renderer(
            self._model, height=cfg.height, width=cfg.width
        )
        self._opt = mujoco.MjvOption()
        self._pert = mujoco.MjvPerturb()
        spacing = cfg.env_spacing or 0.8 * float(self._model.stat.extent)
        self._offsets = _grid_offsets(*_grid_shape(cfg), spacing)
        self._cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self._model, self._cam)
        grid_size = float(np.ptp(self._offsets, axis=0).max())
        self._cam.distance = (
            cfg.distance
            if cfg.distance is not None
            else self._cam.distance + 1.3 * grid_size
        )
        if cfg.azimuth is not None:
            self._cam.azimuth = cfg.azimuth
        if cfg.elevation is not None:
            self._cam.elevation = cfg.elevation
        self._cam.lookat[:] = self._cam.lookat + (cfg.lookat or (0.0, 0.0, 0.0))

    def render(self) -> np.ndarray:
        sim_data = self._env.sim.data
        qpos = sim_data.qpos.cpu().numpy()
        qvel = sim_data.qvel.cpu().numpy()
        scene = self._renderer.scene
        for env_id, offset in enumerate(self._offsets):
            self._data.qpos[:] = qpos[env_id]
            self._data.qvel[:] = qvel[env_id]
            mujoco.mj_forward(self._model, self._data)
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
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()


def _summary(returns: list[float], lengths: list[int], extra: dict[str, float]) -> None:
    print(f"episodes: {len(returns)}")
    print(f"return:   {np.mean(returns):.3f} +- {np.std(returns):.3f}")
    print(f"length:   {np.mean(lengths):.1f}")
    for key, value in extra.items():
        print(f"{key}: {value:.3f}")


def evaluate_cpu(cfg: EvalConfig, checkpoint: Path) -> None:
    import gymnasium as gym

    import myosuite  # noqa: F401  (registers the CPU envs)
    from myosuite.utils.rslrl_policy import load_rslrl_policy

    env = gym.make(cfg.env_id)
    policy = load_rslrl_policy(checkpoint, env.action_space.shape[0])
    model, data = env.unwrapped.model, env.unwrapped.data
    renderer = None
    if cfg.video:
        camera = _video_camera(cfg, model)
        # Offscreen buffer must fit the frame (visual option, no effect on physics).
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, cfg.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, cfg.height)
        renderer = mujoco.Renderer(model, height=cfg.height, width=cfg.width)
    returns, lengths, solved_end, frames = [], [], [], []
    for ep in range(cfg.episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        done, total, steps, info = False, 0.0, 0, {}
        while not done:
            obs, rew, terminated, truncated, info = env.step(policy.act(obs))
            total += float(rew)
            steps += 1
            done = terminated or truncated
            if renderer is not None:
                renderer.update_scene(data, camera=camera)
                frames.append(renderer.render())
        returns.append(total)
        lengths.append(steps)
        solved_end.append(float(bool(info.get("solved", False))))
    env.close()
    if renderer is not None:
        renderer.close()
    _summary(returns, lengths, {"solved at episode end": float(np.mean(solved_end))})
    if cfg.video:
        import imageio

        imageio.mimsave(cfg.video, frames, fps=int(round(1.0 / env.unwrapped.dt)))
        print(f"video:    {cfg.video}")


def evaluate_mjlab(cfg: EvalConfig, checkpoint: Path) -> None:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import myosuite.envs.myo.backends.mjlab  # noqa: F401  (registers the twins)
    from myosuite.utils.rslrl_policy import load_rslrl_policy

    os.environ.setdefault("MUJOCO_GL", "egl")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg(cfg.env_id, play=True)
    cols, rows = _grid_shape(cfg)
    n_envs = cols * rows
    env_cfg.scene.num_envs = n_envs
    env_cfg.seed = cfg.seed
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    grid = GridRenderer(env, cfg) if cfg.video else None
    frames = []
    policy = load_rslrl_policy(checkpoint, env.action_manager.total_action_dim).to(
        device
    )

    obs, _ = env.reset()
    totals = torch.zeros(n_envs, device=device)
    lengths = torch.zeros(n_envs, dtype=torch.long, device=device)
    active = torch.ones(n_envs, dtype=torch.bool, device=device)
    with torch.no_grad():
        for _ in range(env.max_episode_length):
            obs, rew, terminated, truncated, _ = env.step(policy(obs["actor"]))
            if grid is not None:
                frames.append(grid.render())
            totals += rew * active
            lengths += active.long()
            active &= ~(terminated | truncated)
            if not active.any():
                break
    env.close()
    _summary(totals.cpu().tolist(), lengths.cpu().tolist(), {})
    if grid is not None:
        import imageio

        grid.close()
        imageio.mimsave(cfg.video, frames, fps=int(round(1.0 / env.step_dt)))
        print(f"video:    {cfg.video}")


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
