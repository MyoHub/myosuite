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
    """CPU only: write an MP4 of the rollouts (offscreen rendering)."""
    camera: str = "-1"
    """Video camera: a camera name from the model, or an id (``-1`` = free camera)."""
    width: int = 640
    """Video frame width."""
    height: int = 480
    """Video frame height."""
    distance: float | None = None
    """Free camera only: distance to the look-at point (default: MuJoCo's)."""
    azimuth: float | None = None
    """Free camera only: azimuth in degrees."""
    elevation: float | None = None
    """Free camera only: elevation in degrees."""
    lookat: tuple[float, float, float] | None = None
    """Free camera only: look-at point in world coordinates."""


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
    env_cfg.scene.num_envs = cfg.episodes
    env_cfg.seed = cfg.seed
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    policy = load_rslrl_policy(checkpoint, env.action_manager.total_action_dim).to(
        device
    )

    obs, _ = env.reset()
    totals = torch.zeros(cfg.episodes, device=device)
    lengths = torch.zeros(cfg.episodes, dtype=torch.long, device=device)
    active = torch.ones(cfg.episodes, dtype=torch.bool, device=device)
    with torch.no_grad():
        for _ in range(env.max_episode_length):
            obs, rew, terminated, truncated, _ = env.step(policy(obs["actor"]))
            totals += rew * active
            lengths += active.long()
            active &= ~(terminated | truncated)
            if not active.any():
                break
    env.close()
    _summary(totals.cpu().tolist(), lengths.cpu().tolist(), {})


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
