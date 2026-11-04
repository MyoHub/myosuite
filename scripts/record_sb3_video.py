"""Record an MP4 of a policy saved by ``scripts/train_sb3.py`` (or of the env alone).

Loads ``<model>.zip`` (PPO / SAC / TD3, detected automatically) and, if present,
``<model>_vecnormalize.pkl``; rolls out the deterministic policy and writes a video
with the step, return and (for boxing) hit counts drawn on each frame::

    # trained model (default path written by train_sb3.py)
    python scripts/record_sb3_video.py myoChallengeBoxingVsLevel1-v0
    python scripts/record_sb3_video.py myoChallengeBoxingVsLevel1-v0 --model logs/sb3/myoChallengeBoxingVsLevel1-v0/model.zip

    # no model: look at the env with zero / random muscle commands
    python scripts/record_sb3_video.py myoChallengeBoxingVsLevel1-v0 --policy zero

Videos go to ``logs/videos/<env_id>/`` by default (``logs/`` and ``*.mp4`` are
git-ignored: share them through the drive).  Works with envs exposing ``model``
and ``data`` on ``env.unwrapped`` (the boxing and chase-tag envs).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

import myosuite  # noqa: F401  (registers the envs)
from myosuite import make_env
from myosuite.utils.video_io import add_text_to_frame, write_video


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("env_id", nargs="?", default="myoChallengeBoxingVsLevel1-v0")
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="SB3 .zip (default: logs/sb3/<env_id>/model.zip).",
    )
    p.add_argument(
        "--policy",
        choices=("model", "zero", "random"),
        default="model",
        help="'zero'/'random' record the env without a trained model.",
    )
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stochastic", action="store_true", help="Sample actions.")
    p.add_argument("--out", type=Path, default=None, help="Output .mp4 path.")
    p.add_argument("--fps", type=int, default=50, help="Video frame rate.")
    p.add_argument(
        "--speed", type=float, default=1.0, help="Playback speed (0.5 = slow-mo)."
    )
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--azimuth", type=float, default=0.0)
    p.add_argument("--elevation", type=float, default=-12.0)
    p.add_argument("--distance", type=float, default=3.2)
    p.add_argument("--no-text", action="store_true", help="No overlay.")
    return p.parse_args(argv)


def load_model(path: Path, env: gym.Env) -> tuple[Any, Any]:
    """Load an SB3 model (any of PPO/SAC/TD3) and its VecNormalize stats."""
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    if not path.is_file():
        raise SystemExit(
            f"No model at {path}. Train one with scripts/train_sb3.py, pass "
            "--model, or use --policy zero/random to record the env alone."
        )
    errors = []
    for algo in (PPO, SAC, TD3):
        try:
            model = algo.load(str(path), device="cpu")
            break
        except Exception as exc:  # noqa: BLE001  (wrong algorithm class)
            errors.append(f"{algo.__name__}: {exc}")
    else:
        raise SystemExit("Could not load the model:\n" + "\n".join(errors))

    vecnorm = None
    stats = Path(str(path).removesuffix(".zip") + "_vecnormalize.pkl")
    if stats.is_file():
        vecnorm = VecNormalize.load(str(stats), DummyVecEnv([lambda: env]))
        vecnorm.training = False
        print(f"Using observation normalisation from {stats}")
    return model, vecnorm


def _lookat(env: gym.Env) -> np.ndarray:
    """Midpoint between the two boxers' pelvises, else the model centre."""
    u = env.unwrapped
    meta = getattr(u, "_meta", None)
    sites = getattr(meta, "site_ids", None) or {}
    pts = [
        u.data.site_xpos[s["pelvis_site"]] for s in sites.values() if "pelvis_site" in s
    ]
    if pts:
        center = np.mean(pts, axis=0)
        center[2] = 1.0
        return center
    return np.array(u.model.stat.center)


def _overlay(info: dict[str, Any], step: int, ret: float, episode: int) -> str:
    lines = [f"episode {episode}  step {step}  return {ret:+.2f}"]
    if "hits_head/agent_0" in info:
        lines.append(
            "hits head {:.0f}  body {:.0f}  blocked {:.0f}".format(
                *(info["_totals"][k] for k in ("hits_head", "hits_body", "blocked"))
            )
        )
    if info.get("fell/agent_0"):
        lines.append("FELL")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> Path:
    """Roll out and write the video; returns its path."""
    args = parse_args(argv)
    env = make_env(args.env_id)
    u = env.unwrapped
    if not hasattr(u, "model") or not hasattr(u, "data"):
        raise SystemExit(f"{args.env_id}: env.unwrapped has no model/data to render.")

    model = vecnorm = None
    if args.policy == "model":
        model_path = args.model or Path("logs") / "sb3" / args.env_id / "model.zip"
        model, vecnorm = load_model(model_path, env)
        default_name = model_path.stem
    else:
        default_name = args.policy
    out = args.out or Path("logs") / "videos" / args.env_id / f"{default_name}.mp4"

    m = u.model
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), args.width)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), args.height)
    renderer = mujoco.Renderer(m, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = (
        args.azimuth,
        args.elevation,
        args.distance,
    )

    dt = float(getattr(u, "dt", 0.01))
    every = max(1, round(1.0 / (dt * args.fps * args.speed)))
    frames: list[np.ndarray] = []
    env.action_space.seed(args.seed)  # reproducible --policy random

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        ret, totals = 0.0, {"hits_head": 0.0, "hits_body": 0.0, "blocked": 0.0}
        for step in range(args.max_steps):
            if model is not None:
                net_obs = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
                action, _ = model.predict(net_obs, deterministic=not args.stochastic)
            elif args.policy == "random":
                action = env.action_space.sample()
            else:
                space = env.action_space
                action = (
                    {k: np.zeros(s.shape, np.float32) for k, s in space.spaces.items()}
                    if isinstance(space, gym.spaces.Dict)
                    else np.zeros(space.shape, np.float32)
                )
            obs, reward, terminated, truncated, info = env.step(action)
            ret += (
                float(reward)
                if not isinstance(reward, dict)
                else float(next(iter(reward.values())))
            )
            for k in totals:
                totals[k] += float(info.get(f"{k}/agent_0", 0.0))
            done = (
                any(terminated.values()) or any(truncated.values())
                if isinstance(terminated, dict)
                else terminated or truncated
            )
            if step % every == 0 or done:
                cam.lookat[:] = _lookat(env)
                renderer.update_scene(u.data, camera=cam)
                frame = renderer.render().copy()
                if not args.no_text:
                    info["_totals"] = totals
                    frame = np.asarray(
                        add_text_to_frame(
                            frame,
                            _overlay(info, step + 1, ret, ep),
                            color=(255, 255, 255),
                            fontsize=20,
                        )
                    )
                frames.append(frame)
            if done:
                break
        print(f"episode {ep}: {step + 1} steps, return {ret:.2f}, hits {totals}")
        # Hold the last frame for half a second between episodes.
        frames.extend([frames[-1]] * max(1, args.fps // 2))

    renderer.close()
    env.close()
    write_video(out, frames, fps=args.fps)
    print(f"Saved {out}")
    return out


if __name__ == "__main__":
    main()
