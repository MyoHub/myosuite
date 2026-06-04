#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Render SAR benchmark solutions (trained policies) as videos.

Loads checkpoints from a Phase B training run and runs deterministic rollouts
on the CPU myoLegWalk env with rendering, saving one video per backend.
Only backends that have a saved solution (checkpoint) are rendered.

Usage::

    # Render only the solutions (one video per backend that has a checkpoint)
    python benchmarks/sar_backends/render_benchmark_results.py

    # Custom results dir and number of steps
    python benchmarks/sar_backends/render_benchmark_results.py \\
        --results-dir ./benchmark_results \\
        --steps 500

    # Only render specific backends
    python benchmarks/sar_backends/render_benchmark_results.py --backends cpu mjlab

Output: each backend that has a saved solution gets ``render_rollout.mp4`` in
its run directory (e.g. phase_b_training/cpu/render_rollout.mp4).
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys

import numpy as np

# Repo root on path
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Default paths
_SAR_DIR = _REPO_ROOT / "myosuite" / "agents" / "SAR_pretrained" / "locomotion"

# Frame size for offscreen rendering (myoLegWalk uses legacy env without rgb_array)
_RENDER_WIDTH, _RENDER_HEIGHT = 640, 480


def _get_frame_from_env(env, vec_env=None) -> np.ndarray | None:
    """Get an RGB frame from a MyoSuite env.

    myoLegWalk-v0 uses the legacy MujocoEnv chain which does not implement
    render(); we use the unwrapped env's mj_renderer.render_offscreen() instead.
    """
    if vec_env is not None:
        try:
            inner = getattr(vec_env, "venv", vec_env)
            env = inner.envs[0]
        except Exception:
            env = None
    if env is None:
        return None
    base = env
    while hasattr(base, "env"):
        base = base.env
    if hasattr(base, "mj_renderer") and base.mj_renderer is not None:
        try:
            return base.mj_renderer.render_offscreen(
                width=_RENDER_WIDTH, height=_RENDER_HEIGHT, rgb=True
            )
        except Exception:
            return None
    try:
        out = env.render()
        if out is not None and isinstance(out, np.ndarray) and out.ndim >= 2:
            return out if out.ndim == 3 else out[0]
    except NotImplementedError:
        pass
    return None


def _save_video(frames: list[np.ndarray], path: pathlib.Path, fps: int = 50) -> bool:
    """Write frames to an mp4 file. Returns True on success."""
    if not frames:
        return False
    try:
        import imageio.v3 as iio

        iio.imwrite(str(path), frames, fps=fps, codec="libx264")
        return True
    except Exception:
        try:
            import imageio

            with imageio.get_writer(str(path), fps=fps) as writer:
                for f in frames:
                    writer.append_data(f)
            return True
        except Exception as exc:
            log.warning("imageio failed; saving frames as npz: %s", exc)
            np.savez_compressed(str(path.with_suffix(".npz")), frames=np.stack(frames))
            return False


def _render_sb3_backend(
    run_dir: pathlib.Path,
    ica,
    pca,
    normalizer,
    n_steps: int,
    seed: int,
) -> pathlib.Path | None:
    """Render SAC (CPU or MJX) policy on CPU env; save video to run_dir/render_rollout.mp4."""
    import gymnasium as gym
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from benchmarks.sar_backends.run_benchmark import synergy_inverse_transform

    sac_path = run_dir / "sac_model.zip"
    if not sac_path.exists():
        sac_path = run_dir / "sac_model"
    if not sac_path.exists():
        log.warning("No SAC checkpoint at %s", run_dir)
        return None

    vec_norm_path = run_dir / "vec_normalize.pkl"
    if not vec_norm_path.exists():
        log.warning("No vec_normalize.pkl at %s", run_dir)
        return None

    import myosuite

    myosuite.register_all_envs()

    n_syn = pca.components_.shape[0]

    class SynergyWrapper(gym.ActionWrapper):
        def __init__(self, env, ica, pca, normalizer):
            super().__init__(env)
            self._ica, self._pca, self._norm = ica, pca, normalizer
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(n_syn,), dtype=np.float32
            )

        def action(self, act):
            return synergy_inverse_transform(act, self._ica, self._pca, self._norm)

    base_env = gym.make("myoLegWalk-v0", reset_type="init")
    wrapped = SynergyWrapper(base_env, ica, pca, normalizer)
    vec_env = DummyVecEnv([lambda: wrapped])
    norm_env = VecNormalize.load(str(vec_norm_path), vec_env)
    norm_env.training = False
    norm_env.norm_reward = False

    model = SAC.load(str(sac_path), env=norm_env)

    try:
        result = norm_env.reset(seed=seed)
    except TypeError:
        result = norm_env.reset()
    obs = (
        result[0] if isinstance(result, (tuple, list)) and len(result) >= 2 else result
    )

    frames = []
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        step_out = norm_env.step(action)
        # SB3/gym can return (obs, r, done, info) or (obs, r, term, trunc, info)
        if len(step_out) == 5:
            obs, _, terminated, truncated, _ = step_out
        else:
            obs, _, done, _ = step_out
            terminated = np.asarray(done).reshape(-1)
            truncated = np.zeros_like(terminated, dtype=bool)
        frame = _get_frame_from_env(None, vec_env=norm_env)
        if frame is not None:
            frames.append(frame if frame.ndim == 3 else frame[0])
        if np.any(terminated) or np.any(truncated):
            try:
                result = norm_env.reset(seed=seed)
            except TypeError:
                result = norm_env.reset()
            obs = (
                result[0]
                if isinstance(result, (tuple, list)) and len(result) >= 2
                else result
            )

    norm_env.close()
    out_path = run_dir / "render_rollout.mp4"
    if _save_video(frames, out_path):
        log.info("Saved %s (%d frames)", out_path, len(frames))
        return out_path
    return None


def _render_mjlab_backend(
    run_dir: pathlib.Path,
    sar_dir: pathlib.Path,
    n_steps: int,
    seed: int,
) -> pathlib.Path | None:
    """Render mjlab PPO policy on CPU env; save video to run_dir/render_rollout.mp4."""
    import torch

    import gymnasium as gym
    import myosuite

    myosuite.register_all_envs()
    from tensordict import TensorDict

    from benchmarks.sar_backends.run_benchmark import load_mjlab_policy_and_sar

    try:
        policy, sar_transform = load_mjlab_policy_and_sar(
            run_dir, sar_dir, device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
    except Exception as exc:
        log.warning("Could not load mjlab policy from %s: %s", run_dir, exc)
        return None

    env = gym.make("myoLegWalk-v0", reset_type="init")
    obs_cpu, _ = env.reset(seed=seed)
    device = next(policy.parameters()).device
    obs_t = torch.as_tensor(obs_cpu, dtype=torch.float32, device=device).unsqueeze(0)
    # RSL-RL policy expects TensorDict with observation group key "policy"
    obs_td = TensorDict({"policy": obs_t}, batch_size=[1])

    frames = []
    for _ in range(n_steps):
        with torch.no_grad():
            if hasattr(policy, "act_inference"):
                syn_action = policy.act_inference(obs_td)
            else:
                syn_action = policy.act(obs_td)[0]
            muscle_action = sar_transform(syn_action)

        obs_cpu, _, terminated, truncated, _ = env.step(
            muscle_action.squeeze(0).cpu().numpy()
        )
        frame = _get_frame_from_env(env, vec_env=None)
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            obs_cpu, _ = env.reset()
        obs_t = torch.as_tensor(obs_cpu, dtype=torch.float32, device=device).unsqueeze(
            0
        )
        obs_td = TensorDict({"policy": obs_t}, batch_size=[1])

    env.close()
    out_path = run_dir / "render_rollout.mp4"
    if _save_video(frames, out_path):
        log.info("Saved %s (%d frames)", out_path, len(frames))
        return out_path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render Phase B solutions (trained policies) as videos (one per backend)."
    )
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        default=_REPO_ROOT / "sar_benchmark_results",
        help="Root benchmark output directory (contains phase_b_training/).",
    )
    parser.add_argument(
        "--sar-dir",
        type=pathlib.Path,
        default=_SAR_DIR,
        help="Path to SAR pkl files (ica.pkl, pca.pkl, normalizer.pkl).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of steps to render per backend.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for rollout.",
    )
    parser.add_argument(
        "--backends",
        nargs="*",
        choices=["cpu", "mjx_xla", "mjx_warp", "mjlab"],
        default=None,
        help="Backends to render (default: all that have a saved solution).",
    )
    args = parser.parse_args()

    # Enable headless rendering when no DISPLAY (e.g. SSH, CI) so videos are still produced
    if not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = "egl"
        log.info("No DISPLAY set; using MUJOCO_GL=egl for offscreen rendering")

    phase_b_dir = args.results_dir / "phase_b_training"
    if not phase_b_dir.exists():
        log.error("Phase B dir not found: %s", phase_b_dir)
        return 1

    if not args.sar_dir.exists():
        log.error("SAR dir not found: %s", args.sar_dir)
        return 1

    from benchmarks.sar_backends.run_benchmark import load_sar

    ica, pca, normalizer = load_sar(args.sar_dir)

    backends = args.backends
    if backends is None:
        backends = ["cpu", "mjx_xla", "mjx_warp", "mjlab"]

    rendered = []
    for backend in backends:
        run_dir = phase_b_dir / backend
        if not run_dir.is_dir():
            log.debug("Skipping %s (no run dir)", backend)
            continue
        log.info("Rendering %s → %s", backend, run_dir)
        try:
            if backend == "mjlab":
                path = _render_mjlab_backend(
                    run_dir, args.sar_dir, args.steps, args.seed
                )
            else:
                path = _render_sb3_backend(
                    run_dir, ica, pca, normalizer, args.steps, args.seed
                )
            if path is not None:
                rendered.append((backend, str(path)))
        except Exception as exc:
            log.exception("Rendering %s failed: %s", backend, exc)

    if not rendered:
        log.warning("No videos produced.")
        return 1
    log.info("Rendered %d backend(s): %s", len(rendered), [r[0] for r in rendered])
    return 0


if __name__ == "__main__":
    sys.exit(main())
