#!/usr/bin/env python3
"""Headless PPO benchmark for mjlab MuscleMimic fullbody (clip mode).

Compares bare ``RslRlOnPolicyRunnerCfg`` against
:func:`default_mimic_clip_on_policy_runner_cfg`.  Installs
:func:`install_episode_reward_logging_patch` so episode returns are
recorded when ``log_dir=None`` (avoids broken ``tensorboard`` / TF stacks).

Usage::

    python scripts/bench_mjlab_mimic_ppo.py

Env:
    MIMIC_CLIP: optional path to retargeted .npz.
    BENCH_ITERS, BENCH_NUM_ENVS, BENCH_STEPS_PER_ENV, BENCH_SEED: tuning.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import importlib
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

_CLIP_REPO_ID = "amathislab/musclemimic-retargeted"
_CLIP_FILENAME = "MyoFullBody/gmr/KIT/167/walking_medium06_poses.npz"


def _repo_root() -> Path:
    p = Path(__file__).resolve().parent.parent
    if not (p / "myosuite").is_dir():
        raise RuntimeError("Run from myosuite repo root")
    return p


def _default_clip() -> Path:
    env = os.environ.get("MIMIC_CLIP")
    if env:
        return Path(env).expanduser()
    hf_hub_download = importlib.import_module("huggingface_hub").hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=_CLIP_REPO_ID,
            filename=_CLIP_FILENAME,
            repo_type="dataset",
        )
    ).expanduser()


def _build_runner_cfg(
    *,
    num_steps_per_env: int,
    max_iterations: int,
    mode: str,
) -> dict[str, Any]:
    from mjlab.rl import RslRlOnPolicyRunnerCfg

    if mode == "tuned":
        from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
            default_mimic_clip_on_policy_runner_cfg,
        )

        cfg = dataclasses.asdict(default_mimic_clip_on_policy_runner_cfg())
    else:
        cfg = dataclasses.asdict(RslRlOnPolicyRunnerCfg())
    cfg["upload_model"] = False
    cfg["num_steps_per_env"] = num_steps_per_env
    cfg["max_iterations"] = max_iterations
    cfg["save_interval"] = max_iterations + 1
    return cfg


def _mean_reward_from_logger(logger: Any) -> float:
    buf = getattr(logger, "rewbuffer", None)
    if not buf or len(buf) == 0:
        return float("nan")
    return float(statistics.mean(buf))


def run_once(
    *,
    clip_path: Path,
    device: str,
    num_envs: int,
    num_steps_per_env: int,
    max_iterations: int,
    mode: str,
    seed: int,
) -> float:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, register_mjlab_task

    from myosuite.core.trajectory_io import load_motion_clip
    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
        register_mimic_mjlab_tasks_with_clip,
    )
    from myosuite.envs.myo.backends.mjlab.rsl_rl_logger_episode_patch import (
        install_episode_reward_logging_patch,
    )

    torch.manual_seed(seed)
    install_episode_reward_logging_patch()

    clip = load_motion_clip(clip_path, expected_nq=89, expected_nv=88)
    register_mimic_mjlab_tasks_with_clip(
        register_mjlab_task=register_mjlab_task,
        rl_cfg_fn=lambda: __import__(
            "mjlab.rl", fromlist=["RslRlOnPolicyRunnerCfg"]
        ).RslRlOnPolicyRunnerCfg(),
        clip=clip,
        use_lookahead=True,
    )
    env_cfg = load_env_cfg("myoMimicFullbody-v0")
    env_cfg.scene.num_envs = num_envs
    train_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(train_env)

    runner_cfg = _build_runner_cfg(
        num_steps_per_env=num_steps_per_env,
        max_iterations=max_iterations,
        mode=mode,
    )
    runner_cfg["seed"] = seed

    runner = MjlabOnPolicyRunner(
        env=wrapped,
        train_cfg=runner_cfg,
        log_dir=None,
        device=device,
    )
    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)
    return _mean_reward_from_logger(runner.logger)


def main() -> int:
    repo = _repo_root()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "tutorials"))

    clip_path = _default_clip()
    if not clip_path.is_file():
        print(f"SKIP: clip not found at {clip_path}")
        return 0

    if importlib.util.find_spec("mjlab") is None:
        print("SKIP: mjlab not installed")
        return 0

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = int(os.environ.get("BENCH_NUM_ENVS", "6"))
    num_steps_per_env = int(os.environ.get("BENCH_STEPS_PER_ENV", "48"))
    max_iterations = int(os.environ.get("BENCH_ITERS", "55"))
    seed = int(os.environ.get("BENCH_SEED", "42"))

    # Ensure enough steps per env to finish ≥1 timeout episode (1000 ctrl steps).
    steps_per_env_total = max_iterations * num_steps_per_env
    if steps_per_env_total < 1100:
        print(
            f"WARN: BENCH_ITERS*BENCH_STEPS_PER_ENV={steps_per_env_total} < 1100; "
            "rewbuffer may stay empty."
        )

    print(
        f"Bench: device={device} envs={num_envs} steps/env={num_steps_per_env} "
        f"iters={max_iterations} clip={clip_path.name}"
    )
    base = run_once(
        clip_path=clip_path,
        device=device,
        num_envs=num_envs,
        num_steps_per_env=num_steps_per_env,
        max_iterations=max_iterations,
        mode="baseline",
        seed=seed,
    )
    tuned = run_once(
        clip_path=clip_path,
        device=device,
        num_envs=num_envs,
        num_steps_per_env=num_steps_per_env,
        max_iterations=max_iterations,
        mode="tuned",
        seed=seed,
    )

    print(f"Mean episode return (last ≤100 eps) — baseline: {base:.3f}")
    print(f"Mean episode return — tuned:          {tuned:.3f}")

    # Short runs: use ~0.12–0.2; longer runs can tighten (e.g. 0.35).
    margin = float(os.environ.get("BENCH_MARGIN", "0.15"))
    if math.isnan(base) or math.isnan(tuned):
        print("FAIL: nan metrics (increase BENCH_ITERS or check env resets)")
        return 1
    if tuned > base + margin:
        print(f"PASS: tuned ({tuned:.3f}) > baseline + {margin} ({base:.3f})")
        return 0

    print(
        f"FAIL: expected tuned > baseline+{margin}; baseline={base:.3f} tuned={tuned:.3f}"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
