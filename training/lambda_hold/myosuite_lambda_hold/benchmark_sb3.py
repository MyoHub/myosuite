# Copyright (c) MyoSuite Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Like-for-like PPO vs. LambdaHold+PPO benchmark on registered MyoSuite tasks."""

from __future__ import annotations

import argparse
import json
import platform
import time
import tracemalloc
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from myosuite import make_env
from myosuite.utils.action_map import ActionMapWrapper, LambdaHold


@dataclass(frozen=True)
class CurvePoint:
    """Evaluation checkpoint along a training run."""

    env_steps: int
    wall_time_s: float
    mean_return: float
    solved_rate: float


@dataclass(frozen=True)
class ArmSummary:
    """Aggregated metrics across seeds for one arm (PPO or LambdaHold+PPO)."""

    method: str
    env_id: str
    seeds: tuple[int, ...]
    total_steps: int
    final_return_mean: float
    final_return_std: float
    final_solved_mean: float
    final_solved_std: float
    wall_time_mean_s: float
    wall_time_std_s: float
    peak_tracemalloc_mb: float
    curves: list[CurvePoint]


@dataclass(frozen=True)
class BenchmarkSummary:
    """Side-by-side benchmark report across tasks and seeds."""

    hardware: str
    arms: list[ArmSummary]


def _make_single(
    env_id: str,
    seed: int,
    use_lambda_hold: bool,
    kp: float,
    damping_s: float,
    hold_steps: int,
) -> gym.Env:
    env = make_env(env_id)
    if use_lambda_hold:
        model = env.unwrapped.model
        sigmoid = bool(getattr(env.unwrapped, "normalize_act", True))
        amap = LambdaHold.from_model(
            model,
            kp=kp,
            damping_s=damping_s,
            hold_steps=hold_steps,
            muscle_sigmoid=sigmoid,
        )
        env = ActionMapWrapper(env, amap)
    env.reset(seed=seed)
    return env


def _evaluate(
    model: PPO,
    env_id: str,
    use_lambda_hold: bool,
    kp: float,
    damping_s: float,
    hold_steps: int,
    n_episodes: int,
) -> tuple[float, float]:
    env = _make_single(env_id, 10_000, use_lambda_hold, kp, damping_s, hold_steps)
    returns: list[float] = []
    solved: list[float] = []
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=10_000 + ep)
            ep_ret = 0.0
            ep_solved = False
            while True:
                act, _ = model.predict(obs, deterministic=True)
                obs, rwd, term, trunc, info = env.step(act)
                ep_ret += float(rwd)
                if bool(info.get("solved", False)):
                    ep_solved = True
                if term or trunc:
                    break
            returns.append(ep_ret)
            solved.append(1.0 if ep_solved else 0.0)
    finally:
        env.close()
    return float(np.mean(returns)), float(np.mean(solved))


def run_ppo_comparison(
    env_ids: Sequence[str] = ("myoElbowPose1D6MRandom-v0", "myoHandPoseRandom-v0"),
    seeds: Sequence[int] = (0, 1, 2),
    total_steps: int = 16_384,
    eval_interval: int = 4_096,
    n_envs: int = 8,
    n_eval_episodes: int = 20,
    kp: float = 2.5,
    damping_s: float = 0.04,
    hold_steps: int = 2,
) -> BenchmarkSummary:
    """Run PPO vs. LambdaHold+PPO across identical envs, seeds, and step budgets."""
    hw = f"{platform.system()} {platform.release()} | {platform.processor()} | Python {platform.python_version()}"
    summaries: list[ArmSummary] = []

    for env_id in env_ids:
        for use_lh, label in ((False, "PPO"), (True, "LambdaHold+PPO")):
            tracemalloc.start()
            seed_returns, seed_solved, seed_times = [], [], []
            n_chunks = max(1, total_steps // eval_interval)
            chunk_steps = total_steps // n_chunks
            all_curves: list[list[CurvePoint]] = []

            for seed in seeds:
                vec = DummyVecEnv(
                    [
                        (
                            lambda rank=r, s=seed, lh=use_lh: _make_single(
                                env_id, s * 100 + rank, lh, kp, damping_s, hold_steps
                            )
                        )
                        for r in range(n_envs)
                    ]
                )
                try:
                    ppo = PPO(
                        "MlpPolicy",
                        vec,
                        n_steps=max(64, chunk_steps // n_envs),
                        batch_size=256,
                        n_epochs=5,
                        learning_rate=3e-4,
                        seed=seed,
                        device="cpu",
                        verbose=0,
                    )
                    curve: list[CurvePoint] = []
                    r0, s0 = _evaluate(
                        ppo, env_id, use_lh, kp, damping_s, hold_steps, n_eval_episodes
                    )
                    curve.append(CurvePoint(0, 0.0, r0, s0))
                    elapsed = 0.0
                    done_steps = 0
                    for _ in range(n_chunks):
                        t0 = time.perf_counter()
                        ppo.learn(
                            total_timesteps=chunk_steps, reset_num_timesteps=False
                        )
                        elapsed += time.perf_counter() - t0
                        done_steps += chunk_steps
                        rm, sm = _evaluate(
                            ppo,
                            env_id,
                            use_lh,
                            kp,
                            damping_s,
                            hold_steps,
                            n_eval_episodes,
                        )
                        curve.append(CurvePoint(done_steps, elapsed, rm, sm))
                    seed_returns.append(curve[-1].mean_return)
                    seed_solved.append(curve[-1].solved_rate)
                    seed_times.append(elapsed)
                    all_curves.append(curve)
                finally:
                    vec.close()

            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            mean_curve = [
                CurvePoint(
                    env_steps=all_curves[0][k].env_steps,
                    wall_time_s=float(np.mean([c[k].wall_time_s for c in all_curves])),
                    mean_return=float(np.mean([c[k].mean_return for c in all_curves])),
                    solved_rate=float(np.mean([c[k].solved_rate for c in all_curves])),
                )
                for k in range(len(all_curves[0]))
            ]
            summaries.append(
                ArmSummary(
                    method=label,
                    env_id=env_id,
                    seeds=tuple(int(s) for s in seeds),
                    total_steps=total_steps,
                    final_return_mean=float(np.mean(seed_returns)),
                    final_return_std=float(np.std(seed_returns)),
                    final_solved_mean=float(np.mean(seed_solved)),
                    final_solved_std=float(np.std(seed_solved)),
                    wall_time_mean_s=float(np.mean(seed_times)),
                    wall_time_std_s=float(np.std(seed_times)),
                    peak_tracemalloc_mb=round(peak_bytes / (1024 * 1024), 2),
                    curves=mean_curve,
                )
            )
    return BenchmarkSummary(hardware=hw, arms=summaries)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the PPO vs. LambdaHold+PPO benchmark."""
    parser = argparse.ArgumentParser(prog="lambda-hold-bench-sb3")
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["myoElbowPose1D6MRandom-v0", "myoHandPoseRandom-v0"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=16_384)
    parser.add_argument("--eval-interval", type=int, default=4_096)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args(argv)

    summary = run_ppo_comparison(
        env_ids=args.envs,
        seeds=args.seeds,
        total_steps=args.steps,
        eval_interval=args.eval_interval,
    )
    payload = asdict(summary)
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out_json is not None:
        args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
