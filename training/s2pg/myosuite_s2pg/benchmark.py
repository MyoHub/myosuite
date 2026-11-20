# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Side-by-side reproducible benchmark comparing stateless PPO vs. S2PG-PPO on MyoSuite."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import tracemalloc
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import myosuite
import numpy as np
import torch
from myosuite import make_env
from myosuite.core.config import EnvConfig

from myosuite_s2pg.networks import (
    ActorCriticFeedforward,
    ActorCriticS2PG,
    RNNType,
    VALID_RNN_TYPES,
)
from myosuite_s2pg.s2pg_ppo import (
    CpuVectorEnv,
    FeedforwardPPOTrainer,
    S2PGConfig,
    S2PGPPOTrainer,
)

DEFAULT_BENCHMARK_ENVS: tuple[str, ...] = (
    "myoElbowPose1D6MRandom-v0",
    "myoHandPoseRandom-v0",
)
EVAL_SEED_BASE = 10_000


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for side-by-side PPO vs. S2PG-PPO benchmarking."""

    env_ids: tuple[str, ...] = DEFAULT_BENCHMARK_ENVS
    total_steps: int = 102_400
    eval_interval: int = 20_480
    seeds: tuple[int, ...] = (0, 1, 2)
    n_envs: int = 8
    n_eval_episodes: int = 20
    rnn_type: RNNType = "gru"
    hidden_state_dim: int = 64
    ema_coef: float = 1.0
    num_steps_per_env: int = 128
    lr: float = 3e-4
    fatigue: bool = False
    device: str = "cpu"
    out_json: Path | None = None


@torch.no_grad()
def evaluate_s2pg_deterministic(
    policy: ActorCriticS2PG,
    eval_env: Any,
    n_episodes: int,
    clip_actions: float | None = 1.0,
    seed_base: int = EVAL_SEED_BASE,
) -> dict[str, float]:
    """Evaluate S2PG policy deterministically using ``(z_0 = 0, a_{-1} = 0)`` and ``(mean_a, mean_z)``."""
    policy.eval()
    dev = policy.action_mean_head.weight.device
    ep_returns: list[float] = []
    ep_successes: list[float] = []

    for ep_idx in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed_base + ep_idx)
        hidden, prev_act = policy.initial_state(1, device=dev)
        done = False
        ep_ret = 0.0
        solved = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
            mean_a, mean_z = policy.deterministic_act(obs_t, hidden, prev_act)
            env_act = (
                torch.clamp(mean_a, -clip_actions, clip_actions)
                if clip_actions is not None
                else mean_a
            )
            obs, rew, terminated, truncated, info = eval_env.step(
                env_act.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
            )
            ep_ret += float(rew)
            hidden = mean_z
            prev_act = mean_a
            done = bool(terminated or truncated)
            if done and "solved" in info:
                solved = float(bool(info["solved"]))
        ep_returns.append(ep_ret)
        ep_successes.append(solved)

    return {
        "mean_return": float(np.mean(ep_returns)),
        "std_return": float(np.std(ep_returns)),
        "success_rate": float(np.mean(ep_successes)),
    }


@torch.no_grad()
def evaluate_ppo_deterministic(
    policy: ActorCriticFeedforward,
    eval_env: Any,
    n_episodes: int,
    clip_actions: float | None = 1.0,
    seed_base: int = EVAL_SEED_BASE,
) -> dict[str, float]:
    """Evaluate stateless feedforward PPO policy deterministically using ``mean_a``."""
    policy.eval()
    dev = policy.action_mean_head.weight.device
    ep_returns: list[float] = []
    ep_successes: list[float] = []

    for ep_idx in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed_base + ep_idx)
        done = False
        ep_ret = 0.0
        solved = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
            mean_a = policy.deterministic_act(obs_t)
            env_act = (
                torch.clamp(mean_a, -clip_actions, clip_actions)
                if clip_actions is not None
                else mean_a
            )
            obs, rew, terminated, truncated, info = eval_env.step(
                env_act.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
            )
            ep_ret += float(rew)
            done = bool(terminated or truncated)
            if done and "solved" in info:
                solved = float(bool(info["solved"]))
        ep_returns.append(ep_ret)
        ep_successes.append(solved)

    return {
        "mean_return": float(np.mean(ep_returns)),
        "std_return": float(np.std(ep_returns)),
        "success_rate": float(np.mean(ep_successes)),
    }


def _run_single_arm(
    method: str,
    env_id: str,
    seed: int,
    total_steps: int,
    eval_interval: int,
    n_envs: int,
    n_eval_episodes: int,
    cfg: S2PGConfig,
    features: tuple[Any, ...],
    device: str,
    verbose: bool,
) -> dict[str, Any]:
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()

    vec_env = CpuVectorEnv(
        env_id=env_id,
        num_envs=n_envs,
        seed=seed,
        features=features,
        device=device,
    )
    eval_env = make_env(EnvConfig(env_id=env_id, features=features))

    try:
        trainer: S2PGPPOTrainer | FeedforwardPPOTrainer
        if method == "S2PG-PPO":
            trainer = S2PGPPOTrainer(vec_env=vec_env, cfg=cfg, device=device)
        elif method == "PPO":
            trainer = FeedforwardPPOTrainer(vec_env=vec_env, cfg=cfg, device=device)
        else:
            raise ValueError(f"Unknown benchmark method {method!r}.")

        trainer.reset_rollout_state(seed=seed)

        curve: list[dict[str, float]] = []
        total_t0 = time.perf_counter()
        train_elapsed_s = 0.0

        def record_eval() -> None:
            if isinstance(trainer, S2PGPPOTrainer):
                ev = evaluate_s2pg_deterministic(
                    trainer.policy,
                    eval_env,
                    n_episodes=n_eval_episodes,
                    clip_actions=cfg.clip_actions,
                )
            else:
                ev = evaluate_ppo_deterministic(
                    trainer.policy,
                    eval_env,
                    n_episodes=n_eval_episodes,
                    clip_actions=cfg.clip_actions,
                )
            point = {
                "env_steps": float(trainer.env_steps),
                "train_wall_clock_s": float(train_elapsed_s),
                "mean_return": ev["mean_return"],
                "std_return": ev["std_return"],
                "success_rate": ev["success_rate"],
            }
            curve.append(point)
            if verbose:
                print(
                    f"[{env_id} | {method:8s} | seed={seed}] "
                    f"steps={trainer.env_steps:7d} "
                    f"return={ev['mean_return']:8.2f} "
                    f"success={ev['success_rate']:6.1%} "
                    f"train_t={train_elapsed_s:6.2f}s"
                )

        record_eval()

        steps_per_iter = cfg.num_steps_per_env * n_envs
        total_iters = max(1, math.ceil(total_steps / steps_per_iter))
        next_eval_step = eval_interval

        for it in range(total_iters):
            progress = it / max(1, total_iters)
            step_t0 = time.perf_counter()
            trainer.train_step(progress=progress)
            train_elapsed_s += time.perf_counter() - step_t0

            is_last = it == total_iters - 1
            if trainer.env_steps >= next_eval_step or is_last:
                record_eval()
                while next_eval_step <= trainer.env_steps:
                    next_eval_step += eval_interval

        total_wall_s = time.perf_counter() - total_t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        peak_mb = float(peak_bytes) / (1024.0 * 1024.0)
    finally:
        vec_env.close()
        eval_env.close()
        if not was_tracing:
            tracemalloc.stop()

    final_pt = curve[-1]
    return {
        "seed": int(seed),
        "env_steps": int(final_pt["env_steps"]),
        "final_return": float(final_pt["mean_return"]),
        "final_success_rate": float(final_pt["success_rate"]),
        "train_wall_clock_s": float(train_elapsed_s),
        "total_wall_clock_s": float(total_wall_s),
        "peak_tracemalloc_mb": peak_mb,
        "curve": curve,
    }


def _aggregate_seed_runs(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    final_returns = [r["final_return"] for r in seed_runs]
    final_successes = [r["final_success_rate"] for r in seed_runs]
    train_times = [r["train_wall_clock_s"] for r in seed_runs]
    peaks = [r["peak_tracemalloc_mb"] for r in seed_runs]

    num_points = min(len(r["curve"]) for r in seed_runs)
    learning_curve: list[dict[str, float]] = []
    for idx in range(num_points):
        pts = [r["curve"][idx] for r in seed_runs]
        learning_curve.append(
            {
                "env_steps": float(pts[0]["env_steps"]),
                "train_wall_clock_s_mean": float(
                    np.mean([p["train_wall_clock_s"] for p in pts])
                ),
                "return_mean": float(np.mean([p["mean_return"] for p in pts])),
                "return_std": float(np.std([p["mean_return"] for p in pts])),
                "success_mean": float(np.mean([p["success_rate"] for p in pts])),
                "success_std": float(np.std([p["success_rate"] for p in pts])),
            }
        )

    return {
        "final_return_mean": float(np.mean(final_returns)),
        "final_return_std": float(np.std(final_returns)),
        "final_success_mean": float(np.mean(final_successes)),
        "final_success_std": float(np.std(final_successes)),
        "train_wall_clock_s_mean": float(np.mean(train_times)),
        "train_wall_clock_s_std": float(np.std(train_times)),
        "peak_tracemalloc_mb_mean": float(np.mean(peaks)),
        "peak_tracemalloc_mb_max": float(np.max(peaks)),
        "learning_curve": learning_curve,
    }


def format_markdown_summary(results: dict[str, Any]) -> str:
    """Format benchmark results into a Markdown table for README reporting."""
    meta = results["metadata"]
    regime = (
        "Modified task (`FatigueWrapper` enabled)"
        if meta["fatigue"]
        else "Unmodified registered tasks"
    )
    lines = [
        f"### Benchmark Summary ({regime})",
        "",
        "| Environment | Method | Final Return (mean ± std) | Final Success (mean ± std) | Train Time (s) | Peak Memory (MB) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for env_id, methods in results["environments"].items():
        for method_name in ("PPO", "S2PG-PPO"):
            agg = methods[method_name]["aggregate"]
            ret_str = f"{agg['final_return_mean']:.2f} ± {agg['final_return_std']:.2f}"
            succ_str = (
                f"{100.0 * agg['final_success_mean']:.1f}% ± "
                f"{100.0 * agg['final_success_std']:.1f}%"
            )
            time_str = (
                f"{agg['train_wall_clock_s_mean']:.2f} ± "
                f"{agg['train_wall_clock_s_std']:.2f}"
            )
            mem_str = f"{agg['peak_tracemalloc_mb_max']:.1f}"
            lines.append(
                f"| `{env_id}` | {method_name} | {ret_str} | {succ_str} | {time_str} | {mem_str} |"
            )
    return "\n".join(lines)


def run_s2pg_benchmark(
    env_ids: Sequence[str] = DEFAULT_BENCHMARK_ENVS,
    total_steps: int = 102_400,
    eval_interval: int = 20_480,
    seeds: Sequence[int] = (0, 1, 2),
    n_envs: int = 8,
    n_eval_episodes: int = 20,
    rnn_type: RNNType = "gru",
    hidden_state_dim: int = 64,
    ema_coef: float = 1.0,
    num_steps_per_env: int = 128,
    lr: float = 3e-4,
    fatigue: bool = False,
    device: str = "cpu",
    out_json: Path | str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run an apple-to-apple PPO vs. S2PG-PPO benchmark across seeds and environments.

    Args:
        env_ids: Registered MyoSuite ``env_id``s to evaluate.
        total_steps: Environment-step budget per seed and method.
        eval_interval: Environment steps between deterministic evaluations.
        seeds: Random seeds evaluated for both ``PPO`` and ``S2PG-PPO``.
        n_envs: Number of vectorized training environments.
        n_eval_episodes: Number of deterministic evaluation episodes per checkpoint.
        rnn_type: Recurrent transition cell for ``S2PG-PPO``.
        hidden_state_dim: Latent cell width for ``S2PG-PPO``.
        ema_coef: Exponential smoothing weight on ``next_z_mean`` for ``S2PG-PPO``.
        num_steps_per_env: Desired rollout horizon per env (clamped to fit ``eval_interval``).
        lr: Shared Adam learning rate.
        fatigue: If ``True``, wrap envs with MyoSuite's ``FatigueWrapper`` (reported separately).
        device: Torch execution device.
        out_json: Optional output file path for serialized JSON results.
        verbose: Whether to print progress and Markdown summary.

    Returns:
        Structured benchmark report dictionary.
    """
    if total_steps <= 0 or eval_interval <= 0 or n_envs <= 0 or n_eval_episodes <= 0:
        raise ValueError(
            "total_steps, eval_interval, n_envs, and n_eval_episodes must be positive."
        )
    if not env_ids:
        raise ValueError("env_ids must be non-empty.")
    if not seeds:
        raise ValueError("seeds must be non-empty.")

    features: tuple[Any, ...] = ()
    task_modification: str | None = None
    if fatigue:
        from myosuite.envs.wrappers import FatigueWrapper

        features = (FatigueWrapper,)
        task_modification = (
            "FatigueWrapper (3CC-r muscle fatigue; non-Markovian hidden muscle state)"
        )

    effective_steps_per_env = max(
        1,
        min(num_steps_per_env, eval_interval // n_envs, total_steps // n_envs),
    )

    env_results: dict[str, dict[str, Any]] = {}
    for env_id in env_ids:
        env_results[env_id] = {}
        for method in ("PPO", "S2PG-PPO"):
            seed_runs: list[dict[str, Any]] = []
            for seed in seeds:
                cfg = S2PGConfig(
                    hidden_state_dim=hidden_state_dim,
                    rnn_type=rnn_type,
                    ema_coef=ema_coef,
                    lr=lr,
                    num_steps_per_env=effective_steps_per_env,
                    seed=int(seed),
                )
                seed_runs.append(
                    _run_single_arm(
                        method=method,
                        env_id=env_id,
                        seed=int(seed),
                        total_steps=total_steps,
                        eval_interval=eval_interval,
                        n_envs=n_envs,
                        n_eval_episodes=n_eval_episodes,
                        cfg=cfg,
                        features=features,
                        device=device,
                        verbose=verbose,
                    )
                )
            env_results[env_id][method] = {
                "per_seed": seed_runs,
                "aggregate": _aggregate_seed_runs(seed_runs),
            }

    results: dict[str, Any] = {
        "metadata": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "mujoco_version": mujoco.__version__,
            "myosuite_version": myosuite.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "device": device,
            "env_ids": list(env_ids),
            "total_steps": int(total_steps),
            "eval_interval": int(eval_interval),
            "effective_steps_per_env": int(effective_steps_per_env),
            "seeds": [int(s) for s in seeds],
            "n_envs": int(n_envs),
            "n_eval_episodes": int(n_eval_episodes),
            "rnn_type": rnn_type,
            "hidden_state_dim": int(hidden_state_dim),
            "ema_coef": float(ema_coef),
            "lr": float(lr),
            "fatigue": bool(fatigue),
            "task_modification": task_modification,
        },
        "environments": env_results,
    }
    results["markdown_table"] = format_markdown_summary(results)

    if verbose:
        print("\n" + results["markdown_table"])

    if out_json is not None:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for ``s2pg-bench``."""
    parser = argparse.ArgumentParser(
        prog="s2pg-bench",
        description="Side-by-side PPO vs. S2PG-PPO benchmark on MyoSuite environments.",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=list(DEFAULT_BENCHMARK_ENVS),
        help="Registered MyoSuite env_ids to benchmark.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=102_400,
        help="Environment steps per seed and method.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=20_480,
        help="Environment steps between deterministic evaluations.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds run identically for PPO and S2PG-PPO.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of vectorized CPU environments.",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=20,
        help="Number of deterministic evaluation episodes per checkpoint.",
    )
    parser.add_argument(
        "--rnn-type",
        choices=list(VALID_RNN_TYPES),
        default="gru",
        help="Recurrent transition cell for S2PG-PPO.",
    )
    parser.add_argument(
        "--hidden-state-dim",
        type=int,
        default=64,
        help="Latent recurrent state width.",
    )
    parser.add_argument(
        "--ema-coef",
        type=float,
        default=1.0,
        help="Exponential smoothing coefficient on next_z_mean.",
    )
    parser.add_argument(
        "--steps-per-env",
        type=int,
        default=128,
        help="Rollout steps per environment per update.",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Shared learning rate.")
    parser.add_argument(
        "--fatigue",
        action="store_true",
        help="Wrap environments with FatigueWrapper to test non-Markovian muscle fatigue.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (default: cpu).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to write structured benchmark JSON results.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress step-by-step progress output.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for ``s2pg-bench``."""
    args = build_arg_parser().parse_args(sys.argv[1:] if argv is None else argv)
    run_s2pg_benchmark(
        env_ids=tuple(args.envs),
        total_steps=args.total_steps,
        eval_interval=args.eval_interval,
        seeds=tuple(args.seeds),
        n_envs=args.n_envs,
        n_eval_episodes=args.n_eval_episodes,
        rnn_type=args.rnn_type,
        hidden_state_dim=args.hidden_state_dim,
        ema_coef=args.ema_coef,
        num_steps_per_env=args.steps_per_env,
        lr=args.lr,
        fatigue=args.fatigue,
        device=args.device,
        out_json=args.out_json,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
