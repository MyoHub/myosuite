# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Side-by-side reproducible benchmark comparing PPO, BC, GAIL, and AMP on MyoSuite."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import tracemalloc
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from myosuite import make_env
from myosuite_amp.ppo_amp import (
    ActorCritic,
    AMPConfig,
    AMPTrainer,
    BCConfig,
    CpuVecEnv,
    ExpertDataset,
    collect_expert_dataset,
    evaluate_policy_cpu,
    train_bc,
)

BENCHMARK_ARMS: tuple[str, ...] = ("PPO", "BC", "GAIL", "AMP")
DEFAULT_ENV_IDS: tuple[str, ...] = ("myoElbowPose1D6MRandom-v0", "myoHandPoseRandom-v0")


def _hardware_info(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "device": device,
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info


def _run_online_arm(
    env_id: str,
    arm: str,
    seed: int,
    *,
    total_steps: int,
    eval_interval: int,
    n_envs: int,
    n_eval_episodes: int,
    eval_seed: int,
    proportion_env_reward: float,
    motion_dataset: ExpertDataset | None,
    device: str,
) -> dict[str, Any]:
    use_disc = arm in ("AMP", "GAIL")
    if use_disc:
        if motion_dataset is None:
            raise ValueError(f"{arm} requires state-only motion_dataset.")
        if motion_dataset.actions is not None:
            raise ValueError(
                f"{arm} must receive action-free transitions (actions=None)."
            )

    tracemalloc.reset_peak()
    t0 = time.perf_counter()
    checkpoints: list[dict[str, float | int]] = []

    vec_env = CpuVecEnv(env_id, num_envs=n_envs, device=device)
    try:
        rollout_steps = min(128, max(8, eval_interval // max(1, n_envs)))
        cfg = AMPConfig(
            mode="gail" if arm == "GAIL" else "amp",
            use_discriminator=use_disc,
            proportion_env_reward=proportion_env_reward if use_disc else 1.0,
            rollout_steps=rollout_steps,
            seed=seed,
            device=device,
        )
        trainer = AMPTrainer(
            vec_env,
            cfg,
            expert_transitions=motion_dataset if use_disc else None,
        )

        def _on_eval(actor: ActorCritic, steps: int) -> None:
            elapsed = time.perf_counter() - t0
            ev = evaluate_policy_cpu(
                actor, env_id, n_episodes=n_eval_episodes, seed=eval_seed
            )
            checkpoints.append(
                {
                    "env_steps": int(steps),
                    "wall_time_s": float(elapsed),
                    "mean_return": ev.mean_return,
                    "std_return": ev.std_return,
                    "solved_rate": ev.solved_rate,
                }
            )

        trainer.train(
            total_steps=total_steps,
            eval_interval=eval_interval,
            eval_callback=_on_eval,
        )
    finally:
        vec_env.close()

    wall_s = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    peak_mb = float(peak_bytes) / (1024.0 * 1024.0)
    final_ckpt = checkpoints[-1]
    return {
        "seed": int(seed),
        "env_steps": int(total_steps),
        "wall_time_s": float(wall_s),
        "peak_memory_mb": peak_mb,
        "final_mean_return": float(final_ckpt["mean_return"]),
        "final_solved_rate": float(final_ckpt["solved_rate"]),
        "checkpoints": checkpoints,
    }


def _run_bc_arm(
    env_id: str,
    seed: int,
    *,
    expert_dataset: ExpertDataset,
    bc_epochs: int,
    num_eval_points: int,
    n_eval_episodes: int,
    eval_seed: int,
    device: str,
) -> dict[str, Any]:
    if expert_dataset.actions is None:
        raise ValueError("BC arm requires expert actions in expert_dataset.")

    probe_env = make_env(env_id)
    act_low = np.asarray(probe_env.action_space.low, dtype=np.float32)
    act_high = np.asarray(probe_env.action_space.high, dtype=np.float32)
    probe_env.close()

    tracemalloc.reset_peak()
    t0 = time.perf_counter()
    checkpoints: list[dict[str, float | int]] = []
    epoch_stride = max(1, bc_epochs // max(1, num_eval_points - 1))

    def _on_epoch(actor: ActorCritic, epoch: int, mse_loss: float) -> None:
        if epoch != 0 and epoch != bc_epochs and (epoch % epoch_stride != 0):
            return
        elapsed = time.perf_counter() - t0
        ev = evaluate_policy_cpu(
            actor, env_id, n_episodes=n_eval_episodes, seed=eval_seed
        )
        checkpoints.append(
            {
                "env_steps": 0,
                "bc_epoch": int(epoch),
                "bc_mse": 0.0 if np.isnan(mse_loss) else float(mse_loss),
                "wall_time_s": float(elapsed),
                "mean_return": ev.mean_return,
                "std_return": ev.std_return,
                "solved_rate": ev.solved_rate,
            }
        )

    bc_cfg = BCConfig(epochs=bc_epochs, seed=seed, device=device)
    train_bc(
        expert_dataset.obs,
        expert_dataset.actions,
        cfg=bc_cfg,
        action_low=act_low,
        action_high=act_high,
        epoch_callback=_on_epoch,
    )
    wall_s = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    peak_mb = float(peak_bytes) / (1024.0 * 1024.0)
    final_ckpt = checkpoints[-1]
    return {
        "seed": int(seed),
        "env_steps": 0,
        "wall_time_s": float(wall_s),
        "peak_memory_mb": peak_mb,
        "final_mean_return": float(final_ckpt["mean_return"]),
        "final_solved_rate": float(final_ckpt["solved_rate"]),
        "checkpoints": checkpoints,
    }


def _aggregate_arm(
    arm: str,
    seed_runs: list[dict[str, Any]],
    *,
    total_steps: int,
    proportion_env_reward: float,
) -> dict[str, Any]:
    final_returns = np.asarray(
        [r["final_mean_return"] for r in seed_runs], dtype=np.float64
    )
    final_solved = np.asarray(
        [r["final_solved_rate"] for r in seed_runs], dtype=np.float64
    )
    wall_times = np.asarray([r["wall_time_s"] for r in seed_runs], dtype=np.float64)
    peak_mems = np.asarray([r["peak_memory_mb"] for r in seed_runs], dtype=np.float64)

    n_ckpts = min(len(r["checkpoints"]) for r in seed_runs)
    curve: list[dict[str, float | int]] = []
    for idx in range(n_ckpts):
        rets = np.asarray(
            [r["checkpoints"][idx]["mean_return"] for r in seed_runs], dtype=np.float64
        )
        sols = np.asarray(
            [r["checkpoints"][idx]["solved_rate"] for r in seed_runs], dtype=np.float64
        )
        times = np.asarray(
            [r["checkpoints"][idx]["wall_time_s"] for r in seed_runs], dtype=np.float64
        )
        entry: dict[str, float | int] = {
            "env_steps": int(seed_runs[0]["checkpoints"][idx]["env_steps"]),
            "wall_time_mean_s": float(np.mean(times)),
            "return_mean": float(np.mean(rets)),
            "return_std": float(np.std(rets)),
            "solved_mean": float(np.mean(sols)),
            "solved_std": float(np.std(sols)),
        }
        if "bc_epoch" in seed_runs[0]["checkpoints"][idx]:
            entry["bc_epoch"] = int(seed_runs[0]["checkpoints"][idx]["bc_epoch"])
        curve.append(entry)

    return {
        "arm": arm,
        "env_step_budget": 0 if arm == "BC" else int(total_steps),
        "uses_expert_actions": arm == "BC",
        "uses_motion_transitions": arm in ("GAIL", "AMP"),
        "proportion_env_reward": (
            1.0
            if arm == "PPO"
            else (None if arm == "BC" else float(proportion_env_reward))
        ),
        "final_return_mean": float(np.mean(final_returns)),
        "final_return_std": float(np.std(final_returns)),
        "final_solved_mean": float(np.mean(final_solved)),
        "final_solved_std": float(np.std(final_solved)),
        "wall_time_mean_s": float(np.mean(wall_times)),
        "wall_time_std_s": float(np.std(wall_times)),
        "peak_memory_mean_mb": float(np.mean(peak_mems)),
        "peak_memory_std_mb": float(np.std(peak_mems)),
        "curve": curve,
        "seed_runs": seed_runs,
    }


def format_benchmark_markdown(results: dict[str, Any]) -> str:
    """Format aggregated benchmark results as GitHub-Flavored Markdown tables."""
    lines: list[str] = []
    for env_id, env_data in results["environments"].items():
        exp = env_data["expert_setup"]
        lines.append(f"### `{env_id}`")
        lines.append("")
        lines.append(
            f"- **Expert dataset setup (untimed pre-step):** PPO teacher trained for "
            f"`{exp['teacher_train_steps']}` env steps ({exp['generation_wall_time_s']:.2f} s), "
            f"teacher return `{exp['teacher_mean_return']:.2f}`, teacher solved "
            f"`{100.0 * exp['teacher_solved_rate']:.1f}%`, `{exp['num_transitions']}` transitions "
            f"(`actions` stripped for `GAIL`/`AMP`)."
        )
        lines.append("")
        lines.append(
            "| Method | Env Steps | Expert Inputs | Final Return (mean ± std) | "
            "Final Solved % (mean ± std) | Wall Time s (mean ± std) | Peak Mem MB |"
        )
        lines.append("|---|---:|---|---:|---:|---:|---:|")
        for arm in BENCHMARK_ARMS:
            row = env_data["arms"][arm]
            inputs = (
                "none (task rwd)"
                if arm == "PPO"
                else (
                    "`(s_t, a_t)`" if arm == "BC" else "`(s_t, s_{t+1})` + 0.5 task rwd"
                )
            )
            lines.append(
                f"| `{arm}` | {row['env_step_budget']:,} | {inputs} | "
                f"{row['final_return_mean']:.2f} ± {row['final_return_std']:.2f} | "
                f"{100.0 * row['final_solved_mean']:.1f}% ± "
                f"{100.0 * row['final_solved_std']:.1f}% | "
                f"{row['wall_time_mean_s']:.2f} ± {row['wall_time_std_s']:.2f} | "
                f"{row['peak_memory_mean_mb']:.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def run_amp_benchmark(
    env_ids: Sequence[str] = DEFAULT_ENV_IDS,
    total_steps: int = 49_152,
    eval_interval: int = 16_384,
    seeds: Sequence[int] = (0, 1, 2),
    n_envs: int = 8,
    n_eval_episodes: int = 20,
    expert_steps: int = 32_768,
    expert_transitions: int | None = None,
    proportion_env_reward: float = 0.5,
    bc_epochs: int = 50,
    eval_seed: int = 10_000,
    expert_seed: int = 777,
    device: str = "cpu",
    out_json: str | Path | None = None,
) -> dict[str, Any]:
    """Run the 4-arm benchmark (``PPO``, ``BC``, ``GAIL``, ``AMP``) across seeds and tasks.

    Args:
        env_ids: Registered MyoSuite task IDs to benchmark.
        total_steps: Online RL environment-step budget per seed for PPO, GAIL, and AMP.
        eval_interval: Step interval between deterministic evaluations.
        seeds: Random seeds run identically across all 4 arms.
        n_envs: Parallel CPU environments per online training run.
        n_eval_episodes: Deterministic evaluation episodes per checkpoint.
        expert_steps: Step budget used once per task to train the demonstration teacher.
        expert_transitions: Number of expert transitions collected per task (defaults to
            ``2000`` when ``expert_steps >= 2000`` else ``max(200, expert_steps)``).
        proportion_env_reward: Task-reward weight for AMP and GAIL (``0.5`` default).
        bc_epochs: Supervised training epochs for the BC baseline.
        eval_seed: Base deterministic evaluation seed shared across all arms.
        expert_seed: Seed used for demonstration generation.
        device: Torch device string.
        out_json: Optional output JSON file path.

    Returns:
        Dictionary containing metadata, per-env expert setup, arm summaries, and markdown.
    """
    n_demo = (
        int(expert_transitions)
        if expert_transitions is not None
        else (2000 if expert_steps >= 2000 else max(200, int(expert_steps)))
    )
    started_trace = not tracemalloc.is_tracing()
    if started_trace:
        tracemalloc.start()

    num_eval_pts = max(2, int(total_steps // max(1, eval_interval)) + 1)
    env_results: dict[str, Any] = {}

    try:
        for env_idx, env_id in enumerate(env_ids):
            expert_ds = collect_expert_dataset(
                env_id=env_id,
                n_transitions=n_demo,
                seed=expert_seed + env_idx * 1000,
                expert_steps=expert_steps,
                n_envs=n_envs,
            )
            motion_only_ds = expert_ds.without_actions()
            assert motion_only_ds.actions is None

            expert_report = {
                "teacher_train_steps": expert_ds.teacher_train_steps,
                "num_transitions": expert_ds.num_transitions,
                "teacher_mean_return": expert_ds.teacher_mean_return,
                "teacher_solved_rate": expert_ds.teacher_solved_rate,
                "generation_wall_time_s": expert_ds.generation_wall_time_s,
                "amp_gail_actions_stripped": True,
            }

            arms_summary: dict[str, Any] = {}
            for arm in BENCHMARK_ARMS:
                seed_runs: list[dict[str, Any]] = []
                for seed in seeds:
                    if arm == "BC":
                        run_rec = _run_bc_arm(
                            env_id,
                            int(seed),
                            expert_dataset=expert_ds,
                            bc_epochs=bc_epochs,
                            num_eval_points=num_eval_pts,
                            n_eval_episodes=n_eval_episodes,
                            eval_seed=eval_seed,
                            device=device,
                        )
                    else:
                        run_rec = _run_online_arm(
                            env_id,
                            arm,
                            int(seed),
                            total_steps=total_steps,
                            eval_interval=eval_interval,
                            n_envs=n_envs,
                            n_eval_episodes=n_eval_episodes,
                            eval_seed=eval_seed,
                            proportion_env_reward=proportion_env_reward,
                            motion_dataset=motion_only_ds
                            if arm in ("GAIL", "AMP")
                            else None,
                            device=device,
                        )
                    seed_runs.append(run_rec)

                arms_summary[arm] = _aggregate_arm(
                    arm,
                    seed_runs,
                    total_steps=total_steps,
                    proportion_env_reward=proportion_env_reward,
                )

            env_results[env_id] = {
                "expert_setup": expert_report,
                "arms": arms_summary,
            }
    finally:
        if started_trace:
            tracemalloc.stop()

    results: dict[str, Any] = {
        "metadata": {
            "hardware": _hardware_info(device),
            "config": {
                "env_ids": list(env_ids),
                "total_steps": int(total_steps),
                "eval_interval": int(eval_interval),
                "seeds": [int(s) for s in seeds],
                "n_envs": int(n_envs),
                "n_eval_episodes": int(n_eval_episodes),
                "expert_steps": int(expert_steps),
                "expert_transitions": n_demo,
                "proportion_env_reward": float(proportion_env_reward),
                "bc_epochs": int(bc_epochs),
                "eval_seed": int(eval_seed),
            },
        },
        "environments": env_results,
    }
    results["markdown_table"] = format_benchmark_markdown(results)

    if out_json is not None:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible 4-arm benchmark (PPO, BC, GAIL, AMP) on MyoSuite tasks."
    )
    parser.add_argument(
        "--env-ids",
        nargs="+",
        default=list(DEFAULT_ENV_IDS),
        help="Registered MyoSuite environment IDs",
    )
    parser.add_argument("--total-steps", type=int, default=49_152)
    parser.add_argument("--eval-interval", type=int, default=16_384)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--expert-steps", type=int, default=32_768)
    parser.add_argument("--expert-transitions", type=int, default=None)
    parser.add_argument("--proportion-env-reward", type=float, default=0.5)
    parser.add_argument("--bc-epochs", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for ``amp-bench``."""
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    results = run_amp_benchmark(
        env_ids=args.env_ids,
        total_steps=args.total_steps,
        eval_interval=args.eval_interval,
        seeds=args.seeds,
        n_envs=args.n_envs,
        n_eval_episodes=args.n_eval_episodes,
        expert_steps=args.expert_steps,
        expert_transitions=args.expert_transitions,
        proportion_env_reward=args.proportion_env_reward,
        bc_epochs=args.bc_epochs,
        device=args.device,
        out_json=args.out_json,
    )
    print(results["markdown_table"])


if __name__ == "__main__":
    main()
