# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Multi-seed like-for-like benchmark comparing Newton-SHAC against PPO."""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import torch

from newton_shac.diff_env import DifferentiablePoseEnv
from newton_shac.ppo_baseline import PpoConfig, train_ppo
from newton_shac.shac import ShacConfig, TrainRunMetrics, train_shac


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregated multi-seed benchmark summary for one environment and method."""

    env_id: str
    method: str
    seeds: tuple[int, ...]
    eval_steps: tuple[int, ...]
    return_mean_curve: tuple[float, ...]
    return_std_curve: tuple[float, ...]
    solved_mean_curve: tuple[float, ...]
    solved_std_curve: tuple[float, ...]
    wall_time_mean_curve: tuple[float, ...]
    final_return_mean: float
    final_return_std: float
    final_solved_mean: float
    final_solved_std: float
    cpu_return_mean: float
    cpu_return_std: float
    cpu_solved_mean: float
    wall_time_mean_s: float
    wall_time_std_s: float
    peak_gpu_mem_mb: float


def _aggregate(runs: Sequence[TrainRunMetrics]) -> BenchmarkResult:
    ret_mat = np.asarray([r.eval_return_mean for r in runs], dtype=np.float64)
    sol_mat = np.asarray([r.eval_solved_rate for r in runs], dtype=np.float64)
    time_mat = np.asarray([r.eval_wall_time_s for r in runs], dtype=np.float64)
    cpu_ret = np.asarray([r.cpu_eval_return for r in runs], dtype=np.float64)
    cpu_sol = np.asarray([r.cpu_eval_solved_rate for r in runs], dtype=np.float64)
    tot_t = np.asarray([r.total_wall_time_s for r in runs], dtype=np.float64)
    mem = float(np.max([r.peak_gpu_mem_mb for r in runs]))

    return BenchmarkResult(
        env_id=runs[0].env_id,
        method=runs[0].method,
        seeds=tuple(r.seed for r in runs),
        eval_steps=tuple(int(s) for s in runs[0].eval_steps),
        return_mean_curve=tuple(float(v) for v in ret_mat.mean(axis=0)),
        return_std_curve=tuple(float(v) for v in ret_mat.std(axis=0)),
        solved_mean_curve=tuple(float(v) for v in sol_mat.mean(axis=0)),
        solved_std_curve=tuple(float(v) for v in sol_mat.std(axis=0)),
        wall_time_mean_curve=tuple(float(v) for v in time_mat.mean(axis=0)),
        final_return_mean=float(ret_mat[:, -1].mean()),
        final_return_std=float(ret_mat[:, -1].std()),
        final_solved_mean=float(sol_mat[:, -1].mean()),
        final_solved_std=float(sol_mat[:, -1].std()),
        cpu_return_mean=float(cpu_ret.mean()),
        cpu_return_std=float(cpu_ret.std()),
        cpu_solved_mean=float(cpu_sol.mean()),
        wall_time_mean_s=float(tot_t.mean()),
        wall_time_std_s=float(tot_t.std()),
        peak_gpu_mem_mb=mem,
    )


def run_benchmark(
    env_ids: Sequence[str] = ("myoElbowPose1D6MRandom-v0", "myoHandPoseRandom-v0"),
    seeds: Sequence[int] = (0, 1, 2),
    *,
    num_envs: int = 256,
    horizon: int = 16,
    total_env_steps: int = 102_400,
    eval_interval_steps: int = 25_600,
    device: str = "cuda:0",
) -> list[BenchmarkResult]:
    """Run matched multi-seed SHAC and PPO training across ``env_ids``."""
    results: list[BenchmarkResult] = []
    for env_id in env_ids:
        eval_env = DifferentiablePoseEnv(env_id, num_envs=64, device=device, seed=9999)
        shac_cfg = ShacConfig(
            num_envs=num_envs,
            horizon=horizon,
            total_env_steps=total_env_steps,
            eval_interval_steps=eval_interval_steps,
        )
        ppo_cfg = PpoConfig(
            num_envs=num_envs,
            horizon=horizon,
            total_env_steps=total_env_steps,
            eval_interval_steps=eval_interval_steps,
        )

        # Warm up JIT kernels before timed runs so compile latency does not skew seed 0.
        warmup_shac = dataclasses.replace(
            shac_cfg,
            total_env_steps=num_envs * horizon,
            eval_interval_steps=num_envs * horizon,
        )
        _, _ = train_shac(
            env_id, warmup_shac, seed=999, device=device, eval_env=eval_env
        )

        shac_runs: list[TrainRunMetrics] = []
        ppo_runs: list[TrainRunMetrics] = []
        for s in seeds:
            _, m_shac = train_shac(
                env_id, shac_cfg, seed=int(s), device=device, eval_env=eval_env
            )
            print(
                f"[SHAC] {env_id} seed={s} return={m_shac.eval_return_mean[-1]:.2f} "
                f"solved={m_shac.eval_solved_rate[-1] * 100:.1f}% "
                f"cpu_ret={m_shac.cpu_eval_return:.2f} time={m_shac.total_wall_time_s:.2f}s"
            )
            shac_runs.append(m_shac)

            _, m_ppo = train_ppo(
                env_id, ppo_cfg, seed=int(s), device=device, eval_env=eval_env
            )
            print(
                f"[PPO ] {env_id} seed={s} return={m_ppo.eval_return_mean[-1]:.2f} "
                f"solved={m_ppo.eval_solved_rate[-1] * 100:.1f}% "
                f"cpu_ret={m_ppo.cpu_eval_return:.2f} time={m_ppo.total_wall_time_s:.2f}s"
            )
            ppo_runs.append(m_ppo)

        results.append(_aggregate(shac_runs))
        results.append(_aggregate(ppo_runs))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["myoElbowPose1D6MRandom-v0", "myoHandPoseRandom-v0"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--total-steps", type=int, default=102_400)
    parser.add_argument("--eval-interval", type=int, default=25_600)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    results = run_benchmark(
        env_ids=args.envs,
        seeds=args.seeds,
        num_envs=args.num_envs,
        horizon=args.horizon,
        total_env_steps=args.total_steps,
        eval_interval_steps=args.eval_interval,
        device=args.device,
    )
    payload = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "results": [dataclasses.asdict(r) for r in results],
    }
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
