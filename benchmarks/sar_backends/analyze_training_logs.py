#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Utility script to summarise SAR backend training logs.

Given a root output directory produced by ``run_benchmark.py``, this script
reads the per-backend ``training_log.csv`` files from Phase B and prints a
concise table comparing:

  - final episode reward
  - mean episode reward over the last N episodes
  - total wall time at the end of training

Example:

    python benchmarks/sar_backends/analyze_training_logs.py \\
        --root sar_benchmark_results/phase_b_training \\
        --backends cpu mjx_xla mjx_warp mjlab
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Sequence

import numpy as np


@dataclass
class BackendSummary:
    """Summary statistics for one backend."""

    backend: str
    final_timestep: float
    final_ep_reward: float
    mean_last_n: float
    wall_time_s: float


def _load_training_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _summarise_backend(
    backend: str,
    root: Path,
    last_n: int,
) -> BackendSummary | None:
    csv_path = root / backend / "training_log.csv"
    rows = _load_training_rows(csv_path)
    if not rows:
        return None

    # Convert to numpy for convenience.
    rewards = np.array([float(r.get("ep_reward", "nan")) for r in rows])
    timesteps = np.array([float(r.get("timestep", "nan")) for r in rows])
    wall_times = np.array([float(r.get("wall_time_s", "nan")) for r in rows])

    if np.all(~np.isfinite(rewards)):
        return None

    final_ep_reward = float(rewards[-1])
    final_timestep = (
        float(timesteps[-1]) if np.isfinite(timesteps[-1]) else float("nan")
    )
    wall_time_s = float(wall_times[-1]) if np.isfinite(wall_times[-1]) else float("nan")

    if last_n > 0:
        tail = rewards[-last_n:]
        tail = tail[np.isfinite(tail)]
        mean_last_n = float(np.mean(tail)) if tail.size else float("nan")
    else:
        mean_last_n = float("nan")

    return BackendSummary(
        backend=backend,
        final_timestep=final_timestep,
        final_ep_reward=final_ep_reward,
        mean_last_n=mean_last_n,
        wall_time_s=wall_time_s,
    )


def _format_row(fields: Sequence[str]) -> str:
    return " | ".join(fields)


def _print_table(summaries: Iterable[BackendSummary]) -> None:
    summaries = list(summaries)
    if not summaries:
        print("No training_log.csv files found for the requested backends.")
        return

    header = [
        "backend",
        "final_step",
        "final_ep_reward",
        "mean_lastN_ep_reward",
        "wall_time_s",
    ]
    print(_format_row(header))
    print("-" * len(_format_row(header)))

    for s in summaries:
        fields = [
            s.backend,
            f"{s.final_timestep:.0f}"
            if s.final_timestep == s.final_timestep
            else "nan",
            f"{s.final_ep_reward:.3f}"
            if s.final_ep_reward == s.final_ep_reward
            else "nan",
            f"{s.mean_last_n:.3f}" if s.mean_last_n == s.mean_last_n else "nan",
            f"{s.wall_time_s:.1f}" if s.wall_time_s == s.wall_time_s else "nan",
        ]
        print(_format_row(fields))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise SAR backend training logs produced by run_benchmark.py"
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing per-backend subdirectories from Phase B (e.g. sar_benchmark_results/phase_b_training).",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["cpu", "mjx_xla", "mjx_warp", "mjlab"],
        help="Backends to include in the summary.",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=20,
        help="Number of final episodes to average for the mean_lastN_ep_reward column.",
    )
    args = parser.parse_args()

    summaries: list[BackendSummary] = []
    for backend in args.backends:
        s = _summarise_backend(backend, args.root, args.last_n)
        if s is not None:
            summaries.append(s)

    _print_table(summaries)


if __name__ == "__main__":
    main()
