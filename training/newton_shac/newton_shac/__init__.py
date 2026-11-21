# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Differentiable Short-Horizon Actor-Critic (SHAC) for MyoSuite musculoskeletal tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from newton_shac.diff_env import DifferentiablePoseEnv
from newton_shac.networks import ActorCritic, RunningMeanStd
from newton_shac.ppo_baseline import PpoConfig, train_ppo
from newton_shac.shac import ShacConfig, train_shac

if TYPE_CHECKING:
    from newton_shac.benchmark import BenchmarkResult, run_benchmark

__all__ = [
    "ActorCritic",
    "BenchmarkResult",
    "DifferentiablePoseEnv",
    "PpoConfig",
    "RunningMeanStd",
    "ShacConfig",
    "run_benchmark",
    "train_ppo",
    "train_shac",
]


def __getattr__(name: str) -> Any:
    if name in ("BenchmarkResult", "run_benchmark"):
        from newton_shac import benchmark  # noqa: PLC0415

        return getattr(benchmark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
