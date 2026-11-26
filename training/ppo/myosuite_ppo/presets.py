# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reference PPO presets and canonical benchmark task list."""

from __future__ import annotations

from myosuite.benchmarks import FOUR_BENCHMARK_TASKS
from myosuite_ppo.sb3_sweep import WinnerPpoConfig, budget_for

__all__ = [
    "FOUR_BENCHMARK_TASKS",
    "WinnerPpoConfig",
    "budget_for",
]
