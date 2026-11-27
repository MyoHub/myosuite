# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Standalone PPO trainers, callbacks, and presets for MyoSuite."""

from __future__ import annotations

import importlib as _importlib


def _verify_myosuite_runtime_api() -> None:
    try:
        _importlib.import_module("myosuite.policies.export_onnx")
        _importlib.import_module("myosuite.envs.feature_cli")
        _bench = _importlib.import_module("myosuite.benchmarks")
    except ModuleNotFoundError as exc:
        if exc.name == "myosuite" or (exc.name and exc.name.startswith("myosuite.")):
            raise ImportError(
                "myosuite_ppo requires a MyoSuite revision shipping `myosuite.policies`, "
                "`myosuite.benchmarks`, and `myosuite.envs.feature_cli` (`FOUR_BENCHMARK_TASKS`, "
                "`solved_from_info`); install `myosuite` from the matching Git revision."
            ) from exc
        raise
    if not (
        hasattr(_bench, "FOUR_BENCHMARK_TASKS") and hasattr(_bench, "solved_from_info")
    ):
        raise ImportError(
            "myosuite_ppo requires `myosuite.benchmarks.FOUR_BENCHMARK_TASKS` and "
            "`myosuite.benchmarks.solved_from_info`; upgrade `myosuite` to the matching Git revision."
        )


_verify_myosuite_runtime_api()
del _verify_myosuite_runtime_api

from myosuite_ppo.presets import (  # noqa: E402
    FOUR_BENCHMARK_TASKS,
    WinnerPpoConfig,
    budget_for,
)

__all__ = [
    "FOUR_BENCHMARK_TASKS",
    "WinnerPpoConfig",
    "budget_for",
]
