# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Standardized evaluation, perturbation, and performance-card suites."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

FOUR_BENCHMARK_TASKS: tuple[str, ...] = (
    "myoElbowPose1D6MRandom-v0",
    "myoChallengeRelocateP1-v0",
    "myoFullBodyDirectional-v0",
    "myoMimic-v0",
)


def solved_from_info(info: dict[str, Any]) -> bool:
    """Extract a boolean solved flag from a Gymnasium ``info`` dict.

    Args:
        info: Step ``info`` dictionary returned by a MyoSuite environment.

    Returns:
        ``True`` if ``info["solved"]`` or ``info["rwd_dict"]["solved"]`` is truthy.
    """
    solved = info.get("solved")
    if solved is None:
        rwd = info.get("rwd_dict")
        if isinstance(rwd, dict):
            solved = rwd.get("solved")
    if solved is None:
        return False
    try:
        return bool(np.asarray(solved).astype(bool).any())
    except Exception:  # noqa: BLE001
        return bool(solved)


_SUBMODULES = frozenset(
    {
        "env_check",
        "examine_env",
        "examine_sim",
        "musclemimic_fullbody_eval_cli",
        "paired_replay",
        "performance_card",
        "push_metrics",
        "push_protocol",
        "push_recovery",
        "push_results",
        "push_scheduling",
        "term_catalog",
    }
)


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted({*_SUBMODULES, "FOUR_BENCHMARK_TASKS", "solved_from_info"})
