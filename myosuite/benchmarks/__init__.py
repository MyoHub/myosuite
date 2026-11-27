# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Standardized evaluation, perturbation, and performance-card suites."""

from __future__ import annotations

import importlib
from typing import Any

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


__all__ = sorted(_SUBMODULES)
