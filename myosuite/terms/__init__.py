# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible facade for :mod:`myosuite.core.terms`."""

from __future__ import annotations

import importlib
from typing import Any

from myosuite.core.terms import (
    base_action,
    base_event,
    base_obs,
    base_reward,
    base_termination,
)

__all__ = [
    "base_action",
    "base_event",
    "base_obs",
    "base_reward",
    "base_termination",
]

_COMPAT_SUBMODULES = frozenset(
    {
        "_array",
        "catalog",
        "mimic_obs",
        "mimic_reward",
        "mimic_track",
        "multiplayer",
        "opponent_relative_obs",
        "waypoint",
    }
)


def __getattr__(name: str) -> Any:
    if name in _COMPAT_SUBMODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
