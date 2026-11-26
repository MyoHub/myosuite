# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible rollout and reference-motion logging facade."""

from __future__ import annotations

import importlib
from typing import Any

_COMPAT_SUBMODULES = frozenset(
    {
        "examine_logs",
        "examine_reference",
        "grouped_datasets",
        "reference_motion",
        "reference_motion_jax",
        "roboset_logger",
    }
)


def __getattr__(name: str) -> Any:
    if name in _COMPAT_SUBMODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
