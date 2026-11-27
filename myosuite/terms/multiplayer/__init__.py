# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible facade for :mod:`myosuite.core.terms.multiplayer` and multiplayer task terms."""

from __future__ import annotations

import importlib
from typing import Any

_COMPAT_SUBMODULES = frozenset(
    {
        "chase_tag_vs_obs",
        "chase_tag_vs_reward",
        "common",
        "heading",
    }
)


def __getattr__(name: str) -> Any:
    if name in _COMPAT_SUBMODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
