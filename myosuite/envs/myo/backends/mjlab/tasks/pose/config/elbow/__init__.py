# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible package facade for :mod:`myosuite.envs.tasks.arm.pose.mjlab_register`."""

from __future__ import annotations

import importlib as _importlib
from typing import Any

_target = _importlib.import_module("myosuite.envs.tasks.arm.pose.mjlab_register")
__all__ = list(
    getattr(_target, "__all__", [k for k in _target.__dict__ if not k.startswith("_")])
)

_CHILD_SUBMODULES = frozenset(
    {
        "env_cfgs",
        "rl_cfg",
    }
)


def __getattr__(name: str) -> Any:
    if name in _CHILD_SUBMODULES:
        mod = _importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    return getattr(_target, name)
