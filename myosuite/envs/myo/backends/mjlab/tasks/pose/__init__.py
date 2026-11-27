# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible package facade for :mod:`myosuite.envs.backends.mjlab.tasks.pose`."""

from __future__ import annotations

import importlib as _importlib
from typing import Any

_target = _importlib.import_module("myosuite.envs.backends.mjlab.tasks.pose")

_CHILD_SUBMODULES = frozenset({"config", "mdp", "pose_env_cfg"})


def __getattr__(name: str) -> Any:
    if name in _CHILD_SUBMODULES:
        mod = _importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    return getattr(_target, name)
