# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible package facade for :mod:`myosuite.envs.backends.mjlab.tasks`."""

from __future__ import annotations

import importlib as _importlib
from typing import Any

_importlib.import_module("myosuite.envs.backends.mjlab.bootstrap")
_target = _importlib.import_module("myosuite.envs.backends.mjlab.tasks")

_CHILD_SUBMODULES = frozenset(
    {
        "cpu_reference",
        "distribution",
        "leg",
        "mdp",
        "mimic_tracking",
        "pose",
        "reach",
        "registration",
        "rl",
        "waypoint",
    }
)


def __getattr__(name: str) -> Any:
    if name in _CHILD_SUBMODULES:
        mod = _importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    return getattr(_target, name)
