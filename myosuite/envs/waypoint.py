# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible import shim; canonical home is :mod:`myosuite.envs.myo.tasks.basic.full_body.waypoint`."""

from __future__ import annotations

from myosuite.core.model_builder import build_from_recipe
from myosuite.envs.myo.tasks.basic.full_body.waypoint import (
    DEFAULT_MODEL_RECIPE,
    OBS_KEYS,
    RootLayout,
    WaypointEnv,
    WaypointTaskCfg,
    build_model,
    initial_state,
)

__all__ = [
    "DEFAULT_MODEL_RECIPE",
    "OBS_KEYS",
    "RootLayout",
    "WaypointEnv",
    "WaypointTaskCfg",
    "build_from_recipe",
    "build_model",
    "initial_state",
]
