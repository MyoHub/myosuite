# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Modular athletic task×plant×backend factory (msk_envs athletic sweep)."""

from __future__ import annotations

from myosuite.envs.myo.athletic.catalog import (
    PLANTS,
    TASKS,
    make_athletic,
    register_athletic_matrix,
)
from myosuite.envs.myo.athletic.specs import AthleticBundle, PlantSpec, TaskSpec

__all__ = [
    "AthleticBundle",
    "PlantSpec",
    "TaskSpec",
    "PLANTS",
    "TASKS",
    "make_athletic",
    "register_athletic_matrix",
]
