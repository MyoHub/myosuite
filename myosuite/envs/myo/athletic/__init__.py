# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Modular athletic task×model×tendon×backend factory.

``make_athletic("sprint")`` is myofullbody with rigid tendons.
"""

from __future__ import annotations

from myosuite.envs.myo.athletic.catalog import (
    PLANTS,
    TASKS,
    athletic_bundle,
    make_athletic,
    register_athletic_matrix,
)
from myosuite.envs.myo.athletic.specs import (
    DEFAULT_MODEL,
    DEFAULT_TENDONS,
    AthleticBundle,
    PlantSpec,
    TaskSpec,
)

__all__ = [
    "AthleticBundle",
    "DEFAULT_MODEL",
    "DEFAULT_TENDONS",
    "PlantSpec",
    "TaskSpec",
    "PLANTS",
    "TASKS",
    "athletic_bundle",
    "make_athletic",
    "register_athletic_matrix",
]
