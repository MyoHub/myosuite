# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible import shim; canonical home is :mod:`myosuite.envs.myo.tasks.basic.full_body.waypoint_reference`."""

from __future__ import annotations

from myosuite.envs.myo.tasks.basic.full_body.waypoint_reference import (
    GaitParameters,
    compose_waypoint_reference,
    gait_cycle,
    plan_waypoint_reference,
)

__all__ = [
    "GaitParameters",
    "compose_waypoint_reference",
    "gait_cycle",
    "plan_waypoint_reference",
]
