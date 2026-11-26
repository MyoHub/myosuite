# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Full-body basic tasks and registration (``myoFullBodyWaypoint-v0``)."""

from __future__ import annotations

import importlib
import math
from typing import Any

import myosuite.core.registry as _registry
from myosuite.envs.myo.tasks.basic.full_body.waypoint import (
    WaypointEnv,
    WaypointTaskCfg,
)
from myosuite.terms.waypoint import WaypointRouteCfg

# MyoFullBody faces -y at yaw 0, so routes start along -y. 100 Hz control, as MuscleMimic.
FULLBODY_WAYPOINT_TASK = WaypointTaskCfg(
    site_name="pelvis_mimic",
    route=WaypointRouteCfg(
        num_waypoints=4,
        segment_length=(1.0, 2.0),
        turn_angle=(-0.8, 0.8),
        heading_offset=-math.pi / 2,
    ),
    arrival_radius=0.3,
    lookahead=2,
    min_site_height=0.6,
    reset_noise=0.01,
)

_registry.register_env(
    env_id="myoFullBodyWaypoint-v0",
    entry_point="myosuite.envs.myo.tasks.basic.full_body.waypoint:WaypointEnv",
    max_episode_steps=1500,
    mjlab_task_kwargs=True,
    kwargs={
        "task": FULLBODY_WAYPOINT_TASK,
        "frame_skip": 5,
    },
)

_LAZY_REFERENCE_EXPORTS = frozenset(
    {
        "GaitParameters",
        "compose_waypoint_reference",
        "gait_cycle",
        "plan_waypoint_reference",
    }
)


def __getattr__(name: str) -> Any:
    """Load waypoint reference helpers lazily so ``import myosuite`` skips ``scipy.spatial``."""
    if name in _LAZY_REFERENCE_EXPORTS:
        mod = importlib.import_module(f"{__name__}.waypoint_reference")
        value = getattr(mod, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FULLBODY_WAYPOINT_TASK",
    "GaitParameters",
    "WaypointEnv",
    "WaypointTaskCfg",
    "compose_waypoint_reference",
    "gait_cycle",
    "plan_waypoint_reference",
]
