# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Full-body basic tasks and registration (``myoFullBodyWaypoint-v0``)."""

from __future__ import annotations

import math

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

__all__ = [
    "FULLBODY_WAYPOINT_TASK",
    "WaypointEnv",
    "WaypointTaskCfg",
]
