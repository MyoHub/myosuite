# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Registration module for MyoMimic CPU Gymnasium environments.

This suite owns canonical Mimic IDs and their legacy aliases:

- ``myoMimicBimanual-v0``
- ``myoMimicFullbody-v0``
- ``myoMuscleMimicBimanual-v0`` (legacy alias)
- ``myoMuscleMimicFullbody-v0`` (legacy alias)
- ``myoFullBodyWaypoint-v0`` (random ordered waypoints on flat ground)
"""

from __future__ import annotations

import math

import myosuite.core.registry as _registry
from myosuite.envs.waypoint import WaypointTaskCfg
from myosuite.terms.waypoint import WaypointRouteCfg

_BIMANUAL_ENTRY = "myosuite.envs.myo.tasks.mimic.cpu:MuscleMimicBimanualEnv"
_FULLBODY_ENTRY = "myosuite.envs.myo.tasks.mimic.cpu:MuscleMimicFullbodyEnv"
_FULLBODY_DIRECTIONAL_ENTRY = (
    "myosuite.envs.myo.tasks.mimic.cpu:MuscleMimicFullbodyDirectionalEnv"
)

_registry.register_env(
    env_id="myoMimicBimanual-v0",
    entry_point=_BIMANUAL_ENTRY,
    max_episode_steps=1000,
    kwargs={"frame_skip": 5},
)

_registry.register_env(
    env_id="myoMimicFullbody-v0",
    entry_point=_FULLBODY_ENTRY,
    max_episode_steps=1000,
    kwargs={"frame_skip": 5},
)

_registry.register_env(
    env_id="myoMuscleMimicBimanual-v0",
    entry_point=_BIMANUAL_ENTRY,
    max_episode_steps=1000,
    kwargs={"frame_skip": 5},
)

_registry.register_env(
    env_id="myoMuscleMimicFullbody-v0",
    entry_point=_FULLBODY_ENTRY,
    max_episode_steps=1000,
    kwargs={"frame_skip": 5},
)

_registry.register_env(
    env_id="myoFullBodyDirectional-v0",
    entry_point=_FULLBODY_DIRECTIONAL_ENTRY,
    max_episode_steps=500,
    kwargs={"frame_skip": 5},
)

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
    entry_point="myosuite.envs.waypoint:WaypointEnv",
    max_episode_steps=1500,
    kwargs={
        "model_recipe": "musclemimic_fullbody",
        "task": FULLBODY_WAYPOINT_TASK,
        "frame_skip": 5,
    },
)
