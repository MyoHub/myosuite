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
- ``myoFullBodyDirectional-v0``
- ``myoMimic-v0`` (multi-clip motion tracking, :mod:`.tracking`)
"""

from __future__ import annotations

import myosuite.core.registry as _registry
from myosuite.envs.myo.tasks.basic.full_body import (  # noqa: F401 (backward-compat re-export)
    FULLBODY_WAYPOINT_TASK,
)
from myosuite.envs.myo.tasks.mimic.tracking import MIMIC_ENV_ID

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

# The env truncates itself at the task's time limit; TimeLimit only backs it up (300 s
# at the shortest control step, one 2 ms physics step).
_registry.register_env(
    env_id=MIMIC_ENV_ID,
    entry_point="myosuite.envs.myo.tasks.mimic.tracking.env:MimicEnv",
    max_episode_steps=150_000,
    mjlab_task_kwargs=True,
    models=("myofullbody",),
    kwargs={"frame_skip": 10, "preset": "supermimic", "model": "myofullbody"},
)
