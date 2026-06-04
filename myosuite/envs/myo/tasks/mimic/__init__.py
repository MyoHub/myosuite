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
"""

from __future__ import annotations

import myosuite.core.registry as _registry

_BIMANUAL_ENTRY = "myosuite.envs.myo.tasks.mimic.cpu:MuscleMimicBimanualEnv"
_FULLBODY_ENTRY = "myosuite.envs.myo.tasks.mimic.cpu:MuscleMimicFullbodyEnv"

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
