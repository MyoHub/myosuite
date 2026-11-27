# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Task configuration dataclasses for the mjlab/MuJoCo Warp backend.

Each config fully specifies a task: model path, physics timesteps, reward
weights, and task-specific parameters.  Configs are pure Python dataclasses
with no mjlab dependency so they can be imported and validated without a
MuJoCo Warp installation.
"""

from myosuite.envs.tasks.hand.baoding.mjlab_cfg import BaodingCfg
from myosuite.envs.backends.mjlab.configs.elbow_pose_cfg import ElbowPoseCfg
from myosuite.envs.backends.mjlab.configs.hand_reach_cfg import HandReachCfg
from myosuite.envs.tasks.fullbody.bimanual.mjlab_cfg import (
    MuscleMimicBimanualCfg,
)
from myosuite.envs.tasks.mimic.fullbody_mjlab_cfg import (
    MuscleMimicFullbodyCfg,
)
from myosuite.envs.tasks.games.tabletennis.mjlab_cfg import TableTennisCfg
from myosuite.envs.backends.mjlab.configs.walk_cfg import WalkCfg

__all__ = [
    "ElbowPoseCfg",
    "HandReachCfg",
    "WalkCfg",
    "BaodingCfg",
    "TableTennisCfg",
    "MuscleMimicBimanualCfg",
    "MuscleMimicFullbodyCfg",
]

from myosuite.envs.backends.mjlab.bootstrap import (  # noqa: E402
    finish_deferred_mjlab_bootstrap as _finish_deferred_mjlab_bootstrap,
)

_finish_deferred_mjlab_bootstrap(__name__)
del _finish_deferred_mjlab_bootstrap
