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

from myosuite.envs.myo.backends.mjlab.configs.baoding_cfg import BaodingCfg
from myosuite.envs.myo.backends.mjlab.configs.elbow_pose_cfg import ElbowPoseCfg
from myosuite.envs.myo.backends.mjlab.configs.hand_reach_cfg import HandReachCfg
from myosuite.envs.myo.backends.mjlab.configs.musclemimic_bimanual_cfg import (
    MuscleMimicBimanualCfg,
)
from myosuite.envs.myo.backends.mjlab.configs.musclemimic_fullbody_cfg import (
    MuscleMimicFullbodyCfg,
)
from myosuite.envs.myo.backends.mjlab.configs.table_tennis_cfg import TableTennisCfg
from myosuite.envs.myo.backends.mjlab.configs.walk_cfg import WalkCfg

__all__ = [
    "ElbowPoseCfg",
    "HandReachCfg",
    "WalkCfg",
    "BaodingCfg",
    "TableTennisCfg",
    "MuscleMimicBimanualCfg",
    "MuscleMimicFullbodyCfg",
]
