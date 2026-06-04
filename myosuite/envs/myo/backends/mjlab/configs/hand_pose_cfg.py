# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Task config for the hand joint-angle posing task on the mjlab backend."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

_ASSETS = pathlib.Path(__file__).parents[3] / "assets"


@dataclass
class HandPoseCfg:
    """Configuration for the hand pose task on mjlab."""

    model_path: pathlib.Path = field(
        default_factory=lambda: _ASSETS / "hand" / "myohand_pose.xml"
    )
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02
    max_episode_steps: int = 100
    pose_thd: float = 0.7
    angle_reward_weight: float = 1.0
    bonus_weight: float = 4.0
    act_reg_weight: float = 0.01
    num_envs: int = 4096
