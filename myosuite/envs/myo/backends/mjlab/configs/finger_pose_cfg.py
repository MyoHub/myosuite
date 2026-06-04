# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Task config for the finger joint-angle posing task on the mjlab backend."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

_SIMHIVE = pathlib.Path(__file__).parents[5] / "simhive" / "myo_sim"


@dataclass
class FingerPoseCfg:
    """Configuration for the finger pose task on mjlab.

    The task targets the same four finger joints used by Gym and MJX finger
    pose variants.
    """

    model_path: pathlib.Path = field(
        default_factory=lambda: _SIMHIVE / "finger" / "myofinger_v0.xml"
    )
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02
    max_episode_steps: int = 100
    target_jnt_names: tuple[str, ...] = ("IFadb", "IFmcp", "IFpip", "IFdip")
    target_range_fixed: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.0, 0.0),
        (0.75, 0.75),
        (0.75, 0.75),
    )
    target_range_random: tuple[tuple[float, float], ...] = (
        (-0.2, 0.2),
        (-0.4, 1.0),
        (0.1, 1.0),
        (0.1, 1.0),
    )
    pose_thd: float = 0.35
    angle_reward_weight: float = 1.0
    bonus_weight: float = 4.0
    act_reg_weight: float = 0.01
    num_envs: int = 4096
