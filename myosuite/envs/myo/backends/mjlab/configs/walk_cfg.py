# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Task config for the bipedal walking task on the mjlab backend."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

_ASSETS = pathlib.Path(__file__).parents[3] / "assets"


@dataclass
class WalkCfg:
    """Configuration for the bipedal walking task on the mjlab/MuJoCo Warp backend.

    The agent controls the leg muscle actuators to achieve forward locomotion
    while maintaining balance.

    Args:
        model_path: Path to the MJCF model file.
        sim_dt: MuJoCo simulation timestep in seconds.
        ctrl_dt: Control timestep; must be a multiple of ``sim_dt``.
        max_episode_steps: Episode truncation length.
        target_vel: Target forward walking velocity in m/s.
        vel_reward_weight: Scaling weight for velocity-tracking reward.
        alive_bonus: Constant reward added each step the agent stays upright.
        fall_penalty: Penalty applied when the agent falls (torso height
            drops below ``fall_height_threshold``).
        fall_height_threshold: Minimum torso height (m) before fall penalty.
        act_reg_weight: Scaling weight for the activation L2 penalty.
        num_envs: Number of parallel environments (batch size on GPU).
    """

    model_path: pathlib.Path = field(
        default_factory=lambda: _ASSETS / "leg" / "myolegs_chasetag.xml"
    )
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02
    max_episode_steps: int = 1000
    target_vel: float = 1.2
    vel_reward_weight: float = 1.0
    alive_bonus: float = 0.2
    fall_penalty: float = -10.0
    fall_height_threshold: float = 0.8
    act_reg_weight: float = 0.001
    num_envs: int = 4096
