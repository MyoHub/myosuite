# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Task config for the Baoding ball manipulation task on the mjlab backend."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

_ASSETS = pathlib.Path(__file__).parents[3] / "assets"


@dataclass
class BaodingCfg:
    """Configuration for the Baoding ball task on the mjlab/MuJoCo Warp backend.

    The agent controls the 39 hand muscle actuators to rotate two Baoding
    balls continuously in the palm.

    Args:
        model_path: Path to the MJCF model file.
        sim_dt: MuJoCo simulation timestep in seconds.
        ctrl_dt: Control timestep; must be a multiple of ``sim_dt``.
        max_episode_steps: Episode truncation length.
        ball_1_site: Name of the site tracking ball 1 position.
        ball_2_site: Name of the site tracking ball 2 position.
        rotation_speed: Target angular speed of ball rotation (rad/step).
        pos_reward_weight: Scaling weight for ball-position reward.
        act_reg_weight: Scaling weight for the activation L2 penalty.
        drop_penalty: Penalty applied when a ball is dropped (leaves palm).
        num_envs: Number of parallel environments (batch size on GPU).
    """

    model_path: pathlib.Path = field(
        default_factory=lambda: _ASSETS / "hand" / "myohand_baoding.xml"
    )
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02
    max_episode_steps: int = 200
    ball_1_site: str = "Ball1"
    ball_2_site: str = "Ball2"
    rotation_speed: float = 1.0
    pos_reward_weight: float = 1.0
    act_reg_weight: float = 0.01
    drop_penalty: float = -5.0
    num_envs: int = 4096
