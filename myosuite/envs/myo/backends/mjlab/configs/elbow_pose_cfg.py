# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Task config for the elbow joint-angle posing task on the mjlab backend."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

from myosuite.envs.myo.assets._resolve import resolve_elbow_xml


@dataclass
class ElbowPoseCfg:
    """Configuration for the elbow pose task on the mjlab/MuJoCo Warp backend.

    The agent controls six elbow muscle actuators to reach a randomly sampled
    target joint angle for ``r_elbow_flex``.

    Args:
        model_path: Path to the MJCF model file.
        sim_dt: MuJoCo simulation timestep in seconds.
        ctrl_dt: Control timestep; must be a multiple of ``sim_dt``.
        max_episode_steps: Episode truncation length.
        target_jnt_name: Name of the target joint in the model.
        target_range: ``(low, high)`` sampling range for the target angle
            in radians.
        pose_thd: Distance threshold (rad) for the bonus reward.
        angle_reward_weight: Scaling weight for the pose distance component.
        bonus_weight: Scaling weight for the in-threshold bonus.
        act_reg_weight: Scaling weight for the muscle-activation L2 penalty.
        num_envs: Number of parallel environments (batch size on GPU).
    """

    model_path: pathlib.Path = field(
        default_factory=lambda: resolve_elbow_xml("myoelbow_1dof6muscles.xml")
    )
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02
    max_episode_steps: int = 200
    target_jnt_name: str = "r_elbow_flex"
    target_range: tuple[float, float] = (0.0, 2.5)
    pose_thd: float = 0.35
    angle_reward_weight: float = 1.0
    bonus_weight: float = 4.0
    act_reg_weight: float = 0.01
    num_envs: int = 4096
