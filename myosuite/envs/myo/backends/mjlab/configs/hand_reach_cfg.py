# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Task config for the hand fingertip-reaching task on the mjlab backend."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field


def _default_model_path() -> pathlib.Path:
    """Materialize the myo_sim-native "hand_pose" recipe to a real file.

    mjlab reads ``model_path`` as a plain file path (it has no hook into
    myosuite's ModelBuilder/recipe machinery), so the recipe's live MjSpec
    has to be written to disk. Verified numerically equivalent to the
    legacy myohand_pose.xml this replaces: 39/39 muscle names match, 0
    calibration (gainprm/biasprm) mismatches, nq matches exactly (23).
    """
    from myosuite.core.model_recipes import materialize_recipe_xml

    return materialize_recipe_xml("hand_pose")


@dataclass
class HandReachCfg:
    """Configuration for the hand reach task on the mjlab/MuJoCo Warp backend.

    The agent controls the 23 hand muscle actuators to move the index
    fingertip (IFtip_r) to a randomly sampled Cartesian target position.

    Args:
        model_path: Path to the MJCF model file.
        sim_dt: MuJoCo simulation timestep in seconds.
        ctrl_dt: Control timestep; must be a multiple of ``sim_dt``.
        max_episode_steps: Episode truncation length.
        tip_site_name: Name of the end-effector site in the model.
        target_reach_range: ``(low, high)`` bounding box (metres) for
            target sampling, shape ``(3,)`` each.
        reach_thd: Distance threshold (m) for the bonus reward.
        reach_reward_weight: Scaling weight for the distance component.
        bonus_weight: Scaling weight for the in-threshold bonus.
        penalty_weight: Scaling weight for joint-limit penalty.
        num_envs: Number of parallel environments (batch size on GPU).
    """

    model_path: pathlib.Path = field(default_factory=_default_model_path)
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02
    max_episode_steps: int = 200
    tip_site_name: str = "IFtip_r"
    target_reach_range: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ] = (
        (-0.2, -0.6, 1.35),
        (0.0, -0.45, 1.55),
    )
    reach_thd: float = 0.05
    reach_reward_weight: float = 1.0
    bonus_weight: float = 4.0
    penalty_weight: float = 50.0
    num_envs: int = 4096
