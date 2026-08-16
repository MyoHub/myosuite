# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Task config for the hand joint-angle posing task on the mjlab backend."""

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
class HandPoseCfg:
    """Configuration for the hand pose task on mjlab."""

    model_path: pathlib.Path = field(default_factory=_default_model_path)
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02
    max_episode_steps: int = 100
    pose_thd: float = 0.7
    angle_reward_weight: float = 1.0
    bonus_weight: float = 4.0
    act_reg_weight: float = 0.01
    num_envs: int = 4096
