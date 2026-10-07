# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Pose-task terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEntityAccessor
from myosuite.terms.base_reward import pose_reward

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def pose_diverged(
    env: ManagerBasedRlEnv,
    command_name: str,
    far_thd: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """CPU ``done``: joint-space distance to the target exceeds ``far_thd``."""
    task_state = {"target_angles": env.command_manager.get_command(command_name)}
    accessor = MjlabEntityAccessor(env, asset_cfg.name)
    return pose_reward(accessor, task_state, far_thd=far_thd)["done"]
