# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Pose-task observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEntityAccessor
from myosuite.terms.base_obs import pose_error_obs

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def pose_err(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """CPU ``obs["pose_err"]``: target ``qpos`` minus current ``qpos``."""
    return pose_error_obs(
        MjlabEntityAccessor(env, asset_cfg.name),
        target=env.command_manager.get_command(command_name),
    )
