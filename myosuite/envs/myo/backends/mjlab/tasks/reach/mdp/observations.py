# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reach-task observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEntityAccessor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def tip_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """CPU ``obs["tip_pos"]``: flattened tip-site positions (``asset_cfg.site_ids``)."""
    tips = MjlabEntityAccessor(env, asset_cfg.name).site_xpos(asset_cfg.site_ids)
    return tips.reshape(env.num_envs, -1)


def reach_err(
    env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """CPU ``obs["reach_err"]``: target positions minus tip positions."""
    return env.command_manager.get_command(command_name) - tip_pos(env, asset_cfg)
