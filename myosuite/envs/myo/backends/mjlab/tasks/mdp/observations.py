# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Observation terms shared by the MyoSuite mjlab tasks (CPU obs layout)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEntityAccessor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def qpos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """CPU ``obs["qpos"]``: the entity's ``qpos`` in MuJoCo layout."""
    return MjlabEntityAccessor(env, asset_cfg.name).joint_pos()


def qvel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """CPU ``obs["qvel"]``: ``qvel * ctrl_dt``."""
    return MjlabEntityAccessor(env, asset_cfg.name).joint_vel() * env.step_dt


def act(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """CPU ``obs["act"]``: muscle activation state."""
    return MjlabEntityAccessor(env, asset_cfg.name).muscle_act()
