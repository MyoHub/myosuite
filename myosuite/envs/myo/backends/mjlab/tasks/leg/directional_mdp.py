# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MDP terms of the directional leg tasks (mjlab twin of ``myoLegDirectional*-v0``).

The CPU envs are data-driven (``TaskConfig``): every term is a shared function of the
``EnvAccessor`` (``base_obs`` / ``base_reward``), called here with the mjlab accessor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEntityAccessor
from myosuite.terms.base_obs import root_planar_vel_obs
from myosuite.terms.base_reward import act_reg, heading_reward

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def joint_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """CPU ``joint_vel`` observation: raw ``qvel`` (not scaled by the control step)."""
    return MjlabEntityAccessor(env, asset_cfg.name).joint_vel()


def root_planar_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return root_planar_vel_obs(MjlabEntityAccessor(env, asset_cfg.name))


def heading_cmd(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_command(command_name)


def _heading(
    env: ManagerBasedRlEnv, command_name: str, target_speed: float, asset_cfg
) -> dict:
    return heading_reward(
        MjlabEntityAccessor(env, asset_cfg.name),
        {"heading_dir": env.command_manager.get_command(command_name)},
        target_speed=target_speed,
    )


def heading_term(
    env: ManagerBasedRlEnv,
    command_name: str,
    target_speed: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """CPU ``heading`` reward: velocity tracking minus the fall penalty."""
    return _heading(env, command_name, target_speed, asset_cfg)["dense"].float()


def act_reg_term(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """CPU ``act_reg`` reward: ``-mean(act**2)``."""
    accessor = MjlabEntityAccessor(env, asset_cfg.name)
    return act_reg(accessor, {})["dense"].float()


def fallen(
    env: ManagerBasedRlEnv,
    command_name: str,
    target_speed: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """CPU ``done``: the pelvis is below the fall height."""
    return _heading(env, command_name, target_speed, asset_cfg)["done"].bool()


def heading_solved(
    env: ManagerBasedRlEnv,
    command_name: str,
    target_speed: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """CPU ``solved``: upright and within the tolerance of the commanded velocity."""
    return _heading(env, command_name, target_speed, asset_cfg)["solved"].float()
