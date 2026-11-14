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
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEntityAccessor
from myosuite.terms.base_obs import joint_vel_obs, root_planar_vel_obs
from myosuite.terms.base_reward import (
    act_reg,
    heading_reward,
    locomotion_streak,
    locomotion_sustained,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.manager_base import ManagerTermBaseCfg


def joint_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """CPU ``joint_vel`` observation: ``qvel`` scaled by the control step."""
    return joint_vel_obs(MjlabEntityAccessor(env, asset_cfg.name))


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


def heading_on_target(
    env: ManagerBasedRlEnv,
    command_name: str,
    target_speed: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """CPU ``on_target``: upright and within the tolerance of the commanded velocity."""
    return _heading(env, command_name, target_speed, asset_cfg)["solved"].float()


class HeadingSolved(ManagerTermBase):
    """CPU ``solved``: on target on every control step of the last ``window_s`` seconds.

    Holds the count of consecutive on-target steps of every env; a reset clears it.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        self._steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._steps[slice(None) if env_ids is None else env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        target_speed: float,
        asset_cfg: SceneEntityCfg,
        window_s: float,
    ) -> torch.Tensor:
        on_target = _heading(env, command_name, target_speed, asset_cfg)["solved"]
        self._steps = locomotion_streak(self._steps, on_target)
        return locomotion_sustained(self._steps, env.step_dt, window_s).float()
