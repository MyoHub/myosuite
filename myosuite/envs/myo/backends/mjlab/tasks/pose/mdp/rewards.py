# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Pose-task reward components (CPU ``PoseEnvV0.get_reward_dict``).

Each function returns one entry of the CPU reward dict; the reward manager
forms the same weighted sum as the CPU ``dense`` reward.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEntityAccessor
from myosuite.terms.base_reward import pose_reward

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def _components(
    env: ManagerBasedRlEnv,
    command_name: str,
    pose_thd: float,
    far_thd: float,
    asset_cfg: SceneEntityCfg,
) -> dict[str, Any]:
    task_state = {"target_angles": env.command_manager.get_command(command_name)}
    return pose_reward(
        MjlabEntityAccessor(env, asset_cfg.name),
        task_state,
        pose_thd=pose_thd,
        far_thd=far_thd,
    )


def pose_dist(
    env: ManagerBasedRlEnv,
    command_name: str,
    pose_thd: float,
    far_thd: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """``"pose"`` (and ``"sparse"``): negative L2 joint-space distance."""
    return _components(env, command_name, pose_thd, far_thd, asset_cfg)["pose"]


def pose_bonus(
    env: ManagerBasedRlEnv,
    command_name: str,
    pose_thd: float,
    far_thd: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """``"bonus"``: +1 within ``pose_thd``, +1 within ``1.5 * pose_thd``."""
    return _components(env, command_name, pose_thd, far_thd, asset_cfg)["bonus"]


def pose_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    pose_thd: float,
    far_thd: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """``"penalty"``: -1 when the distance exceeds ``far_thd``."""
    return _components(env, command_name, pose_thd, far_thd, asset_cfg)["penalty"]


def pose_solved(
    env: ManagerBasedRlEnv,
    command_name: str,
    pose_thd: float,
    far_thd: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """``"solved"`` as a float (only weighted when a CPU config weights it)."""
    return _components(env, command_name, pose_thd, far_thd, asset_cfg)["solved"].float()


def pose_done(
    env: ManagerBasedRlEnv,
    command_name: str,
    pose_thd: float,
    far_thd: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """``"done"`` as a float (only weighted when a CPU config weights it)."""
    return _components(env, command_name, pose_thd, far_thd, asset_cfg)["done"].float()


def act_norm(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """``"act_reg"``: ``-||act|| / na`` (zero for motor-only models)."""
    act = MjlabEntityAccessor(env, asset_cfg.name).muscle_act()
    if act.shape[-1] == 0:
        return torch.zeros(env.num_envs, device=env.device)
    return -torch.linalg.norm(act, dim=-1) / act.shape[-1]
