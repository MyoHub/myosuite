# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reach-task reward components (shared ``multi_site_reach_reward`` term)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEntityAccessor
from myosuite.terms.base_reward import leg_reach_reward, multi_site_reach_reward

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def reach_components(
    env: ManagerBasedRlEnv,
    command_name: str,
    far_th: float,
    penalty_start_step: int,
    asset_cfg: SceneEntityCfg,
) -> dict[str, Any]:
    """All CPU ``ReachEnvV0`` reward entries for the current state.

    Args:
        env: The environment.
        command_name: Reach target command.
        far_th: Per-site failure distance.
        penalty_start_step: First control step at which the CPU ``data.time``
            exceeds ``2 * ctrl_dt`` (far penalty/termination active from then on).
        asset_cfg: Robot entity with the tip sites in ``site_names``.

    Returns:
        The reward dict of :func:`multi_site_reach_reward`.
    """
    task_state = {
        "tip_site_ids": asset_cfg.site_ids,
        "target_pos": env.command_manager.get_command(command_name),
        "penalty_active": env.episode_length_buf >= penalty_start_step,
    }
    return multi_site_reach_reward(
        MjlabEntityAccessor(env, asset_cfg.name), task_state, far_th=far_th
    )


def reach_term(
    env: ManagerBasedRlEnv,
    key: str,
    command_name: str,
    far_th: float,
    penalty_start_step: int,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """One entry (``key``) of the CPU reward dict, as a float tensor."""
    comps = reach_components(env, command_name, far_th, penalty_start_step, asset_cfg)
    return comps[key].float()


def leg_reach_components(
    env: ManagerBasedRlEnv,
    command_name: str,
    far_th: float,
    penalty_start_step: int,
    asset_cfg: SceneEntityCfg,
) -> dict[str, Any]:
    """All CPU ``LegReachEnvV0`` reward entries for the current state.

    Args:
        env: The environment.
        command_name: Reach target command.
        far_th: Per-site failure distance.
        penalty_start_step: First control step with the far penalty/termination.
        asset_cfg: Robot entity with the tip sites in ``site_names``.

    Returns:
        The reward dict of :func:`leg_reach_reward`.
    """
    task_state = {
        "tip_site_ids": asset_cfg.site_ids,
        "target_pos": env.command_manager.get_command(command_name),
        "penalty_active": env.episode_length_buf >= penalty_start_step,
    }
    return leg_reach_reward(
        MjlabEntityAccessor(env, asset_cfg.name), task_state, far_th=far_th
    )


def leg_reach_term(
    env: ManagerBasedRlEnv,
    key: str,
    command_name: str,
    far_th: float,
    penalty_start_step: int,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """One entry (``key``) of the CPU ``LegReachEnvV0`` reward dict, as a float tensor."""
    comps = leg_reach_components(
        env, command_name, far_th, penalty_start_step, asset_cfg
    )
    return comps[key].float()
