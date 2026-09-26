# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reach-task terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .rewards import leg_reach_components, reach_components

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def reach_failed(
    env: ManagerBasedRlEnv,
    command_name: str,
    far_th: float,
    penalty_start_step: int,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """CPU ``done``: tips further than ``far_th`` per site (after 2 steps)."""
    comps = reach_components(env, command_name, far_th, penalty_start_step, asset_cfg)
    return comps["done"]


def leg_reach_failed(
    env: ManagerBasedRlEnv,
    command_name: str,
    far_th: float,
    penalty_start_step: int,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """CPU ``LegReachEnvV0`` ``done``: tips further than ``far_th`` per site."""
    comps = leg_reach_components(
        env, command_name, far_th, penalty_start_step, asset_cfg
    )
    return comps["done"]
