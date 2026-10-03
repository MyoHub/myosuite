# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Shared helpers for multi-agent combat reward composition."""

from __future__ import annotations

import numpy as np


def combine_combat_reward(
    *,
    delivered_damage: float,
    received_damage: float,
    actions: np.ndarray,
    r_damage_delivered: float,
    r_damage_received: float,
    r_survival_per_step: float,
    r_act_reg: float,
    self_fell: bool = False,
    opponent_fell: bool = False,
    ko_happened: bool = False,
    r_fall_bonus: float = 0.0,
    r_ko_bonus: float = 0.0,
    r_self_fall_penalty: float = 0.0,
) -> float:
    """Compute the common dense+sparse combat reward scalar.

    Args:
        delivered_damage: Damage dealt by this agent this step.
        received_damage: Damage received by this agent this step.
        actions: Applied action vector for regularization.
        r_damage_delivered: Weight for delivered damage.
        r_damage_received: Weight for received damage.
        r_survival_per_step: Per-step survival bonus.
        r_act_reg: Action regularization weight.
        self_fell: Whether this agent fell this step.
        opponent_fell: Whether the opponent fell this step.
        ko_happened: Whether opponent KO happened this step.
        r_fall_bonus: Bonus for opponent fall.
        r_ko_bonus: Bonus for opponent KO.
        r_self_fall_penalty: Penalty when this agent falls.

    Returns:
        Scalar reward value.
    """
    reward = 0.0
    reward += delivered_damage * r_damage_delivered
    reward += received_damage * r_damage_received
    reward += r_survival_per_step

    if opponent_fell:
        reward += r_fall_bonus
    if ko_happened:
        reward += r_ko_bonus
    if self_fell:
        reward += r_self_fall_penalty

    reward += r_act_reg * float(np.sum(actions**2))
    return float(reward)
