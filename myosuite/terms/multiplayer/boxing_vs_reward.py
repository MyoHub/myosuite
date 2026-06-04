# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reward and termination term functions for the two-agent boxing environment.

All functions are pure (no side effects).  They consume MjModel / MjData
directly (CPU-only, matching the P0 boxing task convention).
"""

from __future__ import annotations

import mujoco
import numpy as np

from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_config import BoxingVsConfig
from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_model import (
    BoxingVsModelMeta,
)
from myosuite.terms.multiplayer.common import combine_combat_reward
from myosuite.terms.base_termination import pelvis_fall_termination


# ---------------------------------------------------------------------------
# Hit / damage
# ---------------------------------------------------------------------------


def compute_hit_damage(
    data: mujoco.MjData,
    meta: BoxingVsModelMeta,
    attacker_id: str,
    config: BoxingVsConfig,
) -> tuple[float, float]:
    """Compute damage inflicted by *attacker_id* on its opponent this step.

    Uses the same proximity-based detection as ``boxing_bag_reward`` in the P0
    task: if a fist site is within threshold distance of a scoring-zone site,
    damage is ``(1 − dist/threshold)`` scaled by the zone weight.  This
    soft-proximity formulation rewards landing fists close to the centre of the
    target and degrades gracefully near the threshold boundary.

    Both fists (left and right) are checked against both scoring zones (head
    and body).  The maximum hit signal per fist is used to avoid double-counting
    the same punch against overlapping zones.

    Args:
        data: Current MjData (kinematics must be up-to-date).
        meta: Pre-computed model metadata.
        attacker_id: ``"agent_0"`` or ``"agent_1"`` — the punching agent.
        config: Task configuration (thresholds and zone weights).

    Returns:
        ``(head_damage, body_damage)`` — non-negative floats for this step.
    """
    defender_id = "agent_1" if attacker_id == "agent_0" else "agent_0"

    atk_sites = meta.site_ids[attacker_id]
    def_sites = meta.site_ids[defender_id]

    head_pos = data.site_xpos[def_sites["head_zone"]]
    body_pos = data.site_xpos[def_sites["body_zone"]]

    head_damage = 0.0
    body_damage = 0.0

    for fist_key in ("fist_r", "fist_l"):
        fist_pos = data.site_xpos[atk_sites[fist_key]]

        # Head
        dist_head = float(np.linalg.norm(fist_pos - head_pos))
        if dist_head < config.hit_threshold_head_m:
            head_damage = max(
                head_damage,
                (1.0 - dist_head / config.hit_threshold_head_m) * config.w_head,
            )

        # Body
        dist_body = float(np.linalg.norm(fist_pos - body_pos))
        if dist_body < config.hit_threshold_body_m:
            body_damage = max(
                body_damage,
                (1.0 - dist_body / config.hit_threshold_body_m) * config.w_body,
            )

    return head_damage, body_damage


def compute_total_damage(
    data: mujoco.MjData,
    meta: BoxingVsModelMeta,
    attacker_id: str,
    config: BoxingVsConfig,
) -> float:
    """Total damage inflicted by *attacker_id* this step (head + body).

    Args:
        data: Current MjData.
        meta: Pre-computed model metadata.
        attacker_id: Punching agent.
        config: Task configuration.

    Returns:
        Non-negative float — sum of head and body damage.
    """
    h, b = compute_hit_damage(data, meta, attacker_id, config)
    return h + b


# ---------------------------------------------------------------------------
# Fall detection
# ---------------------------------------------------------------------------


def is_fallen(
    data: mujoco.MjData,
    meta: BoxingVsModelMeta,
    agent_id: str,
    config: BoxingVsConfig,
) -> bool:
    """Return True if *agent_id*'s pelvis is below the fall threshold.

    Uses the pelvis site Z coordinate as a proxy for centre-of-mass height.
    When a full-body model tips over or is knocked down, the pelvis drops
    substantially below 0.7 m (standing pelvis height ≈ 0.9–1.0 m).

    Args:
        data: Current MjData.
        meta: Pre-computed model metadata.
        agent_id: Agent to check.
        config: Task configuration (``fall_pelvis_z_threshold``).

    Returns:
        True if the agent has fallen.
    """
    pelvis_pos = data.site_xpos[meta.site_ids[agent_id]["pelvis_site"]]
    return pelvis_fall_termination(pelvis_pos, config.fall_pelvis_z_threshold)


# ---------------------------------------------------------------------------
# Per-agent reward
# ---------------------------------------------------------------------------


def compute_agent_reward(
    data: mujoco.MjData,
    meta: BoxingVsModelMeta,
    agent_id: str,
    delivered_damage: float,
    received_damage: float,
    self_fell: bool,
    opponent_fell: bool,
    ko_happened: bool,
    own_health: float,
    actions: np.ndarray,
    config: BoxingVsConfig,
) -> float:
    """Compute the scalar reward for *agent_id* this step.

    Reward composition:
    - Damage delivered to opponent  (per-step, dense)
    - Damage received from opponent (per-step, dense, negative)
    - Survival bonus                (per-step, dense)
    - KO / floor bonus              (sparse, positive, on opponent event)
    - Self-fall penalty             (sparse, negative, one-shot)
    - Activation regularisation     (per-step, dense, small negative)

    Args:
        data: Current MjData (unused here but kept for future extensions).
        meta: Pre-computed model metadata.
        agent_id: The rewarded agent.
        delivered_damage: Damage dealt by this agent this step.
        received_damage: Damage received by this agent this step.
        self_fell: True if this agent fell this step.
        opponent_fell: True if the opponent fell this step.
        ko_happened: True if the opponent was KO'd (health cap reached) this step.
        own_health: Current accumulated health for this agent.
        actions: The action vector applied by this agent (for regularisation).
        config: Task configuration.

    Returns:
        Scalar reward.
    """
    return combine_combat_reward(
        delivered_damage=delivered_damage,
        received_damage=received_damage,
        actions=actions,
        r_damage_delivered=config.r_damage_delivered,
        r_damage_received=config.r_damage_received,
        r_survival_per_step=config.r_survival_per_step,
        r_act_reg=config.r_act_reg,
        self_fell=self_fell,
        opponent_fell=opponent_fell,
        ko_happened=ko_happened,
        r_fall_bonus=config.r_fall_bonus,
        r_ko_bonus=config.r_ko_bonus,
        r_self_fall_penalty=config.r_self_fall_penalty,
    )
