# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reward and damage terms for boxer-vs-mannequin task."""

from __future__ import annotations

import mujoco
import numpy as np

from myosuite.envs.myo.tasks.challenge.boxing_mannequin.boxing_mannequin_config import (
    BoxingMannequinConfig,  # noqa: E501
)
from myosuite.envs.myo.tasks.challenge.boxing_mannequin.boxing_mannequin_model import (
    BoxingMannequinModelMeta,  # noqa: E501
)
from myosuite.terms.multiplayer.common import combine_combat_reward
from myosuite.terms.base_termination import pelvis_fall_termination


def compute_zone_damage(
    data: mujoco.MjData,
    meta: BoxingMannequinModelMeta,
    attacker_id: str,
    defender_id: str,
    config: BoxingMannequinConfig,
) -> tuple[float, float, float]:
    """Compute head/chest/waist proximity damage from attacker fists."""
    atk_sites = meta.site_ids[attacker_id]
    def_sites = meta.site_ids[defender_id]
    zones = (
        ("head_zone", config.hit_threshold_head_m, config.w_head),
        ("body_zone", config.hit_threshold_body_m, config.w_body),
        ("waist_zone", config.hit_threshold_waist_m, config.w_waist),
    )
    damage = {"head_zone": 0.0, "body_zone": 0.0, "waist_zone": 0.0}
    for fist_key in ("fist_r", "fist_l"):
        fist_pos = data.site_xpos[atk_sites[fist_key]]
        for zone_name, threshold, weight in zones:
            zone_pos = data.site_xpos[def_sites[zone_name]]
            dist = float(np.linalg.norm(fist_pos - zone_pos))
            if dist < threshold:
                damage[zone_name] = max(
                    damage[zone_name], (1.0 - dist / threshold) * weight
                )
    return damage["head_zone"], damage["body_zone"], damage["waist_zone"]


def compute_total_damage(
    data: mujoco.MjData,
    meta: BoxingMannequinModelMeta,
    attacker_id: str,
    defender_id: str,
    config: BoxingMannequinConfig,
) -> float:
    """Return sum of head/chest/waist damage inflicted this step."""
    h, c, w = compute_zone_damage(data, meta, attacker_id, defender_id, config)
    return h + c + w


def compute_pad_damage(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    meta: BoxingMannequinModelMeta,
    attacker_id: str,
    hit_force_threshold_n: float,
) -> float:
    """Compute pad-contact damage from fist-only contacts for BoxingP0."""
    if attacker_id != "agent_0":
        return 0.0
    if not meta.target_geom_ids:
        return 0.0
    target_geom_to_idx = {
        int(geom_id): idx for idx, geom_id in enumerate(meta.target_geom_ids)
    }
    fist_body_ids = meta.fist_body_ids.get(attacker_id, set())
    if not fist_body_ids:
        return 0.0
    force_buf = np.zeros(6, dtype=np.float64)
    pad_force = np.zeros(len(meta.target_geom_ids), dtype=np.float64)
    for contact_idx in range(int(data.ncon)):
        contact = data.contact[contact_idx]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        target_idx = target_geom_to_idx.get(geom1)
        other_geom = geom2
        if target_idx is None:
            target_idx = target_geom_to_idx.get(geom2)
            other_geom = geom1
        if target_idx is None:
            continue
        other_body = int(model.geom_bodyid[other_geom])
        if other_body not in fist_body_ids:
            continue
        mujoco.mj_contactForce(model, data, contact_idx, force_buf)
        pad_force[target_idx] += abs(float(force_buf[0]))

    force_norm = np.clip(pad_force / max(hit_force_threshold_n, 1e-6), 0.0, 3.0)
    hit_bonus = (pad_force >= hit_force_threshold_n).astype(np.float64)
    return float(np.sum(force_norm + hit_bonus))


def compute_pad_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    meta: BoxingMannequinModelMeta,
    attacker_id: str,
    hit_force_threshold_n: float,
) -> dict[str, float]:
    """Return per-step pad contact diagnostics for BoxingP0."""
    if attacker_id != "agent_0" or not meta.target_geom_ids:
        return {"pad_hits_step": 0.0, "pad_force_sum": 0.0, "pad_force_max": 0.0}
    target_geom_to_idx = {
        int(geom_id): idx for idx, geom_id in enumerate(meta.target_geom_ids)
    }
    fist_body_ids = meta.fist_body_ids.get(attacker_id, set())
    force_buf = np.zeros(6, dtype=np.float64)
    pad_force = np.zeros(len(meta.target_geom_ids), dtype=np.float64)
    for contact_idx in range(int(data.ncon)):
        contact = data.contact[contact_idx]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        target_idx = target_geom_to_idx.get(geom1)
        other_geom = geom2
        if target_idx is None:
            target_idx = target_geom_to_idx.get(geom2)
            other_geom = geom1
        if target_idx is None:
            continue
        other_body = int(model.geom_bodyid[other_geom])
        if other_body not in fist_body_ids:
            continue
        mujoco.mj_contactForce(model, data, contact_idx, force_buf)
        pad_force[target_idx] += abs(float(force_buf[0]))
    hits = float(np.count_nonzero(pad_force >= hit_force_threshold_n))
    return {
        "pad_hits_step": hits,
        "pad_force_sum": float(np.sum(pad_force)),
        "pad_force_max": float(np.max(pad_force) if pad_force.size else 0.0),
    }


def is_fallen(
    data: mujoco.MjData,
    meta: BoxingMannequinModelMeta,
    agent_id: str,
    config: BoxingMannequinConfig,
) -> bool:
    """Return True when pelvis site falls below configured height."""
    pelvis_pos = data.site_xpos[meta.site_ids[agent_id]["pelvis_site"]]
    return pelvis_fall_termination(pelvis_pos, config.fall_pelvis_z_threshold)


def compute_agent_reward(
    delivered_damage: float,
    received_damage: float,
    self_fell: bool,
    opponent_fell: bool,
    ko_happened: bool,
    actions: np.ndarray,
    config: BoxingMannequinConfig,
) -> float:
    """Compute scalar reward with VS-aligned terms."""
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
