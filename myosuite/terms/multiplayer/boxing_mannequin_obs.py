# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Observation terms for boxer-vs-mannequin task."""

from __future__ import annotations

import mujoco
import numpy as np

from myosuite.envs.myo.tasks.challenge.boxing_mannequin.boxing_mannequin_model import (
    BoxingMannequinModelMeta,
)
from myosuite.terms.base_obs import (
    normalized_health_pair,
    qpos_for_joints,
    qvel_for_joints,
    sensor_vel3,
)

# pylint: disable=no-member


def own_kinematics_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    meta: BoxingMannequinModelMeta,
    agent_id: str,
) -> np.ndarray:
    joints = qpos_for_joints(model, data, meta.jnt_ids[agent_id]).astype(np.float32)
    vels = qvel_for_joints(model, data, meta.jnt_ids[agent_id]).astype(np.float32)
    act_idx = meta.act_indices[agent_id]
    acts = (
        data.act[act_idx].astype(np.float32)
        if data.act is not None and len(data.act) >= max(act_idx) + 1
        else data.ctrl[act_idx].astype(np.float32)
    )
    return np.concatenate([joints, vels, acts])


def own_fist_obs(
    data: mujoco.MjData,
    meta: BoxingMannequinModelMeta,
    agent_id: str,
) -> np.ndarray:
    sites = meta.site_ids[agent_id]
    sensors = meta.sensor_adr[agent_id]
    return np.concatenate(
        [
            data.site_xpos[sites["fist_r"]],
            data.site_xpos[sites["fist_l"]],
            sensor_vel3(data, sensors["fist_r_vel"]),
            sensor_vel3(data, sensors["fist_l_vel"]),
        ]
    ).astype(np.float32)


def own_pelvis_obs(
    data: mujoco.MjData,
    meta: BoxingMannequinModelMeta,
    agent_id: str,
) -> np.ndarray:
    sites = meta.site_ids[agent_id]
    sensors = meta.sensor_adr[agent_id]
    return np.concatenate(
        [
            data.site_xpos[sites["pelvis_site"]],
            sensor_vel3(data, sensors["pelvis_site_vel"]),
        ]
    ).astype(np.float32)


def _pseudo_joint_subset_obs(
    data: mujoco.MjData,
    meta: BoxingMannequinModelMeta,
    opp_id: str,
) -> np.ndarray:
    """Create 28-D opponent proxy matching VS arm-subset size."""
    sites = meta.site_ids[opp_id]
    fist_r = data.site_xpos[sites["fist_r"]]
    fist_l = data.site_xpos[sites["fist_l"]]
    head = data.site_xpos[sites["head_zone"]]
    chest = data.site_xpos[sites["body_zone"]]
    pelvis = data.site_xpos[sites["pelvis_site"]]
    sensors = meta.sensor_adr[opp_id]
    fist_r_vel = sensor_vel3(data, sensors["fist_r_vel"])
    fist_l_vel = sensor_vel3(data, sensors["fist_l_vel"])
    pelvis_vel = sensor_vel3(data, sensors["pelvis_site_vel"])

    rel_r = fist_r - pelvis
    rel_l = fist_l - pelvis
    head_rel = head - pelvis
    chest_rel = chest - pelvis
    features = np.concatenate(
        [rel_r, rel_l, head_rel, chest_rel, fist_r_vel, fist_l_vel, pelvis_vel]
    )
    if features.shape[0] < 28:
        features = np.concatenate([features, np.zeros(28 - features.shape[0])])
    return features[:28].astype(np.float32)


def opponent_keypoints_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    meta: BoxingMannequinModelMeta,
    agent_id: str,
) -> np.ndarray:
    opp_id = "agent_1" if agent_id == "agent_0" else "agent_0"
    opp_sites = meta.site_ids[opp_id]
    opp_sensors = meta.sensor_adr[opp_id]
    keypoints = np.concatenate(
        [
            data.site_xpos[opp_sites["fist_r"]],
            data.site_xpos[opp_sites["fist_l"]],
            data.site_xpos[opp_sites["head_zone"]],
            data.site_xpos[opp_sites["body_zone"]],
            data.site_xpos[opp_sites["pelvis_site"]],
            sensor_vel3(data, opp_sensors["fist_r_vel"]),
            sensor_vel3(data, opp_sensors["fist_l_vel"]),
            sensor_vel3(data, opp_sensors["pelvis_site_vel"]),
        ]
    )
    if opp_id == "agent_0":
        arm = np.concatenate(
            [
                qpos_for_joints(model, data, meta.arm_jnt_ids[opp_id]),
                qvel_for_joints(model, data, meta.arm_jnt_ids[opp_id]),
            ]
        )
    else:
        arm = _pseudo_joint_subset_obs(data, meta, opp_id)
    return np.concatenate([keypoints.astype(np.float32), arm.astype(np.float32)])


def health_obs(
    health: dict[str, float], agent_id: str, ko_health_threshold: float
) -> np.ndarray:
    return normalized_health_pair(health, agent_id, ko_health_threshold)


def target_pad_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    meta: BoxingMannequinModelMeta,
    agent_id: str,
    hit_force_threshold_n: float = 15.0,
) -> np.ndarray:
    """Pad-target observations for BoxingP0: pos, normal, force, hit flag.

    Each target contributes 8 values:
    [pos(3), normal(3), normal_contact_force(1), hit(1)].
    """
    target_obs_dim = max(1, len(meta.target_site_ids)) * 8
    if agent_id != "agent_0":
        return np.zeros(target_obs_dim, dtype=np.float32)
    if len(meta.target_site_ids) == 0 or len(meta.target_geom_ids) == 0:
        return np.zeros(target_obs_dim, dtype=np.float32)

    pad_force = np.zeros(len(meta.target_geom_ids), dtype=np.float64)
    boxer_fist_body_ids = meta.fist_body_ids.get("agent_0", set())
    force_buf = np.zeros(6, dtype=np.float64)
    target_geom_to_idx = {
        int(geom_id): idx for idx, geom_id in enumerate(meta.target_geom_ids)
    }
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
        if other_body not in boxer_fist_body_ids:
            continue
        mujoco.mj_contactForce(model, data, contact_idx, force_buf)
        pad_force[target_idx] += abs(float(force_buf[0]))

    parts: list[np.ndarray] = []
    for pad_idx, site_id in enumerate(meta.target_site_ids):
        pos = data.site_xpos[site_id].astype(np.float32)
        # Use site local +Z axis as the outward pad normal in world frame.
        site_mat = data.site_xmat[site_id].reshape(3, 3)
        normal = site_mat[:, 2].astype(np.float32)
        normal_force = np.float32(pad_force[pad_idx])
        hit = np.float32(normal_force >= hit_force_threshold_n)
        parts.append(np.concatenate([pos, normal, [normal_force, hit]]))
    return np.concatenate(parts).astype(np.float32)


def get_agent_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    meta: BoxingMannequinModelMeta,
    agent_id: str,
    health: dict[str, float],
    ko_health_threshold: float,
    target_hit_normal_force_threshold_n: float = 15.0,
) -> np.ndarray:
    return np.concatenate(
        [
            own_kinematics_obs(model, data, meta, agent_id),
            own_fist_obs(data, meta, agent_id),
            own_pelvis_obs(data, meta, agent_id),
            opponent_keypoints_obs(model, data, meta, agent_id),
            target_pad_obs(
                model,
                data,
                meta,
                agent_id,
                hit_force_threshold_n=target_hit_normal_force_threshold_n,
            ),
            health_obs(health, agent_id, ko_health_threshold),
        ]
    )


def obs_dim(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    meta: BoxingMannequinModelMeta,
    agent_id: str,
    health: dict[str, float],
    ko_health_threshold: float,
    target_hit_normal_force_threshold_n: float = 15.0,
) -> int:
    return int(
        get_agent_obs(
            model,
            data,
            meta,
            agent_id,
            health,
            ko_health_threshold,
            target_hit_normal_force_threshold_n,
        ).shape[0]
    )
