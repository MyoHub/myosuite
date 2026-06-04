# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Observation term functions for the two-agent competitive boxing environment.

All functions take ``(model, data, meta, agent_id, health, config)`` and return
a flat ``np.ndarray``.  They are pure (no side effects) and backend-specific
to CPU MuJoCo (``mujoco.MjModel`` / ``mujoco.MjData``).
"""

from __future__ import annotations

import mujoco
import numpy as np

from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_config import BoxingVsConfig
from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_model import (
    BoxingVsModelMeta,
)
from myosuite.terms.base_obs import (
    normalized_health_pair,
    qpos_for_joints,
    qvel_for_joints,
    sensor_vel3,
)

# ---------------------------------------------------------------------------
# Per-agent observation blocks
# ---------------------------------------------------------------------------


def own_kinematics_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    meta: BoxingVsModelMeta,
    agent_id: str,
    config: BoxingVsConfig,
) -> np.ndarray:
    """Own joint positions, velocities, and (optionally) muscle activations.

    Args:
        model: Compiled MjModel.
        data: Current MjData.
        meta: Pre-computed model metadata.
        agent_id: ``"agent_0"`` or ``"agent_1"``.
        config: Task configuration.

    Returns:
        Concatenated kinematic observation vector.
    """
    parts: list[np.ndarray] = []
    jnt_ids = meta.jnt_ids[agent_id]

    if config.observe_own_joints:
        parts.append(qpos_for_joints(model, data, jnt_ids).astype(np.float32))
        parts.append(qvel_for_joints(model, data, jnt_ids).astype(np.float32))

    if config.observe_own_muscles:
        act_idx = meta.act_indices[agent_id]
        # data.act may be shorter than model.nu when not all actuators have dynamics.
        if data.act is not None and len(data.act) >= max(act_idx) + 1:
            parts.append(data.act[act_idx].astype(np.float32))
        else:
            parts.append(data.ctrl[act_idx].astype(np.float32))

    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def own_fist_obs(
    data: mujoco.MjData,
    meta: BoxingVsModelMeta,
    agent_id: str,
) -> np.ndarray:
    """Own right and left fist world-frame positions and linear velocities.

    Returns a ``(12,)`` vector: [fist_r_pos(3), fist_l_pos(3),
    fist_r_vel(3), fist_l_vel(3)].

    Args:
        data: Current MjData.
        meta: Pre-computed model metadata.
        agent_id: ``"agent_0"`` or ``"agent_1"``.

    Returns:
        Shape ``(12,)`` float32 array.
    """
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
    meta: BoxingVsModelMeta,
    agent_id: str,
) -> np.ndarray:
    """Own pelvis world-frame position and linear velocity.

    Returns a ``(6,)`` vector: [pelvis_pos(3), pelvis_vel(3)].

    Args:
        data: Current MjData.
        meta: Pre-computed model metadata.
        agent_id: ``"agent_0"`` or ``"agent_1"``.

    Returns:
        Shape ``(6,)`` float32 array.
    """
    sites = meta.site_ids[agent_id]
    sensors = meta.sensor_adr[agent_id]
    return np.concatenate(
        [
            data.site_xpos[sites["pelvis_site"]],
            sensor_vel3(data, sensors["pelvis_site_vel"]),
        ]
    ).astype(np.float32)


def opponent_keypoints_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    meta: BoxingVsModelMeta,
    agent_id: str,
    config: BoxingVsConfig,
) -> np.ndarray:
    """Opponent fists, scoring zones, pelvis position and velocity.

    Provides the positional context needed for defence (guard timing,
    punch avoidance) and offence (target tracking).

    Returns a vector of size up to 30:
    - fist_r_pos(3), fist_l_pos(3) — where the opponent's fists are
    - fist_r_vel(3), fist_l_vel(3) — incoming punch speed/direction
    - head_zone_pos(3)             — head target location
    - body_zone_pos(3)             — torso target location
    - pelvis_pos(3), pelvis_vel(3) — opponent COM proxy

    Args:
        model: Compiled MjModel.
        data: Current MjData.
        meta: Pre-computed model metadata.
        agent_id: The *observing* agent (opponent obs come from the other one).
        config: Task configuration.

    Returns:
        Float32 array of opponent positional observations.
    """
    opp_id = "agent_1" if agent_id == "agent_0" else "agent_0"
    parts: list[np.ndarray] = []

    if config.observe_opponent_keypoints:
        opp_sites = meta.site_ids[opp_id]
        parts.extend(
            [
                data.site_xpos[opp_sites["fist_r"]],
                data.site_xpos[opp_sites["fist_l"]],
                data.site_xpos[opp_sites["head_zone"]],
                data.site_xpos[opp_sites["body_zone"]],
                data.site_xpos[opp_sites["pelvis_site"]],
            ]
        )

    if config.observe_opponent_velocities:
        opp_sensors = meta.sensor_adr[opp_id]
        parts.extend(
            [
                sensor_vel3(data, opp_sensors["fist_r_vel"]),
                sensor_vel3(data, opp_sensors["fist_l_vel"]),
                sensor_vel3(data, opp_sensors["pelvis_site_vel"]),
            ]
        )

    if config.observe_opponent_joint_subset:
        arm_jnt_ids = meta.arm_jnt_ids[opp_id]
        parts.append(qpos_for_joints(model, data, arm_jnt_ids))
        parts.append(qvel_for_joints(model, data, arm_jnt_ids))

    return (
        np.concatenate(parts).astype(np.float32)
        if parts
        else np.empty(0, dtype=np.float32)
    )


def health_obs(
    health: dict[str, float],
    agent_id: str,
    config: BoxingVsConfig,
) -> np.ndarray:
    """Own and opponent health, each normalised to [0, 1].

    Returns a ``(2,)`` vector: [own_health_norm, opp_health_norm].

    Args:
        health: Current health dict ``{"agent_0": float, "agent_1": float}``.
        agent_id: The observing agent.
        config: Task configuration (provides ``ko_health_threshold``).

    Returns:
        Shape ``(2,)`` float32 array.
    """
    return normalized_health_pair(health, agent_id, config.ko_health_threshold)


# ---------------------------------------------------------------------------
# Full observation assembly
# ---------------------------------------------------------------------------


def get_agent_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    meta: BoxingVsModelMeta,
    agent_id: str,
    health: dict[str, float],
    config: BoxingVsConfig,
) -> np.ndarray:
    """Assemble the full observation vector for one agent.

    Concatenation order:
    1. Own kinematics (joint_pos + joint_vel + muscle_act)
    2. Own fist positions + velocities
    3. Own pelvis position + velocity
    4. Opponent keypoints (fists, zones, pelvis) + velocities + arm joints
    5. Health (own + opponent, normalised)

    Args:
        model: Compiled MjModel.
        data: Current MjData.
        meta: Pre-computed model metadata.
        agent_id: ``"agent_0"`` or ``"agent_1"``.
        health: Current health state for both agents.
        config: Task configuration.

    Returns:
        1-D float32 observation vector.
    """
    parts: list[np.ndarray] = [
        own_kinematics_obs(model, data, meta, agent_id, config),
        own_fist_obs(data, meta, agent_id),
        own_pelvis_obs(data, meta, agent_id),
        opponent_keypoints_obs(model, data, meta, agent_id, config),
    ]
    if config.observe_health:
        parts.append(health_obs(health, agent_id, config))

    return np.concatenate(parts)


def obs_dim(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    meta: BoxingVsModelMeta,
    agent_id: str,
    health: dict[str, float],
    config: BoxingVsConfig,
) -> int:
    """Return the observation dimension for *agent_id* (determined by a dry run).

    Args:
        model: Compiled MjModel.
        data: Current MjData (must have been forward-passed at least once).
        meta: Pre-computed model metadata.
        agent_id: ``"agent_0"`` or ``"agent_1"``.
        health: Dummy health dict (values don't matter for shape).
        config: Task configuration.

    Returns:
        Integer dimension of the observation vector.
    """
    return int(get_agent_obs(model, data, meta, agent_id, health, config).shape[0])
