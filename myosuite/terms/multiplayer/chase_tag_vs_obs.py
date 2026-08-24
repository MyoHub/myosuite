# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Observation term functions for the two-agent chase-tag environment.

All functions are pure (no side effects), CPU MuJoCo only.
"""

from __future__ import annotations

import mujoco
import numpy as np

from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_config import (
    ChaseTagVsConfig,
)
from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_model import (
    ChaseTagVsModelMeta,
)
from myosuite.terms.base_obs import (
    qpos_for_joints,
    qvel_for_joints,
    sensor_vel3,
)
from myosuite.terms.opponent_relative_obs import relative_pose_obs

CHASER_ID = "agent_0"
RUNNER_ID = "agent_1"


def own_kinematics_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    meta: ChaseTagVsModelMeta,
    agent_id: str,
    config: ChaseTagVsConfig,
) -> np.ndarray:
    """Own joint positions, velocities, and (optionally) muscle activations."""
    parts: list[np.ndarray] = []
    jnt_ids = meta.jnt_ids[agent_id]
    if config.observe_own_joints:
        parts.append(qpos_for_joints(model, data, jnt_ids).astype(np.float32))
        parts.append(qvel_for_joints(model, data, jnt_ids).astype(np.float32))
    if config.observe_own_muscles:
        act_idx = meta.act_indices[agent_id]
        if data.act is not None and len(data.act) >= max(act_idx) + 1:
            parts.append(data.act[act_idx].astype(np.float32))
        else:
            parts.append(data.ctrl[act_idx].astype(np.float32))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def own_pelvis_obs(
    data: mujoco.MjData, meta: ChaseTagVsModelMeta, agent_id: str
) -> np.ndarray:
    """Own pelvis world-frame position and linear velocity, shape ``(6,)``."""
    sites = meta.site_ids[agent_id]
    sensors = meta.sensor_adr[agent_id]
    return np.concatenate(
        [
            data.site_xpos[sites["pelvis_site"]],
            sensor_vel3(data, sensors["pelvis_site_vel"]),
        ]
    ).astype(np.float32)


def opponent_relative_obs(
    data: mujoco.MjData,
    meta: ChaseTagVsModelMeta,
    agent_id: str,
    config: ChaseTagVsConfig,
) -> np.ndarray:
    """Opponent pelvis position/velocity relative to *agent_id*, shape ``(7,)``."""
    if not config.observe_opponent_pelvis:
        return np.empty(0, dtype=np.float32)
    opp_id = RUNNER_ID if agent_id == CHASER_ID else CHASER_ID
    own_sites = meta.site_ids[agent_id]
    opp_sites = meta.site_ids[opp_id]
    own_sensors = meta.sensor_adr[agent_id]
    opp_sensors = meta.sensor_adr[opp_id]

    own_pos = data.site_xpos[own_sites["pelvis_site"]]
    opp_pos = data.site_xpos[opp_sites["pelvis_site"]]
    own_vel = sensor_vel3(data, own_sensors["pelvis_site_vel"])
    opp_vel = sensor_vel3(data, opp_sensors["pelvis_site_vel"])
    return relative_pose_obs(own_pos, own_vel, opp_pos, opp_vel)


def role_obs(agent_id: str, task: str, config: ChaseTagVsConfig) -> np.ndarray:
    """One-hot role indicator.

    ``[1, 0]`` when this agent is the chaser, ``[0, 1]`` when runner.
    With ``task="CHASE"``, agent_0 is chaser; with ``task="EVADE"``, agent_1 is.
    """
    if not config.observe_role:
        return np.empty(0, dtype=np.float32)
    is_chaser = (agent_id == CHASER_ID and task == "CHASE") or (
        agent_id == RUNNER_ID and task == "EVADE"
    )
    return np.array([1.0, 0.0] if is_chaser else [0.0, 1.0], dtype=np.float32)


def get_agent_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    meta: ChaseTagVsModelMeta,
    agent_id: str,
    config: ChaseTagVsConfig,
    task: str = "CHASE",
) -> np.ndarray:
    """Assemble the full observation vector for one agent.

    Concatenation order:
    1. Own kinematics (joint_pos + joint_vel + muscle_act)
    2. Own pelvis position + velocity
    3. Opponent-relative pelvis position/velocity + distance
    4. Role one-hot (chaser / runner), conditioned on current task
    """
    parts: list[np.ndarray] = [
        own_kinematics_obs(model, data, meta, agent_id, config),
        own_pelvis_obs(data, meta, agent_id),
        opponent_relative_obs(data, meta, agent_id, config),
        role_obs(agent_id, task, config),
    ]
    return np.concatenate(parts)


def obs_dim(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    meta: ChaseTagVsModelMeta,
    agent_id: str,
    config: ChaseTagVsConfig,
    task: str = "CHASE",
) -> int:
    """Return the observation dimension for *agent_id* (determined by a dry run)."""
    return int(get_agent_obs(model, data, meta, agent_id, config, task).shape[0])
