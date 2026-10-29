# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reward, termination, and score functions for the two-agent chase-tag env.

Rules match MyoChallenge ``chasetag_v0``:
- Binary tag: distance ≤ tag_radius_m → chaser wins, episode ends.
- Evade win: runner survives the full episode (maxTime seconds).
- Score: 1 - t/maxTime for chase win; t/maxTime for evade win; 0 for loss.
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
from myosuite.terms.base_obs import qpos_for_joints, qvel_for_joints
from myosuite.terms.base_termination import pelvis_fall_termination
from myosuite.terms.mimic_reward import (
    mimic_joint_pos_reward,
    mimic_joint_vel_reward,
    mimic_root_vel_reward,
)

CHASER_ID = "agent_0"
RUNNER_ID = "agent_1"


def _pelvis_distance(data: mujoco.MjData, meta: ChaseTagVsModelMeta) -> float:
    chaser_pos = data.site_xpos[meta.site_ids[CHASER_ID]["pelvis_site"]]
    runner_pos = data.site_xpos[meta.site_ids[RUNNER_ID]["pelvis_site"]]
    return float(np.linalg.norm(chaser_pos - runner_pos))


def is_tagged(
    data: mujoco.MjData,
    meta: ChaseTagVsModelMeta,
    config: ChaseTagVsConfig,
) -> bool:
    """Return True when chaser pelvis is within tag_radius_m of runner pelvis."""
    return _pelvis_distance(data, meta) <= config.tag_radius_m


def compute_damage(
    data: mujoco.MjData,
    meta: ChaseTagVsModelMeta,
    config: ChaseTagVsConfig,
) -> dict[str, float]:
    """Damage dict for the MultiAgentTaskEnv health/KO contract.

    A tag delivers ``ko_health_threshold`` damage instantly to the runner,
    which meets the KO condition on that same step and ends the episode.
    No damage is accrued otherwise.
    """
    tagged = is_tagged(data, meta, config)
    return {
        CHASER_ID: config.ko_health_threshold if tagged else 0.0,
        RUNNER_ID: 0.0,
    }


def is_fallen(
    data: mujoco.MjData,
    meta: ChaseTagVsModelMeta,
    agent_id: str,
    config: ChaseTagVsConfig,
) -> bool:
    """Return True if *agent_id*'s pelvis is below the fall threshold."""
    pelvis_pos = data.site_xpos[meta.site_ids[agent_id]["pelvis_site"]]
    return pelvis_fall_termination(pelvis_pos, config.fall_pelvis_z_threshold)


def get_score(time: float, task: str, config: ChaseTagVsConfig) -> float:
    """MyoChallenge-compatible sparse score.

    Args:
        time: Elapsed simulation time at episode end (seconds).
        task: ``"CHASE"`` (agent_0 is chaser) or ``"EVADE"`` (agent_0 is runner).
        config: Task configuration.

    Returns:
        Score in [0, 1].
    """
    t = round(time, 4)
    if task == "CHASE":
        return 1.0 - t / config.maxTime
    return t / config.maxTime


def compute_agent_reward(
    data: mujoco.MjData,
    meta: ChaseTagVsModelMeta,
    agent_id: str,
    task: str,
    tagged: bool,
    self_fell: bool,
    opponent_fell: bool,
    actions: np.ndarray,
    config: ChaseTagVsConfig,
) -> float:
    """Dense reward for *agent_id* this step.

    Args:
        data: Current MjData.
        meta: Model metadata.
        agent_id: The rewarded agent (``CHASER_ID`` or ``RUNNER_ID``).
        task: Role of ``agent_0`` this episode — ``"CHASE"`` or ``"EVADE"``.
        tagged: True if the chaser tagged the runner this step.
        self_fell: True if this agent fell.
        opponent_fell: True if the opponent fell.
        actions: This agent's muscle activation vector.
        config: Task configuration.

    Returns:
        Scalar reward.
    """
    is_chaser = agent_id == CHASER_ID if task == "CHASE" else agent_id == RUNNER_ID

    r = config.r_survival_per_step
    r += config.r_act_reg * float(np.sum(actions**2))

    if self_fell:
        r += config.r_fall_penalty
    if opponent_fell:
        r -= config.r_fall_penalty  # opponent falling is good for us

    dist = _pelvis_distance(data, meta)
    if is_chaser:
        r += config.r_chaser_proximity * np.exp(-dist)
        if tagged:
            r += config.r_win_bonus
    else:
        r += config.r_runner_proximity * np.exp(-dist)
        if tagged:
            r += config.r_loss_penalty

    return float(r)


def _agent_root_linear_vel(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    meta: ChaseTagVsModelMeta,
    agent_id: str,
) -> np.ndarray:
    root_jid = int(meta.jnt_ids[agent_id][0])
    dofadr = int(model.jnt_dofadr[root_jid])
    return np.asarray(data.qvel[dofadr : dofadr + 3])


def compute_gait_directional_reward(
    cur_qpos_local: np.ndarray,
    cur_qvel_local: np.ndarray,
    cur_root_vel: np.ndarray,
    ref_qpos_local: np.ndarray,
    ref_qvel_local: np.ndarray,
    heading_dir: np.ndarray,
    target_speed: float,
    w_jpos: float,
    w_jvel: float,
    w_heading: float,
) -> tuple[float, dict[str, float]]:
    """Gait style imitation + heading reward for a single-agent directional policy."""
    jpos_r = mimic_joint_pos_reward(np, cur_qpos_local, ref_qpos_local)
    jvel_r = mimic_joint_vel_reward(np, cur_qvel_local, ref_qvel_local)

    norm = float(np.linalg.norm(heading_dir))
    direction = heading_dir / norm if norm > 1e-6 else np.array([1.0, 0.0])
    target_vel = np.array(
        [target_speed * direction[0], target_speed * direction[1], 0.0]
    )
    heading_r = mimic_root_vel_reward(np, cur_root_vel, target_vel, scale=1.0)

    dense = (
        w_jpos * float(jpos_r) + w_jvel * float(jvel_r) + w_heading * float(heading_r)
    )
    return dense, {
        "gait_jpos": float(jpos_r),
        "gait_jvel": float(jvel_r),
        "steering_heading": float(heading_r),
    }


def compute_gait_steering_reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    meta: ChaseTagVsModelMeta,
    agent_id: str,
    opponent_id: str,
    gait_qpos_local: np.ndarray,
    gait_qvel_local: np.ndarray,
    frame: int,
    target_speed: float,
    pursue: bool,
    w_jpos: float,
    w_jvel: float,
    w_heading: float,
) -> tuple[float, dict[str, float]]:
    """Gait style + live steering toward/away from opponent."""
    local_jids = meta.jnt_ids[agent_id][1:]
    cur_qpos_local = qpos_for_joints(model, data, local_jids)
    cur_qvel_local = qvel_for_joints(model, data, local_jids)
    jpos_r = mimic_joint_pos_reward(np, cur_qpos_local, gait_qpos_local[frame])
    jvel_r = mimic_joint_vel_reward(np, cur_qvel_local, gait_qvel_local[frame])

    own_pos = data.site_xpos[meta.site_ids[agent_id]["pelvis_site"]]
    opp_pos = data.site_xpos[meta.site_ids[opponent_id]["pelvis_site"]]
    rel_xy = (opp_pos - own_pos)[:2]
    dist = float(np.linalg.norm(rel_xy))
    direction = rel_xy / dist if dist > 1e-6 else np.zeros(2)
    if not pursue:
        direction = -direction
    target_vel = np.array(
        [target_speed * direction[0], target_speed * direction[1], 0.0]
    )
    cur_vel = _agent_root_linear_vel(model, data, meta, agent_id)
    heading_r = mimic_root_vel_reward(np, cur_vel, target_vel, scale=1.0)

    dense = (
        w_jpos * float(jpos_r) + w_jvel * float(jvel_r) + w_heading * float(heading_r)
    )
    return dense, {
        "gait_jpos": float(jpos_r),
        "gait_jvel": float(jvel_r),
        "steering_heading": float(heading_r),
    }
