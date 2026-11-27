# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reward, termination, and score functions for the two-agent chase-tag env.

1v1 rules:
- Binary tag: planar pelvis distance ≤ tag_radius_m → chaser wins, episode ends.
- Evade win: no tag before maxTime, or the pursuer falls.
- Runner fall does not end the episode; the pursuer can still tag.
- Score: 1 - t/maxTime for chase win; t/maxTime for evade win; 0 for loss.
"""

from __future__ import annotations

from typing import Any

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
from myosuite.terms.multiplayer.chase_tag_vs_obs import agent_is_chaser

CHASER_ID = "agent_0"
RUNNER_ID = "agent_1"


def _pelvis_distance(data: mujoco.MjData, meta: ChaseTagVsModelMeta) -> float:
    """Planar pelvis distance so a standing chaser can still tag a downed runner."""
    chaser_pos = data.site_xpos[meta.site_ids[CHASER_ID]["pelvis_site"]]
    runner_pos = data.site_xpos[meta.site_ids[RUNNER_ID]["pelvis_site"]]
    return float(np.linalg.norm(chaser_pos[:2] - runner_pos[:2]))


def is_tagged(
    data: mujoco.MjData,
    meta: ChaseTagVsModelMeta,
    config: ChaseTagVsConfig,
) -> bool:
    """Return True when planar pelvis distance is within tag_radius_m."""
    return _pelvis_distance(data, meta) <= config.tag_radius_m


def compute_damage(
    data: mujoco.MjData,
    meta: ChaseTagVsModelMeta,
    config: ChaseTagVsConfig,
    task: str = "CHASE",
) -> dict[str, float]:
    """Damage dict for the MultiAgentTaskEnv health/KO contract.

    A tag delivers ``ko_health_threshold`` damage instantly from the active
    chaser to the active runner, meeting the KO condition on that same step.
    No damage is accrued otherwise.
    """
    tagged = is_tagged(data, meta, config)
    chaser = CHASER_ID if task == "CHASE" else RUNNER_ID
    runner = RUNNER_ID if task == "CHASE" else CHASER_ID
    return {
        chaser: config.ko_health_threshold if tagged else 0.0,
        runner: 0.0,
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
    raw = 1.0 - t / config.maxTime if task == "CHASE" else t / config.maxTime
    return float(np.clip(raw, 0.0, 1.0))


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
    prev_distance: float | None = None,
) -> float:
    """Dense reward for *agent_id* this step.

    The proximity terms are potential-based (Ng et al., 1999): they pay the change
    of ``exp(-d)`` since the previous step, so the sum over an episode is bounded
    by the weight and staying close earns nothing; only the tag pays. The per-step
    survival reward goes to the runner only, since paying it to the chaser would
    reward putting the tag off.

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
        prev_distance: Pelvis distance after the previous step; ``None`` on the
            first step of an episode (no proximity term).

    Returns:
        Scalar reward.
    """
    is_chaser = agent_is_chaser(agent_id, task)

    r = 0.0 if is_chaser else config.r_survival_per_step
    r += config.r_act_reg * float(np.sum(actions**2))

    if self_fell:
        r += config.r_fall_penalty
    if opponent_fell:
        r -= config.r_fall_penalty  # opponent falling is good for us

    dist = _pelvis_distance(data, meta)
    closer = 0.0 if prev_distance is None else np.exp(-dist) - np.exp(-prev_distance)
    if is_chaser:
        r += config.r_chaser_proximity * closer
        if tagged:
            r += config.r_win_bonus
    else:
        r += config.r_runner_proximity * closer
        if tagged:
            r += config.r_loss_penalty

    return float(r)


def chase_tag_dense_reward(
    xp: Any,
    *,
    is_chaser: Any,
    tagged: Any,
    self_fell: Any,
    opponent_fell: Any,
    act_sq_sum: Any,
    dist: Any,
    config: ChaseTagVsConfig,
) -> Any:
    """Backend-agnostic dense chase-tag reward of one agent (scalars or batches).

    Args:
        xp: Array module (``numpy`` or ``torch``).
        is_chaser: Whether the agent chases this episode.
        tagged: Whether the chaser tagged the runner this step.
        self_fell: Whether the agent fell.
        opponent_fell: Whether the opponent fell.
        act_sq_sum: Sum of the squared (unclipped) actions of the agent.
        dist: Planar pelvis distance between the agents.
        config: Task configuration (reward weights).

    Returns:
        Reward, shaped like the inputs.
    """
    survival = config.r_survival_per_step
    if config.r_chaser_survival_per_step is not None:
        survival = xp.where(is_chaser, config.r_chaser_survival_per_step, survival)
    r = survival + config.r_act_reg * act_sq_sum
    r = r + config.r_fall_penalty * self_fell
    r = r + config.r_opponent_fall_bonus * opponent_fell
    proximity = xp.where(
        is_chaser, config.r_chaser_proximity, config.r_runner_proximity
    )
    r = r + proximity * xp.exp(-dist)
    return r + xp.where(is_chaser, config.r_win_bonus, config.r_loss_penalty) * tagged


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
    own_pos = data.site_xpos[meta.site_ids[agent_id]["pelvis_site"]]
    opp_pos = data.site_xpos[meta.site_ids[opponent_id]["pelvis_site"]]
    rel_xy = (opp_pos - own_pos)[:2]
    dense, parts = steering_reward(
        np,
        qpos_local=qpos_for_joints(model, data, local_jids),
        qvel_local=qvel_for_joints(model, data, local_jids),
        ref_qpos_local=gait_qpos_local[frame],
        ref_qvel_local=gait_qvel_local[frame],
        rel_xy=rel_xy,
        dist_xy=float(np.linalg.norm(rel_xy)),
        root_lin_vel=_agent_root_linear_vel(model, data, meta, agent_id),
        pursue=pursue,
        target_speed=target_speed,
        w_jpos=w_jpos,
        w_jvel=w_jvel,
        w_heading=w_heading,
    )
    return float(dense), {k: float(v) for k, v in parts.items()}


def steering_reward(
    xp: Any,
    *,
    qpos_local: Any,
    qvel_local: Any,
    ref_qpos_local: Any,
    ref_qvel_local: Any,
    rel_xy: Any,
    dist_xy: Any,
    root_lin_vel: Any,
    pursue: Any,
    target_speed: float,
    w_jpos: float,
    w_jvel: float,
    w_heading: float,
) -> tuple[Any, dict[str, Any]]:
    """Backend-agnostic gait-style + steering reward (scalars or batches).

    Args:
        xp: Array module (``numpy`` or ``torch``).
        qpos_local: Joint positions without the root, ``(..., nq - 7)``.
        qvel_local: Joint velocities without the root, ``(..., nv - 6)``.
        ref_qpos_local: Gait-clip joint positions of the current frame.
        ref_qvel_local: Gait-clip joint velocities of the current frame.
        rel_xy: Planar opponent position relative to the agent, ``(..., 2)``.
        dist_xy: Norm of *rel_xy*, ``(..., 1)`` (or a scalar).
        root_lin_vel: World-frame root linear velocity, ``(..., 3)``.
        pursue: Steer toward (``True``) or away from the opponent.
        target_speed: Target walking speed (m/s).
        w_jpos: Weight of the joint-position term.
        w_jvel: Weight of the joint-velocity term.
        w_heading: Weight of the steering term.

    Returns:
        ``(dense, {"gait_jpos", "gait_jvel", "steering_heading"})``.
    """
    jpos_r = mimic_joint_pos_reward(xp, qpos_local, ref_qpos_local)
    jvel_r = mimic_joint_vel_reward(xp, qvel_local, ref_qvel_local)
    far = dist_xy > 1e-6
    direction = xp.where(far, rel_xy / xp.where(far, dist_xy, 1.0), 0.0)
    direction = xp.where(pursue, direction, -direction)
    zero = direction[..., 0] * 0.0
    target_vel = xp.stack(
        [target_speed * direction[..., 0], target_speed * direction[..., 1], zero],
        -1,
    )
    heading_r = mimic_root_vel_reward(xp, root_lin_vel, target_vel, scale=1.0)
    dense = w_jpos * jpos_r + w_jvel * jvel_r + w_heading * heading_r
    return dense, {
        "gait_jpos": jpos_r,
        "gait_jvel": jvel_r,
        "steering_heading": heading_r,
    }
