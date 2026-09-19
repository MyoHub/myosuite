# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""
Reward term functions for MyoSuite environments.

All functions receive an EnvAccessor and a task_state dict, and return
a dict with scalar reward components plus boolean "solved" and "done" flags.
The native array type (numpy / jax.Array / torch.Tensor) is determined by
the backend.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from myosuite.core.protocols import EnvAccessor

# ---------------------------------------------------------------------------
# Named constants — avoids magic numbers in reward computations below.
# These match the values used in published MyoChallenge baselines; changing
# them will alter reward scale and trained-policy behaviour.
# ---------------------------------------------------------------------------

# ── pose_reward ──────────────────────────────────────────────────────────────
_POSE_BONUS_FAR_MULTIPLIER: float = 1.5  # secondary bonus at 1.5× threshold
_POSE_PENALTY_THRESHOLD: float = 2 * math.pi  # penalise wraps > one full rotation

# ── reach_reward ─────────────────────────────────────────────────────────────
_REACH_BONUS_FAR_MULTIPLIER: float = 2.0  # secondary bonus at 2× threshold

# ── walk_env_reward ───────────────────────────────────────────────────────────
_WALK_HIP_AMPLITUDE: float = 0.8  # sinusoidal hip target amplitude (rad)
_WALK_REF_ROT_SCALE: float = 5.0  # scaling for quaternion distance
_WALK_JOINT_ANGLE_SCALE: float = 5.0  # exp(-scale * mean |hip_angles|)
_WALK_W_VEL: float = 5.0  # dense weight: forward velocity reward
_WALK_W_FALL: float = -100.0  # dense weight: fall/termination penalty
_WALK_W_CYCLIC: float = -10.0  # dense weight: cyclic hip deviation
_WALK_W_REF_ROT: float = 10.0  # dense weight: reference rotation match
_WALK_W_JOINT: float = 5.0  # dense weight: joint angle regularisation

# ── heading_reward ───────────────────────────────────────────────────────────
_HEADING_FALL_HEIGHT: float = 0.7  # pelvis z (m) below which the agent has fallen
_HEADING_FALL_PENALTY: float = 1.0  # dense reward subtracted on fall


def _xp_asarray(xp: Any, value: Any, *, dtype: Any | None = None) -> Any:
    """Convert *value* to the backend array type used by *xp*."""

    if getattr(xp, "__name__", "") == "torch":
        tensor = xp.as_tensor(value)
        return tensor.to(dtype=dtype) if dtype is not None else tensor
    return xp.asarray(value, dtype=dtype) if dtype is not None else xp.asarray(value)


def _maybe_item(value: Any) -> Any:
    """Convert scalar backend arrays/tensors to Python scalars for CPU callers."""

    ndim = getattr(value, "ndim", None)
    if ndim == 0 and hasattr(value, "item"):
        return value.item()
    return value


def pose_reward(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    pose_thd: float = 0.35,
    **kwargs: Any,
) -> dict[str, Any]:
    """Reward for reaching a target joint configuration.

    Returns a negative distance reward plus discrete bonuses for being
    within threshold and a penalty for wildly out-of-range poses.

    Args:
        accessor: Environment state accessor.
        task_state: Must contain ``"target_angles"`` — target joint positions.
        pose_thd: Distance threshold (radians) for the bonus reward.
        **kwargs: Unused extra keyword arguments.

    Returns:
        Dict with keys: ``pose``, ``bonus``, ``penalty``, ``dense``,
        ``solved`` (bool scalar), ``done`` (bool scalar).
    """
    xp = accessor.array_module()
    dist = xp.linalg.norm(task_state["target_angles"] - accessor.joint_pos(), axis=-1)
    pose = -dist
    bonus = 1.0 * (dist < pose_thd) + 1.0 * (
        dist < _POSE_BONUS_FAR_MULTIPLIER * pose_thd
    )
    penalty = -1.0 * (dist > _POSE_PENALTY_THRESHOLD)
    dense = pose + bonus + penalty
    return {
        "pose": pose,
        "bonus": bonus,
        "penalty": penalty,
        "dense": dense,
        "solved": dist < pose_thd,
        "done": dist > _POSE_PENALTY_THRESHOLD,
    }


def reach_reward(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    reach_thd: float = 0.05,
    **kwargs: Any,
) -> dict[str, Any]:
    """Reward for moving a fingertip close to a target position.

    Args:
        accessor: Environment state accessor.
        task_state: Must contain ``"target_pos"`` (shape (3,)) and
            ``"tip_site_ids"`` for the end-effector sites.
        reach_thd: Distance threshold (metres) for the bonus reward.
        **kwargs: Unused extra keyword arguments.

    Returns:
        Dict with keys: ``reach``, ``bonus``, ``dense``, ``solved``, ``done``.
    """
    xp = accessor.array_module()
    tip_pos = accessor.site_xpos(task_state["tip_site_ids"])
    # Use mean tip position when multiple sites are provided
    if tip_pos.ndim > 1:
        tip_pos = xp.mean(tip_pos, axis=0)
    dist = xp.linalg.norm(task_state["target_pos"] - tip_pos)
    reach = -dist
    bonus = 1.0 * (dist < reach_thd) + 1.0 * (
        dist < _REACH_BONUS_FAR_MULTIPLIER * reach_thd
    )
    dense = reach + bonus
    return {
        "reach": reach,
        "bonus": bonus,
        "dense": dense,
        "solved": dist < reach_thd,
        "done": False,
    }


def act_reg(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    weight: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """L2 regularisation penalty on muscle activations.

    Args:
        accessor: Environment state accessor.
        task_state: Unused; present for uniform call signature.
        weight: Scaling weight applied to the L2 norm.
        **kwargs: Unused extra keyword arguments.

    Returns:
        Dict with key ``act_reg`` (negative scalar).
    """
    xp = accessor.array_module()
    act = accessor.muscle_act()
    penalty = -weight * xp.mean(act**2, axis=-1)
    return {"act_reg": penalty, "dense": penalty, "solved": False, "done": False}


def movement_efficiency_reward(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Reward efficient muscle usage via ``1 - mean(normalized activation^2)``.

    Args:
        accessor: Environment state accessor.
        task_state: Unused shared task-state dict.
        **kwargs: Unused extra reward configuration values.

    Returns:
        Dict containing the scalar ``movement_efficiency`` reward and its
        matching ``dense`` contribution.
    """
    del task_state, kwargs
    xp = accessor.array_module()
    act = accessor.muscle_act()
    mean_squared_activation = xp.mean(act**2, axis=-1)
    efficiency = 1.0 - mean_squared_activation
    return {
        "movement_efficiency": efficiency,
        "mean_squared_activation": mean_squared_activation,
        "dense": efficiency,
        "solved": False,
        "done": False,
    }


def upright_posture_reward(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    upright_posture_threshold: float = 0.90,
    **kwargs: Any,
) -> dict[str, Any]:
    """Reward maintaining an upright torso posture.

    Args:
        accessor: Environment state accessor.
        task_state: Must contain ``"upright_posture"`` in ``[0, 1]``.
        **kwargs: Unused compatibility keys from task config.

    Returns:
        Dict with ``upright_posture`` and ``dense`` equal to the upright score.
    """
    del kwargs
    xp = accessor.array_module()
    posture = _xp_asarray(
        xp,
        task_state.get("upright_posture", 0.0),
        dtype=getattr(xp, "float32", None),
    )
    reward = posture - float(upright_posture_threshold)
    done = posture < float(upright_posture_threshold)
    return {
        "upright_posture": _maybe_item(posture),
        "dense": _maybe_item(reward),
        "solved": False,
        "done": _maybe_item(done),
    }


def heading_reward(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    heading_dir: tuple[float, float] = (0.0, 1.0),
    target_speed: float = 1.0,
    fall_height: float = _HEADING_FALL_HEIGHT,
    fall_penalty: float = _HEADING_FALL_PENALTY,
    **kwargs: Any,
) -> dict[str, Any]:
    """Reward tracking a commanded planar velocity ``target_speed * heading_dir``.

    Assumes the model's first joint is a ``freejoint`` so ``joint_vel()[:2]``
    is the root's world-frame horizontal velocity and ``joint_pos()[2]`` is
    its height, used for fall detection.

    Args:
        accessor: Environment state accessor.
        task_state: Per-episode task state. If it carries ``heading_dir`` (set
            by a command-randomized ``reset_task``), it overrides the static
            ``heading_dir`` kwarg so the reward tracks the sampled command.
        heading_dir: Commanded unit direction ``(dx, dy)`` (fallback default).
        target_speed: Commanded speed (m/s) along ``heading_dir``.
        fall_height: Pelvis height (m) below which the agent is considered fallen.
        fall_penalty: Dense reward subtracted when fallen.
        **kwargs: Unused extra keyword arguments.

    Returns:
        Dict with keys: ``heading_tracking``, ``dense``, ``solved``, ``done``.
    """
    if isinstance(task_state, dict) and "heading_dir" in task_state:
        heading_dir = task_state["heading_dir"]
    xp = accessor.array_module()
    planar_vel = accessor.joint_vel()[:2]
    height = accessor.joint_pos()[2]
    direction = _xp_asarray(xp, heading_dir, dtype=getattr(xp, "float32", None))
    target_vel = target_speed * direction
    tracking = xp.exp(-xp.sum((target_vel - planar_vel) ** 2))
    fallen = height < fall_height
    dense = tracking - fall_penalty * fallen
    return {
        "heading_tracking": _maybe_item(tracking),
        "dense": _maybe_item(dense),
        "solved": False,
        "done": _maybe_item(fallen),
    }


def joint_penalty(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    weight: float = 50.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Penalty for approaching joint range-of-motion limits.

    Args:
        accessor: Environment state accessor.
        task_state: Unused; present for uniform call signature.
        weight: Scaling weight for the penalty.
        **kwargs: Unused extra keyword arguments.

    Returns:
        Dict with key ``joint_penalty`` (non-positive scalar).
    """
    xp = accessor.array_module()
    ctrl_range = accessor.ctrl_range()
    qpos = accessor.joint_pos()
    lo, hi = ctrl_range[:, 0], ctrl_range[:, 1]
    margin = 0.05 * (hi - lo)
    # xp.clip(x, 0.0, None) is used in preference to xp.maximum(0.0, x) because
    # torch.maximum requires both arguments to be Tensors, while clip/clamp accept
    # a scalar min bound universally across numpy, jax, and torch.
    violation = xp.sum(
        xp.clip(lo + margin - qpos, 0.0, None)
        + xp.clip(qpos - (hi - margin), 0.0, None)
    )
    penalty = -weight * violation
    return {"joint_penalty": penalty, "dense": penalty, "solved": False, "done": False}


def walk_env_reward(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    *,
    hip_flex_indices: tuple[int, int],
    hip_angle_indices: tuple[int, int, int, int],
    target_rot: Any,
    target_vel: tuple[float, float] = (0.0, 1.2),
    min_height: float = 0.8,
    max_rot: float = 0.8,
    com_height_index: int | None = None,
    com_vel_indices: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Backend-agnostic WalkEnvV0-style reward based on EnvAccessor data.

    This implementation assumes that ``task_state`` and the accessor together
    expose enough information to reconstruct the key WalkEnvV0 components:
    - COM velocity in the sagittal plane,
    - termination signal based on height and torso rotation,
    - cyclic hip trajectory tracking,
    - hip adduction/rotation regularisation.

    Args:
        accessor: Environment state accessor (CPU, MJX, or mjlab).
        task_state: Task configuration/state dictionary. Expected to contain:
            - ``"qpos"``: joint positions (same layout as WalkEnvV0.qpos),
            - ``"height"``: COM height scalar (if ``com_height_index`` is None),
            - ``"com_vel"``: COM velocity vector (if ``com_vel_indices`` is None).
        hip_indices: Tuple of 4 indices into ``qpos`` corresponding to
            (hip_adduction_l, hip_adduction_r, hip_rotation_l, hip_rotation_r).
        target_rot: Target torso quaternion (shape (4,)).
        target_vel: Tuple (target_x_vel, target_y_vel) in m/s.
        min_height: Minimum COM height before termination.
        max_rot: Maximum |(R @ e_x)[0]| before termination.
        com_height_index: Optional index into ``task_state["height_like"]`` if
            COM height is stored in a vector. If None, ``task_state["height"]``
            is treated as a scalar.
        com_vel_indices: Optional (ix, iy) indices into a COM-velocity-like
            array in ``task_state``. If None, ``task_state["com_vel"]`` is
            expected to be a length-2 vector.

    Returns:
        Dict with keys:
            - ``vel_reward``
            - ``done``
            - ``cyclic_hip``
            - ``ref_rot``
            - ``joint_angle_rew``
            - ``dense`` (weighted sum using WalkEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS)
    """
    xp = accessor.array_module()

    # Resolve COM velocity.
    if com_vel_indices is None:
        com_vel = task_state["com_vel"]
        vx, vy = com_vel[0], com_vel[1]
    else:
        com_vel_arr = task_state["com_vel"]
        vx, vy = com_vel_arr[com_vel_indices[0]], com_vel_arr[com_vel_indices[1]]

    target_x_vel, target_y_vel = target_vel
    vel_reward = xp.exp(-xp.square(target_y_vel - vy)) + xp.exp(
        -xp.square(target_x_vel - vx)
    )

    # Resolve COM height.
    if com_height_index is None:
        height = task_state["height"]
    else:
        height_vec = task_state["height_like"]
        height = height_vec[com_height_index]

    # Rotation condition using root quaternion from qpos[3:7].
    qpos = task_state["qpos"]
    quat = qpos[3:7]
    # (R @ [1, 0, 0])[0] for a unit quaternion.
    # We approximate via the standard formula: R[0,0] = 1 - 2*(qy^2 + qz^2).
    qy = quat[2]
    qz = quat[3]
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)

    done_height = height < min_height
    done_rot = xp.abs(r00) > max_rot
    done = xp.logical_or(done_height, done_rot)

    # Cyclic hip reward.
    phase = xp.asarray(task_state.get("phase_var", xp.array(0.0)))
    phase_scalar = phase[0] if phase.shape else phase
    des_l = _WALK_HIP_AMPLITUDE * xp.cos(phase_scalar * 2.0 * xp.pi + xp.pi)
    des_r = _WALK_HIP_AMPLITUDE * xp.cos(phase_scalar * 2.0 * xp.pi)

    hip_flex_l_idx, hip_flex_r_idx = hip_flex_indices
    hip_flex = xp.asarray(
        [qpos[hip_flex_l_idx], qpos[hip_flex_r_idx]], dtype=xp.asarray(qpos).dtype
    )
    des = xp.asarray([des_l, des_r])
    cyclic_hip = xp.linalg.norm(des - hip_flex)

    # Ref rotation reward.
    quat_np = xp.asarray(quat)
    target_rot_arr = xp.asarray(target_rot)
    ref_rot = xp.exp(-xp.linalg.norm(_WALK_REF_ROT_SCALE * (quat_np - target_rot_arr)))

    # Joint angle reward.
    hip_adduct_l, hip_adduct_r, hip_rot_l, hip_rot_r = hip_angle_indices
    hip_angles = xp.asarray(
        [qpos[hip_adduct_l], qpos[hip_adduct_r], qpos[hip_rot_l], qpos[hip_rot_r]]
    )
    joint_angle_rew = xp.exp(-_WALK_JOINT_ANGLE_SCALE * xp.mean(xp.abs(hip_angles)))

    dense = (
        _WALK_W_VEL * vel_reward
        + _WALK_W_FALL * done.astype(xp.float32)
        + _WALK_W_CYCLIC * cyclic_hip
        + _WALK_W_REF_ROT * ref_rot
        + _WALK_W_JOINT * joint_angle_rew
    )

    done_flag = done.astype(xp.float32)
    return {
        "vel_reward": vel_reward,
        "done": done_flag,
        "solved": xp.array(False),
        "cyclic_hip": cyclic_hip,
        "ref_rot": ref_rot,
        "joint_angle_rew": joint_angle_rew,
        "dense": dense,
    }
