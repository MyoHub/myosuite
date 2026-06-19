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

# ── saber_target_pool_reward ─────────────────────────────────────────────────
_SABER_HIT_DIST_CAP: float = 1e3  # non-hit distance cap (normalisation)
_SABER_HIT_PRECISION_CAP: float = 1e6  # hit offset precision cap
_SABER_OBSTACLE_CAP: float = 1e5  # obstacle distance normalisation cap


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


def saber_keyframe_pose_reward(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    saber_keyframe_pose_error_scale: float = 40.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Reward keeping both sabers near their reset/keyframe pose.

    Args:
        accessor: Unused environment accessor kept for the shared reward API.
        task_state: Shared task-state dict containing
            ``"saber_keyframe_pose_error"``.
        saber_keyframe_pose_error_scale: Exponential decay scale applied to the
            keyframe pose error.
        **kwargs: Unused extra reward configuration values.

    Returns:
        Dict containing the decayed ``saber_keyframe_pose`` reward, the raw
        pose error, and the matching ``dense`` contribution.
    """
    del kwargs
    xp = accessor.array_module()
    pose_error = _xp_asarray(
        xp,
        task_state.get("saber_keyframe_pose_error", 1.0),
        dtype=getattr(xp, "float32", None),
    )
    reward = xp.exp(-float(saber_keyframe_pose_error_scale) * pose_error)
    return {
        "saber_keyframe_pose": _maybe_item(reward),
        "saber_keyframe_pose_error": _maybe_item(pose_error),
        "dense": _maybe_item(reward),
        "solved": False,
        "done": False,
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


def boxing_bag_reward(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    fist_site_name: str = "fist_r",
    bag_head_site_name: str = "bag_head_zone",
    bag_body_site_name: str = "bag_body_zone",
    hit_threshold: float = 0.15,
    w_head: float = 2.0,
    w_body: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Proximity-based boxing bag hit detection reward.

    A hit is registered when the fist site is within ``hit_threshold`` metres
    of a bag scoring site. Head hits are weighted higher than body hits.

    Args:
        accessor: Environment state accessor.
        task_state: Unused; present for uniform call signature.
        fist_site_name: Name of the fist tracking site.
        bag_head_site_name: Name of the bag head scoring zone site.
        bag_body_site_name: Name of the bag body scoring zone site.
        hit_threshold: Distance threshold in metres for a hit to register.
        w_head: Reward weight for head zone hits.
        w_body: Reward weight for body zone hits.
        **kwargs: Unused extra keyword arguments.

    Returns:
        Dict with keys: ``head_hits``, ``body_hits``, ``r_head``, ``r_body``,
        ``dense``, ``solved`` (bool), ``done`` (bool).
    """
    xp = accessor.array_module()

    fist_id = accessor.site_id(fist_site_name)
    head_id = accessor.site_id(bag_head_site_name)
    body_id = accessor.site_id(bag_body_site_name)

    fist_pos = xp.asarray(accessor.site_xpos(fist_id))
    head_pos = xp.asarray(accessor.site_xpos(head_id))
    body_pos = xp.asarray(accessor.site_xpos(body_id))

    dist_head = float(xp.linalg.norm(head_pos - fist_pos))
    dist_body = float(xp.linalg.norm(body_pos - fist_pos))

    head_hits = 1.0 if dist_head < hit_threshold else 0.0
    body_hits = 1.0 if dist_body < hit_threshold else 0.0

    r_head = w_head * head_hits
    r_body = w_body * body_hits
    dense = r_head + r_body
    solved = (head_hits + body_hits) > 0

    return {
        "head_hits": head_hits,
        "body_hits": body_hits,
        "r_head": r_head,
        "r_body": r_body,
        "dense": dense,
        "solved": solved,
        "done": False,
    }


def saber_targets_reward(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    left_saber_site_name: str = "left_saber_tip",
    right_saber_site_name: str = "right_saber_tip",
    target_a_site_name: str = "saber_target_a_site",
    target_b_site_name: str = "saber_target_b_site",
    hit_threshold: float = 0.25,
    w_hit: float = 2.0,
    w_precision: float = 0.25,
    wrong_pairing_threshold: float = 0.3,
    wrong_pairing_penalty: float = 1.0,
    episode_done_y_below: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Dual-saber strike reward (Beat Saber / ``dual_beat_saber`` style).

    Correct pairings are **left saber → target *a*** and **right saber → target
    *b*** (geometry names in the MJCF). Dense reward combines proximity on
    correct pairs, discrete hit bonuses, and a penalty when the wrong saber
    crosses too close to the opposite block (discourages arm crossing).

    Args:
        accessor: Environment state accessor.
        task_state: Unused; present for uniform call signature.
        left_saber_site_name: Left-hand saber tip site.
        right_saber_site_name: Right-hand saber tip site.
        target_a_site_name: Strike block *a* tracking site.
        target_b_site_name: Strike block *b* tracking site.
        hit_threshold: Metres; tip-to-correct-target distance for a hit.
        w_hit: Weight on binary hit indicators (correct pair only).
        w_precision: Weight on negative sum of correct-pair distances.
        wrong_pairing_threshold: Metres; wrong-pair distance below this
            triggers ``wrong_pairing_penalty``.
        wrong_pairing_penalty: Subtracted once per step if any wrong pair is
            closer than ``wrong_pairing_threshold``.
        episode_done_y_below: If set, episode ends when **both** target site
            world-frame *y* coordinates are below this value (blocks passed the
            player on a ``-y`` approach). ``None`` disables this signal.
        **kwargs: Absorbs legacy keys (e.g. ``motion_amplitude``,
            ``drive_saber_targets_kinematics``) from task configs.

    Returns:
        Dict with hit counters, ``dense``, ``solved``, and ``done``.
    """
    del task_state  # unused
    _ = kwargs  # legacy / task-config keys ignored here

    xp = accessor.array_module()
    lid = accessor.site_id(left_saber_site_name)
    rid = accessor.site_id(right_saber_site_name)
    aid = accessor.site_id(target_a_site_name)
    bid = accessor.site_id(target_b_site_name)

    left = xp.asarray(accessor.site_xpos(lid), dtype=xp.float32)
    right = xp.asarray(accessor.site_xpos(rid), dtype=xp.float32)
    ta = xp.asarray(accessor.site_xpos(aid), dtype=xp.float32)
    tb = xp.asarray(accessor.site_xpos(bid), dtype=xp.float32)

    d_la = float(xp.linalg.norm(ta - left))
    d_rb = float(xp.linalg.norm(tb - right))
    d_lb = float(xp.linalg.norm(tb - left))
    d_ra = float(xp.linalg.norm(ta - right))

    left_hit = 1.0 if d_la < hit_threshold else 0.0
    right_hit = 1.0 if d_rb < hit_threshold else 0.0
    total_hits = left_hit + right_hit

    wrong_left = d_lb < wrong_pairing_threshold and d_la >= hit_threshold
    wrong_right = d_ra < wrong_pairing_threshold and d_rb >= hit_threshold
    wrong_close = wrong_left or wrong_right
    penalty = float(wrong_pairing_penalty) if wrong_close else 0.0

    dense = float(
        w_hit * (left_hit + right_hit) - w_precision * (d_la + d_rb) - penalty
    )

    solved = bool(left_hit > 0.5 and right_hit > 0.5)

    done = False
    if episode_done_y_below is not None:
        ya = float(xp.asarray(ta[1]))
        yb = float(xp.asarray(tb[1]))
        done = bool(ya < episode_done_y_below and yb < episode_done_y_below)

    return {
        "left_saber_hits": left_hit,
        "right_saber_hits": right_hit,
        "total_hits": total_hits,
        "dense": dense,
        "solved": solved,
        "done": done,
    }


def saber_target_pool_reward(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    w_hit: float = 2.0,
    w_correct_hand: float = 4.0,
    w_correct_face: float = 3.0,
    max_target_misses: int | None = None,
    use_health_status: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Reward the saber target pool with success, slicing, and timing terms.

    Uses ``task_state`` populated by :class:`~myosuite.envs.modular_env.ModularTaskEnv`
    when ``enable_saber_target_pool`` is set.

    Args:
        accessor: Environment state accessor.
        task_state: Must contain pool step flags set by the env.
        w_hit: Success-score weight for any target hit.
        w_correct_hand: Success-score weight for using the requested hand.
        w_correct_face: Success-score weight for striking the requested face.
        max_target_misses: Episode terminates once cumulative misses reach this
            count. ``None`` disables miss-count termination.
        **kwargs: Ignored compatibility keys from task config.

    Returns:
        Reward dict with ``dense``, ``solved``, ``done``, and logging keys.
    """
    del kwargs
    xp = accessor.array_module()
    float_dtype = getattr(xp, "float32", None)
    bool_dtype = getattr(xp, "bool", bool)
    hit = _xp_asarray(
        xp,
        task_state.get("target_pool_hit_this_step", False),
        dtype=float_dtype,
    )
    correct_hand = _xp_asarray(
        xp,
        task_state.get("target_pool_correct_hand_this_step", False),
        dtype=float_dtype,
    )
    correct_face = _xp_asarray(
        xp,
        task_state.get("target_pool_correct_face_this_step", False),
        dtype=float_dtype,
    )
    wrong_hand = _xp_asarray(
        xp,
        task_state.get("target_pool_wrong_hand_this_step", False),
        dtype=float_dtype,
    )
    wrong_face = _xp_asarray(
        xp,
        task_state.get("target_pool_wrong_face_this_step", False),
        dtype=float_dtype,
    )
    hit_mask = hit > 0.5
    zeros = xp.zeros_like(hit, dtype=float_dtype)
    hit_y = xp.where(
        hit_mask,
        _xp_asarray(
            xp,
            task_state.get("target_pool_hit_y_at_hit", 0.0),
            dtype=float_dtype,
        ),
        zeros,
    )
    hit_y_deviation = xp.where(
        hit_mask,
        _xp_asarray(
            xp,
            task_state.get("target_pool_hit_y_deviation_at_hit", 0.0),
            dtype=float_dtype,
        ),
        zeros,
    )
    face_surface_distance = xp.where(
        hit_mask,
        _xp_asarray(
            xp,
            task_state.get("target_pool_face_surface_distance_at_hit", 0.0),
            dtype=float_dtype,
        ),
        zeros,
    )
    slicing_accuracy = xp.where(
        hit_mask,
        _xp_asarray(
            xp,
            task_state.get("target_pool_slicing_accuracy_this_step", 0.0),
            dtype=float_dtype,
        ),
        zeros,
    )
    timing_accuracy = xp.where(
        hit_mask,
        _xp_asarray(
            xp,
            task_state.get("target_pool_timing_accuracy_this_step", 0.0),
            dtype=float_dtype,
        ),
        zeros,
    )
    success_score = (
        float(w_hit) * hit
        + float(w_correct_hand) * correct_hand
        + float(w_correct_face) * correct_face
    )
    bad_cut = _xp_asarray(
        xp,
        (wrong_hand > 0.5) | (wrong_face > 0.5),
        dtype=float_dtype,
    )
    obstacle_contact = _xp_asarray(
        xp,
        task_state.get("target_pool_obstacle_contact_this_step", False),
        dtype=float_dtype,
    )
    obstacle_d_min = _xp_asarray(
        xp,
        task_state.get("target_pool_obstacle_d_min", 1e6),
        dtype=float_dtype,
    )
    dense = success_score + slicing_accuracy + timing_accuracy

    hits_total = _xp_asarray(
        xp,
        task_state.get("target_pool_hits", 0),
        dtype=float_dtype,
    )
    misses_total = _xp_asarray(
        xp,
        task_state.get("target_pool_misses", 0),
        dtype=float_dtype,
    )
    correct_hand_total = _xp_asarray(
        xp,
        task_state.get("target_pool_correct_hand_hits", 0),
        dtype=float_dtype,
    )
    correct_face_total = _xp_asarray(
        xp,
        task_state.get("target_pool_correct_face_hits", 0),
        dtype=float_dtype,
    )
    wrong_hand_total = _xp_asarray(
        xp,
        task_state.get("target_pool_wrong_hand_hits", 0),
        dtype=float_dtype,
    )
    wrong_face_total = _xp_asarray(
        xp,
        task_state.get("target_pool_wrong_face_hits", 0),
        dtype=float_dtype,
    )
    spawned_total = _xp_asarray(
        xp,
        task_state.get("target_pool_total_spawned", 0),
        dtype=float_dtype,
    )
    obstacle_contacts_total = _xp_asarray(
        xp,
        task_state.get("target_pool_obstacle_contacts", 0),
        dtype=float_dtype,
    )
    health_status = _xp_asarray(
        xp,
        task_state.get("health_status", 0.5),
        dtype=float_dtype,
    )
    if use_health_status:
        health_depleted = _xp_asarray(
            xp,
            task_state.get("target_pool_health_depleted_this_step", False),
            dtype=bool_dtype,
        ) | (health_status <= 0.0)
    else:
        health_depleted = xp.zeros_like(hit, dtype=bool_dtype)
    if max_target_misses is not None:
        miss_limit_reached = misses_total >= float(max_target_misses)
    else:
        miss_limit_reached = xp.zeros_like(hit, dtype=bool_dtype)
    done = miss_limit_reached | health_depleted

    return {
        "hit_this_step": _maybe_item(hit),
        "success_score": _maybe_item(success_score),
        "hit_y_at_hit": _maybe_item(hit_y),
        "hit_y_deviation_at_hit": _maybe_item(hit_y_deviation),
        "face_surface_distance_at_hit": _maybe_item(face_surface_distance),
        "slicing_accuracy": _maybe_item(slicing_accuracy),
        "timing_accuracy": _maybe_item(timing_accuracy),
        "correct_hand_this_step": _maybe_item(correct_hand),
        "correct_face_this_step": _maybe_item(correct_face),
        "wrong_hand_this_step": _maybe_item(wrong_hand),
        "wrong_face_this_step": _maybe_item(wrong_face),
        "bad_cut_this_step": _maybe_item(bad_cut),
        "hits_total": _maybe_item(hits_total),
        "misses_total": _maybe_item(misses_total),
        "correct_hand_total": _maybe_item(correct_hand_total),
        "correct_face_total": _maybe_item(correct_face_total),
        "wrong_hand_total": _maybe_item(wrong_hand_total),
        "wrong_face_total": _maybe_item(wrong_face_total),
        "total_spawned": _maybe_item(spawned_total),
        "health_status": _maybe_item(health_status),
        "health_depleted": _maybe_item(health_depleted),
        "miss_limit_reached": _maybe_item(miss_limit_reached),
        "obstacle_contact_this_step": _maybe_item(obstacle_contact),
        "obstacle_contacts_total": _maybe_item(obstacle_contacts_total),
        "obstacle_d_min": _maybe_item(obstacle_d_min),
        "dense": _maybe_item(dense),
        "solved": _maybe_item(
            hit_mask.any() if hasattr(hit_mask, "any") else bool(hit_mask)
        ),
        "done": _maybe_item(done),
    }
