# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""
Observation term functions for MyoSuite environments.

All functions receive an EnvAccessor and return the native array type
for the current backend (numpy / jax.Array / torch.Tensor).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import mujoco

    from myosuite.core.protocols import EnvAccessor


def qpos_for_joints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_ids: list[int],
) -> np.ndarray:
    """Concatenate qpos slices for a list of joint ids.

    Args:
        model: Compiled MuJoCo model.
        data: Current MuJoCo simulation data.
        joint_ids: Joint ids to extract from ``qpos``.

    Returns:
        Concatenated joint position values, or an empty array when ``joint_ids``
        is empty.
    """
    import mujoco  # noqa: PLC0415

    parts: list[np.ndarray] = []
    for joint_id in joint_ids:
        address = int(model.jnt_qposadr[joint_id])
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            width = 7
        elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            width = 4
        else:
            width = 1
        parts.append(data.qpos[address : address + width])
    return np.concatenate(parts) if parts else np.empty(0)


def qvel_for_joints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_ids: list[int],
) -> np.ndarray:
    """Concatenate qvel slices for a list of joint ids.

    Args:
        model: Compiled MuJoCo model.
        data: Current MuJoCo simulation data.
        joint_ids: Joint ids to extract from ``qvel``.

    Returns:
        Concatenated joint velocity values, or an empty array when ``joint_ids``
        is empty.
    """
    import mujoco  # noqa: PLC0415

    parts: list[np.ndarray] = []
    for joint_id in joint_ids:
        address = int(model.jnt_dofadr[joint_id])
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            width = 6
        elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            width = 3
        else:
            width = 1
        parts.append(data.qvel[address : address + width])
    return np.concatenate(parts) if parts else np.empty(0)


def sensor_vel3(data: mujoco.MjData, sensor_adr: int) -> np.ndarray:
    """Extract a 3D linear velocity from ``data.sensordata``.

    Args:
        data: Current MuJoCo simulation data.
        sensor_adr: Sensor start index in ``sensordata``.

    Returns:
        A copied ``(3,)`` linear-velocity vector.
    """
    return data.sensordata[sensor_adr : sensor_adr + 3].copy()


def normalized_health_pair(
    health: dict[str, float], agent_id: str, health_scale: float
) -> np.ndarray:
    """Return own and opponent normalized health values.

    Args:
        health: Health dictionary keyed by agent id.
        agent_id: Observing agent id.
        health_scale: Positive denominator used to normalize health.

    Returns:
        Float32 array ``[own_health_norm, opponent_health_norm]``.
    """
    opponent_id = "agent_1" if agent_id == "agent_0" else "agent_0"
    return np.array(
        [health[agent_id] / health_scale, health[opponent_id] / health_scale],
        dtype=np.float32,
    )


def joint_pos_obs(accessor: EnvAccessor, **kwargs: Any) -> Any:
    """Return joint positions as the observation.

    Args:
        accessor: Environment state accessor.
        **kwargs: Unused; for uniform call signature.

    Returns:
        Joint position array, shape (nq,) or (N, nq).
    """
    return accessor.joint_pos()


def joint_vel_obs(accessor: EnvAccessor, **kwargs: Any) -> Any:
    """Return joint velocities as the observation.

    Args:
        accessor: Environment state accessor.
        **kwargs: Unused; for uniform call signature.

    Returns:
        Joint velocity array, shape (nv,) or (N, nv).
    """
    return accessor.joint_vel()


def muscle_act_obs(accessor: EnvAccessor, **kwargs: Any) -> Any:
    """Return muscle activations as the observation.

    Args:
        accessor: Environment state accessor.
        **kwargs: Unused; for uniform call signature.

    Returns:
        Muscle activation array, shape (na,) or (N, na).
    """
    return accessor.muscle_act()


def tip_pos_obs(accessor: EnvAccessor, site_ids: Any, **kwargs: Any) -> Any:
    """Return fingertip / end-effector Cartesian positions.

    Args:
        accessor: Environment state accessor.
        site_ids: Site indices or names to query.
        **kwargs: Unused; for uniform call signature.

    Returns:
        Site position array, shape (len(site_ids), 3) or (N, len(site_ids), 3).
    """
    return accessor.site_xpos(site_ids)


def pose_error_obs(
    accessor: EnvAccessor,
    target: Any = None,
    target_angles: Any = None,
    **kwargs: Any,
) -> Any:
    """Return the signed difference between target and current joint positions.

    Accepts either ``target`` or ``target_angles`` as the goal array so that
    task states produced by goal-sampling (which uses ``"target_angles"`` as
    the key) work without an explicit alias mapping.

    Args:
        accessor: Environment state accessor.
        target: Target joint angle array, same shape as ``joint_pos()``.
            Mutually exclusive with ``target_angles``; ``target`` takes
            precedence when both are provided.
        target_angles: Alias for ``target``.  Used when the task state dict
            contains ``"target_angles"`` (the default key from
            :func:`~myosuite.envs.modular_env._sample_goal`).
        **kwargs: Unused; for uniform call signature.

    Returns:
        Pose error array (target - current), same shape as ``joint_pos()``.

    Raises:
        ValueError: If neither ``target`` nor ``target_angles`` is provided.
    """
    t = target if target is not None else target_angles
    if t is None:
        raise ValueError(
            "pose_error_obs requires 'target' or 'target_angles' in kwargs"
        )
    return t - accessor.joint_pos()


def time_obs(accessor: EnvAccessor, **kwargs: Any) -> Any:
    """Return current simulation time.

    Args:
        accessor: Environment state accessor.
        **kwargs: Unused; for uniform call signature.

    Returns:
        Scalar simulation time in seconds.
    """
    return accessor.time()


def saber_sites_obs(
    accessor: EnvAccessor,
    site_names: list[str],
    include_target_approach_vectors: bool = False,
    **kwargs: Any,
) -> Any:
    """Concatenate saber tip and strike-block site positions (Beat Saber).

    When ``include_target_approach_vectors`` is ``True``, appends vectors from
    each saber tip toward its paired target (same pairing as
    ``saber_targets_reward`` in ``myo_reward_terms``): left tip → target *a*,
    right tip → target *b*, i.e. an **18**-vector like the ``dual_beat_saber``
    prototype (four positions plus two relative vectors).

    Args:
        accessor: Environment state accessor.
        site_names: Four site names in order
            ``[left_saber, right_saber, target_a, target_b]``.
        include_target_approach_vectors: If ``True``, append
            ``(target_a - left)`` and ``(target_b - right)`` (each length 3).
        **kwargs: Ignored; present for a uniform call signature.

    Returns:
        1-D array of length ``12`` or ``18``.

    Raises:
        ValueError: If ``site_names`` does not contain exactly four names.
    """
    xp = accessor.array_module()
    if len(site_names) != 4:
        raise ValueError("saber_sites_obs expects exactly four site_names")
    left_n, right_n, ta_n, tb_n = site_names
    left = xp.asarray(accessor.site_xpos(accessor.site_id(left_n)), dtype=xp.float32)
    right = xp.asarray(accessor.site_xpos(accessor.site_id(right_n)), dtype=xp.float32)
    ta = xp.asarray(accessor.site_xpos(accessor.site_id(ta_n)), dtype=xp.float32)
    tb = xp.asarray(accessor.site_xpos(accessor.site_id(tb_n)), dtype=xp.float32)
    parts = [left.ravel(), right.ravel(), ta.ravel(), tb.ravel()]
    if include_target_approach_vectors:
        parts.append((ta - left).ravel())
        parts.append((tb - right).ravel())
    return xp.concatenate(parts)


def saber_target_pool_obs(accessor: EnvAccessor, **kwargs: Any) -> Any:
    """Top-K proximity-sorted target slots for the 100-sphere saber pool.

    Each slot is 8 values:
    relative position (3) from the nearer saber tip, velocity (3),
    preferred hit area (1), preferred hand (1). Remaining slots are zero-padded.

    Args:
        accessor: Environment state accessor.
        **kwargs: ``pool_size``, ``num_visible``, ``site_prefix``, ``site_suffix``,
            ``graveyard_z``, ``left_saber_site_name``, ``right_saber_site_name``,
            and task-state keys ``target_pool_active``, ``target_pool_vel``,
            ``target_pool_hit_area``, ``target_pool_preferred_hand`` (optional),
            and ``target_pool_pos`` (optional batched tensor of shape
            ``(N, pool_size, 3)`` or ``(pool_size, 3)``; skips per-site
            ``accessor.site_xpos`` lookups when provided).

    Returns:
        Array of shape ``(N, 8 * num_visible)`` when inputs are batched
        (torch, leading N dimension) or 1-D array of length ``8 * num_visible``
        for single-env (numpy) use.
    """
    xp = accessor.array_module()
    pool_size = int(kwargs.get("pool_size", 100))
    num_visible = int(kwargs.get("num_visible", 10))
    left_n = str(kwargs.get("left_saber_site_name", "left_saber_tip"))
    right_n = str(kwargs.get("right_saber_site_name", "right_saber_tip"))

    active = kwargs.get("target_pool_active")
    vel_tb = kwargs.get("target_pool_vel")
    hit_area_tb = kwargs.get("target_pool_hit_area")
    preferred_hand_tb = kwargs.get("target_pool_preferred_hand")
    target_pos_pre = kwargs.get("target_pool_pos")

    # --- batched torch fast-path (mjlab) ---
    # Used when target positions are passed in directly (avoids per-site MuJoCo lookups).
    if getattr(xp, "__name__", "") == "torch" and target_pos_pre is not None:
        import torch

        target_pos = torch.as_tensor(target_pos_pre, dtype=torch.float32)
        # support both (pool_size, 3) and (N, pool_size, 3)
        if target_pos.ndim == 2:
            target_pos = target_pos.unsqueeze(0)
        n = target_pos.shape[0]
        device = target_pos.device

        left = torch.as_tensor(
            accessor.site_xpos(accessor.site_id(left_n)), dtype=torch.float32
        ).to(device)
        right = torch.as_tensor(
            accessor.site_xpos(accessor.site_id(right_n)), dtype=torch.float32
        ).to(device)
        # Normalise tip tensors to (N, 1, 3) regardless of whether site_xpos
        # returned (3,), (N, 3), or (N, 1, 3).
        if left.ndim == 1:  # (3,) — unbatched scalar lookup
            left = left.view(1, 1, 3)
            right = right.view(1, 1, 3)
        elif left.ndim == 2:  # (N, 3) — batched scalar lookup
            left = left[:, None, :]
            right = right[:, None, :]
        # ndim == 3 → already (N, 1, 3) or similar; use as-is

        if active is None:
            act = torch.zeros(n, pool_size, dtype=torch.bool, device=device)
        else:
            act = torch.as_tensor(active, dtype=torch.bool, device=device)
            if act.ndim == 1:
                act = act.unsqueeze(0).expand(n, -1)

        vel = (
            torch.zeros(n, pool_size, 3, dtype=torch.float32, device=device)
            if vel_tb is None
            else torch.as_tensor(vel_tb, dtype=torch.float32, device=device)
        )
        if vel.ndim == 2:
            vel = vel.unsqueeze(0).expand(n, -1, -1)

        hit = (
            torch.zeros(n, pool_size, dtype=torch.float32, device=device)
            if hit_area_tb is None
            else torch.as_tensor(hit_area_tb, dtype=torch.float32, device=device)
        )
        if hit.ndim == 1:
            hit = hit.unsqueeze(0).expand(n, -1)

        hand = (
            torch.zeros(n, pool_size, dtype=torch.float32, device=device)
            if preferred_hand_tb is None
            else torch.as_tensor(preferred_hand_tb, dtype=torch.float32, device=device)
        )
        if hand.ndim == 1:
            hand = hand.unsqueeze(0).expand(n, -1)

        d_l = torch.linalg.norm(target_pos - left, dim=-1)
        d_r = torch.linalg.norm(target_pos - right, dim=-1)
        nearest_left = d_l <= d_r
        d_min = torch.where(act, torch.minimum(d_l, d_r), torch.full_like(d_l, 1e9))

        topk = min(num_visible, pool_size)
        _, order = torch.topk(d_min, k=topk, dim=1, largest=False)
        bi = torch.arange(n, device=device).unsqueeze(1)
        c_act = act[bi, order]
        c_left = nearest_left[bi, order]
        c_pos = target_pos[bi, order]
        c_vel = vel[bi, order]
        c_hit = hit[bi, order]
        c_hand = hand[bi, order]

        ref = torch.where(
            c_left.unsqueeze(-1),
            left.expand(n, topk, 3),
            right.expand(n, topk, 3),
        )
        rel = c_pos - ref
        out = torch.zeros(n, topk, 8, dtype=torch.float32, device=device)
        mask = c_act.unsqueeze(-1)
        out[..., 0:3] = torch.where(mask, rel, torch.zeros_like(rel))
        out[..., 3:6] = torch.where(mask, c_vel, torch.zeros_like(c_vel))
        out[..., 6] = torch.where(c_act, c_hit, torch.zeros_like(c_hit))
        out[..., 7] = torch.where(c_act, c_hand, torch.zeros_like(c_hand))
        return out.reshape(n, -1)

    # --- single-env path (numpy / CPU) ---
    site_prefix = str(kwargs.get("site_prefix", "saber_target_"))
    site_suffix = str(kwargs.get("site_suffix", "_site"))

    left = xp.asarray(accessor.site_xpos(accessor.site_id(left_n)), dtype=xp.float32)
    right = xp.asarray(accessor.site_xpos(accessor.site_id(right_n)), dtype=xp.float32)

    if active is None:
        active = xp.zeros(pool_size, dtype=bool)
    else:
        active = xp.asarray(active, dtype=bool)
    if vel_tb is None:
        vel_tb = xp.zeros((pool_size, 3), dtype=xp.float32)
    else:
        vel_tb = xp.asarray(vel_tb, dtype=xp.float32)
    if hit_area_tb is None:
        hit_area_tb = xp.zeros(pool_size, dtype=xp.float32)
    else:
        hit_area_tb = xp.asarray(hit_area_tb, dtype=xp.float32)
    if preferred_hand_tb is None:
        preferred_hand_tb = xp.zeros(pool_size, dtype=xp.float32)
    else:
        preferred_hand_tb = xp.asarray(preferred_hand_tb, dtype=xp.float32)

    active_indices: list[int] = []
    active_positions: list[Any] = []
    for i in range(pool_size):
        if not bool(active[i]):
            continue
        sname = f"{site_prefix}{i}{site_suffix}"
        active_positions.append(
            xp.asarray(accessor.site_xpos(accessor.site_id(sname)), dtype=xp.float32)
        )
        active_indices.append(i)

    out = xp.zeros(num_visible * 8, dtype=xp.float32)
    if not active_indices:
        return out

    positions = xp.stack(active_positions, axis=0)
    d_l = xp.linalg.norm(positions - left, axis=-1)
    d_r = xp.linalg.norm(positions - right, axis=-1)
    d_min = xp.minimum(d_l, d_r)

    order = xp.argsort(d_min)[:num_visible]
    for slot, oi in enumerate(order):
        i = active_indices[int(oi)]
        tp = active_positions[int(oi)]
        rel = tp - left if float(d_l[oi]) <= float(d_r[oi]) else tp - right
        row = slot * 8
        out[row : row + 3] = rel.ravel()
        out[row + 3 : row + 6] = vel_tb[i]
        out[row + 6] = hit_area_tb[i]
        out[row + 7] = preferred_hand_tb[i]

    return out


def health_status_obs(accessor: EnvAccessor, **kwargs: Any) -> Any:
    """Return the current normalized health status as a scalar observation."""

    xp = accessor.array_module()
    health_status = kwargs.get("health_status", 0.5)
    if getattr(xp, "__name__", "") == "torch":
        value = xp.as_tensor(health_status, dtype=xp.float32)
        if value.ndim == 0:
            return value.unsqueeze(0)
        if value.ndim == 1:
            return value.unsqueeze(-1)
        return value
    value = xp.asarray(health_status, dtype=xp.float32)
    if value.ndim == 0:
        return xp.asarray([value], dtype=xp.float32)
    if value.ndim == 1:
        return value.reshape(-1, 1)
    return value
