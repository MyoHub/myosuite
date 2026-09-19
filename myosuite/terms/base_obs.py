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


def root_planar_vel_obs(accessor: EnvAccessor, **kwargs: Any) -> Any:
    """Return the root free-joint's planar (x, y) linear velocity.

    Assumes the model's first joint is a ``freejoint`` so ``joint_vel()[:2]``
    is the world-frame horizontal velocity of the root body (pelvis).

    Args:
        accessor: Environment state accessor.
        **kwargs: Unused; for uniform call signature.

    Returns:
        Length-2 array ``[vx, vy]``.
    """
    return accessor.joint_vel()[:2]


def heading_cmd_obs(
    accessor: EnvAccessor,
    heading_dir: tuple[float, float] = (0.0, 1.0),
    **kwargs: Any,
) -> Any:
    """Return the commanded planar heading direction as an observation.

    Args:
        accessor: Environment state accessor.
        heading_dir: Commanded unit direction ``(dx, dy)``.
        **kwargs: Unused; for uniform call signature.

    Returns:
        Length-2 array equal to ``heading_dir``.
    """
    xp = accessor.array_module()
    return xp.asarray(heading_dir, dtype=xp.float32)
