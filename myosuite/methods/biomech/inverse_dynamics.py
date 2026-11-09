# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Joint kinematics and inverse dynamics of motion clips.

The steps after inverse kinematics in a musculoskeletal analysis (OpenSim's ID tool):
:func:`clip_kinematics` smooths a clip's generalized coordinates and differentiates
them on the configuration manifold, and :func:`inverse_dynamics` returns the
generalized forces the model needs to follow them::

    from myosuite.core.trajectory_io import expand_motion_clip_to_model
    from myosuite.methods.biomech.inverse_dynamics import clip_kinematics, inverse_dynamics

    qpos = expand_motion_clip_to_model(clip, model).qpos
    kin = clip_kinematics(model, qpos, 1 / clip.frequency_hz, lowpass_hz=6.0)
    tau = inverse_dynamics(model, kin.qpos, kin.qvel, kin.qacc)

On a floating-base model the free-joint rows of ``tau`` are the residual wrench
that external loads (ground reactions) must supply; pass known loads, such as
force-plate wrenches mapped with ``mujoco.mj_applyFT``, as ``external_qfrc``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy import signal

from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings


@dataclass(frozen=True)
class ClipKinematics:
    """Generalized positions, velocities and accelerations of a clip.

    Attributes:
        qpos: ``(T, nq)`` positions (smoothed, couplings projected).
        qvel: ``(T, nv)`` velocities, in MuJoCo's convention (free-joint angular
            velocity in the body frame).
        qacc: ``(T, nv)`` accelerations, the time derivative of ``qvel``.
        dt: Frame period [s].
    """

    qpos: np.ndarray
    qvel: np.ndarray
    qacc: np.ndarray
    dt: float


def _quaternion_addresses(model: mujoco.MjModel) -> list[int]:
    """First qpos address of every free- and ball-joint quaternion."""
    free, ball = mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL
    return [
        int(model.jnt_qposadr[j]) + (3 if model.jnt_type[j] == free else 0)
        for j in range(model.njnt)
        if model.jnt_type[j] in (free, ball)
    ]


def clip_kinematics(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    dt: float,
    *,
    lowpass_hz: float | None = None,
    filter_order: int = 2,
    project_couplings: bool = True,
) -> ClipKinematics:
    """Smooth a clip's coordinates and differentiate them twice.

    Velocities and accelerations are central differences of
    ``mj_differentiatePos`` between neighbouring frames, so free and ball joints
    are differentiated on the rotation manifold; both are second-order accurate,
    including the first and last frames (linear extrapolation).

    Args:
        model: Compiled model the clip is expressed in.
        qpos: ``(T, nq)`` positions, ``T >= 4``, e.g. from
            ``expand_motion_clip_to_model(clip, model).qpos``.
        dt: Frame period [s].
        lowpass_hz: Cut-off of a zero-phase (forward-backward) Butterworth filter
            applied to every coordinate before differentiation; ``None`` (as in
            OpenSim's ID tool) skips filtering. Quaternions are filtered
            component-wise after removing sign flips, then renormalised. The
            clip is padded by odd reflection over two cut-off periods; within
            about ``1 / lowpass_hz`` of either end the accelerations are less
            reliable, so trim those frames from the analysis.
        filter_order: Order of each pass; the default 2 is Winter's
            fourth-order zero-lag filter.
        project_couplings: Put the dependent joints of active joint equalities
            on their polynomials (``mjEQ_JOINT``) after filtering, so the
            kinematics satisfy the model's couplings.

    Returns:
        The positions, velocities and accelerations of every frame.

    Raises:
        ValueError: If the shapes, ``dt`` or the cut-off are invalid.
    """
    qpos = np.array(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or len(qpos) < 4:
        raise ValueError(f"qpos must be (T >= 4, nq={model.nq}), got {qpos.shape}")
    if not np.isfinite(qpos).all():
        raise ValueError("qpos must be finite")
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError(f"dt must be finite and positive, got {dt}")

    quats = _quaternion_addresses(model)
    for a in quats:
        q = qpos[:, a : a + 4]
        flips = np.sign(np.einsum("ij,ij->i", q[1:], q[:-1]))
        flips[flips == 0] = 1.0
        q[1:] *= np.cumprod(flips)[:, None]

    if lowpass_hz is not None:
        if not 0 < lowpass_hz < 0.5 / dt:
            raise ValueError(
                f"lowpass_hz must lie in (0, {0.5 / dt:g}) Hz (Nyquist), got {lowpass_hz}"
            )
        sos = signal.butter(filter_order, lowpass_hz, fs=1.0 / dt, output="sos")
        padlen = min(len(qpos) - 1, int(np.ceil(2.0 / (lowpass_hz * dt))))
        qpos = signal.sosfiltfilt(sos, qpos, axis=0, padlen=padlen)
    for a in quats:
        qpos[:, a : a + 4] /= np.linalg.norm(qpos[:, a : a + 4], axis=1, keepdims=True)

    if project_couplings:
        project_joint_couplings(JointCouplings.from_model(model), qpos)

    # Velocity over each interval, at the interval midpoints.
    step = np.empty((len(qpos) - 1, model.nv))
    for t in range(len(qpos) - 1):
        mujoco.mj_differentiatePos(model, step[t], dt, qpos[t], qpos[t + 1])
    qvel = np.empty((len(qpos), model.nv))
    qvel[1:-1] = 0.5 * (step[:-1] + step[1:])
    qvel[0] = 1.5 * step[0] - 0.5 * step[1]
    qvel[-1] = 1.5 * step[-1] - 0.5 * step[-2]
    qacc = np.empty_like(qvel)
    qacc[1:-1] = (step[1:] - step[:-1]) / dt
    qacc[0] = 2.0 * qacc[1] - qacc[2]
    qacc[-1] = 2.0 * qacc[-2] - qacc[-3]
    return ClipKinematics(qpos=qpos, qvel=qvel, qacc=qacc, dt=float(dt))


def required_qfrc(
    model: mujoco.MjModel, data: mujoco.MjData, qacc: np.ndarray
) -> np.ndarray:
    """``M qacc + qfrc_bias - qfrc_passive`` at the state already in *data*.

    Args:
        model: Compiled model.
        data: Data after ``mj_fwdPosition`` and ``mj_fwdVelocity`` at the state.
        qacc: ``(nv,)`` accelerations.

    Returns:
        ``(nv,)`` generalized force actuators and constraints must apply.
    """
    tau = np.empty(model.nv)
    mujoco.mj_mulM(model, data, tau, np.ascontiguousarray(qacc, dtype=np.float64))
    return tau + data.qfrc_bias - data.qfrc_passive


def inverse_dynamics(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    qacc: np.ndarray,
    *,
    external_qfrc: np.ndarray | None = None,
) -> np.ndarray:
    """Generalized forces that make the model follow a motion.

    See :func:`required_qfrc` for one state.

    Per frame, ``tau = M(q) qacc + qfrc_bias(q, qvel) - qfrc_passive(q, qvel) -
    external_qfrc``: what actuators and constraints together must apply. This is
    ``mj_inverse`` with constraints disabled; joint limits, contacts and
    equality constraints are left to the caller (e.g. static optimization).
    ``qfrc_passive`` holds the model's joint and tendon springs and dampers;
    armature is part of ``M``. Muscle passive forces are actuator forces and are
    not included.

    Args:
        model: Compiled model.
        qpos: ``(T, nq)`` positions.
        qvel: ``(T, nv)`` velocities.
        qacc: ``(T, nv)`` accelerations.
        external_qfrc: Optional ``(T, nv)`` known external generalized forces
            (e.g. force plates via ``mujoco.mj_applyFT``).

    Returns:
        ``(T, nv)`` required generalized forces [N or N m per dof].

    Raises:
        ValueError: If the array shapes do not match the model.
    """
    qpos, qvel, qacc = (np.asarray(x, dtype=np.float64) for x in (qpos, qvel, qacc))
    frames = len(qpos)
    for name, x, width in (
        ("qpos", qpos, model.nq),
        ("qvel", qvel, model.nv),
        ("qacc", qacc, model.nv),
    ):
        if x.shape != (frames, width):
            raise ValueError(f"{name} must be ({frames}, {width}), got {x.shape}")
    data = mujoco.MjData(model)
    tau = np.empty((frames, model.nv))
    for t in range(frames):
        data.qpos[:], data.qvel[:] = qpos[t], qvel[t]
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_fwdVelocity(model, data)
        tau[t] = required_qfrc(model, data, qacc[t])
    if external_qfrc is not None:
        external_qfrc = np.asarray(external_qfrc, dtype=np.float64)
        if external_qfrc.shape != tau.shape:
            raise ValueError(
                f"external_qfrc must be {tau.shape}, got {external_qfrc.shape}"
            )
        tau -= external_qfrc
    return tau
