# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# myosuite/envs/myo/tasks/mimic/tracking/ for the notices of the code it includes.
"""Motion-tracking terms of ``myoMimic-v0`` as pure, backend-agnostic functions.

Every function takes the array module ``xp`` (numpy or torch) and a :class:`TrackState`
(the body state) and/or a :class:`TrackRef` (the reference at one frame), batched over
leading axes. The CPU env fills the state from ``MjData``, the mjlab twin from
``entity.data``, and a learning method from its own model of the body, so all of them
score one MDP. In torch the functions stay differentiable: no in-place ops, no
``.item()``, no branching on data.

Quantities follow MuscleMimic (github.com/amathislab/musclemimic) as ported by
muscle-supermimic (``tasks/tracking/mdp``, commit 8a6244b): the 16 relative sites are the
tracked bodies minus the first one (the pelvis), the reward is MimicReward
(``exp(-kernel * mean squared error)`` per component) and the failure rule is
MuscleMimic's mean relative site / root position / root orientation deviation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


@dataclass
class TrackState:
    """Body state of a tracking agent (arrays with leading batch axes ``...``).

    Attributes:
        root_pos: ``(..., 3)`` free-joint position (env frame).
        root_quat: ``(..., 4)`` free-joint orientation (wxyz).
        root_qvel: ``(..., 6)`` free-joint velocity ``[v world, w body]``.
        joint_pos: ``(..., J)`` every non-root joint, model order.
        joint_vel: ``(..., J)``.
        body_pos: ``(..., B, 3)`` tracked body origins (env frame), pelvis first.
        body_rot: ``(..., B, 3, 3)`` tracked body rotations.
        body_lin_vel: ``(..., B, 3)`` linear velocity of the body origins (world).
        body_ang_vel: ``(..., B, 3)`` angular velocity (world).
        act: ``(..., M)`` muscle activations.
        excitation: ``(..., M)`` the last action (the commanded excitation).
        touch: ``(..., 4)`` normal ground force on the heels and toes (N).
        muscle_length: ``(..., M)`` optional.
        muscle_velocity: ``(..., M)`` optional.
        muscle_force: ``(..., M)`` optional.
    """

    root_pos: Any
    root_quat: Any
    root_qvel: Any
    joint_pos: Any
    joint_vel: Any
    body_pos: Any
    body_rot: Any
    body_lin_vel: Any
    body_ang_vel: Any
    act: Any
    excitation: Any
    touch: Any
    muscle_length: Any = None
    muscle_velocity: Any = None
    muscle_force: Any = None


@dataclass
class TrackRef:
    """Reference at one frame per batch entry, in the conventions of :class:`TrackState`.

    Attributes:
        root_pos: ``(..., 3)``.
        root_rot: ``(..., 3, 3)`` free-joint body rotation.
        root_qvel: ``(..., 6)`` ``[v world, w body]``.
        root_vel_local: ``(..., 6)`` ``[R^T v, w body]``.
        joint_pos: ``(..., J)``.
        joint_vel: ``(..., J)``.
        site_rpos: ``(..., B-1, 3)`` see :func:`relative_sites`.
        site_rangles: ``(..., B-1, 3)``.
        site_rvel: ``(..., B-1, 6)``.
    """

    root_pos: Any
    root_rot: Any
    root_qvel: Any
    root_vel_local: Any
    joint_pos: Any
    joint_vel: Any
    site_rpos: Any
    site_rangles: Any
    site_rvel: Any


@dataclass
class TrackFeatures:
    """Quantities of a state that several terms share (:func:`features`)."""

    root_rot: Any
    root_vel_local: Any
    site_rpos: Any
    site_rangles: Any
    site_rvel: Any


def array_module(x: Any) -> Any:
    """The array module of an array or container of arrays (numpy or torch)."""
    if isinstance(x, (TrackState, TrackRef)):
        x = x.root_pos
    elif isinstance(x, TrackFeatures):
        x = x.root_rot
    if type(x).__module__.startswith("torch"):
        import torch  # noqa: PLC0415

        return torch
    import numpy  # noqa: PLC0415

    return numpy


# ---------------------------------------------------------------------------
# Rotations (wxyz quaternions, matrices act on column vectors)
# ---------------------------------------------------------------------------


def _t(x: Any) -> Any:
    return x.swapaxes(-1, -2)


def _mv(m: Any, v: Any) -> Any:
    return (m @ v[..., None])[..., 0]


def quat_to_mat(xp: Any, q: Any) -> Any:
    """Unit quaternion (wxyz) -> rotation matrix."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rows = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )
    return xp.stack([xp.stack(r, axis=-1) for r in rows], axis=-2)


def mat_to_quat(xp: Any, m: Any) -> Any:
    """Rotation matrix -> unit quaternion (wxyz, ``w >= 0``), robust branch choice."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    qw = xp.sqrt(xp.clip(1 + m00 + m11 + m22, 1e-10, None)) / 2
    qx = xp.sqrt(xp.clip(1 + m00 - m11 - m22, 1e-10, None)) / 2
    qy = xp.sqrt(xp.clip(1 - m00 + m11 - m22, 1e-10, None)) / 2
    qz = xp.sqrt(xp.clip(1 - m00 - m11 + m22, 1e-10, None)) / 2
    cands = (
        (qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)),
        ((m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)),
        ((m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)),
        ((m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz),
    )
    best = xp.stack((qw, qx, qy, qz), axis=-1).argmax(-1)
    q = xp.stack(cands[3], axis=-1)
    for k in (2, 1, 0):
        q = xp.where((best == k)[..., None], xp.stack(cands[k], axis=-1), q)
    q = q / xp.sqrt((q * q).sum(-1))[..., None]
    return xp.where(q[..., :1] < 0, -q, q)


def quat_to_rotvec(xp: Any, q: Any) -> Any:
    """Unit quaternion (wxyz) -> rotation vector, stable near 0 and pi."""
    q = xp.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    v_sq = (v * v).sum(-1)[..., None]
    sin_half = xp.sqrt(xp.clip(v_sq, 1e-30, None))
    half = xp.arctan2(sin_half, q[..., :1])
    scale = xp.where(
        v_sq < 1e-12,
        2.0 / xp.clip(q[..., :1], 1e-6, None),
        2.0 * half / sin_half,
    )
    return v * scale


def so3_log(xp: Any, R: Any) -> Any:
    """Rotation matrix -> rotation vector."""
    return quat_to_rotvec(xp, mat_to_quat(xp, R))


def rotation_angle(xp: Any, Ra: Any, Rb: Any) -> Any:
    """Geodesic angle between rotations (via the relative quaternion; exact near 0)."""
    q = mat_to_quat(xp, _t(Ra) @ Rb)
    return 2 * xp.arctan2(
        xp.sqrt(xp.clip((q[..., 1:] ** 2).sum(-1), 1e-30, None)), xp.abs(q[..., 0])
    )


# ---------------------------------------------------------------------------
# Shared quantities
# ---------------------------------------------------------------------------


def relative_sites(xp: Any, P: Any, R: Any, V: Any, W: Any) -> tuple[Any, Any, Any]:
    """MuscleMimic's relative site quantities, the first body (pelvis) as main site.

    Args:
        xp: Array module.
        P: ``(..., B, 3)`` body positions.
        R: ``(..., B, 3, 3)`` body rotations.
        V: ``(..., B, 3)`` body linear velocities (world).
        W: ``(..., B, 3)`` body angular velocities (world).

    Returns:
        ``rpos (..., B-1, 3)`` positions minus the main body's (world axes);
        ``rangles (..., B-1, 3)`` rotation vectors of ``R_main^T R_body``;
        ``rvel (..., B-1, 6)`` ``[(R_main^T R_body)^T w_body - w_main,
        R_main (v_main - v_body)]`` (MuscleMimic's frames, kept).
    """
    P0, R0, V0, W0 = P[..., :1, :], R[..., :1, :, :], V[..., :1, :], W[..., :1, :]
    rrot = _t(R0) @ R[..., 1:, :, :]
    rpos = P[..., 1:, :] - P0
    lin = _mv(R0, V0 - V[..., 1:, :])
    ang = _mv(_t(rrot), W[..., 1:, :]) - W0
    return rpos, so3_log(xp, rrot), xp.concatenate((ang, lin), axis=-1)


def root_vel_local(xp: Any, root_rot: Any, root_qvel: Any) -> Any:
    """``[R^T v, w body]`` (MuscleMimic calls the body-frame angular velocity global)."""
    return xp.concatenate(
        (_mv(_t(root_rot), root_qvel[..., :3]), root_qvel[..., 3:]), axis=-1
    )


def features(xp: Any, s: TrackState) -> TrackFeatures:
    """The shared quantities of a state (relative sites, root rotation and velocity)."""
    R = quat_to_mat(xp, s.root_quat)
    rpos, rangles, rvel = relative_sites(
        xp, s.body_pos, s.body_rot, s.body_lin_vel, s.body_ang_vel
    )
    return TrackFeatures(R, root_vel_local(xp, R, s.root_qvel), rpos, rangles, rvel)


def _flat(x: Any) -> Any:
    return x.reshape(*x.shape[:-2], -1)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def _root(xp: Any, s: TrackState, f: TrackFeatures) -> Any:
    q = s.root_quat
    q = xp.where(q[..., :1] < 0, -q, q)
    return xp.concatenate((s.root_pos[..., 2:3], q), axis=-1)


OBSERVATIONS = {
    "root_pos": _root,
    "joint_pos": lambda xp, s, f: s.joint_pos,
    "root_vel": lambda xp, s, f: s.root_qvel,
    "joint_vel": lambda xp, s, f: s.joint_vel,
    "muscle_length": lambda xp, s, f: s.muscle_length,
    "muscle_velocity": lambda xp, s, f: s.muscle_velocity,
    "muscle_force": lambda xp, s, f: s.muscle_force,
    "muscle_excitation": lambda xp, s, f: s.excitation,
    "muscle_activation": lambda xp, s, f: s.act,
    "touch": lambda xp, s, f: s.touch,
    "site_rpos": lambda xp, s, f: _flat(f.site_rpos),
    "site_rangles": lambda xp, s, f: _flat(f.site_rangles),
    "site_rvel": lambda xp, s, f: _flat(f.site_rvel),
}
"""The body-state observation vocabulary: ``name -> fn(xp, state, features)``."""


def observation(xp: Any, s: TrackState, f: TrackFeatures, name: str) -> Any:
    """One body-state observation term (``"root_pos"``: height and wxyz quaternion, ``w >= 0``).

    Args:
        xp: Array module.
        s: The state.
        f: Its :func:`features`.
        name: A key of :data:`OBSERVATIONS`.

    Returns:
        ``(..., n)``.
    """
    return OBSERVATIONS[name](xp, s, f)


def goal_observation(xp: Any, now: TrackRef, future: TrackRef, phase: Any) -> Any:
    """The reference-only goal: GoalTrajMimic's concise lookahead and the motion phase.

    Args:
        xp: Array module.
        now: Reference at the current frame (batch ``...``).
        future: Reference at the K future frames (batch ``..., K``).
        phase: ``(...,)`` progress through the clip.

    Returns:
        ``(..., 3 (B-1) + K (9 + 3 (B-1)) + 1)``.
    """
    fut = xp.concatenate(
        (
            future.root_pos - now.root_pos[..., None, :],
            future.root_qvel - now.root_qvel[..., None, :],
            _flat(future.site_rpos),
        ),
        axis=-1,
    )
    return xp.concatenate((_flat(now.site_rpos), _flat(fut), phase[..., None]), axis=-1)


# ---------------------------------------------------------------------------
# MimicReward components: exp(-kernel * mean squared error)
# ---------------------------------------------------------------------------


def _exp_mse(xp: Any, err: Any, kernel: float, axes: Any = -1) -> Any:
    return xp.exp(-kernel * (err * err).mean(axes))


def _qpos(xp: Any, s: TrackState, f: TrackFeatures, r: TrackRef, kernel: float) -> Any:
    e = xp.concatenate(
        ((r.root_pos - s.root_pos) ** 2, (r.joint_pos - s.joint_pos) ** 2), axis=-1
    ).mean(-1)
    return xp.exp(-kernel * (e + rotation_angle(xp, r.root_rot, f.root_rot)))


def _qvel(xp: Any, s: TrackState, f: TrackFeatures, r: TrackRef, kernel: float) -> Any:
    err = xp.concatenate(
        (r.root_qvel - s.root_qvel, r.joint_vel - s.joint_vel), axis=-1
    )
    return _exp_mse(xp, err, kernel)


REWARDS = {
    "qpos": _qpos,
    "qvel": _qvel,
    "root_pos": lambda xp, s, f, r, k: _exp_mse(xp, r.root_pos - s.root_pos, k),
    "root_vel": lambda xp, s, f, r, k: _exp_mse(
        xp, r.root_vel_local - f.root_vel_local, k
    ),
    "rpos": lambda xp, s, f, r, k: _exp_mse(xp, r.site_rpos - f.site_rpos, k, (-1, -2)),
    "rquat": lambda xp, s, f, r, k: _exp_mse(
        xp, r.site_rangles - f.site_rangles, k, (-1, -2)
    ),
    "rvel_rot": lambda xp, s, f, r, k: _exp_mse(
        xp, r.site_rvel[..., :3] - f.site_rvel[..., :3], k, (-1, -2)
    ),
    "rvel_lin": lambda xp, s, f, r, k: _exp_mse(
        xp, r.site_rvel[..., 3:] - f.site_rvel[..., 3:], k, (-1, -2)
    ),
}
"""MimicReward's components ``name -> fn(xp, state, features, ref, kernel)`` in [0, 1]:
``qpos`` (root position, every joint and the root angle), ``qvel`` (free-joint qvel and
joints), ``root_pos``, ``root_vel`` (``[R^T v, w body]``), ``rpos`` / ``rquat`` /
``rvel_rot`` / ``rvel_lin`` (relative sites)."""


def reward(
    xp: Any, s: TrackState, f: TrackFeatures, r: TrackRef, name: str, kernel: float
) -> Any:
    """One MimicReward component (a key of :data:`REWARDS`), ``(...,)`` in [0, 1]."""
    return REWARDS[name](xp, s, f, r, kernel)


def weighted_reward(
    xp: Any, s: TrackState, f: TrackFeatures, r: TrackRef,
    terms: Any, step_weights: dict[str, float],
) -> Any:  # fmt: skip
    """``sum_k step_weight_k * component_k`` over ``terms`` (``(name, kernel)`` pairs)."""
    return sum(step_weights[n] * reward(xp, s, f, r, n, k) for n, k in terms)


# ---------------------------------------------------------------------------
# Deviations and the failure rule
# ---------------------------------------------------------------------------


def _norm(xp: Any, x: Any) -> Any:
    return xp.sqrt((x * x).sum(-1) + 1e-12)  # finite gradient at 0


DEVIATIONS = {
    "rel_site": lambda xp, s, f, r: _norm(xp, r.site_rpos - f.site_rpos).mean(-1),
    "root_pos": lambda xp, s, f, r: _norm(xp, r.root_pos - s.root_pos),
    "root_ori": lambda xp, s, f, r: rotation_angle(xp, r.root_rot, f.root_rot),
}
"""Deviations from the reference ``name -> fn(xp, state, features, ref)``: mean distance
of the relative sites (m), root position distance (m, drift included), root geodesic
angle (rad, heading included)."""


def deviation(xp: Any, s: TrackState, f: TrackFeatures, r: TrackRef, name: str) -> Any:
    """One deviation (a key of :data:`DEVIATIONS`), ``(...,)``."""
    return DEVIATIONS[name](xp, s, f, r)


def exceeds(
    xp: Any, s: TrackState, f: TrackFeatures, r: TrackRef, name: str, threshold: float
) -> Any:
    """The hard failure rule of one deviation: ``deviation > threshold`` (bool)."""
    return deviation(xp, s, f, r, name) > threshold


def soft_exceeds(xp: Any, error: Any, threshold: float, width: float) -> Any:
    """Smooth surrogate of the failure rule for imagined states: ``sigmoid((e - thr) / width)``.

    Not part of the MDP: the simulator only uses :func:`exceeds`.
    """
    x = (error - threshold) / width
    sigmoid = getattr(xp, "sigmoid", None)  # torch's fused, saturating sigmoid
    return sigmoid(x) if sigmoid is not None else 1.0 / (1.0 + xp.exp(-x))
