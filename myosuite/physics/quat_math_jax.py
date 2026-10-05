# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""JAX quaternion math, ``[w, x, y, z]`` (MuJoCo) convention.

Same API as ``quat_math.py``; the quaternion and vector helpers broadcast over
leading batch axes (last axis = components).  ``mat2quat`` takes one matrix.
"""

import jax.numpy as jp
import jax

# Constants for floating-point precision
_FLOAT_EPS = jp.finfo(jp.float32).eps
_EPS4 = _FLOAT_EPS * 4.0


def mul_quat(qa, qb):
    """Hamilton product ``qa * qb``."""
    qa, qb = jp.asarray(qa), jp.asarray(qb)
    aw, ax, ay, az = qa[..., 0], qa[..., 1], qa[..., 2], qa[..., 3]
    bw, bx, by, bz = qb[..., 0], qb[..., 1], qb[..., 2], qb[..., 3]
    return jp.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def neg_quat(quat):
    """Conjugate (the inverse of a unit quaternion)."""
    return jp.asarray(quat) * jp.array([1.0, -1.0, -1.0, -1.0])


def quat2Vel(quat, dt=1):
    """Angular velocity that applies rotation *quat* over *dt*, as ``(speed, axis)``.

    ``speed * axis`` equals ``mujoco.mju_quat2Vel`` (rotations by more than pi
    are taken the short way round).
    """
    quat = jp.asarray(quat)
    axis = quat[..., 1:]
    sin_a_2 = jp.sqrt(jp.sum(axis**2, axis=-1))
    axis = axis / (sin_a_2[..., None] + 1e-8)
    speed = 2 * jp.arctan2(sin_a_2, quat[..., 0])
    speed = jp.where(speed > jp.pi, speed - 2 * jp.pi, speed)
    return speed / dt, axis


def diff_quat(quat1, quat2):
    neg = neg_quat(quat1)
    return mul_quat(quat2, neg)


def quat_diff_to_vel(quat1, quat2, dt):
    diff = diff_quat(quat1, quat2)
    return quat2Vel(diff, dt)


def axis_angle2quat(axis, angle):
    """Quaternion of a rotation by *angle* about the unit vector *axis*."""
    half = jp.asarray(angle)[..., None] / 2
    xyz = jp.sin(half) * jp.asarray(axis)
    w = jp.broadcast_to(jp.cos(half), xyz.shape[:-1] + (1,))
    return jp.concatenate([w, xyz], axis=-1)


def euler2mat(euler):
    """Intrinsic X-Y'-Z'' Euler angles (scipy ``"XYZ"``) to rotation matrix."""
    euler = jp.asarray(euler, dtype=jp.float32)
    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = jp.sin(ai), jp.sin(aj), jp.sin(ak)
    ci, cj, ck = jp.cos(ai), jp.cos(aj), jp.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = jp.empty(euler.shape[:-1] + (3, 3), dtype=jp.float32)
    mat = mat.at[..., 2, 2].set(cj * ck)
    mat = mat.at[..., 2, 1].set(sj * sc - cs)
    mat = mat.at[..., 2, 0].set(sj * cc + ss)
    mat = mat.at[..., 1, 2].set(cj * sk)
    mat = mat.at[..., 1, 1].set(sj * ss + cc)
    mat = mat.at[..., 1, 0].set(sj * cs - sc)
    mat = mat.at[..., 0, 2].set(-sj)
    mat = mat.at[..., 0, 1].set(cj * si)
    mat = mat.at[..., 0, 0].set(cj * ci)
    return mat


def euler2quat(euler):
    """Intrinsic X-Y'-Z'' Euler angles (scipy ``"XYZ"``) to quaternion."""
    euler = jp.asarray(euler, dtype=jp.float32)
    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = jp.sin(ai), jp.sin(aj), jp.sin(ak)
    ci, cj, ck = jp.cos(ai), jp.cos(aj), jp.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = jp.empty(euler.shape[:-1] + (4,), dtype=jp.float32)
    quat = quat.at[..., 0].set(cj * cc + sj * ss)
    quat = quat.at[..., 3].set(cj * sc - sj * cs)
    quat = quat.at[..., 2].set(-(cj * ss + sj * cc))
    quat = quat.at[..., 1].set(cj * cs - sj * sc)
    return quat


def mat2euler(mat):
    """Rotation matrix to intrinsic X-Y'-Z'' Euler angles (scipy ``"XYZ"``)."""
    mat = jp.asarray(mat, dtype=jp.float32)
    cy = jp.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = jp.empty(mat.shape[:-2] + (3,), dtype=jp.float32)
    euler = euler.at[..., 2].set(
        jp.where(
            condition,
            -jp.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
            -jp.arctan2(-mat[..., 1, 0], mat[..., 1, 1]),
        )
    )
    euler = euler.at[..., 1].set(
        jp.where(
            condition,
            -jp.arctan2(-mat[..., 0, 2], cy),
            -jp.arctan2(-mat[..., 0, 2], cy),
        )
    )
    euler = euler.at[..., 0].set(
        jp.where(condition, -jp.arctan2(mat[..., 1, 2], mat[..., 2, 2]), 0.0)
    )
    return euler


def mat2quat(mat):
    """Convert Rotation Matrix to Quaternion using JAX"""
    mat = jp.asarray(mat, dtype=jp.float32)
    assert mat.shape == (3, 3), f"Invalid shape matrix {mat.shape}"

    def case_1(mat):
        trace = 1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]
        s = 2.0 * jp.sqrt(trace)
        s = jp.where(mat[1, 2] < mat[2, 1], -s, s)
        q1 = 0.25 * s
        s = 1.0 / s
        q0 = (mat[1, 2] - mat[2, 1]) * s
        q2 = (mat[0, 1] + mat[1, 0]) * s
        q3 = (mat[2, 0] + mat[0, 2]) * s
        return jp.array([q0, q1, q2, q3])

    def case_2(mat):
        trace = 1.0 - mat[0, 0] + mat[1, 1] - mat[2, 2]
        s = 2.0 * jp.sqrt(trace)
        s = jp.where(mat[2, 0] < mat[0, 2], -s, s)
        q2 = 0.25 * s
        s = 1.0 / s
        q0 = (mat[2, 0] - mat[0, 2]) * s
        q1 = (mat[0, 1] + mat[1, 0]) * s
        q3 = (mat[1, 2] + mat[2, 1]) * s
        return jp.array([q0, q1, q2, q3])

    def case_3(mat):
        trace = 1.0 - mat[0, 0] - mat[1, 1] + mat[2, 2]
        s = 2.0 * jp.sqrt(trace)
        s = jp.where(mat[0, 1] < mat[1, 0], -s, s)
        q3 = 0.25 * s
        s = 1.0 / s
        q0 = (mat[0, 1] - mat[1, 0]) * s
        q1 = (mat[2, 0] + mat[0, 2]) * s
        q2 = (mat[1, 2] + mat[2, 1]) * s
        return jp.array([q0, q1, q2, q3])

    def case_4(mat):
        trace = 1.0 + mat[0, 0] + mat[1, 1] + mat[2, 2]
        s = 2.0 * jp.sqrt(trace)
        q0 = 0.25 * s
        s = 1.0 / s
        q1 = (mat[1, 2] - mat[2, 1]) * s
        q2 = (mat[2, 0] - mat[0, 2]) * s
        q3 = (mat[0, 1] - mat[1, 0]) * s
        return jp.array([q0, q1, q2, q3])

    # Conditional execution for efficiency
    q = jax.lax.cond(
        mat[2, 2] < 0.0,
        lambda mat: jax.lax.cond(mat[0, 0] > mat[1, 1], case_1, case_2, mat),
        lambda mat: jax.lax.cond(mat[0, 0] < -mat[1, 1], case_3, case_4, mat),
        mat,
    )

    q = q.at[1:].set(-q[1:])
    return q


def quat2euler(quat):
    return mat2euler(quat2mat(quat))


def quat2mat(quat):
    quat = jp.asarray(quat, dtype=jp.float32)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = jp.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = jp.empty(quat.shape[:-1] + (3, 3), dtype=jp.float32)
    mat = mat.at[..., 0, 0].set(1.0 - (yY + zZ))
    mat = mat.at[..., 0, 1].set(xY - wZ)
    mat = mat.at[..., 0, 2].set(xZ + wY)
    mat = mat.at[..., 1, 0].set(xY + wZ)
    mat = mat.at[..., 1, 1].set(1.0 - (xX + zZ))
    mat = mat.at[..., 1, 2].set(yZ - wX)
    mat = mat.at[..., 2, 0].set(xZ - wY)
    mat = mat.at[..., 2, 1].set(yZ + wX)
    mat = mat.at[..., 2, 2].set(1.0 - (xX + yY))
    return jp.where((Nq > _FLOAT_EPS)[..., jp.newaxis, jp.newaxis], mat, jp.eye(3))


def rot_vec_mat_t(vec, mat):
    """Multiply *vec* by the transpose of the rotation matrix *mat*."""
    vec, mat = jp.asarray(vec), jp.asarray(mat)
    return jp.stack(
        [
            mat[..., 0, 0] * vec[..., 0]
            + mat[..., 1, 0] * vec[..., 1]
            + mat[..., 2, 0] * vec[..., 2],
            mat[..., 0, 1] * vec[..., 0]
            + mat[..., 1, 1] * vec[..., 1]
            + mat[..., 2, 1] * vec[..., 2],
            mat[..., 0, 2] * vec[..., 0]
            + mat[..., 1, 2] * vec[..., 1]
            + mat[..., 2, 2] * vec[..., 2],
        ],
        axis=-1,
    )


def rot_vec_mat(vec, mat):
    """Multiply *vec* by the rotation matrix *mat*."""
    vec, mat = jp.asarray(vec), jp.asarray(mat)
    return jp.stack(
        [
            mat[..., 0, 0] * vec[..., 0]
            + mat[..., 0, 1] * vec[..., 1]
            + mat[..., 0, 2] * vec[..., 2],
            mat[..., 1, 0] * vec[..., 0]
            + mat[..., 1, 1] * vec[..., 1]
            + mat[..., 1, 2] * vec[..., 2],
            mat[..., 2, 0] * vec[..., 0]
            + mat[..., 2, 1] * vec[..., 1]
            + mat[..., 2, 2] * vec[..., 2],
        ],
        axis=-1,
    )


def rot_vec_quat(vec, quat):
    """Rotate *vec* by the quaternion *quat*."""
    return rot_vec_mat(vec, quat2mat(quat))


def quat2euler_intrinsic(quat):
    """Quaternion to ``[roll, pitch, yaw]``, extrinsic x-y-z (scipy ``"xyz"``).

    Inverse of :func:`intrinsic_euler2quat`; pitch lies in ``[-pi/2, pi/2]``.
    """
    quat = jp.asarray(quat)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    roll = jp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = jp.arcsin(jp.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = jp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return jp.stack([roll, pitch, yaw], axis=-1)


def intrinsic_euler2quat(euler):
    """``[roll, pitch, yaw]`` to quaternion, extrinsic x-y-z (scipy ``"xyz"``).

    Despite the name: roll about the fixed X, then pitch about the fixed Y, then
    yaw about the fixed Z (equivalently intrinsic Z-Y'-X'').
    """
    half = jp.asarray(euler) * 0.5
    sr, cr = jp.sin(half[..., 0]), jp.cos(half[..., 0])
    sp, cp = jp.sin(half[..., 1]), jp.cos(half[..., 1])
    sy, cy = jp.sin(half[..., 2]), jp.cos(half[..., 2])
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return jp.stack([w, x, y, z], axis=-1)
