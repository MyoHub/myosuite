# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""NumPy quaternion math, ``[w, x, y, z]`` (MuJoCo) convention.

Quaternion and vector helpers broadcast over leading batch axes (last axis =
components), like the torch and JAX twins in ``quat_math_torch.py`` /
``quat_math_jax.py``.
"""

import math
import numpy as np

# Near-zero cutoffs (quat2mat norm, mat2euler gimbal lock). Float32 values, as
# in the torch/JAX twins, so that all three backends take the same branch.
_FLOAT_EPS = np.finfo(np.float32).eps
_EPS4 = _FLOAT_EPS * 4.0

_CONJ = np.array([1.0, -1.0, -1.0, -1.0])


def mul_quat(qa, qb):
    """Hamilton product ``qa * qb``."""
    qa = np.asarray(qa, dtype=np.float64)
    qb = np.asarray(qb, dtype=np.float64)
    aw, ax, ay, az = qa[..., 0], qa[..., 1], qa[..., 2], qa[..., 3]
    bw, bx, by, bz = qb[..., 0], qb[..., 1], qb[..., 2], qb[..., 3]
    return np.stack(
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
    return np.asarray(quat, dtype=np.float64) * _CONJ


def quat2Vel(quat, dt=1):
    """Angular velocity that applies rotation *quat* over *dt*, as ``(speed, axis)``.

    ``speed * axis`` equals ``mujoco.mju_quat2Vel``: rotations by more than pi
    are taken the short way round, so ``q`` and ``-q`` (the same rotation) give
    the same velocity.
    """
    quat = np.asarray(quat, dtype=np.float64)
    axis = quat[..., 1:]
    sin_a_2 = np.sqrt(np.sum(axis**2, axis=-1))
    axis = axis / (sin_a_2[..., None] + 1e-8)
    speed = 2 * np.arctan2(sin_a_2, quat[..., 0])
    speed = speed - 2 * np.pi * (speed > np.pi)
    return speed / dt, axis


def diff_quat(quat1, quat2):
    neg = neg_quat(quat1)
    diff = mul_quat(quat2, neg)
    return diff


def quat_diff_to_vel(quat1, quat2, dt):
    diff = diff_quat(quat1, quat2)
    return quat2Vel(diff, dt)


def axis_angle2quat(axis, angle):
    """Quaternion of a rotation by *angle* about the unit vector *axis*."""
    half = np.asarray(angle, dtype=np.float64)[..., None] / 2
    xyz = np.sin(half) * np.asarray(axis, dtype=np.float64)
    w = np.broadcast_to(np.cos(half), xyz.shape[:-1] + (1,))
    return np.concatenate([w, xyz], axis=-1)


def euler2mat(euler):
    """Euler angles to rotation matrix, intrinsic X-Y'-Z'' (scipy ``"XYZ"``)."""
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, f"Invalid shaped euler {euler}"

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat


def euler2quat(euler):
    """Euler angles to quaternion, intrinsic X-Y'-Z'' (scipy ``"XYZ"``)."""
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, f"Invalid shape euler {euler}"

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat


def mat2euler(mat):
    """Rotation matrix to intrinsic X-Y'-Z'' Euler angles (scipy ``"XYZ"``)."""
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), f"Invalid shape matrix {mat}"

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(
        condition,
        -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
        -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]),
    )
    euler[..., 1] = np.where(
        condition, -np.arctan2(-mat[..., 0, 2], cy), -np.arctan2(-mat[..., 0, 2], cy)
    )
    euler[..., 0] = np.where(
        condition, -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]), 0.0
    )
    return euler


def mat2quat(mat):
    """Convert Rotation Matrix to Quaternion"""
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), f"Invalid shape matrix {mat}"

    Qxx, Qyx, Qzx = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    Qxy, Qyy, Qzy = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    Qxz, Qyz, Qzz = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(mat.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=["multi_index"])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q


def quat2euler(quat):
    """Quaternion to intrinsic X-Y'-Z'' Euler angles (scipy ``"XYZ"``)."""
    return mat2euler(quat2mat(quat))


def quat2mat(quat):
    """Quaternion to rotation matrix (identity for a near-zero quaternion)."""
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, f"Invalid shape quat {quat}"

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def quat2yaw(quat: np.ndarray) -> np.float64:
    """Yaw of one quaternion: ``quat2euler(quat)[2]`` without the full conversion.

    Runs only the operations of :func:`quat2mat` and :func:`mat2euler` that the
    yaw depends on, in the same order and with numpy's ``sum`` and ``arctan2``,
    so the result is bit-identical; the per-step heightmap needs only the yaw.

    Args:
        quat: Quaternion ``(w, x, y, z)``, shape ``(4,)``.

    Returns:
        The yaw angle in radians.
    """
    quat = np.asarray(quat, dtype=np.float64)
    Nq = np.sum(quat * quat, axis=-1)
    if not Nq > _FLOAT_EPS:  # quat2mat falls back to the identity matrix
        return -np.arctan2(0.0, 1.0)
    w, x, y, z = quat.tolist()
    s = 2.0 / float(Nq)
    X, Y, Z = x * s, y * s, z * s
    wX, wZ = w * X, w * Z
    xX, xY = x * X, x * Y
    yY, yZ, zZ = y * Y, y * Z, z * Z
    m12, m22 = yZ - wX, 1.0 - (xX + yY)
    if math.sqrt(m22 * m22 + m12 * m12) > _EPS4:
        return -np.arctan2(xY - wZ, 1.0 - (yY + zZ))
    return -np.arctan2(-(xY + wZ), 1.0 - (xX + zZ))


# multiply vector by 3D rotation matrix transpose
def rot_vec_mat_t(vec, mat):
    """Multiply *vec* by the transpose of the rotation matrix *mat*."""
    vec, mat = np.asarray(vec), np.asarray(mat)
    return np.stack(
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
    vec, mat = np.asarray(vec), np.asarray(mat)
    return np.stack(
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
    """Quaternion to ``[roll, pitch, yaw]``, the inverse of :func:`intrinsic_euler2quat`.

    Despite the name, the angles are extrinsic x-y-z (scipy ``"xyz"``); pitch
    lies in ``[-pi/2, pi/2]``.
    """
    quat = np.asarray(quat, dtype=np.float64)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    # Clip: rounding can push |sin(pitch)| past 1 at the poles (-> +-pi/2).
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=-1)


def intrinsic_euler2quat(euler):
    """``[roll, pitch, yaw]`` to quaternion.

    Despite the name, this is the extrinsic x-y-z convention (scipy ``"xyz"``):
    roll about the fixed X axis, then pitch about the fixed Y, then yaw about
    the fixed Z (equivalently intrinsic Z-Y'-X'': yaw, pitch, roll).
    """
    half = np.asarray(euler, dtype=np.float64) * 0.5
    sr, cr = np.sin(half[..., 0]), np.cos(half[..., 0])
    sp, cp = np.sin(half[..., 1]), np.cos(half[..., 1])
    sy, cy = np.sin(half[..., 2]), np.cos(half[..., 2])
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.stack([w, x, y, z], axis=-1)


def quat_from_euler_xyz_deg(euler_deg: list[float]) -> list[float]:
    """Convert ``[roll, pitch, yaw]`` in degrees to a quaternion.

    Uses ``intrinsic_euler2quat``, i.e. extrinsic x-y-z rotations (scipy
    ``"xyz"``), the convention of the historical half-angle composition used
    for glove/helmet mesh placement.  Do not use ``euler2quat`` here: that
    helper is intrinsic X-Y'-Z'' (scipy ``"XYZ"``).

    Args:
        euler_deg: [roll_deg, pitch_deg, yaw_deg] about the fixed X, Y, Z axes.

    Returns:
        Quaternion as [w, x, y, z].
    """
    euler_rad = np.deg2rad(np.asarray(euler_deg, dtype=np.float64))
    return intrinsic_euler2quat(euler_rad).astype(np.float64).tolist()


def calculate_cosine(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Return cos(theta) between two vectors, supporting batch dimensions.

    Args:
        vec1: First vector (optionally batched).
        vec2: Second vector with the same shape as *vec1*.

    Returns:
        Cosine similarity with the same batch shape as the inputs.
    """
    if np.shape(vec1) != np.shape(vec2):
        raise ValueError(
            f"vec1 and vec2 must have the same shape, got {np.shape(vec1)} vs {np.shape(vec2)}"
        )
    norm_product = np.linalg.norm(vec1, axis=-1) * np.linalg.norm(vec2, axis=-1)
    if np.any(norm_product == 0):
        norm_product = np.where(norm_product == 0, 1.0, norm_product)
    return np.einsum("...i,...i", vec1, vec2) / norm_product
