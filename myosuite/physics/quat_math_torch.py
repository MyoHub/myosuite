# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Torch backend for quaternion math — same API as quat_math.py and quat_math_jax.py.

All functions accept and return torch.Tensor objects.  Quaternion convention:
[w, x, y, z] throughout, matching MuJoCo and the other backends.
"""

import torch

_EPS = torch.finfo(torch.float32).eps
_EPS4 = _EPS * 4.0


def mul_quat(qa: torch.Tensor, qb: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two unit quaternions."""
    return torch.stack(
        [
            qa[0] * qb[0] - qa[1] * qb[1] - qa[2] * qb[2] - qa[3] * qb[3],
            qa[0] * qb[1] + qa[1] * qb[0] + qa[2] * qb[3] - qa[3] * qb[2],
            qa[0] * qb[2] - qa[1] * qb[3] + qa[2] * qb[0] + qa[3] * qb[1],
            qa[0] * qb[3] + qa[1] * qb[2] - qa[2] * qb[1] + qa[3] * qb[0],
        ]
    )


def neg_quat(quat: torch.Tensor) -> torch.Tensor:
    """Conjugate (inverse for unit quaternions)."""
    return torch.stack([quat[0], -quat[1], -quat[2], -quat[3]])


def quat2Vel(quat: torch.Tensor, dt: float = 1) -> tuple[torch.Tensor, torch.Tensor]:
    axis = quat[1:].clone()
    sin_a_2 = torch.sqrt(torch.sum(axis**2))
    axis = axis / (sin_a_2 + 1e-8)
    speed = 2 * torch.atan2(sin_a_2, quat[0]) / dt
    return speed, axis


def diff_quat(quat1: torch.Tensor, quat2: torch.Tensor) -> torch.Tensor:
    return mul_quat(quat2, neg_quat(quat1))


def quat_diff_to_vel(
    quat1: torch.Tensor, quat2: torch.Tensor, dt: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return quat2Vel(diff_quat(quat1, quat2), dt)


def axis_angle2quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle / 2)
    s = torch.sin(angle / 2)
    return torch.stack([c, s * axis[0], s * axis[1], s * axis[2]])


def euler2mat(euler: torch.Tensor) -> torch.Tensor:
    """Extrinsic XYZ Euler angles → rotation matrix."""
    euler = euler.to(torch.float32)
    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = torch.empty(
        euler.shape[:-1] + (3, 3), dtype=torch.float32, device=euler.device
    )
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


def euler2quat(euler: torch.Tensor) -> torch.Tensor:
    """Extrinsic XYZ Euler angles → unit quaternion."""
    euler = euler.to(torch.float32)
    ai = euler[..., 2] / 2
    aj = -euler[..., 1] / 2
    ak = euler[..., 0] / 2
    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = torch.empty(
        euler.shape[:-1] + (4,), dtype=torch.float32, device=euler.device
    )
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat


def mat2euler(mat: torch.Tensor) -> torch.Tensor:
    """Rotation matrix → extrinsic XYZ Euler angles."""
    mat = mat.to(torch.float32)
    cy = torch.sqrt(mat[..., 2, 2] ** 2 + mat[..., 1, 2] ** 2)
    condition = cy > _EPS4
    euler = torch.empty(mat.shape[:-1], dtype=torch.float32, device=mat.device)
    euler[..., 2] = torch.where(
        condition,
        -torch.atan2(mat[..., 0, 1], mat[..., 0, 0]),
        -torch.atan2(-mat[..., 1, 0], mat[..., 1, 1]),
    )
    euler[..., 1] = -torch.atan2(-mat[..., 0, 2], cy)
    euler[..., 0] = torch.where(
        condition, -torch.atan2(mat[..., 1, 2], mat[..., 2, 2]), torch.zeros_like(cy)
    )
    return euler


def quat2mat(quat: torch.Tensor) -> torch.Tensor:
    """Unit quaternion → rotation matrix."""
    quat = quat.to(torch.float32)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = torch.sum(quat * quat, dim=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = torch.empty(quat.shape[:-1] + (3, 3), dtype=torch.float32, device=quat.device)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)

    eye = torch.eye(3, dtype=torch.float32, device=quat.device)
    return torch.where(
        (Nq > _EPS)[..., None, None].expand_as(mat), mat, eye.expand_as(mat)
    )


def mat2quat(mat: torch.Tensor) -> torch.Tensor:
    """Rotation matrix → unit quaternion (Shepperd method)."""
    mat = mat.to(torch.float32)
    trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    # Four candidate discriminants — pick the numerically largest
    diag = torch.stack(
        [
            1.0 + mat[..., 0, 0] - mat[..., 1, 1] - mat[..., 2, 2],  # 4x²
            1.0 - mat[..., 0, 0] + mat[..., 1, 1] - mat[..., 2, 2],  # 4y²
            1.0 - mat[..., 0, 0] - mat[..., 1, 1] + mat[..., 2, 2],  # 4z²
            1.0 + trace,  # 4w²
        ],
        dim=-1,
    )
    choice = torch.argmax(diag, dim=-1)

    def _case(i: int) -> torch.Tensor:
        s = torch.sqrt(torch.clamp(diag[..., i], min=_EPS)) * 2  # 4 * component
        if i == 3:  # w is largest
            w = s / 4
            x = (mat[..., 2, 1] - mat[..., 1, 2]) / s
            y = (mat[..., 0, 2] - mat[..., 2, 0]) / s
            z = (mat[..., 1, 0] - mat[..., 0, 1]) / s
        elif i == 0:  # x is largest
            x = s / 4
            w = (mat[..., 2, 1] - mat[..., 1, 2]) / s
            y = (mat[..., 0, 1] + mat[..., 1, 0]) / s
            z = (mat[..., 0, 2] + mat[..., 2, 0]) / s
        elif i == 1:  # y is largest
            y = s / 4
            w = (mat[..., 0, 2] - mat[..., 2, 0]) / s
            x = (mat[..., 0, 1] + mat[..., 1, 0]) / s
            z = (mat[..., 1, 2] + mat[..., 2, 1]) / s
        else:  # z is largest
            z = s / 4
            w = (mat[..., 1, 0] - mat[..., 0, 1]) / s
            x = (mat[..., 0, 2] + mat[..., 2, 0]) / s
            y = (mat[..., 1, 2] + mat[..., 2, 1]) / s
        return torch.stack([w, x, y, z], dim=-1)

    cases = torch.stack([_case(i) for i in range(4)], dim=-1)
    idx = choice[..., None, None].expand(*choice.shape, 4, 1)
    q = cases.gather(-1, idx).squeeze(-1)
    # prefer w ≥ 0
    q = torch.where(q[..., :1] < 0, -q, q)
    return q


def quat2euler(quat: torch.Tensor) -> torch.Tensor:
    return mat2euler(quat2mat(quat))


def rot_vec_mat_t(vec: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            mat[0, 0] * vec[0] + mat[1, 0] * vec[1] + mat[2, 0] * vec[2],
            mat[0, 1] * vec[0] + mat[1, 1] * vec[1] + mat[2, 1] * vec[2],
            mat[0, 2] * vec[0] + mat[1, 2] * vec[1] + mat[2, 2] * vec[2],
        ]
    )


def rot_vec_mat(vec: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            mat[0, 0] * vec[0] + mat[0, 1] * vec[1] + mat[0, 2] * vec[2],
            mat[1, 0] * vec[0] + mat[1, 1] * vec[1] + mat[1, 2] * vec[2],
            mat[2, 0] * vec[0] + mat[2, 1] * vec[1] + mat[2, 2] * vec[2],
        ]
    )


def rot_vec_quat(vec: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    return rot_vec_mat(vec, quat2mat(quat))


def quat2euler_intrinsic(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.copysign(torch.tensor(torch.pi / 2), sinp),
        torch.asin(sinp),
    )
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw])


def intrinsic_euler2quat(euler: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = euler[0], euler[1], euler[2]
    hr, hp, hy = roll * 0.5, pitch * 0.5, yaw * 0.5
    sr, cr = torch.sin(hr), torch.cos(hr)
    sp, cp = torch.sin(hp), torch.cos(hp)
    sy, cy = torch.sin(hy), torch.cos(hy)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z])


def calculate_cosine(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between two vectors, supporting batch dimensions."""
    if vec1.shape != vec2.shape:
        raise ValueError(
            f"vec1 and vec2 must have the same shape, got {vec1.shape} vs {vec2.shape}"
        )
    norm = torch.linalg.norm(vec1, dim=-1) * torch.linalg.norm(vec2, dim=-1)
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)
    return torch.einsum("...i,...i->...", vec1, vec2) / norm
