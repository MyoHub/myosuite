# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# this package for the notices of the code it includes.
"""Batched rotation utilities (quaternions are wxyz, rotation matrices act on column vectors).

From supermimic_vibed (github.com/zita-ch/supermimic_vibed, MIT, Copyright (c) 2026 ETH Zurich, Chong Zhang).

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at commit
8a6244b, ``src/muscle_supermimic/utils/rotations.py``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))  # fmt: skip


def mat_to_quat(m: torch.Tensor) -> torch.Tensor:
    """Robust matrix -> quaternion (wxyz, w >= 0)."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    qw = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=1e-10)) / 2
    qx = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=1e-10)) / 2
    qy = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=1e-10)) / 2
    qz = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=1e-10)) / 2
    cands = torch.stack(
        (
            torch.stack(
                (
                    qw,
                    (m21 - m12) / (4 * qw),
                    (m02 - m20) / (4 * qw),
                    (m10 - m01) / (4 * qw),
                ),
                -1,
            ),
            torch.stack(
                (
                    (m21 - m12) / (4 * qx),
                    qx,
                    (m01 + m10) / (4 * qx),
                    (m02 + m20) / (4 * qx),
                ),
                -1,
            ),
            torch.stack(
                (
                    (m02 - m20) / (4 * qy),
                    (m01 + m10) / (4 * qy),
                    qy,
                    (m12 + m21) / (4 * qy),
                ),
                -1,
            ),
            torch.stack(
                (
                    (m10 - m01) / (4 * qz),
                    (m02 + m20) / (4 * qz),
                    (m12 + m21) / (4 * qz),
                    qz,
                ),
                -1,
            ),
        ),
        dim=-2,
    )
    idx = torch.stack((qw, qx, qy, qz), -1).argmax(-1)
    q = torch.gather(
        cands, -2, idx[..., None, None].expand(idx.shape + (1, 4))
    ).squeeze(-2)
    q = F.normalize(q, dim=-1)
    return q * torch.where(q[..., :1] < 0, -1.0, 1.0)


def axis_angle_to_mat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues for a fixed unit axis (..., 3) and angle (...)."""
    x, y, z = axis.unbind(-1)
    c, s = torch.cos(angle), torch.sin(angle)
    C = 1 - c
    return torch.stack(
        (
            c + x * x * C, x * y * C - z * s, x * z * C + y * s,
            y * x * C + z * s, c + y * y * C, y * z * C - x * s,
            z * x * C - y * s, z * y * C + x * s, c + z * z * C,
        ),
        dim=-1,
    ).reshape(angle.shape + (3, 3))  # fmt: skip


def mv3(R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """R x for (..., 3, 3) matrices and (..., 3) vectors, as a product and a sum: as a batched GEMM a 3x3 product is
    slow on the GPU, and torch.compile cannot fuse it with its neighbours."""
    return (R * x.unsqueeze(-2)).sum(-1)


def mm3(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A B for (..., 3, 3) matrices, as a product and a sum (see :func:`mv3`)."""
    return (A.unsqueeze(-1) * B.unsqueeze(-3)).sum(-2)


def so3_exp(v: torch.Tensor) -> torch.Tensor:
    """Rotation vector (..., 3) -> matrix, smooth at zero."""
    theta2 = (v * v).sum(-1, keepdim=True)
    theta = torch.sqrt(theta2 + 1e-12)
    small = theta2 < 1e-8
    a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / (theta2 + 1e-12))
    K = skew(v)
    eye = torch.eye(3, device=v.device, dtype=v.dtype).expand(K.shape)
    return eye + a[..., None] * K + b[..., None] * mm3(K, K)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Matrix -> rotation vector via the quaternion (stable near 0 and pi)."""
    q = mat_to_quat(R)
    return quat_to_rotvec(q)


def quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)
    v = q[..., 1:]
    sin_half = torch.linalg.norm(v, dim=-1, keepdim=True)
    half = torch.atan2(sin_half, q[..., :1])
    scale = torch.where(
        sin_half < 1e-6,
        2.0 / torch.clamp(q[..., :1], min=1e-6),
        2 * half / (sin_half + 1e-12),
    )
    return v * scale


def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(
        v.shape[:-1] + (3, 3)
    )


def rot_angle_sq(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """Squared geodesic angle between rotations, differentiable (via acos of trace, clamped)."""
    tr = (Ra * Rb).sum((-1, -2))  # trace(Ra^T Rb)
    cos = torch.clamp((tr - 1) / 2, -1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos) ** 2


def yaw_mat(R: torch.Tensor) -> torch.Tensor:
    """Heading (yaw-only) rotation of R, from the projected x axis."""
    fx, fy = R[..., 0, 0], R[..., 1, 0]
    n = torch.sqrt(fx * fx + fy * fy + 1e-9)
    c, s = fx / n, fy / n
    o, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, o, s, c, o, o, o, one), -1).reshape(
        R.shape[:-2] + (3, 3)
    )


def mat_to_6d(R: torch.Tensor) -> torch.Tensor:
    """First two columns, flattened (6D continuous rotation representation)."""
    return R[..., :, :2].reshape(R.shape[:-2] + (6,))


def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    s = torch.sin(theta)
    t = t[..., None]
    w0 = torch.where(s < 1e-6, 1 - t, torch.sin((1 - t) * theta) / (s + 1e-12))
    w1 = torch.where(s < 1e-6, t, torch.sin(t * theta) / (s + 1e-12))
    return F.normalize(w0 * q0 + w1 * q1, dim=-1)
