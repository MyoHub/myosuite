# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Yaw-rotate a motion-clip NPZ around the first-frame root XY."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from myosuite.physics import quat_math

_Z_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def yaw_quat(yaw: float) -> np.ndarray:
    """Unit quaternion for a rotation of *yaw* radians about Z."""
    return quat_math.axis_angle2quat(_Z_AXIS, float(yaw)).astype(np.float64)


def rotate_xy(xy: np.ndarray, yaw: float, origin_xy: np.ndarray) -> np.ndarray:
    """Rotate planar coordinates about *origin_xy*."""
    q = yaw_quat(yaw)
    pts = np.asarray(xy, dtype=np.float64)
    origin = np.asarray(origin_xy, dtype=np.float64)[:2]
    rel = np.zeros(pts.shape[:-1] + (3,), dtype=np.float64)
    rel[..., :2] = pts[..., :2] - origin
    rot = quat_math.rot_vec_quat(rel, q)
    out = np.array(pts, copy=True, dtype=np.float64)
    out[..., :2] = rot[..., :2] + origin
    return out


def _rotate_cvel(cvel: np.ndarray, yaw: float) -> np.ndarray:
    q = yaw_quat(yaw)
    out = np.array(cvel, copy=True, dtype=np.float64)
    if out.ndim < 1 or out.shape[-1] != 6:
        return out
    ang = np.zeros(out.shape[:-1] + (3,), dtype=np.float64)
    lin = np.zeros(out.shape[:-1] + (3,), dtype=np.float64)
    ang[..., :2] = out[..., :2]
    lin[..., :2] = out[..., 3:5]
    ang_r = quat_math.rot_vec_quat(ang, q)
    lin_r = quat_math.rot_vec_quat(lin, q)
    out[..., :2] = ang_r[..., :2]
    out[..., 3:5] = lin_r[..., :2]
    return out


def write_rotated_clip_npz(src: Path, dst: Path, yaw: float) -> Path:
    """Write *dst* as *src* yaw-rotated about the first-frame root XY.

    Rotates free-joint ``qpos``/``qvel``, plus ``site_xpos``, ``site_xmat``,
    ``subtree_com``, and ``cvel`` when those keys are present so a
    :class:`~myosuite.integrations.musclemimic.fullbody_local_policy.FullbodyObsAdapter`
    can reload the file.

    Args:
        src: Source motion NPZ.
        dst: Destination path (created, parents ok).
        yaw: Rotation about world Z, radians.

    Returns:
        Resolved *dst*.
    """
    src = Path(src)
    dst = Path(dst)
    data = dict(np.load(src, allow_pickle=True))
    qpos = np.array(data["qpos"], copy=True, dtype=np.float64)
    origin = qpos[0, :2].copy()
    qz = yaw_quat(yaw)
    qpos[:, :2] = rotate_xy(qpos[:, :2], yaw, origin)
    qpos[:, 3:7] = quat_math.mul_quat(qz, qpos[:, 3:7])
    data["qpos"] = qpos
    if "qvel" in data:
        qvel = np.array(data["qvel"], copy=True, dtype=np.float64)
        lin = np.zeros((qvel.shape[0], 3), dtype=np.float64)
        lin[:, :2] = qvel[:, :2]
        lin = quat_math.rot_vec_quat(lin, qz)
        qvel[:, :2] = lin[:, :2]
        ang = np.zeros((qvel.shape[0], 3), dtype=np.float64)
        ang[:, :2] = qvel[:, 3:5]
        ang = quat_math.rot_vec_quat(ang, qz)
        qvel[:, 3:5] = ang[:, :2]
        data["qvel"] = qvel
    for key in ("site_xpos", "subtree_com"):
        if key in data:
            arr = np.array(data[key], copy=True, dtype=np.float64)
            arr[..., :2] = rotate_xy(arr[..., :2], yaw, origin)
            data[key] = arr
    if "site_xmat" in data:
        xmat = np.array(data["site_xmat"], copy=True, dtype=np.float64)
        r = quat_math.quat2mat(qz)
        m = xmat.reshape(*xmat.shape[:-1], 3, 3)
        data["site_xmat"] = np.einsum("ij,...jk->...ik", r, m).reshape(xmat.shape)
    if "cvel" in data:
        data["cvel"] = _rotate_cvel(np.asarray(data["cvel"]), yaw)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **data)
    return dst.resolve()


def clip_natural_yaw(qpos: np.ndarray, qvel: np.ndarray | None) -> float:
    """Mean travel heading of a clip from planar velocity or qpos delta."""
    if qvel is not None and qvel.shape[0] > 0:
        mean = np.asarray(qvel[:, :2], dtype=np.float64).mean(axis=0)
    else:
        d = np.diff(np.asarray(qpos[:, :2], dtype=np.float64), axis=0)
        mean = d.mean(axis=0) if d.size else np.array([1.0, 0.0])
    return heading_from_xy(mean)


def heading_from_xy(xy: np.ndarray) -> float:
    v = np.asarray(xy, dtype=np.float64)
    return float(np.arctan2(float(v[1]), float(v[0])))
