# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Targetless multi-camera extrinsic calibration from walking human motion.

Adapted from the algorithms of HumanCalib (Delaplace, 2026, MIT) and
Lee et al. (IEEE RA-L, 2022): 3D leg-swing subject selection, Umeyama/Procrustes
camera initialisation, outlier-frame rejection, analytic-Jacobian bundle
adjustment, and reference-free milliradian acceptance gating.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from myosuite.biomechanics.camera import (
    DEFAULT_HEAD_BAND_RATIO,
    PinholeCamera,
    SequenceTriangulationResult,
    WalkFrameAlignment,
    align_walk_frame,
    triangulate_sequence,
)

MAX_ACCEPT_MRE_MRAD = 15.0


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _so3_left_jacobian(rvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rvec))
    k = _skew(rvec)
    if theta < 1e-6:
        return np.eye(3) + 0.5 * k + (1.0 / 6.0) * (k @ k)
    t2 = theta * theta
    return (
        np.eye(3)
        + ((1.0 - np.cos(theta)) / t2) * k
        + ((theta - np.sin(theta)) / (t2 * theta)) * (k @ k)
    )


def umeyama_alignment(
    src: np.ndarray, dst: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit ``dst ~= scale * rotation @ src + translation`` with ``det(rotation) = +1``."""
    a, b = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] != 3 or len(a) < 3:
        raise ValueError("src and dst must have matching shape (N >= 3, 3)")
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    ac, bc = a - mu_a, b - mu_b
    var_a = float(np.mean(np.sum(ac * ac, axis=1)))
    if var_a < 1e-12:
        raise ValueError("Degenerate source point set for Umeyama alignment")
    cov = (bc.T @ ac) / len(a)
    u, s_vals, vt = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[2] = -1.0
    rot = u @ np.diag(d) @ vt
    scale = float(np.sum(s_vals * d) / var_a)
    trans = mu_b - scale * (rot @ mu_a)
    return scale, rot, trans


def place_skeleton_in_camera(
    joints_3d_rel: np.ndarray,
    pixels_2d: np.ndarray,
    intrinsic: np.ndarray,
    *,
    confidences: np.ndarray | None = None,
    min_confidence: float = 0.1,
) -> np.ndarray:
    """Place root-relative ``(T, J, 3)`` lifted poses in camera coordinates from 2D pixels."""
    rel = np.asarray(joints_3d_rel, dtype=float)
    uv = np.asarray(pixels_2d, dtype=float)
    k_inv = np.linalg.inv(np.asarray(intrinsic, dtype=float))
    if rel.ndim != 3 or rel.shape[2] != 3 or uv.shape != (*rel.shape[:2], 2):
        raise ValueError("joints_3d_rel (T, J, 3) and pixels_2d (T, J, 2) must match")
    conf = (
        np.ones(rel.shape[:2])
        if confidences is None
        else np.asarray(confidences, dtype=float)
    )
    placed = np.full_like(rel, np.nan)
    for t in range(len(rel)):
        ok = (
            np.isfinite(rel[t]).all(axis=1)
            & np.isfinite(uv[t]).all(axis=1)
            & (conf[t] >= min_confidence)
        )
        idx = np.flatnonzero(ok)
        if len(idx) < 3:
            continue
        norm_xy = (np.column_stack((uv[t, idx], np.ones(len(idx)))) @ k_inv.T)[:, :2]
        w = np.sqrt(conf[t, idx])
        a = np.zeros((2 * len(idx), 3))
        b = np.zeros(2 * len(idx))
        p = rel[t, idx]
        a[0::2, 0], a[0::2, 2] = w, -w * norm_xy[:, 0]
        a[1::2, 1], a[1::2, 2] = w, -w * norm_xy[:, 1]
        b[0::2] = w * (norm_xy[:, 0] * p[:, 2] - p[:, 0])
        b[1::2] = w * (norm_xy[:, 1] * p[:, 2] - p[:, 1])
        root, *_ = np.linalg.lstsq(a, b, rcond=None)
        if root[2] > 0.05:
            placed[t, idx] = p + root
    return placed


def select_walking_subject(
    candidates_2d: Sequence[Sequence[np.ndarray]],
    candidates_3d: Sequence[Sequence[np.ndarray]],
    candidates_conf: Sequence[Sequence[np.ndarray]],
    *,
    hip_indices: tuple[int, int],
    ankle_indices: tuple[int, int],
    min_swing_speed: float = 0.15,
    min_camera_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the walking person per camera and frame via viewpoint-invariant 3D leg swing."""
    n_cam = len(candidates_2d)
    n_frames = len(candidates_2d[0])
    n_joints = next(
        arr.shape[1]
        for cam_frames in candidates_2d
        for arr in cam_frames
        if len(arr) > 0
    )
    sel_2d = np.full((n_cam, n_frames, n_joints, 2), np.nan)
    sel_3d = np.full((n_cam, n_frames, n_joints, 3), np.nan)
    sel_cf = np.zeros((n_cam, n_frames, n_joints))
    active = np.zeros((n_cam, n_frames), dtype=bool)

    for c in range(n_cam):
        for t in range(n_frames):
            p2, p3, cf = candidates_2d[c][t], candidates_3d[c][t], candidates_conf[c][t]
            ref_t = t - 1 if t > 0 else (1 if n_frames > 1 else 0)
            p3_ref = candidates_3d[c][ref_t]
            if len(p3) == 0:
                continue
            hips = 0.5 * (p3[:, hip_indices[0]] + p3[:, hip_indices[1]])
            ankles = p3[:, list(ankle_indices)] - hips[:, None, :]
            leg_len = np.maximum(np.linalg.norm(ankles, axis=-1).mean(axis=1), 1e-3)
            if ref_t != t and len(p3_ref) > 0:
                ref_hips = 0.5 * (p3_ref[:, hip_indices[0]] + p3_ref[:, hip_indices[1]])
                ref_ankles = p3_ref[:, list(ankle_indices)] - ref_hips[:, None, :]
                dists = np.linalg.norm(hips[:, None, :] - ref_hips[None, :, :], axis=-1)
                nearest = np.argmin(dists, axis=1)
                matched = dists[np.arange(len(p3)), nearest] < 0.8 * leg_len
                swing = (
                    np.linalg.norm(ankles - ref_ankles[nearest], axis=-1).mean(axis=1)
                    / leg_len
                )
                scores = np.where(matched, swing, 0.0)
                best = int(np.argmax(scores))
                active[c, t] = bool(scores[best] >= min_swing_speed)
            else:
                best = int(np.argmax(cf.mean(axis=1)))
                active[c, t] = True
            sel_2d[c, t], sel_3d[c, t], sel_cf[c, t] = p2[best], p3[best], cf[best]

    keep = active.mean(axis=0) >= min_camera_fraction
    if keep.any():
        sel_2d[:, ~keep] = np.nan
        sel_3d[:, ~keep] = np.nan
        sel_cf[:, ~keep] = 0.0
    return sel_2d, sel_3d, sel_cf


def init_extrinsics_procrustes(
    intrinsics: Sequence[np.ndarray],
    poses_3d_cam: np.ndarray,
    *,
    camera_names: Sequence[str] | None = None,
    distortions: Sequence[Sequence[float]] | None = None,
    image_sizes: Sequence[tuple[int, int] | None] | None = None,
    fisheye: Sequence[bool] | None = None,
    ref_cam: int | None = None,
) -> list[PinholeCamera]:
    """Initialise multi-camera extrinsics via pairwise Umeyama alignment of 3D skeletons."""
    n_cam = len(intrinsics)
    p3 = np.asarray(poses_3d_cam, dtype=float)
    if n_cam < 2 or p3.ndim != 4 or p3.shape[0] != n_cam or p3.shape[3] != 3:
        raise ValueError("poses_3d_cam must have shape (C >= 2, T, J, 3)")
    flat = p3.reshape(n_cam, -1, 3)
    valid = np.isfinite(flat).all(axis=-1)

    residuals = np.full((n_cam, n_cam), np.inf)
    transforms: dict[tuple[int, int], tuple[float, np.ndarray, np.ndarray]] = {}
    for ref in range(n_cam):
        residuals[ref, ref] = 0.0
        transforms[ref, ref] = (1.0, np.eye(3), np.zeros(3))
        for c in range(n_cam):
            if c == ref:
                continue
            shared = np.flatnonzero(valid[ref] & valid[c])
            if len(shared) < 6:
                continue
            sc, rot, tr = umeyama_alignment(flat[ref, shared], flat[c, shared])
            pred = sc * (flat[ref, shared] @ rot.T) + tr
            residuals[ref, c] = float(
                np.mean(np.linalg.norm(pred - flat[c, shared], axis=1))
            )
            transforms[ref, c] = (sc, rot, tr)

    chosen = (
        int(np.argmin(np.mean(residuals, axis=1))) if ref_cam is None else int(ref_cam)
    )
    if not np.isfinite(residuals[chosen]).all():
        raise ValueError(
            "Cameras lack sufficient co-visible 3D joints for initialisation"
        )

    names = (
        [f"cam{c + 1:02d}" for c in range(n_cam)]
        if camera_names is None
        else list(camera_names)
    )
    dists = [() for _ in range(n_cam)] if distortions is None else list(distortions)
    sizes = [None for _ in range(n_cam)] if image_sizes is None else list(image_sizes)
    fish = [False for _ in range(n_cam)] if fisheye is None else list(fisheye)
    return [
        PinholeCamera(
            intrinsic=np.asarray(intrinsics[c], dtype=float),
            rotation=transforms[chosen, c][1],
            translation=transforms[chosen, c][2],
            calibration_id=names[c],
            distortion=dists[c],
            image_size=sizes[c],
            fisheye=fish[c],
        )
        for c in range(n_cam)
    ]


def drop_outlier_frames(
    cameras: Sequence[PinholeCamera],
    pixels_2d: np.ndarray,
    confidences: np.ndarray | None = None,
    *,
    abs_threshold_px: float = 50.0,
    median_factor: float = 5.0,
) -> np.ndarray:
    """Return a boolean ``(C, T)`` mask of outlier frames exceeding abs and median MRE."""
    tri = triangulate_sequence(
        cameras, pixels_2d, confidences=confidences, max_reprojection_px=1e6
    )
    with np.errstate(invalid="ignore"):
        frame_err = np.nanmean(tri.reprojection_px, axis=2)
        cam_med = np.nanmedian(frame_err, axis=1, keepdims=True)
    limit = np.maximum(
        abs_threshold_px, median_factor * np.where(np.isfinite(cam_med), cam_med, 0.0)
    )
    return np.isfinite(frame_err) & (frame_err > limit)


def _ba_problem(
    cameras: Sequence[PinholeCamera],
    uv_undist: np.ndarray,
    weights: np.ndarray,
    x_init: np.ndarray,
    bones: np.ndarray | None,
    bone_weight: float,
    max_nfev: int,
) -> tuple[list[PinholeCamera], np.ndarray]:
    n_cam = len(cameras)
    n_frames, n_joints, _ = x_init.shape
    n_pts = n_frames * n_joints
    k_mats = np.array([c.intrinsic for c in cameras])
    r0, t0 = cameras[0].rotation, cameras[0].translation

    obs_mask = (
        np.isfinite(uv_undist).all(axis=-1)
        & (weights > 0)
        & np.isfinite(x_init).all(axis=-1)[None, :, :]
    )
    obs_c, obs_t, obs_j = np.nonzero(obs_mask)
    obs_p = obs_t * n_joints + obs_j
    obs_uv = uv_undist[obs_c, obs_t, obs_j]
    obs_w = weights[obs_c, obs_t, obs_j]
    n_obs = len(obs_c)

    bone_pairs = (
        np.empty((0, 2), dtype=int)
        if bones is None
        else np.asarray(bones, dtype=int).reshape(-1, 2)
    )
    valid_bones: list[tuple[int, int, np.ndarray]] = []
    if len(bone_pairs) and bone_weight > 0:
        for j1, j2 in bone_pairs:
            t_ok = np.flatnonzero(
                np.isfinite(x_init[:, j1]).all(axis=1)
                & np.isfinite(x_init[:, j2]).all(axis=1)
            )
            if len(t_ok) >= 2:
                valid_bones.append((int(j1), int(j2), t_ok))
    n_bone_res = sum(len(t_ok) for _, _, t_ok in valid_bones)

    n_cam_vars = 6 * (n_cam - 1)
    theta0 = np.zeros(n_cam_vars + 3 * n_pts)
    for c in range(1, n_cam):
        base = 6 * (c - 1)
        theta0[base : base + 3] = Rotation.from_matrix(cameras[c].rotation).as_rotvec()
        theta0[base + 3 : base + 6] = cameras[c].translation
    x_filled = np.where(np.isfinite(x_init), x_init, np.nanmedian(x_init, axis=(0, 1)))
    theta0[n_cam_vars:] = x_filled.ravel()

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rots = np.empty((n_cam, 3, 3))
        trans = np.empty((n_cam, 3))
        rots[0], trans[0] = r0, t0
        for c in range(1, n_cam):
            base = 6 * (c - 1)
            rots[c] = Rotation.from_rotvec(theta[base : base + 3]).as_matrix()
            trans[c] = theta[base + 3 : base + 6]
        return rots, trans, theta[n_cam_vars:].reshape(n_pts, 3)

    def fun(theta: np.ndarray) -> np.ndarray:
        rots, trans, pts = unpack(theta)
        xc = np.einsum("kij,kj->ki", rots[obs_c], pts[obs_p]) + trans[obs_c]
        z = np.maximum(xc[:, 2], 1e-6)
        k = k_mats[obs_c]
        u_hat = k[:, 0, 0] * (xc[:, 0] / z) + k[:, 0, 1] * (xc[:, 1] / z) + k[:, 0, 2]
        v_hat = k[:, 1, 1] * (xc[:, 1] / z) + k[:, 1, 2]
        res_reproj = np.empty(2 * n_obs)
        res_reproj[0::2] = obs_w * (obs_uv[:, 0] - u_hat)
        res_reproj[1::2] = obs_w * (obs_uv[:, 1] - v_hat)
        if not n_bone_res:
            return res_reproj
        pts_tj = pts.reshape(n_frames, n_joints, 3)
        res_b = []
        for j1, j2, t_ok in valid_bones:
            lengths = np.linalg.norm(pts_tj[t_ok, j1] - pts_tj[t_ok, j2], axis=1)
            res_b.append(bone_weight * (lengths - lengths.mean()))
        return np.concatenate((res_reproj, *res_b))

    def jac(theta: np.ndarray) -> sparse.csc_matrix:
        rots, trans, pts = unpack(theta)
        j_left = np.zeros((n_cam, 3, 3))
        for c in range(1, n_cam):
            j_left[c] = _so3_left_jacobian(theta[6 * (c - 1) : 6 * (c - 1) + 3])

        xc = np.einsum("kij,kj->ki", rots[obs_c], pts[obs_p]) + trans[obs_c]
        z = np.maximum(xc[:, 2], 1e-6)
        inv_z, inv_z2 = 1.0 / z, 1.0 / (z * z)
        k = k_mats[obs_c]
        fx, s_skew, fy = k[:, 0, 0], k[:, 0, 1], k[:, 1, 1]

        g = np.zeros((n_obs, 2, 3))
        g[:, 0, 0] = -obs_w * fx * inv_z
        g[:, 0, 1] = -obs_w * s_skew * inv_z
        g[:, 0, 2] = obs_w * (fx * xc[:, 0] + s_skew * xc[:, 1]) * inv_z2
        g[:, 1, 1] = -obs_w * fy * inv_z
        g[:, 1, 2] = obs_w * fy * xc[:, 1] * inv_z2

        rows_l, cols_l, vals_l = [], [], []
        g_pt = np.einsum("kab,kbc->kac", g, rots[obs_c])
        r_idx = np.repeat(2 * np.arange(n_obs), 3)[:, None] + np.array([0, 1])[None, :]
        r_pair = np.stack(
            (
                np.repeat(2 * np.arange(n_obs), 3),
                np.repeat(2 * np.arange(n_obs) + 1, 3),
            ),
            axis=0,
        )
        p_cols = (n_cam_vars + 3 * obs_p[:, None] + np.arange(3)[None, :]).ravel()
        rows_l.extend((r_pair[0], r_pair[1]))
        cols_l.extend((p_cols, p_cols))
        vals_l.extend((g_pt[:, 0, :].ravel(), g_pt[:, 1, :].ravel()))

        moving = np.flatnonzero(obs_c > 0)
        if len(moving):
            rx = np.einsum("kij,kj->ki", rots[obs_c[moving]], pts[obs_p[moving]])
            skew_rx = np.zeros((len(moving), 3, 3))
            skew_rx[:, 0, 1], skew_rx[:, 0, 2] = -rx[:, 2], rx[:, 1]
            skew_rx[:, 1, 0], skew_rx[:, 1, 2] = rx[:, 2], -rx[:, 0]
            skew_rx[:, 2, 0], skew_rx[:, 2, 1] = -rx[:, 1], rx[:, 0]
            d_rvec = -np.einsum(
                "kab,kbc,kcd->kad", g[moving], skew_rx, j_left[obs_c[moving]]
            )
            c_base = 6 * (obs_c[moving] - 1)
            r_m0 = np.repeat(2 * moving, 3)
            r_m1 = r_m0 + 1
            col_r = (c_base[:, None] + np.arange(3)[None, :]).ravel()
            col_t = (c_base[:, None] + 3 + np.arange(3)[None, :]).ravel()
            rows_l.extend((r_m0, r_m1, r_m0, r_m1))
            cols_l.extend((col_r, col_r, col_t, col_t))
            vals_l.extend(
                (
                    d_rvec[:, 0, :].ravel(),
                    d_rvec[:, 1, :].ravel(),
                    g[moving, 0, :].ravel(),
                    g[moving, 1, :].ravel(),
                )
            )

        offset = 2 * n_obs
        pts_tj = pts.reshape(n_frames, n_joints, 3)
        for j1, j2, t_ok in valid_bones:
            m = len(t_ok)
            diff = pts_tj[t_ok, j1] - pts_tj[t_ok, j2]
            unit = diff / np.maximum(np.linalg.norm(diff, axis=1, keepdims=True), 1e-9)
            for row_i, ti in enumerate(t_ok):
                r_num = offset + row_i
                for s_i, ts in enumerate(t_ok):
                    coeff = bone_weight * ((1.0 if s_i == row_i else 0.0) - 1.0 / m)
                    u_s = coeff * unit[s_i]
                    c1 = n_cam_vars + 3 * (ts * n_joints + j1) + np.arange(3)
                    c2 = n_cam_vars + 3 * (ts * n_joints + j2) + np.arange(3)
                    rows_l.append(np.full(6, r_num))
                    cols_l.append(np.concatenate((c1, c2)))
                    vals_l.append(np.concatenate((u_s, -u_s)))
            offset += m

        del r_idx
        return sparse.coo_matrix(
            (np.concatenate(vals_l), (np.concatenate(rows_l), np.concatenate(cols_l))),
            shape=(2 * n_obs + n_bone_res, len(theta)),
        ).tocsc()

    sol = least_squares(fun, theta0, jac=jac, method="trf", max_nfev=max_nfev)
    rots_f, trans_f, pts_f = unpack(sol.x)
    refined = [
        PinholeCamera(
            cam.intrinsic,
            rots_f[c],
            trans_f[c],
            cam.calibration_id,
            cam.distortion,
            cam.image_size,
            cam.fisheye,
        )
        for c, cam in enumerate(cameras)
    ]
    return refined, pts_f.reshape(n_frames, n_joints, 3)


@dataclass(frozen=True)
class HumanCalibrationResult:
    """Calibrated cameras, triangulated 3D motion, and reference-free quality metrics."""

    cameras: tuple[PinholeCamera, ...]
    triangulation: SequenceTriangulationResult
    alignment: WalkFrameAlignment | None
    worst_camera_mre_mrad: float
    accepted: bool


def calibrate_cameras_from_motion(
    intrinsics: Sequence[np.ndarray],
    pixels_2d: np.ndarray,
    poses_3d: np.ndarray,
    *,
    confidences: np.ndarray | None = None,
    poses_are_root_relative: bool = False,
    bones: np.ndarray | Sequence[tuple[int, int]] | None = None,
    camera_names: Sequence[str] | None = None,
    distortions: Sequence[Sequence[float]] | None = None,
    image_sizes: Sequence[tuple[int, int] | None] | None = None,
    fisheye: Sequence[bool] | None = None,
    frame_budget: int = 80,
    bone_weight: float = 5.0,
    max_nfev: int = 80,
    height_m: float | None = None,
    head_indices: Sequence[int] = (),
    sole_indices: Sequence[int] = (),
    left_heel_index: int | None = None,
    right_heel_index: int | None = None,
    head_ratio: float = DEFAULT_HEAD_BAND_RATIO,
    max_mre_mrad: float = MAX_ACCEPT_MRE_MRAD,
) -> HumanCalibrationResult:
    """Calibrate multi-camera extrinsics, metric scale and gravity from walking poses."""
    n_cam = len(intrinsics)
    uv_raw = np.asarray(pixels_2d, dtype=float)
    p3_raw = np.asarray(poses_3d, dtype=float)
    conf = (
        np.ones(uv_raw.shape[:3])
        if confidences is None
        else np.asarray(confidences, dtype=float)
    )
    dists = [() for _ in range(n_cam)] if distortions is None else list(distortions)
    fish = [False for _ in range(n_cam)] if fisheye is None else list(fisheye)

    # Undistort pixels once before placement, linear Procrustes init, and BA.
    temp_cams = [
        PinholeCamera(
            np.asarray(intrinsics[c], float),
            np.eye(3),
            np.zeros(3),
            f"c{c}",
            dists[c],
            None,
            fish[c],
        )
        for c in range(n_cam)
    ]
    uv = np.array(
        [
            temp_cams[c].undistort(uv_raw[c]) if dists[c] else uv_raw[c]
            for c in range(n_cam)
        ]
    )
    if poses_are_root_relative:
        p3_cam = np.array(
            [
                place_skeleton_in_camera(
                    p3_raw[c], uv[c], temp_cams[c].intrinsic, confidences=conf[c]
                )
                for c in range(n_cam)
            ]
        )
    else:
        p3_cam = p3_raw.copy()

    cams = init_extrinsics_procrustes(
        intrinsics,
        p3_cam,
        camera_names=camera_names,
        distortions=dists,
        image_sizes=image_sizes,
        fisheye=fish,
    )
    outliers = drop_outlier_frames(
        cams, uv, conf, abs_threshold_px=50.0, median_factor=5.0
    )
    if outliers.any():
        p3_clean, uv_clean, conf_clean = p3_cam.copy(), uv.copy(), conf.copy()
        p3_clean[outliers] = np.nan
        uv_clean[outliers] = np.nan
        conf_clean[outliers] = 0.0
        cams = init_extrinsics_procrustes(
            intrinsics,
            p3_clean,
            camera_names=camera_names,
            distortions=dists,
            image_sizes=image_sizes,
            fisheye=fish,
        )
    else:
        uv_clean, conf_clean = uv, conf

    n_frames = uv_clean.shape[1]
    stride = max(1, n_frames // max(1, frame_budget))
    sub_idx = np.arange(0, n_frames, stride)
    tri0 = triangulate_sequence(
        cams,
        uv_clean[:, sub_idx],
        confidences=conf_clean[:, sub_idx],
        undistort=False,
        max_reprojection_px=1e5,
    )
    cams_p1, pts_p1 = _ba_problem(
        cams,
        uv_clean[:, sub_idx],
        conf_clean[:, sub_idx],
        tri0.positions,
        None if bones is None else np.asarray(bones),
        bone_weight,
        max_nfev,
    )

    # Pass 2: drop sampled frames whose reprojection error exceeds 2x median (HumanCalib).
    tri1 = triangulate_sequence(
        cams_p1,
        uv_clean[:, sub_idx],
        confidences=conf_clean[:, sub_idx],
        undistort=False,
        max_reprojection_px=1e5,
    )
    with np.errstate(invalid="ignore"):
        f_err = np.nanmean(tri1.reprojection_px, axis=(0, 2))
        med = float(np.nanmedian(f_err))
    keep = np.isfinite(f_err) & (f_err <= max(2.0 * med, 1.0))
    sub2 = sub_idx[keep] if int(keep.sum()) >= 5 else sub_idx
    x2 = pts_p1[keep] if int(keep.sum()) >= 5 else pts_p1
    cams_f, _ = _ba_problem(
        cams_p1,
        uv_clean[:, sub2],
        conf_clean[:, sub2],
        x2,
        None if bones is None else np.asarray(bones),
        bone_weight,
        max_nfev,
    )

    alignment: WalkFrameAlignment | None = None
    tri_full = triangulate_sequence(
        cams_f,
        uv_clean,
        confidences=conf_clean,
        undistort=False,
        max_reprojection_px=50.0,
    )
    if (
        height_m is not None
        and head_indices
        and sole_indices
        and left_heel_index is not None
        and right_heel_index is not None
    ):
        alignment = align_walk_frame(
            tri_full.positions,
            height_m,
            head_indices=head_indices,
            sole_indices=sole_indices,
            left_heel_index=left_heel_index,
            right_heel_index=right_heel_index,
            head_ratio=head_ratio,
            cameras=cams_f,
        )
        cams_f = list(alignment.cameras)
        tri_full = triangulate_sequence(
            cams_f,
            uv_clean,
            confidences=conf_clean,
            undistort=False,
            max_reprojection_px=50.0,
        )

    worst_mrad = float(np.nanmax(tri_full.camera_mre_mrad))
    accepted = bool(np.isfinite(worst_mrad) and worst_mrad <= max_mre_mrad)
    return HumanCalibrationResult(
        tuple(cams_f), tri_full, alignment, worst_mrad, accepted
    )
