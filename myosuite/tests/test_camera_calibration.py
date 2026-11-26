"""Unit tests for targetless multi-camera human-motion calibration."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from myosuite.biomechanics.camera import DEFAULT_HEAD_BAND_RATIO, PinholeCamera
from myosuite.biomechanics.camera_calibration import (
    _so3_left_jacobian,
    calibrate_cameras_from_motion,
    place_skeleton_in_camera,
    select_walking_subject,
)

pytestmark = pytest.mark.tier2


def _synthetic_walk(
    n_frames: int = 30, height: float = 1.78
) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 2.0, n_frames)
    phase = 2.0 * np.pi * t
    # 6 joints: 0=head, 1=l_hip, 2=r_hip, 3=l_ankle/heel, 4=r_ankle/heel, 5=neck
    pts = np.zeros((n_frames, 6, 3))
    x = 1.1 * t - 1.1
    l_sw = np.maximum(0.0, np.sin(phase))
    r_sw = np.maximum(0.0, -np.sin(phase))
    pts[:, 0] = np.column_stack(
        (x + 0.08, np.zeros_like(t), np.full_like(t, DEFAULT_HEAD_BAND_RATIO * height))
    )
    pts[:, 1] = np.column_stack((x, np.full_like(t, 0.11), np.full_like(t, 0.92)))
    pts[:, 2] = np.column_stack((x, np.full_like(t, -0.11), np.full_like(t, 0.92)))
    pts[:, 3] = np.column_stack(
        (x + 0.22 * np.sin(phase), np.full_like(t, 0.11), 0.05 * l_sw)
    )
    pts[:, 4] = np.column_stack(
        (x - 0.22 * np.sin(phase), np.full_like(t, -0.11), 0.05 * r_sw)
    )
    pts[:, 5] = np.column_stack((x + 0.06, np.zeros_like(t), np.full_like(t, 1.45)))
    bones = np.array([[1, 2], [1, 3], [2, 4], [1, 5], [5, 0]], dtype=int)
    return pts, bones


def _four_camera_rig() -> list[PinholeCamera]:
    k = np.array([[920.0, 0.0, 640.0], [0.0, 918.0, 360.0], [0.0, 0.0, 1.0]])
    centers = [
        np.array([-2.8, -2.6, 1.6]),
        np.array([2.8, -2.5, 1.7]),
        np.array([2.7, 2.7, 1.5]),
        np.array([-2.6, 2.8, 1.8]),
    ]
    cams = []
    for i, c in enumerate(centers):
        fwd = np.array([0.0, 0.0, 0.9]) - c
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        r_w2c = np.vstack((right, down, fwd))
        cams.append(
            PinholeCamera(
                k,
                r_w2c,
                -r_w2c @ c,
                f"cam0{i + 1}",
                distortion=(-0.06, 0.015, 0.0002, -0.0001),
                image_size=(1280, 720),
            )
        )
    return cams


def test_so3_left_jacobian_matches_finite_differences() -> None:
    rvec = np.array([0.42, -0.31, 0.67])
    pt = np.array([0.7, -1.2, 2.3])
    r0 = Rotation.from_rotvec(rvec).as_matrix()
    rx = r0 @ pt
    skew_rx = np.array(
        [[0.0, -rx[2], rx[1]], [rx[2], 0.0, -rx[0]], [-rx[1], rx[0], 0.0]]
    )
    analytic = -skew_rx @ _so3_left_jacobian(rvec)

    eps = 1e-7
    numeric = np.zeros((3, 3))
    for i in range(3):
        step = np.zeros(3)
        step[i] = eps
        rp = Rotation.from_rotvec(rvec + step).as_matrix() @ pt
        rm = Rotation.from_rotvec(rvec - step).as_matrix() @ pt
        numeric[:, i] = (rp - rm) / (2.0 * eps)
    np.testing.assert_allclose(analytic, numeric, atol=1e-8)


def test_place_skeleton_in_camera_and_walking_subject_selection() -> None:
    cams = _four_camera_rig()
    walk, _ = _synthetic_walk(n_frames=12)
    cam = cams[0]
    cam_pts = walk @ cam.rotation.T + cam.translation
    root = 0.5 * (cam_pts[:, 1:2] + cam_pts[:, 2:3])
    rel = cam_pts - root
    uv = cam.project(walk)
    placed = place_skeleton_in_camera(rel, uv, cam.intrinsic)
    np.testing.assert_allclose(placed, cam_pts, atol=1e-9)

    # Add a foreground stationary bystander closer to cam0 (larger 2D bbox).
    bystander_w = np.tile(walk[0], (12, 1, 1)) * 0.9
    bystander_w[:, :, 0] = -1.6
    bystander_w[:, :, 1] = -1.5
    c2d, c3d, ccf = [], [], []
    for c in cams[:2]:
        frames_2d, frames_3d, frames_cf = [], [], []
        for t in range(12):
            w_2d = c.project(walk[t])
            b_2d = c.project(bystander_w[t])
            w_3d = walk[t] @ c.rotation.T + c.translation
            b_3d = bystander_w[t] @ c.rotation.T + c.translation
            # Put bystander first so naive index-0 or largest-box selection would fail.
            frames_2d.append(np.stack((b_2d, w_2d), axis=0))
            frames_3d.append(np.stack((b_3d, w_3d), axis=0))
            frames_cf.append(np.ones((2, 6)))
        c2d.append(frames_2d)
        c3d.append(frames_3d)
        ccf.append(frames_cf)

    sel_2d, sel_3d, _ = select_walking_subject(
        c2d, c3d, ccf, hip_indices=(1, 2), ankle_indices=(3, 4), min_swing_speed=0.05
    )
    valid_t = np.isfinite(sel_2d[0, :, 0, 0])
    assert valid_t.sum() >= 10
    expected_cam0 = walk[valid_t] @ cams[0].rotation.T + cams[0].translation
    np.testing.assert_allclose(sel_3d[0, valid_t], expected_cam0, atol=1e-9)


def test_calibrate_cameras_from_root_relative_motion_with_outliers() -> None:
    rng = np.random.default_rng(12)
    height = 1.78
    walk, bones = _synthetic_walk(n_frames=32, height=height)
    gt_cams = _four_camera_rig()

    pixels_2d = np.stack([c.project(walk, distort=True) for c in gt_cams], axis=0)
    pixels_2d += rng.normal(0.0, 0.35, size=pixels_2d.shape)

    # Root-relative lifted 3D skeletons with noisy per-camera scale & depth jitter.
    poses_rel = np.empty((len(gt_cams), len(walk), walk.shape[1], 3))
    for i, c in enumerate(gt_cams):
        cam_pts = walk @ c.rotation.T + c.translation
        pelvis = 0.5 * (cam_pts[:, 1:2] + cam_pts[:, 2:3])
        poses_rel[i] = 0.83 * (cam_pts - pelvis) + rng.normal(
            0.0, 0.006, size=cam_pts.shape
        )

    # Inject a corrupted outlier frame in camera 2 (> 150 px jump).
    pixels_2d[2, 9] += 180.0

    result = calibrate_cameras_from_motion(
        [c.intrinsic for c in gt_cams],
        pixels_2d,
        poses_rel,
        poses_are_root_relative=True,
        bones=bones,
        camera_names=[c.calibration_id for c in gt_cams],
        distortions=[c.distortion for c in gt_cams],
        image_sizes=[c.image_size for c in gt_cams],
        height_m=height,
        head_indices=[0],
        sole_indices=[3, 4],
        left_heel_index=3,
        right_heel_index=4,
    )
    assert result.accepted
    assert result.worst_camera_mre_mrad < 5.0

    # Pairwise relative camera rotations match ground truth within 0.5 degrees.
    for i in range(len(gt_cams)):
        for j in range(i + 1, len(gt_cams)):
            r_gt = gt_cams[i].rotation @ gt_cams[j].rotation.T
            r_est = result.cameras[i].rotation @ result.cameras[j].rotation.T
            deg = (
                np.linalg.norm(Rotation.from_matrix(r_gt @ r_est.T).as_rotvec())
                * 180.0
                / np.pi
            )
            assert deg < 0.5

    # Pairwise camera baselines match ground-truth metric scale within 2%.
    b_gt = np.linalg.norm(gt_cams[1].center - gt_cams[0].center)
    b_est = np.linalg.norm(result.cameras[1].center - result.cameras[0].center)
    assert b_est == pytest.approx(b_gt, rel=0.02)
