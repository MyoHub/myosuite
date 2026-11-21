"""Calibrated reconstruction recovers known metric points and rejects ambiguity."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from myosuite.biomechanics.camera import (
    DEFAULT_HEAD_BAND_RATIO,
    LEG_STATURE_RATIO,
    TRUNK_STATURE_RATIO,
    PinholeCamera,
    align_walk_frame,
    angular_reprojection_mrad,
    distort_pixels,
    lift_depth,
    load_pose2sim_cameras,
    save_pose2sim_cameras,
    triangulate_point,
    triangulate_sequence,
    undistort_pixels,
)

pytestmark = pytest.mark.tier2


def cameras() -> list[PinholeCamera]:
    k = np.array([[800.0, 0, 320], [0, 800, 240], [0, 0, 1]])
    return [
        PinholeCamera(k, np.eye(3), np.array([x, 0.0, 0.0]), f"cal{x}")
        for x in (0.0, -0.7, 0.5)
    ]


def test_multiview_and_depth_recover_metric_geometry() -> None:
    views = cameras()
    truth = np.array([0.2, -0.3, 3.2])
    pixels = np.array([camera.project(truth) for camera in views])
    result = triangulate_point(views, pixels)
    np.testing.assert_allclose(result.position, truth, atol=1e-12)
    assert result.reprojection_errors.max() < 1e-10
    np.testing.assert_allclose(lift_depth(views[1], pixels[1], 3.2), truth)
    noisy = pixels + np.array([[0.4, -0.2], [-0.3, 0.3], [0.1, -0.1]])
    np.testing.assert_allclose(
        triangulate_point(views, noisy).position, truth, atol=0.01
    )


def test_geometry_rejects_low_parallax_bad_matches_and_negative_depth() -> None:
    views = cameras()[:2]
    truth = np.array([0.2, 0.3, 3.0])
    pixels = np.array([camera.project(truth) for camera in views])
    with pytest.raises(ValueError, match="parallax"):
        triangulate_point([views[0], views[0]], [pixels[0], pixels[0]])
    mismatch = pixels.copy()
    mismatch[1, 1] += 100
    with pytest.raises(ValueError, match="reprojection"):
        triangulate_point(views, mismatch)
    behind = np.array([[320.0, 240.0], [500.0, 240.0]])
    with pytest.raises(ValueError, match="front"):
        triangulate_point(views, behind)
    for depth in (-1, 0, float("nan")):
        with pytest.raises(ValueError, match="depth"):
            lift_depth(views[0], pixels[0], depth)


@pytest.mark.parametrize("rotation", [np.diag([1, 1, -1]), np.eye(3) * 2])
def test_camera_rejects_invalid_calibration(rotation: np.ndarray) -> None:
    with pytest.raises(ValueError, match="rotation"):
        PinholeCamera(np.eye(3), rotation, np.zeros(3), "bad")


def test_rotated_camera_depth_uses_optical_z_and_roundtrips_calibration() -> None:
    first = cameras()[0]
    rotation = Rotation.from_euler("y", 20, degrees=True).as_matrix()
    center = np.array([1.0, 0.2, -0.1])
    camera = PinholeCamera(first.intrinsic, rotation, -rotation @ center, "rotated")
    truth = np.array([0.1, -0.2, 3.2])
    depth = (rotation @ truth + camera.translation)[2]
    pixel = camera.project(truth)
    np.testing.assert_allclose(lift_depth(camera, pixel, depth), truth, atol=1e-12)
    clone = PinholeCamera(**camera.record())
    np.testing.assert_allclose(clone.project(truth), pixel)


def test_brown_conrady_and_fisheye_undistortion_roundtrip() -> None:
    k = np.array([[1050.0, 0.0, 960.0], [0.0, 1048.0, 540.0], [0.0, 0.0, 1.0]])
    grid_u, grid_v = np.meshgrid(
        np.linspace(200.0, 1720.0, 6), np.linspace(120.0, 960.0, 5)
    )
    ideal = np.stack((grid_u, grid_v), axis=-1)

    bc = (-0.18, 0.045, 0.0008, -0.0005, -0.006)
    distorted_bc = distort_pixels(ideal, k, bc, fisheye=False)
    assert np.linalg.norm(distorted_bc - ideal, axis=-1).max() > 15.0
    recovered_bc = undistort_pixels(distorted_bc, k, bc, fisheye=False)
    np.testing.assert_allclose(recovered_bc, ideal, atol=1e-9)

    fe = (-0.03, 0.012, -0.004, 0.001)
    distorted_fe = distort_pixels(ideal, k, fe, fisheye=True)
    recovered_fe = undistort_pixels(distorted_fe, k, fe, fisheye=True)
    np.testing.assert_allclose(recovered_fe, ideal, atol=1e-9)


def test_pose2sim_toml_roundtrip_and_distorted_sequence_triangulation(
    tmp_path: Path,
) -> None:
    k = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    rig = [
        PinholeCamera(
            k,
            Rotation.from_euler("y", deg, degrees=True).as_matrix(),
            np.array([tx, 0.1, 3.5]),
            f"cam0{i + 1}",
            distortion=(-0.09, 0.02, 0.0003, -0.0002),
            image_size=(1280, 720),
        )
        for i, (deg, tx) in enumerate(((-22.0, -0.8), (0.0, 0.0), (24.0, 0.9)))
    ]
    toml_path = save_pose2sim_cameras(tmp_path / "Calib_scene.toml", rig)
    loaded = load_pose2sim_cameras(toml_path)
    assert [c.calibration_id for c in loaded] == ["cam01", "cam02", "cam03"]
    for src, dst in zip(rig, loaded):
        np.testing.assert_allclose(dst.intrinsic, src.intrinsic, atol=1e-8)
        np.testing.assert_allclose(dst.rotation, src.rotation, atol=1e-8)
        np.testing.assert_allclose(dst.translation, src.translation, atol=1e-8)
        assert dst.image_size == (1280, 720)

    t = np.linspace(0.0, 1.0, 8)
    truth = np.stack(
        [
            np.column_stack((0.3 * t - 0.15, np.full_like(t, -0.2), np.zeros_like(t))),
            np.column_stack(
                (0.3 * t - 0.15, np.full_like(t, 0.4), 0.1 * np.sin(4 * t))
            ),
        ],
        axis=1,
    )
    obs = np.stack([c.project(truth, distort=True) for c in loaded], axis=0)
    conf = np.ones((3, 8, 2))
    obs[2, 3, 0] = np.nan
    conf[0, 5, 1] = 0.01
    conf[1, 5, 1] = 0.01

    seq = triangulate_sequence(loaded, obs, confidences=conf, min_confidence=0.1)
    assert np.isnan(seq.positions[5, 1]).all()
    mask = np.ones((8, 2), dtype=bool)
    mask[5, 1] = False
    np.testing.assert_allclose(seq.positions[mask], truth[mask], atol=1e-7)
    assert np.nanmax(seq.camera_mre_px) < 1e-6
    assert np.nanmax(seq.camera_mre_mrad) < 1e-5

    shifted = loaded[0].project(truth[0, 0]) + np.array([9.0, 0.0])
    mrad = angular_reprojection_mrad(loaded[0], truth[0, 0], shifted)
    assert mrad == pytest.approx(10.0, rel=0.08)


def test_align_walk_frame_cancels_trunk_lean_scales_stature_and_updates_cameras() -> (
    None
):
    rng = np.random.default_rng(7)
    height = 1.80
    n_frames = 40
    x_prog = np.linspace(0.0, 2.4, n_frames)
    phase = np.linspace(0.0, 4.0 * np.pi, n_frames)

    # 10 landmarks: 0=head_band, 1=l_heel, 2=r_heel, 3=l_ankle, 4=r_ankle,
    # 5=l_knee, 6=r_knee, 7=l_hip, 8=r_hip, 9=neck
    pts_true = np.zeros((n_frames, 10, 3))
    lean_x = 0.14  # ~8.5 deg forward trunk lean along +X walking axis
    pts_true[:, 0] = np.column_stack(
        (
            x_prog + lean_x,
            np.zeros(n_frames),
            np.full(n_frames, DEFAULT_HEAD_BAND_RATIO * height),
        )
    )
    l_swing = np.maximum(0.0, np.sin(phase))
    r_swing = np.maximum(0.0, -np.sin(phase))
    l_x = np.cumsum(0.12 * l_swing)
    r_x = np.cumsum(0.12 * r_swing)
    l_x *= 2.4 / l_x[-1]
    r_x *= 2.4 / r_x[-1]
    pts_true[:, 1] = np.column_stack((l_x, np.full(n_frames, 0.10), 0.04 * l_swing))
    pts_true[:, 2] = np.column_stack((r_x, np.full(n_frames, -0.10), 0.04 * r_swing))
    leg_len = LEG_STATURE_RATIO * height
    pts_true[:, 3] = pts_true[:, 1] + [0.0, 0.0, 0.07]
    pts_true[:, 4] = pts_true[:, 2] + [0.0, 0.0, 0.07]
    pts_true[:, 5] = pts_true[:, 3] + [0.0, 0.0, 0.5 * leg_len]
    pts_true[:, 6] = pts_true[:, 4] + [0.0, 0.0, 0.5 * leg_len]
    pts_true[:, 7] = pts_true[:, 5] + [0.0, 0.0, 0.5 * leg_len]
    pts_true[:, 8] = pts_true[:, 6] + [0.0, 0.0, 0.5 * leg_len]
    pts_true[:, 9] = pts_true[:, 7] + [0.0, 0.0, TRUNK_STATURE_RATIO * height]

    # Apply an arbitrary non-metric similarity transform (tilted camera frame, scale 0.37).
    arbitrary_scale = 0.37
    arbitrary_rot = Rotation.from_euler(
        "xyz", [37.0, -24.0, 61.0], degrees=True
    ).as_matrix()
    arbitrary_shift = rng.uniform(-2.0, 2.0, size=3)
    pts_raw = arbitrary_scale * (pts_true @ arbitrary_rot.T) + arbitrary_shift

    k = np.array([[850.0, 0.0, 640.0], [0.0, 850.0, 360.0], [0.0, 0.0, 1.0]])
    cam_centers_raw = [
        arbitrary_scale * (np.array([1.2, -3.2, 1.4]) @ arbitrary_rot.T)
        + arbitrary_shift,
        arbitrary_scale * (np.array([1.2, 3.2, 1.5]) @ arbitrary_rot.T)
        + arbitrary_shift,
    ]
    raw_cams = []
    target_raw = np.mean(pts_raw, axis=(0, 1))
    for i, c_pos in enumerate(cam_centers_raw):
        fwd = target_raw - c_pos
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, arbitrary_rot[:, 2])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        r_w2c = np.vstack((right, down, fwd))
        raw_cams.append(PinholeCamera(k, r_w2c, -r_w2c @ c_pos, f"raw{i}"))

    px_before = [c.project(pts_raw[10]) for c in raw_cams]

    aligned = align_walk_frame(
        pts_raw,
        height,
        head_indices=[0],
        sole_indices=[1, 2],
        left_heel_index=1,
        right_heel_index=2,
        vertical_method="walk",
        scale_method="stature",
        cameras=raw_cams,
    )
    assert aligned.scale == pytest.approx(1.0 / arbitrary_scale, rel=0.015)
    true_z_in_raw = arbitrary_rot[:, 2]
    recovered_z_in_raw = aligned.rotation[2]
    tilt_deg = np.degrees(
        np.arccos(np.clip(abs(np.dot(true_z_in_raw, recovered_z_in_raw)), -1.0, 1.0))
    )
    assert tilt_deg < 0.5

    # Naive single-frame vertical without trunk-lean projection is ~5 deg tilted.
    frame_only = align_walk_frame(
        pts_raw,
        height,
        head_indices=[0],
        sole_indices=[1, 2],
        left_heel_index=1,
        right_heel_index=2,
        vertical_method="frame",
    )
    naive_tilt_deg = np.degrees(
        np.arccos(
            np.clip(abs(np.dot(true_z_in_raw, frame_only.rotation[2])), -1.0, 1.0)
        )
    )
    assert naive_tilt_deg > 4.0

    # Transformed cameras keep identical pixel projections and triangulate in metres.
    for cam_after, uv_expected in zip(aligned.cameras, px_before):
        np.testing.assert_allclose(
            cam_after.project(aligned.positions[10]), uv_expected, atol=1e-9
        )
    tri = triangulate_point(
        aligned.cameras, np.array([px_before[0][0], px_before[1][0]])
    )
    np.testing.assert_allclose(tri.position, aligned.positions[10, 0], atol=1e-9)

    # Segment-based anthropometric scaling (legs + trunk) also recovers metric scale.
    seg_aligned = align_walk_frame(
        pts_raw,
        height,
        head_indices=[0],
        sole_indices=[1, 2],
        left_heel_index=1,
        right_heel_index=2,
        leg_chains=((7, 5, 3), (8, 6, 4)),
        trunk_pair=(7, 9),
        scale_method="segments",
    )
    assert seg_aligned.scale == pytest.approx(1.0 / arbitrary_scale, rel=1e-5)
