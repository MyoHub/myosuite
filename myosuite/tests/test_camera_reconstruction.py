"""Calibrated reconstruction recovers known metric points and rejects ambiguity."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from myosuite.biomechanics.camera import PinholeCamera, lift_depth, triangulate_point

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
