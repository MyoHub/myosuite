"""Metric reconstruction from calibrated, undistorted camera observations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PinholeCamera:
    """Calibrated camera with optical +Z forward and world coordinates in metres.

    Args:
        intrinsic: 3x3 pixel intrinsic matrix, after image undistortion.
        rotation: World-to-camera proper rotation matrix.
        translation: World-to-camera translation in metres.
        calibration_id: Identifier of the measured camera calibration.

    Pixels must already be undistorted with the matching intrinsic matrix.
    Resized/cropped images require correspondingly adjusted intrinsics.
    """

    intrinsic: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    calibration_id: str

    def __post_init__(self) -> None:
        for name, shape in (
            ("intrinsic", (3, 3)),
            ("rotation", (3, 3)),
            ("translation", (3,)),
        ):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name))
        k, r = self.intrinsic, self.rotation
        if (
            k[0, 0] <= 0
            or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1])
            or not np.allclose(k[1, 0], 0)
        ):
            raise ValueError("intrinsic must be a calibrated pinhole matrix")
        if not np.allclose(r @ r.T, np.eye(3), atol=1e-7) or not np.isclose(
            np.linalg.det(r), 1, atol=1e-7
        ):
            raise ValueError("rotation must be orthonormal with determinant +1")
        if not self.calibration_id:
            raise ValueError("calibration_id is required")

    @property
    def projection(self) -> np.ndarray:
        """Return the world-to-pixel 3x4 projection matrix."""
        return self.intrinsic @ np.column_stack((self.rotation, self.translation))

    @property
    def center(self) -> np.ndarray:
        """Return the optical centre in world metres."""
        return -self.rotation.T @ self.translation

    def project(self, world: np.ndarray) -> np.ndarray:
        """Project (..., 3) world points; reject points behind the camera."""
        points = np.asarray(world, dtype=float)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("world must contain finite 3D points")
        camera = points @ self.rotation.T + self.translation
        if np.any(camera[..., 2] <= 0):
            raise ValueError("points must be in front of the camera")
        pixel = camera @ self.intrinsic.T
        return pixel[..., :2] / pixel[..., 2:3]

    def ray(self, pixel: np.ndarray) -> np.ndarray:
        """Return a world-frame unit ray for one undistorted pixel."""
        uv = _array(pixel, (2,), "pixel")
        ray = self.rotation.T @ np.linalg.solve(self.intrinsic, [*uv, 1.0])
        return ray / np.linalg.norm(ray)

    def record(self) -> dict:
        """Return all metric calibration parameters for reproducible reconstruction."""
        return {
            "intrinsic": self.intrinsic.tolist(),
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "calibration_id": self.calibration_id,
        }


@dataclass(frozen=True)
class TriangulationResult:
    """A reconstructed point and diagnostics for every contributing camera."""

    position: np.ndarray
    reprojection_errors: np.ndarray
    ray_angle_deg: float


def triangulate_point(
    cameras: Sequence[PinholeCamera],
    pixels: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    max_reprojection_px: float = 5.0,
    min_ray_angle_deg: float = 1.0,
) -> TriangulationResult:
    """Triangulate simultaneous observations of the same anatomical landmark.

    Args:
        cameras: At least two cameras calibrated in the same metric world.
        pixels: Undistorted pixels, shape (number of cameras, 2).
        weights: Positive observation weights; defaults to equal weights.
        max_reprojection_px: Reject a point exceeding this error in any view.
        min_ray_angle_deg: Reject insufficient parallax, including parallel rays.

    Returns:
        The position, reprojection errors and largest triangulation angle.

    Raises:
        ValueError: Invalid, degenerate, behind-camera or inconsistent data.
    """
    count = len(cameras)
    if count < 2:
        raise ValueError("triangulation requires at least two calibrated cameras")
    uv = _array(pixels, (count, 2), "pixels")
    w = np.ones(count) if weights is None else _array(weights, (count,), "weights")
    if np.any(w <= 0):
        raise ValueError("weights must be positive")
    if not (
        np.isfinite(max_reprojection_px)
        and max_reprojection_px > 0
        and 0 < min_ray_angle_deg < 90
    ):
        raise ValueError("invalid reconstruction thresholds")
    rays = np.array([camera.ray(p) for camera, p in zip(cameras, uv)])
    dots = np.abs(np.clip(rays @ rays.T, -1.0, 1.0))
    angle = float(np.degrees(np.arccos(dots)).max())
    if angle < min_ray_angle_deg:
        raise ValueError("insufficient camera parallax")
    rows = []
    for camera, (u, v), weight in zip(cameras, uv, w):
        p = camera.projection
        rows.extend(
            (np.sqrt(weight) * (u * p[2] - p[0]), np.sqrt(weight) * (v * p[2] - p[1]))
        )
    _, singular, vt = np.linalg.svd(np.asarray(rows))
    point = vt[-1]
    if abs(point[3]) < 1e-12 or singular[-2] < 1e-12:
        raise ValueError("degenerate camera geometry")
    world = point[:3] / point[3]
    errors = np.array(
        [np.linalg.norm(c.project(world) - p) for c, p in zip(cameras, uv)]
    )
    if errors.max() > max_reprojection_px:
        raise ValueError("reprojection error exceeds threshold")
    return TriangulationResult(world, errors, angle)


def lift_depth(camera: PinholeCamera, pixel: np.ndarray, depth_m: float) -> np.ndarray:
    """Lift an undistorted pixel using calibrated optical-axis Z depth in metres.

    Args:
        camera: Calibration for the pixel/depth image pair.
        pixel: Undistorted pixel of a landmark, shape (2,).
        depth_m: Positive metric camera-Z depth, not ray range or relative depth.

    Returns:
        The landmark in metric world coordinates.
    """
    uv = _array(pixel, (2,), "pixel")
    if not np.isfinite(depth_m) or depth_m <= 0:
        raise ValueError("depth_m must be positive calibrated metric Z depth")
    camera_point = np.linalg.solve(camera.intrinsic, [*uv, 1.0]) * depth_m
    return camera.rotation.T @ (camera_point - camera.translation)
