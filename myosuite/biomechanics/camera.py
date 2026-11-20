# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Metric reconstruction, Pose2Sim calibration I/O, undistortion and walk-frame scaling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

LEG_STATURE_RATIO = 0.245 + 0.246
TRUNK_STATURE_RATIO = 0.288
DEFAULT_HEAD_BAND_RATIO = 0.9255


def _array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    array.setflags(write=False)
    return array


def _distortion_coeffs(
    distortion: Sequence[float] | np.ndarray, fisheye: bool
) -> np.ndarray:
    raw = np.asarray(distortion, dtype=float).ravel()
    if not np.isfinite(raw).all() or len(raw) > 8:
        raise ValueError("distortion must contain at most 8 finite coefficients")
    out = np.zeros(4 if fisheye else 5, dtype=float)
    out[: min(len(raw), len(out))] = raw[: len(out)]
    return out


def distort_pixels(
    pixels: np.ndarray,
    intrinsic: np.ndarray,
    distortion: Sequence[float] | np.ndarray,
    *,
    fisheye: bool = False,
) -> np.ndarray:
    """Apply radial-tangential or equidistant fisheye distortion to ``(..., 2)`` pixels."""
    uv = np.asarray(pixels, dtype=float)
    if uv.shape[-1:] != (2,):
        raise ValueError("pixels must have trailing dimension 2")
    k, d = (
        _array(intrinsic, (3, 3), "intrinsic"),
        _distortion_coeffs(distortion, fisheye),
    )
    if not np.any(d):
        return uv.copy()
    x, y = (uv[..., 0] - k[0, 2]) / k[0, 0], (uv[..., 1] - k[1, 2]) / k[1, 1]
    if fisheye:
        r = np.hypot(x, y)
        th = np.arctan(r)
        t2 = th * th
        th_d = th * (1.0 + t2 * (d[0] + t2 * (d[1] + t2 * (d[2] + t2 * d[3]))))
        s = np.where(r > 1e-12, th_d / np.where(r > 1e-12, r, 1.0), 1.0)
        xd, yd = x * s, y * s
    else:
        r2, xy = x * x + y * y, 2.0 * x * y
        rad = 1.0 + r2 * (d[0] + r2 * (d[1] + r2 * d[4]))
        xd = x * rad + d[2] * xy + d[3] * (r2 + 2.0 * x * x)
        yd = y * rad + d[2] * (r2 + 2.0 * y * y) + d[3] * xy
    return np.stack(
        (k[0, 0] * xd + k[0, 1] * yd + k[0, 2], k[1, 1] * yd + k[1, 2]), axis=-1
    )


def undistort_pixels(
    pixels: np.ndarray,
    intrinsic: np.ndarray,
    distortion: Sequence[float] | np.ndarray,
    *,
    fisheye: bool = False,
    iterations: int = 10,
) -> np.ndarray:
    """Invert radial-tangential or fisheye distortion on ``(..., 2)`` pixels."""
    uv = np.asarray(pixels, dtype=float)
    if uv.shape[-1:] != (2,):
        raise ValueError("pixels must have trailing dimension 2")
    k, d = (
        _array(intrinsic, (3, 3), "intrinsic"),
        _distortion_coeffs(distortion, fisheye),
    )
    if not np.any(d):
        return uv.copy()
    yd = (uv[..., 1] - k[1, 2]) / k[1, 1]
    xd = (uv[..., 0] - k[0, 2] - k[0, 1] * yd) / k[0, 0]
    if fisheye:
        th_d = np.hypot(xd, yd)
        th = th_d.copy()
        for _ in range(iterations):
            t2 = th * th
            poly = 1.0 + t2 * (d[0] + t2 * (d[1] + t2 * (d[2] + t2 * d[3])))
            der = 1.0 + t2 * (
                3.0 * d[0] + t2 * (5.0 * d[1] + t2 * (7.0 * d[2] + 9.0 * d[3] * t2))
            )
            th = th - (th * poly - th_d) / np.where(np.abs(der) > 1e-12, der, 1.0)
        s = np.where(th_d > 1e-12, np.tan(th) / np.where(th_d > 1e-12, th_d, 1.0), 1.0)
        x, y = xd * s, yd * s
    else:
        x, y = xd.copy(), yd.copy()
        for _ in range(iterations):
            r2 = x * x + y * y
            rad = 1.0 + r2 * (d[0] + r2 * (d[1] + r2 * d[4]))
            dr = 2.0 * (d[0] + r2 * (2.0 * d[1] + 3.0 * d[4] * r2))
            fx = x * rad + 2.0 * d[2] * x * y + d[3] * (r2 + 2.0 * x * x) - xd
            fy = y * rad + d[2] * (r2 + 2.0 * y * y) + 2.0 * d[3] * x * y - yd
            j00 = rad + x * x * dr + 2.0 * d[2] * y + 6.0 * d[3] * x
            j01 = x * y * dr + 2.0 * d[2] * x + 2.0 * d[3] * y
            j11 = rad + y * y * dr + 6.0 * d[2] * y + 2.0 * d[3] * x
            det = np.where(
                np.abs(j00 * j11 - j01 * j01) > 1e-12, j00 * j11 - j01 * j01, 1.0
            )
            x, y = x - (j11 * fx - j01 * fy) / det, y - (-j01 * fx + j00 * fy) / det
    return np.stack(
        (k[0, 0] * x + k[0, 1] * y + k[0, 2], k[1, 1] * y + k[1, 2]), axis=-1
    )


@dataclass(frozen=True)
class PinholeCamera:
    """Calibrated camera with optical +Z forward and world coordinates in metres."""

    intrinsic: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    calibration_id: str
    distortion: tuple[float, ...] = ()
    image_size: tuple[int, int] | None = None
    fisheye: bool = False

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
        if not np.allclose(r @ r.T, np.eye(3), atol=1e-6) or not np.isclose(
            np.linalg.det(r), 1, atol=1e-6
        ):
            raise ValueError("rotation must be orthonormal with determinant +1")
        if not self.calibration_id:
            raise ValueError("calibration_id is required")
        coeffs = tuple(float(c) for c in self.distortion)
        _distortion_coeffs(coeffs, self.fisheye)
        object.__setattr__(self, "distortion", coeffs)
        if self.image_size is not None:
            w, h = int(self.image_size[0]), int(self.image_size[1])
            if w <= 0 or h <= 0:
                raise ValueError("image_size must contain positive (width, height)")
            object.__setattr__(self, "image_size", (w, h))

    @property
    def projection(self) -> np.ndarray:
        """Return the world-to-pixel 3x4 projection matrix."""
        return self.intrinsic @ np.column_stack((self.rotation, self.translation))

    @property
    def center(self) -> np.ndarray:
        """Return the optical centre in world metres."""
        return -self.rotation.T @ self.translation

    def undistort(self, pixels: np.ndarray) -> np.ndarray:
        """Undistort ``(..., 2)`` raw pixels using this camera's lens coefficients."""
        return undistort_pixels(
            pixels, self.intrinsic, self.distortion, fisheye=self.fisheye
        )

    def project(self, world: np.ndarray, *, distort: bool = False) -> np.ndarray:
        """Project ``(..., 3)`` world points; reject points behind the camera."""
        points = np.asarray(world, dtype=float)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("world must contain finite 3D points")
        camera = points @ self.rotation.T + self.translation
        if np.any(camera[..., 2] <= 0):
            raise ValueError("points must be in front of the camera")
        pixel = camera @ self.intrinsic.T
        uv = pixel[..., :2] / pixel[..., 2:3]
        return (
            distort_pixels(uv, self.intrinsic, self.distortion, fisheye=self.fisheye)
            if distort and self.distortion
            else uv
        )

    def ray(self, pixel: np.ndarray) -> np.ndarray:
        """Return a world-frame unit ray for one undistorted pixel."""
        uv = _array(pixel, (2,), "pixel")
        ray = self.rotation.T @ np.linalg.solve(self.intrinsic, [*uv, 1.0])
        return ray / np.linalg.norm(ray)

    def record(self) -> dict:
        """Return all metric calibration parameters for reproducible reconstruction."""
        out: dict = {
            "intrinsic": self.intrinsic.tolist(),
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "calibration_id": self.calibration_id,
        }
        if self.distortion:
            out["distortion"] = list(self.distortion)
        if self.image_size is not None:
            out["image_size"] = list(self.image_size)
        if self.fisheye:
            out["fisheye"] = True
        return out


def load_pose2sim_cameras(
    path: str | Path, *, camera_names: Sequence[str] | None = None
) -> list[PinholeCamera]:
    """Load calibrated or intrinsics-only cameras from a Pose2Sim ``.toml`` file."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    keys = (
        list(camera_names)
        if camera_names is not None
        else [k for k, v in data.items() if isinstance(v, dict) and "matrix" in v]
    )
    if not keys:
        raise ValueError(f"No camera sections with 'matrix' found in {path}")
    cameras: list[PinholeCamera] = []
    for key in keys:
        sec = data[key]
        raw_rot = sec.get("rotation")
        arr = None if raw_rot is None else np.asarray(raw_rot, dtype=float)
        rot = (
            np.eye(3)
            if arr is None
            else (Rotation.from_rotvec(arr).as_matrix() if arr.shape == (3,) else arr)
        )
        sz = sec.get("size")
        cameras.append(
            PinholeCamera(
                intrinsic=np.asarray(sec["matrix"], dtype=float),
                rotation=rot,
                translation=np.asarray(
                    sec.get("translation", [0.0, 0.0, 0.0]), dtype=float
                ),
                calibration_id=str(sec.get("name") or key),
                distortion=tuple(float(c) for c in sec.get("distortions", ())),
                image_size=(int(round(sz[0])), int(round(sz[1])))
                if sz is not None
                else None,
                fisheye=bool(sec.get("fisheye", False)),
            )
        )
    return cameras


def save_pose2sim_cameras(path: str | Path, cameras: Sequence[PinholeCamera]) -> Path:
    """Write ``cameras`` in Pose2Sim ``Calib_scene.toml`` format."""
    if not cameras:
        raise ValueError("cameras must be non-empty")
    blocks: list[str] = []
    for cam in cameras:
        rvec = Rotation.from_matrix(cam.rotation).as_rotvec().tolist()
        dist = list(cam.distortion) if cam.distortion else [0.0, 0.0, 0.0, 0.0]
        sz = (
            list(cam.image_size)
            if cam.image_size is not None
            else [
                int(round(2 * cam.intrinsic[0, 2])),
                int(round(2 * cam.intrinsic[1, 2])),
            ]
        )
        rows = ", ".join(
            "[" + ", ".join(f"{v:.10g}" for v in row) + "]" for row in cam.intrinsic
        )
        blocks.append(
            f'[{cam.calibration_id}]\nname = "{cam.calibration_id}"\n'
            f"size = [{float(sz[0]):.1f}, {float(sz[1]):.1f}]\nmatrix = [{rows}]\n"
            f"distortions = [{', '.join(f'{v:.10g}' for v in dist)}]\n"
            f"rotation = [{', '.join(f'{v:.10g}' for v in rvec)}]\n"
            f"translation = [{', '.join(f'{v:.10g}' for v in cam.translation)}]\n"
            f"fisheye = {'true' if cam.fisheye else 'false'}\n"
        )
    out = Path(path)
    out.write_text("\n".join(blocks), encoding="utf-8")
    return out


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
    """Triangulate simultaneous undistorted observations of one 3D landmark."""
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
    angle = float(
        np.degrees(np.arccos(np.abs(np.clip(rays @ rays.T, -1.0, 1.0)))).max()
    )
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


@dataclass(frozen=True)
class SequenceTriangulationResult:
    """Multi-frame triangulated landmarks and per-camera pixel/angular diagnostics."""

    positions: np.ndarray
    reprojection_px: np.ndarray
    camera_mre_px: np.ndarray
    camera_mre_mrad: np.ndarray


def angular_reprojection_mrad(
    camera: PinholeCamera, world: np.ndarray, undistorted_pixels: np.ndarray
) -> np.ndarray:
    """Return focal-length-invariant angular reprojection error in milliradians."""
    pts, uv = (
        np.asarray(world, dtype=float),
        np.asarray(undistorted_pixels, dtype=float),
    )
    if (
        pts.shape[:-1] != uv.shape[:-1]
        or pts.shape[-1:] != (3,)
        or uv.shape[-1:] != (2,)
    ):
        raise ValueError("world (..., 3) and undistorted_pixels (..., 2) must match")
    cam_pts = pts @ camera.rotation.T + camera.translation
    cam_dir = cam_pts / np.maximum(
        np.linalg.norm(cam_pts, axis=-1, keepdims=True), 1e-12
    )
    obs_dir = (
        np.concatenate((uv, np.ones((*uv.shape[:-1], 1))), axis=-1)
        @ np.linalg.inv(camera.intrinsic).T
    )
    obs_dir = obs_dir / np.linalg.norm(obs_dir, axis=-1, keepdims=True)
    return np.arccos(np.clip(np.sum(cam_dir * obs_dir, axis=-1), -1.0, 1.0)) * 1000.0


def triangulate_sequence(
    cameras: Sequence[PinholeCamera],
    pixels: np.ndarray,
    *,
    confidences: np.ndarray | None = None,
    min_confidence: float = 0.1,
    undistort: bool = True,
    max_reprojection_px: float = 25.0,
    min_ray_angle_deg: float = 1.0,
) -> SequenceTriangulationResult:
    """Triangulate ``(C, T, J, 2)`` multi-camera keypoint tracks with quality diagnostics."""
    n_cam = len(cameras)
    uv_raw = np.asarray(pixels, dtype=float)
    if (
        n_cam < 2
        or uv_raw.ndim != 4
        or uv_raw.shape[0] != n_cam
        or uv_raw.shape[3] != 2
    ):
        raise ValueError(f"pixels must have shape ({n_cam}, T, J, 2) with >= 2 cameras")
    _, n_frames, n_joints, _ = uv_raw.shape
    conf = (
        np.ones((n_cam, n_frames, n_joints))
        if confidences is None
        else np.asarray(confidences, dtype=float)
    )
    if conf.shape != (n_cam, n_frames, n_joints):
        raise ValueError(f"confidences must have shape {(n_cam, n_frames, n_joints)}")
    uv = np.array(
        [
            c.undistort(uv_raw[i]) if undistort and c.distortion else uv_raw[i]
            for i, c in enumerate(cameras)
        ]
    )
    positions = np.full((n_frames, n_joints, 3), np.nan)
    reproj_px = np.full((n_cam, n_frames, n_joints), np.nan)
    reproj_mrad = np.full((n_cam, n_frames, n_joints), np.nan)
    valid = np.isfinite(uv).all(axis=-1) & np.isfinite(conf) & (conf >= min_confidence)
    for t in range(n_frames):
        for j in range(n_joints):
            vis = np.flatnonzero(valid[:, t, j])
            if len(vis) < 2:
                continue
            sub = [cameras[c] for c in vis]
            try:
                res = triangulate_point(
                    sub,
                    uv[vis, t, j],
                    weights=conf[vis, t, j],
                    max_reprojection_px=max_reprojection_px,
                    min_ray_angle_deg=min_ray_angle_deg,
                )
            except ValueError:
                continue
            positions[t, j], reproj_px[vis, t, j] = (
                res.position,
                res.reprojection_errors,
            )
            for k_i, c in enumerate(vis):
                reproj_mrad[c, t, j] = angular_reprojection_mrad(
                    sub[k_i], res.position, uv[c, t, j]
                )
    with np.errstate(invalid="ignore"):
        return SequenceTriangulationResult(
            positions,
            reproj_px,
            np.nanmean(reproj_px, axis=(1, 2)),
            np.nanmean(reproj_mrad, axis=(1, 2)),
        )


def lift_depth(camera: PinholeCamera, pixel: np.ndarray, depth_m: float) -> np.ndarray:
    """Lift an undistorted pixel using calibrated optical-axis Z depth in metres."""
    uv = _array(pixel, (2,), "pixel")
    if not np.isfinite(depth_m) or depth_m <= 0:
        raise ValueError("depth_m must be positive calibrated metric Z depth")
    return camera.rotation.T @ (
        np.linalg.solve(camera.intrinsic, [*uv, 1.0]) * depth_m - camera.translation
    )


@dataclass(frozen=True)
class WalkFrameAlignment:
    """Similarity transform mapping an arbitrary 3D reconstruction into metric Z-up."""

    positions: np.ndarray
    scale: float
    rotation: np.ndarray
    origin: np.ndarray
    cameras: tuple[PinholeCamera, ...] = ()


def transform_cameras(
    cameras: Sequence[PinholeCamera],
    *,
    scale: float,
    rotation: np.ndarray,
    origin: np.ndarray,
) -> tuple[PinholeCamera, ...]:
    """Update cameras under ``X_new = scale * rotation @ (X_old - origin)``."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    rot, orig = _array(rotation, (3, 3), "rotation"), _array(origin, (3,), "origin")
    if not np.allclose(rot @ rot.T, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rot), 1.0, atol=1e-6
    ):
        raise ValueError("rotation must be a proper 3x3 rotation matrix")
    return tuple(
        PinholeCamera(
            cam.intrinsic,
            cam.rotation @ rot.T,
            scale * (cam.rotation @ orig + cam.translation),
            cam.calibration_id,
            cam.distortion,
            cam.image_size,
            cam.fisheye,
        )
        for cam in cameras
    )


def align_walk_frame(
    positions: np.ndarray,
    height_m: float,
    *,
    head_indices: Sequence[int],
    sole_indices: Sequence[int],
    left_heel_index: int,
    right_heel_index: int,
    leg_chains: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
    trunk_pair: tuple[int, int] | None = None,
    vertical_method: Literal["walk", "frame"] = "walk",
    scale_method: Literal["stature", "segments"] = "stature",
    head_ratio: float = DEFAULT_HEAD_BAND_RATIO,
    ref_frame: int | None = None,
    cameras: Sequence[PinholeCamera] = (),
) -> WalkFrameAlignment:
    """Recover metric Z-up coordinates and scale from a walking skeleton (HumanCalib)."""
    pts = np.asarray(positions, dtype=float)
    if (
        pts.ndim != 3
        or pts.shape[2] != 3
        or not len(pts)
        or not np.isfinite(height_m)
        or height_m <= 0
    ):
        raise ValueError("positions must be (T, J, 3) and height_m must be positive")
    if (
        not head_indices
        or not sole_indices
        or vertical_method not in ("walk", "frame")
        or scale_method not in ("stature", "segments")
    ):
        raise ValueError("invalid landmark indices, vertical_method or scale_method")
    head = np.nanmean(pts[:, list(head_indices)], axis=1)
    l_heel, r_heel = pts[:, left_heel_index], pts[:, right_heel_index]
    mid_heel = 0.5 * (l_heel + r_heel)
    valid = np.isfinite(head).all(axis=1) & np.isfinite(mid_heel).all(axis=1)
    if not valid.any():
        raise ValueError("No frame has finite head and bilateral heel landmarks")
    valid_idx = np.flatnonzero(valid)
    r_idx = int(valid_idx[0]) if ref_frame is None else int(ref_frame)
    if not valid[r_idx]:
        raise ValueError(f"ref_frame {r_idx} lacks finite head or heel landmarks")

    body_vecs = head[valid] - mid_heel[valid]
    if vertical_method == "frame" or len(valid_idx) < 4:
        up, walk_vec = body_vecs[valid_idx.tolist().index(r_idx)], np.zeros(3)
    else:
        up, quarter = np.nanmedian(body_vecs, axis=0), max(1, len(valid_idx) // 4)
        soles = pts[valid][:, list(sole_indices)]
        overall = np.nanmean(mid_heel[valid][-quarter:], axis=0) - np.nanmean(
            mid_heel[valid][:quarter], axis=0
        )
        if len(valid_idx) >= 6:
            speeds = np.linalg.norm(np.diff(soles, axis=0), axis=-1)
            stance = soles[:-1][speeds <= np.nanmedian(speeds)]
            centered = stance - np.nanmean(stance, axis=0)
            walk_vec = np.linalg.svd(
                centered[np.isfinite(centered).all(axis=1)], full_matrices=False
            )[2][0]
            if np.dot(walk_vec, overall) < 0:
                walk_vec = -walk_vec
        else:
            walk_vec = overall
        if np.linalg.norm(walk_vec) > 1e-8:
            w_hat = walk_vec / np.linalg.norm(walk_vec)
            up = up - np.dot(up, w_hat) * w_hat

    z_axis = up / np.linalg.norm(up)
    proj_walk = walk_vec - np.dot(walk_vec, z_axis) * z_axis
    if np.linalg.norm(proj_walk) > 1e-6:
        x_axis = proj_walk / np.linalg.norm(proj_walk)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
    else:
        lat = (l_heel[r_idx] - r_heel[r_idx]) - np.dot(
            l_heel[r_idx] - r_heel[r_idx], z_axis
        ) * z_axis
        y_axis = lat / np.linalg.norm(lat)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
    rot = np.vstack((x_axis, y_axis, z_axis))

    if scale_method == "stature":
        sole_z = np.nanmin(pts[valid][:, list(sole_indices)] @ z_axis, axis=1)
        raw_span, expected = (
            float(np.nanpercentile(head[valid] @ z_axis - sole_z, 95.0)),
            head_ratio * height_m,
        )
    else:
        if leg_chains is None:
            raise ValueError("leg_chains is required when scale_method='segments'")
        seg = 0.5 * sum(
            np.linalg.norm(pts[valid, h] - pts[valid, k], axis=1)
            + np.linalg.norm(pts[valid, k] - pts[valid, a], axis=1)
            for h, k, a in leg_chains
        )
        ratio = LEG_STATURE_RATIO + (
            TRUNK_STATURE_RATIO if trunk_pair is not None else 0.0
        )
        if trunk_pair is not None:
            seg = seg + np.linalg.norm(
                pts[valid, trunk_pair[1]] - pts[valid, trunk_pair[0]], axis=1
            )
        raw_span, expected = float(np.nanmedian(seg)), ratio * height_m
    if not np.isfinite(raw_span) or raw_span <= 1e-9:
        raise ValueError("Degenerate skeleton span; cannot determine metric scale")
    scale = expected / raw_span
    floor_z = float(
        np.nanpercentile(
            np.nanmin(pts[valid][:, list(sole_indices)] @ z_axis, axis=1), 5.0
        )
    )
    ref_xy = mid_heel[r_idx] - (mid_heel[r_idx] @ z_axis - floor_z) * z_axis
    return WalkFrameAlignment(
        scale * ((pts - ref_xy) @ rot.T),
        float(scale),
        rot,
        ref_xy,
        transform_cameras(cameras, scale=scale, rotation=rot, origin=ref_xy),
    )
