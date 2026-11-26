# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Pre-capture multi-camera rig placement optimization over musculoskeletal observability.

Maximizes worst-case target-DOF singular values, D-optimal log-volume, and stereo
landmark coverage under ``mj_ray`` body self-occlusion, inspired by
``flodelaplace/lab-camera-optimizer`` (Delaplace, 2026, MIT).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from myosuite.biomechanics.camera import PinholeCamera
from myosuite.biomechanics.observability import (
    CameraObservabilityResult,
    camera_landmark_observability,
)
from myosuite.biomechanics.retarget import Landmark


@dataclass(frozen=True)
class RoomBounds:
    """Physical room bounds and mount clearance constraints in world metres."""

    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]
    height_bounds: tuple[float, float] = (1.1, 2.5)
    min_subject_distance_m: float = 1.0
    min_camera_separation_m: float = 0.6
    mount_mode: Literal["perimeter", "free"] = "perimeter"

    def __post_init__(self) -> None:
        for name, (lo, hi) in (
            ("x_bounds", self.x_bounds),
            ("y_bounds", self.y_bounds),
            ("height_bounds", self.height_bounds),
        ):
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise ValueError(f"{name} must satisfy lo < hi, got {(lo, hi)}")
        if self.min_subject_distance_m <= 0 or self.min_camera_separation_m < 0:
            raise ValueError("Clearance distances must be nonnegative")
        if self.mount_mode not in ("perimeter", "free"):
            raise ValueError(f"Unsupported mount_mode: {self.mount_mode!r}")

    def perimeter_point(self, frac: float, height: float) -> np.ndarray:
        """Map a normalized perimeter coordinate ``frac`` in ``[0, 1)`` to world ``(x, y, z)``."""
        x0, x1 = self.x_bounds
        y0, y1 = self.y_bounds
        dx, dy = x1 - x0, y1 - y0
        perim = 2.0 * (dx + dy)
        s = float(np.mod(frac, 1.0)) * perim
        z = float(np.clip(height, self.height_bounds[0], self.height_bounds[1]))
        if s < dx:
            return np.array([x0 + s, y0, z], dtype=np.float64)
        s -= dx
        if s < dy:
            return np.array([x1, y0 + s, z], dtype=np.float64)
        s -= dy
        if s < dx:
            return np.array([x1 - s, y1, z], dtype=np.float64)
        s -= dx
        return np.array([x0, y1 - s, z], dtype=np.float64)


@dataclass(frozen=True)
class CameraRigOptimizationResult:
    """Optimized multi-camera rig and musculoskeletal observability diagnostics."""

    cameras: tuple[PinholeCamera, ...]
    positions: np.ndarray
    image_sizes: tuple[tuple[int, int] | None, ...]
    per_frame_observability: tuple[CameraObservabilityResult, ...]
    worst_dof_sigma: float
    mean_log_sigma: float
    stereo_fraction: float
    score: float
    evaluations: int


def look_at_camera(
    calibration_id: str,
    position: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
    intrinsic: np.ndarray,
    *,
    up: Sequence[float] | np.ndarray = (0.0, 0.0, 1.0),
    yaw_offset_rad: float = 0.0,
    pitch_offset_rad: float = 0.0,
) -> PinholeCamera:
    """Construct a calibrated OpenCV :class:`PinholeCamera` aimed from *position* at *target*."""
    pos = np.asarray(position, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    up_v = np.asarray(up, dtype=np.float64)
    if pos.shape != (3,) or tgt.shape != (3,) or up_v.shape != (3,):
        raise ValueError("position, target, and up must be 3-vectors")
    fwd = tgt - pos
    dist = float(np.linalg.norm(fwd))
    if dist < 1e-8:
        raise ValueError("Camera position and target must be distinct")
    fwd /= dist
    right = np.cross(fwd, up_v)
    rn = float(np.linalg.norm(right))
    if rn < 1e-8:
        alt = (
            np.array([1.0, 0.0, 0.0])
            if abs(fwd[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        right = np.cross(fwd, alt)
        rn = float(np.linalg.norm(right))
    right /= rn
    down = np.cross(fwd, right)
    r_base = np.vstack([right, down, fwd])
    if yaw_offset_rad != 0.0 or pitch_offset_rad != 0.0:
        r_off = Rotation.from_euler(
            "yx", [yaw_offset_rad, pitch_offset_rad]
        ).as_matrix()
        r_base = r_off @ r_base
    t_vec = -(r_base @ pos)
    return PinholeCamera(
        intrinsic=np.asarray(intrinsic, dtype=np.float64),
        rotation=r_base,
        translation=t_vec,
        calibration_id=calibration_id,
    )


def evaluate_camera_rig(
    model: mujoco.MjModel,
    poses: np.ndarray,
    landmarks: Sequence[Landmark],
    cameras: Sequence[PinholeCamera],
    *,
    image_sizes: Sequence[tuple[int, int] | None] | None = None,
    check_occlusion: bool = True,
    min_views: int = 2,
    target_dofs: Sequence[int] | None = None,
    fixed_dofs: Sequence[int] = (),
    dof_scales: Mapping[int, float] | None = None,
) -> tuple[tuple[CameraObservabilityResult, ...], float, float, float, float]:
    """Evaluate a camera rig across *poses*, returning observability and scalar summary."""
    results: list[CameraObservabilityResult] = []
    worst_sig = float("inf")
    log_sigs: list[float] = []
    stereo_hits = 0
    total_lm = 0
    for q in poses:
        res = camera_landmark_observability(
            model,
            q,
            landmarks,
            cameras,
            image_sizes=image_sizes,
            check_occlusion=check_occlusion,
            min_views=min_views,
            target_dofs=target_dofs,
            fixed_dofs=fixed_dofs,
            dof_scales=dof_scales,
        )
        results.append(res)
        sig = res.observability.singular_values
        if len(sig) > 0:
            worst_sig = min(worst_sig, float(sig.min()))
            log_sigs.append(float(np.mean(np.log(sig + 1e-4))))
        else:
            worst_sig = 0.0
            log_sigs.append(float(np.log(1e-4)))
        stereo_hits += int(np.count_nonzero(res.views_per_landmark >= min_views))
        total_lm += len(res.views_per_landmark)

    if not np.isfinite(worst_sig):
        worst_sig = 0.0
    mean_log = float(np.mean(log_sigs)) if log_sigs else float(np.log(1e-4))
    stereo_frac = stereo_hits / max(total_lm, 1)
    # Combined E-optimality (worst DOF), D-optimality (log volume), and stereo coverage.
    score = worst_sig + 0.15 * mean_log + 2.0 * stereo_frac
    return tuple(results), worst_sig, mean_log, stereo_frac, score


def _subsample_poses(poses: np.ndarray, max_frames: int) -> np.ndarray:
    if len(poses) <= max_frames:
        return poses
    idx = np.linspace(0, len(poses) - 1, max_frames).round().astype(int)
    return poses[np.unique(idx)]


def optimize_camera_rig(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    landmarks: Sequence[Landmark],
    n_cameras: int,
    intrinsic: np.ndarray | Sequence[np.ndarray],
    room: RoomBounds,
    *,
    image_size: tuple[int, int] | None = (1920, 1080),
    check_occlusion: bool = True,
    min_views: int = 2,
    target_dofs: Sequence[int] | None = None,
    fixed_dofs: Sequence[int] = (),
    dof_scales: Mapping[int, float] | None = None,
    max_eval_frames: int = 6,
    n_candidates: int = 24,
    refine_steps: int = 12,
    seed: int = 0,
) -> CameraRigOptimizationResult:
    """Optimize *n_cameras* mounts and orientations inside *room* for *qpos* and *landmarks*."""
    if n_cameras < 1:
        raise ValueError(f"n_cameras must be >= 1, got {n_cameras}")
    if n_candidates < n_cameras:
        raise ValueError(
            f"n_candidates ({n_candidates}) must be >= n_cameras ({n_cameras})"
        )
    q_arr = np.asarray(qpos, dtype=np.float64)
    if q_arr.ndim == 1:
        q_arr = q_arr[None, :]
    if q_arr.ndim != 2 or q_arr.shape[1] != model.nq:
        raise ValueError(f"qpos must have shape ({model.nq},) or (T, {model.nq})")
    poses = _subsample_poses(q_arr, max(1, max_eval_frames))

    if isinstance(intrinsic, np.ndarray) and intrinsic.ndim == 2:
        intrinsics = [intrinsic.copy() for _ in range(n_cameras)]
    else:
        intrinsics = [np.asarray(k, dtype=np.float64) for k in intrinsic]
        if len(intrinsics) != n_cameras:
            raise ValueError(
                f"Expected {n_cameras} intrinsic matrices, got {len(intrinsics)}"
            )

    # Compute motion-volume landmark centroid and bounding positions for subject clearance.
    data = mujoco.MjData(model)
    lm_pts_list: list[np.ndarray] = []
    site_names = {model.site(i).name for i in range(model.nsite)}
    for q in poses:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        for lm in landmarks:
            if lm.site is not None and lm.site in site_names:
                st = model.site(lm.site)
                b_id, off = int(st.bodyid[0]), st.pos
            else:
                b_id, off = model.body(lm.body).id, np.asarray(lm.pos, dtype=np.float64)
            lm_pts_list.append(data.xpos[b_id] + data.xmat[b_id].reshape(3, 3) @ off)
    all_lm_pts = np.stack(lm_pts_list, axis=0)
    centroid = np.mean(all_lm_pts, axis=0)

    rng = np.random.default_rng(seed)
    z_lo, z_hi = room.height_bounds
    heights_grid = np.linspace(
        z_lo + 0.1 * (z_hi - z_lo), z_hi - 0.1 * (z_hi - z_lo), 3
    )

    cand_params: list[np.ndarray] = []
    cand_positions: list[np.ndarray] = []
    for k in range(n_candidates):
        if room.mount_mode == "perimeter":
            u_frac = (k + 0.5 * float(rng.uniform(-0.3, 0.3))) / n_candidates
            z_val = float(heights_grid[k % len(heights_grid)])
            pos = room.perimeter_point(u_frac, z_val)
            param = np.array([u_frac, z_val, 0.0, 0.0], dtype=np.float64)
        else:
            pos = np.array(
                [
                    float(rng.uniform(*room.x_bounds)),
                    float(rng.uniform(*room.y_bounds)),
                    float(rng.uniform(z_lo, z_hi)),
                ],
                dtype=np.float64,
            )
            param = np.array([pos[0], pos[1], pos[2], 0.0, 0.0], dtype=np.float64)
        if (
            float(np.min(np.linalg.norm(all_lm_pts[:, :2] - pos[:2], axis=1)))
            >= room.min_subject_distance_m
        ):
            cand_params.append(param)
            cand_positions.append(pos)

    if len(cand_params) < n_cameras:
        raise ValueError(
            f"Only {len(cand_params)} valid candidate mounts satisfy min_subject_distance_m="
            f"{room.min_subject_distance_m} inside room bounds"
        )

    def build_cam(idx: int, param: np.ndarray) -> tuple[PinholeCamera, np.ndarray]:
        if room.mount_mode == "perimeter":
            pos = room.perimeter_point(float(param[0]), float(param[1]))
            yaw, pitch = float(param[2]), float(param[3])
        else:
            pos = np.array(
                [
                    float(np.clip(param[0], *room.x_bounds)),
                    float(np.clip(param[1], *room.y_bounds)),
                    float(np.clip(param[2], *room.height_bounds)),
                ]
            )
            yaw, pitch = float(param[3]), float(param[4])
        cam = look_at_camera(
            f"cam_{idx}",
            pos,
            centroid,
            intrinsics[idx],
            yaw_offset_rad=yaw,
            pitch_offset_rad=pitch,
        )
        return cam, pos

    eval_count = 0

    def score_params(params_list: Sequence[np.ndarray]) -> float:
        nonlocal eval_count
        cams_and_pos = [build_cam(i, p) for i, p in enumerate(params_list)]
        cams = [c for c, _ in cams_and_pos]
        pts = np.stack([pt for _, pt in cams_and_pos], axis=0)
        if (
            float(
                np.min(
                    np.linalg.norm(all_lm_pts[:, None, :2] - pts[None, :, :2], axis=-1)
                )
            )
            < room.min_subject_distance_m
        ):
            return -1e9
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if (
                    float(np.linalg.norm(pts[i] - pts[j]))
                    < room.min_camera_separation_m
                ):
                    return -1e9
        eval_count += 1
        eff_min_views = min(min_views, len(cams))
        img_sizes = [image_size] * len(cams)
        _, _, _, _, sc = evaluate_camera_rig(
            model,
            poses,
            landmarks,
            cams,
            image_sizes=img_sizes,
            check_occlusion=check_occlusion,
            min_views=eff_min_views,
            target_dofs=target_dofs,
            fixed_dofs=fixed_dofs,
            dof_scales=dof_scales,
        )
        return sc

    # Stage 1: Greedy submodular mount selection.
    chosen_indices: list[int] = []
    chosen_params: list[np.ndarray] = []
    for slot in range(n_cameras):
        best_sc = -1e12
        best_k = -1
        for k, p_cand in enumerate(cand_params):
            if k in chosen_indices:
                continue
            sc = score_params([*chosen_params, p_cand])
            if sc > best_sc:
                best_sc = sc
                best_k = k
        if best_k < 0:
            best_k = next(k for k in range(len(cand_params)) if k not in chosen_indices)
        chosen_indices.append(best_k)
        chosen_params.append(cand_params[best_k].copy())

    # Stage 2: Seeded coordinate / random local refinement of position + pan/tilt.
    cur_score = score_params(chosen_params)
    step_scale = (
        np.array([0.04, 0.20, 0.08, 0.06])
        if room.mount_mode == "perimeter"
        else np.array([0.25, 0.25, 0.20, 0.08, 0.06])
    )
    for it in range(refine_steps):
        decay = 1.0 - 0.6 * (it / max(refine_steps, 1))
        for c_i in range(n_cameras):
            delta = rng.normal(0.0, step_scale * decay)
            trial_p = chosen_params[c_i] + delta
            if room.mount_mode == "perimeter":
                trial_p[0] = float(np.mod(trial_p[0], 1.0))
                trial_p[1] = float(np.clip(trial_p[1], z_lo, z_hi))
                trial_p[2:] = np.clip(trial_p[2:], -0.35, 0.35)
            else:
                trial_p[0] = float(np.clip(trial_p[0], *room.x_bounds))
                trial_p[1] = float(np.clip(trial_p[1], *room.y_bounds))
                trial_p[2] = float(np.clip(trial_p[2], z_lo, z_hi))
                trial_p[3:] = np.clip(trial_p[3:], -0.35, 0.35)
            candidate_list = list(chosen_params)
            candidate_list[c_i] = trial_p
            sc = score_params(candidate_list)
            if sc > cur_score:
                chosen_params[c_i] = trial_p
                cur_score = sc

    final_pairs = [build_cam(i, p) for i, p in enumerate(chosen_params)]
    final_cams = tuple(c for c, _ in final_pairs)
    final_pos = np.stack([pt for _, pt in final_pairs], axis=0)
    final_sizes = tuple([image_size] * n_cameras)
    obs_list, worst_sig, mean_log, stereo_frac, final_sc = evaluate_camera_rig(
        model,
        poses,
        landmarks,
        final_cams,
        image_sizes=final_sizes,
        check_occlusion=check_occlusion,
        min_views=min_views,
        target_dofs=target_dofs,
        fixed_dofs=fixed_dofs,
        dof_scales=dof_scales,
    )
    return CameraRigOptimizationResult(
        cameras=final_cams,
        positions=final_pos,
        image_sizes=final_sizes,
        per_frame_observability=obs_list,
        worst_dof_sigma=worst_sig,
        mean_log_sigma=mean_log,
        stereo_fraction=stereo_frac,
        score=final_sc,
        evaluations=eval_count,
    )


__all__ = [
    "CameraRigOptimizationResult",
    "RoomBounds",
    "evaluate_camera_rig",
    "look_at_camera",
    "optimize_camera_rig",
]
