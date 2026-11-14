# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""InstantHMR MHR70 scaling, inverse kinematics, retargeting and muscle reconstruction."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from myosuite.biomechanics.inverse_dynamics import clip_kinematics
from myosuite.biomechanics.retarget import (
    ClipTargets,
    LandmarkIK,
    LandmarkMap,
    RefitConfig,
    RefitThresholds,
    _body_landmarks,
    _calibration_pose,
    _foot_contacts,
    _geom_lowest_z,
    _ground_clip,
    _hinge_table,
    _knee_flexion_meta,
    _mat_to_quat,
    _orientation_targets,
    _segment_angle,
    _smooth,
    _smooth_quat,
    _start_pose,
    merge_quality_flags,
    validate_refit,
)
from myosuite.core.trajectory_io import MotionClip
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

if TYPE_CHECKING:
    from myosuite.viz.muscle_meshes import MuscleMeshReconstruction

MHR70_JOINTS: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_big_toe_tip", "left_small_toe_tip", "left_heel",
    "right_big_toe_tip", "right_small_toe_tip", "right_heel",
    "right_thumb_tip", "right_thumb_first_joint", "right_thumb_second_joint", "right_thumb_third_joint",
    "right_index_tip", "right_index_first_joint", "right_index_second_joint", "right_index_third_joint",
    "right_middle_tip", "right_middle_first_joint", "right_middle_second_joint", "right_middle_third_joint",
    "right_ring_tip", "right_ring_first_joint", "right_ring_second_joint", "right_ring_third_joint",
    "right_pinky_tip", "right_pinky_first_joint", "right_pinky_second_joint", "right_pinky_third_joint",
    "right_wrist",
    "left_thumb_tip", "left_thumb_first_joint", "left_thumb_second_joint", "left_thumb_third_joint",
    "left_index_tip", "left_index_first_joint", "left_index_second_joint", "left_index_third_joint",
    "left_middle_tip", "left_middle_first_joint", "left_middle_second_joint", "left_middle_third_joint",
    "left_ring_tip", "left_ring_first_joint", "left_ring_second_joint", "left_ring_third_joint",
    "left_pinky_tip", "left_pinky_first_joint", "left_pinky_second_joint", "left_pinky_third_joint",
    "left_wrist",
    "left_olecranon", "right_olecranon", "left_cubital_fossa", "right_cubital_fossa",
    "left_acromion", "right_acromion", "neck",
)  # fmt: skip
"""InstantHMR / SAM-3D-Body MHR70 keypoint names (0..69)."""

_JIDX: dict[str, int] = {name: i for i, name in enumerate(MHR70_JOINTS)}

CAM_TO_WORLD = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
"""OpenCV camera (``+x`` right, ``+y`` down, ``+z`` forward) to world (``+x`` forward, ``+y`` left, ``+z`` up)."""


def _check_rotation(R: np.ndarray, name: str = "cam_to_world") -> np.ndarray:
    R = np.asarray(R, np.float64)
    if R.shape != (3, 3) or not np.isfinite(R).all():
        raise ValueError(f"{name} must be a finite (3, 3) matrix, got {R.shape}")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-4) or np.linalg.det(R) < 0.999:
        raise ValueError(f"{name} must be a proper rotation matrix (det = +1)")
    return R


def _interpolate_gaps(joints: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if valid.all():
        return joints
    out = joints.copy()
    t = np.arange(len(joints), dtype=np.float64)
    ok, flat = t[valid], out.reshape(len(joints), -1)
    for col in range(flat.shape[1]):
        flat[:, col] = np.interp(t, ok, flat[valid, col])
    return out


def _parse_quality_flags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str | bytes):
        text = value.decode() if isinstance(value, bytes) else value
        if text.startswith("["):
            import json  # noqa: PLC0415

            return [str(x) for x in json.loads(text) if str(x)]
        return [text] if text else []
    arr = np.asarray(value)
    if arr.ndim == 0:
        return _parse_quality_flags(arr.item())
    return [str(x) for x in arr.ravel().tolist() if str(x)]


def _extract_valid_segments(
    raw: dict[str, Any],
    label: str | Path = "<memory>",
    *,
    max_gap_s: float = 0.25,
    on_long_gap: str = "reject",
    min_frames: int = 2,
) -> list[dict[str, Any]]:
    if on_long_gap not in ("reject", "split"):
        raise ValueError(
            f"on_long_gap must be 'reject' or 'split', got {on_long_gap!r}"
        )
    if not np.isfinite(max_gap_s) or max_gap_s < 0:
        raise ValueError(f"max_gap_s must be finite and non-negative, got {max_gap_s}")
    if min_frames < 2:
        raise ValueError(f"min_frames must be >= 2, got {min_frames}")

    files = set(raw)
    if "joints_3d_cam" in files:
        joints = np.asarray(raw["joints_3d_cam"], np.float64)
    elif "joints_3d_local" in files or "joints_3d" in files:
        key = "joints_3d_local" if "joints_3d_local" in files else "joints_3d"
        joints = np.asarray(raw[key], np.float64)
        if "cam_trans" in files and raw["cam_trans"] is not None:
            trans = np.asarray(raw["cam_trans"], np.float64)
            if trans.shape != (len(joints), 3):
                raise ValueError(f"cam_trans shape {trans.shape} != ({len(joints)}, 3)")
            joints = joints + trans[:, None, :]
    else:
        raise ValueError(f"{label}: expected 'joints_3d_cam' or 'joints_3d_local'")

    fps_key = (
        "fps" if "fps" in files else ("frequency" if "frequency" in files else None)
    )
    if fps_key is None:
        raise ValueError(f"{label}: missing 'fps' (or 'frequency')")
    fps = float(raw[fps_key])
    valid_raw = (
        np.asarray(raw["valid"], bool)
        if "valid" in files and raw["valid"] is not None
        else None
    )
    cam_R = (
        _check_rotation(raw["cam_to_world"])
        if "cam_to_world" in files and raw["cam_to_world"] is not None
        else None
    )
    shape_params = (
        np.asarray(raw["shape_params"], np.float64)
        if "shape_params" in files and raw["shape_params"] is not None
        else None
    )
    base_flags = _parse_quality_flags(raw.get("quality_flags"))
    base_trimmed = int(raw.get("endpoint_trimmed_frames", 0))
    base_offset = (
        int(raw["segment_range"][0]) if raw.get("segment_range") is not None else 0
    )

    if joints.ndim != 3 or joints.shape[1:] != (len(MHR70_JOINTS), 3):
        raise ValueError(
            f"{label}: expected joints shape (T, 70, 3), got {joints.shape}"
        )
    if len(joints) < min_frames:
        raise ValueError(
            f"{label}: clip needs at least {min_frames} frames, got {len(joints)}"
        )
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{label}: fps must be finite and positive, got {fps}")

    finite = np.isfinite(joints).all(axis=(1, 2))
    if valid_raw is not None:
        if valid_raw.shape != (len(joints),):
            raise ValueError(
                f"{label}: valid mask shape {valid_raw.shape} != ({len(joints)},)"
            )
        valid = valid_raw & finite
    else:
        valid = finite

    ok_idx = np.flatnonzero(valid)
    if len(ok_idx) < min_frames:
        raise ValueError(
            f"{label}: need at least {min_frames} valid frames, got {len(ok_idx)}"
        )

    first_ok, last_ok = int(ok_idx[0]), int(ok_idx[-1])
    trimmed_lead = first_ok
    trimmed_trail = len(joints) - 1 - last_ok
    max_gap_frames = int(np.floor(max_gap_s * fps + 1e-9))
    gaps = np.diff(ok_idx) - 1
    long_idx = np.flatnonzero(gaps > max_gap_frames)

    if len(long_idx) > 0 and on_long_gap == "reject":
        longest = int(gaps.max())
        raise ValueError(
            f"{label}: missing-frame gap of {longest / fps:.3f}s ({longest} frames) "
            f"exceeds max_gap_s={max_gap_s:.3f}s"
        )

    starts = [first_ok, *(int(ok_idx[i + 1]) for i in long_idx)]
    ends = [*(int(ok_idx[i]) + 1 for i in long_idx), last_ok + 1]
    n_orig = len(joints)

    segments: list[dict[str, Any]] = []
    for seg_i, (s, e) in enumerate(zip(starts, ends)):
        j_seg = joints[s:e]
        v_seg = valid[s:e]
        if int(v_seg.sum()) < min_frames or len(j_seg) < min_frames:
            continue
        seg_trimmed = (
            base_trimmed + trimmed_lead + trimmed_trail
            if len(starts) == 1
            else (trimmed_lead if seg_i == 0 else 0)
            + (trimmed_trail if seg_i == len(starts) - 1 else 0)
        )
        interp_count = int((~v_seg).sum())
        new_flags: list[str] = []
        if seg_trimmed > 0:
            new_flags.append("trimmed_endpoints")
        if len(long_idx) > 0:
            new_flags.append("split_long_gap")
        if interp_count > 0:
            new_flags.append("interpolated_gaps")
        sp_seg = (
            shape_params[s:e]
            if shape_params is not None
            and shape_params.ndim >= 2
            and len(shape_params) == n_orig
            else shape_params
        )
        segments.append(
            {
                "joints_3d_cam": _interpolate_gaps(j_seg, v_seg),
                "fps": fps,
                "valid": v_seg,
                "interpolated_frames": interp_count,
                "endpoint_trimmed_frames": seg_trimmed,
                "segment_range": [base_offset + s, base_offset + e],
                "quality_flags": merge_quality_flags(base_flags, new_flags),
                "cam_to_world": cam_R,
                "shape_params": sp_seg,
            }
        )

    if not segments:
        raise ValueError(
            f"{label}: no contiguous valid segment of >= {min_frames} frames "
            f"within max_gap_s={max_gap_s:.3f}s"
        )
    return segments


def _normalize_clip_dict(
    raw: dict[str, Any],
    label: str | Path = "<memory>",
    *,
    max_gap_s: float = 0.25,
) -> dict[str, Any]:
    return _extract_valid_segments(
        raw, label, max_gap_s=max_gap_s, on_long_gap="reject"
    )[0]


def load_instanthmr(
    path: str | Path,
    *,
    max_gap_s: float = 0.25,
) -> dict[str, Any]:
    """Read an InstantHMR ``.npz`` clip, trim missing endpoints and interpolate short gaps.

    Args:
        path: InstantHMR ``.npz`` path.
        max_gap_s: Maximum missing-frame gap duration (s) interpolated across; longer
            interior gaps raise :class:`ValueError` (use :func:`split_instanthmr_clips`
            to split across long occlusions).

    Returns:
        Normalized clip dictionary with ``joints_3d_cam``, ``fps``, ``valid``,
        ``interpolated_frames``, ``endpoint_trimmed_frames``, ``segment_range`` and
        ``quality_flags``.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        raw = {k: data[k] for k in data.files}
    return _normalize_clip_dict(raw, path, max_gap_s=max_gap_s)


def split_instanthmr_clips(
    clip: str | Path | dict[str, Any],
    *,
    max_gap_s: float = 0.25,
    min_frames: int = 2,
) -> list[dict[str, Any]]:
    """Split an InstantHMR clip at missing-frame gaps longer than *max_gap_s*.

    Leading and trailing missing frames are trimmed, gaps up to *max_gap_s* are
    linearly interpolated, and gaps exceeding *max_gap_s* split the clip into
    sub-clips carrying ``segment_range`` and ``quality_flags``.
    """
    if isinstance(clip, dict):
        return _extract_valid_segments(
            clip, max_gap_s=max_gap_s, on_long_gap="split", min_frames=min_frames
        )
    with np.load(Path(clip), allow_pickle=False) as data:
        raw = {k: data[k] for k in data.files}
    return _extract_valid_segments(
        raw, clip, max_gap_s=max_gap_s, on_long_gap="split", min_frames=min_frames
    )


def _as_clip(
    clip: str | Path | dict[str, Any], *, max_gap_s: float = 0.25
) -> dict[str, Any]:
    if isinstance(clip, dict):
        return _normalize_clip_dict(clip, max_gap_s=max_gap_s)
    return load_instanthmr(clip, max_gap_s=max_gap_s)


@dataclass(frozen=True)
class InstantHMRPoint:
    """Target tree node pairing MHR70 keypoints with a model body."""

    name: str
    mhr: tuple[str, ...]
    body: str
    model: tuple[str, ...]
    parent: str | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class InstantHMROrientation:
    """Orthonormal anatomical triad target built from MHR70 keypoints."""

    name: str
    body: str
    primary: tuple[tuple[str, ...], tuple[str, ...]]
    secondary: tuple[tuple[str, ...], tuple[str, ...]] = ()
    model_primary: tuple[tuple[str, ...], tuple[str, ...]] = ()
    model_secondary: tuple[tuple[str, ...], tuple[str, ...]] = ()
    canonical_primary: tuple[float, float, float] = (1.0, 0.0, 0.0)
    canonical_secondary: tuple[float, float, float] = (0.0, 1.0, 0.0)
    kind: str = "triad"
    side: str = ""
    weight: float = 0.05
    axis_to: str | None = None
    axis_weight: float = 1.0


@dataclass(frozen=True)
class InstantHMRHinge:
    """Model hinge driven per frame by the angle between two MHR70 segments."""

    joint: str
    mhr: tuple[str, str, str]
    model: tuple[str, str, str]
    weight: float = 0.1


@dataclass(frozen=True)
class InstantHMRMap:
    """Model-specific specification for InstantHMR MHR70 retargeting."""

    name: str
    version: str
    points: tuple[InstantHMRPoint, ...]
    orientations: tuple[InstantHMROrientation, ...]
    calibration: dict[str, float] = field(default_factory=dict)
    feet: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    contact_keypoints: tuple[tuple[str, ...], tuple[str, ...]] = (
        ("left_heel", "left_big_toe_tip", "left_small_toe_tip"),
        ("right_heel", "right_big_toe_tip", "right_small_toe_tip"),
    )  # fmt: skip
    leg_edges: tuple[tuple[str, str], ...] = (
        ("L_Hip", "L_Knee"), ("L_Knee", "L_Ankle"), ("R_Hip", "R_Knee"), ("R_Knee", "R_Ankle"),
    )  # fmt: skip
    twist_pairs: tuple[tuple[str, str], ...] = ()
    twist_reference: str = ""
    coupling_term: str = ""
    period_joint: str = ""
    posture: tuple[tuple[str, float, float], ...] = ()
    hinges: tuple[InstantHMRHinge, ...] = ()


def _leg_points() -> tuple[InstantHMRPoint, ...]:
    pts = []
    for S, side, m in (("L_", "left", "_l"), ("R_", "right", "_r")):
        pts += [
            InstantHMRPoint(f"{S}Hip", (f"{side}_hip",), f"femur{m}", (f"femur{m}",), "Hips"),
            InstantHMRPoint(f"{S}Knee", (f"{side}_knee",), f"tibia{m}", (f"tibia{m}",), f"{S}Hip"),
            InstantHMRPoint(f"{S}Ankle", (f"{side}_ankle",), f"talus{m}", (f"talus{m}",), f"{S}Knee"),
            InstantHMRPoint(f"{S}Heel", (f"{side}_heel",), f"calcn{m}", (f"calcn{m}",), f"{S}Ankle", weight=0.5),
            InstantHMRPoint(f"{S}Forefoot", (f"{side}_big_toe_tip", f"{side}_small_toe_tip"),
                            f"toes{m}", (f"toes{m}",), f"{S}Ankle"),
        ]  # fmt: skip
    return tuple(pts)


def _arm_points() -> tuple[InstantHMRPoint, ...]:
    pts = []
    for S, side, m in (("L_", "left", "_l"), ("R_", "right", "_r")):
        pts += [
            InstantHMRPoint(f"{S}Shoulder", (f"{side}_shoulder",), f"humerus{m}", (f"humerus{m}",), "Shoulders", weight=0.3),
            InstantHMRPoint(f"{S}Elbow", (f"{side}_elbow",), f"ulna{m}", (f"ulna{m}",), f"{S}Shoulder"),
            InstantHMRPoint(f"{S}Wrist", (f"{side}_wrist",), f"lunate{m}", (f"lunate{m}",), f"{S}Elbow"),
        ]  # fmt: skip
    return tuple(pts)


def _leg_orientations() -> tuple[InstantHMROrientation, ...]:
    ors = []
    for S, side, m in (("L_", "left", "_l"), ("R_", "right", "_r")):
        fore = (f"{side}_big_toe_tip", f"{side}_small_toe_tip")
        toe_lat = ((f"{side}_big_toe_tip",), (f"{side}_small_toe_tip",)) if side == "left" else ((f"{side}_small_toe_tip",), (f"{side}_big_toe_tip",))  # fmt: skip
        ors += [
            InstantHMROrientation(f"{S}Femur", f"femur{m}", ((f"{side}_hip",), (f"{side}_knee",)),
                                  model_primary=((f"femur{m}",), (f"tibia{m}",)), kind="femur", side=side, weight=0.1),
            InstantHMROrientation(f"{S}Tibia", f"tibia{m}", ((f"{side}_knee",), (f"{side}_ankle",)),
                                  ((f"{side}_heel",), fore), ((f"tibia{m}",), (f"talus{m}",)),
                                  ((f"calcn{m}",), (f"toes{m}",)), weight=0.1, axis_to=f"talus{m}", axis_weight=0.1),
            InstantHMROrientation(f"{S}Foot", f"calcn{m}", ((f"{side}_heel",), fore), toe_lat,
                                  ((f"calcn{m}",), (f"toes{m}",)), canonical_secondary=(0.0, 1.0, 0.0),
                                  weight=0.1, axis_to=f"toes{m}", axis_weight=0.4),
        ]  # fmt: skip
    return tuple(ors)


def _arm_orientations() -> tuple[InstantHMROrientation, ...]:
    ors = []
    for S, side, m in (("L_", "left", "_l"), ("R_", "right", "_r")):
        ors += [
            InstantHMROrientation(f"{S}Humerus", f"humerus{m}", ((f"{side}_shoulder",), (f"{side}_elbow",)),
                                  ((f"{side}_olecranon",), (f"{side}_cubital_fossa",)),
                                  ((f"humerus{m}",), (f"ulna{m}",)), canonical_secondary=(1.0, 0.0, 0.0), weight=0.05),
            InstantHMROrientation(f"{S}Hand", f"lunate{m}", ((f"{side}_wrist",), (f"{side}_middle_third_joint",)),
                                  ((f"{side}_pinky_third_joint",), (f"{side}_index_third_joint",)),
                                  ((f"lunate{m}",), (f"3proxph{m}",)), ((f"5proxph{m}",), (f"2proxph{m}",)),
                                  canonical_primary=(0.0, 1.0 if side == "left" else -1.0, 0.0),
                                  canonical_secondary=(1.0, 0.0, 0.0), weight=0.05),
        ]  # fmt: skip
    return tuple(ors)


INSTANTHMR_MYOFULLBODY = InstantHMRMap(
    name="instanthmr-mhr70->myofullbody",
    version="1",
    points=(
        InstantHMRPoint(
            "Hips", ("left_hip", "right_hip"), "pelvis", ("femur_l", "femur_r")
        ),
        InstantHMRPoint(
            "Shoulders",
            ("left_shoulder", "right_shoulder"),
            "torso",
            ("humerus_l", "humerus_r"),
            "Hips",
            weight=0.0,
        ),
        InstantHMRPoint(
            "Head",
            ("left_ear", "right_ear"),
            "head",
            ("head",),
            "Shoulders",
            weight=0.3,
        ),
        *_leg_points(),
        *_arm_points(),
    ),  # fmt: skip
    orientations=(
        InstantHMROrientation(
            "Pelvis",
            "pelvis",
            (("right_hip",), ("left_hip",)),
            (("left_hip", "right_hip"), ("left_shoulder", "right_shoulder")),
            (("femur_r",), ("femur_l",)),
            (("femur_l", "femur_r"), ("humerus_l", "humerus_r")),
            canonical_secondary=(0.0, 0.0, 1.0),
            weight=0.5,
            axis_to="femur_l",
            axis_weight=0.25,
        ),
        InstantHMROrientation(
            "Torso",
            "torso",
            (("left_hip", "right_hip"), ("left_shoulder", "right_shoulder")),
            (("right_shoulder",), ("left_shoulder",)),
            (("femur_l", "femur_r"), ("humerus_l", "humerus_r")),
            (("humerus_r",), ("humerus_l",)),
            weight=0.1,
        ),
        InstantHMROrientation(
            "Head",
            "head",
            (("right_ear",), ("left_ear",)),
            (("left_ear", "right_ear"), ("nose", "left_eye", "right_eye")),
            canonical_primary=(0.0, 1.0, 0.0),
            canonical_secondary=(1.0, 0.0, 0.0),
            weight=0.03,
        ),
        *_leg_orientations(),
        *_arm_orientations(),
    ),  # fmt: skip
    calibration={"shoulder_elv_l": np.pi / 2, "shoulder_elv_r": np.pi / 2},
    feet=(("calcn_l", "toes_l"), ("calcn_r", "toes_r")),
    twist_pairs=(
        ("femur_l", "femur_r"),
        ("tibia_l", "tibia_r"),
        ("calcn_l", "calcn_r"),
    ),
    twist_reference="pelvis",
    coupling_term="knee_angle_rotation3_l",
    period_joint="knee_angle_r",
    posture=(
        ("flex_extension", 0.0, 0.02),
        ("elv_angle_l", 0.0, 0.01),
        ("elv_angle_r", 0.0, 0.01),
        ("mtp_angle_l", 0.0, 0.02),
        ("mtp_angle_r", 0.0, 0.02),
    ),  # fmt: skip
    hinges=tuple(
        InstantHMRHinge(
            f"elbow_flexion{m}",
            (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"),
            (f"humerus{m}", f"ulna{m}", f"lunate{m}"),
        )
        for side, m in (("left", "_l"), ("right", "_r"))
    ),  # fmt: skip
)

INSTANTHMR_MYOLEGS = InstantHMRMap(
    name="instanthmr-mhr70->myolegs",
    version="1",
    points=(
        InstantHMRPoint(
            "Hips", ("left_hip", "right_hip"), "pelvis", ("femur_l", "femur_r")
        ),
        *_leg_points(),
    ),  # fmt: skip
    orientations=(
        InstantHMROrientation(
            "Pelvis",
            "pelvis",
            (("right_hip",), ("left_hip",)),
            (("left_hip", "right_hip"), ("left_shoulder", "right_shoulder")),
            (("femur_r",), ("femur_l",)),
            canonical_secondary=(0.0, 0.0, 1.0),
            weight=0.5,
            axis_to="femur_l",
            axis_weight=0.35,
        ),
        *_leg_orientations(),
    ),  # fmt: skip
    feet=(("calcn_l", "toes_l"), ("calcn_r", "toes_r")),
    twist_pairs=(
        ("femur_l", "femur_r"),
        ("tibia_l", "tibia_r"),
        ("calcn_l", "calcn_r"),
    ),
    twist_reference="pelvis",
    coupling_term="knee_angle_rotation3_l",
    period_joint="knee_angle_r",
    posture=(("mtp_angle_l", 0.0, 0.02), ("mtp_angle_r", 0.0, 0.02)),
)

INSTANTHMR_MAPS: dict[str, InstantHMRMap] = {
    "myofullbody": INSTANTHMR_MYOFULLBODY,
    "myolegs": INSTANTHMR_MYOLEGS,
}


@dataclass
class Calibration:
    """An :class:`InstantHMRMap` resolved on a compiled target model."""

    lmap: LandmarkMap
    model_pos: np.ndarray
    rot_offset: np.ndarray
    qpos: np.ndarray
    leg_length: float
    canon_to_model: np.ndarray
    hinge_table: tuple[np.ndarray, ...] = ()


def _unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def _triad_checked(
    u: np.ndarray,
    v: np.ndarray,
    *,
    eps: float = 1e-8,
    sin_tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Build orthonormal right-handed ``(..., 3, 3)`` rotation matrices and validity mask.

    Entries whose inputs are non-finite, near-zero, or collinear (``||e0 x v_hat|| <= sin_tol``)
    are marked ``False`` in the returned mask and replaced by a deterministic proper
    rotation in ``SO(3)`` (``det = +1``) so downstream quaternion conversion never sees
    a singular matrix.
    """
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    u_fin = np.isfinite(u_arr).all(axis=-1)
    v_fin = np.isfinite(v_arr).all(axis=-1)
    u_safe = np.where(u_fin[..., None], u_arr, 0.0)
    v_safe = np.where(v_fin[..., None], v_arr, 0.0)

    u_norm = np.linalg.norm(u_safe, axis=-1)
    v_norm = np.linalg.norm(v_safe, axis=-1)
    e0 = u_safe / np.maximum(u_norm[..., None], eps)
    v_hat = v_safe / np.maximum(v_norm[..., None], eps)

    cross = np.cross(e0, v_hat)
    cross_norm = np.linalg.norm(cross, axis=-1)
    valid = u_fin & v_fin & (u_norm > eps) & (v_norm > eps) & (cross_norm > sin_tol)

    e2 = cross / np.maximum(cross_norm[..., None], eps)
    e1 = np.cross(e2, e0)
    R = np.stack([e0, e1, e2], axis=-1)

    valid_arr = np.asarray(valid, dtype=bool)
    if not valid_arr.all():
        R_flat = R.reshape(-1, 3, 3)
        e0_flat = e0.reshape(-1, 3)
        u_ok_flat = np.asarray(u_fin & (u_norm > eps), dtype=bool).reshape(-1)
        for idx in np.flatnonzero(~valid_arr.reshape(-1)):
            if u_ok_flat[idx]:
                axis = e0_flat[idx]
                basis = np.zeros(3, dtype=np.float64)
                basis[int(np.argmin(np.abs(axis)))] = 1.0
                e2_fb = np.cross(axis, basis)
                e2_fb /= np.linalg.norm(e2_fb)
                e1_fb = np.cross(e2_fb, axis)
                R_flat[idx] = np.column_stack([axis, e1_fb, e2_fb])
            else:
                R_flat[idx] = np.eye(3, dtype=np.float64)
    return R, valid_arr


def _triad(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _triad_checked(u, v)[0]


def _femur_secondary(
    u_thigh: np.ndarray, u_shank: np.ndarray, u_foot: np.ndarray
) -> np.ndarray:
    return np.cross(u_thigh, u_shank) + 0.25 * _unit(np.cross(u_foot, u_shank))


def _canonical_axes(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    up = np.array([0.0, 0.0, 1.0])
    lr = d.xpos[m.body("femur_l").id] - d.xpos[m.body("femur_r").id]
    left = _unit(lr - np.dot(lr, up) * up)
    return np.column_stack([np.cross(left, up), left, up])


def calibrate(model: mujoco.MjModel, imap: InstantHMRMap) -> Calibration:
    """Resolve *imap* on *model* in its T-pose calibration configuration."""
    m = model
    d, q = _calibration_pose(m, imap.calibration)
    R_canon = _canonical_axes(m, d)
    body_names = {m.body(i).name for i in range(m.nbody)}
    names = [p.name for p in imap.points]
    pos = np.stack(
        [np.mean([d.xpos[m.body(b).id] for b in p.model], axis=0) for p in imap.points]
    )
    landmarks = _body_landmarks(m, d, imap.points, pos)

    def mvec(
        pair: tuple[tuple[str, ...], tuple[str, ...]], canon: tuple[float, float, float]
    ) -> np.ndarray:
        if pair and all(b in body_names for grp in pair for b in grp):
            return np.mean([d.xpos[m.body(b).id] for b in pair[1]], 0) - np.mean([d.xpos[m.body(b).id] for b in pair[0]], 0)  # fmt: skip
        return R_canon @ np.asarray(canon, np.float64)

    rot_offsets = []
    for o in imap.orientations:
        u = mvec(o.model_primary, o.canonical_primary)
        if o.kind == "femur":
            sfx = "_l" if o.side == "left" else "_r"
            v = _femur_secondary(
                _unit(u),
                _unit(
                    d.xpos[m.body(f"talus{sfx}").id] - d.xpos[m.body(f"tibia{sfx}").id]
                ),
                _unit(
                    d.xpos[m.body(f"toes{sfx}").id] - d.xpos[m.body(f"calcn{sfx}").id]
                ),
            )
        else:
            v = mvec(o.model_secondary, o.canonical_secondary)
        R_cal, ok_cal = _triad_checked(u, v)
        if not bool(ok_cal):
            raise ValueError(f"{o.name}: degenerate calibration triad on model")
        rot_offsets.append(R_cal.T @ d.xmat[m.body(o.body).id].reshape(3, 3))

    orientations = _orientation_targets(m, d, imap.orientations, source_attr="name")
    lmap = LandmarkMap(
        name=imap.name, version=imap.version, landmarks=landmarks,
        orientations=orientations, feet=imap.feet, twist_pairs=imap.twist_pairs,
        twist_reference=imap.twist_reference, coupling_term=imap.coupling_term,
        period_joint=imap.period_joint,
        posture=imap.posture + tuple((h.joint, 0.0, h.weight) for h in imap.hinges),
    )  # fmt: skip
    leg_len = float(
        0.5
        * sum(
            np.linalg.norm(pos[names.index(b)] - pos[names.index(a)])
            for a, b in imap.leg_edges
        )
    )
    return Calibration(lmap, pos, np.stack(rot_offsets), q, leg_len, R_canon, tuple(_hinge_table(m, d, q, h) for h in imap.hinges))  # fmt: skip


def segment_orientations_with_mask(
    joints_world: np.ndarray, imap: InstantHMRMap
) -> tuple[np.ndarray, np.ndarray]:
    """Construct ``(T, O, 3, 3)`` proper segment rotations and ``(T, O)`` non-degeneracy mask."""

    def jvec(pair: tuple[tuple[str, ...], tuple[str, ...]]) -> np.ndarray:
        return joints_world[:, [_JIDX[j] for j in pair[1]]].mean(1) - joints_world[:, [_JIDX[j] for j in pair[0]]].mean(1)  # fmt: skip

    frames, masks = [], []
    for o in imap.orientations:
        u = jvec(o.primary)
        extra_ok: np.ndarray | bool = True
        if o.kind == "femur":
            s = o.side
            shank = (
                joints_world[:, _JIDX[f"{s}_ankle"]]
                - joints_world[:, _JIDX[f"{s}_knee"]]
            )
            fore = 0.5 * (
                joints_world[:, _JIDX[f"{s}_big_toe_tip"]]
                + joints_world[:, _JIDX[f"{s}_small_toe_tip"]]
            )
            foot = fore - joints_world[:, _JIDX[f"{s}_heel"]]
            extra_ok = (np.linalg.norm(shank, axis=-1) > 1e-8) & (
                np.linalg.norm(foot, axis=-1) > 1e-8
            )
            v = _femur_secondary(_unit(u), _unit(shank), _unit(foot))
        else:
            v = jvec(o.secondary)
        R_o, ok_o = _triad_checked(u, v)
        frames.append(R_o)
        masks.append(ok_o & extra_ok)
    return np.stack(frames, axis=1), np.stack(masks, axis=1)


def segment_orientations(joints_world: np.ndarray, imap: InstantHMRMap) -> np.ndarray:
    """Construct ``(T, O, 3, 3)`` proper segment rotation matrices from world MHR70 keypoints."""
    return segment_orientations_with_mask(joints_world, imap)[0]


def knee_flexion(joints_world: np.ndarray) -> dict[str, np.ndarray]:
    """Signed ``(T,)`` knee flexion angle (rad; negative is hyperextension) per leg."""
    out: dict[str, np.ndarray] = {}
    for side, key in (("left", "l"), ("right", "r")):
        u_thigh = _unit(
            joints_world[:, _JIDX[f"{side}_knee"]]
            - joints_world[:, _JIDX[f"{side}_hip"]]
        )
        u_shank = _unit(
            joints_world[:, _JIDX[f"{side}_ankle"]]
            - joints_world[:, _JIDX[f"{side}_knee"]]
        )
        fore = 0.5 * (
            joints_world[:, _JIDX[f"{side}_big_toe_tip"]]
            + joints_world[:, _JIDX[f"{side}_small_toe_tip"]]
        )
        u_foot = _unit(fore - joints_world[:, _JIDX[f"{side}_heel"]])
        ang = np.arccos(np.clip((u_thigh * u_shank).sum(-1), -1.0, 1.0))
        sign = np.where(
            (np.cross(u_thigh, u_shank) * np.cross(u_foot, u_shank)).sum(-1) < 0.0,
            -1.0,
            1.0,
        )
        out[key] = sign * ang
    return out


@dataclass
class InstantHMRSource:
    """InstantHMR MHR70 ``.npz`` clips as a :class:`~.retarget.ClipSource`."""

    imap: InstantHMRMap
    calibration: Calibration
    cam_to_world: np.ndarray | None = None
    trim: tuple[int, int] = (0, 0)
    zero_xy_start: bool = True
    contact_height: float = 0.035
    contact_speed: float = 0.5
    ground_sigma: float = 0.05
    target_sigma: float = 0.015
    max_gap_s: float = 0.25
    metric_targets: bool = False
    name: str = "instanthmr-mhr70"
    _last: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def _world_joints(self, clip: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if self.cam_to_world is not None:
            R_cw = _check_rotation(self.cam_to_world)
        elif clip["cam_to_world"] is not None:
            R_cw = clip["cam_to_world"]
        else:
            R_cw = self.calibration.canon_to_model @ CAM_TO_WORLD
        n = len(clip["joints_3d_cam"])
        sl = slice(self.trim[0], n - self.trim[1] if self.trim[1] > 0 else n)
        j_cam = clip["joints_3d_cam"][sl]
        if len(j_cam) < 2:
            raise ValueError(
                f"trimmed clip needs at least 2 frames, got {len(j_cam)} from {n}"
            )
        return j_cam @ R_cw.T, clip["valid"][sl]

    def targets(
        self,
        path: str | Path | dict[str, Any],
        model: mujoco.MjModel,
        lmap: LandmarkMap,
    ) -> ClipTargets:
        """Construct scaled landmark, orientation, contact and hinge targets for *path*."""
        c, imap = self.calibration, self.imap
        clip = _as_clip(path, max_gap_s=self.max_gap_s)
        jw, valid = self._world_joints(clip)
        fps, T = clip["fps"], len(jw)
        names = [p.name for p in imap.points]

        x = [jw[:, [_JIDX[j] for j in p.mhr]].mean(axis=1) for p in imap.points]
        subj_len = {}
        for k, p in enumerate(imap.points[1:], 1):
            d_edge = np.linalg.norm(x[k] - x[names.index(p.parent)], axis=-1)
            subj_len[k] = max(
                float(np.median(d_edge[valid] if valid.any() else d_edge)), 1e-6
            )

        s_root = (
            1.0
            if self.metric_targets
            else c.leg_length
            / max(0.5 * sum(subj_len[names.index(b)] for _, b in imap.leg_edges), 1e-6)
        )
        sole = jw[:, [_JIDX[j] for side in imap.contact_keypoints for j in side]]
        sole_floor = np.percentile(sole[..., 2], 2, axis=0)
        floor = float(np.median(sole_floor))

        out = np.empty((len(names), T, 3))
        root = x[0].copy()
        root[:, :2] = s_root * (
            root[:, :2] - root[0, :2] if self.zero_xy_start else root[:, :2]
        )
        root[:, 2] = s_root * (root[:, 2] - floor)
        out[0] = root

        ratio: dict[int, float] = {}
        for k, p in enumerate(imap.points[1:], 1):
            par = names.index(p.parent)
            model_edge = c.model_pos[k] - c.model_pos[par]
            model_len = float(np.linalg.norm(model_edge))
            ratio[k] = model_len / subj_len[k]
            edge = x[k] - x[par]
            edge_norm = np.linalg.norm(edge, axis=-1, keepdims=True)
            cal_dir = model_edge / max(model_len, 1e-9)
            unit_edge = np.where(
                edge_norm > 1e-8, edge / np.maximum(edge_norm, 1e-9), cal_dir
            )
            out[k] = out[par] + model_len * unit_edge

        pos = out[[names.index(lm.source) for lm in lmap.landmarks]].transpose(1, 0, 2)
        R_seg, ori_valid = segment_orientations_with_mask(jw, imap)
        quat = _mat_to_quat(np.einsum("toab,obc->toac", R_seg, c.rot_offset))

        if self.target_sigma > 0:
            pos = _smooth(pos, self.target_sigma * fps)
            quat = _smooth_quat(quat, self.target_sigma * fps, mask=ori_valid)

        posture = None
        if imap.hinges:
            posture = np.tile([v for _, v, _ in lmap.posture], (T, 1))
            for k, (h, tab) in enumerate(zip(imap.hinges, c.hinge_table)):
                a, b, e = (jw[:, _JIDX[j]] for j in h.mhr)
                theta = _smooth(_segment_angle(a, b, e), self.target_sigma * fps)
                posture[:, len(imap.posture) + k] = np.interp(theta, tab[1], tab[0])

        contact = _foot_contacts(
            sole,
            sole_floor,
            fps,
            len(imap.contact_keypoints[0]),
            self.contact_height,
            self.contact_speed,
        )

        k_pelvis = [o.body for o in lmap.orientations].index("pelvis")
        quat0 = quat[0].copy()
        if not ori_valid[0, k_pelvis]:
            ok_p = np.flatnonzero(ori_valid[:, k_pelvis])
            if len(ok_p):
                quat0[k_pelvis] = quat[int(ok_p[0]), k_pelvis]
            else:
                d_cal = mujoco.MjData(model)
                d_cal.qpos[:] = c.qpos
                mujoco.mj_kinematics(model, d_cal)
                quat0[k_pelvis] = d_cal.xquat[model.body("pelvis").id]
        seed = _start_pose(model, c.qpos, c.model_pos[0], out[0, 0], quat0, lmap)

        degen_frames = int((~ori_valid).any(axis=1).sum())
        flags = merge_quality_flags(
            clip.get("quality_flags"),
            ["degenerate_orientation"] if degen_frames > 0 else None,
        )
        quat_mask = None if ori_valid.all() else ori_valid.astype(np.float64)
        self._last = {
            "floor": floor,
            "s_root": float(s_root),
            "contact": contact,
            "frames_source": len(clip["joints_3d_cam"]),
            "interpolated_frames": int((~valid).sum()),
            "endpoint_trimmed_frames": int(clip.get("endpoint_trimmed_frames", 0)),
            "segment_range": list(
                clip.get("segment_range", [0, len(clip["joints_3d_cam"])])
            ),
            "degenerate_orientation_frames": degen_frames,
            "quality_flags": flags,
            "edge_ratio": {names[k]: float(r) for k, r in ratio.items()},
            "knee_flexion": knee_flexion(jw),
        }
        return ClipTargets(fps, pos, quat, seed, contact, posture, quat_mask=quat_mask)

    def finish(
        self, ik: LandmarkIK, targets: ClipTargets, qpos: np.ndarray
    ) -> tuple[np.ndarray, ClipTargets, dict[str, Any]]:
        """Ground stance frames to ``z = 0`` while interpolating aerial flight phases."""
        info = self._last
        contact = info["contact"]
        qpos, targets, dz = _ground_clip(
            ik.model, ik.lmap, targets, qpos, contact, self.ground_sigma
        )
        meta = {
            "instanthmr_map": self.imap.name,
            "instanthmr_map_version": self.imap.version,
            "trim": list(self.trim),
            "segment_range": info["segment_range"],
            "frames_source": info["frames_source"],
            "interpolated_frames": info["interpolated_frames"],
            "endpoint_trimmed_frames": info["endpoint_trimmed_frames"],
            "degenerate_orientation_frames": info["degenerate_orientation_frames"],
            "quality_flags": list(info["quality_flags"]),
            "subject_floor_m": info["floor"],
            "root_scale": info["s_root"],
            "edge_ratio": info["edge_ratio"],
            "ground_shift_m": {
                "mean": float(dz.mean()),
                "min": float(dz.min()),
                "max": float(dz.max()),
            },
            "contact_fraction": [float(x) for x in contact.mean(0)],
            "source_knee_flexion_deg": _knee_flexion_meta(info["knee_flexion"]),
        }
        return qpos, targets, meta


def instanthmr_source(
    model: mujoco.MjModel, imap: InstantHMRMap = INSTANTHMR_MYOFULLBODY, **kwargs: Any
) -> tuple[InstantHMRSource, LandmarkMap]:
    """Construct a calibrated :class:`InstantHMRSource` and solver :class:`LandmarkMap`."""
    cal = calibrate(model, imap)
    return InstantHMRSource(imap, cal, **kwargs), cal.lmap


def landmark_bodies(imap: InstantHMRMap) -> tuple[str, ...]:
    """Sorted model body names carrying landmark, orientation or hinge targets."""
    names = {b for p in imap.points for b in (p.body, *p.model)}
    names |= {b for o in imap.orientations for b in (o.body, o.axis_to) if b}
    names |= {b for h in imap.hinges for b in h.model}
    return tuple(sorted(names))


def map_dict(imap: InstantHMRMap) -> dict[str, Any]:
    """JSON-serialisable dictionary form of *imap*."""
    return asdict(imap)


def default_instanthmr_map(
    model_or_spec: mujoco.MjModel | mujoco.MjSpec,
) -> InstantHMRMap:
    """Select :data:`INSTANTHMR_MYOFULLBODY` or :data:`INSTANTHMR_MYOLEGS` from body names."""
    if isinstance(model_or_spec, mujoco.MjModel):
        names = {model_or_spec.body(i).name for i in range(model_or_spec.nbody)}
    else:
        names = {b.name for b in model_or_spec.bodies}
    if {"pelvis", "torso", "humerus_l", "humerus_r", "head"} <= names:
        return INSTANTHMR_MYOFULLBODY
    if {"pelvis", "femur_l", "femur_r", "tibia_l", "tibia_r"} <= names:
        return INSTANTHMR_MYOLEGS
    raise ValueError(
        "Could not infer InstantHMRMap from model bodies; pass imap explicitly"
    )


def estimate_instanthmr_scales(
    clip: str | Path | dict[str, Any],
    model: mujoco.MjModel,
    imap: InstantHMRMap | None = None,
    *,
    cam_to_world: np.ndarray | None = None,
    trim: tuple[int, int] = (0, 0),
    max_gap_s: float = 0.25,
    bilateral_symmetry: bool = True,
    bounds: tuple[float, float] = (0.5, 2.0),
) -> dict[str, float]:
    """Estimate isotropic per-segment scale factors from metric MHR70 3D landmarks.

    Segment lengths are measured as medians over valid frames of *clip* and divided by
    the corresponding bone-frame distances of *model* in its calibration T-pose.

    Args:
        clip: InstantHMR ``.npz`` path or loaded clip dictionary.
        model: Unscaled target MuJoCo model.
        imap: Optional :class:`InstantHMRMap` (inferred from *model* when ``None``).
        cam_to_world: Optional ``(3, 3)`` camera-to-world rotation.
        trim: ``(start, end)`` frames dropped before measuring segment lengths.
        max_gap_s: Maximum missing-frame gap duration (s) allowed when loading *clip*.
        bilateral_symmetry: Average left and right limb scales to keep the model
            laterally symmetric.
        bounds: ``(min_scale, max_scale)`` clamp on estimated ratios.

    Returns:
        ``{body_name: scale_factor}`` mapping segment root bodies to scalar scales.
    """
    if bounds[0] <= 0 or bounds[0] > bounds[1]:
        raise ValueError(f"invalid scale bounds {bounds}")
    imap = imap or default_instanthmr_map(model)
    cal = calibrate(model, imap)
    src = InstantHMRSource(
        imap, cal, cam_to_world=cam_to_world, trim=trim, max_gap_s=max_gap_s
    )
    jw, valid = src._world_joints(_as_clip(clip, max_gap_s=max_gap_s))

    d = mujoco.MjData(model)
    d.qpos[:] = cal.qpos
    mujoco.mj_kinematics(model, d)
    body_names = {model.body(i).name for i in range(model.nbody)}

    def med_dist(mhr_a: tuple[str, ...], mhr_b: tuple[str, ...]) -> float:
        pa = jw[:, [_JIDX[j] for j in mhr_a]].mean(axis=1)
        pb = jw[:, [_JIDX[j] for j in mhr_b]].mean(axis=1)
        dist = np.linalg.norm(pb - pa, axis=-1)
        return max(float(np.median(dist[valid] if valid.any() else dist)), 1e-6)

    def mod_dist(bodies_a: tuple[str, ...], bodies_b: tuple[str, ...]) -> float:
        pa = np.mean([d.xpos[model.body(b).id] for b in bodies_a], axis=0)
        pb = np.mean([d.xpos[model.body(b).id] for b in bodies_b], axis=0)
        return max(float(np.linalg.norm(pb - pa)), 1e-6)

    def clamp(val: float) -> float:
        return float(np.clip(val, bounds[0], bounds[1]))

    scales: dict[str, float] = {}
    s_pelvis = clamp(
        med_dist(("left_hip",), ("right_hip",)) / mod_dist(("femur_l",), ("femur_r",))
    )
    scales["pelvis"] = s_pelvis

    L_torso = med_dist(("left_hip", "right_hip"), ("left_shoulder", "right_shoulder"))
    if {"sacrum", "humerus_l", "humerus_r"} <= body_names:
        root_0 = d.xpos[model.body("sacrum").id]
        v_lo = (
            0.5 * (d.xpos[model.body("femur_l").id] + d.xpos[model.body("femur_r").id])
            - root_0
        )
        has_clav = {"clavicle_l", "clavicle_r"} <= body_names
        if has_clav:
            clav_mid = 0.5 * (
                d.xpos[model.body("clavicle_l").id]
                + d.xpos[model.body("clavicle_r").id]
            )
            hum_mid = 0.5 * (
                d.xpos[model.body("humerus_l").id] + d.xpos[model.body("humerus_r").id]
            )
            v_up = clav_mid - root_0
            v_hum_off = hum_mid - clav_mid
            L_sh = med_dist(("left_shoulder",), ("right_shoulder",))
            c_diff = (
                d.xpos[model.body("clavicle_l").id]
                - d.xpos[model.body("clavicle_r").id]
            )
            h_diff = (
                d.xpos[model.body("humerus_l").id] - d.xpos[model.body("clavicle_l").id]
            ) - (
                d.xpos[model.body("humerus_r").id] - d.xpos[model.body("clavicle_r").id]
            )
        else:
            v_up = (
                0.5
                * (
                    d.xpos[model.body("humerus_l").id]
                    + d.xpos[model.body("humerus_r").id]
                )
                - root_0
            )
            v_hum_off = np.zeros(3)
            L_sh = 0.0
            c_diff = h_diff = np.zeros(3)

        s_trunk = s_clav = s_pelvis
        a = float(np.dot(v_up, v_up))
        ah = float(np.dot(h_diff, h_diff))
        for _ in range(4):
            v_rem = s_clav * v_hum_off - s_pelvis * v_lo
            b_coef = 2.0 * float(np.dot(v_up, v_rem))
            c_coef = float(np.dot(v_rem, v_rem)) - L_torso**2
            disc = max(b_coef * b_coef - 4.0 * a * c_coef, 0.0)
            s_trunk = clamp((-b_coef + float(np.sqrt(disc))) / max(2.0 * a, 1e-12))
            if has_clav and ah > 1e-8:
                bh = 2.0 * s_trunk * float(np.dot(c_diff, h_diff))
                ch = (s_trunk**2) * float(np.dot(c_diff, c_diff)) - L_sh**2
                s_clav = clamp(
                    (-bh + float(np.sqrt(max(bh * bh - 4.0 * ah * ch, 0.0))))
                    / (2.0 * ah)
                )
        scales["sacrum"] = s_trunk
        if has_clav and ah > 1e-8:
            scales["clavicle_l"] = s_clav
            scales["clavicle_r"] = s_clav
    elif "torso" in body_names:
        scales["torso"] = clamp(L_torso / 0.5824)

    limb_pairs: list[
        tuple[
            str,
            str,
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
        ]
    ] = [
        (
            "femur_l",
            "femur_r",
            ("left_hip",),
            ("left_knee",),
            ("right_hip",),
            ("right_knee",),
            ("femur_l",),
            ("tibia_l",),
            ("femur_r",),
            ("tibia_r",),
        ),
        (
            "tibia_l",
            "tibia_r",
            ("left_knee",),
            ("left_ankle",),
            ("right_knee",),
            ("right_ankle",),
            ("tibia_l",),
            ("talus_l",),
            ("tibia_r",),
            ("talus_r",),
        ),
        (
            "talus_l",
            "talus_r",
            ("left_heel",),
            ("left_big_toe_tip", "left_small_toe_tip"),
            ("right_heel",),
            ("right_big_toe_tip", "right_small_toe_tip"),
            ("calcn_l",),
            ("toes_l",),
            ("calcn_r",),
            ("toes_r",),
        ),
    ]
    if {
        "humerus_l",
        "humerus_r",
        "ulna_l",
        "ulna_r",
        "lunate_l",
        "lunate_r",
    } <= body_names:
        limb_pairs.extend(
            [
                (
                    "humerus_l",
                    "humerus_r",
                    ("left_shoulder",),
                    ("left_elbow",),
                    ("right_shoulder",),
                    ("right_elbow",),
                    ("humerus_l",),
                    ("ulna_l",),
                    ("humerus_r",),
                    ("ulna_r",),
                ),
                (
                    "ulna_l",
                    "ulna_r",
                    ("left_elbow",),
                    ("left_wrist",),
                    ("right_elbow",),
                    ("right_wrist",),
                    ("ulna_l",),
                    ("lunate_l",),
                    ("ulna_r",),
                    ("lunate_r",),
                ),
            ]
        )
    if {"lunate_l", "lunate_r", "3proxph_l", "3proxph_r"} <= body_names:
        limb_pairs.append(
            (
                "lunate_l",
                "lunate_r",
                ("left_wrist",),
                ("left_middle_third_joint",),
                ("right_wrist",),
                ("right_middle_third_joint",),
                ("lunate_l",),
                ("3proxph_l",),
                ("lunate_r",),
                ("3proxph_r",),
            )
        )

    for bl, br, mpl, mdl, mpr, mdr, bpl, bdl, bpr, bdr in limb_pairs:
        sl = med_dist(mpl, mdl) / mod_dist(bpl, bdl)
        sr = med_dist(mpr, mdr) / mod_dist(bpr, bdr)
        if bilateral_symmetry:
            scales[bl] = scales[br] = clamp(0.5 * (sl + sr))
        else:
            scales[bl], scales[br] = clamp(sl), clamp(sr)
    return scales


def _sole_lowest_z(model: mujoco.MjModel, qpos: np.ndarray) -> float | None:
    body_names = {model.body(i).name for i in range(model.nbody)}
    foot_names = ("calcn_l", "toes_l", "calcn_r", "toes_r")
    if not all(b in body_names for b in foot_names):
        return None
    foot_ids = {model.body(b).id for b in foot_names}
    geoms = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] in foot_ids
        and (model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0)
    ]
    if not geoms:
        return None
    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    mujoco.mj_kinematics(model, d)
    return float(min(_geom_lowest_z(model, d, g) for g in geoms))


def scale_spec_from_instanthmr(
    spec: mujoco.MjSpec,
    scales: dict[str, float | tuple[float, float, float]],
    *,
    body_mass: float | None = None,
) -> mujoco.MjSpec:
    """Return a subject-scaled copy of *spec* from per-segment scale factors.

    Delegates to ``myo_sim.scaling.scale_spec`` when installed, and otherwise applies
    exact isotropic per-segment scaling across bone offsets, slide joints and polynomial
    couplings, wrapping/collision geoms, meshes, body masses/inertias, spatial tendons,
    muscle ``lengthrange`` (preserving peak isometric force) and sole floor alignment.
    """
    if body_mass is not None and (not np.isfinite(body_mass) or body_mass <= 0):
        raise ValueError(f"body_mass must be finite and positive, got {body_mass}")

    if importlib.util.find_spec("myo_sim.scaling") is not None:
        mod = importlib.import_module("myo_sim.scaling")
        if hasattr(mod, "scale_spec"):
            return mod.scale_spec(spec, scales, total_mass=body_mass)

    scalar_scales: dict[str, float] = {}
    for k, v in scales.items():
        arr = np.asarray(v, dtype=np.float64).ravel()
        if arr.size == 1:
            s = float(arr[0])
        elif arr.size == 3 and np.allclose(arr, arr[0], rtol=1e-5, atol=1e-6):
            s = float(arr[0])
        else:
            raise ValueError(
                f"Isotropic scaler requires scalar or uniform 3-vector for {k!r}, got {v}"
            )
        if not np.isfinite(s) or s <= 0:
            raise ValueError(f"Scale for {k!r} must be finite and positive, got {s}")
        scalar_scales[k] = s

    m0 = spec.compile()
    body_names = {b.name for b in spec.bodies}
    unknown = sorted(set(scalar_scales) - body_names)
    if unknown:
        raise ValueError(f"Unknown body names in scales: {unknown}")

    q0 = m0.qpos0.copy()
    project_joint_couplings(JointCouplings.from_model(m0), q0)
    d0 = mujoco.MjData(m0)
    d0.qpos[:] = q0
    mujoco.mj_forward(m0, d0)
    z0 = _sole_lowest_z(m0, q0)

    out = spec.copy()
    for i, ac in enumerate(out.actuators):
        if ac.gaintype == mujoco.mjtGain.mjGAIN_MUSCLE and ac.gainprm[2] <= 0:
            f_peak = float(m0.actuator_gainprm[i, 3]) / max(
                float(m0.actuator_acc0[i]), 1e-10
            )
            gp = np.array(ac.gainprm, dtype=np.float64)
            gp[2] = f_peak
            ac.gainprm = gp
            if ac.biastype == mujoco.mjtBias.mjBIAS_MUSCLE:
                bp = np.array(ac.biasprm, dtype=np.float64)
                bp[2] = f_peak
                ac.biasprm = bp

    top_roots = [
        b
        for b in out.worldbody.bodies
        if any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in b.joints)
    ] or list(out.worldbody.bodies)

    subject_bodies: list[Any] = []
    body_scales: dict[str, float] = {}
    jnt_parent_scale: dict[str, float] = {}
    jnt_types: dict[str, Any] = {}
    mesh_scales: dict[str, float] = {}

    def walk(body: Any, s_par: float, is_root: bool) -> None:
        s_b = scalar_scales.get(body.name, s_par)
        body_scales[body.name] = s_b
        subject_bodies.append(body)
        if not is_root:
            body.pos = s_par * np.asarray(body.pos, dtype=np.float64)
        for j in body.joints:
            jnt_parent_scale[j.name] = s_par
            jnt_types[j.name] = j.type
            if j.type != mujoco.mjtJoint.mjJNT_FREE:
                j.pos = s_b * np.asarray(j.pos, dtype=np.float64)
            if j.type == mujoco.mjtJoint.mjJNT_SLIDE:
                j.range = s_par * np.asarray(j.range, dtype=np.float64)
                j.ref = s_par * float(j.ref)
                j.springref = s_par * float(j.springref)
                j.margin = s_par * float(j.margin)
                j.stiffness = np.asarray(j.stiffness, dtype=np.float64) / s_par
                j.damping = np.asarray(j.damping, dtype=np.float64) / s_par
        for st in body.sites:
            st.pos = s_b * np.asarray(st.pos, dtype=np.float64)
            st.size = s_b * np.asarray(st.size, dtype=np.float64)
        for frm in getattr(body, "frames", ()):
            frm.pos = s_b * np.asarray(frm.pos, dtype=np.float64)
        for g in body.geoms:
            g.pos = s_b * np.asarray(g.pos, dtype=np.float64)
            ft = np.asarray(g.fromto, dtype=np.float64)
            if np.all(np.isfinite(ft)):
                g.fromto = s_b * ft
            if np.isfinite(g.mass) and g.mass > 0:
                g.mass = float(g.mass) * (s_b**3)
            if g.meshname:
                mesh_scales[g.meshname] = s_b
            elif g.type != mujoco.mjtGeom.mjGEOM_PLANE:
                g.size = s_b * np.asarray(g.size, dtype=np.float64)
        if body.mass > 0:
            body.mass = float(body.mass) * (s_b**3)
            body.ipos = s_b * np.asarray(body.ipos, dtype=np.float64)
            fi = np.asarray(body.fullinertia, dtype=np.float64)
            if np.all(np.isfinite(fi)):
                body.fullinertia = (s_b**5) * fi
            else:
                body.inertia = (s_b**5) * np.asarray(body.inertia, dtype=np.float64)
        for child in body.bodies:
            walk(child, s_b, False)

    for rb in top_roots:
        walk(rb, scalar_scales.get(rb.name, 1.0), True)

    for mesh in out.meshes:
        if mesh.name in mesh_scales:
            sm = mesh_scales[mesh.name]
            mesh.scale = sm * np.asarray(mesh.scale, dtype=np.float64)
            mesh.refpos = sm * np.asarray(mesh.refpos, dtype=np.float64)

    for eq in out.equalities:
        if eq.type == mujoco.mjtEq.mjEQ_JOINT:
            a = (
                jnt_parent_scale.get(eq.name1, 1.0)
                if jnt_types.get(eq.name1) == mujoco.mjtJoint.mjJNT_SLIDE
                else 1.0
            )
            bm = (
                jnt_parent_scale.get(eq.name2, 1.0)
                if jnt_types.get(eq.name2) == mujoco.mjtJoint.mjJNT_SLIDE
                else 1.0
            )
            edata = np.array(eq.data, dtype=np.float64)
            for k in range(5):
                edata[k] = a * edata[k] / (bm**k)
            eq.data = edata

    for key in out.keys:
        kq = np.array(key.qpos, dtype=np.float64) if len(key.qpos) == m0.nq else None
        kv = np.array(key.qvel, dtype=np.float64) if len(key.qvel) == m0.nv else None
        for jname, jtype in jnt_types.items():
            if jtype == mujoco.mjtJoint.mjJNT_SLIDE:
                jid = m0.joint(jname).id
                sp = jnt_parent_scale[jname]
                if kq is not None:
                    kq[m0.jnt_qposadr[jid]] *= sp
                if kv is not None:
                    kv[m0.jnt_dofadr[jid]] *= sp
        if kq is not None:
            key.qpos = kq
        if kv is not None:
            key.qvel = kv

    m1 = out.compile()
    if body_mass is not None:
        cur_mass = float(sum(m1.body_mass[m1.body(b.name).id] for b in subject_bodies))
        if cur_mass <= 0:
            raise ValueError("Cannot normalize body_mass on a massless model")
        m_ratio = float(body_mass) / cur_mass
        for b in subject_bodies:
            if b.mass > 0:
                b.mass = float(b.mass) * m_ratio
                fi = np.asarray(b.fullinertia, dtype=np.float64)
                if np.all(np.isfinite(fi)):
                    b.fullinertia = m_ratio * fi
                else:
                    b.inertia = m_ratio * np.asarray(b.inertia, dtype=np.float64)
            else:
                for g in b.geoms:
                    if np.isfinite(g.mass) and g.mass > 0:
                        g.mass = float(g.mass) * m_ratio
                    elif np.isfinite(g.density) and g.density > 0:
                        g.density = float(g.density) * m_ratio
    q1 = m1.qpos0.copy()
    project_joint_couplings(JointCouplings.from_model(m1), q1)
    d1 = mujoco.MjData(m1)
    d1.qpos[:] = q1
    mujoco.mj_forward(m1, d1)

    ten_ratio = np.ones(m0.ntendon, dtype=np.float64)
    for t in range(m0.ntendon):
        if d0.ten_length[t] > 1e-6 and d1.ten_length[t] > 1e-6:
            ten_ratio[t] = float(d1.ten_length[t] / d0.ten_length[t])

    for t_idx, td in enumerate(out.tendons):
        rt = ten_ratio[t_idx]
        if abs(rt - 1.0) > 1e-12:
            sl = np.asarray(td.springlength, dtype=np.float64)
            td.springlength = np.where(sl >= 0.0, sl * rt, sl)
            td.stiffness = np.asarray(td.stiffness, dtype=np.float64) / rt
            td.damping = np.asarray(td.damping, dtype=np.float64) / rt
            td.margin = float(td.margin) * rt
            if td.limited == mujoco.mjtLimited.mjLIMITED_TRUE:
                td.range = rt * np.asarray(td.range, dtype=np.float64)

    for a_idx, ac in enumerate(out.actuators):
        if ac.trntype == mujoco.mjtTrn.mjTRN_TENDON:
            tid = int(m0.actuator_trnid[a_idx, 0])
            rt = ten_ratio[tid]
            lr = np.asarray(ac.lengthrange, dtype=np.float64)
            if np.any(lr != 0.0):
                ac.lengthrange = rt * lr

    z1 = _sole_lowest_z(m1, q1)
    if z0 is not None and z1 is not None and top_roots:
        dz = z0 - z1
        rb = top_roots[0]
        rpos = np.array(rb.pos, dtype=np.float64)
        rpos[2] += dz
        rb.pos = rpos
        free_jnts = [
            j for j in range(m0.njnt) if m0.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ]
        if free_jnts:
            z_adr = int(m0.jnt_qposadr[free_jnts[0]]) + 2
            for key in out.keys:
                if len(key.qpos) == m0.nq:
                    kq = np.array(key.qpos, dtype=np.float64)
                    kq[z_adr] += dz
                    key.qpos = kq
    return out


def scale_model_from_instanthmr(
    spec: mujoco.MjSpec,
    clip: str | Path | dict[str, Any],
    imap: InstantHMRMap | None = None,
    *,
    body_mass: float | None = None,
    cam_to_world: np.ndarray | None = None,
    trim: tuple[int, int] = (0, 0),
    max_gap_s: float = 0.25,
    bilateral_symmetry: bool = True,
    bounds: tuple[float, float] = (0.5, 2.0),
) -> tuple[mujoco.MjModel, mujoco.MjSpec, dict[str, float]]:
    """Estimate anthropometric scales from *clip* and compile a subject-scaled model."""
    nominal = spec.compile()
    scales = estimate_instanthmr_scales(
        clip,
        nominal,
        imap,
        cam_to_world=cam_to_world,
        trim=trim,
        max_gap_s=max_gap_s,
        bilateral_symmetry=bilateral_symmetry,
        bounds=bounds,
    )
    scaled_spec = scale_spec_from_instanthmr(spec, scales, body_mass=body_mass)
    return scaled_spec.compile(), scaled_spec, scales


def _short_clip_derivatives(
    model: mujoco.MjModel, qpos: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    n = len(qpos)
    step = np.empty((n - 1, model.nv), dtype=np.float64)
    for t in range(n - 1):
        mujoco.mj_differentiatePos(model, step[t], dt, qpos[t], qpos[t + 1])
    qvel = np.empty((n, model.nv), dtype=np.float64)
    qvel[0], qvel[-1] = step[0], step[-1]
    if n == 3:
        qvel[1] = 0.5 * (step[0] + step[1])
        qacc = np.tile((step[1] - step[0]) / dt, (3, 1))
    else:
        qacc = np.zeros((n, model.nv), dtype=np.float64)
    return qvel, qacc


@dataclass(frozen=True)
class InstantHMRIKResult:
    """Subject-scaled or nominal IK solution for an InstantHMR MHR70 clip."""

    model: mujoco.MjModel
    clip: MotionClip
    qacc: np.ndarray
    targets: ClipTargets
    landmark_error: np.ndarray
    metrics: dict[str, Any]
    scales: dict[str, float] | None = None
    spec: mujoco.MjSpec | None = None

    @property
    def qpos(self) -> np.ndarray:
        return self.clip.qpos

    @property
    def qvel(self) -> np.ndarray:
        assert self.clip.qvel is not None
        return self.clip.qvel

    @property
    def fps(self) -> float:
        assert self.clip.frequency_hz is not None
        return float(self.clip.frequency_hz)


def solve_instanthmr_ik(
    clip: str | Path | dict[str, Any],
    model: mujoco.MjModel | None = None,
    *,
    spec: mujoco.MjSpec | None = None,
    imap: InstantHMRMap | None = None,
    scale: bool = False,
    body_mass: float | None = None,
    bilateral_symmetry: bool = True,
    metric_targets: bool | None = None,
    config: RefitConfig | None = None,
    thresholds: RefitThresholds | None = None,
    lowpass_hz: float | None = None,
    **source_kwargs: Any,
) -> InstantHMRIKResult:
    """Solve full-body or lower-limb IK from an InstantHMR clip, optionally scaling first.

    Args:
        clip: InstantHMR ``.npz`` path or in-memory clip dictionary.
        model: Compiled target ``MjModel`` (optional if *spec* is supplied).
        spec: ``MjSpec`` of the target model (required when ``scale=True``).
        imap: Optional :class:`InstantHMRMap` (auto-detected when ``None``).
        scale: Estimate anthropometric scales from *clip* and scale *spec* before IK.
        body_mass: Optional total subject body mass in kg applied when ``scale=True``.
        bilateral_symmetry: Keep left/right limb scales symmetric when ``scale=True``.
        metric_targets: Preserve metric 1:1 root displacement (defaults to ``scale``).
        config: Optional :class:`~.retarget.RefitConfig` for :class:`~.retarget.LandmarkIK`.
        thresholds: Optional :class:`~.retarget.RefitThresholds` for validation.
        lowpass_hz: Optional zero-phase Butterworth cut-off (Hz) before differentiating
            ``qpos`` into ``qvel`` and ``qacc``.
        **source_kwargs: Extra keyword arguments forwarded to :func:`instanthmr_source`.

    Returns:
        An :class:`InstantHMRIKResult` with the fitted :class:`~myosuite.core.trajectory_io.MotionClip`,
        manifold accelerations ``qacc``, per-landmark errors (m) and validation metrics.
    """
    if scale and spec is None:
        raise ValueError("spec must be provided when scale=True")
    if model is None and spec is None:
        raise ValueError("Provide at least one of model or spec")

    scales: dict[str, float] | None = None
    out_spec = spec
    if scale:
        assert spec is not None
        model, out_spec, scales = scale_model_from_instanthmr(
            spec,
            clip,
            imap,
            body_mass=body_mass,
            cam_to_world=source_kwargs.get("cam_to_world"),
            trim=source_kwargs.get("trim", (0, 0)),
            max_gap_s=source_kwargs.get("max_gap_s", 0.25),
            bilateral_symmetry=bilateral_symmetry,
        )
    elif model is None:
        assert spec is not None
        model = spec.compile()

    assert model is not None
    imap = imap or default_instanthmr_map(model)
    use_metric = scale if metric_targets is None else bool(metric_targets)
    src, lmap = instanthmr_source(
        model, imap, metric_targets=use_metric, **source_kwargs
    )
    ik = LandmarkIK(model, lmap, config)
    targets = src.targets(clip, model, lmap)
    qpos_fit, iters = ik.fit(
        targets.pos, targets.quat, targets.seed, targets.posture, targets.quat_mask
    )
    qpos_fit, targets, extra = src.finish(ik, targets, qpos_fit)

    dt = 1.0 / targets.frequency
    if len(qpos_fit) >= 4:
        kin = clip_kinematics(model, qpos_fit, dt, lowpass_hz=lowpass_hz)
        qpos_out, qvel, qacc = kin.qpos, kin.qvel, kin.qacc
    else:
        if lowpass_hz is not None:
            raise ValueError("lowpass_hz filtering requires at least 4 frames")
        qpos_out = qpos_fit
        qvel, qacc = _short_clip_derivatives(model, qpos_out, dt)

    landmark_err = np.stack(
        [
            np.linalg.norm(ik.landmarks(q) - p, axis=-1)
            for q, p in zip(qpos_out, targets.pos)
        ]
    )
    val_metrics = validate_refit(ik, targets, qpos_out, thresholds)
    joint_names = [
        model.joint(j).name
        for j in sorted(range(model.njnt), key=lambda i: int(model.jnt_qposadr[i]))
        if model.joint(j).name
    ]
    motion_clip = MotionClip(
        qpos=qpos_out,
        qvel=qvel,
        site_xpos=None,
        site_names=None,
        qpos_joint_names=joint_names or None,
        qvel_joint_names=joint_names or None,
        qpos_model_indices=np.arange(model.nq, dtype=np.int32),
        qvel_model_indices=np.arange(model.nv, dtype=np.int32),
        frequency_hz=float(targets.frequency),
        source_path=Path(clip) if isinstance(clip, str | Path) else None,
    )
    metrics: dict[str, Any] = {
        **val_metrics,
        **extra,
        "iterations_mean": float(iters.mean()),
        "iterations_max": int(iters.max()),
        "scaled": bool(scale),
        **({"scales": scales} if scales is not None else {}),
    }
    return InstantHMRIKResult(
        model=model,
        clip=motion_clip,
        qacc=qacc,
        targets=targets,
        landmark_error=landmark_err,
        metrics=metrics,
        scales=scales,
        spec=out_spec,
    )


@dataclass(frozen=True)
class InstantHMRReconstruction:
    """End-to-end scaling, IK and 3D muscle mesh reconstruction from InstantHMR."""

    ik: InstantHMRIKResult
    muscles: MuscleMeshReconstruction

    @property
    def model(self) -> mujoco.MjModel:
        return self.ik.model

    @property
    def scales(self) -> dict[str, float] | None:
        return self.ik.scales

    @property
    def clip(self) -> MotionClip:
        return self.ik.clip


def reconstruct_from_instanthmr(
    clip: str | Path | dict[str, Any],
    model: mujoco.MjModel | None = None,
    *,
    spec: mujoco.MjSpec | None = None,
    imap: InstantHMRMap | None = None,
    scale: bool = True,
    body_mass: float | None = None,
    bilateral_symmetry: bool = True,
    metric_targets: bool | None = None,
    config: RefitConfig | None = None,
    thresholds: RefitThresholds | None = None,
    lowpass_hz: float | None = None,
    activation_mode: str = "shortening",
    mesh_mode: str | Path = "volumetric",
    muscle_scale: float = 1.0,
    so_config: Any | None = None,
    **source_kwargs: Any,
) -> InstantHMRReconstruction:
    """Run subject scaling, IK and 3D muscle mesh reconstruction from an InstantHMR clip."""
    from myosuite.viz.muscle_meshes import reconstruct_muscle_meshes  # noqa: PLC0415

    if scale and spec is None and model is not None:
        scale = False
    ik_res = solve_instanthmr_ik(
        clip,
        model=model,
        spec=spec,
        imap=imap,
        scale=scale,
        body_mass=body_mass,
        bilateral_symmetry=bilateral_symmetry,
        metric_targets=metric_targets,
        config=config,
        thresholds=thresholds,
        lowpass_hz=lowpass_hz,
        **source_kwargs,
    )
    muscles = reconstruct_muscle_meshes(
        ik_res.model,
        ik_res.qpos,
        ik_res.fps,
        qvel=ik_res.qvel,
        qacc=ik_res.qacc,
        activation_mode=activation_mode,
        mesh_mode=mesh_mode,
        muscle_scale=muscle_scale,
        so_config=so_config,
    )
    return InstantHMRReconstruction(ik=ik_res, muscles=muscles)


__all__ = [
    "CAM_TO_WORLD", "INSTANTHMR_MAPS", "INSTANTHMR_MYOFULLBODY", "INSTANTHMR_MYOLEGS",
    "MHR70_JOINTS", "Calibration", "InstantHMRHinge", "InstantHMRIKResult",
    "InstantHMRMap", "InstantHMROrientation", "InstantHMRPoint",
    "InstantHMRReconstruction", "InstantHMRSource", "calibrate",
    "default_instanthmr_map", "estimate_instanthmr_scales", "instanthmr_source",
    "knee_flexion", "landmark_bodies", "load_instanthmr", "map_dict",
    "reconstruct_from_instanthmr", "scale_model_from_instanthmr",
    "scale_spec_from_instanthmr", "segment_orientations",
    "segment_orientations_with_mask", "solve_instanthmr_ik", "split_instanthmr_clips",
]  # fmt: skip
