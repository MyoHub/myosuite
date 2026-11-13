# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""InstantHMR MHR70 motions straight onto a body model: a :class:`~.retarget.ClipSource`."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite.biomechanics.retarget import (
    ClipTargets,
    Landmark,
    LandmarkIK,
    LandmarkMap,
    OrientationTarget,
    _foot_kinematics,
)
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

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


def load_instanthmr(path: str | Path) -> dict[str, Any]:
    """Read an InstantHMR ``.npz`` clip and linearly interpolate dropped frames."""
    with np.load(Path(path), allow_pickle=False) as data:
        files = set(data.files)
        if "joints_3d_cam" in files:
            joints = np.asarray(data["joints_3d_cam"], np.float64)
        elif "joints_3d_local" in files or "joints_3d" in files:
            key = "joints_3d_local" if "joints_3d_local" in files else "joints_3d"
            joints = np.asarray(data[key], np.float64)
            if "cam_trans" in files:
                trans = np.asarray(data["cam_trans"], np.float64)
                if trans.shape != (len(joints), 3):
                    raise ValueError(
                        f"cam_trans shape {trans.shape} != ({len(joints)}, 3)"
                    )
                joints = joints + trans[:, None, :]
        else:
            raise ValueError(f"{path}: expected 'joints_3d_cam' or 'joints_3d_local'")
        fps_key = (
            "fps" if "fps" in files else ("frequency" if "frequency" in files else None)
        )
        if fps_key is None:
            raise ValueError(f"{path}: missing 'fps' (or 'frequency')")
        fps = float(data[fps_key])
        valid_raw = np.asarray(data["valid"], bool) if "valid" in files else None
        cam_R = (
            _check_rotation(data["cam_to_world"]) if "cam_to_world" in files else None
        )

    if joints.ndim != 3 or joints.shape[1:] != (len(MHR70_JOINTS), 3):
        raise ValueError(
            f"{path}: expected joints shape (T, 70, 3), got {joints.shape}"
        )
    if len(joints) < 2:
        raise ValueError(f"{path}: clip needs at least 2 frames, got {len(joints)}")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{path}: fps must be finite and positive, got {fps}")

    finite = np.isfinite(joints).all(axis=(1, 2))
    if valid_raw is not None:
        if valid_raw.shape != (len(joints),):
            raise ValueError(
                f"{path}: valid mask shape {valid_raw.shape} != ({len(joints)},)"
            )
        valid = valid_raw & finite
    else:
        valid = finite
    if int(valid.sum()) < 2:
        raise ValueError(
            f"{path}: need at least 2 valid frames, got {int(valid.sum())}"
        )
    return {
        "joints_3d_cam": _interpolate_gaps(joints, valid),
        "fps": fps,
        "valid": valid,
        "interpolated_frames": int((~valid).sum()),
        "cam_to_world": cam_R,
    }


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


def _triad(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    e0 = _unit(u)
    e2 = _unit(np.cross(e0, v))
    return np.stack([e0, np.cross(e2, e0), e2], axis=-1)


def _femur_secondary(
    u_thigh: np.ndarray, u_shank: np.ndarray, u_foot: np.ndarray
) -> np.ndarray:
    return np.cross(u_thigh, u_shank) + 0.25 * _unit(np.cross(u_foot, u_shank))


def _segment_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    u, v = b - a, c - b
    cos = (u * v).sum(-1) / np.linalg.norm(u, axis=-1) / np.linalg.norm(v, axis=-1)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _hinge_table(
    m: mujoco.MjModel, d: mujoco.MjData, q: np.ndarray, h: InstantHMRHinge
) -> np.ndarray:
    j = m.joint(h.joint).id
    prox, mid, dist = (m.body(b).id for b in h.model)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    tip = d.xmat[mid].reshape(3, 3).T @ (d.xpos[dist] - d.xpos[mid])
    phi = np.linspace(*m.jnt_range[j], 181)
    theta = np.empty_like(phi)
    x = q.copy()
    for k, v in enumerate(phi):
        x[m.jnt_qposadr[j]] = v
        d.qpos[:] = x
        mujoco.mj_kinematics(m, d)
        theta[k] = _segment_angle(
            d.xpos[prox], d.xpos[mid], d.xpos[mid] + d.xmat[mid].reshape(3, 3) @ tip
        )
    if np.any(np.diff(theta) <= 0):
        raise ValueError(f"{h.joint}: segment angle not increasing over the range")
    return np.stack([phi, theta])


def _canonical_axes(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    up = np.array([0.0, 0.0, 1.0])
    lr = d.xpos[m.body("femur_l").id] - d.xpos[m.body("femur_r").id]
    left = _unit(lr - np.dot(lr, up) * up)
    return np.column_stack([np.cross(left, up), left, up])


def calibrate(model: mujoco.MjModel, imap: InstantHMRMap) -> Calibration:
    """Resolve *imap* on *model* in its T-pose calibration configuration."""
    m = model
    d = mujoco.MjData(m)
    q = m.qpos0.copy()
    for name, v in imap.calibration.items():
        q[m.jnt_qposadr[m.joint(name).id]] = v
    project_joint_couplings(JointCouplings.from_model(m), q)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)

    R_canon = _canonical_axes(m, d)
    body_names = {m.body(i).name for i in range(m.nbody)}
    names = [p.name for p in imap.points]
    pos = np.stack(
        [np.mean([d.xpos[m.body(b).id] for b in p.model], axis=0) for p in imap.points]
    )

    landmarks = []
    for k, p in enumerate(imap.points):
        if p.weight > 0:
            b = m.body(p.body).id
            off = d.xmat[b].reshape(3, 3).T @ (pos[k] - d.xpos[b])
            landmarks.append(
                Landmark(p.name, p.body, None, tuple(map(float, off)), p.weight)
            )

    def mvec(
        pair: tuple[tuple[str, ...], tuple[str, ...]], canon: tuple[float, float, float]
    ) -> np.ndarray:
        if pair and all(b in body_names for grp in pair for b in grp):
            return np.mean([d.xpos[m.body(b).id] for b in pair[1]], 0) - np.mean([d.xpos[m.body(b).id] for b in pair[0]], 0)  # fmt: skip
        return R_canon @ np.asarray(canon, np.float64)

    rot_offsets, orientations = [], []
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
        rot_offsets.append(_triad(u, v).T @ d.xmat[m.body(o.body).id].reshape(3, 3))
        axis = None
        if o.axis_to is not None:
            b = m.body(o.body).id
            a = d.xmat[b].reshape(3, 3).T @ (d.xpos[m.body(o.axis_to).id] - d.xpos[b])
            axis = tuple(map(float, a / np.linalg.norm(a)))
        orientations.append(
            OrientationTarget(o.name, o.body, o.weight, axis, o.axis_weight)
        )

    lmap = LandmarkMap(
        name=imap.name, version=imap.version, landmarks=tuple(landmarks),
        orientations=tuple(orientations), feet=imap.feet, twist_pairs=imap.twist_pairs,
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


def segment_orientations(joints_world: np.ndarray, imap: InstantHMRMap) -> np.ndarray:
    """Construct ``(T, O, 3, 3)`` proper segment rotation matrices from world MHR70 keypoints."""

    def jvec(pair: tuple[tuple[str, ...], tuple[str, ...]]) -> np.ndarray:
        return joints_world[:, [_JIDX[j] for j in pair[1]]].mean(1) - joints_world[:, [_JIDX[j] for j in pair[0]]].mean(1)  # fmt: skip

    frames = []
    for o in imap.orientations:
        u = jvec(o.primary)
        if o.kind == "femur":
            s = o.side
            fore = 0.5 * (
                joints_world[:, _JIDX[f"{s}_big_toe_tip"]]
                + joints_world[:, _JIDX[f"{s}_small_toe_tip"]]
            )
            v = _femur_secondary(
                _unit(u),
                _unit(
                    joints_world[:, _JIDX[f"{s}_ankle"]]
                    - joints_world[:, _JIDX[f"{s}_knee"]]
                ),
                _unit(fore - joints_world[:, _JIDX[f"{s}_heel"]]),
            )
        else:
            v = jvec(o.secondary)
        frames.append(_triad(u, v))
    return np.stack(frames, axis=1)


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


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    out = np.empty((*R.shape[:-2], 4))
    flat_R, flat_q = R.reshape(-1, 9), out.reshape(-1, 4)
    for i in range(len(flat_R)):
        mujoco.mju_mat2Quat(flat_q[i], flat_R[i])
    return out


def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x
    r = max(1, int(round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, pad)


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
        self, path: str | Path, model: mujoco.MjModel, lmap: LandmarkMap
    ) -> ClipTargets:
        """Construct scaled landmark, orientation, contact and hinge targets for *path*."""
        c, imap = self.calibration, self.imap
        clip = load_instanthmr(path)
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

        s_root = c.leg_length / max(
            0.5 * sum(subj_len[names.index(b)] for _, b in imap.leg_edges), 1e-6
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
            model_len = float(np.linalg.norm(c.model_pos[k] - c.model_pos[par]))
            ratio[k] = model_len / subj_len[k]
            out[k] = out[par] + model_len * _unit(x[k] - x[par])

        pos = out[[names.index(lm.source) for lm in lmap.landmarks]].transpose(1, 0, 2)
        quat = _mat_to_quat(
            np.einsum("toab,obc->toac", segment_orientations(jw, imap), c.rot_offset)
        )

        if self.target_sigma > 0:
            pos = _smooth(pos, self.target_sigma * fps)
            sign = np.cumprod(
                np.where((quat[1:] * quat[:-1]).sum(-1) < 0, -1.0, 1.0), axis=0
            )
            quat[1:] *= sign[..., None]
            quat = _smooth(quat, self.target_sigma * fps)
            quat /= np.linalg.norm(quat, axis=-1, keepdims=True)

        posture = None
        if imap.hinges:
            posture = np.tile([v for _, v, _ in lmap.posture], (T, 1))
            for k, (h, tab) in enumerate(zip(imap.hinges, c.hinge_table)):
                a, b, e = (jw[:, _JIDX[j]] for j in h.mhr)
                theta = _smooth(_segment_angle(a, b, e), self.target_sigma * fps)
                posture[:, len(imap.posture) + k] = np.interp(theta, tab[1], tab[0])

        near = ((sole[..., 2] - sole_floor) < self.contact_height) & (
            np.linalg.norm(np.gradient(sole[..., :2], 1.0 / fps, axis=0), axis=-1)
            < self.contact_speed
        )
        n_l = len(imap.contact_keypoints[0])
        contact = np.stack([near[:, :n_l].any(1), near[:, n_l:].any(1)], axis=1)

        seed = self._start_pose(model, pos[0], quat[0], lmap)
        self._last = {
            "floor": floor, "s_root": float(s_root), "contact": contact,
            "frames_source": len(clip["joints_3d_cam"]), "interpolated_frames": int((~valid).sum()),
            "edge_ratio": {names[k]: float(r) for k, r in ratio.items()}, "knee_flexion": knee_flexion(jw),
        }  # fmt: skip
        return ClipTargets(fps, pos, quat, seed, contact, posture)

    def _start_pose(
        self, m: mujoco.MjModel, pos: np.ndarray, quat: np.ndarray, lmap: LandmarkMap
    ) -> np.ndarray:
        c = self.calibration
        q = c.qpos.copy()
        k = [o.body for o in lmap.orientations].index("pelvis")
        d = mujoco.MjData(m)
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        Rp = np.empty(9)
        mujoco.mju_quat2Mat(Rp, quat[k])
        R_delta = Rp.reshape(3, 3) @ d.xmat[m.body("pelvis").id].reshape(3, 3).T
        root_jnt = next(
            j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        )
        root_body = int(m.jnt_bodyid[root_jnt])
        qr = np.empty(4)
        mujoco.mju_mat2Quat(qr, (R_delta @ d.xmat[root_body].reshape(3, 3)).ravel())
        adr = int(m.jnt_qposadr[root_jnt])
        q[adr + 3 : adr + 7] = qr
        q[adr : adr + 3] = pos[0] - R_delta @ (c.model_pos[0] - d.xpos[root_body])
        return q

    def finish(
        self, ik: LandmarkIK, targets: ClipTargets, qpos: np.ndarray
    ) -> tuple[np.ndarray, ClipTargets, dict[str, Any]]:
        """Ground stance frames to ``z = 0`` while interpolating aerial flight phases."""
        info = self._last
        contact = info["contact"]
        low, _ = _foot_kinematics(ik.model, ik.lmap, qpos)
        any_c = contact.any(1)
        dz = np.full(len(qpos), np.nan)
        dz[any_c] = -np.where(contact, low, np.inf)[any_c].min(1)
        if any_c.any():
            t = np.arange(len(qpos))
            dz = np.interp(t, t[any_c], dz[any_c])
        else:
            dz[:] = -low.min()
        lift = -low.min(1)
        dz = np.maximum(
            _smooth(np.maximum(dz, lift), self.ground_sigma * targets.frequency), lift
        )

        root_jnt = next(
            j
            for j in range(ik.model.njnt)
            if ik.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        )
        qpos = qpos.copy()
        qpos[:, int(ik.model.jnt_qposadr[root_jnt]) + 2] += dz
        pos = targets.pos.copy()
        pos[..., 2] += dz[:, None]
        targets = ClipTargets(targets.frequency, pos, targets.quat, targets.seed, targets.contact, targets.posture)  # fmt: skip
        meta = {
            "instanthmr_map": self.imap.name, "instanthmr_map_version": self.imap.version,
            "trim": list(self.trim), "frames_source": info["frames_source"],
            "interpolated_frames": info["interpolated_frames"], "subject_floor_m": info["floor"],
            "root_scale": info["s_root"], "edge_ratio": info["edge_ratio"],
            "ground_shift_m": {"mean": float(dz.mean()), "min": float(dz.min()), "max": float(dz.max())},
            "contact_fraction": [float(x) for x in contact.mean(0)],
            "source_knee_flexion_deg": {
                side: {"min": float(np.degrees(f.min())),
                       "frames_below_minus2": float((f < np.radians(-2)).mean()),
                       "frames_below_minus5": float((f < np.radians(-5)).mean())}
                for side, f in info["knee_flexion"].items()
            },
        }  # fmt: skip
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


__all__ = [
    "CAM_TO_WORLD", "INSTANTHMR_MAPS", "INSTANTHMR_MYOFULLBODY", "INSTANTHMR_MYOLEGS",
    "MHR70_JOINTS", "Calibration", "InstantHMRHinge", "InstantHMRMap",
    "InstantHMROrientation", "InstantHMRPoint", "InstantHMRSource", "calibrate",
    "instanthmr_source", "knee_flexion", "landmark_bodies", "load_instanthmr",
    "map_dict", "segment_orientations",
]  # fmt: skip
