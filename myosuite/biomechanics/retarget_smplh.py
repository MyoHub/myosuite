# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""AMASS SMPL+H motions straight onto a body model: a :class:`~.retarget.ClipSource`.

Route B of the KIT clips: the AMASS source (SMPL+H, subject shape) is retargeted onto
myofullbody once, instead of SMPL+H -> MuscleMimic body (GMR) -> myofullbody.

* SMPL+H forward kinematics (:mod:`.smplh`) with the subject's gender-specific model and
  all 16 betas gives joint positions, joint orientations and a few surface vertices.
* One explicit table (:class:`SMPLHMap`) pairs SMPL+H joints/vertices with model
  points and bodies. Model points and orientation offsets come from the model itself in
  a calibration pose that matches SMPL's rest pose (T-pose, palms down), so a new model
  version (myofullbody v2) only needs this table checked, not new numbers.
* Proportions (per segment, relative to the root): limb edges are the model's own
  segments turned by the SMPL+H joint that carries them, so they agree with the
  orientation targets; edges across the spine and shoulder girdle keep the subject's
  direction at the model's length, as do the thighs, shanks, feet and forearms, whose
  joints have a different zero in the two bodies (SMPL+H's rest knee is 4-6 deg flexed
  between the joint centres, unequal left and right; the model's 1.4 deg). The root
  follows the subject's mid-hip, scaled by the hip-height ratio and started at the
  origin.
  Orientation targets on the limbs carry the axial twist that joint centres leave open;
  where the model couples a twist to a hinge (the knee), the distal segment's twist is
  weighted down, so the twist the model cannot follow stays at that joint.
* Hinges (the elbows) follow SMPL+H's segment angle in the model's own convention
  (:class:`SMPLHHinge`): the two bodies' angles at zero flexion differ.
* Ground: after the fit the root height is shifted per frame, smoothly, so the foot the
  subject stands on touches the ground (contacts from the subject's heel and toe
  vertices); flight phases are interpolated.

The approach follows GMR (Araujo et al. 2025, "Retargeting Matters") as MuscleMimic
used it for ``smplh -> myofullbody``: SMPL joint frames as IK targets, a T-pose
calibration of rotation offsets and proportion scaling relative to the root, then a
ground offset. Here the scaling is per segment on the subject's own shape (GMR fits one
SMPL shape to the robot instead), and the solve is :class:`~.retarget.LandmarkIK` on
the model's exact couplings.
"""

from __future__ import annotations

import functools
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
from myosuite.biomechanics.smplh import (
    SMPLH,
    SMPLH_JOINTS,
    SMPLH_VERTICES,
    load_amass,
)
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

SMPL_TO_MODEL = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
"""SMPL's canonical frame (x left, y up, z forward) in a model world frame facing +x
with z up (columns: SMPL x, y, z)."""


@dataclass(frozen=True)
class SMPLHPoint:
    """A node of the target tree: an SMPL+H point and its model counterpart.

    Attributes:
        name: Landmark name.
        smplh: SMPL+H joints whose mean is the point, or one vertex name
            (:data:`~.smplh.SMPLH_VERTICES`).
        body: Model body carrying the point.
        model: Model bodies whose origins (mean, in the calibration pose) are the point;
            empty for a vertex (placed from the neutral SMPL+H template, scaled).
        parent: Parent node (``None``: the root).
        scale_like: For a vertex: the node whose edge length ratio scales it.
        carrier: SMPL+H joint whose orientation carries the edge from the parent: the
            edge is the model's (calibration pose) turned by that joint, so it agrees
            with the orientation targets. ``None``: the subject's edge direction
            scaled to the model's length (edges across the spine and shoulder girdle).
        subject_direction: Use the subject's edge direction (at the model's length)
            even with a carrier; orientation targets that ``align`` with this point
            are turned by the rotation from the carried to the subject's direction.
        weight: Position weight; 0 keeps the node in the tree only.
    """

    name: str
    smplh: tuple[str, ...]
    body: str
    model: tuple[str, ...] = ()
    parent: str | None = None
    scale_like: str | None = None
    carrier: str | None = None
    subject_direction: bool = False
    weight: float = 1.0


@dataclass(frozen=True)
class SMPLHOrientation:
    """A model body following an SMPL+H joint's orientation (offset from calibration).

    Attributes:
        joint: SMPL+H joint.
        body: Model body.
        weight: Metres of position error one radian is worth.
        axis_to: Model body whose origin (calibration pose) gives ``body``'s long axis;
            the rotation about it is weighted by ``axis_weight``. ``None``: isotropic.
        axis_weight: Weight factor of the axial rotation.
        align: A point with ``subject_direction``: the target turns with its edge.
    """

    joint: str
    body: str
    weight: float = 0.05
    axis_to: str | None = None
    axis_weight: float = 1.0
    align: str | None = None


@dataclass(frozen=True)
class SMPLHHinge:
    """A model hinge set per frame from the angle between two SMPL+H segments.

    The angle between the proximal and distal segment directions (joint centres) is
    matched in the model's own convention: the model's hinge value whose segment angle
    (model joint centres, calibration pose) equals SMPL+H's. The two bodies' angles at
    zero differ (myofullbody's elbow: 12.8 deg carrying angle, SMPL+H: about 7 deg), so
    the hinge value is not SMPL+H's angle minus a constant.

    Attributes:
        joint: Model hinge (an independent joint).
        smplh: ``(proximal, hinge, distal)`` SMPL+H joints.
        model: ``(proximal, hinge, distal)`` model bodies; the distal point is fixed on
            the hinge body (where the distal body's origin is in the calibration pose).
        weight: Metres of position error one radian is worth.
    """

    joint: str
    smplh: tuple[str, str, str]
    model: tuple[str, str, str]
    weight: float = 0.1


@dataclass(frozen=True)
class SMPLHMap:
    """Everything model-specific of an SMPL+H retarget, in one table.

    Attributes:
        name: Map name.
        version: Bumped whenever an entry or weight changes.
        points: Target tree (root first, parents before children).
        orientations: Orientation targets.
        calibration: Model joint values (rad) of SMPL's rest pose; other joints at 0.
        smpl_to_model: SMPL canonical axes in the calibration pose's world frame.
        feet: ``(left, right)`` model bodies whose collision geoms are the soles.
        contact_vertices: ``(left, right)`` SMPL+H vertices of the foot contacts.
        twist_pairs, twist_reference, coupling_term, period_joint: The symmetry check,
            as in :class:`~.retarget.LandmarkMap`.
        posture: Priors of the solver (:class:`~.retarget.LandmarkMap`).
        hinges: Hinges set per frame from SMPL+H segment angles (the elbows).
    """

    name: str
    version: str
    points: tuple[SMPLHPoint, ...]
    orientations: tuple[SMPLHOrientation, ...]
    calibration: dict[str, float] = field(default_factory=dict)
    smpl_to_model: tuple[tuple[float, ...], ...] = tuple(map(tuple, SMPL_TO_MODEL))
    feet: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    contact_vertices: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    twist_pairs: tuple[tuple[str, str], ...] = ()
    twist_reference: str = ""
    coupling_term: str = ""
    period_joint: str = ""
    posture: tuple[tuple[str, float, float], ...] = ()
    hinges: tuple[SMPLHHinge, ...] = ()


def _limb_points() -> tuple[SMPLHPoint, ...]:
    pts = []
    for S, m in (("L_", "_l"), ("R_", "_r")):
        pts += [
            SMPLHPoint(f"{S}Hip", (f"{S}Hip",), f"femur{m}", (f"femur{m}",), "Hips",
                       carrier="Pelvis"),
            # the subject's thigh, shank and foot directions: knee and ankle angles
            # between the joint centres, not SMPL+H's joint rotations (whose zero is a
            # flexed, side-dependent rest knee)
            SMPLHPoint(f"{S}Knee", (f"{S}Knee",), f"tibia{m}", (f"tibia{m}",), f"{S}Hip",
                       carrier=f"{S}Hip", subject_direction=True),
            SMPLHPoint(f"{S}Ankle", (f"{S}Ankle",), f"talus{m}", (f"talus{m}",), f"{S}Knee",
                       carrier=f"{S}Knee", subject_direction=True),
            SMPLHPoint(f"{S}Foot", (f"{S}Foot",), f"toes{m}", (f"toes{m}",), f"{S}Ankle",
                       carrier=f"{S}Ankle", subject_direction=True),
            SMPLHPoint(f"{S}Heel", (f"{S}Heel",), f"calcn{m}", (), f"{S}Ankle",
                       scale_like=f"{S}Foot", carrier=f"{S}Ankle", weight=0.5),
            # SMPL's toe rotation jitters (KIT has no toe markers): a light pull
            SMPLHPoint(f"{S}BigToe", (f"{S}BigToe",), f"toes{m}", (), f"{S}Foot",
                       scale_like=f"{S}Foot", carrier=f"{S}Foot", weight=0.2),
            SMPLHPoint(f"{S}Shoulder", (f"{S}Shoulder",), f"humerus{m}", (f"humerus{m}",),
                       "Shoulders", weight=0.3),
            # the elbow relative to the thorax, not along SMPL's humerus: SMPL's
            # shoulder joint sits lateral of the glenohumeral centre, and its hanging
            # upper arm would drive the model's humerus through the thorax
            SMPLHPoint(f"{S}Elbow", (f"{S}Elbow",), f"ulna{m}", (f"ulna{m}",), "Shoulders"),
            # the subject's forearm direction: the model's forearm turned by SMPL's
            # elbow would carry the model's 12.8 deg carrying angle on top of SMPL's 7
            # (the hand targets turn with it)
            SMPLHPoint(f"{S}Wrist", (f"{S}Wrist",), f"lunate{m}", (f"lunate{m}",), f"{S}Elbow",
                       carrier=f"{S}Elbow", subject_direction=True),
        ]  # fmt: skip
    return tuple(pts)


SMPLH_MYOFULLBODY = SMPLHMap(
    name="amass-smplh->myofullbody",
    version="3",
    points=(
        SMPLHPoint("Hips", ("L_Hip", "R_Hip"), "pelvis", ("femur_l", "femur_r")),
        SMPLHPoint(
            "Shoulders",
            ("L_Shoulder", "R_Shoulder"),
            "torso",
            ("humerus_l", "humerus_r"),
            "Hips",
            weight=0.0,
        ),
        # the model has no neck and a shoulder girdle driven by the elevation: head and
        # shoulder centres are followed loosely
        SMPLHPoint("Head", ("Head",), "head", ("head",), "Shoulders", weight=0.3),
        *_limb_points(),
    ),  # fmt: skip
    orientations=(
        SMPLHOrientation("Pelvis", "pelvis", 0.5),
        SMPLHOrientation("Spine3", "torso", 0.1),
        SMPLHOrientation("Head", "head", 0.02),
        # the knee couples its axial rotation to flexion: SMPL's free knee twist is
        # left at the knee (shank twist weak); the shank target turns with its edge
        SMPLHOrientation("L_Hip", "femur_l", 0.1),
        SMPLHOrientation("R_Hip", "femur_r", 0.1),
        SMPLHOrientation("L_Knee", "tibia_l", 0.1, "talus_l", 0.1, align="L_Ankle"),
        SMPLHOrientation("R_Knee", "tibia_r", 0.1, "talus_r", 0.1, align="R_Ankle"),
        SMPLHOrientation("L_Ankle", "calcn_l", 0.1, align="L_Foot"),
        SMPLHOrientation("R_Ankle", "calcn_r", 0.1, align="R_Foot"),
        SMPLHOrientation("L_Shoulder", "humerus_l", 0.05),
        SMPLHOrientation("R_Shoulder", "humerus_r", 0.05),
        # KIT has wrist but no hand markers: the hand orientation is weakly known, and a
        # stronger pull flips the forearm between pronation and supination
        SMPLHOrientation("L_Wrist", "lunate_l", 0.02, align="L_Wrist"),
        SMPLHOrientation("R_Wrist", "lunate_r", 0.02, align="R_Wrist"),
    ),
    # SMPL's rest pose: arms abducted to horizontal, palms down (GMR's T-pose)
    calibration={"shoulder_elv_l": np.pi / 2, "shoulder_elv_r": np.pi / 2},
    feet=(("calcn_l", "toes_l"), ("calcn_r", "toes_r")),
    contact_vertices=(
        ("L_Heel", "L_BigToe", "L_SmallToe"),
        ("R_Heel", "R_BigToe", "R_SmallToe"),
    ),  # fmt: skip
    twist_pairs=(
        ("femur_l", "femur_r"),
        ("tibia_l", "tibia_r"),
        ("calcn_l", "calcn_r"),
    ),
    twist_reference="pelvis",
    coupling_term="knee_angle_rotation3_l",
    period_joint="knee_angle_r",
    # a hanging arm leaves the plane of elevation open (it and the humeral rotation
    # turn about the same axis): a weak prior keeps it from wandering. The toes (+-30
    # deg) would otherwise snap onto their limits when the foot heading is off.
    posture=(
        ("elv_angle_l", 0.0, 0.01),
        ("elv_angle_r", 0.0, 0.01),
        ("mtp_angle_l", 0.0, 0.02),
        ("mtp_angle_r", 0.0, 0.02),
    ),
    hinges=tuple(
        SMPLHHinge(
            f"elbow_flexion{m}",
            (f"{S}Shoulder", f"{S}Elbow", f"{S}Wrist"),
            (f"humerus{m}", f"ulna{m}", f"lunate{m}"),
        )
        for S, m in (("L_", "_l"), ("R_", "_r"))
    ),
)
"""AMASS SMPL+H (KIT and other subsets) on myo_sim's myofullbody."""

SMPLH_MAPS: dict[str, SMPLHMap] = {"myofullbody": SMPLH_MYOFULLBODY}
"""SMPL+H map per target model name."""


# ------------------------------------------------------------------------ calibration


@dataclass
class Calibration:
    """A map resolved on a model (subject independent).

    Attributes:
        lmap: The :class:`~.retarget.LandmarkMap` of the solver (weighted points and
            orientations, with model-local offsets; the hinges are its last posture
            priors).
        model_pos: ``(P, 3)`` every tree node in the calibration pose.
        rot_offset: ``(O, 3, 3)`` body orientation in its SMPL joint's frame.
        qpos: The calibration pose.
        hip_height: Model mid-hip height above the soles in the calibration pose.
        hinge_table: Per hinge, ``(2, K)`` hinge values and the model's segment angle
            (rad), increasing.
    """

    lmap: LandmarkMap
    model_pos: np.ndarray
    rot_offset: np.ndarray
    qpos: np.ndarray
    hip_height: float
    hinge_table: tuple[np.ndarray, ...] = ()


def _segment_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle (rad) between the segments a->b and b->c."""
    u, v = b - a, c - b
    cos = (u * v).sum(-1) / np.linalg.norm(u, axis=-1) / np.linalg.norm(v, axis=-1)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _hinge_table(
    m: mujoco.MjModel, d: mujoco.MjData, q: np.ndarray, h: SMPLHHinge
) -> np.ndarray:
    """The model's segment angle over the hinge's range (calibration pose otherwise)."""
    j = m.joint(h.joint).id
    prox, mid, dist = (m.body(b).id for b in h.model)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    R = d.xmat[mid].reshape(3, 3)
    tip = R.T @ (d.xpos[dist] - d.xpos[mid])  # distal point, fixed on the hinge body
    phi = np.linspace(*m.jnt_range[j], 181)
    theta = np.empty_like(phi)
    x = q.copy()
    for k, v in enumerate(phi):
        x[m.jnt_qposadr[j]] = v
        d.qpos[:] = x
        mujoco.mj_kinematics(m, d)
        end = d.xpos[mid] + d.xmat[mid].reshape(3, 3) @ tip
        theta[k] = _segment_angle(d.xpos[prox], d.xpos[mid], end)
    if np.any(np.diff(theta) <= 0):
        raise ValueError(f"{h.joint}: segment angle not increasing over the range")
    return np.stack([phi, theta])


def _lowest_sole(m: mujoco.MjModel, smap: SMPLHMap, qpos: np.ndarray) -> float:
    proxy = LandmarkMap("", "", (), feet=smap.feet)
    return float(_foot_kinematics(m, proxy, qpos[None])[0].min())


def calibrate(model: mujoco.MjModel, smap: SMPLHMap, template: SMPLH) -> Calibration:
    """Resolve *smap* on *model*: model points, local offsets and rotation offsets.

    Args:
        model: Target model.
        smap: The table.
        template: An SMPL+H model; its zero-shape template places the vertex points.

    Returns:
        The calibration.
    """
    m = model
    d = mujoco.MjData(m)
    q = m.qpos0.copy()
    for name, v in smap.calibration.items():
        q[m.jnt_qposadr[m.joint(name).id]] = v
    project_joint_couplings(JointCouplings.from_model(m), q)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    Rc = np.asarray(smap.smpl_to_model)
    v0, J0 = template.shaped(np.zeros(template.num_betas))
    jidx = {n: i for i, n in enumerate(SMPLH_JOINTS)}
    names = [p.name for p in smap.points]
    pos = np.empty((len(names), 3))
    ref = np.empty((len(names), 3))  # template position of each node (SMPL frame)
    for k, p in enumerate(smap.points):
        if p.model:
            pos[k] = np.mean([d.xpos[m.body(b).id] for b in p.model], 0)
            ref[k] = J0[[jidx[j] for j in p.smplh]].mean(0)
    for k, p in enumerate(smap.points):
        if not p.model:  # vertex: template offset from its parent, scaled like an edge
            par, like = names.index(p.parent), names.index(p.scale_like)
            lpar = names.index(smap.points[like].parent)
            s = np.linalg.norm(pos[like] - pos[lpar]) / np.linalg.norm(
                ref[like] - ref[lpar]
            )
            ref[k] = v0[SMPLH_VERTICES[p.smplh[0]]]
            pos[k] = pos[par] + s * Rc @ (ref[k] - ref[par])
    landmarks = []
    for k, p in enumerate(smap.points):
        if p.weight > 0:
            b = m.body(p.body).id
            off = d.xmat[b].reshape(3, 3).T @ (pos[k] - d.xpos[b])
            landmarks.append(
                Landmark(p.name, p.body, None, tuple(map(float, off)), p.weight)
            )
    rot = np.stack(
        [Rc.T @ d.xmat[m.body(o.body).id].reshape(3, 3) for o in smap.orientations]
    )
    orientations = []
    for o in smap.orientations:
        axis = None
        if o.axis_to is not None:  # long axis in the body frame
            b = m.body(o.body).id
            a = d.xmat[b].reshape(3, 3).T @ (d.xpos[m.body(o.axis_to).id] - d.xpos[b])
            axis = tuple(map(float, a / np.linalg.norm(a)))
        orientations.append(
            OrientationTarget(o.joint, o.body, o.weight, axis, o.axis_weight)
        )
    lmap = LandmarkMap(
        name=smap.name, version=smap.version, landmarks=tuple(landmarks),
        orientations=tuple(orientations),
        feet=smap.feet, twist_pairs=smap.twist_pairs,
        twist_reference=smap.twist_reference, coupling_term=smap.coupling_term,
        period_joint=smap.period_joint,
        posture=smap.posture + tuple((h.joint, 0.0, h.weight) for h in smap.hinges),
    )  # fmt: skip
    hip = pos[0, 2] - _lowest_sole(m, smap, q)
    tables = tuple(_hinge_table(m, d, q, h) for h in smap.hinges)
    return Calibration(lmap, pos, rot, q, float(hip), tables)


# ---------------------------------------------------------------------------- targets


@functools.lru_cache(maxsize=4)
def _body_model(path: str) -> SMPLH:
    return SMPLH(path)


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    out = np.empty((*R.shape[:-2], 4))
    flat_R, flat_q = R.reshape(-1, 9), out.reshape(-1, 4)
    for i in range(len(flat_R)):
        mujoco.mju_mat2Quat(flat_q[i], flat_R[i])
    return out


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(T, 3, 3)`` shortest rotations taking the directions *a* onto *b*."""
    orig_shape = np.shape(a)[:-1]
    a_flat = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b_flat = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    na = np.linalg.norm(a_flat, axis=-1, keepdims=True)
    nb = np.linalg.norm(b_flat, axis=-1, keepdims=True)
    valid = (na[:, 0] > 1e-12) & (nb[:, 0] > 1e-12)
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    a_u = np.where(na > 1e-12, a_flat / np.where(na > 1e-12, na, 1.0), ex)
    b_u = np.where(nb > 1e-12, b_flat / np.where(nb > 1e-12, nb, 1.0), ex)
    v = np.cross(a_u, b_u)
    cos = (a_u * b_u).sum(-1)
    zero = np.zeros_like(cos)
    K = np.stack(
        [zero, -v[:, 2], v[:, 1], v[:, 2], zero, -v[:, 0], -v[:, 1], v[:, 0], zero], -1
    ).reshape(-1, 3, 3)
    regular = (1.0 + cos) > 1e-8
    denom = np.where(regular, 1.0 + cos, 1.0)
    R_rod = np.eye(3) + K + (K @ K) / denom[:, None, None]
    ref_axis = np.where(np.abs(a_u[:, :1]) < 0.9, ex, ey)
    ortho = np.cross(a_u, ref_axis)
    ortho_norm = np.linalg.norm(ortho, axis=-1, keepdims=True)
    u = ortho / np.where(ortho_norm > 1e-12, ortho_norm, 1.0)
    R_180 = 2.0 * (u[:, :, None] * u[:, None, :]) - np.eye(3)
    R = np.where(regular[:, None, None], R_rod, R_180)
    R = np.where(valid[:, None, None], R, np.eye(3))
    return R.reshape(*orig_shape, 3, 3)


def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along axis 0 (edge-padded)."""
    if sigma <= 0:
        return x
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, pad)


def knee_flexion(rotations: np.ndarray) -> dict[str, np.ndarray]:
    """``(T,)`` SMPL+H knee flexion (rad, rotation about SMPL's x axis) per side.

    Args:
        rotations: ``(T, J, 3, 3)`` world rotations of the SMPL+H joints.

    Returns:
        ``{"l": ..., "r": ...}``; negative values are hyperextension.
    """
    jidx = {n: i for i, n in enumerate(SMPLH_JOINTS)}
    out = {}
    for side, S in (("l", "L_"), ("r", "R_")):
        R = (
            np.swapaxes(rotations[:, jidx[S + "Hip"]], -1, -2)
            @ rotations[:, jidx[S + "Knee"]]
        )
        out[side] = np.arctan2(R[:, 2, 1], R[:, 1, 1])
    return out


@dataclass
class SMPLHSource:
    """AMASS SMPL+H ``*_poses.npz`` files as a :class:`~.retarget.ClipSource`.

    Args:
        smap: The table.
        calibration: Its calibration on the model (:func:`calibrate`).
        body_models: Folder with ``{male,female,neutral}/model.npz``.
        trim: Source frames dropped at the start and end (3, 3 matches the GMR KIT
            clips frame for frame).
        contact_height: A foot is in contact while a contact vertex is this close to
            the subject's floor (m) ...
        contact_speed: ... and moves slower than this (m/s).
        ground_sigma: Smoothing (s) of the per-frame root-height correction.
        target_sigma: Gaussian smoothing (s) of the targets: the raw 100 Hz mocap
            jitters by a few degrees per frame at the wrists and feet.
    """

    smap: SMPLHMap
    calibration: Calibration
    body_models: str | Path
    trim: tuple[int, int] = (0, 0)
    contact_height: float = 0.03
    contact_speed: float = 0.5
    ground_sigma: float = 0.05
    target_sigma: float = 0.01
    name: str = "amass-smplh"
    # the last clip's subject data, for finish() (targets() and finish() run in pairs)
    _last: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def _subject(self, path: str | Path) -> dict[str, Any]:
        a = load_amass(path)
        body = _body_model(str(Path(self.body_models) / a["gender"] / "model.npz"))
        n = len(a["poses"])
        sl = slice(self.trim[0], n - self.trim[1])
        verts = {v: SMPLH_VERTICES[v] for v in SMPLH_VERTICES}
        pose = body.forward(a["poses"][sl], a["betas"], a["trans"][sl], verts)
        rest_v, rest_j = body.shaped(a["betas"])
        return {
            "amass": a,
            "body": body,
            "pose": pose,
            "rest_v": rest_v,
            "rest_j": rest_j,
        }

    def targets(
        self, path: str | Path, model: mujoco.MjModel, lmap: LandmarkMap
    ) -> ClipTargets:
        """Targets of an AMASS clip on *model* (see the module docstring)."""
        c, smap = self.calibration, self.smap
        s = self._subject(path)
        pose, rest_v, rest_j = s["pose"], s["rest_v"], s["rest_j"]
        jidx = {n: i for i, n in enumerate(SMPLH_JOINTS)}
        vnames = list(pose.vertex_names)
        names = [p.name for p in smap.points]

        def subj(p: SMPLHPoint, rest: bool) -> np.ndarray:
            if p.model:
                ids = [jidx[j] for j in p.smplh]
                return rest_j[ids].mean(0) if rest else pose.joints[:, ids].mean(1)
            v = p.smplh[0]
            return (
                rest_v[SMPLH_VERTICES[v]] if rest else pose.vertices[:, vnames.index(v)]
            )

        x = [subj(p, False) for p in smap.points]
        xr = [subj(p, True) for p in smap.points]
        # each contact vertex's own floor level (skin and mocap offsets differ per
        # vertex), the subject's floor their median
        sole = pose.vertices[:, [vnames.index(v) for side in smap.contact_vertices
                                 for v in side]]  # fmt: skip
        sole_floor = np.percentile(sole[..., 2], 2, axis=0)
        floor = float(np.median(sole_floor))
        hip_rest = xr[0][1] - rest_v[:, 1].min()  # SMPL rest is y-up
        s_root = c.hip_height / hip_rest
        T = len(x[0])
        out = np.empty((len(names), T, 3))
        root = x[0].copy()
        # horizontal: scaled path, mid-hip starting at the origin (as the GMR clips)
        root[:, :2] = s_root * (root[:, :2] - root[0, :2])
        root[:, 2] = s_root * (root[:, 2] - floor)
        out[0] = root
        Rc = np.asarray(smap.smpl_to_model)
        ratio, turn = {}, {}
        for k, p in enumerate(smap.points[1:], 1):
            par = names.index(p.parent)
            edge = c.model_pos[k] - c.model_pos[par]
            ratio[k] = np.linalg.norm(edge) / np.linalg.norm(xr[k] - xr[par])
            subject = ratio[k] * (
                x[k] - x[par]
            )  # the subject's direction, model length
            if p.carrier:  # the model's edge, turned like the carrier joint
                carried = pose.rotations[:, jidx[p.carrier]] @ (Rc.T @ edge)
                if p.subject_direction:
                    turn[p.name] = _rotation_between(carried, subject)
                    out[k] = out[par] + subject
                else:
                    out[k] = out[par] + carried
            else:
                out[k] = out[par] + subject
        keep = [names.index(lm.source) for lm in lmap.landmarks]
        pos = out[keep].transpose(1, 0, 2)
        R = pose.rotations[:, [jidx[o.joint] for o in smap.orientations]].copy()
        for i, o in enumerate(smap.orientations):
            if o.align is not None:
                R[:, i] = turn[o.align] @ R[:, i]
        quat = _mat_to_quat(np.einsum("toab,obc->toac", R, c.rot_offset))
        fps = s["amass"]["fps"]
        if self.target_sigma > 0:
            pos = _smooth(pos, self.target_sigma * fps)
            # quaternions: same hemisphere as the previous frame, smoothed, normalised
            sign = np.cumprod(
                np.where((quat[1:] * quat[:-1]).sum(-1) < 0, -1.0, 1.0), 0
            )
            quat[1:] *= sign[..., None]
            quat = _smooth(quat, self.target_sigma * fps)
            quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
        # hinges: SMPL+H's segment angle as the model's hinge value (per-frame priors)
        posture = None
        if smap.hinges:
            posture = np.tile([v for _, v, _ in lmap.posture], (T, 1))
            for k, (h, tab) in enumerate(zip(smap.hinges, c.hinge_table)):
                a, b, e = (pose.joints[:, jidx[j]] for j in h.smplh)
                theta = _smooth(_segment_angle(a, b, e), self.target_sigma * fps)
                posture[:, len(smap.posture) + k] = np.interp(theta, tab[1], tab[0])
        # contacts: a contact vertex near the floor and slow
        h = sole[..., 2] - sole_floor
        v = np.linalg.norm(np.gradient(sole[..., :2], 1.0 / fps, axis=0), axis=-1)
        near = (h < self.contact_height) & (v < self.contact_speed)
        n_l = len(smap.contact_vertices[0])
        contact = np.stack([near[:, :n_l].any(1), near[:, n_l:].any(1)], 1)
        # start pose: calibration pose turned and moved onto the first frame's pelvis
        seed = self._start_pose(model, pos[0], quat[0], lmap)
        self._last = {"floor": floor, "s_root": float(s_root), "contact": contact,
                      "gender": s["amass"]["gender"], "frames_source": len(s["amass"]["poses"]),
                      "edge_ratio": {names[k]: float(r) for k, r in ratio.items()},
                      "knee_flexion": knee_flexion(pose.rotations)}  # fmt: skip
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
        R_root = Rp.reshape(3, 3) @ d.xmat[m.body("pelvis").id].reshape(3, 3).T
        qr = np.empty(4)
        mujoco.mju_mat2Quat(qr, R_root.ravel())
        q[3:7] = qr
        q[:3] = pos[0] - R_root @ (c.model_pos[0] - q[:3])
        return q

    def finish(
        self, ik: LandmarkIK, targets: ClipTargets, qpos: np.ndarray
    ) -> tuple[np.ndarray, ClipTargets, dict[str, Any]]:
        """Ground the fit: shift the root (and the targets) so standing feet touch z = 0.

        The lowest sole point of the feet in contact goes to the ground (interpolated
        over flight phases, smoothed); frames where a foot would still be below the
        ground are lifted out of it (as GMR does per frame).
        """
        info = self._last
        contact = info["contact"]
        low, _ = _foot_kinematics(ik.model, ik.lmap, qpos)
        any_c = contact.any(1)
        dz = np.full(len(qpos), np.nan)
        dz[any_c] = -np.where(contact, low, np.inf)[any_c].min(1)
        if any_c.any():
            t = np.arange(len(qpos))
            dz = np.interp(t, t[any_c], dz[any_c])
        else:  # never on the ground (should not happen): lowest point to 0
            dz[:] = -low.min()
        lift = -low.min(1)
        dz = _smooth(np.maximum(dz, lift), self.ground_sigma * targets.frequency)
        dz = np.maximum(dz, lift)
        qpos = qpos.copy()
        qpos[:, 2] += dz
        pos = targets.pos.copy()
        pos[..., 2] += dz[:, None]
        targets = ClipTargets(targets.frequency, pos, targets.quat, targets.seed,
                              targets.contact, targets.posture)  # fmt: skip
        meta = {
            "smplh_map": self.smap.name,
            "smplh_map_version": self.smap.version,
            "trim": list(self.trim),
            "gender": info["gender"],
            "frames_source": info["frames_source"],
            "subject_floor_m": info["floor"],
            "root_scale": info["s_root"],
            "edge_ratio": info["edge_ratio"],
            "ground_shift_m": {"mean": float(dz.mean()), "min": float(dz.min()),
                               "max": float(dz.max())},
            "contact_fraction": [float(x) for x in contact.mean(0)],
            # the source's knee hyperextension (e.g. a fit beyond the model's knee range)
            "source_knee_flexion_deg": {
                side: {"min": float(np.degrees(f.min())),
                       "frames_below_minus2": float((f < np.radians(-2)).mean()),
                       "frames_below_minus5": float((f < np.radians(-5)).mean())}
                for side, f in info["knee_flexion"].items()
            },
        }  # fmt: skip
        return qpos, targets, meta


def smplh_source(
    model: mujoco.MjModel,
    body_models: str | Path,
    smap: SMPLHMap = SMPLH_MYOFULLBODY,
    **kwargs: Any,
) -> tuple[SMPLHSource, LandmarkMap]:
    """A ready :class:`SMPLHSource` and the solver's landmark map for *model*.

    Args:
        model: Target model.
        body_models: Folder with ``{male,female,neutral}/model.npz``.
        smap: The table.
        **kwargs: Further :class:`SMPLHSource` options (e.g. ``trim``).
    """
    template = _body_model(str(Path(body_models) / "neutral" / "model.npz"))
    cal = calibrate(model, smap, template)
    return SMPLHSource(smap, cal, body_models, **kwargs), cal.lmap


def landmark_bodies(smap: SMPLHMap) -> tuple[str, ...]:
    """Model bodies the map places targets on: the points' carriers and calibration
    bodies, the orientation bodies and their long-axis ends, and the hinge segments."""
    names = {b for p in smap.points for b in (p.body, *p.model)}
    names |= {b for o in smap.orientations for b in (o.body, o.axis_to) if b}
    names |= {b for h in smap.hinges for b in h.model}
    return tuple(sorted(names))


def map_dict(smap: SMPLHMap) -> dict[str, Any]:
    """JSON-serialisable form of a table."""
    return asdict(smap)


__all__ = [
    "SMPLH_MAPS",
    "SMPLH_MYOFULLBODY",
    "Calibration",
    "SMPLHHinge",
    "SMPLHMap",
    "SMPLHOrientation",
    "SMPLHPoint",
    "SMPLHSource",
    "calibrate",
    "knee_flexion",
    "landmark_bodies",
    "map_dict",
    "smplh_source",
]
