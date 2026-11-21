# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Biovision Hierarchy (``.bvh`` / LAFAN1) motions straight onto a body model.

Pure NumPy/MuJoCo ``.bvh`` parsing, vectorized forward kinematics, per-segment
proportion scaling from each clip's rest hierarchy, elbow hinge-angle tables,
and foot-contact grounding onto :class:`~.retarget.LandmarkIK`'s coupled solver.

Conversion contracts:
- **Units:** ``scale`` converts raw BVH ``OFFSET`` and ``*position`` channel values into
  metres [m] (default ``0.01`` for centimetre datasets such as Ubisoft LAFAN1); Euler
  ``*rotation`` channels are in degrees [deg], converted to radians internally.
- **Timing:** Uniform sampling rate ``fps = 1 / Frame Time`` [Hz] (strictly positive).
- **Coordinate frames:** ``parse_bvh`` outputs ``BVHClip`` positions and ``3x3`` rotation
  matrices in the file's BVH world frame; ``BVHSource`` maps them into the MuJoCo world
  frame (``+x`` forward, ``+y`` left, ``+z`` up) via proper ``3x3`` rotation
  ``BVHMap.bvh_to_model`` (default :data:`BVH_TO_MODEL` for ``+x`` left, ``+y`` up,
  ``+z`` forward BVH skeletons). Orientation targets in :class:`~.retarget.ClipTargets`
  use MuJoCo ``wxyz`` unit quaternions.
- **Channel order & missing data:** Arbitrary per-joint ``X/Y/Zposition`` and intrinsic
  ``X/Y/Zrotation`` channel sequences are evaluated in exact declared ``CHANNELS`` order.
  ``.bvh`` files are dense numeric tables; any ``NaN``/``Inf`` in ``OFFSET`` or ``MOTION``
  is rejected with ``ValueError``.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite.biomechanics.retarget import (
    ClipTargets,
    LandmarkIK,
    LandmarkMap,
    body_landmarks,
    calibration_pose,
    foot_contacts,
    foot_kinematics,
    ground_clip,
    hinge_table,
    knee_flexion_meta,
    mat_to_quat,
    orientation_targets,
    rotation_between,
    segment_angle,
    smooth_quat,
    smooth_signal,
    start_pose,
)

BVH_TO_MODEL = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
"""Standard BVH canonical frame (``+x`` left, ``+y`` up, ``+z`` forward) in a model
world frame facing ``+x`` with ``+y`` left and ``+z`` up."""

DEFAULT_BVH_ALIASES: dict[str, tuple[str, ...]] = {
    "LeftToe": ("LeftToe", "LeftToeBase", "LToeBase"),
    "RightToe": ("RightToe", "RightToeBase", "RToeBase"),
    "LeftToe_End": ("LeftToe_End", "LeftToeBase_End", "LToeBase_End"),
    "RightToe_End": ("RightToe_End", "RightToeBase_End", "RToeBase_End"),
    "Spine2": ("Spine2", "Spine3", "Chest", "UpperChest"),
}
"""Common joint-name aliases across Ubisoft LAFAN1, CMU and Mixamo ``.bvh`` skeletons."""

_VALID_CHANNELS: frozenset[str] = frozenset(
    {
        "Xposition",
        "Yposition",
        "Zposition",
        "Xrotation",
        "Yrotation",
        "Zrotation",
    }
)
_POS_AXIS: dict[str, int] = {"Xposition": 0, "Yposition": 1, "Zposition": 2}
_ROT_AXIS: dict[str, int] = {"Xrotation": 0, "Yrotation": 1, "Zrotation": 2}
_MOTION_RE = re.compile(r"(?m)^\s*MOTION\s*$")
_FRAMES_RE = re.compile(r"^\s*Frames\s*:\s*([+-]?\d+)\s*$")
_FRAME_TIME_RE = re.compile(r"^\s*Frame\s+Time\s*:\s*(\S+)\s*$")


# -------------------------------------------------------------------------- bvh parser


@dataclass(frozen=True)
class BVHJoint:
    """One node (root, joint or end site) of a ``.bvh`` skeletal hierarchy.

    Attributes:
        name: Joint name (``f"{parent}_End"`` for ``End Site`` blocks).
        parent: Parent index in :attr:`BVHClip.joints`, or ``-1`` for the root.
        offset: Static translation from the parent in metres (after ``scale``).
        channels: Ordered motion channel names on this joint.
        channel_offset: Starting column index in the raw motion table.
        is_end_site: ``True`` for terminal ``End Site`` nodes.
    """

    name: str
    parent: int
    offset: tuple[float, float, float]
    channels: tuple[str, ...]
    channel_offset: int
    is_end_site: bool = False


@dataclass(frozen=True)
class BVHClip:
    """Parsed ``.bvh`` hierarchy and forward-kinematics motion in BVH world space.

    Attributes:
        joints: Topological joint tuple (parents precede children).
        joint_names: Joint names matching :attr:`joints`.
        fps: Frame rate in Hz (``1 / frame_time``).
        positions: ``(T, J, 3)`` joint positions in metres (BVH world frame).
        rotations: ``(T, J, 3, 3)`` world rotation matrices (BVH world frame).
        rest_positions: ``(J, 3)`` zero-channel T-pose positions in metres.
    """

    joints: tuple[BVHJoint, ...]
    joint_names: tuple[str, ...]
    fps: float
    positions: np.ndarray
    rotations: np.ndarray
    rest_positions: np.ndarray


def _axis_rotation(axis: int, rad: np.ndarray) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    z, o = np.zeros_like(c), np.ones_like(c)
    if axis == 0:
        rows = (o, z, z, z, c, -s, z, s, c)
    elif axis == 1:
        rows = (c, z, s, z, o, z, -s, z, c)
    else:
        rows = (c, -s, z, s, c, z, z, z, o)
    return np.stack(rows, axis=-1).reshape(-1, 3, 3)


def _parse_hierarchy(
    text: str, scale: float, label: Path
) -> tuple[tuple[BVHJoint, ...], int]:
    tokens = text.replace("{", " { ").replace("}", " } ").split()
    if len(tokens) < 2 or tokens[0] != "HIERARCHY" or tokens[1] != "ROOT":
        raise ValueError(f"{label}: expected 'HIERARCHY' followed by 'ROOT'")

    joints: list[BVHJoint | None] = []
    seen_names: set[str] = set()
    idx = 1
    chan_cursor = 0

    def next_tok(desc: str) -> str:
        nonlocal idx
        if idx >= len(tokens):
            raise ValueError(
                f"{label}: unexpected end of HIERARCHY while reading {desc}"
            )
        tok = tokens[idx]
        idx += 1
        return tok

    def unique_end_name(parent_name: str) -> str:
        cand = f"{parent_name}_End"
        if cand not in seen_names:
            return cand
        k = 2
        while f"{cand}_{k}" in seen_names:
            k += 1
        return f"{cand}_{k}"

    def parse_node(parent: int, kind: str, parent_name: str) -> None:
        nonlocal chan_cursor
        is_end = kind == "End"
        if is_end:
            site_kw = next_tok("'Site'")
            if site_kw != "Site":
                raise ValueError(f"{label}: expected 'End Site', got 'End {site_kw}'")
            name = unique_end_name(parent_name)
        else:
            name = next_tok(f"{kind} name")
            if name in ("{", "}"):
                raise ValueError(f"{label}: missing name after {kind}")
            if name in seen_names:
                raise ValueError(f"{label}: duplicate joint name {name!r}")
        seen_names.add(name)

        if next_tok("'{'") != "{":
            raise ValueError(f"{label}: expected '{{' opening {name!r}")

        my_idx = len(joints)
        joints.append(None)
        offset: tuple[float, float, float] | None = None
        channels: tuple[str, ...] = ()
        has_channels = False
        my_chan_offset = chan_cursor

        while True:
            kw = next_tok(f"body of {name!r}")
            if kw == "}":
                break
            if kw == "OFFSET":
                try:
                    vals = tuple(
                        float(next_tok("OFFSET value")) * scale for _ in range(3)
                    )
                except ValueError as err:
                    raise ValueError(f"{label}: invalid OFFSET in {name!r}") from err
                if not np.isfinite(vals).all():
                    raise ValueError(f"{label}: non-finite OFFSET in {name!r}")
                offset = (vals[0], vals[1], vals[2])
            elif kw == "CHANNELS":
                if is_end:
                    raise ValueError(
                        f"{label}: End Site {name!r} cannot declare CHANNELS"
                    )
                try:
                    n_ch = int(next_tok("channel count"))
                except ValueError as err:
                    raise ValueError(
                        f"{label}: invalid channel count in {name!r}"
                    ) from err
                if n_ch < 0:
                    raise ValueError(f"{label}: negative channel count in {name!r}")
                ch_list = []
                for _ in range(n_ch):
                    ch = next_tok("channel name")
                    if ch not in _VALID_CHANNELS:
                        raise ValueError(f"{label}: invalid channel {ch!r} in {name!r}")
                    ch_list.append(ch)
                my_chan_offset = chan_cursor
                chan_cursor += n_ch
                channels = tuple(ch_list)
                has_channels = True
            elif kw in ("JOINT", "End"):
                if is_end:
                    raise ValueError(
                        f"{label}: End Site {name!r} cannot have child nodes"
                    )
                parse_node(my_idx, kw, name)
            else:
                raise ValueError(f"{label}: unexpected token {kw!r} in {name!r}")

        if offset is None:
            raise ValueError(f"{label}: missing OFFSET in {name!r}")
        if not is_end and not has_channels:
            raise ValueError(f"{label}: missing CHANNELS in {name!r}")
        joints[my_idx] = BVHJoint(
            name=name,
            parent=parent,
            offset=offset,
            channels=channels,
            channel_offset=my_chan_offset,
            is_end_site=is_end,
        )

    root_kw = next_tok("'ROOT'")
    assert root_kw == "ROOT"
    parse_node(-1, "ROOT", "")
    if idx != len(tokens):
        raise ValueError(
            f"{label}: unexpected token {tokens[idx]!r} after root HIERARCHY block"
        )
    if chan_cursor == 0:
        raise ValueError(f"{label}: HIERARCHY defines zero motion channels")
    return tuple(j for j in joints if j is not None), chan_cursor


def parse_bvh(path: str | Path, *, scale: float = 0.01) -> BVHClip:
    """Parse a Biovision Hierarchy (``.bvh``) file and run vectorized forward kinematics.

    Args:
        path: Path to the ``.bvh`` file.
        scale: Linear unit factor converting raw offsets and position channels to metres
            (default ``0.01`` converts centimetre datasets such as Ubisoft LAFAN1).

    Returns:
        A :class:`BVHClip` with world positions, rotation matrices and T-pose rest points.

    Raises:
        ValueError: On invalid ``scale``, malformed hierarchy/motion headers, or frame /
            channel count mismatches.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be finite and positive, got {scale}")
    bvh_path = Path(path)
    text = bvh_path.read_text(encoding="utf-8")
    m_motion = _MOTION_RE.search(text)
    if m_motion is None:
        raise ValueError(f"{bvh_path}: missing MOTION section")

    joints, total_channels = _parse_hierarchy(text[: m_motion.start()], scale, bvh_path)

    rem = text[m_motion.end() :]
    pos = 0
    header_lines: list[str] = []
    while len(header_lines) < 2:
        nl = rem.find("\n", pos)
        if nl < 0:
            line = rem[pos:]
            pos = len(rem)
        else:
            line = rem[pos:nl]
            pos = nl + 1
        if line.strip():
            header_lines.append(line)
        elif pos >= len(rem):
            break
    if len(header_lines) < 2:
        raise ValueError(
            f"{bvh_path}: incomplete MOTION header (need Frames and Frame Time)"
        )

    m_frames = _FRAMES_RE.match(header_lines[0])
    if m_frames is None:
        raise ValueError(f"{bvh_path}: invalid Frames header {header_lines[0]!r}")
    n_frames = int(m_frames.group(1))
    if n_frames < 1:
        raise ValueError(f"{bvh_path}: Frames must be >= 1, got {n_frames}")

    m_ft = _FRAME_TIME_RE.match(header_lines[1])
    if m_ft is None:
        raise ValueError(f"{bvh_path}: invalid Frame Time header {header_lines[1]!r}")
    try:
        frame_time = float(m_ft.group(1))
    except ValueError as err:
        raise ValueError(
            f"{bvh_path}: invalid Frame Time value {m_ft.group(1)!r}"
        ) from err
    if not np.isfinite(frame_time) or frame_time <= 0:
        raise ValueError(
            f"{bvh_path}: Frame Time must be positive and finite, got {frame_time}"
        )
    fps = 1.0 / frame_time

    motion_str = rem[pos:].strip()
    try:
        vals = (
            np.fromstring(motion_str, dtype=np.float64, sep=" ")
            if motion_str
            else np.empty(0, dtype=np.float64)
        )
    except ValueError as err:
        raise ValueError(f"{bvh_path}: malformed numeric MOTION data: {err}") from err
    expected = n_frames * total_channels
    if len(vals) != expected:
        raise ValueError(
            f"{bvh_path}: expected {expected} motion values "
            f"({n_frames} frames x {total_channels} channels), got {len(vals)}"
        )
    if not np.isfinite(vals).all():
        raise ValueError(f"{bvh_path}: non-finite values in MOTION data")
    motion = vals.reshape(n_frames, total_channels)

    n_joints = len(joints)
    rest_positions = np.empty((n_joints, 3), dtype=np.float64)
    positions = np.empty((n_frames, n_joints, 3), dtype=np.float64)
    rotations = np.empty((n_frames, n_joints, 3, 3), dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)

    for j, jnt in enumerate(joints):
        off = np.asarray(jnt.offset, dtype=np.float64)
        rest_positions[j] = off if jnt.parent < 0 else rest_positions[jnt.parent] + off

        t_local = np.broadcast_to(off, (n_frames, 3)).copy()
        R_local = np.broadcast_to(eye, (n_frames, 3, 3)).copy()
        has_rot = False
        for k, ch in enumerate(jnt.channels):
            col = motion[:, jnt.channel_offset + k]
            if ch in _POS_AXIS:
                ax = _POS_AXIS[ch]
                if not has_rot:
                    t_local[:, ax] += col * scale
                else:
                    step = np.zeros((n_frames, 3), dtype=np.float64)
                    step[:, ax] = col * scale
                    t_local += np.einsum("tij,tj->ti", R_local, step)
            else:
                R_local = R_local @ _axis_rotation(_ROT_AXIS[ch], np.radians(col))
                has_rot = True

        if jnt.parent < 0:
            positions[:, j] = t_local
            rotations[:, j] = R_local
        else:
            p = jnt.parent
            positions[:, j] = positions[:, p] + np.einsum(
                "tij,tj->ti", rotations[:, p], t_local
            )
            rotations[:, j] = (
                rotations[:, p] if jnt.is_end_site else rotations[:, p] @ R_local
            )

    return BVHClip(
        joints=joints,
        joint_names=tuple(jnt.name for jnt in joints),
        fps=fps,
        positions=positions,
        rotations=rotations,
        rest_positions=rest_positions,
    )


# ----------------------------------------------------------------------------- mapping


@dataclass(frozen=True)
class BVHPoint:
    """A node of the target tree pairing BVH joint(s) with a model counterpart.

    Attributes:
        name: Landmark name.
        bvh: BVH joint names whose mean position forms this node.
        body: Model body carrying the landmark.
        model: Model bodies whose origins (mean in calibration pose) place the point.
        parent: Parent node name (``None`` for the root).
        carrier: BVH joint whose world orientation carries the model edge from ``parent``.
        subject_direction: Follow the subject's edge direction (scaled to model length)
            and record the alignment rotation for matching :class:`BVHOrientation` entries.
        weight: Position residual weight; ``0`` keeps the node in the scaling tree only.
    """

    name: str
    bvh: tuple[str, ...]
    body: str
    model: tuple[str, ...]
    parent: str | None = None
    carrier: str | None = None
    subject_direction: bool = False
    weight: float = 1.0


@dataclass(frozen=True)
class BVHOrientation:
    """A model body tracking a BVH joint's orientation (offset from T-pose calibration).

    Attributes:
        joint: Canonical BVH joint name.
        body: Target model body.
        weight: Metres of position error one radian is worth.
        axis_to: Optional child model body defining ``body``'s long axis for anisotropic
            axial weighting.
        axis_weight: Weight multiplier on rotations about the long axis.
        align: Optional :class:`BVHPoint` name with ``subject_direction=True`` whose edge
            turn rotates this orientation target.
    """

    joint: str
    body: str
    weight: float = 0.05
    axis_to: str | None = None
    axis_weight: float = 1.0
    align: str | None = None


@dataclass(frozen=True)
class BVHHinge:
    """A model hinge driven per frame from the angle between two BVH segments.

    Attributes:
        joint: Independent model hinge joint name.
        bvh: ``(proximal, hinge, distal)`` BVH joint names.
        model: ``(proximal, hinge, distal)`` model body names.
        weight: Metres of position error one radian is worth.
    """

    joint: str
    bvh: tuple[str, str, str]
    model: tuple[str, str, str]
    weight: float = 0.1


@dataclass(frozen=True)
class BVHMap:
    """Model-specific specification for BVH retargeting.

    Attributes:
        name: Map identifier.
        version: Version bumped whenever targets, weights or offsets change.
        points: Topological target tree (root first, parents before children).
        orientations: Body orientation targets.
        calibration: Model joint overrides (rad) forming the T-pose rest calibration.
        bvh_to_model: ``(3, 3)`` matrix mapping BVH canonical axes into model world axes.
        feet: ``(left, right)`` model foot body tuples carrying sole collision geoms.
        contact_joints: ``(left, right)`` BVH joint names used to detect foot contacts.
        aliases: Candidate alias tuples per canonical BVH joint name.
        twist_pairs: ``(left, right)`` model body pairs for gait twist-symmetry checks.
        twist_reference: Reference heading body for twist-symmetry checks.
        coupling_term: Coupled joint name for twist-symmetry regression.
        period_joint: Joint used to estimate stride period.
        posture: ``(joint, value, weight)`` independent-joint posture priors.
        hinges: Segment-angle hinge priors (e.g. elbows).
    """

    name: str
    version: str
    points: tuple[BVHPoint, ...]
    orientations: tuple[BVHOrientation, ...]
    calibration: dict[str, float] = field(default_factory=dict)
    bvh_to_model: tuple[tuple[float, ...], ...] = tuple(map(tuple, BVH_TO_MODEL))
    feet: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    contact_joints: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    aliases: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_BVH_ALIASES)
    )
    twist_pairs: tuple[tuple[str, str], ...] = ()
    twist_reference: str = ""
    coupling_term: str = ""
    period_joint: str = ""
    posture: tuple[tuple[str, float, float], ...] = ()
    hinges: tuple[BVHHinge, ...] = ()


def _limb_points() -> tuple[BVHPoint, ...]:
    pts = []
    for S, B, m in (("L_", "Left", "_l"), ("R_", "Right", "_r")):
        pts += [
            BVHPoint(
                f"{S}Hip",
                (f"{B}UpLeg",),
                f"femur{m}",
                (f"femur{m}",),
                "Hips",
                carrier="Hips",
            ),
            BVHPoint(
                f"{S}Knee",
                (f"{B}Leg",),
                f"tibia{m}",
                (f"tibia{m}",),
                f"{S}Hip",
                carrier=f"{B}UpLeg",
                subject_direction=True,
            ),
            BVHPoint(
                f"{S}Ankle",
                (f"{B}Foot",),
                f"talus{m}",
                (f"talus{m}",),
                f"{S}Knee",
                carrier=f"{B}Leg",
                subject_direction=True,
            ),
            BVHPoint(
                f"{S}Foot",
                (f"{B}Toe",),
                f"toes{m}",
                (f"toes{m}",),
                f"{S}Ankle",
                carrier=f"{B}Foot",
                subject_direction=True,
            ),
            BVHPoint(
                f"{S}Shoulder",
                (f"{B}Arm",),
                f"humerus{m}",
                (f"humerus{m}",),
                "Shoulders",
                weight=0.3,
            ),
            BVHPoint(
                f"{S}Elbow",
                (f"{B}ForeArm",),
                f"ulna{m}",
                (f"ulna{m}",),
                f"{S}Shoulder",
                carrier=f"{B}Arm",
                subject_direction=True,
            ),
            BVHPoint(
                f"{S}Wrist",
                (f"{B}Hand",),
                f"lunate{m}",
                (f"lunate{m}",),
                f"{S}Elbow",
                carrier=f"{B}ForeArm",
                subject_direction=True,
            ),
        ]
    return tuple(pts)


LAFAN1_MYOFULLBODY = BVHMap(
    name="lafan1-bvh->myofullbody",
    version="1",
    points=(
        BVHPoint("Hips", ("LeftUpLeg", "RightUpLeg"), "pelvis", ("femur_l", "femur_r")),
        BVHPoint(
            "Shoulders",
            ("LeftArm", "RightArm"),
            "torso",
            ("humerus_l", "humerus_r"),
            "Hips",
            weight=0.0,
        ),
        BVHPoint("Head", ("Head",), "head", ("head",), "Shoulders", weight=0.3),
        *_limb_points(),
    ),
    orientations=(
        BVHOrientation("Hips", "pelvis", 0.5),
        BVHOrientation("Spine2", "torso", 0.1),
        BVHOrientation("Head", "head", 0.02),
        BVHOrientation("LeftUpLeg", "femur_l", 0.1),
        BVHOrientation("RightUpLeg", "femur_r", 0.1),
        BVHOrientation("LeftLeg", "tibia_l", 0.1, "talus_l", 0.1, align="L_Ankle"),
        BVHOrientation("RightLeg", "tibia_r", 0.1, "talus_r", 0.1, align="R_Ankle"),
        BVHOrientation("LeftFoot", "calcn_l", 0.1, align="L_Foot"),
        BVHOrientation("RightFoot", "calcn_r", 0.1, align="R_Foot"),
        BVHOrientation("LeftArm", "humerus_l", 0.05, align="L_Elbow"),
        BVHOrientation("RightArm", "humerus_r", 0.05, align="R_Elbow"),
        BVHOrientation("LeftHand", "lunate_l", 0.02, align="L_Wrist"),
        BVHOrientation("RightHand", "lunate_r", 0.02, align="R_Wrist"),
    ),
    calibration={"shoulder_elv_l": np.pi / 2, "shoulder_elv_r": np.pi / 2},
    feet=(("calcn_l", "toes_l"), ("calcn_r", "toes_r")),
    contact_joints=(("LeftFoot", "LeftToe"), ("RightFoot", "RightToe")),
    twist_pairs=(
        ("femur_l", "femur_r"),
        ("tibia_l", "tibia_r"),
        ("calcn_l", "calcn_r"),
    ),
    twist_reference="pelvis",
    coupling_term="knee_angle_rotation3_l",
    period_joint="knee_angle_r",
    posture=(
        ("elv_angle_l", 0.0, 0.01),
        ("elv_angle_r", 0.0, 0.01),
        ("mtp_angle_l", 0.0, 0.02),
        ("mtp_angle_r", 0.0, 0.02),
    ),
    hinges=tuple(
        BVHHinge(
            f"elbow_flexion{m}",
            (f"{B}Arm", f"{B}ForeArm", f"{B}Hand"),
            (f"humerus{m}", f"ulna{m}", f"lunate{m}"),
        )
        for B, m in (("Left", "_l"), ("Right", "_r"))
    ),
)
"""Ubisoft LAFAN1 (and compatible CMU / Mixamo) ``.bvh`` skeletons onto ``myofullbody``."""

BVH_MAPS: dict[str, BVHMap] = {"myofullbody": LAFAN1_MYOFULLBODY}
"""BVH map per target model name."""


# ------------------------------------------------------------------------ calibration


@dataclass
class BVHCalibration:
    """A :class:`BVHMap` resolved on a compiled target ``MjModel``.

    Attributes:
        lmap: Solver :class:`~.retarget.LandmarkMap` with body-local offsets and priors.
        model_pos: ``(P, 3)`` positions of every tree node in the T-pose calibration.
        rot_offset: ``(O, 3, 3)`` body orientation offsets relative to BVH canonical axes.
        qpos: Calibration pose on ``model``'s couplings.
        hip_height: Mid-hip height above the lowest foot collision sole in calibration.
        hinge_table: Per hinge, ``(2, K)`` joint angles and corresponding segment angles.
    """

    lmap: LandmarkMap
    model_pos: np.ndarray
    rot_offset: np.ndarray
    qpos: np.ndarray
    hip_height: float
    hinge_table: tuple[np.ndarray, ...] = ()


def _check_proper_rotation(rot: np.ndarray) -> np.ndarray:
    r = np.asarray(rot, dtype=np.float64)
    if (
        r.shape != (3, 3)
        or not np.allclose(r @ r.T, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(r), 1.0, atol=1e-6)
    ):
        raise ValueError("bvh_to_model must be a proper 3x3 rotation matrix")
    return r


def _lowest_sole(m: mujoco.MjModel, bmap: BVHMap, qpos: np.ndarray) -> float:
    proxy = LandmarkMap("", "", (), feet=bmap.feet)
    return float(foot_kinematics(m, proxy, qpos[None])[0].min())


def calibrate_bvh(
    model: mujoco.MjModel, bmap: BVHMap = LAFAN1_MYOFULLBODY
) -> BVHCalibration:
    """Resolve *bmap* on *model* directly from *model*'s T-pose geometry."""
    m = model
    d, q = calibration_pose(m, bmap.calibration)
    Rc = _check_proper_rotation(bmap.bvh_to_model)
    pos = np.stack(
        [np.mean([d.xpos[m.body(b).id] for b in p.model], axis=0) for p in bmap.points]
    )
    landmarks = body_landmarks(m, d, bmap.points, pos)
    rot = np.stack(
        [Rc.T @ d.xmat[m.body(o.body).id].reshape(3, 3) for o in bmap.orientations]
    )
    orientations = orientation_targets(m, d, bmap.orientations, source_attr="joint")
    lmap = LandmarkMap(
        name=bmap.name,
        version=bmap.version,
        landmarks=landmarks,
        orientations=orientations,
        feet=bmap.feet,
        twist_pairs=bmap.twist_pairs,
        twist_reference=bmap.twist_reference,
        coupling_term=bmap.coupling_term,
        period_joint=bmap.period_joint,
        posture=bmap.posture + tuple((h.joint, 0.0, h.weight) for h in bmap.hinges),
    )
    hip = pos[0, 2] - _lowest_sole(m, bmap, q)
    tables = tuple(hinge_table(m, d, q, h) for h in bmap.hinges)
    return BVHCalibration(lmap, pos, rot, q, float(hip), tables)


# ---------------------------------------------------------------------------- targets


def _resolve_joint(
    name: str,
    jidx: dict[str, int],
    aliases: dict[str, tuple[str, ...]],
    *,
    required: bool = True,
) -> int | None:
    candidates = (name, *(c for c in aliases.get(name, ()) if c != name))
    for cand in candidates:
        if cand in jidx:
            return jidx[cand]
    stripped = {k.split(":")[-1]: v for k, v in jidx.items() if ":" in k}
    for cand in candidates:
        if cand in stripped:
            return stripped[cand]
    if required:
        raise ValueError(
            f"BVH clip is missing joint {name!r}; available joints: {sorted(jidx)}"
        )
    return None


def _resolve_contact_side(
    side_joints: tuple[str, ...],
    jidx: dict[str, int],
    aliases: dict[str, tuple[str, ...]],
) -> list[int]:
    ids: list[int] = []
    for name in side_joints:
        jid = _resolve_joint(name, jidx, aliases, required=True)
        assert jid is not None
        if jid not in ids:
            ids.append(jid)
        end_id = _resolve_joint(f"{name}_End", jidx, aliases, required=False)
        if end_id is not None and end_id not in ids:
            ids.append(end_id)
    return ids


def knee_flexion(
    clip: BVHClip,
    bmap: BVHMap = LAFAN1_MYOFULLBODY,
    rotations: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """``(T,)`` source knee flexion in radians per leg (negative is hyperextension)."""
    jidx = {n: i for i, n in enumerate(clip.joint_names)}
    rots = clip.rotations if rotations is None else rotations
    out: dict[str, np.ndarray] = {}
    for side, B in (("l", "Left"), ("r", "Right")):
        i_hip = _resolve_joint(f"{B}UpLeg", jidx, bmap.aliases)
        i_knee = _resolve_joint(f"{B}Leg", jidx, bmap.aliases)
        assert i_hip is not None and i_knee is not None
        R = np.swapaxes(rots[:, i_hip], -1, -2) @ rots[:, i_knee]
        out[side] = np.arctan2(R[:, 2, 1], R[:, 1, 1])
    return out


@dataclass
class BVHSource:
    """Biovision Hierarchy (``.bvh``) files as a :class:`~.retarget.ClipSource`.

    Args:
        bmap: Target mapping specification.
        calibration: Model calibration (:func:`calibrate_bvh`).
        scale: Linear unit factor converting BVH coordinates to metres (``0.01`` for cm).
        trim: Source frames dropped from the start and end of each clip.
        contact_height: Maximum vertical height above floor (m) for foot contact.
        contact_speed: Maximum horizontal speed (m/s) for foot contact.
        ground_sigma: Gaussian smoothing (s) of per-frame root-height grounding.
        target_sigma: Gaussian smoothing (s) of landmark positions and orientations.
    """

    bmap: BVHMap
    calibration: BVHCalibration
    scale: float = 0.01
    trim: tuple[int, int] = (0, 0)
    contact_height: float = 0.03
    contact_speed: float = 0.5
    ground_sigma: float = 0.05
    target_sigma: float = 0.01
    name: str = "bvh"
    _last: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def targets(
        self, path: str | Path | BVHClip, model: mujoco.MjModel, lmap: LandmarkMap
    ) -> ClipTargets:
        """Build :class:`~.retarget.ClipTargets` of a ``.bvh`` clip on *model*."""
        c, bmap = self.calibration, self.bmap
        clip = path if isinstance(path, BVHClip) else parse_bvh(path, scale=self.scale)
        n_total = len(clip.positions)
        t0, t1 = self.trim
        if t0 < 0 or t1 < 0 or (n_total - t1 - t0) < 1:
            raise ValueError(
                f"{path}: trim={self.trim} leaves no frames from {n_total} source frames"
            )
        sl = slice(t0, n_total - t1)

        Rc = _check_proper_rotation(bmap.bvh_to_model)
        pos_w = clip.positions[sl] @ Rc.T
        rest_w = clip.rest_positions @ Rc.T
        rot_bvh = clip.rotations[sl]
        rot_w = np.einsum("ab,tjbc->tjac", Rc, rot_bvh)

        jidx = {n: i for i, n in enumerate(clip.joint_names)}
        names = [p.name for p in bmap.points]

        def resolve_ids(joint_names: tuple[str, ...]) -> list[int]:
            out_ids: list[int] = []
            for jn in joint_names:
                jid = _resolve_joint(jn, jidx, bmap.aliases, required=True)
                assert jid is not None
                out_ids.append(jid)
            return out_ids

        x = [pos_w[:, resolve_ids(p.bvh)].mean(axis=1) for p in bmap.points]
        xr = [rest_w[resolve_ids(p.bvh)].mean(axis=0) for p in bmap.points]

        left_c = _resolve_contact_side(bmap.contact_joints[0], jidx, bmap.aliases)
        right_c = _resolve_contact_side(bmap.contact_joints[1], jidx, bmap.aliases)
        all_c = left_c + right_c
        sole = pos_w[:, all_c]
        sole_floor = np.percentile(sole[..., 2], 2, axis=0)
        floor = float(sole_floor.min()) if sole_floor.size else 0.0

        rest_low = float(rest_w[all_c, 2].min()) if all_c else float(rest_w[:, 2].min())
        hip_rest = max(float(xr[0][2] - rest_low), 1e-6)
        s_root = c.hip_height / hip_rest

        T = len(x[0])
        out = np.empty((len(names), T, 3), dtype=np.float64)
        root = x[0].copy()
        root[:, :2] = s_root * (root[:, :2] - root[0, :2])
        root[:, 2] = s_root * (root[:, 2] - floor)
        out[0] = root

        ratio: dict[int, float] = {}
        turn: dict[str, np.ndarray] = {}
        for k, p in enumerate(bmap.points[1:], 1):
            assert p.parent is not None
            par = names.index(p.parent)
            edge = c.model_pos[k] - c.model_pos[par]
            subj_len = max(float(np.linalg.norm(xr[k] - xr[par])), 1e-8)
            ratio[k] = float(np.linalg.norm(edge)) / subj_len
            subject = ratio[k] * (x[k] - x[par])
            if p.carrier:
                cid = _resolve_joint(p.carrier, jidx, bmap.aliases, required=True)
                assert cid is not None
                carried = rot_w[:, cid] @ (Rc.T @ edge)
                if p.subject_direction:
                    turn[p.name] = rotation_between(carried, subject)
                    out[k] = out[par] + subject
                else:
                    out[k] = out[par] + carried
            else:
                out[k] = out[par] + subject

        keep = [names.index(lm.source) for lm in lmap.landmarks]
        pos = out[keep].transpose(1, 0, 2)

        ori_ids = [
            _resolve_joint(o.joint, jidx, bmap.aliases, required=True)
            for o in bmap.orientations
        ]
        R = rot_w[:, [i for i in ori_ids if i is not None]].copy()
        for i, o in enumerate(bmap.orientations):
            if o.align is not None:
                R[:, i] = turn[o.align] @ R[:, i]
        quat = mat_to_quat(np.einsum("toab,obc->toac", R, c.rot_offset))

        fps = clip.fps
        if self.target_sigma > 0 and T >= 2:
            pos = smooth_signal(pos, self.target_sigma * fps)
            quat = smooth_quat(quat, self.target_sigma * fps)

        posture = None
        if bmap.hinges:
            posture = np.tile([v for _, v, _ in lmap.posture], (T, 1))
            for k, (h, tab) in enumerate(zip(bmap.hinges, c.hinge_table)):
                h_ids = resolve_ids(h.bvh)
                a, b_pt, e = (pos_w[:, i] for i in h_ids)
                theta = segment_angle(a, b_pt, e)
                if self.target_sigma > 0 and T >= 2:
                    theta = smooth_signal(theta, self.target_sigma * fps)
                posture[:, len(bmap.posture) + k] = np.interp(theta, tab[1], tab[0])

        if T >= 2 and all_c:
            contact = foot_contacts(
                sole,
                sole_floor,
                fps,
                len(left_c),
                self.contact_height,
                self.contact_speed,
            )
        else:
            contact = np.ones((T, 2), dtype=bool)

        seed = start_pose(model, c.qpos, c.model_pos[0], pos[0, 0], quat[0], lmap)
        self._last = {
            "floor": floor,
            "s_root": float(s_root),
            "contact": contact,
            "frames_source": n_total,
            "edge_ratio": {names[k]: float(r) for k, r in ratio.items()},
            "knee_flexion": knee_flexion(clip, bmap, rotations=rot_bvh),
        }
        return ClipTargets(fps, pos, quat, seed, contact, posture)

    def finish(
        self, ik: LandmarkIK, targets: ClipTargets, qpos: np.ndarray
    ) -> tuple[np.ndarray, ClipTargets, dict[str, Any]]:
        """Ground the fit so standing feet touch ``z = 0`` while flight stays airborne."""
        info = self._last
        contact = info["contact"]
        sigma = self.ground_sigma if len(qpos) >= 2 else 0.0
        qpos, targets, dz = ground_clip(
            ik.model, ik.lmap, targets, qpos, contact, sigma
        )
        meta = {
            "bvh_map": self.bmap.name,
            "bvh_map_version": self.bmap.version,
            "scale": float(self.scale),
            "trim": list(self.trim),
            "frames_source": info["frames_source"],
            "subject_floor_m": info["floor"],
            "root_scale": info["s_root"],
            "edge_ratio": info["edge_ratio"],
            "ground_shift_m": {
                "mean": float(dz.mean()),
                "min": float(dz.min()),
                "max": float(dz.max()),
            },
            "contact_fraction": [float(x) for x in contact.mean(0)],
            "source_knee_flexion_deg": knee_flexion_meta(info["knee_flexion"]),
        }
        return qpos, targets, meta


def bvh_source(
    model: mujoco.MjModel,
    bmap: BVHMap = LAFAN1_MYOFULLBODY,
    **kwargs: Any,
) -> tuple[BVHSource, LandmarkMap]:
    """Build a calibrated :class:`BVHSource` and solver :class:`~.retarget.LandmarkMap`."""
    cal = calibrate_bvh(model, bmap)
    return BVHSource(bmap, cal, **kwargs), cal.lmap


def landmark_bodies(bmap: BVHMap) -> tuple[str, ...]:
    """Model bodies referenced by *bmap*'s points, orientations and hinges."""
    names = {b for p in bmap.points for b in (p.body, *p.model)}
    names |= {b for o in bmap.orientations for b in (o.body, o.axis_to) if b}
    names |= {b for h in bmap.hinges for b in h.model}
    return tuple(sorted(names))


def map_dict(bmap: BVHMap) -> dict[str, Any]:
    """JSON-serialisable dictionary form of *bmap*."""
    return asdict(bmap)


__all__ = [
    "BVH_MAPS",
    "BVH_TO_MODEL",
    "DEFAULT_BVH_ALIASES",
    "LAFAN1_MYOFULLBODY",
    "BVHCalibration",
    "BVHClip",
    "BVHHinge",
    "BVHJoint",
    "BVHMap",
    "BVHOrientation",
    "BVHPoint",
    "BVHSource",
    "bvh_source",
    "calibrate_bvh",
    "knee_flexion",
    "landmark_bodies",
    "map_dict",
    "parse_bvh",
]
