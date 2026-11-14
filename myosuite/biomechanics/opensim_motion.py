# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""OpenSim motions (``.mot`` / ``.sto``) on MyoSuite models.

OpenSim's inverse kinematics writes generalized coordinates by name. MyoSuite's leg
and body models keep OpenSim's joint names and body frames, but place the pelvis
on a free joint in a Z-up world instead of six pelvis coordinates in OpenSim's
Y-up ground. :func:`opensim_motion_to_qpos` maps one onto the other::

    motion = read_opensim_motion("subject01_walk1_ik.mot")
    clip = opensim_motion_to_qpos(model, motion, rate=100.0)  # 10 physics steps
    kin = clip_kinematics(model, clip.qpos, clip.dt, lowpass_hz=6.0)

The pelvis follows OpenSim's ``ground_pelvis`` joint of gait2392 and its
successors: ``pelvis_tilt``, ``pelvis_list`` and ``pelvis_rotation`` rotate about
the pelvis's z, x and y axes in that body-fixed order, and ``pelvis_tx/ty/tz``
place its origin in the ground frame. The model is not scaled to the subject, so
the feet can float above or sink into the floor (OpenSim's subject01 walk floats
8-10 cm on MyoLeg): lower the root until the stance feet touch the floor, or scale
the model, before estimating ground reactions.
"""

from __future__ import annotations

from collections.abc import Mapping
import warnings
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline

from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

# OpenSim's Y-up ground in a Z-up world, its x axis (walking direction) kept.
GROUND_Y_UP = (float(np.sqrt(0.5)), float(np.sqrt(0.5)), 0.0, 0.0)
# gait2392's knees flex negative, MyoSuite's positive.
GAIT2392_SIGNS: Mapping[str, float] = {"knee_angle_r": -1.0, "knee_angle_l": -1.0}
# A mapped joint this far (fraction of its range) outside its limits is reported.
RANGE_TOLERANCE = 0.05
PELVIS_COORDINATES = (
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
    "pelvis_tx",
    "pelvis_ty",
    "pelvis_tz",
)


@dataclass(frozen=True)
class OpenSimMotion:
    """Columns of an OpenSim storage file.

    Attributes:
        time: ``(T,)`` frame times [s].
        labels: Column labels after ``time``.
        values: ``(T, len(labels))`` values as stored.
        in_degrees: Whether rotational coordinates are in degrees.
    """

    time: np.ndarray
    labels: tuple[str, ...]
    values: np.ndarray
    in_degrees: bool

    def column(self, label: str) -> np.ndarray:
        """Values of the first column named *label*."""
        return self.values[:, self.labels.index(label)]


def read_opensim_motion(
    path: str | Path, *, in_degrees: bool | None = None
) -> OpenSimMotion:
    """Read an OpenSim ``.mot`` or ``.sto`` file.

    Args:
        path: File with a header ending in ``endheader``, a line of column
            labels starting with ``time``, and one row per frame.
        in_degrees: Units of rotational coordinates; ``None`` reads the header's
            ``inDegrees``, and degrees when it has none (the SIMM ``.mot``
            convention).

    Returns:
        The file's columns.

    Raises:
        ValueError: Without an ``endheader`` line or a leading ``time`` column.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    try:
        end = next(i for i, line in enumerate(lines) if line.strip() == "endheader")
    except StopIteration:
        raise ValueError(f"{path}: no 'endheader' line") from None
    header = {}
    for line in lines[:end]:
        key, sep, value = line.partition("=")
        if sep:
            header[key.strip().lower()] = value.strip().lower()
    labels = lines[end + 1].split("\t" if "\t" in lines[end + 1] else None)
    labels = [label.strip() for label in labels]
    if labels[0].lower() != "time":
        raise ValueError(f"{path}: the first column is {labels[0]!r}, not 'time'")
    data = np.loadtxt(lines[end + 2 :], ndmin=2)
    if in_degrees is None:
        in_degrees = header.get("indegrees", "yes") == "yes"
    return OpenSimMotion(
        time=data[:, 0],
        labels=tuple(labels[1:]),
        values=data[:, 1:],
        in_degrees=bool(in_degrees),
    )


@dataclass(frozen=True)
class MappedMotion:
    """An OpenSim motion as model positions.

    Attributes:
        qpos: ``(T, nq)`` positions; joints the motion lacks stay at ``qpos0``.
        time: ``(T,)`` frame times [s].
        dt: Frame period [s].
        coordinates: Labels mapped onto model joints (pelvis included).
        unmapped: Labels left out: coordinates without a model joint (e.g.
            ``lumbar_extension`` on MyoLeg), markers, forces.
    """

    qpos: np.ndarray
    time: np.ndarray
    dt: float
    coordinates: tuple[str, ...]
    unmapped: tuple[str, ...]


def _quat(axis: int, angle: np.ndarray) -> np.ndarray:
    """``(T, 4)`` rotations by *angle* about coordinate axis *axis*."""
    q = np.zeros((len(angle), 4))
    q[:, 0], q[:, 1 + axis] = np.cos(0.5 * angle), np.sin(0.5 * angle)
    return q


def _mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise quaternion products ``a b``."""
    out = np.empty(np.broadcast_shapes(a.shape, b.shape))
    for i in range(len(out)):
        mujoco.mju_mulQuat(out[i], a[i % len(a)], b[i % len(b)])
    return out


def _free_body(model: mujoco.MjModel, body: int) -> tuple[int, np.ndarray, np.ndarray]:
    """The free joint above *body*, and *body*'s pose in that joint's body frame."""
    chain = [body]
    while chain[-1] != 0:
        b = chain[-1]
        joints = range(
            model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]
        )
        free = [j for j in joints if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if free:
            data = mujoco.MjData(model)
            mujoco.mj_kinematics(model, data)
            inv = np.empty(4)
            mujoco.mju_negQuat(inv, data.xquat[b])
            pos, quat = np.empty(3), np.empty(4)
            mujoco.mju_rotVecQuat(pos, data.xpos[body] - data.xpos[b], inv)
            mujoco.mju_mulQuat(quat, inv, data.xquat[body])
            return free[0], pos, quat
        if model.body_jntnum[b]:
            break
        chain.append(model.body_parentid[b])
    raise ValueError(
        f"no free joint above body {model.body(body).name!r} without other joints"
    )


def _level_ground(model: mujoco.MjModel, body: int) -> np.ndarray:
    """OpenSim's ground, Y up, its x axis along *body*'s heading at ``qpos0``."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    forward = data.xmat[body].reshape(3, 3)[:, 0]
    yaw = np.array([np.arctan2(forward[1], forward[0])])
    return _mul(_quat(2, yaw), _quat(0, np.array([0.5 * np.pi])))[0]


def _warn_out_of_range(
    model: mujoco.MjModel, qpos: np.ndarray, names: set[str]
) -> None:
    """Warn about mapped limited joints that leave their range (wrong sign or units)."""
    outside = []
    for name in sorted(names):
        j = model.joint(name).id
        if not model.jnt_limited[j]:
            continue
        lo, hi = model.jnt_range[j]
        q = qpos[:, model.jnt_qposadr[j]]
        excess = np.maximum(lo - q, q - hi) / (hi - lo)
        share = float(np.mean(excess > RANGE_TOLERANCE))
        if share:
            outside.append(f"{name} ({share:.0%} of frames)")
    if outside:
        warnings.warn(
            "mapped coordinates outside their joint range: "
            + ", ".join(outside)
            + "; check `signs` (Rajagopal-family files need signs={}) and the units",
            UserWarning,
            stacklevel=3,
        )


def opensim_motion_to_qpos(
    model: mujoco.MjModel,
    motion: OpenSimMotion,
    *,
    rate: float | None = None,
    root_body: str | None = "pelvis",
    ground_quat: tuple[float, float, float, float] | None = None,
    signs: Mapping[str, float] = GAIT2392_SIGNS,
    project_couplings: bool = True,
) -> MappedMotion:
    """Model positions for an OpenSim motion.

    Hinge and slide joints take the coordinates of the same name (radians and
    meters, times ``signs``). The free joint above *root_body* takes the pelvis
    coordinates: *root_body* is placed at OpenSim's pelvis pose in OpenSim's
    ground, which *ground_quat* places in the world.

    Args:
        model: Compiled model with OpenSim's joint names and body frames.
        motion: From :func:`read_opensim_motion`.
        rate: Resample the coordinates to this frame rate [Hz] with cubic
            splines, e.g. to a whole number of physics steps per frame for
            computed muscle control; ``None`` keeps the file's frames.
        root_body: Body that OpenSim's pelvis coordinates place; ``None`` leaves
            the root at ``qpos0``.
        ground_quat: OpenSim's ground frame in the world (``w, x, y, z``).
            ``None`` keeps it level (Y up to Z up) and turns its x axis (the
            walking direction) to *root_body*'s heading at ``qpos0``, so that
            zero pelvis angles give the model's default pose; on MyoLeg the walk
            then goes along -y. :data:`GROUND_Y_UP` keeps OpenSim's x axis.
        signs: Per coordinate, the sign that turns OpenSim's into the model's
            convention. The default is gait2392's (knee flexion negative);
            Rajagopal-family models (OpenCap, AddBiomechanics) store knee flexion
            positive and need ``signs={}``. A mapped joint more than
            :data:`RANGE_TOLERANCE` of its range outside its limits raises a
            ``UserWarning`` naming it, which points at the wrong ``signs`` or units.
        project_couplings: Put coupled joints on their polynomials, as
            :func:`~myosuite.biomechanics.inverse_dynamics.clip_kinematics` does.

    Returns:
        Positions per frame, the frame times and period, and which labels were
        used.

    Raises:
        ValueError: If pelvis coordinates or the free joint are missing, or the
            frames are not evenly spaced and no *rate* is given.
    """
    time, values = motion.time, motion.values
    if rate is not None:
        time = np.arange(time[0], time[-1] + 1e-9, 1.0 / rate)
        values = CubicSpline(motion.time, motion.values)(time)
        dt = 1.0 / rate
    else:
        # Files often print times rounded to 1 ms, so check against an even grid.
        dt = float(time[-1] - time[0]) / max(len(time) - 1, 1)
        grid = time[0] + dt * np.arange(len(time))
        if np.abs(time - grid).max() > 0.1 * dt:
            raise ValueError("frames are not evenly spaced; pass a rate to resample")

    def column(label: str) -> np.ndarray:
        return values[:, motion.labels.index(label)]

    angle = np.deg2rad if motion.in_degrees else (lambda x: x)
    qpos = np.tile(model.qpos0, (len(time), 1))
    used = set()

    # Plain ints: some MuJoCo builds' enums do not compare equal to NumPy scalars.
    hinge, slide = int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
    types = model.jnt_type.tolist()
    for j in range(model.njnt):
        name = model.joint(j).name
        if types[j] in (hinge, slide) and name in motion.labels:
            value = column(name)
            if types[j] == hinge:
                value = angle(value)
            qpos[:, model.jnt_qposadr[j]] = signs.get(name, 1.0) * value
            used.add(name)
    _warn_out_of_range(model, qpos, used)

    if root_body is not None:
        missing = [c for c in PELVIS_COORDINATES if c not in motion.labels]
        if missing:
            raise ValueError(f"pelvis coordinates missing: {missing}")
        tilt, lst, rot = (angle(column(c)) for c in PELVIS_COORDINATES[:3])
        origin = np.stack([column(c) for c in PELVIS_COORDINATES[3:]], axis=1)
        body = model.body(root_body).id
        if ground_quat is None:
            ground = _level_ground(model, body)[None]
        else:
            ground = np.asarray(ground_quat, dtype=np.float64)[None]
            ground = ground / np.linalg.norm(ground)
        # Pelvis in the world: ground rotation, then body-fixed z-x-y rotations.
        pelvis = _mul(ground, _mul(_quat(2, tilt), _mul(_quat(0, lst), _quat(1, rot))))
        position = np.empty_like(origin)
        for t in range(len(origin)):
            mujoco.mju_rotVecQuat(position[t], origin[t], ground[0])
        joint, rel_pos, rel_quat = _free_body(model, body)
        inv_rel = np.empty(4)
        mujoco.mju_negQuat(inv_rel, rel_quat)
        root = _mul(pelvis, inv_rel[None])
        adr = model.jnt_qposadr[joint]
        for t in range(len(root)):
            offset = np.empty(3)
            mujoco.mju_rotVecQuat(offset, rel_pos, root[t])
            qpos[t, adr : adr + 3] = position[t] - offset
        qpos[:, adr + 3 : adr + 7] = root
        used.update(PELVIS_COORDINATES)

    if project_couplings:
        project_joint_couplings(JointCouplings.from_model(model), qpos)
    return MappedMotion(
        qpos=qpos,
        time=np.array(time),
        dt=float(dt),
        coordinates=tuple(label for label in motion.labels if label in used),
        unmapped=tuple(label for label in motion.labels if label not in used),
    )
