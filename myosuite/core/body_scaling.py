# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Scale a musculoskeletal model to a subject, as OpenSim's Scale Tool does.

Each body gets per-axis scale factors in its own frame (OpenSim's convention):

* everything a body carries (geoms, meshes, sites, joint anchors, cameras,
  lights, frames, the inertial frame) is scaled by that body's factors, and a
  child body's position by its parent's;
* slide-joint translations, their ranges and their joint couplings scale with
  the parent body along the joint axis;
* every muscle's optimal fibre length and tendon slack length scale by the change
  of its muscle-tendon length in the default pose (``qpos0``). MuJoCo derives
  both from ``lengthrange``, so ``lengthrange`` is scaled by that ratio;
* peak muscle force is kept (OpenSim does not scale strength) unless
  ``force_scale`` is given.

The topology (joints, actuators, sites, tendons) does not change, so the
observation and action spaces of an env stay the same.

Scale factors can come from an OpenSim Scale Tool result, e.g. a model scaled by
AddBiomechanics or a ScaleSet file (:func:`read_opensim_scales`). Segment names
of the Rajagopal model map onto MyoSuite bodies with
:data:`RAJAGOPAL_MYOLEG_SEGMENTS` and :data:`RAJAGOPAL_MYOFULLBODY_SEGMENTS`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from myosuite.core.muscle_conditions import _peak_force

_LEG_SEGMENTS = (
    "femur",
    "tibia",
    "patella",
    "talus",
    "calcn",
    "toes",
)

#: Rajagopal model segment -> first body of that segment in MyoLeg
#: (``myolegs_with_torso``). Every body takes its nearest mapped ancestor's scale.
RAJAGOPAL_MYOLEG_SEGMENTS: dict[str, tuple[str, ...]] = {
    "pelvis": ("root",),
    "torso": ("torso",),
    **{f"{seg}_{side}": (f"{seg}_{side}",) for seg in _LEG_SEGMENTS for side in "rl"},
}

#: Rajagopal model segment -> first body of that segment in the MuscleMimic
#: MyoFullBody. The lumbar spine, thorax, shoulder girdle and head follow
#: ``torso``; the 27 hand bodies follow ``hand``.
RAJAGOPAL_MYOFULLBODY_SEGMENTS: dict[str, tuple[str, ...]] = {
    "pelvis": ("Full Body",),
    "torso": ("lumbar5",),
    **{f"{seg}_{side}": (f"{seg}_{side}",) for seg in _LEG_SEGMENTS for side in "rl"},
    **{
        f"{seg}_{side}": (f"{seg}_{side}",)
        for seg in ("humerus", "ulna", "radius")
        for side in "rl"
    },
    "hand_r": ("lunate_r",),
    "hand_l": ("lunate_l",),
}

_ORIENT = mujoco.mjtOrientation
_UNIT = np.ones(3)


def read_opensim_scales(
    path: str | Path, reference: str | Path | None = None
) -> dict[str, np.ndarray]:
    """Read per-segment scale factors from an OpenSim file.

    Accepts a ScaleSet (``<Scale><segment>…<scales>sx sy sz</scales>``, as in a
    Scale Tool setup or result) or a scaled model (``.osim``), whose bodies carry
    the applied factors as their geometry's ``scale_factors``.

    Args:
        path: ScaleSet XML or scaled ``.osim`` model.
        reference: The unscaled model, when its geometry already has scale
            factors other than one; the result is divided by them.

    Returns:
        Segment name -> ``(sx, sy, sz)`` in the segment's own frame.
    """
    scales = _scale_set(ET.parse(path).getroot()) or _model_scales(
        ET.parse(path).getroot()
    )
    if reference is not None:
        base = _model_scales(ET.parse(reference).getroot())
        scales = {name: s / base.get(name, _UNIT) for name, s in scales.items()}
    return scales


def _scale_set(root: ET.Element) -> dict[str, np.ndarray]:
    scales = {}
    for scale in root.iter("Scale"):
        if (scale.findtext("apply") or "true").strip().lower() == "false":
            continue
        segment, values = scale.findtext("segment"), scale.findtext("scales")
        if segment and values:
            scales[segment.strip()] = np.array(values.split(), dtype=float)
    return scales


def _model_scales(root: ET.Element) -> dict[str, np.ndarray]:
    scales = {}
    for body in root.iter("Body"):
        name, factors = body.get("name"), body.find(".//scale_factors")
        if name and factors is not None and factors.text:
            scales[name] = np.array(factors.text.split(), dtype=float)
    return scales


def scale_bodies(
    spec: mujoco.MjSpec,
    scales: Mapping[str, float | ArrayLike],
    *,
    segments: Mapping[str, Sequence[str]] | None = None,
    mass: Literal["volume", "keep"] = "volume",
    total_mass: float | None = None,
    force_scale: float = 1.0,
) -> mujoco.MjSpec:
    """Scale bodies of *spec* in place (see the module docstring for the rules).

    Args:
        spec: Model to edit; edited in place and returned.
        scales: Body name (or segment name with *segments*) -> one factor or
            per-axis factors ``(sx, sy, sz)`` in that body's frame.
        segments: Segment -> its first bodies (e.g.
            :data:`RAJAGOPAL_MYOFULLBODY_SEGMENTS`). Each body then takes the
            scale of its nearest ancestor (or itself) that starts a segment, and
            a segment missing from *scales* stays unscaled. Segments of *scales*
            the mapping lacks (e.g. arms for MyoLeg) are ignored.
        mass: ``"volume"`` scales each body's mass with its volume
            (``sx * sy * sz``); ``"keep"`` keeps it.
        total_mass: Rescale all body masses to this total (kg) afterwards.
            ``mass="keep"`` with ``total_mass`` is OpenSim's "preserve mass
            distribution".
        force_scale: Factor on every muscle's peak force.

    Returns:
        The edited spec. With no factor other than one, no ``total_mass`` and
        ``force_scale=1`` it is left untouched.

    Raises:
        KeyError: If a body (or a segment's first body) is not in the model.
        NotImplementedError: For a mesh shared by geoms needing different
            scales, or a connect/weld equality or height field on a scaled body.
    """
    body_scales = _body_scales(spec, scales, segments)
    if not body_scales and total_mass is None and force_scale == 1.0:
        return spec

    # Compile copies only: compiling *spec* itself would freeze its frames.
    ref = spec.copy().compile()
    ref_data = mujoco.MjData(ref)
    mujoco.mj_forward(ref, ref_data)
    ref_actuator_length = ref_data.actuator_length.copy()
    ref_tendon_length = ref_data.ten_length.copy()
    peak_force = _peak_force(ref)

    _check_equalities(spec, body_scales)
    slide = _scale_geometry(spec, ref, body_scales)
    _scale_inertia(spec, ref, body_scales, mass, total_mass)
    _scale_joint_couplings(spec, slide)

    scaled = spec.copy().compile()
    data = mujoco.MjData(scaled)
    mujoco.mj_forward(scaled, data)
    tendon_ratio = _ratio(data.ten_length, ref_tendon_length)
    _scale_tendons(spec, tendon_ratio)
    muscle_ratio = _ratio(data.actuator_length, ref_actuator_length)
    for i, actuator in enumerate(spec.actuators):
        if ref.actuator_gaintype[i] != mujoco.mjtGain.mjGAIN_MUSCLE:
            continue
        actuator.lengthrange = ref.actuator_lengthrange[i] * muscle_ratio[i]
        # MuJoCo derives an automatic peak force from the (now scaled) mass.
        if force_scale != 1.0 or ref.actuator_gainprm[i, 2] < 0:
            actuator.gainprm[2] = peak_force[i] * force_scale
            if ref.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_MUSCLE:
                actuator.biasprm[2] = peak_force[i] * force_scale
    return spec


def _body_scales(
    spec: mujoco.MjSpec,
    scales: Mapping[str, float | ArrayLike],
    segments: Mapping[str, Sequence[str]] | None,
) -> dict[str, np.ndarray]:
    """Body name -> per-axis factors, for bodies with a factor other than one."""
    names = {b.name for b in spec.bodies}
    if segments is None:
        per_body = {name: _factors(s) for name, s in scales.items()}
    else:
        # A segment missing from *scales* stays unscaled, as in OpenSim.
        starts = {
            body: _factors(scales.get(segment, 1.0))
            for segment, first_bodies in segments.items()
            for body in first_bodies
        }
        missing = set(starts) - names
        if missing:
            raise KeyError(f"Segment bodies not in the model: {sorted(missing)}")
        per_body = {}
        for body in spec.bodies[1:]:
            ancestor = body
            while ancestor.name not in starts and ancestor.parent is not None:
                ancestor = ancestor.parent
            if ancestor.name in starts:
                if not body.name:
                    raise ValueError(
                        f"An unnamed body below {ancestor.name!r} cannot be scaled."
                    )
                per_body[body.name] = starts[ancestor.name]
    unknown = set(per_body) - names
    if unknown:
        raise KeyError(f"Bodies not in the model: {sorted(unknown)}")
    return {name: s for name, s in per_body.items() if not np.array_equal(s, _UNIT)}


def _factors(scale: float | ArrayLike) -> np.ndarray:
    factors = np.broadcast_to(np.asarray(scale, dtype=float), (3,)).copy()
    if np.any(factors <= 0):
        raise ValueError(f"Scale factors must be positive, got {factors}.")
    return factors


def _ratio(new: np.ndarray, old: np.ndarray) -> np.ndarray:
    ratio = np.ones_like(old)
    nonzero = np.abs(old) > mujoco.mjMINVAL
    ratio[nonzero] = new[nonzero] / old[nonzero]
    return ratio


def _quat(spec: mujoco.MjSpec, element) -> np.ndarray:
    """Orientation of *element* in its parent frame, resolving euler/axisangle/…"""
    alt, quat = element.alt, np.zeros(4)
    kind = int(alt.type)
    to_rad = np.pi / 180.0 if spec.compiler.degree else 1.0
    if kind == int(_ORIENT.mjORIENTATION_QUAT):
        quat[:] = element.quat
        quat /= np.linalg.norm(quat)
    elif kind == int(_ORIENT.mjORIENTATION_EULER):
        mujoco.mju_euler2Quat(
            quat,
            np.asarray(alt.euler, dtype=float) * to_rad,
            "".join(spec.compiler.eulerseq),
        )
    elif kind == int(_ORIENT.mjORIENTATION_AXISANGLE):
        axis = np.asarray(alt.axisangle[:3], dtype=float)
        mujoco.mju_axisAngle2Quat(
            quat, axis / np.linalg.norm(axis), alt.axisangle[3] * to_rad
        )
    elif kind == int(_ORIENT.mjORIENTATION_ZAXIS):
        mujoco.mju_quatZ2Vec(quat, np.asarray(alt.zaxis, dtype=float))
    else:  # xyaxes: x, then y made orthogonal to it, as MuJoCo does
        x = np.asarray(alt.xyaxes[:3], dtype=float)
        x /= np.linalg.norm(x)
        y = np.asarray(alt.xyaxes[3:], dtype=float)
        y -= x * (x @ y)
        y /= np.linalg.norm(y)
        mujoco.mju_mat2Quat(quat, np.column_stack([x, y, np.cross(x, y)]).ravel())
    return quat


def _rot(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


def _frame_pose(spec: mujoco.MjSpec, frame) -> tuple[np.ndarray, np.ndarray]:
    """Origin and rotation of a (possibly nested) frame in its body's coordinates."""
    if frame is None:
        return np.zeros(3), np.eye(3)
    origin, rot = _frame_pose(spec, frame.frame)
    return origin + rot @ np.asarray(frame.pos, dtype=float), rot @ _rot(
        _quat(spec, frame)
    )


def _frame_rot(spec: mujoco.MjSpec, frame) -> np.ndarray:
    """Rotation of a (possibly nested) frame in its body's coordinates."""
    return _frame_pose(spec, frame)[1]


def _scaled_point(spec: mujoco.MjSpec, frame, s: np.ndarray, point) -> np.ndarray:
    """A point given in *frame* coordinates, moved as its body scales by ``diag(s)``.

    Frames themselves are left alone: MuJoCo ignores edits of a frame's position
    once the spec has been compiled, so the scaling goes into the point instead.
    """
    origin, rot = _frame_pose(spec, frame)
    return rot.T @ (s * (origin + rot @ np.asarray(point, dtype=float)) - origin)


def _axis_factors(s: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Scale along each axis of a frame rotated by *rot* in body coordinates."""
    return np.linalg.norm(np.diag(s) @ rot, axis=0)


def _scale_geometry(
    spec: mujoco.MjSpec, ref: mujoco.MjModel, body_scales: Mapping[str, np.ndarray]
) -> dict[str, float]:
    """Scale positions and sizes; return slide joint -> translation factor."""
    slide = {}
    mesh_scales: dict[str, np.ndarray] = {}
    for body in spec.bodies[1:]:
        parent_s = body_scales.get(body.parent.name, _UNIT) if body.parent else _UNIT
        body_rot = _frame_rot(spec, body.frame) @ _rot(_quat(spec, body))
        for joint in body.joints:
            if joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
                axis = body_rot @ np.asarray(joint.axis, dtype=float)
                factor = float(np.linalg.norm(parent_s * axis) / np.linalg.norm(axis))
                if factor != 1.0:
                    slide[joint.name] = factor
                    joint.range = np.asarray(joint.range) * factor
                    joint.ref *= factor
                    joint.springref *= factor
        if not np.array_equal(parent_s, _UNIT):
            body.pos = _scaled_point(spec, body.frame, parent_s, body.pos)
        s = body_scales.get(body.name)
        if s is None:
            continue
        for element in (*body.joints, *body.cameras, *body.lights):
            element.pos = _scaled_point(spec, element.frame, s, element.pos)
        for site in body.sites:
            _scale_placement(spec, site, s)
            site.size = np.asarray(site.size) * float(np.prod(s) ** (1 / 3))
        for geom in body.geoms:
            _scale_geom(spec, geom, s, mesh_scales)
    shared = {
        g.meshname
        for g in spec.geoms
        if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.parent.name not in body_scales
    } & set(mesh_scales)
    if shared:
        raise NotImplementedError(
            f"Meshes {sorted(shared)} are also used by unscaled bodies; give each its own mesh asset."
        )
    for name, factors in mesh_scales.items():
        mesh = spec.mesh(name)
        mesh.scale = np.asarray(mesh.scale) * factors
    _scale_keyframes(spec, ref, slide)
    return slide


def _scale_placement(spec: mujoco.MjSpec, element, s: np.ndarray) -> None:
    """Scale the position, or the ``fromto`` end points, of a geom or site."""
    if np.isnan(element.fromto[0]):
        element.pos = _scaled_point(spec, element.frame, s, element.pos)
    else:
        ends = np.asarray(element.fromto, dtype=float).reshape(2, 3)
        element.fromto = np.concatenate(
            [_scaled_point(spec, element.frame, s, end) for end in ends]
        )


def _scale_geom(
    spec: mujoco.MjSpec, geom, s: np.ndarray, mesh_scales: dict[str, np.ndarray]
) -> None:
    frame_rot = _frame_rot(spec, geom.frame)
    kind = geom.type
    if kind in (mujoco.mjtGeom.mjGEOM_HFIELD, mujoco.mjtGeom.mjGEOM_SDF):
        raise NotImplementedError(f"Cannot scale geom {geom.name!r} of type {kind}.")
    if not np.isnan(geom.fromto[0]):
        ends = np.asarray(geom.fromto, dtype=float).reshape(2, 3)
        geom.size[0] *= _radial_factor(s, frame_rot @ (ends[1] - ends[0]))
        _scale_placement(spec, geom, s)
        return
    _scale_placement(spec, geom, s)
    factors = _axis_factors(s, frame_rot @ _rot(_quat(spec, geom)))
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        previous = mesh_scales.setdefault(geom.meshname, factors)
        if not np.allclose(previous, factors):
            raise NotImplementedError(
                f"Mesh {geom.meshname!r} is shared by geoms needing different scales; "
                "give each its own mesh asset."
            )
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        geom.size[0] *= float(np.prod(factors) ** (1 / 3))
    elif kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        geom.size[0] *= float(np.sqrt(factors[0] * factors[1]))
        geom.size[1] *= float(factors[2])
    elif kind in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        geom.size = np.asarray(geom.size) * factors
    elif kind == mujoco.mjtGeom.mjGEOM_PLANE:
        geom.size[:2] = np.asarray(geom.size[:2]) * factors[:2]


def _radial_factor(s: np.ndarray, axis: np.ndarray) -> float:
    """Geometric mean scale of two directions perpendicular to *axis*."""
    axis = axis / np.linalg.norm(axis)
    helper = np.eye(3)[np.argmin(np.abs(axis))]
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return float(np.sqrt(np.linalg.norm(s * u) * np.linalg.norm(s * v)))


def _scale_keyframes(
    spec: mujoco.MjSpec, ref: mujoco.MjModel, slide: Mapping[str, float]
) -> None:
    for key in spec.keys:
        qpos = np.asarray(key.qpos, dtype=float)
        if qpos.size != ref.nq:
            continue
        for name, factor in slide.items():
            if name:
                qpos[ref.jnt_qposadr[ref.joint(name).id]] *= factor
        key.qpos = qpos


def _scale_inertia(
    spec: mujoco.MjSpec,
    ref: mujoco.MjModel,
    body_scales: Mapping[str, np.ndarray],
    mass: str,
    total_mass: float | None,
) -> None:
    """Scale mass and inertia of scaled bodies (and rescale all to *total_mass*)."""
    if mass not in ("volume", "keep"):
        raise ValueError(f"mass must be 'volume' or 'keep', got {mass!r}.")
    # Compiled body ids, matched by name (unnamed bodies keep their list position).
    ids = [ref.body(b.name).id if b.name else i for i, b in enumerate(spec.bodies)]
    new_mass = ref.body_mass.copy()
    for body, i in zip(spec.bodies[1:], ids[1:]):
        s = body_scales.get(body.name)
        if s is None:
            continue
        m = ref.body_mass[i]
        new_mass[i] = m * np.prod(s) if mass == "volume" else m
        inertia, iquat = _scaled_inertia(ref, i, s, new_mass[i] / m if m > 0 else 0.0)
        _set_inertial(body, new_mass[i], s * ref.body_ipos[i], inertia, iquat)
    if total_mass is None:
        return
    k = total_mass / new_mass[1:].sum()
    for body, i in zip(spec.bodies[1:], ids[1:]):
        if new_mass[i] > 0:
            if body.name not in body_scales:
                _set_inertial(
                    body,
                    ref.body_mass[i],
                    ref.body_ipos[i],
                    ref.body_inertia[i],
                    ref.body_iquat[i],
                )
            body.mass *= k
            body.inertia = _on_triangle(np.asarray(body.inertia) * k)


def _scaled_inertia(
    ref: mujoco.MjModel, i: int, s: np.ndarray, mass_ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """Principal inertia and frame of body *i* after scaling its shape by *s*."""
    if np.ptp(s) == 0:  # isotropic: same principal frame
        return ref.body_inertia[i] * mass_ratio * s[0] ** 2, ref.body_iquat[i].copy()
    rot = _rot(ref.body_iquat[i])
    moments = ref.body_inertia[i]
    second = rot @ np.diag(0.5 * moments.sum() - moments) @ rot.T  # sum m x x^T
    second = mass_ratio * np.diag(s) @ second @ np.diag(s)
    inertia, axes = np.linalg.eigh(np.trace(second) * np.eye(3) - second)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    iquat = np.zeros(4)
    mujoco.mju_mat2Quat(iquat, axes.ravel())
    return inertia, iquat


def _on_triangle(moments) -> np.ndarray:
    """Undo rounding past the triangle inequality ``I_i <= I_j + I_k``.

    A flat body sits exactly on it; one rounding step over makes MuJoCo's
    ``balanceinertia`` replace all three moments by their mean.
    """
    moments = np.array(moments, dtype=float)
    for i in range(3):
        bound = moments[(i + 1) % 3] + moments[(i + 2) % 3]
        if bound < moments[i] <= bound * (1 + 1e-9):
            moments[i] = bound
    return moments


def _set_inertial(body, mass: float, ipos, inertia, iquat) -> None:
    body.explicitinertial = True
    body.mass = float(mass)
    body.ipos = np.asarray(ipos, dtype=float)
    body.inertia = _on_triangle(inertia)
    body.iquat = np.asarray(iquat, dtype=float)
    body.ialt.type = _ORIENT.mjORIENTATION_QUAT
    body.fullinertia = np.full(6, np.nan)


def _check_equalities(
    spec: mujoco.MjSpec, body_scales: Mapping[str, np.ndarray]
) -> None:
    """Body anchors of connect/weld equalities would need scaling too."""
    for eq in spec.equalities:
        if (
            eq.type in (mujoco.mjtEq.mjEQ_CONNECT, mujoco.mjtEq.mjEQ_WELD)
            and eq.objtype == mujoco.mjtObj.mjOBJ_BODY
            and {eq.name1, eq.name2} & set(body_scales)
        ):
            raise NotImplementedError(
                f"Scaling a body of the connect/weld equality {eq.name!r} is not supported."
            )


def _scale_joint_couplings(spec: mujoco.MjSpec, slide: Mapping[str, float]) -> None:
    """``y1 = sum c_k y2^k``: with ``y1 -> a y1`` and ``y2 -> b y2``, ``c_k -> a c_k / b^k``."""
    for eq in spec.equalities:
        if eq.type == mujoco.mjtEq.mjEQ_JOINT:
            a = slide.get(eq.name1, 1.0)
            b = slide.get(eq.name2, 1.0) if eq.name2 else 1.0
            if a != 1.0 or b != 1.0:
                eq.data[:5] = [a * c / b**k for k, c in enumerate(eq.data[:5])]


def _scale_tendons(spec: mujoco.MjSpec, ratio: np.ndarray) -> None:
    """Tendon lengths, limits and couplings scale with each tendon's length change."""
    for tendon, r in zip(spec.tendons, ratio, strict=True):
        if r == 1.0:
            continue
        tendon.range = np.asarray(tendon.range) * r
        tendon.margin *= r
        springlength = np.asarray(tendon.springlength, dtype=float)
        if not np.all(springlength == -1):
            tendon.springlength = springlength * r
    index = {t.name: i for i, t in enumerate(spec.tendons) if t.name}
    for eq in spec.equalities:
        if eq.type == mujoco.mjtEq.mjEQ_TENDON:
            a = ratio[index[eq.name1]]
            b = ratio[index[eq.name2]] if eq.name2 else 1.0
            eq.data[:5] = [a * c / b**k for k, c in enumerate(eq.data[:5])]


__all__ = [
    "RAJAGOPAL_MYOFULLBODY_SEGMENTS",
    "RAJAGOPAL_MYOLEG_SEGMENTS",
    "read_opensim_scales",
    "scale_bodies",
]
