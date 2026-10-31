# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Scale a musculoskeletal model to a subject, as OpenSim's Scale Tool does.

:func:`scale_model` returns a scaled, compiled copy of a model. Its ``scales``
argument is

* a mapping of body name to one factor, or to three factors along the body's
  own axes; or
* the path of an OpenSim scale file: a ScaleSet, or a model scaled by the Scale
  Tool (e.g. by AddBiomechanics). Its segments are those of the Rajagopal model
  and are mapped onto the bodies of MyoLeg or the MuscleMimic MyoFullBody.

Each body gets per-axis scale factors in its own frame (OpenSim's convention):

* everything a body carries (geoms, meshes, sites, joint anchors, cameras,
  lights, frames, the inertial frame) is scaled by that body's factors, and a
  child body's position by its parent's;
* slide-joint translations, their ranges and their joint couplings scale with
  the parent body along the joint axis;
* every muscle's optimal fibre length and tendon slack length scale by the change
  of its muscle-tendon length in the default pose (``qpos0``). MuJoCo derives
  both from ``lengthrange``, so ``lengthrange`` is scaled by that ratio;
* mass scales with volume (``sx * sy * sz``) and peak muscle force is kept
  (OpenSim does not scale strength); ``total_mass`` then rescales the subject.

The topology (joints, actuators, sites, tendons) does not change, so the
observation and action spaces of an env stay the same.

Rotated geoms use the scale along each of their own axes. Anisotropic scaling
can introduce shear, which primitive sizes and mesh scale factors approximate.
Spheres and cylinder radii use OpenSim's arithmetic mean convention.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from typing import Any
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

# Rajagopal model segment -> first body of that segment in MyoLeg
# (``myolegs_with_torso``). Every body takes its nearest mapped ancestor's scale.
_MYOLEG_SEGMENTS: dict[str, tuple[str, ...]] = {
    "pelvis": ("root",),
    "torso": ("torso",),
    **{f"{seg}_{side}": (f"{seg}_{side}",) for seg in _LEG_SEGMENTS for side in "rl"},
}

# Rajagopal model segment -> first body of that segment in the MuscleMimic
# MyoFullBody. The lumbar spine, thorax, shoulder girdle and head follow
# ``torso``; the 27 hand bodies follow ``hand``.
_MYOFULLBODY_SEGMENTS: dict[str, tuple[str, ...]] = {
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


def scale_model(
    spec: mujoco.MjSpec,
    scales: Mapping[str, float | ArrayLike] | str | os.PathLike[str],
    *,
    total_mass: float | None = None,
) -> mujoco.MjModel:
    """Scale a copy of *spec* to a subject and compile it.

    Args:
        spec: Nominal model; it is not changed.
        scales: Body name -> one factor or ``(sx, sy, sz)`` along that body's
            axes; or the path of an OpenSim ScaleSet or scaled ``.osim`` model,
            for MyoLeg or the MuscleMimic MyoFullBody.
        total_mass: Rescale the subject's body masses to this total (kg). The
            subject is every kinematic tree holding a body named in *scales*
            (a factor of one counts), or else the model's one articulated tree;
            static scenery and other agents keep their mass.

    Returns:
        The compiled scaled model. With no factor other than one and no
        ``total_mass`` it equals ``spec.compile()``.

    Raises:
        KeyError: If a body is not in the model.
        ValueError: For factors or a ``total_mass`` that are not finite and
            positive, an OpenSim file for a model without a segment mapping, or
            a ``total_mass`` whose subject is ambiguous.
        NotImplementedError: For a mesh shared by geoms needing different
            scales, or a connect/weld equality or height field on a scaled body.
    """
    spec = spec.copy()
    _scale_spec(spec, scales, total_mass=total_mass)
    return spec.compile()


def _scale_spec(
    spec: mujoco.MjSpec,
    scales: Mapping[str, float | ArrayLike] | str | os.PathLike[str],
    *,
    total_mass: float | None = None,
) -> None:
    """Scale *spec* in place (see :func:`scale_model`)."""
    if total_mass is not None and (not np.isfinite(total_mass) or total_mass <= 0):
        raise ValueError(f"total_mass must be finite and positive, got {total_mass}.")
    if isinstance(scales, (str, os.PathLike)):
        segments = _segments_for(spec)
        body_scales = _body_scales(spec, _read_opensim_scales(scales), segments)
        # The segment bodies define the subject, even those left unscaled.
        subject = set().union(*segments.values())
    else:
        body_scales = _body_scales(spec, scales, None)
        subject = set(scales)
    _scale_bodies(spec, body_scales, subject, total_mass)


def _segments_for(spec: mujoco.MjSpec) -> Mapping[str, tuple[str, ...]]:
    """The Rajagopal segment mapping whose first bodies are all in *spec*.

    MyoFullBody also has bodies named like MyoLeg's (``root``, ``torso``), so
    the mapping with the most segments wins.
    """
    names = {b.name for b in spec.bodies[1:]}
    tables = [
        table
        for table in (_MYOFULLBODY_SEGMENTS, _MYOLEG_SEGMENTS)
        if all(body in names for bodies in table.values() for body in bodies)
    ]
    if not tables:
        raise ValueError(
            "No OpenSim segment mapping fits this model (MyoLeg and MyoFullBody "
            "have one); pass per-body factors instead."
        )
    return max(tables, key=len)


def _read_opensim_scales(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Segment name -> ``(sx, sy, sz)`` from a ScaleSet or a scaled ``.osim`` model.

    A scaled model carries the applied factors as its geometry's
    ``scale_factors`` (one for the unscaled Rajagopal model).
    """
    root = ET.parse(path).getroot()
    return _scale_set(root) or _model_scales(root)


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


def _scale_bodies(
    spec: mujoco.MjSpec,
    body_scales: Mapping[str, np.ndarray],
    subject: set[str],
    total_mass: float | None,
) -> None:
    """Scale *spec* in place by resolved per-body factors.

    Args:
        spec: Model to edit.
        body_scales: Body name -> factors other than one, in that body's frame.
        subject: Body names whose kinematic trees ``total_mass`` rescales.
        total_mass: Target mass of the subject (kg), or None.
    """
    if not body_scales and total_mass is None:
        return

    # Compile copies only: compiling *spec* itself would freeze its frames.
    ref = spec.copy().compile()
    ref_data = mujoco.MjData(ref)
    mujoco.mj_forward(ref, ref_data)
    ref_actuator_length = ref_data.actuator_length.copy()
    ref_tendon_length = ref_data.ten_length.copy()
    peak_force = _peak_force(ref)

    # Refuse before the first edit, so a refused scaling leaves *spec* intact.
    _check_equalities(spec, body_scales)
    mesh_scales = _mesh_scales(spec, body_scales)
    # Compiled body ids, matched by name (unnamed bodies keep their list position).
    ids = [ref.body(b.name).id if b.name else i for i, b in enumerate(spec.bodies)]
    rescaled = None
    if total_mass is not None:
        rescaled = _subject_bodies(spec, ref, ids, subject)
        if not any(ref.body_mass[i] > 0 for _, i in rescaled):
            raise ValueError("total_mass needs a subject body with mass.")

    slide = _scale_geometry(spec, ref, body_scales, mesh_scales)
    _scale_inertia(spec, ref, ids, body_scales, total_mass, rescaled)
    _scale_joint_couplings(spec, slide)

    scaled = spec.copy().compile()
    data = mujoco.MjData(scaled)
    mujoco.mj_forward(scaled, data)
    tendon_ratio = _ratio(data.ten_length, ref_tendon_length)
    _scale_tendons(spec, tendon_ratio)
    muscle_ratio = _ratio(data.actuator_length, ref_actuator_length)
    for i, actuator in enumerate(spec.actuators):
        if int(ref.actuator_gaintype[i]) != int(mujoco.mjtGain.mjGAIN_MUSCLE):
            continue
        actuator.lengthrange = ref.actuator_lengthrange[i] * muscle_ratio[i]
        # MuJoCo derives an automatic peak force from the (now scaled) mass.
        if ref.actuator_gainprm[i, 2] < 0:
            actuator.gainprm[2] = peak_force[i]
            if int(ref.actuator_biastype[i]) == int(mujoco.mjtBias.mjBIAS_MUSCLE):
                actuator.biasprm[2] = peak_force[i]


def _body_scales(
    spec: mujoco.MjSpec,
    scales: Mapping[str, float | ArrayLike],
    segments: Mapping[str, Sequence[str]] | None,
) -> dict[str, np.ndarray]:
    """Body name -> per-axis factors, for bodies with a factor other than one.

    With *segments* (segment -> first bodies), each body takes the factors of
    its nearest ancestor (or itself) that starts a segment; a segment missing
    from *scales* stays unscaled, and segments the mapping lacks are ignored.
    """
    names = {b.name for b in spec.bodies[1:]}
    # Validate even unused segments, so malformed estimates cannot pass silently.
    scales = {name: _factors(scale) for name, scale in scales.items()}
    if segments is None:
        per_body = {name: _factors(s) for name, s in scales.items()}
    else:
        # A segment missing from *scales* stays unscaled, as in OpenSim.
        starts = {}
        for segment, first_bodies in segments.items():
            for start_body in first_bodies:
                if start_body in starts:
                    raise ValueError(
                        f"Body {start_body!r} starts more than one segment."
                    )
                starts[start_body] = _factors(scales.get(segment, 1.0))
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
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0):
        raise ValueError(f"Scale factors must be finite and positive, got {factors}.")
    return factors


def _ratio(new: np.ndarray, old: np.ndarray) -> np.ndarray:
    ratio = np.ones_like(old)
    nonzero = np.abs(old) > mujoco.mjMINVAL
    ratio[nonzero] = new[nonzero] / old[nonzero]
    return ratio


def _quat(spec: mujoco.MjSpec, element: Any) -> np.ndarray:
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


def _frame_pose(spec: mujoco.MjSpec, frame: Any) -> tuple[np.ndarray, np.ndarray]:
    """Origin and rotation of a (possibly nested) frame in its body's coordinates."""
    if frame is None:
        return np.zeros(3), np.eye(3)
    origin, rot = _frame_pose(spec, frame.frame)
    return origin + rot @ np.asarray(frame.pos, dtype=float), rot @ _rot(
        _quat(spec, frame)
    )


def _frame_rot(spec: mujoco.MjSpec, frame: Any) -> np.ndarray:
    """Rotation of a (possibly nested) frame in its body's coordinates."""
    return _frame_pose(spec, frame)[1]


def _scaled_point(
    spec: mujoco.MjSpec, frame: Any, s: np.ndarray, point: ArrayLike
) -> np.ndarray:
    """A point given in *frame* coordinates, moved as its body scales by ``diag(s)``.

    Frames themselves are left alone: MuJoCo ignores edits of a frame's position
    once the spec has been compiled, so the scaling goes into the point instead.
    """
    origin, rot = _frame_pose(spec, frame)
    return rot.T @ (s * (origin + rot @ np.asarray(point, dtype=float)) - origin)


def _axis_factors(s: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Scale along each axis of a frame rotated by *rot* in body coordinates."""
    return np.linalg.norm(np.diag(s) @ rot, axis=0)


def _mesh_scales(
    spec: mujoco.MjSpec, body_scales: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Mesh -> scale along its own axes; refuses geometry that cannot be scaled.

    Edits nothing, so it runs before the first edit of a scaling.
    """
    mesh_scales: dict[str, np.ndarray] = {}
    for geom in spec.geoms:
        s = body_scales.get(geom.parent.name)
        if s is None:
            continue
        if geom.type in (mujoco.mjtGeom.mjGEOM_HFIELD, mujoco.mjtGeom.mjGEOM_SDF):
            raise NotImplementedError(
                f"Cannot scale geom {geom.name!r} of type {geom.type}."
            )
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            rot = _frame_rot(spec, geom.frame) @ _rot(_quat(spec, geom))
            factors = _axis_factors(s, rot)
            previous = mesh_scales.setdefault(geom.meshname, factors)
            if not np.allclose(previous, factors):
                raise NotImplementedError(
                    f"Mesh {geom.meshname!r} is shared by geoms needing different "
                    "scales; give each its own mesh asset."
                )
    shared = {
        g.meshname
        for g in spec.geoms
        if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.parent.name not in body_scales
    } & set(mesh_scales)
    if shared:
        raise NotImplementedError(
            f"Meshes {sorted(shared)} are also used by unscaled bodies; give each its own mesh asset."
        )
    for name in mesh_scales:
        if spec.mesh(name).plugin.active:
            raise NotImplementedError(f"Cannot scale plugin-generated mesh {name!r}.")
    return mesh_scales


def _scale_geometry(
    spec: mujoco.MjSpec,
    ref: mujoco.MjModel,
    body_scales: Mapping[str, np.ndarray],
    mesh_scales: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Scale positions and sizes; return slide joint -> translation factor."""
    slide = {}
    slide_qpos: dict[int, float] = {}
    for body_index, body in enumerate(spec.bodies[1:], start=1):
        parent_s = body_scales.get(body.parent.name, _UNIT) if body.parent else _UNIT
        body_rot = _frame_rot(spec, body.frame) @ _rot(_quat(spec, body))
        body_id = ref.body(body.name).id if body.name else body_index
        joint_ids = np.flatnonzero(ref.jnt_bodyid == body_id)
        for joint, joint_id in zip(body.joints, joint_ids, strict=True):
            if joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
                axis = (
                    body_rot
                    @ _frame_rot(spec, joint.frame)
                    @ np.asarray(joint.axis, dtype=float)
                )
                factor = float(np.linalg.norm(parent_s * axis) / np.linalg.norm(axis))
                if factor != 1.0:
                    if joint.name:
                        slide[joint.name] = factor
                    slide_qpos[int(ref.jnt_qposadr[joint_id])] = factor
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
            _scale_geom(spec, geom, s)
    for name, factors in mesh_scales.items():
        _rescale_mesh(spec, name, factors)
    _scale_keyframes(spec, ref, slide_qpos)
    return slide


def _rescale_mesh(spec: mujoco.MjSpec, name: str, factors: np.ndarray) -> None:
    """Recreate an edited mesh so precompiled bounds cannot survive its new scale."""
    mesh = spec.mesh(name)
    # MuJoCo 3.11--3.15 retains private mesh AABB extrema after recompilation.
    # Copy authored fields, not XML or compiled vertices/principal-axis frames.
    attributes: dict[str, Any] = {}
    arrays = {
        "refpos",
        "refquat",
        "scale",
        "uservert",
        "usernormal",
        "usertexcoord",
        "userface",
        "userfacenormal",
        "userfacetexcoord",
        "usernode",
        "usertet",
    }
    for field in (
        "name",
        "content_type",
        "file",
        "refpos",
        "refquat",
        "scale",
        "inertia",
        "smoothnormal",
        "needsdf",
        "maxhullvert",
        "uservert",
        "usernormal",
        "usertexcoord",
        "userface",
        "userfacenormal",
        "userfacetexcoord",
        "usernode",
        "usertet",
        "material",
        "octree_maxdepth",
        "info",
    ):
        if hasattr(mesh, field):
            value = getattr(mesh, field)
            attributes[field] = np.array(value, copy=True) if field in arrays else value
    attributes["scale"] *= factors
    classname = mesh.classname
    spec.delete(mesh)
    replacement = spec.add_mesh(**attributes)
    replacement.classname = classname


def _scale_placement(spec: mujoco.MjSpec, element: Any, s: np.ndarray) -> None:
    """Scale the position, or the ``fromto`` end points, of a geom or site."""
    if np.isnan(element.fromto[0]):
        element.pos = _scaled_point(spec, element.frame, s, element.pos)
    else:
        ends = np.asarray(element.fromto, dtype=float).reshape(2, 3)
        element.fromto = np.concatenate(
            [_scaled_point(spec, element.frame, s, end) for end in ends]
        )


def _scale_geom(spec: mujoco.MjSpec, geom: Any, s: np.ndarray) -> None:
    """Scale a primitive's size and place; meshes are rescaled by :func:`_rescale_mesh`."""
    frame_rot = _frame_rot(spec, geom.frame)
    kind = geom.type
    if not np.isnan(geom.fromto[0]):
        ends = np.asarray(geom.fromto, dtype=float).reshape(2, 3)
        geom.size[0] *= _radial_factor(s, ends[0] - ends[1], frame_rot)
        _scale_placement(spec, geom, s)
        return
    _scale_placement(spec, geom, s)
    factors = _axis_factors(s, frame_rot @ _rot(_quat(spec, geom)))
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        geom.size[0] *= float(np.mean(s))
    elif kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        geom.size[0] *= float(np.mean(factors[:2]))
        geom.size[1] *= float(factors[2])
    elif kind in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        geom.size = np.asarray(geom.size) * factors
    elif kind == mujoco.mjtGeom.mjGEOM_PLANE:
        geom.size[:2] = np.asarray(geom.size[:2]) * factors[:2]


def _radial_factor(s: np.ndarray, axis: np.ndarray, frame_rot: np.ndarray) -> float:
    """Mean transverse scale using MuJoCo's orientation for a ``fromto`` geom."""
    quat = np.zeros(4)
    mujoco.mju_quatZ2Vec(quat, axis)
    return float(np.mean(_axis_factors(s, frame_rot @ _rot(quat))[:2]))


def _scale_keyframes(
    spec: mujoco.MjSpec, ref: mujoco.MjModel, slide: Mapping[int, float]
) -> None:
    for index, key in enumerate(spec.keys):
        qpos = ref.key_qpos[index].copy()
        for address, factor in slide.items():
            qpos[address] *= factor
        key.qpos = qpos


def _scale_inertia(
    spec: mujoco.MjSpec,
    ref: mujoco.MjModel,
    ids: Sequence[int],
    body_scales: Mapping[str, np.ndarray],
    total_mass: float | None,
    subject_bodies: Sequence[tuple[mujoco.MjsBody, int]] | None,
) -> None:
    """Scale mass (by volume) and inertia of scaled bodies; rescale the subject.

    Args:
        spec: Model to edit.
        ref: The compiled model before scaling.
        ids: Compiled id of each body of *spec*.
        body_scales: Body name -> factors other than one.
        total_mass: Target mass of *subject_bodies* (kg), or None.
        subject_bodies: Bodies (with ids) that *total_mass* rescales.
    """
    if spec.compiler.settotalmass > 0:
        # Preserve the compiled distribution before disabling renormalization.
        for body, i in zip(spec.bodies[1:], ids[1:]):
            _set_inertial(
                body,
                ref.body_mass[i],
                ref.body_ipos[i],
                ref.body_inertia[i],
                ref.body_iquat[i],
            )
        spec.compiler.settotalmass = -1
    new_mass = ref.body_mass.copy()
    for body, i in zip(spec.bodies[1:], ids[1:]):
        s = body_scales.get(body.name)
        if s is None:
            continue
        m = ref.body_mass[i]
        new_mass[i] = m * np.prod(s)
        inertia, iquat = _scaled_inertia(ref, i, s, new_mass[i] / m if m > 0 else 0.0)
        _set_inertial(body, new_mass[i], s * ref.body_ipos[i], inertia, iquat)
    if total_mass is None or subject_bodies is None:
        return
    k = total_mass / sum(new_mass[i] for _, i in subject_bodies)
    for body, i in subject_bodies:
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


def _subject_bodies(
    spec: mujoco.MjSpec,
    ref: mujoco.MjModel,
    ids: Sequence[int],
    subject: set[str],
) -> list[tuple[mujoco.MjsBody, int]]:
    """Bodies whose mass *total_mass* sets, with their compiled ids.

    The subject is every kinematic tree (child of the world body) holding a
    body named in *subject*, or, if none is named, the one tree with a joint.
    Static scenery and other agents in a composed scene keep their mass.
    """
    roots = ref.body_rootid
    pairs = list(zip(spec.bodies[1:], ids[1:]))
    trees = {roots[i] for body, i in pairs if body.name in subject}
    if not trees:
        trees = {roots[i] for i in range(1, ref.nbody) if ref.body_dofnum[i] > 0}
        if len(trees) > 1:
            names = sorted(ref.body(int(r)).name for r in trees)
            raise ValueError(
                f"total_mass is ambiguous: several articulated trees {names}; "
                "name a body of the subject in scales (a factor of 1 is fine)."
            )
    return [(body, i) for body, i in pairs if roots[i] in trees]


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


def _on_triangle(moments: ArrayLike) -> np.ndarray:
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


def _set_inertial(
    body: mujoco.MjsBody,
    mass: float,
    ipos: ArrayLike,
    inertia: ArrayLike,
    iquat: ArrayLike,
) -> None:
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


def _model_digest(model: mujoco.MjModel) -> str:
    """Hex SHA-256 of the MuJoCo version and the model's binary (``mj_saveModel``)."""
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return hashlib.sha256(mujoco.__version__.encode() + buffer.tobytes()).hexdigest()


__all__ = ["scale_model"]
