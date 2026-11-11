# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Compiled MuJoCo anatomy interchange for the optional MuSkeMo Blender add-on."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite.core.muscle_conditions import _peak_force
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

SCHEMA_VERSION = 1


def model_fingerprint(model: mujoco.MjModel) -> str:
    """Hash the complete compiled model, including embedded meshes and options.

    Args:
        model: Source model.

    Returns:
        SHA-256 of its MuJoCo-version-specific binary representation.
    """
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def _names(model: mujoco.MjModel, kind: str, count: int) -> list[str]:
    return [getattr(model, kind)(i).name or f"{kind}_{i}" for i in range(count)]


def _quat(matrix: np.ndarray) -> list[float]:
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, matrix)
    return quaternion.tolist()


def _inertia(diagonal: np.ndarray, quaternion: np.ndarray) -> list[list[float]]:
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quaternion)
    rotation = rotation.reshape(3, 3)
    return (rotation @ np.diag(diagonal) @ rotation.T).tolist()


def export_model(
    model: mujoco.MjModel,
    output: str | Path,
    *,
    qpos: np.ndarray | None = None,
) -> dict[str, Any]:
    """Export resolved anatomy, original paths and MuJoCo-evaluated poses to JSON.

    Args:
        model: Compiled source model; it is never modified.
        output: Destination JSON file. Mesh vertices are embedded.
        qpos: Optional ``(frames, nq)`` poses. Joint equalities are projected with
            the shared coupling implementation before evaluating each pose.

    Returns:
        The JSON-compatible document. Coordinates are metres and radians,
        quaternions are wxyz, and inertias are about the COM in body axes.

    Raises:
        ValueError: Invalid poses or unsupported flex/skin geometry.
    """
    if model.nflex or model.nskin:
        raise ValueError(
            "Flex and skin geometry cannot be represented by this importer"
        )
    poses = np.asarray(model.qpos0[None] if qpos is None else qpos, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != model.nq or not len(poses):
        raise ValueError(f"qpos must have nonempty shape (frames, {model.nq})")
    if not np.isfinite(poses).all():
        raise ValueError("qpos must be finite")
    names = {
        kind: _names(model, kind, count)
        for kind, count in (
            ("body", model.nbody),
            ("joint", model.njnt),
            ("site", model.nsite),
            ("geom", model.ngeom),
            ("actuator", model.nu),
            ("tendon", model.ntendon),
        )
    }
    muscles = np.flatnonzero(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE)
    if any(model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_TENDON for a in muscles):
        raise ValueError("Muscles without tendon transmission are unsupported")
    force = _peak_force(model)
    bodies = [
        dict(
            id=i,
            name=names["body"][i],
            source_name=model.body(i).name,
            parent=int(model.body_parentid[i]),
            mass=float(model.body_mass[i]),
            com=model.body_ipos[i].tolist(),
            inertia=_inertia(model.body_inertia[i], model.body_iquat[i]),
        )
        for i in range(model.nbody)
    ]
    joints = [
        dict(
            id=i,
            name=names["joint"][i],
            body=int(model.jnt_bodyid[i]),
            type=mujoco.mjtJoint(int(model.jnt_type[i])).name.removeprefix("mjJNT_"),
            pos=model.jnt_pos[i].tolist(),
            axis=model.jnt_axis[i].tolist(),
            qpos_address=int(model.jnt_qposadr[i]),
            limited=bool(model.jnt_limited[i]),
            range=model.jnt_range[i].tolist(),
        )
        for i in range(model.njnt)
    ]
    sites = [
        dict(
            id=i,
            name=names["site"][i],
            source_name=model.site(i).name,
            body=int(model.site_bodyid[i]),
            pos=model.site_pos[i].tolist(),
        )
        for i in range(model.nsite)
    ]
    tendons = []
    for t in range(model.ntendon):
        elements = []
        for w in range(model.tendon_adr[t], model.tendon_adr[t] + model.tendon_num[t]):
            elements.append(
                dict(
                    type=mujoco.mjtWrap(int(model.wrap_type[w])).name.removeprefix(
                        "mjWRAP_"
                    ),
                    object=int(model.wrap_objid[w]),
                    parameter=float(model.wrap_prm[w]),
                )
            )
        tendons.append(dict(id=t, name=names["tendon"][t], elements=elements))
    actuators = [
        dict(
            id=int(a),
            name=names["actuator"][a],
            tendon=int(model.actuator_trnid[a, 0]),
            force=float(force[a]),
            gear=model.actuator_gear[a].tolist(),
            gain=model.actuator_gainprm[a].tolist(),
            bias=model.actuator_biasprm[a].tolist(),
            dynamics=model.actuator_dynprm[a].tolist(),
            lengthrange=model.actuator_lengthrange[a].tolist(),
        )
        for a in muscles
    ]
    geoms = []
    for i in range(model.ngeom):
        geom = dict(
            id=i,
            name=names["geom"][i],
            body=int(model.geom_bodyid[i]),
            type=mujoco.mjtGeom(int(model.geom_type[i])).name.removeprefix("mjGEOM_"),
            size=model.geom_size[i].tolist(),
            rgba=model.geom_rgba[i].tolist(),
        )
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[i]
            va, vn = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            fa, fn = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            geom.update(
                vertices=model.mesh_vert[va : va + vn].tolist(),
                faces=model.mesh_face[fa : fa + fn].tolist(),
            )
        elif model.geom_type[i] == mujoco.mjtGeom.mjGEOM_HFIELD:
            raise ValueError(
                "Heightfields are unsupported; export an anatomy-only model"
            )
        geoms.append(geom)
    equalities = [
        dict(
            id=i,
            type=mujoco.mjtEq(int(model.eq_type[i])).name.removeprefix("mjEQ_"),
            active=bool(model.eq_active0[i]),
            object1=int(model.eq_obj1id[i]),
            object2=int(model.eq_obj2id[i]),
            data=model.eq_data[i].tolist(),
        )
        for i in range(model.neq)
    ]
    data, coupling = mujoco.MjData(model), JointCouplings.from_model(model)
    frames = []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_normalizeQuat(model, data.qpos)
        project_joint_couplings(coupling, data.qpos)
        mujoco.mj_forward(model, data)
        moment = np.zeros((model.nu, model.nv))
        mujoco.mju_sparse2dense(
            moment,
            data.actuator_moment,
            data.moment_rownnz,
            data.moment_rowadr,
            data.moment_colind,
        )
        frames.append(
            dict(
                qpos=data.qpos.tolist(),
                body_pos=data.xpos.tolist(),
                body_quat=data.xquat.tolist(),
                geom_pos=data.geom_xpos.tolist(),
                geom_quat=[_quat(r) for r in data.geom_xmat],
                site_pos=data.site_xpos.tolist(),
                muscle_length=data.actuator_length[muscles].tolist(),
                muscle_tendon_length=data.ten_length[
                    model.actuator_trnid[muscles, 0]
                ].tolist(),
                moment_arms=moment[muscles].tolist(),
            )
        )
    document = dict(
        schema=SCHEMA_VERSION,
        fingerprint=model_fingerprint(model),
        mujoco_version=mujoco.__version__,
        units="metres-radians-kilograms",
        bodies=bodies,
        joints=joints,
        sites=sites,
        tendons=tendons,
        muscles=actuators,
        geoms=geoms,
        equalities=equalities,
        qpos0=model.qpos0.tolist(),
        frames=frames,
    )
    Path(output).write_text(
        json.dumps(document, allow_nan=False) + "\n", encoding="utf-8"
    )
    return document


def joint_sweep(model: mujoco.MjModel, joint: str, samples: int = 21) -> np.ndarray:
    """Return poses spanning a limited scalar joint's range, for geometry checks.

    Args:
        model: Source model.
        joint: Name of a limited hinge or slide joint.
        samples: Number of evenly spaced poses, at least two.

    Returns:
        ``(samples, nq)`` poses; export projects dependent coordinates.
    """
    j = model.joint(joint)
    if j.type[0] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        raise ValueError("A sweep requires a scalar hinge or slide joint")
    if not j.limited[0] or samples < 2:
        raise ValueError("A sweep requires a limited joint and at least two samples")
    if any(e["object1"] == j.id and e["active"] for e in _joint_equalities(model)):
        raise ValueError("Sweep an independent joint, not a dependent coordinate")
    qpos = np.tile(model.qpos0, (samples, 1))
    qpos[:, j.qposadr[0]] = np.linspace(*j.range, samples)
    return qpos


def _joint_equalities(model: mujoco.MjModel) -> list[dict[str, Any]]:
    return [
        dict(object1=int(model.eq_obj1id[i]), active=bool(model.eq_active0[i]))
        for i in range(model.neq)
        if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT
    ]
