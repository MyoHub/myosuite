# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# this package for the notices of the code it includes.
"""Body models the tracking task runs on (the model layer), and their bookkeeping.

A model adapter (:class:`MimicModelCfg`) names the bodies the task needs (17 tracked
bodies, pelvis first, and the 4 foot bodies of the touch observation) and builds the
body spec. The scene adds a ground plane and the foot contact sensors, as the mjlab
twin's terrain and contact sensor do. ``myofullbody`` (myo_sim, fingers disabled) is
the only model so far; register others in :data:`MODELS`.

:class:`ModelInfo` is ported from muscle-supermimic (commit 8a6244b,
``src/muscle_supermimic/assets/robots/model_info.py``).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from myosuite.physics.joint_coupling import JointCouplings

GROUND_BODY = "terrain"
"""Name of the ground body and geom (mjlab's plane terrain)."""
TOUCH_SENSOR = "foot_touch"
"""Prefix of the foot contact sensors (mjlab's ``ContactSensorCfg`` name)."""
_CONTACT_FORCE_BIT = 1  # mjlab _CONTACT_DATA_MAP["force"]
_CONTACT_NETFORCE = 3  # mjlab _CONTACT_REDUCE_MAP["netforce"]


@dataclass(frozen=True)
class MimicModelCfg:
    """A body the task can run on.

    Attributes:
        name: Model name (``EnvConfig.model``).
        build_spec: ``build_spec(self_collisions) -> MjSpec`` of the body alone.
        tracked_bodies: The 17 tracked bodies, the one welded to the root first.
        foot_bodies: Heel and toe bodies of the touch observation.
        muscle_tendon_expr: Regexes of the muscle tendons (mjlab's actuator targets).
    """

    name: str
    build_spec: Callable[[bool], mujoco.MjSpec]
    tracked_bodies: tuple[str, ...]
    foot_bodies: tuple[str, ...]
    muscle_tendon_expr: tuple[str, ...]


ROOT_BODY = "Full Body"
INIT_POS = (0.0, 0.0, 1.0)
INIT_ROT = (0.7071068, 0.0, 0.0, 0.7071068)
"""+90 deg about z: myofullbody's default pose faces +x."""


def _build_mimic_body_spec(
    model_name: str = "myofullbody",
    self_collisions: bool = True,
    build_kwargs: dict[str, object] | None = None,
) -> mujoco.MjSpec:
    from ml_collections import config_dict  # noqa: PLC0415

    from myosuite.integrations.musclemimic.fullbody_model import (  # noqa: PLC0415
        FULLBODY_BODY2SITES_FOR_MIMIC,
        build_native_mimic_fullbody_spec,
    )

    spec = build_native_mimic_fullbody_spec(
        config_dict.create(
            disable_fingers=True,
            model_name=model_name,
            build_kwargs=dict(build_kwargs or {}),
        )
    )
    spec.njmax = spec.nconmax = spec.nuser_jnt = -1
    spec.option.timestep = mujoco.MjSpec().option.timestep
    for key in list(spec.keys):
        spec.delete(key)
    for geom in list(spec.worldbody.geoms):
        spec.delete(geom)
    for light in list(spec.worldbody.lights):
        spec.delete(light)
    root = spec.body(ROOT_BODY)
    root.pos, root.quat = INIT_POS, INIT_ROT
    for site in list(spec.sites):
        if site.name in FULLBODY_BODY2SITES_FOR_MIMIC.values():
            spec.delete(site)
    for sensor in list(spec.sensors):
        if sensor.type != mujoco.mjtSensor.mjSENS_TOUCH:
            spec.delete(sensor)
    if not self_collisions:
        for pair in list(spec.pairs):
            spec.delete(pair)
    return spec


def _myofullbody_spec(self_collisions: bool = True) -> mujoco.MjSpec:
    return _build_mimic_body_spec("myofullbody", self_collisions=self_collisions)


_TRACKED_FULLBODY: tuple[str, ...] = (
    "pelvis",
    "lumbar1",
    "head",
    "humerus_l",
    "ulna_l",
    "lunate_l",
    "humerus_r",
    "ulna_r",
    "lunate_r",
    "femur_l",
    "tibia_l",
    "talus_l",
    "toes_l",
    "femur_r",
    "tibia_r",
    "talus_r",
    "toes_r",
)
_FOOT_BODIES: tuple[str, ...] = ("calcn_r", "toes_r", "calcn_l", "toes_l")
_TENDON_EXPR: tuple[str, ...] = (r".*_tendon(_l|_left)?",)

MODELS: dict[str, MimicModelCfg] = {
    "myofullbody": MimicModelCfg(
        name="myofullbody",
        build_spec=_myofullbody_spec,
        tracked_bodies=_TRACKED_FULLBODY,
        foot_bodies=_FOOT_BODIES,
        muscle_tendon_expr=_TENDON_EXPR,
    ),
    "myofullbody_rigid": MimicModelCfg(
        name="myofullbody_rigid",
        build_spec=functools.partial(_build_mimic_body_spec, "myofullbody_rigid"),
        tracked_bodies=_TRACKED_FULLBODY,
        foot_bodies=_FOOT_BODIES,
        muscle_tendon_expr=_TENDON_EXPR,
    ),
    "myofullbody_simple": MimicModelCfg(
        name="myofullbody_simple",
        build_spec=functools.partial(_build_mimic_body_spec, "myofullbody_simple"),
        tracked_bodies=_TRACKED_FULLBODY,
        foot_bodies=_FOOT_BODIES,
        muscle_tendon_expr=_TENDON_EXPR,
    ),
    "myofullbody_simple_locked_st": MimicModelCfg(
        name="myofullbody_simple_locked_st",
        build_spec=functools.partial(
            _build_mimic_body_spec,
            "myofullbody_simple",
            build_kwargs={"keep_scapula_dofs": False},
        ),
        tracked_bodies=_TRACKED_FULLBODY,
        foot_bodies=_FOOT_BODIES,
        muscle_tendon_expr=_TENDON_EXPR,
    ),
    "myofullbody_simple_rigid": MimicModelCfg(
        name="myofullbody_simple_rigid",
        build_spec=functools.partial(_build_mimic_body_spec, "myofullbody_simple_rigid"),
        tracked_bodies=_TRACKED_FULLBODY,
        foot_bodies=_FOOT_BODIES,
        muscle_tendon_expr=_TENDON_EXPR,
    ),
}
DEFAULT_MODEL = "myofullbody"


def model_cfg(name: str) -> MimicModelCfg:
    """The adapter of model *name*.

    Raises:
        KeyError: For a model the task does not support.
    """
    if name not in MODELS:
        raise KeyError(f"myoMimic has no model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name]


def scene_spec(model: MimicModelCfg, self_collisions: bool = True) -> mujoco.MjSpec:
    """The body on a ground plane with the foot contact sensors (the CPU scene).

    Matches the mjlab twin: a plane terrain body ``terrain`` and one contact sensor
    per foot body reading the net force of the ground on it (world frame).
    """
    spec = model.build_spec(self_collisions)
    spec.worldbody.add_body(name=GROUND_BODY).add_geom(
        name=GROUND_BODY, type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.01)
    )
    for body in model.foot_bodies:
        spec.add_sensor(
            name=f"{TOUCH_SENSOR}_{body}_force",
            type=mujoco.mjtSensor.mjSENS_CONTACT,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            objname=body,
            reftype=mujoco.mjtObj.mjOBJ_BODY,
            refname=GROUND_BODY,
            intprm=[1 << _CONTACT_FORCE_BIT, _CONTACT_NETFORCE, 1],
        )
    from myosuite.physics.compliant_plant import prepare_spec_for_plant  # noqa: PLC0415

    return prepare_spec_for_plant(spec)


def model_version(name: str) -> str:
    """Version of the package model *name* comes from (myo_sim for every model so far)."""
    model_cfg(name)
    import myo_sim  # noqa: PLC0415

    return str(myo_sim.__version__)


@functools.lru_cache(maxsize=4)
def kinematic_fingerprint(name: str) -> str:
    """Kinematic fingerprint of model *name*'s body: what its clips are valid for.

    :func:`~myosuite.biomechanics.retarget.kinematic_fingerprint` of
    :func:`compiled_body` (joints, body tree, coupling equalities, foot collision
    geoms), with the landmark bodies and feet of the model's SMPL+H map. Clips whose
    metadata carries it load whatever myo_sim version they were fitted under.
    """
    from myosuite.biomechanics.retarget import (  # noqa: PLC0415
        kinematic_fingerprint as fingerprint,
    )
    from myosuite.integrations.retarget_smplh import (  # noqa: PLC0415
        SMPLH_MAPS,
        landmark_bodies,
    )

    smap = SMPLH_MAPS[model_cfg(name).name]
    return fingerprint(compiled_body(name), landmark_bodies(smap), smap.feet)


@functools.lru_cache(maxsize=4)
def compiled_scene(name: str, self_collisions: bool = True) -> mujoco.MjModel:
    """Compiled :func:`scene_spec` (cached; copy before editing)."""
    return scene_spec(model_cfg(name), self_collisions).compile()


@functools.lru_cache(maxsize=4)
def compiled_body(name: str) -> mujoco.MjModel:
    """The body compiled on its own (no scene), for bookkeeping."""
    return model_cfg(name).build_spec(True).compile()


class ModelInfo:
    """Joint and muscle bookkeeping of a compiled body.

    * root: the body carrying the model's free joint;
    * independent joints: every non-root joint that is not the driven side of a joint
      ``<equality>``, in model order (the joint coordinates of the clips);
    * coupled joints: set from an independent joint by the equality's polynomial;
    * muscles: in actuator order, each driving one tendon.
    """

    def __init__(self, m: mujoco.MjModel):
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1:
            raise ValueError(f"expected one free joint (the root), found {len(free)}")
        self.root_joint_name = m.joint(free[0]).name
        self.root_id = int(m.jnt_bodyid[free[0]])
        self.root_qadr = int(m.jnt_qposadr[free[0]])
        self.root_dadr = int(m.jnt_dofadr[free[0]])
        self.m = m
        self.root_name = m.body(self.root_id).name
        coupled = {}
        for e in range(m.neq):
            if m.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT or not m.eq_active0[e]:
                continue
            j2, c = int(m.eq_obj2id[e]), m.eq_data[e, :5].copy()
            if (
                j2 < 0
            ):  # held at qpos0 + polycoef[0]; MuJoCo ignores the other coefficients
                j2, c[1:] = None, 0.0
            coupled[int(m.eq_obj1id[e])] = (j2, c)
        self.coupled = coupled
        """Coupled joint id -> (its independent joint id or ``None``, polycoef[5])."""
        self.couplings = JointCouplings.from_model(m)
        self.joint_ids = [
            j for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
        ]
        self.joint_names = [m.joint(j).name for j in self.joint_ids]

        def _qw(j: int) -> int:
            return 4 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1

        def _dw(j: int) -> int:
            return 3 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1

        self.joint_qadr = (
            np.concatenate(
                [
                    np.arange(m.jnt_qposadr[j], m.jnt_qposadr[j] + _qw(j), dtype=np.intp)
                    for j in self.joint_ids
                ]
            )
            if self.joint_ids
            else np.empty(0, dtype=np.intp)
        )
        self.joint_dadr = (
            np.concatenate(
                [
                    np.arange(m.jnt_dofadr[j], m.jnt_dofadr[j] + _dw(j), dtype=np.intp)
                    for j in self.joint_ids
                ]
            )
            if self.joint_ids
            else np.empty(0, dtype=np.intp)
        )
        self.indep_ids = [j for j in self.joint_ids if j not in coupled]
        self.indep_names = [m.joint(j).name for j in self.indep_ids]
        self.indep_cols = [self.joint_ids.index(j) for j in self.indep_ids]
        self.indep_qadr = (
            np.concatenate(
                [
                    np.arange(m.jnt_qposadr[j], m.jnt_qposadr[j] + _qw(j), dtype=np.intp)
                    for j in self.indep_ids
                ]
            )
            if self.indep_ids
            else np.empty(0, dtype=np.intp)
        )
        self.indep_dadr = (
            np.concatenate(
                [
                    np.arange(m.jnt_dofadr[j], m.jnt_dofadr[j] + _dw(j), dtype=np.intp)
                    for j in self.indep_ids
                ]
            )
            if self.indep_ids
            else np.empty(0, dtype=np.intp)
        )
        scalar_indep = [
            j for j in self.indep_ids if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_BALL
        ]
        if scalar_indep and not m.jnt_limited[scalar_indep].all():
            raise ValueError("every scalar independent joint needs a range")
        lo_parts, hi_parts = [], []
        for j in self.indep_ids:
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL:
                lo_parts.append(np.full(4, -1.0))
                hi_parts.append(np.full(4, 1.0))
            else:
                lo_parts.append(np.array([m.jnt_range[j, 0]]))
                hi_parts.append(np.array([m.jnt_range[j, 1]]))
        self.q_lo = np.concatenate(lo_parts) if lo_parts else np.empty(0)
        self.q_hi = np.concatenate(hi_parts) if hi_parts else np.empty(0)
        allowed_trn = (
            int(mujoco.mjtTrn.mjTRN_TENDON),
            int(mujoco.mjtTrn.mjTRN_JOINT),
        )
        if not np.isin(m.actuator_trntype, allowed_trn).all():
            raise ValueError("every actuator must drive a tendon or joint")
        self.num_muscles = m.nu
        self.muscle_names = [m.actuator(a).name for a in range(m.nu)]
        self.muscle_tendon_names = [
            m.tendon(int(tid)).name
            if int(trn) == int(mujoco.mjtTrn.mjTRN_TENDON)
            else m.joint(int(tid)).name
            for trn, tid in zip(m.actuator_trntype, m.actuator_trnid[:, 0])
        ]
        self.qpos0 = m.qpos0.copy()
        self.act_adr = m.actuator_actadr.copy()


def root_offset(m: mujoco.MjModel) -> np.ndarray:
    """``R_off`` with ``R_root @ R_off = I`` in the default pose (canonical clip frame)."""
    adr = m.jnt_qposadr[m.body_jntadr[ModelInfo(m).root_id]]
    R_rest = np.zeros(9)
    mujoco.mju_quat2Mat(R_rest, m.qpos0[adr + 3 : adr + 7])
    return R_rest.reshape(3, 3).T.copy()
