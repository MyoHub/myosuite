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


def _myofullbody_spec(self_collisions: bool = True) -> mujoco.MjSpec:
    from ml_collections import config_dict  # noqa: PLC0415

    from myosuite.integrations.musclemimic.fullbody_model import (  # noqa: PLC0415
        FULLBODY_BODY2SITES_FOR_MIMIC,
        build_native_mimic_fullbody_spec,
    )

    spec = build_native_mimic_fullbody_spec(
        config_dict.create(disable_fingers=True, model_name="myofullbody")
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


MODELS: dict[str, MimicModelCfg] = {
    "myofullbody": MimicModelCfg(
        name="myofullbody",
        build_spec=_myofullbody_spec,
        tracked_bodies=(
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
        ),  # fmt: skip
        foot_bodies=("calcn_r", "toes_r", "calcn_l", "toes_l"),
        muscle_tendon_expr=(r".*_tendon(_l|_left)?",),
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
    return spec


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
    from myosuite.biomechanics.retarget_smplh import (  # noqa: PLC0415
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
        coupled = {
            int(m.eq_obj1id[e])
            for e in range(m.neq)
            if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and m.eq_active0[e]
        }
        self.couplings = JointCouplings.from_model(m)
        self.joint_ids = [
            j for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
        ]
        self.joint_names = [m.joint(j).name for j in self.joint_ids]
        self.joint_qadr = m.jnt_qposadr[self.joint_ids].copy()
        self.joint_dadr = m.jnt_dofadr[self.joint_ids].copy()
        self.indep_ids = [j for j in self.joint_ids if j not in coupled]
        self.indep_names = [m.joint(j).name for j in self.indep_ids]
        self.indep_cols = [self.joint_ids.index(j) for j in self.indep_ids]
        self.indep_qadr = m.jnt_qposadr[self.indep_ids].copy()
        self.indep_dadr = m.jnt_dofadr[self.indep_ids].copy()
        if not m.jnt_limited[self.indep_ids].all():
            raise ValueError("every independent joint needs a range")
        self.q_lo, self.q_hi = m.jnt_range[self.indep_ids].T.copy()
        if not (m.actuator_trntype == mujoco.mjtTrn.mjTRN_TENDON).all():
            raise ValueError("every actuator must be a tendon-driven muscle")
        self.num_muscles = m.nu
        self.act_adr = m.actuator_actadr.copy()


def root_offset(m: mujoco.MjModel) -> np.ndarray:
    """``R_off`` with ``R_root @ R_off = I`` in the default pose (canonical clip frame)."""
    adr = m.jnt_qposadr[m.body_jntadr[ModelInfo(m).root_id]]
    R_rest = np.zeros(9)
    mujoco.mju_quat2Mat(R_rest, m.qpos0[adr + 3 : adr + 7])
    return R_rest.reshape(3, 3).T.copy()
