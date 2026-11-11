# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# this package for the notices of the code it includes.
"""Body models the tracking task runs on (the model layer), and their bookkeeping.

A model adapter (:class:`MimicModelCfg`) names the bodies the task needs (the tracked
bodies, the first one the anchor of the relative sites, and the contact bodies of the
touch observation) and builds the body spec. The root is free (one free joint, e.g. the
full body) or fixed (no free joint, e.g. an arm on a fixed torso); actuators are muscles
or, by name, motors. With contact bodies the scene adds a ground plane and their contact
sensors, as the mjlab twin's terrain and contact sensor do. ``myofullbody`` (myo_sim,
fingers disabled) is the built-in model; :func:`register_model` adds others.

:class:`ModelInfo` is ported from muscle-supermimic (commit 8a6244b,
``src/muscle_supermimic/assets/robots/model_info.py``).
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from myosuite.physics.joint_coupling import JointCouplings

GROUND_BODY = "terrain"
"""Name of the ground body and geom (mjlab's plane terrain)."""
TOUCH_SENSOR = "foot_touch"
"""Prefix of the foot contact sensors (mjlab's ``ContactSensorCfg`` name)."""
_CONTACT_FORCE_BIT = 1  # mjlab _CONTACT_DATA_MAP["force"]
_CONTACT_NETFORCE = 3  # mjlab _CONTACT_REDUCE_MAP["netforce"]
MIRROR_SIDES = (("_r", "_l"), ("_R", "_L"), ("_right", "_left"))
"""Left / right name suffixes of the bodies, joints and muscles of a symmetric body."""


def _myo_sim_version() -> str:
    import myo_sim  # noqa: PLC0415

    return str(myo_sim.__version__)


@dataclass(frozen=True)
class MimicModelCfg:
    """A body the task can run on.

    The root kind follows from the body: one free joint (a free root) or none (a fixed
    root, welded to the world).

    Attributes:
        name: Model name (``EnvConfig.model``).
        build_spec: ``build_spec(self_collisions) -> MjSpec`` of the body alone.
        tracked_bodies: The tracked bodies (any number, at least two); the first is the
            anchor of the relative sites (with a free root it must be welded to the root).
        foot_bodies: Contact bodies of the touch observation (heels and toes); empty:
            no touch observation and no ground.
        muscle_tendon_expr: Regexes of the muscle tendons (mjlab's actuator targets).
        motor_expr: Regexes of the actuator names that are motors (joint transmission,
            no or first-order dynamics); every other actuator is a tendon muscle.
        mirror_sides: Left / right suffix pairs of a left-right symmetric body (mirror
            augmentation of a learning method); ``None``: not symmetric.
        effectors: Tracked bodies whose world position error is a success metric
            (e.g. fingertips); empty: none.
        effector_metric: Name of that metric.
        effector_scale: Its unit per metre (100: cm).
        version: Version of the package the body comes from (the clips' refit metadata).
        njmax: MuJoCo-Warp constraint rows per world.
        nconmax: MuJoCo-Warp contacts per world.
    """

    name: str
    build_spec: Callable[[bool], mujoco.MjSpec]
    tracked_bodies: tuple[str, ...]
    foot_bodies: tuple[str, ...]
    muscle_tendon_expr: tuple[str, ...]
    motor_expr: tuple[str, ...] = ()
    mirror_sides: tuple[tuple[str, str], ...] | None = None
    effectors: tuple[str, ...] = ()
    effector_metric: str = "effector_err_mm"
    effector_scale: float = 1000.0
    version: Callable[[], str] = _myo_sim_version
    njmax: int = 768
    nconmax: int = 96

    @property
    def ground(self) -> bool:
        """The scene has a ground plane (a body with contact bodies)."""
        return bool(self.foot_bodies)


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


ELBOW_TIP = "hand_tip"
"""Massless body at the middle fingertip of ``myoelbow2dof`` (its effector)."""


def _myoelbow2dof_spec(self_collisions: bool = True) -> mujoco.MjSpec:
    """myo_sim's 2-DoF elbow (shoulder elevation, elbow flexion, 6 muscles, fixed base)
    with a fingertip body."""
    import myo_sim  # noqa: PLC0415

    del self_collisions  # the model has no contact pairs
    spec = mujoco.MjSpec.from_file(
        str(myo_sim.get_path("legacy/elbow/myoelbow_2dof6muscles.xml"))
    )
    for key in list(spec.keys):
        spec.delete(key)
    spec.body("r_ulna_radius_hand").add_body(name=ELBOW_TIP, pos=(0.027, -0.416, 0.065))
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
        mirror_sides=MIRROR_SIDES,
    ),
    "myoelbow2dof": MimicModelCfg(
        name="myoelbow2dof",
        build_spec=_myoelbow2dof_spec,
        tracked_bodies=("base", "r_humerus", "r_ulna_radius_hand", ELBOW_TIP),
        foot_bodies=(),
        muscle_tendon_expr=(".*",),
        effectors=(ELBOW_TIP,),
        effector_metric="tip_err_cm",
        effector_scale=100.0,
        njmax=64,
        nconmax=16,
    ),
}
DEFAULT_MODEL = "myofullbody"


def register_model(cfg: MimicModelCfg) -> None:
    """Add a body the task can run on (select it with ``task_kwargs={"model": name}``).

    Raises:
        ValueError: If another model has that name.
    """
    if cfg.name in MODELS and MODELS[cfg.name] is not cfg:
        raise ValueError(f"myoMimic already has a model {cfg.name!r}")
    MODELS[cfg.name] = cfg


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
    per foot body reading the net force of the ground on it (world frame). A body
    without contact bodies gets no ground.
    """
    spec = model.build_spec(self_collisions)
    if not model.ground:
        return spec
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
    """Version of the package model *name* comes from (myo_sim for the built-in model)."""
    return model_cfg(name).version()


@functools.lru_cache(maxsize=4)
def kinematic_fingerprint(name: str) -> str:
    """Kinematic fingerprint of model *name*'s body: what its clips are valid for.

    :func:`~myosuite.biomechanics.retarget.kinematic_fingerprint` of
    :func:`compiled_body` (joints, body tree, coupling equalities, foot collision
    geoms), with the landmark bodies and feet of the model's SMPL+H map (a body
    without one, e.g. a submodel: no landmarks, its contact bodies as the feet).
    Clips whose metadata carries it load whatever myo_sim version they were fitted
    under.
    """
    from myosuite.biomechanics.retarget import (  # noqa: PLC0415
        kinematic_fingerprint as fingerprint,
    )
    from myosuite.biomechanics.retarget_smplh import (  # noqa: PLC0415
        SMPLH_MAPS,
        landmark_bodies,
    )

    mc = model_cfg(name)
    smap = SMPLH_MAPS.get(mc.name)
    if smap is None:
        return fingerprint(compiled_body(name), (), [[b] for b in mc.foot_bodies])
    return fingerprint(compiled_body(name), landmark_bodies(smap), smap.feet)


@functools.lru_cache(maxsize=4)
def compiled_scene(name: str, self_collisions: bool = True) -> mujoco.MjModel:
    """Compiled :func:`scene_spec` (cached; copy before editing)."""
    return scene_spec(model_cfg(name), self_collisions).compile()


@functools.lru_cache(maxsize=4)
def compiled_body(name: str) -> mujoco.MjModel:
    """The body compiled on its own (no scene), for bookkeeping."""
    return model_cfg(name).build_spec(True).compile()


def body_info(name: str) -> ModelInfo:
    """:class:`ModelInfo` of model *name* (its motors classified by ``motor_expr``)."""
    return ModelInfo(compiled_body(name), model_cfg(name).motor_expr or None)


def task_on_model(task: Any, name: str) -> Any:
    """The task config *task* (a ``MimicTaskCfg``) adapted to model *name*
    (:func:`~.config.for_body`)."""
    from .config import for_body  # noqa: PLC0415

    return for_body(task, body_info(name).fixed_root, model_cfg(name).ground)


class ModelInfo:
    """Joint and actuator bookkeeping of a compiled body.

    * root: the body carrying the model's free joint, or the world (body 0) for a
      fixed root (no free joint; ``root_joint_name``, ``root_qadr`` and ``root_dadr``
      are ``None``);
    * independent joints: every non-root joint that is not the driven side of a joint
      ``<equality>``, in model order (the joint coordinates of the clips);
    * coupled joints: set from an independent joint by the equality's polynomial;
    * actuators: muscles (each driving one tendon) and motors (joint transmission, no
      or first-order dynamics), in actuator order.

    Args:
        m: The compiled body.
        motor_expr: Regexes of the motors' actuator names (``None``: every non-muscle
            actuator is a motor).
    """

    def __init__(self, m: mujoco.MjModel, motor_expr: Sequence[str] | None = None):
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) > 1:
            raise ValueError(
                f"expected at most one free joint (the root), found {len(free)}"
            )
        self.fixed_root = not free
        if free:
            self.root_joint_name = m.joint(free[0]).name
            self.root_id = int(m.jnt_bodyid[free[0]])
            self.root_qadr = int(m.jnt_qposadr[free[0]])
            self.root_dadr = int(m.jnt_dofadr[free[0]])
        else:
            self.root_joint_name = self.root_qadr = self.root_dadr = None
            self.root_id = 0
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
        names = [m.actuator(a).name for a in range(m.nu)]
        is_muscle = m.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        if motor_expr is None:
            is_motor = ~is_muscle
        else:
            pats = [re.compile(e) for e in motor_expr]
            is_motor = np.array(
                [any(p.fullmatch(n) for p in pats) for n in names], bool
            )
        muscles = [a for a in range(m.nu) if not is_motor[a]]
        motors = [a for a in range(m.nu) if is_motor[a]]
        if not (
            is_muscle[muscles].all()
            and (m.actuator_trntype[muscles] == mujoco.mjtTrn.mjTRN_TENDON).all()
        ):
            raise ValueError(
                "every actuator but the motors must be a tendon-driven muscle"
            )
        motor_dyn = (mujoco.mjtDyn.mjDYN_NONE, mujoco.mjtDyn.mjDYN_FILTER)
        if motors and not (
            np.isin(m.actuator_dyntype[motors], motor_dyn).all()
            and (m.actuator_trntype[motors] == mujoco.mjtTrn.mjTRN_JOINT).all()
        ):
            raise ValueError(
                "motors need a joint transmission and no or filter dynamics"
            )
        self.num_actuators = m.nu
        self.actuator_names = names
        self.muscle_ids = np.array(muscles, np.int64)
        """Actuator ids (= action columns) of the muscles."""
        self.motor_ids = np.array(motors, np.int64)
        """Actuator ids (= action columns) of the motors."""
        self.num_muscles = len(muscles)
        self.muscle_names = [names[a] for a in muscles]
        self.muscle_tendon_names = [
            m.tendon(int(t)).name for t in m.actuator_trnid[muscles, 0]
        ]
        self.motor_joint_names = [
            m.joint(int(j)).name for j in m.actuator_trnid[motors, 0]
        ]
        self.qpos0 = m.qpos0.copy()
        self.act_adr = m.actuator_actadr[muscles].copy()
        """``act`` addresses of the muscles."""
        lo, hi = m.actuator_ctrlrange[motors].T
        self.motor_lo, self.motor_hi = lo.copy(), hi.copy()
        """Control range of the motors (an action in [0, 1] maps onto it)."""

    def action_to_ctrl(self, xp: Any, action: Any) -> Any:
        """Control of an action in [0, 1] (clipped): the muscles' excitation as is, the
        motors' mapped affinely onto their control range."""
        ctrl = (
            xp.clip(action, 0.0, 1.0)
            if xp.__name__ == "numpy"
            else action.clamp(0.0, 1.0)
        )
        if not len(self.motor_ids):
            return ctrl
        lo, hi = (_as(xp, x, ctrl) for x in (self.motor_lo, self.motor_hi))
        mot = lo + ctrl[..., self.motor_ids] * (hi - lo)
        if xp.__name__ == "numpy":
            ctrl = ctrl.copy()
            ctrl[..., self.motor_ids] = mot
            return ctrl
        return ctrl.index_copy(-1, _as(xp, self.motor_ids, ctrl, long=True), mot)


def _as(xp: Any, x: np.ndarray, like: Any, long: bool = False) -> Any:
    if xp.__name__ == "numpy":
        return x
    return xp.as_tensor(x, device=like.device, dtype=xp.long if long else like.dtype)


def root_offset(m: mujoco.MjModel) -> np.ndarray:
    """``R_off`` with ``R_root @ R_off = I`` in the default pose (canonical clip frame);
    the identity for a fixed root."""
    info = ModelInfo(m)
    if info.fixed_root:
        return np.eye(3)
    adr = m.jnt_qposadr[m.body_jntadr[info.root_id]]
    R_rest = np.zeros(9)
    mujoco.mju_quat2Mat(R_rest, m.qpos0[adr + 3 : adr + 7])
    return R_rest.reshape(3, 3).T.copy()
