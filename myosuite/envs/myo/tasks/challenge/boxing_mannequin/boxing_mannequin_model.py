# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Model builder for boxer-vs-mannequin boxing scene."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path

import mujoco
import numpy as np

from myosuite.envs.myo.tasks.challenge.boxing_mannequin.boxing_mannequin_config import (
    BoxingMannequinConfig,
)
from myosuite.envs.myo.tasks.challenge.combat_model_meta import CombatModelMetaBase
from myosuite.envs.myo.tasks.challenge.boxing_visuals import (
    BOXING_RING_MESH,
    RING_POS,
    RING_SCALE,
    add_helmet,
    replace_hand_visuals_with_gloves,
)
from myosuite.integrations.musclemimic.fullbody_model import (
    build_mimic_fullbody_spec,
    default_mimic_fullbody_config,
)
from myosuite.integrations.musclemimic.two_agent_scene import (
    build_combined_spec,
    extract_prefix_indices,
    extract_sensor_addrs,
    extract_site_ids,
    mj_name2id_strict,
)
from myosuite.physics.quat_math import euler2quat

logger = logging.getLogger(__name__)

_ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"

AGENTS: tuple[str, str] = ("agent_0", "agent_1")
_AGENT_PREFIXES: dict[str, str] = {"agent_0": "a0_", "agent_1": "a1_"}
_ARM_JOINT_TOKENS: tuple[str, ...] = (
    "shoulder_elv_l",
    "shoulder_elv_r",
    "elv_angle_l",
    "elv_angle_r",
    "shoulder_rot_l",
    "shoulder_rot_r",
    "elbow_flex_l",
    "elbow_flex_r",
    "pro_sup_l",
    "pro_sup_r",
    "deviation_l",
    "deviation_r",
    "flexion_l",
    "flexion_r",
)


@dataclass
class BoxingMannequinModelMeta(CombatModelMetaBase):
    """Cached ids/indices used by obs/reward/scripted-opponent code."""

    body_ids: dict[str, int] = field(default_factory=dict)
    target_site_ids: list[int] = field(default_factory=list)
    target_geom_ids: list[int] = field(default_factory=list)
    agent_body_ids: dict[str, set[int]] = field(default_factory=dict)
    fist_body_ids: dict[str, set[int]] = field(default_factory=dict)


def _add_boxer_sites_and_sensors(spec: mujoco.MjSpec, prefix: str) -> None:
    """Add fist/target/pelvis sites and linear-velocity sensors for boxer."""
    spec.body(f"{prefix}3proxph_r").add_site(name=f"{prefix}fist_r", size=[0.015, 0, 0])
    spec.body(f"{prefix}3proxph_l").add_site(name=f"{prefix}fist_l", size=[0.015, 0, 0])
    spec.body(f"{prefix}head").add_site(name=f"{prefix}head_zone", size=[0.02, 0, 0])
    spec.body(f"{prefix}lumbar1").add_site(name=f"{prefix}body_zone", size=[0.02, 0, 0])
    spec.body(f"{prefix}pelvis").add_site(
        name=f"{prefix}pelvis_site", size=[0.01, 0, 0]
    )
    spec.body(f"{prefix}pelvis").add_site(
        name=f"{prefix}waist_zone",
        pos=[0.0, 0.0, -0.12],
        size=[0.02, 0, 0],
    )
    for suffix in ("fist_r", "fist_l", "pelvis_site"):
        spec.add_sensor(
            name=f"{prefix}{suffix}_vel",
            type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=f"{prefix}{suffix}",
        )


def _build_mannequin_agent_spec(config: BoxingMannequinConfig) -> mujoco.MjSpec:
    """Build mannequin-only spec to be attached as agent_1."""
    spec = mujoco.MjSpec()
    spec.modelname = "boxing_mannequin_agent"
    spec.option.timestep = config.sim_dt
    spec.compiler.meshdir = str(_ASSETS_DIR)
    spec.compiler.autolimits = True
    spec.compiler.inertiafromgeom = True

    mannequin_mesh = spec.add_mesh()
    mannequin_mesh.name = "mannequin_mesh"
    mannequin_mesh.file = "BoxingDummy.stl"
    mannequin_mesh.scale = [1.1, 1.1, 1.1]

    glove_r_mesh = spec.add_mesh()
    glove_r_mesh.name = "glove_r_mesh"
    glove_r_mesh.file = "boxing_glove_r.stl"
    glove_r_mesh.scale = [0.06, 0.06, 0.06]

    glove_l_mesh = spec.add_mesh()
    glove_l_mesh.name = "glove_l_mesh"
    glove_l_mesh.file = "boxing_glove_r.stl"
    glove_l_mesh.scale = [-0.06, 0.06, 0.06]

    mannequin_mat = spec.add_material()
    mannequin_mat.name = "mannequin_mat"
    mannequin_mat.rgba = [0.75, 0.68, 0.58, 1.0]
    head_mat = spec.add_material()
    head_mat.name = "mannequin_head_mat"
    head_mat.rgba = [0.8, 0.72, 0.62, 1.0]
    glove_r_mat = spec.add_material()
    glove_r_mat.name = "mannequin_glove_r_mat"
    glove_r_mat.rgba = [0.14, 0.18, 0.84, 1.0]
    glove_l_mat = spec.add_material()
    glove_l_mat.name = "mannequin_glove_l_mat"
    glove_l_mat.rgba = [0.14, 0.18, 0.84, 1.0]

    mannequin = spec.worldbody.add_body(name="mannequin")
    mannequin.pos = [0.0, 0.0, 0.0]
    mannequin.add_joint(
        name="mannequin_x",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[1, 0, 0],
        range=[-1.5, 1.5],
    )
    mannequin.add_joint(
        name="mannequin_y",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, 1, 0],
        range=[-1.5, 1.5],
    )
    mannequin.add_joint(
        name="mannequin_rotation",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 0, 1],
        range=[-3.14, 3.14],
    )

    mannequin_visual = mannequin.add_geom(name="mannequin_visual")
    mannequin_visual.type = mujoco.mjtGeom.mjGEOM_MESH
    mannequin_visual.meshname = "mannequin_mesh"
    mannequin_visual.material = "mannequin_mat"
    mannequin_visual.contype = 0
    mannequin_visual.conaffinity = 0

    chest = mannequin.add_geom(name="mannequin_chest_collision")
    chest.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
    chest.pos = [0.0, 0.07, 1.33]
    chest.size = [0.35, 0.17, 0.17]

    waist = mannequin.add_geom(name="mannequin_waist_collision")
    waist.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
    waist.pos = [0.0, 0.035, 1.05]
    waist.size = [0.23, 0.17, 0.25]

    head = mannequin.add_geom(name="mannequin_head_collision")
    head.type = mujoco.mjtGeom.mjGEOM_SPHERE
    head.pos = [0.0, 0.025, 1.68]
    head.size = [0.17, 0.0, 0.0]
    head.mass = 4.0
    head.material = "mannequin_head_mat"

    mannequin.add_site(name="pelvis_site", pos=[0, 0, 1.068], size=[0.01, 0, 0])
    mannequin.add_site(name="head_zone", pos=[0.0, 0.025, 1.68], size=[0.02, 0, 0])
    mannequin.add_site(name="body_zone", pos=[0.0, 0.07, 1.33], size=[0.02, 0, 0])
    mannequin.add_site(name="waist_zone", pos=[0.0, 0.035, 1.05], size=[0.02, 0, 0])

    right = mannequin.add_body(name="right_glove_anchor")
    right.pos = [0.402, 0.126, 1.068]
    right.add_joint(
        name="right_glove_x",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[1, 0, 0],
        range=[-0.65, 0.0],
    )
    right.add_joint(
        name="right_glove_y",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, 1, 0],
        range=[0.0, 0.55],
    )
    right.add_joint(
        name="right_glove_z",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, 0, 1],
        range=[0.0, 0.45],
    )
    right_visual = right.add_geom(name="right_glove_visual")
    right_visual.type = mujoco.mjtGeom.mjGEOM_MESH
    right_visual.meshname = "glove_r_mesh"
    right_visual.material = "mannequin_glove_r_mat"
    right_visual.contype = 0
    right_visual.conaffinity = 0
    right_collision = right.add_geom(name="right_glove_collision")
    right_collision.type = mujoco.mjtGeom.mjGEOM_SPHERE
    right_collision.pos = [0.0, 0.082, -0.006]
    right_collision.size = [0.07, 0.0, 0.0]
    right_collision.mass = 1.2
    right.add_site(name="fist_r", pos=[0.05, 0.0, 0.013], size=[0.012, 0, 0])

    left = mannequin.add_body(name="left_glove_anchor")
    left.pos = [-0.402, 0.126, 1.068]
    left.add_joint(
        name="left_glove_x",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[1, 0, 0],
        range=[0.0, 0.65],
    )
    left.add_joint(
        name="left_glove_y",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, 1, 0],
        range=[0.0, 0.55],
    )
    left.add_joint(
        name="left_glove_z",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, 0, 1],
        range=[0.0, 0.45],
    )
    left_visual = left.add_geom(name="left_glove_visual")
    left_visual.type = mujoco.mjtGeom.mjGEOM_MESH
    left_visual.meshname = "glove_l_mesh"
    left_visual.material = "mannequin_glove_l_mat"
    left_visual.contype = 0
    left_visual.conaffinity = 0
    left_collision = left.add_geom(name="left_glove_collision")
    left_collision.type = mujoco.mjtGeom.mjGEOM_SPHERE
    left_collision.pos = [0.0, 0.082, -0.006]
    left_collision.size = [0.07, 0.0, 0.0]
    left_collision.mass = 1.2
    left.add_site(name="fist_l", pos=[-0.05, 0.0, 0.013], size=[0.012, 0, 0])

    for suffix in ("fist_r", "fist_l", "pelvis_site"):
        spec.add_sensor(
            name=f"{suffix}_vel",
            type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=suffix,
        )

    def add_position(name: str, joint: str, low: float, high: float) -> None:
        act = spec.add_actuator(
            name=name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            dyntype=mujoco.mjtDyn.mjDYN_NONE,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            biastype=mujoco.mjtBias.mjBIAS_NONE,
        )
        act.target = joint
        act.ctrlrange = [low, high]
        act.ctrllimited = True
        act.gainprm[0] = 1800.0

    add_position("mannequin_x_ctrl", "mannequin_x", -1.5, 1.5)
    add_position("mannequin_y_ctrl", "mannequin_y", -1.5, 1.5)
    add_position("mannequin_rotation_ctrl", "mannequin_rotation", -3.14, 3.14)
    add_position("right_glove_x_ctrl", "right_glove_x", -0.65, 0.0)
    add_position("right_glove_y_ctrl", "right_glove_y", 0.0, 0.55)
    add_position("right_glove_z_ctrl", "right_glove_z", 0.0, 0.45)
    add_position("left_glove_x_ctrl", "left_glove_x", 0.0, 0.65)
    add_position("left_glove_y_ctrl", "left_glove_y", 0.0, 0.55)
    add_position("left_glove_z_ctrl", "left_glove_z", 0.0, 0.45)
    return spec


def _quat_from_euler_deg(roll_pitch_yaw_deg: tuple[float, float, float]) -> list[float]:
    """Convert MuJoCo XML euler degrees to quat [w, x, y, z]."""
    quat = euler2quat(np.deg2rad(np.asarray(roll_pitch_yaw_deg, dtype=np.float64)))
    return quat.astype(np.float64).tolist()


def _build_targets_agent_spec(config: BoxingMannequinConfig) -> mujoco.MjSpec:
    """Build 6-target boxing machine as the opponent (agent_1) spec."""
    spec = mujoco.MjSpec()
    spec.modelname = "boxing_targets_agent"
    spec.option.timestep = config.sim_dt
    spec.compiler.meshdir = str(_ASSETS_DIR)
    spec.compiler.autolimits = True
    spec.compiler.inertiafromgeom = True

    targets_mesh = spec.add_mesh()
    targets_mesh.name = "boxing_targets_mesh"
    targets_mesh.file = "Boxing8targets.stl"

    mesh_gray = spec.add_material()
    mesh_gray.name = "boxing_targets_mesh_gray"
    mesh_gray.rgba = [0.4, 0.4, 0.4, 1.0]
    mesh_gray.shininess = 0.2

    target_red = spec.add_material()
    target_red.name = "boxing_targets_red"
    target_red.rgba = [0.8, 0.0, 0.0, 1.0]

    machine = spec.worldbody.add_body(name="mannequin", pos=[0.0, 0.0, 0.0])
    machine.add_joint(
        name="mannequin_x",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[1, 0, 0],
        range=[-1.5, 1.5],
    )
    machine.add_joint(
        name="mannequin_y",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, 1, 0],
        range=[-1.5, 1.5],
    )
    machine.add_joint(
        name="mannequin_rotation",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 0, 1],
        range=[-3.14, 3.14],
    )
    machine_mesh = machine.add_geom(
        name="mannequin_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        contype=0,
        conaffinity=0,
    )
    machine_mesh.meshname = "boxing_targets_mesh"
    machine_mesh.material = "boxing_targets_mesh_gray"

    pad_data = (
        (
            "pad_1",
            [0.0, 0.1, 1.65],
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            [0.14, 0.01, 0.0],
            (60.0, 0.0, 0.0),
            (90.0, 0.0, 0.0),
        ),
        (
            "pad_2",
            [0.35, 0.0, 1.55],
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            [0.10, 0.01, 0.0],
            (80.0, 55.0, 40.0),
            (90.0, 0.0, -45.0),
        ),
        (
            "pad_3",
            [0.0, 0.18, 1.2],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            [0.16, 0.0, 0.0],
            None,
            (90.0, 0.0, -135.0),
        ),
        (
            "pad_4",
            [0.28, 0.23, 1.05],
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            [0.08, 0.01, 0.0],
            (50.0, 57.0, 40.0),
            (90.0, 0.0, -135.0),
        ),
        (
            "pad_5",
            [-0.26, 0.25, 1.05],
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            [0.08, 0.01, 0.0],
            (35.0, -50.0, 40.0),
            (90.0, 0.0, -225.0),
        ),
        (
            "pad_6",
            [-0.33, 0.05, 1.53],
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            [0.10, 0.01, 0.0],
            (80.0, -55.0, 40.0),
            (90.0, 0.0, -270.0),
        ),
    )
    for name, pos, geom_type, size, geom_euler, site_euler in pad_data:
        pad = machine.add_body(name=name, pos=pos)
        geom = pad.add_geom(
            name=f"{name}_geom",
            type=geom_type,
            size=size,
            material="boxing_targets_red",
            contype=1,
            conaffinity=1,
        )
        if geom_euler is not None:
            geom.quat = _quat_from_euler_deg(geom_euler)
        site = pad.add_site(
            name=f"{name}_site",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.081, 0.011, 0.0],
            rgba=[0.0, 0.0, 0.0, 0.0],
        )
        site.quat = _quat_from_euler_deg(site_euler)

    machine.add_site(name="pelvis_site", pos=[0.0, 0.0, 1.068], size=[0.01, 0.0, 0.0])
    machine.add_site(name="head_zone", pos=[0.0, 0.1, 1.65], size=[0.02, 0.0, 0.0])
    machine.add_site(name="body_zone", pos=[0.0, 0.18, 1.2], size=[0.02, 0.0, 0.0])
    machine.add_site(name="waist_zone", pos=[0.0, 0.0, 1.05], size=[0.02, 0.0, 0.0])

    machine.add_site(name="fist_r", pos=[0.35, 0.0, 1.55], size=[0.012, 0.0, 0.0])
    machine.add_site(name="fist_l", pos=[-0.33, 0.05, 1.53], size=[0.012, 0.0, 0.0])

    for suffix in ("fist_r", "fist_l", "pelvis_site"):
        spec.add_sensor(
            name=f"{suffix}_vel",
            type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=suffix,
        )

    def add_position(name: str, joint: str, low: float, high: float) -> None:
        act = spec.add_actuator(
            name=name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            dyntype=mujoco.mjtDyn.mjDYN_NONE,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            biastype=mujoco.mjtBias.mjBIAS_NONE,
        )
        act.target = joint
        act.ctrlrange = [low, high]
        act.ctrllimited = True
        act.gainprm[0] = 1800.0

    add_position("mannequin_x_ctrl", "mannequin_x", -1.5, 1.5)
    add_position("mannequin_y_ctrl", "mannequin_y", -1.5, 1.5)
    add_position("mannequin_rotation_ctrl", "mannequin_rotation", -3.14, 3.14)
    return spec


def _replace_floor_visual_with_boxing_ring(spec: mujoco.MjSpec) -> None:
    ring_mesh = spec.add_mesh()
    ring_mesh.name = "boxing_ring_mesh"
    ring_mesh.file = str(BOXING_RING_MESH)
    ring_mesh.scale = RING_SCALE
    ring_material = spec.add_material()
    ring_material.name = "boxing_ring_mat"
    ring_material.rgba = [1.0, 1.0, 1.0, 1.0]
    ring_material.specular = 0.05
    ring_material.shininess = 0.05
    for geom in spec.geoms:
        if (geom.name or "") == "floor":
            geom.rgba = [0.0, 0.0, 0.0, 0.0]
            break
    ring_geom = spec.worldbody.add_geom(
        name="boxing_ring_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        pos=RING_POS,
        contype=0,
        conaffinity=0,
    )
    ring_geom.meshname = "boxing_ring_mesh"
    ring_geom.material = "boxing_ring_mat"


def _build_boxing_mannequin_spec(config: BoxingMannequinConfig) -> mujoco.MjSpec:
    fullbody_cfg = default_mimic_fullbody_config()
    fullbody_cfg.disable_fingers = config.disable_fingers
    fullbody_cfg.sim_dt = config.sim_dt
    boxer_spec, _ = build_mimic_fullbody_spec(fullbody_cfg)
    replace_hand_visuals_with_gloves(boxer_spec, rgba=[0.84, 0.12, 0.12, 1.0])
    add_helmet(boxer_spec, rgba=[0.80, 0.10, 0.10, 1.0])
    mannequin_spec = (
        _build_targets_agent_spec(config)
        if config.include_boxing_targets_scene
        else _build_mannequin_agent_spec(config)
    )

    spec = build_combined_spec(
        boxer_spec,
        mannequin_spec,
        separation_m=config.agent_separation_m,
        sim_dt=config.sim_dt,
        model_name="boxing_mannequin",
        floor_rgba=[0.75, 0.75, 0.75, 1.0],
        floor_collision=True,
    )
    # The mannequin mesh is authored facing +Y; unlike VS we keep agent_1
    # unrotated so it presents the front toward the boxer at reset.
    for frame in spec.worldbody.frames:
        if frame.pos[1] < 0.0:
            frame.quat = [1.0, 0.0, 0.0, 0.0]
            break
    _replace_floor_visual_with_boxing_ring(spec)
    _add_boxer_sites_and_sensors(spec, "a0_")
    return spec


def _extract_meta(model: mujoco.MjModel) -> BoxingMannequinModelMeta:
    meta = BoxingMannequinModelMeta(n_act=model.nu)
    for agent_id, prefix in _AGENT_PREFIXES.items():
        arm_tokens = _ARM_JOINT_TOKENS if agent_id == "agent_0" else ()
        act_idx, jnt_ids, arm_jnt_ids = extract_prefix_indices(
            model, prefix, arm_joint_tokens=arm_tokens
        )
        meta.act_indices[agent_id] = act_idx
        meta.jnt_ids[agent_id] = jnt_ids
        meta.arm_jnt_ids[agent_id] = arm_jnt_ids
        meta.site_ids[agent_id] = extract_site_ids(
            model,
            prefix,
            ("fist_r", "fist_l", "head_zone", "body_zone", "waist_zone", "pelvis_site"),
        )
        meta.sensor_adr[agent_id] = extract_sensor_addrs(
            model, prefix, ("fist_r", "fist_l", "pelvis_site")
        )
    meta.body_ids["a0_pelvis"] = mj_name2id_strict(
        model, mujoco.mjtObj.mjOBJ_BODY, "a0_pelvis"
    )
    meta.body_ids["a1_mannequin"] = mj_name2id_strict(
        model, mujoco.mjtObj.mjOBJ_BODY, "a1_mannequin"
    )
    for agent_id, prefix in _AGENT_PREFIXES.items():
        body_ids: set[int] = set()
        for body_idx in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_idx)
            if body_name is not None and body_name.startswith(prefix):
                body_ids.add(body_idx)
        meta.agent_body_ids[agent_id] = body_ids
        fist_body_ids: set[int] = set()
        for fist_site in ("fist_r", "fist_l"):
            site_id = meta.site_ids[agent_id].get(fist_site)
            if site_id is None:
                continue
            fist_body_ids.add(int(model.site_bodyid[site_id]))
        meta.fist_body_ids[agent_id] = fist_body_ids

    target_site_ids: list[int] = []
    target_geom_ids: list[int] = []
    for pad_idx in range(1, 7):
        site_name = f"a1_pad_{pad_idx}_site"
        geom_name = f"a1_pad_{pad_idx}_geom"
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if site_id != -1 and geom_id != -1:
            target_site_ids.append(site_id)
            target_geom_ids.append(geom_id)
    meta.target_site_ids = target_site_ids
    meta.target_geom_ids = target_geom_ids
    return meta


def build_boxing_mannequin_model(
    config: BoxingMannequinConfig,
) -> tuple[mujoco.MjModel, mujoco.MjData, BoxingMannequinModelMeta]:
    """Build boxer-vs-mannequin model and extract metadata."""
    spec = _build_boxing_mannequin_spec(config)
    model = spec.compile()
    model.opt.timestep = config.sim_dt
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    meta = _extract_meta(model)
    logger.info(
        "Boxing mannequin model compiled: nq=%d nv=%d nu=%d",
        model.nq,
        model.nv,
        model.nu,
    )
    return model, data, meta
