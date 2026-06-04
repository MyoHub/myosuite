# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Standalone MJCF assets for the mjlab saber scene."""

from __future__ import annotations

from pathlib import Path

import mujoco

from myosuite.envs.myo.tasks.challenge.saber.saber_target_pool import (
    SABER_POOL_GRAVEYARD_Z,
    SABER_POOL_SPHERE_RADIUS,
    SABER_TARGET_POOL_SIZE,
    graveyard_offset_xy,
)
from myosuite.integrations.musclemimic.myotorso_bimanual_model import (
    build_myotorso_bimanual_mimic_spec,
    default_myotorso_bimanual_mimic_config,
)
from myosuite.utils.simhive_path import get_simhive_asset_root, resolve_model_xml_path

SABER_BODY_XML = "myoarm_saber_body.xml"
LEFT_LIGHTSABER_XML = "left_lightsaber.xml"
RIGHT_LIGHTSABER_XML = "right_lightsaber.xml"
SABER_TARGETS_XML = "saber_targets.xml"

_MOUNT_QUAT = (0.7071067811865476, 0.0, 0.7071067811865476, 0.0)
_LEFT_SABER_POS = (-0.26, -0.12, 1.15)
_RIGHT_SABER_POS = (0.06, -0.12, 1.15)
_BLADE_CENTER_Z = 0.28
_BLADE_HALF_LENGTH = 0.18
_BLADE_TIP_Z = _BLADE_CENTER_Z + _BLADE_HALF_LENGTH
_MARKER_HALF_SPAN = 0.40 * SABER_POOL_SPHERE_RADIUS * 1.75


_FENCING_HELMET_MESH = (
    Path(__file__).resolve().parents[2] / "assets" / "fencing_helmet.stl"
)
_FENCING_HELMET_SCALE = 0.28
_FENCING_HELMET_POS = [0.05, 0.14, 0.0]
_FENCING_HELMET_QUAT = [0.5, -0.5, 0.5, 0.5]


def _add_saber_hand_contact_patches(spec: mujoco.MjSpec) -> None:
    """Add palm/thumb collision patches so saber can physically contact hand."""
    patches = (
        ("lunate_l", "left_palm_contact", [0.0, -0.01, 0.0], [0.026, 0.0, 0.0]),
        ("lunate_r", "right_palm_contact", [0.0, -0.01, 0.0], [0.026, 0.0, 0.0]),
        (
            "proximal_thumb_l",
            "left_thumb_prox_contact",
            [0.0, 0.0, 0.0],
            [0.012, 0.0, 0.0],
        ),
        (
            "distal_thumb_l",
            "left_thumb_dist_contact",
            [0.0, 0.0, 0.0],
            [0.010, 0.0, 0.0],
        ),
        (
            "proximal_thumb_r",
            "right_thumb_prox_contact",
            [0.0, 0.0, 0.0],
            [0.012, 0.0, 0.0],
        ),
        (
            "distal_thumb_r",
            "right_thumb_dist_contact",
            [0.0, 0.0, 0.0],
            [0.010, 0.0, 0.0],
        ),
    )
    for body_name, geom_name, pos, size in patches:
        try:
            body = spec.body(body_name)
        except (KeyError, ValueError):
            continue
        geom = body.add_geom(
            name=geom_name,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=pos,
            size=size,
            contype=1,
            conaffinity=1,
            rgba=[0.2, 1.0, 0.2, 0.0],
            mass=0.001,
        )
        geom.condim = 4


def _add_saber_helmet(spec: mujoco.MjSpec) -> None:
    """Attach the visual-only fencing helmet used in saber scenes."""

    helmet_mesh = spec.add_mesh()
    helmet_mesh.name = "fencing_helmet_mesh"
    helmet_mesh.file = str(_FENCING_HELMET_MESH.resolve())
    helmet_mesh.scale = [
        _FENCING_HELMET_SCALE,
        _FENCING_HELMET_SCALE,
        _FENCING_HELMET_SCALE,
    ]
    helmet_mat = spec.add_material(name="fencing_helmet_mat")
    helmet_mat.rgba = [0.2, 0.2, 0.2, 1.0]
    helmet_mat.specular = 0.35
    helmet_mat.shininess = 0.45
    helmet_mat.emission = 0.15
    helmet_mat.reflectance = 0.0
    neck = spec.body("cervical_spine")
    shell = neck.add_geom(
        name="fencing_helmet_shell",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        pos=_FENCING_HELMET_POS,
        quat=_FENCING_HELMET_QUAT,
        rgba=[0.2, 0.2, 0.2, 1.0],
        contype=0,
        conaffinity=0,
    )
    shell.meshname = "fencing_helmet_mesh"
    shell.material = "fencing_helmet_mat"


def saber_scene_asset_dir() -> Path:
    """Return the directory that stores standalone saber scene XML files."""
    return get_simhive_asset_root("myo_sim")


def saber_scene_asset_paths() -> dict[str, Path]:
    """Return absolute paths for the standalone mjlab saber scene assets."""
    root = saber_scene_asset_dir()
    return {
        "body": root / SABER_BODY_XML,
        "left_saber": root / LEFT_LIGHTSABER_XML,
        "right_saber": root / RIGHT_LIGHTSABER_XML,
        "targets": root / SABER_TARGETS_XML,
    }


def ensure_saber_scene_assets() -> dict[str, Path]:
    """Ensure the standalone saber scene XML files exist on disk."""
    paths = saber_scene_asset_paths()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if not missing:
        return paths

    paths["body"].write_text(_build_saber_body_xml_text(), encoding="utf-8")
    paths["left_saber"].write_text(_build_left_saber_xml_text(), encoding="utf-8")
    paths["right_saber"].write_text(_build_right_saber_xml_text(), encoding="utf-8")
    paths["targets"].write_text(_build_saber_targets_xml_text(), encoding="utf-8")
    return paths


def load_saber_scene_spec(path: Path) -> mujoco.MjSpec:
    """Load a saber scene asset with package-relative paths resolved."""
    return mujoco.MjSpec.from_file(str(resolve_model_xml_path(path)))


def build_saber_body_spec() -> mujoco.MjSpec:
    """Load the standalone saber robot-body spec."""
    return load_saber_scene_spec(ensure_saber_scene_assets()["body"])


def build_left_saber_spec() -> mujoco.MjSpec:
    """Load the standalone left-saber spec."""
    return load_saber_scene_spec(ensure_saber_scene_assets()["left_saber"])


def build_right_saber_spec() -> mujoco.MjSpec:
    """Load the standalone right-saber spec."""
    return load_saber_scene_spec(ensure_saber_scene_assets()["right_saber"])


def build_saber_targets_spec() -> mujoco.MjSpec:
    """Load the standalone full target-pool spec."""
    return load_saber_scene_spec(ensure_saber_scene_assets()["targets"])


def build_isolated_saber_target_spec(target_index: int) -> mujoco.MjSpec:
    """Load one target-pool body as its own mocap-root spec.

    The source ``saber_targets.xml`` contains an inert anchor plus all
    ``saber_target_{i}`` bodies. For per-target entities we must delete the
    anchor and every non-selected target so the remaining root body is the
    mocap body itself.
    """

    keep_name = f"saber_target_{int(target_index)}"
    spec = build_saber_targets_spec()
    found_keep = False
    for body in list(spec.bodies):
        if body.name == "world":
            continue
        if body.name == keep_name:
            found_keep = True
            continue
        spec.delete(body)
    if not found_keep:
        raise ValueError(f"Unknown saber target body {keep_name!r}.")
    return spec


def build_saber_scene_spec() -> mujoco.MjSpec:
    """Compose the split saber assets back into one unprefixed scene spec."""

    spec = build_saber_body_spec()
    for child in (
        build_left_saber_spec(),
        build_right_saber_spec(),
        build_saber_targets_spec(),
    ):
        frame = spec.worldbody.add_frame()
        spec.attach(child=child, frame=frame, prefix="")
    return spec


def build_saber_scene_mjmodel_and_spec() -> tuple[mujoco.MjModel, mujoco.MjSpec]:
    """Compile and return the combined saber scene model/spec pair."""

    spec = build_saber_scene_spec()
    return spec.compile(), spec


def saber_scene_callable(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Drop-in scene callable for SaberP0Task.

    Ensures that the CPU ``ModularTaskEnv`` and the mjlab backend share the
    same XML-asset scene assembly code.  The ``spec`` argument (the base model
    loaded from ``task_config.model``) is intentionally ignored — the full scene
    is built from the standalone saber XML files managed by this module.
    """

    del spec
    scene = build_saber_scene_spec()
    return scene


def build_saber_body_mjmodel_and_spec() -> tuple[mujoco.MjModel, mujoco.MjSpec]:
    """Compile and return the standalone saber robot-body model/spec pair."""

    spec = build_saber_body_spec()
    return spec.compile(), spec


def _build_saber_body_xml_text() -> str:
    cfg = default_myotorso_bimanual_mimic_config()
    cfg.disable_fingers = False
    spec, _ = build_myotorso_bimanual_mimic_spec(cfg)
    _add_saber_hand_contact_patches(spec)
    _add_saber_helmet(spec)
    spec.visual.quality.shadowsize = 0
    spec.visual.map.haze = 0.0
    spec.visual.map.alpha = 0.0
    spec.visual.headlight.ambient[:] = 0.25
    spec.visual.headlight.diffuse[:] = 0.25
    spec.visual.headlight.specular[:] = 0.25
    xml = spec.to_xml()
    xml = xml.replace(
        '<mujoco model="myotorso_arm_chain_host">',
        '<mujoco model="myoarm_saber_body">',
        1,
    )
    xml = xml.replace(
        'meshdir="../../../../simhive/myo_sim/"',
        'meshdir="."',
    ).replace(
        'texturedir="../../../../simhive/myo_sim/"',
        'texturedir="."',
    )
    return xml


def _build_left_saber_xml_text() -> str:
    return _build_lightsaber_xml_text(
        model_name="left_lightsaber",
        body_name="left_lightsaber",
        joint_name="left_saber_freejoint",
        hilt_name="left_saber_hilt",
        blade_name="left_saber_geom",
        aura_name="left_blade_aura",
        light_name="left_saber_light",
        light_diffuse="0.0 0.5 1.0",
        tip_name="left_saber_tip",
        blade_base_name="left_saber_blade_base",
        material_core_name="left_saber_blue_core",
        material_aura_name="left_saber_blue_aura",
        material_core_rgba="0.02 0.30 1.0 1.0",
        material_aura_rgba="0.1 0.35 1.0 0.05",
        blade_rgba="1.0 1.0 1.0 1.0",
        aura_rgba="0.10 0.35 1.0 0.09",
        tip_rgba="0.2 0.8 1.0 0.2",
        blade_base_rgba="0.2 0.8 1.0 0.0",
        body_pos=_LEFT_SABER_POS,
    )


def _build_right_saber_xml_text() -> str:
    return _build_lightsaber_xml_text(
        model_name="right_lightsaber",
        body_name="right_lightsaber",
        joint_name="right_saber_freejoint",
        hilt_name="right_saber_hilt",
        blade_name="right_saber_geom",
        aura_name="right_blade_aura",
        light_name="right_saber_light",
        light_diffuse="1.0 0.35 0.55",
        tip_name="right_saber_tip",
        blade_base_name="right_saber_blade_base",
        material_core_name="right_saber_red_core",
        material_aura_name="right_saber_red_aura",
        material_core_rgba="1.0 0.08 0.08 1.0",
        material_aura_rgba="1.0 0.18 0.32 0.05",
        blade_rgba="1.0 1.0 1.0 1.0",
        aura_rgba="1.0 0.18 0.32 0.09",
        tip_rgba="1.0 0.35 0.35 0.2",
        blade_base_rgba="1.0 0.35 0.35 0.0",
        body_pos=_RIGHT_SABER_POS,
    )


def _build_lightsaber_xml_text(
    *,
    model_name: str,
    body_name: str,
    joint_name: str,
    hilt_name: str,
    blade_name: str,
    aura_name: str,
    light_name: str,
    light_diffuse: str,
    tip_name: str,
    blade_base_name: str,
    material_core_name: str,
    material_aura_name: str,
    material_core_rgba: str,
    material_aura_rgba: str,
    blade_rgba: str,
    aura_rgba: str,
    tip_rgba: str,
    blade_base_rgba: str,
    body_pos: tuple[float, float, float],
) -> str:
    pos = " ".join(str(v) for v in body_pos)
    quat = " ".join(str(v) for v in _MOUNT_QUAT)
    return f"""<mujoco model="{model_name}">
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <asset>
    <material name="{material_core_name}" rgba="{material_core_rgba}" emission="2" specular="0.05" shininess="0.05"/>
    <material name="{material_aura_name}" rgba="{material_aura_rgba}" emission="2"/>
  </asset>
  <worldbody>
    <body name="{body_name}" pos="{pos}" quat="{quat}">
      <freejoint name="{joint_name}"/>
      <geom name="{hilt_name}" type="cylinder" pos="0 0 0.06" quat="1 0 0 0" size="0.018 0.08 0" rgba="0.4 0.4 0.4 1" mass="0.5" contype="1" conaffinity="1" condim="4"/>
      <geom name="{blade_name}" type="capsule" pos="0 0 {_BLADE_CENTER_Z}" quat="1 0 0 0" size="0.01 {_BLADE_HALF_LENGTH} 0" rgba="{blade_rgba}" mass="0.001" contype="1" conaffinity="1" condim="4" material="{material_core_name}" friction="0.01 0.005 0.0001"/>
      <geom name="{aura_name}" type="capsule" pos="0 0 {_BLADE_CENTER_Z}" quat="1 0 0 0" size="0.025 {_BLADE_HALF_LENGTH + 0.005} 0" rgba="{aura_rgba}" contype="0" conaffinity="0" material="{material_aura_name}"/>
      <light name="{light_name}" pos="0 0 0.35" dir="0 0 1" diffuse="{light_diffuse}" specular="{light_diffuse}" range="2.75" cutoff="180" castshadow="true"/>
      <site name="{tip_name}" pos="0 0 {_BLADE_TIP_Z}" size="0.01 0 0" rgba="{tip_rgba}"/>
      <site name="{blade_base_name}" pos="0 0 {_BLADE_CENTER_Z - _BLADE_HALF_LENGTH}" size="0.01 0 0" rgba="{blade_base_rgba}"/>
    </body>
  </worldbody>
  <sensor>
    <framelinvel name="{tip_name}_linvel" objtype="site" objname="{tip_name}"/>
    <velocimeter name="{tip_name}_velocimeter" site="{tip_name}"/>
  </sensor>
</mujoco>
"""


def _build_saber_targets_xml_text() -> str:
    bodies = []
    for index in range(SABER_TARGET_POOL_SIZE):
        offset_x, offset_y = graveyard_offset_xy(index, SABER_TARGET_POOL_SIZE)
        bodies.append(
            f"""    <body name="saber_target_{index}" pos="{offset_x} {offset_y} {SABER_POOL_GRAVEYARD_Z}" mocap="true">
      <geom name="saber_target_geom_{index}" type="box" size="{SABER_POOL_SPHERE_RADIUS} {SABER_POOL_SPHERE_RADIUS} {SABER_POOL_SPHERE_RADIUS}" rgba="0.5 0.5 0.5 0.2" mass="0.02" contype="2" conaffinity="1" condim="4" margin="0.0" gap="1.0" material="target_pool_cube"/>
      <geom name="saber_target_geom_{index}_marker_front" type="box" pos="0 {SABER_POOL_SPHERE_RADIUS + 0.012} 0" size="{_MARKER_HALF_SPAN} 0.008 {_MARKER_HALF_SPAN}" rgba="0 0 0 0" contype="0" conaffinity="0" material="target_pool_face_left"/>
      <geom name="saber_target_geom_{index}_marker_top" type="box" pos="0 0 {SABER_POOL_SPHERE_RADIUS + 0.012}" size="{_MARKER_HALF_SPAN} {_MARKER_HALF_SPAN} 0.008" rgba="0 0 0 0" contype="0" conaffinity="0" material="target_pool_face_left"/>
      <geom name="saber_target_geom_{index}_marker_bottom" type="box" pos="0 0 -{SABER_POOL_SPHERE_RADIUS + 0.012}" size="{_MARKER_HALF_SPAN} {_MARKER_HALF_SPAN} 0.008" rgba="0 0 0 0" contype="0" conaffinity="0" material="target_pool_face_left"/>
      <geom name="saber_target_geom_{index}_marker_right" type="box" pos="{SABER_POOL_SPHERE_RADIUS + 0.012} 0 0" size="0.008 {_MARKER_HALF_SPAN} {_MARKER_HALF_SPAN}" rgba="0 0 0 0" contype="0" conaffinity="0" material="target_pool_face_left"/>
      <geom name="saber_target_geom_{index}_marker_left" type="box" pos="-{SABER_POOL_SPHERE_RADIUS + 0.012} 0 0" size="0.008 {_MARKER_HALF_SPAN} {_MARKER_HALF_SPAN}" rgba="0 0 0 0" contype="0" conaffinity="0" material="target_pool_face_left"/>
      <site name="saber_target_{index}_site" pos="0 0 0" size="0.01 0 0" rgba="1 1 1 0"/>
    </body>"""
        )
    bodies_xml = "\n".join(bodies)
    return f"""<mujoco model="saber_targets">
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <asset>
    <material name="target_pool_cube" rgba="0.60 0.60 0.60 1.0" emission="0.55" specular="0.9" shininess="0.85"/>
    <material name="target_pool_face_left" rgba="0.05 0.25 1.0 1.0" emission="0.6" specular="0.9" shininess="0.9"/>
    <material name="target_pool_face_right" rgba="1.0 0.10 0.10 1.0" emission="0.6" specular="0.9" shininess="0.9"/>
  </asset>
  <worldbody>
    <body name="saber_targets_anchor" pos="0 0 0"/>
{bodies_xml}
  </worldbody>
</mujoco>
"""


__all__ = [
    "LEFT_LIGHTSABER_XML",
    "RIGHT_LIGHTSABER_XML",
    "SABER_BODY_XML",
    "SABER_TARGETS_XML",
    "build_isolated_saber_target_spec",
    "build_left_saber_spec",
    "build_right_saber_spec",
    "build_saber_body_mjmodel_and_spec",
    "build_saber_body_spec",
    "build_saber_scene_mjmodel_and_spec",
    "build_saber_scene_spec",
    "build_saber_targets_spec",
    "ensure_saber_scene_assets",
    "load_saber_scene_spec",
    "saber_scene_asset_dir",
    "saber_scene_asset_paths",
]
