# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""CPU MuJoCo model builder for MuscleMimic-compatible MyoFullBody.

Mirrors ``MuscleMimic`` ``MyoFullBody._apply_spec_changes`` (finger removal,
mimic sites, muscle ``ctrlrange``) for the ``musclemimic_models`` full-body
MJCF — importable without MJX or ``mujoco_playground``.

Does **not** apply ``_modify_spec_for_mjx`` (MJX contact stripping / warp
budgets); that path is only used when building the JAX/Warp training env.

Finger joint/muscle name lists are shared with
:mod:`myosuite.integrations.musclemimic.bimanual_model` (same names as upstream
``MyoFullBody``).
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import warnings
import xml.etree.ElementTree as ET

import mujoco
from ml_collections import config_dict

from myosuite.integrations.musclemimic.bimanual_model import (
    FINGER_JOINT_TOKENS,
    FINGER_MUSCLE_TOKENS,
    _translate_finger_token,
)
from myosuite.terms.mimic_reward import MimicTrackingConfig

NATIVE_FULLBODY_FALLBACK_WARNING = (
    "musclemimic_models is not installed; using myo_sim's own myofullbody "
    "composition with the same mimic sites/finger removal/ctrlrange edits "
    "applied instead. This is NOT bit-exact parity with the external "
    "MuscleMimic codebase's model (github.com/amathislab/musclemimic) - "
    "checkpoints trained against the real musclemimic_models MJCF are not "
    "guaranteed to transfer. Install 'musclemimic_models>=1.0.2' or "
    "'myosuite[musclemimic]' for exact parity."
)

# Body → mimic site names (``MyoFullBody.body2sites_for_mimic``).
FULLBODY_BODY2SITES_FOR_MIMIC = {
    "pelvis": "pelvis_mimic",
    "lumbar1": "upper_body_mimic",
    "head": "head_mimic",
    "humerus_l": "left_shoulder_mimic",
    "ulna_l": "left_elbow_mimic",
    "lunate_l": "left_hand_mimic",
    "humerus_r": "right_shoulder_mimic",
    "ulna_r": "right_elbow_mimic",
    "lunate_r": "right_hand_mimic",
    "femur_l": "left_hip_mimic",
    "tibia_l": "left_knee_mimic",
    "talus_l": "left_ankle_mimic",
    "toes_l": "left_toes_mimic",
    "femur_r": "right_hip_mimic",
    "tibia_r": "right_knee_mimic",
    "talus_r": "right_ankle_mimic",
    "toes_r": "right_toes_mimic",
}


def resolve_mimic_fullbody_xml(config: config_dict.ConfigDict) -> str:
    """Return absolute path to the MuscleMimic MyoFullBody MJCF.

    When ``config.model_path`` is set, it is used. Otherwise the packaged
    ``musclemimic_models`` ``myofullbody`` entry XML is used (same as
    ``MuscleMimic`` ``MjxMyoFullBody`` / ``MyoFullBody``).

    Args:
        config: Task configuration with optional ``model_path``.

    Returns:
        Path string for :func:`mujoco.MjSpec.from_file`.

    Raises:
        ImportError: If the default asset is requested but
            ``musclemimic_models`` is not installed.
    """
    mp = getattr(config, "model_path", None)
    if mp is not None:
        return mp.as_posix() if hasattr(mp, "as_posix") else str(mp)
    try:
        from musclemimic_models import get_xml_path
    except ImportError as err:
        raise ImportError(
            "MyoFullBody parity requires the same MJCF as "
            "https://github.com/amathislab/musclemimic (package "
            "`musclemimic_models`). Install with: pip install "
            "'musclemimic_models>=1.0.2' or pip install "
            "'myosuite[musclemimic]'."
        ) from err
    return get_xml_path("myofullbody").as_posix()


def _prepare_scene_xml_keep_only_floor(scene_xml: Path) -> Path:
    """Materialize a scene XML with worldbody reduced to only floor geom.

    Args:
        scene_xml: Source scene XML path.

    Returns:
        Temporary XML path in the same directory as ``scene_xml``.
    """
    root = ET.fromstring(scene_xml.read_text(encoding="utf-8"))

    # Match MyoSuite floor appearance while keeping upstream floor physics.
    try:
        from myosuite.utils.asset_path_resolver import get_sim_asset_root

        myosuite_floor_texture = get_sim_asset_root("myo_sim") / "scene" / "floor0.png"
    except FileNotFoundError:
        myosuite_floor_texture = Path()
    use_myosuite_floor_texture = myosuite_floor_texture.is_file()

    # Remove MuscleMimic sky branding/colors and force a neutral skybox.
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    for elem in list(asset):
        if elem.tag == "texture" and (elem.get("type") or "") == "skybox":
            asset.remove(elem)
            continue
        if (
            use_myosuite_floor_texture
            and elem.tag == "texture"
            and (elem.get("name") or "") == "texfloor"
        ):
            asset.remove(elem)
            continue
        if (
            use_myosuite_floor_texture
            and elem.tag == "material"
            and (elem.get("name") or "") == "matfloor"
        ):
            asset.remove(elem)
            continue
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "sky",
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0 0 0",
            "rgb2": "0 0 0",
            "width": "100",
            "height": "100",
        },
    )
    if use_myosuite_floor_texture:
        ET.SubElement(
            asset,
            "texture",
            {
                "name": "texfloor",
                "type": "2d",
                "height": "1",
                "width": "1",
                "file": myosuite_floor_texture.as_posix(),
            },
        )
        ET.SubElement(
            asset,
            "material",
            {
                "name": "matfloor",
                "reflectance": "0.01",
                "texture": "texfloor",
                "texrepeat": "1 1",
                "texuniform": "true",
            },
        )

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    rgba = visual.find("rgba")
    if rgba is None:
        rgba = ET.SubElement(visual, "rgba")
    rgba.set("haze", "0 0 0 0")

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for child in list(worldbody):
            if child.tag != "geom":
                worldbody.remove(child)
                continue
            geom_name = (child.get("name") or "").lower()
            geom_type = (child.get("type") or "").lower()
            is_floor = geom_name == "floor" or geom_type == "plane"
            if not is_floor:
                worldbody.remove(child)
                continue
            if use_myosuite_floor_texture:
                child.set("material", "matfloor")

    with tempfile.NamedTemporaryFile(
        "w",
        dir=scene_xml.parent,
        suffix=".xml",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(ET.tostring(root, encoding="unicode"))
        return Path(tmp.name)


def _load_spec_with_floor_only_upstream_scene(xml_path: str) -> mujoco.MjSpec:
    """Load XML replacing upstream scene include with floor-only variant.

    Args:
        xml_path: Source model XML path.

    Returns:
        MuJoCo spec with the scene include rewritten to floor-only variant.
    """
    src = Path(xml_path)
    scene_temp_path: Path | None = None
    temp_model_path: Path | None = None
    try:
        model_root = ET.fromstring(src.read_text(encoding="utf-8"))
        include_rewritten = False
        for include_elem in model_root.findall("include"):
            include_file = include_elem.get("file")
            if include_file is None or "scene" not in include_file.lower():
                continue
            include_path = (src.parent / include_file).resolve()
            scene_temp_path = _prepare_scene_xml_keep_only_floor(include_path)
            include_elem.set("file", scene_temp_path.as_posix())
            include_rewritten = True
            break

        if not include_rewritten:
            return mujoco.MjSpec.from_file(src.as_posix())

        with tempfile.NamedTemporaryFile(
            "w",
            dir=src.parent,
            suffix=".xml",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            temp_model_path = Path(tmp.name)
            tmp.write(ET.tostring(model_root, encoding="unicode"))
        return mujoco.MjSpec.from_file(temp_model_path.as_posix())
    finally:
        if temp_model_path is not None and temp_model_path.exists():
            temp_model_path.unlink()
        if scene_temp_path is not None and scene_temp_path.exists():
            scene_temp_path.unlink()


def _apply_mimic_fullbody_spec_changes(
    spec: mujoco.MjSpec,
    config: config_dict.ConfigDict,
    finger_joint_tokens: tuple[str, ...],
    finger_muscle_tokens: tuple[str, ...],
) -> mujoco.MjSpec:
    """Apply the shared Mimic full-body edits (finger removal, sites, ctrlrange).

    Args:
        spec: Loaded, uncompiled full-body MjSpec.
        config: At least ``disable_fingers`` (see
            :func:`default_mimic_fullbody_config`).
        finger_joint_tokens: Finger joint names to remove when
            ``config.disable_fingers``, in the naming convention of *spec*.
        finger_muscle_tokens: Finger muscle/tendon name substrings to remove,
            in the naming convention of *spec*.

    Returns:
        The same spec, edited in place.
    """
    if config.disable_fingers:
        joints_to_remove = [j for j in spec.joints if j.name in finger_joint_tokens]
        for joint in joints_to_remove:
            spec.delete(joint)

        actuators_to_remove = [
            a for a in spec.actuators if a.name in finger_muscle_tokens
        ]
        for actuator in actuators_to_remove:
            spec.delete(actuator)

        tendons_to_remove = [
            t for t in spec.tendons if any(m in t.name for m in finger_muscle_tokens)
        ]
        for tendon in tendons_to_remove:
            spec.delete(tendon)

    for body_name, site_name in FULLBODY_BODY2SITES_FOR_MIMIC.items():
        body = spec.body(body_name)
        body.add_site(
            name=site_name,
            group=4,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.075, 0.05, 0.025],
            rgba=[1.0, 0.0, 0.0, 0.5],
            pos=[0.0, 0.0, 0.0],
        )

    for actuator in spec.actuators:
        if actuator.dyntype == mujoco.mjtDyn.mjDYN_MUSCLE:
            actuator.ctrlrange = [-1.0, 1.0]
            actuator.ctrllimited = True

    return spec


def build_mimic_fullbody_spec(
    config: config_dict.ConfigDict,
) -> tuple[mujoco.MjSpec, str]:
    """Build edited :class:`mujoco.MjSpec` for Mimic full-body (pre-compile).

    Uses the same MJCF as ``musclemimic_models`` when it's installed (or an
    explicit ``config.model_path``); falls back to a myo_sim-native full-body
    spec with the same edits applied (mimic sites, finger removal, muscle
    ctrlrange) when it isn't. The native fallback is NOT bit-exact parity
    with the external MuscleMimic codebase's model — see
    :func:`build_native_mimic_fullbody_spec` — but is usable without the
    optional dependency.

    Args:
        config: At least ``disable_fingers`` and ``sim_dt`` (see
            :func:`default_mimic_fullbody_config`).

    Returns:
        Tuple of ``MjSpec`` after edits and resolved source XML path string
        (or ``"myo_sim:myofullbody"`` when using the native fallback).
    """
    mp = getattr(config, "model_path", None)
    if mp is None:
        try:
            import musclemimic_models  # noqa: F401
        except ImportError:
            warnings.warn(NATIVE_FULLBODY_FALLBACK_WARNING, UserWarning, stacklevel=2)
            return build_native_mimic_fullbody_spec(config), "myo_sim:myofullbody"

    xml_path = resolve_mimic_fullbody_xml(config)
    # Keep only ground from the upstream MuscleMimic scene, preserving floor
    # physics/height from that source scene.
    spec = _load_spec_with_floor_only_upstream_scene(xml_path)
    _apply_mimic_fullbody_spec_changes(
        spec, config, FINGER_JOINT_TOKENS, FINGER_MUSCLE_TOKENS
    )
    return spec, xml_path


def build_native_mimic_fullbody_spec(config: config_dict.ConfigDict) -> mujoco.MjSpec:
    """Build a myo_sim-native Mimic-compatible full-body MjSpec.

    Applies the same edits as :func:`build_mimic_fullbody_spec` (mimic
    tracking sites, optional finger removal, muscle ctrlrange) to myo_sim's
    own composed ``myofullbody`` model instead of the external
    ``musclemimic_models`` MJCF. Every body :data:`FULLBODY_BODY2SITES_FOR_MIMIC`
    needs (pelvis, lumbar1, head, and the 7 limb-segment pairs) already
    exists in myo_sim's ``myofullbody`` under the same names.

    This is NOT bit-exact parity with the external MuscleMimic codebase's
    model (different muscle/mesh calibration) — checkpoints trained against
    the real ``musclemimic_models`` MJCF are not guaranteed to transfer.
    It exists so Mimic full-body envs remain usable without the optional
    ``musclemimic_models`` dependency installed.

    Args:
        config: At least ``disable_fingers`` (see
            :func:`default_mimic_fullbody_config`).

    Returns:
        Edited, uncompiled MjSpec.
    """
    import myo_sim

    spec = myo_sim.load_spec("myofullbody")
    if config.disable_fingers:
        model = spec.compile()
        import mujoco as _mujoco

        joint_pool = frozenset(
            _mujoco.mj_id2name(model, _mujoco.mjtObj.mjOBJ_JOINT, i) or ""
            for i in range(model.njnt)
        )
        actuator_pool = frozenset(
            _mujoco.mj_id2name(model, _mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            for i in range(model.nu)
        )
        joint_tokens = tuple(
            _translate_finger_token(t, joint_pool) for t in FINGER_JOINT_TOKENS
        )
        muscle_tokens = tuple(
            _translate_finger_token(t, actuator_pool) for t in FINGER_MUSCLE_TOKENS
        )
    else:
        joint_tokens = FINGER_JOINT_TOKENS
        muscle_tokens = FINGER_MUSCLE_TOKENS

    return _apply_mimic_fullbody_spec_changes(spec, config, joint_tokens, muscle_tokens)


def compile_mimic_fullbody_mjmodel(
    config: config_dict.ConfigDict,
) -> tuple[mujoco.MjModel, mujoco.MjSpec, str]:
    """Compile a CPU :class:`mujoco.MjModel` matching ``MyoFullBody`` (no MJX).

    Args:
        config: At least ``disable_fingers`` and ``sim_dt`` (see
            :func:`default_mimic_fullbody_config`).

    Returns:
        Compiled model, ``MjSpec`` after edits, and resolved XML path.
    """
    spec, xml_path = build_mimic_fullbody_spec(config)

    mj_model = spec.compile()
    # Match MuscleMimic ``MyoFullBody`` (CPU): LocoEnv only applies
    # ``timestep`` via ``model_option_conf``, not solver iterations /
    # disableflags (those are MJX tuning).
    mj_model.opt.timestep = float(config.sim_dt)
    return mj_model, spec, xml_path


def default_mimic_fullbody_config() -> config_dict.ConfigDict:
    """Defaults aligned with MuscleMimic ``MyoFullBody`` (CPU compile path).

    :func:`compile_mimic_fullbody_mjmodel` only applies ``sim_dt`` as the
    MuJoCo timestep on the host model (Loco-style CPU). Solver iterations and
    ``disableflags`` are applied when building the MJX env
    (
    :class:`~myosuite.envs.myo.backends.mjx.musclemimic_fullbody_env.MjxMuscleMimicFullbodyEnv`
    ). Extra keys (observation toggles, ``target_site_range``, ``nconmax``)
    are for that MJX task wrapper.
    """
    tracking = MimicTrackingConfig()

    return config_dict.create(
        ctrl_dt=0.01,
        sim_dt=0.002,
        num_envs=1,
        mjx_impl=None,
        norm_actions=False,
        max_episode_steps=1000,
        model_iterations=4,
        model_ls_iterations=8,
        model_disableflags=int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP),
        model_path=None,
        disable_fingers=True,
        nconmax=4096,
        enable_joint_pos_observations=True,
        enable_joint_vel_observations=True,
        enable_muscle_length_observations=False,
        enable_muscle_velocity_observations=False,
        enable_muscle_force_observations=False,
        enable_muscle_excitation_observations=False,
        enable_muscle_activation_observations=False,
        mimic_site_names=tuple(FULLBODY_BODY2SITES_FOR_MIMIC.values()),
        target_site_range=config_dict.create(
            low=(-0.50, -0.50, 0.20),
            high=(0.50, 0.50, 1.90),
        ),
        tracking_reward_scale=tracking.reward_scale,
        tracking_success_threshold=tracking.success_threshold,
    )


# Backward-compatible aliases.
resolve_musclemimic_fullbody_xml = resolve_mimic_fullbody_xml
compile_musclemimic_fullbody_mjmodel = compile_mimic_fullbody_mjmodel
default_musclemimic_fullbody_config = default_mimic_fullbody_config


__all__ = [
    "FULLBODY_BODY2SITES_FOR_MIMIC",
    "build_mimic_fullbody_spec",
    "resolve_mimic_fullbody_xml",
    "compile_mimic_fullbody_mjmodel",
    "default_mimic_fullbody_config",
    "resolve_musclemimic_fullbody_xml",
    "compile_musclemimic_fullbody_mjmodel",
    "default_musclemimic_fullbody_config",
]
