# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""CPU MuJoCo model builder for MuscleMimic-compatible bimanual tasks.

This module is importable **without** MJX or ``mujoco_playground`` so CPU tests
and parity tooling can compile the same :class:`mujoco.MjModel` as
https://github.com/amathislab/musclemimic ``MjxMyoBimanualArm``.

The MJX task wrapper lives at
:mod:`myosuite.envs.myo.backends.mjx.musclemimic_env`.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

from etils import epath
from ml_collections import config_dict
import mujoco

from myosuite.integrations.musclemimic.mimic_common import MimicTrackingConfig

BODY2SITES_FOR_MIMIC = {
    "thorax": "upper_body_mimic",
    "humerus_l": "left_shoulder_mimic",
    "ulna_l": "left_elbow_mimic",
    "lunate_l": "left_hand_mimic",
    "humerus_r": "right_shoulder_mimic",
    "ulna_r": "right_elbow_mimic",
    "lunate_r": "right_hand_mimic",
}

FINGER_JOINT_TOKENS = (
    "cmc_flexion_r",
    "cmc_abduction_r",
    "mp_flexion_r",
    "ip_flexion_r",
    "mcp2_flexion_r",
    "mcp2_abduction_r",
    "mcp3_flexion_r",
    "mcp3_abduction_r",
    "mcp4_flexion_r",
    "mcp4_abduction_r",
    "mcp5_flexion_r",
    "mcp5_abduction_r",
    "md2_flexion_r",
    "md3_flexion_r",
    "md4_flexion_r",
    "md5_flexion_r",
    "pm2_flexion_r",
    "pm3_flexion_r",
    "pm4_flexion_r",
    "pm5_flexion_r",
    "cmc_flexion_l",
    "cmc_abduction_l",
    "mp_flexion_l",
    "ip_flexion_l",
    "mcp2_flexion_l",
    "mcp2_abduction_l",
    "mcp3_flexion_l",
    "mcp3_abduction_l",
    "mcp4_flexion_l",
    "mcp4_abduction_l",
    "mcp5_flexion_l",
    "mcp5_abduction_l",
    "md2_flexion_l",
    "md3_flexion_l",
    "md4_flexion_l",
    "md5_flexion_l",
    "pm2_flexion_l",
    "pm3_flexion_l",
    "pm4_flexion_l",
    "pm5_flexion_l",
)

FINGER_MUSCLE_TOKENS = (
    "FDS2",
    "FDS3",
    "FDS4",
    "FDS5",
    "FDP2",
    "FDP3",
    "FDP4",
    "FDP5",
    "EDC2",
    "EDC3",
    "EDC4",
    "EDC5",
    "EDM",
    "EIP",
    "EPL",
    "EPB",
    "FPL",
    "APL",
    "OP",
    "RI2",
    "RI3",
    "RI4",
    "RI5",
    "LU_RB2",
    "LU_RB3",
    "LU_RB4",
    "LU_RB5",
    "UI_UB2",
    "UI_UB3",
    "UI_UB4",
    "UI_UB5",
    "FDS2_left",
    "FDS3_left",
    "FDS4_left",
    "FDS5_left",
    "FDP2_left",
    "FDP3_left",
    "FDP4_left",
    "FDP5_left",
    "EDC2_left",
    "EDC3_left",
    "EDC4_left",
    "EDC5_left",
    "EDM_left",
    "EIP_left",
    "EPL_left",
    "EPB_left",
    "FPL_left",
    "APL_left",
    "OP_left",
    "RI2_left",
    "RI3_left",
    "RI4_left",
    "RI5_left",
    "LU_RB2_left",
    "LU_RB3_left",
    "LU_RB4_left",
    "LU_RB5_left",
    "UI_UB2_left",
    "UI_UB3_left",
    "UI_UB4_left",
    "UI_UB5_left",
)


def resolve_mimic_bimanual_xml(config: config_dict.ConfigDict) -> str:
    """Return absolute path to the MuscleMimic bimanual MJCF.

    When ``config.model_path`` is set, it is used (for tests or custom
    assets). Otherwise the same entry MJCF as
    ``amathislab/musclemimic`` is loaded via the ``musclemimic_models``
    distribution package.

    Args:
        config: Task configuration.

    Returns:
        Filesystem path suitable for :func:`mujoco.MjSpec.from_file`.

    Raises:
        ImportError: If the default upstream MJCF is requested but
            ``musclemimic_models`` is not installed.
    """
    mp = getattr(config, "model_path", None)
    if mp is not None:
        return mp.as_posix() if hasattr(mp, "as_posix") else str(mp)
    try:
        from musclemimic_models import get_xml_path
    except ImportError as err:
        raise ImportError(
            "MjxMimicBimanual-v0 needs the same bimanual MJCF as "
            "https://github.com/amathislab/musclemimic (package "
            "`musclemimic_models`). Install with: pip install "
            "'musclemimic_models>=1.0.2' or pip install "
            "'myosuite[musclemimic]'."
        ) from err
    return get_xml_path("bimanual").as_posix()


def apply_mimic_bimanual_spec_edits(
    spec: mujoco.MjSpec, config: config_dict.ConfigDict
) -> None:
    """Apply MuscleMimic bimanual post-load edits (finger strip, mimic sites).

    Mutates *spec* in place. Shared by :func:`build_mimic_bimanual_spec` and
    ``build_myotorso_bimanual_mimic_spec``.

    Args:
        spec: Loaded bimanual (or MyoTorso + bimanual) ``MjSpec``.
        config: Same shape as :func:`default_mimic_config`.
    """
    if config.disable_fingers:
        joints_to_remove = []
        for joint in spec.joints:
            if any(token in joint.name for token in FINGER_JOINT_TOKENS):
                joints_to_remove.append(joint)
        for joint in joints_to_remove:
            spec.delete(joint)

        actuators_to_remove = []
        for actuator in spec.actuators:
            if any(token in actuator.name for token in FINGER_MUSCLE_TOKENS):
                actuators_to_remove.append(actuator)
        for actuator in actuators_to_remove:
            spec.delete(actuator)

        tendons_to_remove = []
        for tendon in spec.tendons:
            if any(token in tendon.name for token in FINGER_MUSCLE_TOKENS):
                tendons_to_remove.append(tendon)
        for tendon in tendons_to_remove:
            spec.delete(tendon)

    for body_name, site_name in BODY2SITES_FOR_MIMIC.items():
        body = spec.body(body_name)
        body.add_site(
            name=site_name,
            group=4,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.075, 0.05, 0.025],
            rgba=[1.0, 0.0, 0.0, 0.5],
            pos=[0.0, 0.0, 0.0],
        )


def build_mimic_bimanual_spec(
    config: config_dict.ConfigDict,
) -> tuple[mujoco.MjSpec, str]:
    """Build edited :class:`mujoco.MjSpec` for Mimic bimanual (pre-compile).

    Used by mjlab ``EntityCfg(spec_fn=...)`` and by
    :func:`compile_mimic_bimanual_mjmodel`.

    Args:
        config: Same shape as :func:`default_mimic_config`.

    Returns:
        Tuple of ``MjSpec`` after edits and resolved source XML path string.
    """
    xml_path = resolve_mimic_bimanual_xml(config)
    spec = _load_spec_with_default_scene(xml_path)
    apply_mimic_bimanual_spec_edits(spec, config)
    return spec, xml_path


def compile_mimic_bimanual_mjmodel(
    config: config_dict.ConfigDict,
) -> tuple[mujoco.MjModel, mujoco.MjSpec, str]:
    """Build the CPU :class:`mujoco.MjModel` for MuscleMimic bimanual parity.

    Applies the same ``MjSpec`` edits as
    :class:`~myosuite.envs.myo.backends.mjx.musclemimic_env.MjxMuscleMimicIOEnv`
    (finger pruning, mimic sites, solver options). Does not require MJX or
    mujoco_playground; safe for unit tests and parity checks.

    Args:
        config: Same shape as :func:`default_mimic_config`.

    Returns:
        Tuple of compiled model, the post-edit spec, and resolved XML path.
    """
    spec, xml_path = build_mimic_bimanual_spec(config)

    mj_model = spec.compile()
    mj_model.opt.timestep = float(config.sim_dt)
    mj_model.opt.iterations = int(config.model_iterations)
    mj_model.opt.ls_iterations = int(config.model_ls_iterations)
    mj_model.opt.disableflags = int(config.model_disableflags)
    return mj_model, spec, xml_path


def _default_myosuite_scene_xml() -> Path:
    """Resolve the canonical MyoSuite scene include path."""
    return (
        Path(epath.resource_path("myosuite"))
        / "simhive"
        / "myo_sim"
        / "scene"
        / "myosuite_scene.xml"
    ).resolve()


def _prepare_scene_xml_for_include(scene_xml: Path) -> Path:
    """Materialize a scene XML with normalized local mesh/texture paths.

    The bundled MyoSuite scene XML can contain paths like
    ``../myo_sim/scene/myosuite_logo.msh``. When the scene itself already lives
    under ``.../myo_sim/scene/``, these expand to a non-existent duplicated
    segment (``.../myo_sim/myo_sim/scene``). Normalize to local filenames.
    """
    root = ET.fromstring(scene_xml.read_text(encoding="utf-8"))

    # Normalize legacy path prefixes so scene-local assets resolve correctly.
    for elem in root.findall(".//*[@file]"):
        file_attr = elem.get("file")
        if file_attr:
            elem.set("file", file_attr.replace("../myo_sim/scene/", ""))

    # Keep only the floor geom from the default scene worldbody.
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

    with tempfile.NamedTemporaryFile(
        "w",
        dir=scene_xml.parent,
        suffix=".xml",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(ET.tostring(root, encoding="unicode"))
        return Path(tmp.name)


def _load_spec_with_default_scene(xml_path: str) -> mujoco.MjSpec:
    """Load XML while replacing custom scene include with MyoSuite default."""
    if os.environ.get("MYOSUITE_MIMIC_STRICT_UPSTREAM_SCENE", "0") == "1":
        # Match upstream MuscleMimic scene includes exactly.
        return mujoco.MjSpec.from_file(Path(xml_path).as_posix())
    src = Path(xml_path)
    scene_xml = _prepare_scene_xml_for_include(_default_myosuite_scene_xml())
    replaced = False
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=src.parent,
            suffix=".xml",
            delete=False,
        ) as tmp:
            temp_path = Path(tmp.name)
            for line in src.read_text().splitlines():
                lowered = line.lower()
                if "<include" in lowered and "scene" in lowered:
                    if not replaced:
                        include_line = f'<include file="{scene_xml.as_posix()}"/>\n'
                        tmp.write(include_line)
                        replaced = True
                    continue
                tmp.write(f"{line}\n")
        target = temp_path if replaced else src
        return mujoco.MjSpec.from_file(target.as_posix())
    finally:
        if replaced and temp_path is not None and temp_path.exists():
            temp_path.unlink()
        if scene_xml.name != _default_myosuite_scene_xml().name and scene_xml.exists():
            scene_xml.unlink()


def default_mimic_config() -> config_dict.ConfigDict:
    """Config aligned with ``MjxMyoBimanualArm`` defaults in musclemimic."""
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
        # None: load upstream MuscleMimic MJCF via musclemimic_models (see
        # :func:`resolve_mimic_bimanual_xml`).
        model_path=None,
        disable_fingers=True,
        enable_joint_pos_observations=True,
        enable_joint_vel_observations=True,
        enable_muscle_length_observations=False,
        enable_muscle_velocity_observations=False,
        enable_muscle_force_observations=False,
        enable_muscle_excitation_observations=False,
        enable_muscle_activation_observations=False,
        mimic_site_names=tuple(BODY2SITES_FOR_MIMIC.values()),
        target_site_range=config_dict.create(
            low=(-0.30, -0.85, 0.95),
            high=(0.30, -0.25, 1.45),
        ),
        tracking_reward_scale=tracking.reward_scale,
        tracking_success_threshold=tracking.success_threshold,
    )


# Backward-compatible aliases.
resolve_musclemimic_bimanual_xml = resolve_mimic_bimanual_xml
compile_musclemimic_bimanual_mjmodel = compile_mimic_bimanual_mjmodel
default_musclemimic_config = default_mimic_config


__all__ = [
    "BODY2SITES_FOR_MIMIC",
    "FINGER_JOINT_TOKENS",
    "FINGER_MUSCLE_TOKENS",
    "apply_mimic_bimanual_spec_edits",
    "build_mimic_bimanual_spec",
    "resolve_mimic_bimanual_xml",
    "compile_mimic_bimanual_mjmodel",
    "default_mimic_config",
    "resolve_musclemimic_bimanual_xml",
    "compile_musclemimic_bimanual_mjmodel",
    "default_musclemimic_config",
]
