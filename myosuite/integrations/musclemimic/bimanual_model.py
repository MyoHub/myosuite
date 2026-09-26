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
import warnings
from pathlib import Path
import xml.etree.ElementTree as ET

from ml_collections import config_dict
import mujoco

from myosuite.terms.mimic_reward import MimicTrackingConfig

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


def _translate_finger_token(name: str, pool: frozenset[str]) -> str:
    """Translate a musclemimic_models finger joint/muscle name to myo_sim's convention.

    myo_sim leaves the right side unmarked (``cmc_flexion``) where
    musclemimic_models suffixes it (``cmc_flexion_r``), and uses ``_l`` where
    musclemimic_models uses ``_left`` (``FDS2_l`` vs. ``FDS2_left``).

    Raises:
        KeyError: If no variant of *name* resolves in *pool*, so a naming
            drift between myo_sim releases surfaces immediately instead of
            silently leaving a finger joint/muscle un-disabled.
    """
    if name in pool:
        return name
    if name.endswith("_r") and name[:-2] in pool:
        return name[:-2]
    if name.endswith("_left") and (name[:-5] + "_l") in pool:
        return name[:-5] + "_l"
    raise KeyError(
        f"Could not translate finger token {name!r} to myo_sim's naming convention"
    )


def apply_mimic_bimanual_spec_edits(
    spec: mujoco.MjSpec,
    config: config_dict.ConfigDict,
    body2sites: dict[str, str] | None = None,
    finger_joint_tokens: tuple[str, ...] | None = None,
    finger_muscle_tokens: tuple[str, ...] | None = None,
) -> None:
    """Apply MuscleMimic bimanual post-load edits (finger strip, mimic sites).

    Mutates *spec* in place. Shared by :func:`build_mimic_bimanual_spec` and
    ``build_myotorso_bimanual_mimic_spec``.

    Args:
        spec: Loaded bimanual (or MyoTorso + bimanual) ``MjSpec``.
        config: Same shape as :func:`default_mimic_config`.
        body2sites: Body name -> mimic site name mapping. Defaults to
            :data:`BODY2SITES_FOR_MIMIC`; override when *spec* uses a
            different naming convention (e.g. myo_sim's native composition,
            which has no ``"thorax"`` body).
        finger_joint_tokens: Defaults to :data:`FINGER_JOINT_TOKENS`.
        finger_muscle_tokens: Defaults to :data:`FINGER_MUSCLE_TOKENS`.
    """
    body2sites = BODY2SITES_FOR_MIMIC if body2sites is None else body2sites
    finger_joint_tokens = (
        FINGER_JOINT_TOKENS if finger_joint_tokens is None else finger_joint_tokens
    )
    finger_muscle_tokens = (
        FINGER_MUSCLE_TOKENS if finger_muscle_tokens is None else finger_muscle_tokens
    )

    if config.disable_fingers:
        joints_to_remove = []
        for joint in spec.joints:
            if any(token in joint.name for token in finger_joint_tokens):
                joints_to_remove.append(joint)
        for joint in joints_to_remove:
            spec.delete(joint)

        actuators_to_remove = []
        for actuator in spec.actuators:
            if any(token in actuator.name for token in finger_muscle_tokens):
                actuators_to_remove.append(actuator)
        for actuator in actuators_to_remove:
            spec.delete(actuator)

        tendons_to_remove = []
        for tendon in spec.tendons:
            if any(token in tendon.name for token in finger_muscle_tokens):
                tendons_to_remove.append(tendon)
        for tendon in tendons_to_remove:
            spec.delete(tendon)

    for body_name, site_name in body2sites.items():
        body = spec.body(body_name)
        body.add_site(
            name=site_name,
            group=4,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.075, 0.05, 0.025],
            rgba=[1.0, 0.0, 0.0, 0.5],
            pos=[0.0, 0.0, 0.0],
        )


_NATIVE_BIMANUAL_TAG = "myo_sim:myoarms"

# myo_sim's own composed torso has no "thorax" body (that's a synthetic stub
# musclemimic's bimanual arm package invents to anchor bi-articular
# pectoralis/lat wrap sites for a standalone arm); the equivalent upper-torso
# attachment point in myo_sim's real, connected torso is "torso".
_NATIVE_BODY2SITES_FOR_MIMIC = {
    ("torso" if body_name == "thorax" else body_name): site_name
    for body_name, site_name in BODY2SITES_FOR_MIMIC.items()
}

NATIVE_BIMANUAL_FALLBACK_WARNING = (
    "musclemimic_models is not installed; using myo_sim's own myoarms "
    "composition (passive torso scaffold + both arms) with the same mimic "
    "sites/finger removal/ctrlrange edits applied instead. This is NOT "
    "bit-exact parity with the external MuscleMimic codebase's model "
    "(github.com/amathislab/musclemimic) - checkpoints trained against the "
    "real musclemimic_models MJCF are not guaranteed to transfer. Install "
    "'musclemimic_models>=1.0.2' or 'myosuite[musclemimic]' for exact parity."
)


def build_native_mimic_bimanual_spec(config: config_dict.ConfigDict) -> mujoco.MjSpec:
    """Build a myo_sim-native Mimic-compatible bimanual-arms MjSpec.

    Uses myo_sim's own ``myoarms`` composition (passive anatomical torso
    scaffold + right arm + mirrored-left arm, full anatomy including the
    bi-articular pectoralis/lat wrap sites that live on the real torso)
    instead of musclemimic_models' ``myoarm_bimanual_body.xml`` (which
    invents a synthetic "thorax" stub to provide those same wrap sites for a
    standalone arm).

    This is NOT bit-exact parity with the external MuscleMimic codebase's
    model (different muscle/mesh calibration) - checkpoints trained against
    the real ``musclemimic_models`` MJCF are not guaranteed to transfer.

    Args:
        config: Same shape as :func:`default_mimic_config`.

    Returns:
        Edited, uncompiled MjSpec.
    """
    import myo_sim

    spec = myo_sim.load_spec("myoarms")

    if config.disable_fingers:
        model = spec.compile()
        joint_pool = frozenset(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
            for i in range(model.njnt)
        )
        actuator_pool = frozenset(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
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

    apply_mimic_bimanual_spec_edits(
        spec,
        config,
        body2sites=_NATIVE_BODY2SITES_FOR_MIMIC,
        finger_joint_tokens=joint_tokens,
        finger_muscle_tokens=muscle_tokens,
    )
    return spec


def build_mimic_bimanual_spec(
    config: config_dict.ConfigDict,
) -> tuple[mujoco.MjSpec, str]:
    """Build edited :class:`mujoco.MjSpec` for Mimic bimanual (pre-compile).

    Uses the same MJCF as ``musclemimic_models`` when it's installed (or an
    explicit ``config.model_path``); falls back to a myo_sim-native spec
    (see :func:`build_native_mimic_bimanual_spec`) when it isn't.

    Used by mjlab ``EntityCfg(spec_fn=...)`` and by
    :func:`compile_mimic_bimanual_mjmodel`.

    Args:
        config: Same shape as :func:`default_mimic_config`.

    Returns:
        Tuple of ``MjSpec`` after edits and resolved source XML path string,
        or ``"myo_sim:myoarms"`` when using the native fallback.
    """
    mp = getattr(config, "model_path", None)
    if mp is None:
        try:
            import musclemimic_models  # noqa: F401
        except ImportError:
            warnings.warn(NATIVE_BIMANUAL_FALLBACK_WARNING, UserWarning, stacklevel=2)
            return build_native_mimic_bimanual_spec(config), _NATIVE_BIMANUAL_TAG

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
    """Resolve the canonical MyoSuite scene include path (pip or submodule)."""
    from myosuite.utils.asset_path_resolver import get_sim_asset_root

    return get_sim_asset_root("myo_sim") / "scene" / "myosuite_scene.xml"


def _prepare_scene_xml_for_include(scene_xml: Path) -> Path:
    """Materialize a scene XML with normalized local mesh/texture paths.

    The bundled MyoSuite scene XML can contain paths like
    ``../myo_sim/scene/myosuite_logo.msh``. When the scene itself already lives
    under ``.../myo_sim/scene/``, these expand to a non-existent duplicated
    segment (``.../myo_sim/myo_sim/scene``). Normalize to local filenames.
    """
    root = ET.fromstring(scene_xml.read_text(encoding="utf-8"))

    # Determine effective mesh/texture base dirs from the scene compiler element
    # (MuJoCo resolves asset file= paths relative to meshdir/texturedir, which
    # may be set to ".." or another relative path within the scene XML).
    scene_dir = scene_xml.parent
    mesh_base = scene_dir
    tex_base = scene_dir
    for compiler_elem in root.iter("compiler"):
        for attr, store in (("meshdir", "mesh_base"), ("texturedir", "tex_base")):
            v = compiler_elem.get(attr)
            if v:
                p = Path(v)
                resolved = p if p.is_absolute() else (scene_dir / p).resolve()
                if attr == "meshdir":
                    mesh_base = resolved
                else:
                    tex_base = resolved

    # Absolutize all relative file= paths so the temp copy (written to the
    # same directory) resolves them regardless of sub-directory prefixes.
    for elem in root.findall(".//*[@file]"):
        file_attr = elem.get("file")
        if not file_attr or Path(file_attr).is_absolute():
            continue
        base = (
            mesh_base
            if elem.tag in ("mesh",)
            else tex_base
            if elem.tag in ("texture",)
            else scene_dir
        )
        candidate = (base / file_attr).resolve()
        if candidate.exists():
            elem.set("file", str(candidate))

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
