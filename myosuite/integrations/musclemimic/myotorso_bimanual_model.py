# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MyoTorso spine + MuscleMimic bimanual upper body (dual arms).

Uses the same **torso assets**, **head assets**, **leg chain**, and **leg
assets** as MuscleMimic ``myofullbody.xml`` (all under the
``musclemimic_models`` ``model/`` tree), plus ``myotorso_bimanual_chain.xml``
and bimanual arm assets.
This keeps pelvis / hip framing consistent with the Mimic full-body MJCF
(MyoSuite's leg/torso files can differ, e.g. pelvis orientation).

``myoarm_assets.xml`` in the host is swapped for ``myoarm_bimanual_assets`` so
mesh names do not collide with the bimanual arm package.

Legs use ``myolegs_chain`` geoms / bodies but **all leg DOFs are stripped**
(``<joint/>`` lines removed) so shanks/feet are rigidly welded to the pelvis.
The temp host drops table-tennis ``full_body`` *xy* offset and **π yaw** so
the figure matches Mimic / saber world axes (incoming targets advance along
``+Y``). It also removes ``pelvis_x`` / ``pelvis_y`` slide joints and their
actuators (no horizontal root translation).

Requires ``musclemimic_models`` (same as :func:`resolve_mimic_bimanual_xml`).
"""

from __future__ import annotations

import re
import tempfile
import warnings
from pathlib import Path

from ml_collections import config_dict
import mujoco

from myosuite.integrations.musclemimic.bimanual_model import (
    BODY2SITES_FOR_MIMIC,
    FINGER_JOINT_TOKENS,
    FINGER_MUSCLE_TOKENS,
    _translate_finger_token,
    apply_mimic_bimanual_spec_edits,
    default_mimic_config,
)

# ``import myosuite`` must not run at module import time: ``myosuite`` executes
# ``register_all_envs()`` which imports challenge tasks (e.g. saber_vs) that
# import this module again → partially initialized circular import on Colab/notebooks.
_MYOSUITE_PKG_ROOT = Path(__file__).resolve().parents[2]

_MYOTORSO_BIMANUAL_TAG = "myotorso_bimanual_mimic"

# Canonical MyoLegs chain include (same path as ``myotorso_arm_chain_host``).
_MYOLEGS_CHAIN_HOST_INCLUDE = (
    '<include file="../myo_sim/leg/assets/myolegs_chain.xml"/>'
)

_TORSO_ASSETS_HOST_INCLUDE = (
    '<include file="../myo_sim/torso/assets/myotorso_assets.xml"/>\n'
    '  <include file="../myo_sim/torso/assets/myotorso_muscle.xml"/>\n'
    '  <include file="../myo_sim/torso/assets/myotorso_tendon.xml"/>'
)
_HEAD_ASSETS_HOST_INCLUDE = (
    '<include file="../myo_sim/head/assets/myohead_simple_assets.xml"/>'
)


def _musclemimic_model_file(*relative_parts: str) -> Path:
    """Resolve a file under the packaged MuscleMimic ``model/`` directory."""
    root = _musclemimic_model_root()
    path = root.joinpath(*relative_parts)
    if not path.is_file():
        raise RuntimeError(f"Missing MuscleMimic model file: {path}")
    return path


def _myolegs_chain_source_xml() -> Path:
    """Path to mmc ``myolegs_chain.xml`` (same legs as ``myofullbody``)."""
    return _musclemimic_model_file("leg", "assets", "myolegs_chain.xml")


def _bones_only_xml_text(src: Path) -> str:
    """MJCF text from *src* with every ``<joint .../>`` line removed."""
    text = src.read_text(encoding="utf-8")
    stripped, n = re.subn(
        r"^\s*<joint\b[^>]*/>\s*\n",
        "",
        text,
        flags=re.MULTILINE,
    )
    if n < 20:
        raise RuntimeError(
            f"Expected many <joint/> lines in {src}, removed only {n}; "
            f"check {src.name} format."
        )
    return stripped


def _myolegs_chain_bones_only_xml_text() -> str:
    """``myolegs_chain`` MJCF with every ``<joint .../>`` line removed."""
    return _bones_only_xml_text(_myolegs_chain_source_xml())


def _myolegs_assets_bones_only_xml_text() -> str:
    """Return mmc ``myolegs_assets`` text without the ``<equality>`` block."""
    src = _musclemimic_model_file("leg", "assets", "myolegs_assets.xml")
    text = src.read_text(encoding="utf-8")
    stripped, n = re.subn(
        r"\s*<equality>.*?</equality>\s*",
        "\n",
        text,
        count=1,
        flags=re.DOTALL,
    )
    if n != 1:
        raise RuntimeError(
            f"Expected one <equality> block in {src}, matched {n}; "
            "update _myolegs_assets_bones_only_xml_text()."
        )
    return stripped


def _musclemimic_model_root() -> Path:
    """Return ``musclemimic_models/model`` directory root."""
    try:
        from musclemimic_models import get_xml_path
    except ImportError as err:
        raise ImportError(
            "MyoTorso + bimanual needs musclemimic_models "
            "(pip install 'musclemimic_models>=1.0.2' or "
            "pip install 'myosuite[musclemimic]')."
        ) from err
    # myofullbody.xml lives in model/body/
    return Path(get_xml_path("myofullbody")).resolve().parent.parent


def _myotorso_bimanual_chain_xml() -> Path:
    """Path to mmc ``myotorso_bimanual_chain.xml``."""
    return _musclemimic_model_file("torso", "assets", "myotorso_bimanual_chain.xml")


def _myoarm_bimanual_assets_xml() -> Path:
    """Path to ``myoarm_bimanual_assets.xml``."""
    return _musclemimic_model_file("arm", "assets", "myoarm_bimanual_assets.xml")


def _materialize_myotorso_bimanual_host_xml() -> tuple[Path, Path, Path]:
    """Write temp host MJCF and bones-only leg chain; return both paths.

    Same layout as ``myotorso_arm_chain_host.xml``, but torso/head/leg asset
    includes point at ``musclemimic_models`` (same stack as
    ``myofullbody.xml``), plus bimanual arm assets and
    ``myotorso_bimanual_chain.xml``. Legs use a
    temp copy of mmc ``myolegs_chain.xml`` with all leg ``<joint/>`` lines
    removed.  Also strips ``pelvis_x`` / ``pelvis_y`` and their actuators, and
    sets ``full_body`` to ``pos="0 0 0.95"`` without the table-tennis π yaw.

    Returns:
        ``(tmp_host_path, tmp_legs_path, tmp_leg_assets_path)``. Caller deletes
        all three after load.

    Raises:
        ImportError: If ``musclemimic_models`` is not installed.
        RuntimeError: If expected upstream files are missing.
    """
    chain_abs = _myotorso_bimanual_chain_xml().as_posix()
    assets_abs = _myoarm_bimanual_assets_xml().as_posix()

    root = _MYOSUITE_PKG_ROOT
    arm_dir = root / "envs" / "myo" / "assets" / "arm"
    host_tpl = arm_dir / "myotorso_arm_chain_host.xml"
    host_src = host_tpl.read_text(encoding="utf-8")
    # Arm asset include — swap bundled arm assets for musclemimic bimanual assets.
    _ARM_ASSETS_INC = '<include file="../arm/assets/myoarm_assets.xml"/>'
    chain_inc = '<include file="../torso/assets/myotorso_arm_chain.xml"/>'
    host_src = host_src.replace(_ARM_ASSETS_INC, f'<include file="{assets_abs}"/>', 1)
    host_src = host_src.replace(
        chain_inc,
        f'<include file="{chain_abs}"/>',
        1,
    )

    # Resolve compiler meshdir/texturedir and quad-scene include to pip path.
    from myosuite.utils.asset_path_resolver import get_sim_asset_root

    _myo_sim_root = get_sim_asset_root("myo_sim").as_posix()
    host_src = host_src.replace('meshdir="../myo_sim"', f'meshdir="{_myo_sim_root}"', 1)
    host_src = host_src.replace(
        'texturedir="../myo_sim"', f'texturedir="{_myo_sim_root}"', 1
    )
    _quad_scene_inc = '<include file="../myo_sim/scene/myosuite_quad.xml"/>'
    _quad_scene_abs = (
        get_sim_asset_root("myo_sim") / "scene" / "myosuite_quad.xml"
    ).as_posix()
    host_src = host_src.replace(
        _quad_scene_inc, f'<include file="{_quad_scene_abs}"/>', 1
    )

    torso_assets_abs = _musclemimic_model_file(
        "torso", "assets", "myotorso_assets.xml"
    ).as_posix()
    head_assets_abs = _musclemimic_model_file(
        "head", "assets", "myohead_simple_assets.xml"
    ).as_posix()
    host_src = host_src.replace(
        _TORSO_ASSETS_HOST_INCLUDE,
        f'<include file="{torso_assets_abs}"/>',
        1,
    )
    host_src = host_src.replace(
        _HEAD_ASSETS_HOST_INCLUDE,
        f'<include file="{head_assets_abs}"/>',
        1,
    )

    leg_assets_dir = _musclemimic_model_file(
        "leg", "assets", "myolegs_assets.xml"
    ).parent
    tmp_legs = leg_assets_dir / "_tmp_myolegs_chain_bones_only.xml"
    tmp_legs.write_text(_myolegs_chain_bones_only_xml_text(), encoding="utf-8")
    tmp_leg_assets = leg_assets_dir / "_tmp_myolegs_assets_bones_only.xml"
    tmp_leg_assets.write_text(
        _myolegs_assets_bones_only_xml_text(),
        encoding="utf-8",
    )
    legs_assets_inc = '<include file="../myo_sim/leg/assets/myolegs_assets.xml"/>'
    host_src = host_src.replace(
        legs_assets_inc,
        f'<include file="{tmp_leg_assets.resolve().as_posix()}"/>',
        1,
    )
    host_src = host_src.replace(
        _MYOLEGS_CHAIN_HOST_INCLUDE,
        f'<include file="{tmp_legs.resolve().as_posix()}"/>',
        1,
    )

    # ``myotorso_arm_chain_host`` offsets ``full_body`` for the table-tennis
    # court (x=1.6). MuscleMimic / saber scenes assume a near-origin layout so
    # target portals and demo cameras match the packaged bimanual-only MJCF.
    _HOST_FULL_BODY_OFFSET = '<body name="full_body" pos="1.6 0 0.95" euler="0 0 3.14">'
    _MIMIC_FULL_BODY_ORIGIN = '<body name="full_body" pos="0 0 0.95">'
    if _HOST_FULL_BODY_OFFSET not in host_src:
        raise RuntimeError(
            "Expected table-tennis full_body pose in "
            "myotorso_arm_chain_host.xml; update _HOST_FULL_BODY_OFFSET."
        )
    host_src = host_src.replace(
        _HOST_FULL_BODY_OFFSET,
        _MIMIC_FULL_BODY_ORIGIN,
        1,
    )

    _PELVIS_SLIDE_LINES = (
        '      <joint name="pelvis_x" class="pelvis_move" axis="1 0 0" '
        'range="-1 -0.05"/>\n'
        '      <joint name="pelvis_y" class="pelvis_move" axis="0 1 0" '
        'range="-1 1"/>\n'
    )
    if _PELVIS_SLIDE_LINES not in host_src:
        raise RuntimeError(
            "Expected pelvis_x / pelvis_y slide joints in host template; "
            "update _PELVIS_SLIDE_LINES."
        )
    host_src = host_src.replace(_PELVIS_SLIDE_LINES, "", 1)

    _PELVIS_ACTUATOR_BLOCK = (
        "  <actuator>\n"
        '    <position class="pelvis_move" name="pelvis_x" joint="pelvis_x" '
        'ctrlrange="-1 0.05"/>\n'
        '    <position class="pelvis_move" name="pelvis_y" joint="pelvis_y" '
        'ctrlrange="-1 1"/>\n'
        "  </actuator>\n"
    )
    if _PELVIS_ACTUATOR_BLOCK not in host_src:
        raise RuntimeError(
            "Expected pelvis actuator block in host template; "
            "update _PELVIS_ACTUATOR_BLOCK."
        )
    host_src = host_src.replace(_PELVIS_ACTUATOR_BLOCK, "", 1)

    tmp_host = tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_myotorso_bimanual_host.xml",
        dir=str(arm_dir),
        delete=False,
        encoding="utf-8",
    )
    tmp_host.write(host_src)
    tmp_host.flush()
    tmp_host_path = Path(tmp_host.name)
    tmp_host.close()

    return tmp_host_path, tmp_legs, tmp_leg_assets


_NATIVE_BIMANUAL_TAG = "myo_sim:myotorso_arms"

# myo_sim's own composed torso has no "thorax" body (that's a synthetic stub
# musclemimic's bimanual arm package invents to anchor bi-articular
# pectoralis/lat wrap sites for a standalone arm); the equivalent upper-torso
# attachment point in myo_sim's real, connected torso is "torso".
_NATIVE_BODY2SITES_FOR_MIMIC = {
    ("torso" if body_name == "thorax" else body_name): site_name
    for body_name, site_name in BODY2SITES_FOR_MIMIC.items()
}

NATIVE_BIMANUAL_FALLBACK_WARNING = (
    "musclemimic_models is not installed; using myo_sim's own myotorso_arms "
    "composition (torso + both arms, bones-only legs) with the same mimic "
    "sites/finger removal/ctrlrange edits applied instead. This is NOT "
    "bit-exact parity with the external MuscleMimic codebase's model "
    "(github.com/amathislab/musclemimic) - checkpoints trained against the "
    "real musclemimic_models MJCF are not guaranteed to transfer. Install "
    "'musclemimic_models>=1.0.2' or 'myosuite[musclemimic]' for exact parity."
)


def build_native_myotorso_bimanual_mimic_spec(
    config: config_dict.ConfigDict,
) -> mujoco.MjSpec:
    """Build a myo_sim-native MyoTorso + bimanual-arms MjSpec.

    Uses myo_sim's own ``myotorso_arms`` composition (torso + right arm +
    mirrored-left arm, full anatomy including the bi-articular pectoralis/lat
    wrap sites that live on the real torso) instead of musclemimic_models'
    ``myotorso_bimanual_chain.xml`` + ``myoarm_bimanual_body.xml`` (which
    invent a synthetic "thorax" stub to provide those same wrap sites for a
    standalone arm). Legs are myo_sim's own ``leg/assets/myolegs_chain.xml``
    with every ``<joint/>`` stripped (bones-only, matching the external
    package's rigidly-welded legs), attached under the root body.

    This is NOT bit-exact parity with the external MuscleMimic codebase's
    model (different muscle/mesh calibration) - checkpoints trained against
    the real ``musclemimic_models`` MJCF are not guaranteed to transfer.

    Args:
        config: Same shape as :func:`default_mimic_config`.

    Returns:
        Edited, uncompiled MjSpec.
    """
    import myo_sim

    spec = myo_sim.load_spec("myotorso_arms")

    legs_chain_path = myo_sim.get_path("leg/assets/myolegs_chain.xml")
    legs_body_text = _bones_only_xml_text(legs_chain_path)
    # myolegs_chain.xml is a <mujocoinclude> fragment (bodies only, meant to
    # be <include>d into a host); strip its wrapper tag and re-wrap as a
    # standalone compilable model so it can be attached via spec.attach().
    legs_body_text = re.sub(r"^<mujocoinclude[^>]*>", "", legs_body_text, count=1)
    legs_body_text = re.sub(r"</mujocoinclude>\s*$", "", legs_body_text, count=1)
    legs_assets_path = myo_sim.get_path("leg/assets/myolegs_assets.xml")
    legs_xml_text = (
        f'<mujoco model="myolegs_bones_only">'
        f'<compiler meshdir="{myo_sim.MODELS_DIR}" '
        f'texturedir="{myo_sim.MODELS_DIR}"/>'
        f'<include file="{legs_assets_path}"/>'
        f"<worldbody>{legs_body_text}</worldbody></mujoco>"
    )
    legs_spec = mujoco.MjSpec.from_string(legs_xml_text)
    full_body = spec.body("Full Body")
    legs_frame = full_body.add_frame(name="legs_attach")
    spec.attach(legs_spec, prefix="", suffix="", frame=legs_frame)

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


def build_myotorso_bimanual_mimic_spec(
    config: config_dict.ConfigDict,
) -> tuple[mujoco.MjSpec, str]:
    """Build ``MjSpec`` for MyoTorso + MuscleMimic bimanual arms (pre-compile).

    Uses the same MJCF as ``musclemimic_models`` when it's installed (or an
    explicit ``config.model_path``); falls back to a myo_sim-native spec
    (see :func:`build_native_myotorso_bimanual_mimic_spec`) when it isn't.

    Args:
        config: Same shape as :func:`default_mimic_config`.

    Returns:
        Tuple of edited ``MjSpec`` and the tag string
        ``myotorso_bimanual_mimic`` (for logging parity with
        :func:`build_mimic_bimanual_spec`, which returns a filesystem path),
        or ``"myo_sim:myotorso_arms"`` when using the native fallback.
    """
    mp = getattr(config, "model_path", None)
    if mp is None:
        try:
            import musclemimic_models  # noqa: F401
        except ImportError:
            warnings.warn(NATIVE_BIMANUAL_FALLBACK_WARNING, UserWarning, stacklevel=2)
            return build_native_myotorso_bimanual_mimic_spec(
                config
            ), _NATIVE_BIMANUAL_TAG

    tmp_host, tmp_legs, tmp_leg_assets = _materialize_myotorso_bimanual_host_xml()
    try:
        spec = mujoco.MjSpec.from_file(tmp_host.as_posix())
    finally:
        tmp_host.unlink(missing_ok=True)
        tmp_legs.unlink(missing_ok=True)
        tmp_leg_assets.unlink(missing_ok=True)

    apply_mimic_bimanual_spec_edits(spec, config)
    return spec, _MYOTORSO_BIMANUAL_TAG


def compile_myotorso_bimanual_mimic_mjmodel(
    config: config_dict.ConfigDict,
) -> tuple[mujoco.MjModel, mujoco.MjSpec, str]:
    """Compile CPU ``MjModel`` for MyoTorso + bimanual (Mimic edits).

    Applies the same solver option overrides as
    :func:`compile_mimic_bimanual_mjmodel`.

    Args:
        config: Same shape as :func:`default_mimic_config`.

    Returns:
        Tuple of compiled model, post-edit spec, and tag string.
    """
    spec, tag = build_myotorso_bimanual_mimic_spec(config)
    mj_model = spec.compile()
    mj_model.opt.timestep = float(config.sim_dt)
    mj_model.opt.iterations = int(config.model_iterations)
    mj_model.opt.ls_iterations = int(config.model_ls_iterations)
    mj_model.opt.disableflags = int(config.model_disableflags)
    return mj_model, spec, tag


def default_myotorso_bimanual_mimic_config() -> config_dict.ConfigDict:
    """Return Mimic config defaults (same as :func:`default_mimic_config`)."""
    return default_mimic_config()


def save_myotorso_bimanual_mimic_xml(
    dest: Path | None = None,
    *,
    config: config_dict.ConfigDict | None = None,
) -> Path:
    """Write a monolithic MJCF for MyoTorso + bimanual (Mimic edits) to disk.

    Uses :meth:`mujoco.MjSpec.to_xml` on the composite spec, then rewrites
    ``meshdir``/``texturedir`` to ``.`` so mesh/texture ``file`` attributes
    (baked in as absolute paths by ``to_xml()``) resolve relative to the
    output directory.

    Args:
        dest: Output path. Default: an OS temp directory
            (``<tempdir>/myosuite_musclemimic/myotorso_bimanual_mimic.xml``) —
            never the read-only installed myo_sim pip package directory.
        config: Mimic config; default is :func:`default_mimic_config`.

    Returns:
        Path to the written file.

    Raises:
        ImportError: If ``musclemimic_models`` is not installed.
    """
    cfg = config if config is not None else default_mimic_config()
    spec, tag = build_myotorso_bimanual_mimic_spec(cfg)
    xml = spec.to_xml()
    xml = xml.replace(
        'meshdir="../myo_sim"',
        'meshdir="."',
    ).replace(
        'texturedir="../myo_sim"',
        'texturedir="."',
    )
    xml = xml.replace(
        '<mujoco model="myotorso_arm_chain_host">',
        '<mujoco model="myotorso_bimanual_mimic">',
        1,
    )
    if tag == _NATIVE_BIMANUAL_TAG:
        fresh = (
            "  <!-- Copyright (c) MyoSuite Authors. Apache-2.0.\n"
            "       Monolithic export: myo_sim-native myotorso_arms + bones-only\n"
            "       leg chain (no leg DOFs). NOT bit-exact parity with the\n"
            "       external MuscleMimic codebase's model - see\n"
            "       build_native_myotorso_bimanual_mimic_spec() docstring.\n"
            "       Regenerate via save_myotorso_bimanual_mimic_xml(). -->\n"
        )
        insert_after = xml.index(">", xml.index("<mujoco")) + 1
        xml = xml[:insert_after] + "\n" + fresh + xml[insert_after:]
    else:
        stale = r"<!-- Same asset.*?parity tests\. -->"
        fresh = (
            "  <!-- Copyright (c) MyoSuite Authors. Apache-2.0.\n"
            "       Monolithic export: mmc torso/head/legs + bimanual_chain + "
            "leg bones (no leg DOFs, no pelvis slides).\n"
            "       Regenerate via save_myotorso_bimanual_mimic_xml(). -->"
        )
        xml, n_sub = re.subn(stale, fresh, xml, count=1, flags=re.DOTALL)
        if n_sub != 1:
            raise RuntimeError(
                "Could not replace stale MJCF comment from MjSpec.to_xml(); "
                "update save_myotorso_bimanual_mimic_xml() regex."
            )

    out = dest
    if out is None:
        import tempfile

        out = (
            Path(tempfile.gettempdir())
            / "myosuite_musclemimic"
            / "myotorso_bimanual_mimic.xml"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml, encoding="utf-8")
    return out.resolve()


__all__ = [
    "build_myotorso_bimanual_mimic_spec",
    "compile_myotorso_bimanual_mimic_mjmodel",
    "default_myotorso_bimanual_mimic_config",
    "save_myotorso_bimanual_mimic_xml",
]
