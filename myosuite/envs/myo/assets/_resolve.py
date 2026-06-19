# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Shared XML-path resolvers for bundled myo model assets.

Resolution order for each model family:
  1. Local task-assets directory (custom overrides / task-specific additions)
  2. myo_sim pip package — ``models/legacy/<family>/`` (feat/legacy-models-pip+)
  3. myo_sim pip package — ``models/<family>/`` (backward compat for older builds)
  4. Fall through to the local path (caller decides how to handle missing file)
"""

from __future__ import annotations

import pathlib
import tempfile
import warnings
import xml.etree.ElementTree as ET

_ASSETS_ROOT = pathlib.Path(__file__).parent

# PoseEnvV0's viz_site_targets=("wrist",) needs a "wrist_target" site that the
# upstream myo_sim pip elbow fragments don't define. Rather than maintaining a
# duplicate copy of the (otherwise-identical) pip XML just to add one site,
# generate a small patched file on the fly: it includes the pip fragments by
# absolute reference (no body/asset content is copied) and adds only the site.
_WRIST_TARGET_SITE = {
    "name": "wrist_target",
    "pos": "0.007 -.261 0.065",
    "size": ".005",
}
_WRIST_TARGET_CACHE: dict[pathlib.Path, pathlib.Path] = {}


def _with_wrist_target_site(pip_path: pathlib.Path) -> pathlib.Path:
    """Return a small patched copy of *pip_path* with a "wrist_target" site added.

    Include paths and the compiler meshdir/texturedir are rewritten to
    absolute paths so the generated file works regardless of where it's
    written, without copying any of myo_sim's own body/asset content.
    """
    if pip_path in _WRIST_TARGET_CACHE:
        return _WRIST_TARGET_CACHE[pip_path]

    tree = ET.parse(pip_path)
    root = tree.getroot()
    for include in root.findall("include"):
        file_attr = include.get("file")
        if file_attr:
            include.set("file", str((pip_path.parent / file_attr).resolve()))
    compiler = root.find("compiler")
    if compiler is not None:
        for attr in ("meshdir", "texturedir"):
            value = compiler.get(attr)
            if value:
                compiler.set(attr, str((pip_path.parent / value).resolve()))
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "site", _WRIST_TARGET_SITE)

    out_dir = pathlib.Path(tempfile.gettempdir()) / "myosuite_patched_xml"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"wrist_target_{pip_path.stem}_{abs(hash(pip_path))}.xml"
    tree.write(out_path, encoding="unicode")
    _WRIST_TARGET_CACHE[pip_path] = out_path
    return out_path


TORSO_PIP_CALIBRATION_WARNING = (
    "This env's torso uses myo_sim pip's muscle/tendon calibration, not the "
    "original MyoChallenge competition models. myo_sim pip mirrors right-side "
    "torso muscle parameters (gainprm/biasprm/lengthrange) onto the left side; "
    "the original competition models had independently-fit, slightly "
    "asymmetric left/right calibration. Results are not expected to match "
    "the original competition submissions exactly."
)


def warn_torso_pip_calibration_divergence() -> None:
    """Warn that this env's torso muscles diverge from the original competition models.

    See the resolve_torso_xml docstring above for the asymmetric-vs-mirrored
    left-muscle calibration divergence between the legacy/competition torso
    models and myo_sim pip's torso muscle/tendon files.
    """
    warnings.warn(TORSO_PIP_CALIBRATION_WARNING, UserWarning, stacklevel=3)


_ELBOW_POSE_MODELS = frozenset(
    {"myoelbow_1dof6muscles.xml", "myoelbow_1dof6muscles_1dofexo.xml"}
)


def resolve_elbow_xml(filename: str = "myoelbow_1dof6muscles.xml") -> pathlib.Path:
    try:
        import myo_sim  # type: ignore[import-untyped]

        p = myo_sim.MODELS_DIR / "legacy" / "elbow" / filename
        if p.exists():
            return _with_wrist_target_site(p) if filename in _ELBOW_POSE_MODELS else p
        p = myo_sim.MODELS_DIR / "elbow" / filename
        if p.exists():
            return _with_wrist_target_site(p) if filename in _ELBOW_POSE_MODELS else p
    except ImportError:
        pass
    local = _ASSETS_ROOT / "elbow" / filename
    return local


def resolve_finger_xml(filename: str) -> pathlib.Path:
    local = _ASSETS_ROOT / "finger" / filename
    if local.exists():
        return local
    try:
        import myo_sim  # type: ignore[import-untyped]

        p = myo_sim.MODELS_DIR / "legacy" / "finger" / filename
        if p.exists():
            return p
    except ImportError:
        pass
    return local


def resolve_arm_xml(filename: str) -> pathlib.Path:
    local = _ASSETS_ROOT / "arm" / filename
    if local.exists():
        return local
    try:
        import myo_sim  # type: ignore[import-untyped]

        p = myo_sim.MODELS_DIR / "legacy" / "arm" / filename
        if p.exists():
            return p
        p = myo_sim.MODELS_DIR / "arm" / filename
        if p.exists():
            return p
    except ImportError:
        pass
    return local


def resolve_osl_xml(filename: str) -> pathlib.Path:
    local = _ASSETS_ROOT / "leg" / filename
    if local.exists():
        return local
    try:
        import myo_sim  # type: ignore[import-untyped]

        p = myo_sim.MODELS_DIR / "legacy" / "osl" / filename
        if p.exists():
            return p
    except ImportError:
        pass
    return local


def resolve_torso_xml(filename: str) -> pathlib.Path:
    """Resolve torso host XML: local bundled override first, then pip myo_sim."""
    local = _ASSETS_ROOT / "torso" / filename
    if local.exists():
        return local
    try:
        import myo_sim  # type: ignore[import-untyped]

        p = myo_sim.MODELS_DIR / "legacy" / "torso" / filename
        if p.exists():
            return p
        p = myo_sim.MODELS_DIR / "torso" / filename
        if p.exists():
            return p
    except ImportError:
        pass
    return local


def resolve_leg_xml(filename: str) -> pathlib.Path:
    """Resolve leg host XML: local bundled override first, then pip myo_sim.

    ``myolegs_with_torso.xml`` is bundled locally because the currently pinned
    myo_sim pip package's ``leg/myolegs.xml`` has no "torso" body (legs+pelvis
    only); the local file wraps the pip-resolved leg assets/tendon/muscle/chain
    in the bundled rigid-torso fallback so envs that need ``model.body("torso")``
    (e.g. ``LegWalkEnvV0._get_torso_angle``) keep working.
    """
    local = _ASSETS_ROOT / "leg" / filename
    if local.exists():
        return local
    try:
        import myo_sim  # type: ignore[import-untyped]

        p = myo_sim.MODELS_DIR / "legacy" / "leg" / filename
        if p.exists():
            return p
        p = myo_sim.MODELS_DIR / "leg" / filename
        if p.exists():
            return p
    except ImportError:
        pass
    return local
