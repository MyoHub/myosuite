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


# myo_sim's shared myosuite_scene.xml includes an unnamed, un-textured
# mjGEOM_CYLINDER prop (radius ~1.05m) centered at the origin — a "pedestal"
# intended for object-manipulation tasks (baoding balls, die relocate) that
# rest a prop on a raised platform. Leg-family walking tasks pull in this same
# shared scene (for lights/cameras/floor texture) but have no use for the
# pedestal: it sits right under the character's feet and renders as a visible
# disc under the walker. Strip it out of a leg-only copy of the scene rather
# than touching myo_sim's shared scene file, which other task families
# legitimately depend on.
_LEG_TORSO_MODELS = frozenset(
    {"myolegs_with_torso.xml", "myolegs_with_torso_plane.xml"}
)
_LEG_SCENE_CACHE: dict[pathlib.Path, pathlib.Path] = {}
_LEG_HOST_CACHE: dict[pathlib.Path, pathlib.Path] = {}


def _absolutize_paths(root: ET.Element, base_dir: pathlib.Path) -> None:
    """Rewrite relative include/mesh/texture/meshdir/texturedir paths to absolute.

    Needed because the patched file is written to a different (tempdir)
    location than the source, so paths that were relative to the source's own
    directory must be resolved before the move.
    """
    for include in root.findall("include"):
        file_attr = include.get("file")
        if file_attr and not pathlib.Path(file_attr).is_absolute():
            include.set("file", str((base_dir / file_attr).resolve()))
    for compiler in root.findall("compiler"):
        for attr in ("meshdir", "texturedir"):
            value = compiler.get(attr)
            if value and not pathlib.Path(value).is_absolute():
                compiler.set(attr, str((base_dir / value).resolve()))


def _leg_scene_without_pedestal(scene_path: pathlib.Path) -> pathlib.Path:
    """Return myo_sim's official pedestal-free scene variant, path-resolved.

    myo_sim ships a pre-authored ``myosuite_scene_noPedestal.xml`` alongside
    ``myosuite_scene.xml`` — verified byte-diff to differ *only* in the
    floor/backdrop-mesh/cylinder geoms (lights, cameras, textures are
    identical). Prefer it over hand-stripping the cylinder from a copy of the
    pedestal scene: the pedestal scene's floor sits at ``z=-0.4`` with the
    cylinder's raised top surface (``z=-0.19+0.205=0.015``) acting as the
    *real* ground-contact surface for the leg model's standing keyframe
    (feet rest at ``z=0.05-0.11``) — stripping only the cylinder geom left
    the floor at its original ``z=-0.4``, ~0.45m below where the character
    actually stands, silently breaking standing/walking physics (episodes
    collapsed in ~20 steps instead of surviving ~100+) while looking "fixed"
    in a static screenshot. The official noPedestal variant's floor is
    already correctly repositioned to ``z=0`` to match.
    """
    scene_path = scene_path.resolve()
    if scene_path in _LEG_SCENE_CACHE:
        return _LEG_SCENE_CACHE[scene_path]

    no_pedestal_path = (
        scene_path.parent / f"{scene_path.stem}_noPedestal{scene_path.suffix}"
    )
    if not no_pedestal_path.exists():
        raise FileNotFoundError(
            f"Expected myo_sim's official pedestal-free scene variant at "
            f"{no_pedestal_path}, but it does not exist. myo_sim's shared "
            f"scene composition may have changed; re-verify "
            f"{scene_path.name} vs its *_noPedestal.xml sibling."
        )

    tree = ET.parse(no_pedestal_path)
    root = tree.getroot()
    _absolutize_paths(root, no_pedestal_path.parent)

    out_dir = pathlib.Path(tempfile.gettempdir()) / "myosuite_patched_xml"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"no_pedestal_{scene_path.stem}_{abs(hash(scene_path))}.xml"
    tree.write(out_path, encoding="unicode")
    _LEG_SCENE_CACHE[scene_path] = out_path
    return out_path


def _without_pedestal(host_path: pathlib.Path) -> pathlib.Path:
    """Return a copy of a leg host XML whose scene include has no pedestal geom."""
    host_path = host_path.resolve()
    if host_path in _LEG_HOST_CACHE:
        return _LEG_HOST_CACHE[host_path]

    tree = ET.parse(host_path)
    root = tree.getroot()
    patched_scene: pathlib.Path | None = None
    for include in root.findall("include"):
        file_attr = include.get("file")
        if file_attr and pathlib.Path(file_attr).name == "myosuite_scene.xml":
            # The relative "../myo_sim/scene/myosuite_scene.xml" path isn't a
            # real local file — it's resolved against the pip myo_sim package
            # at compile time (see asset_path_resolver.py). Resolve it the
            # same way rather than against host_path.parent.
            scene_path = (host_path.parent / file_attr).resolve()
            if not scene_path.exists():
                import myo_sim  # type: ignore[import-untyped]

                scene_path = myo_sim.MODELS_DIR / "scene" / "myosuite_scene.xml"
            patched_scene = _leg_scene_without_pedestal(scene_path)
            include.set("file", str(patched_scene))
    if patched_scene is None:
        _LEG_HOST_CACHE[host_path] = host_path
        return host_path
    # Only the scene include is rewritten to an absolute path above. The
    # host's other includes (torso/leg assets/tendon/muscle) stay relative
    # ("../myo_sim/...") — resolve_model_xml_path's pattern-based fallback
    # resolves those against the pip myo_sim package regardless of where this
    # patched copy is written, so they must not be pre-absolutized against
    # host_path.parent (which would point into a nonexistent local mirror).

    out_dir = pathlib.Path(tempfile.gettempdir()) / "myosuite_patched_xml"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"no_pedestal_{host_path.stem}_{abs(hash(host_path))}.xml"
    tree.write(out_path, encoding="unicode")
    _LEG_HOST_CACHE[host_path] = out_path
    return out_path


def resolve_leg_xml(filename: str) -> pathlib.Path:
    """Resolve leg host XML: local bundled override first, then pip myo_sim.

    ``myolegs_with_torso.xml`` is bundled locally because the currently pinned
    myo_sim pip package's ``leg/myolegs.xml`` has no "torso" body (legs+pelvis
    only); the local file wraps the pip-resolved leg assets/tendon/muscle/chain
    in the bundled rigid-torso fallback so envs that need ``model.body("torso")``
    (e.g. ``LegWalkEnvV0._get_torso_angle``) keep working.

    ``myolegs_with_torso.xml`` and ``myolegs_with_torso_plane.xml`` pull in
    myo_sim's shared ``myosuite_scene.xml`` for lights/camera/floor texture;
    that shared scene also carries an unnamed pedestal-cylinder prop meant for
    object-manipulation tasks, which is stripped out here (see
    ``_without_pedestal``) since it has no purpose for leg locomotion and
    otherwise renders as a visible disc under the walker.
    """
    local = _ASSETS_ROOT / "leg" / filename
    if local.exists():
        return _without_pedestal(local) if filename in _LEG_TORSO_MODELS else local
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
