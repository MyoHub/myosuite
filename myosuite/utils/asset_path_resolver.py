# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Resolve cross-package sim-asset paths in main model XML files.

The MyoSuite MJCF scene files reference assets from sibling pip packages
(myo_sim, ycb_sim, furniture_sim, mpl_sim, object_sim) via paths like:

    ../../../../simhive/YCB_sim/includes/defaults_ycb.xml
    ../furniture_sim/common/textures/stone1.png

The first form is a legacy path from the old git-submodule layout (a
"simhive/" directory holding all sim-asset submodules); the second is the
short package-relative form used now that assets are pip-installed. When
the legacy form is encountered, or a relative path doesn't resolve as-is,
this module rewrites it to the absolute pip-package root before MuJoCo
loads the model.

Only the main model XML is rewritten; pip-package internal XML files are
correct as of v0.2.0+ (upstream path fixes applied).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
import tempfile

_SIM_PACKAGE_MAP: dict[str, tuple[str, str]] = {
    "myo_sim": ("myo_sim", "myo_sim"),
    "furniture_sim": ("furniture_sim", "furniture_sim"),
    "MPL_sim": ("mpl_sim", "MPL_sim"),
    "object_sim": ("object_sim", "object_sim"),
    "YCB_sim": ("ycb_sim", "YCB_sim"),
}

# Longest-first: more specific prefixes before shorter ones.
_REL_PACKAGE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("./../../MPL_sim/", "MPL_sim"),
    ("../../MPL_sim/", "MPL_sim"),
    ("./../../MPL_sim", "MPL_sim"),
    ("../../MPL_sim", "MPL_sim"),
    ("../furniture_sim/", "furniture_sim"),
    ("../object_sim/", "object_sim"),
    ("../MPL_sim/", "MPL_sim"),
    ("../YCB_sim/", "YCB_sim"),
    ("../myo_sim/", "myo_sim"),
    ("../furniture_sim", "furniture_sim"),
    ("../object_sim", "object_sim"),
    ("../MPL_sim", "MPL_sim"),
    ("../YCB_sim", "YCB_sim"),
    ("../myo_sim", "myo_sim"),
)

_PATH_ATTRIBUTES = frozenset({"file", "meshdir", "texturedir"})


def _local_override_root() -> Path:
    """Optional local checkout root for sim-asset packages (dev override)."""
    return Path(__file__).resolve().parents[1] / "simhive"


def _installed_package_root(module_name: str) -> Path | None:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    return Path(module.__file__).resolve().parent


def get_sim_asset_root(sim_name: str) -> Path:
    """Return the root path for a sim-asset family (local override or pip)."""
    if sim_name not in _SIM_PACKAGE_MAP:
        raise KeyError(f"Unsupported sim-asset family: {sim_name}")
    module_name, local_dir = _SIM_PACKAGE_MAP[sim_name]
    local_root = _local_override_root() / local_dir
    if local_root.exists():
        return local_root
    pkg_root = _installed_package_root(module_name)
    if pkg_root is not None:
        models_dir = pkg_root / "models"
        return models_dir if models_dir.exists() else pkg_root
    if sim_name == "myo_sim":
        raise FileNotFoundError(
            "Unable to resolve myo_sim assets. Install via pip (`pip install myo-sim`)."
        )
    raise FileNotFoundError(
        f"Unable to resolve sim assets for {sim_name}. "
        f"Install via pip or place files under {local_root}."
    )


_MYOSUITE_ASSETS = Path(__file__).resolve().parents[1] / "envs" / "myo" / "assets"

# Files that existed in the old myo_sim git-submodule layout but are absent
# from the pip package. They are bundled in myosuite/envs/myo/assets/ as a
# compatibility shim until the upstream pip package ships them.
_MYO_SIM_BUNDLED_FALLBACKS: dict[str, Path] = {
    "arm/assets/myoarm_assets.xml": _MYOSUITE_ASSETS
    / "arm"
    / "assets"
    / "myoarm_assets.xml",
    "arm/assets/myoarm_body.xml": _MYOSUITE_ASSETS
    / "arm"
    / "assets"
    / "myoarm_body.xml",
    "hand/assets/myohand_assets.xml": _MYOSUITE_ASSETS
    / "hand"
    / "assets"
    / "myohand_assets.xml",
    "hand/assets/myohand_body.xml": _MYOSUITE_ASSETS
    / "hand"
    / "assets"
    / "myohand_body.xml",
    "torso/assets/myotorso_arm_chain.xml": _MYOSUITE_ASSETS
    / "torso"
    / "assets"
    / "myotorso_arm_chain.xml",
    "torso/assets/myotorso_rigid_assets.xml": _MYOSUITE_ASSETS
    / "torso"
    / "assets"
    / "myotorso_rigid_assets.xml",
    "torso/assets/myotorso_rigid_chain.xml": _MYOSUITE_ASSETS
    / "torso"
    / "assets"
    / "myotorso_rigid_chain.xml",
}


def _resolve_legacy_submodule_path(include_file: str) -> Path | None:
    """Map legacy ../../../../simhive/X_sim/... paths to pip or local roots."""
    posix_parts = PurePosixPath(include_file).parts
    if "simhive" not in posix_parts:
        return None
    sim_idx = posix_parts.index("simhive") + 1
    if sim_idx >= len(posix_parts):
        return None
    sim_name = posix_parts[sim_idx]
    if sim_name not in _SIM_PACKAGE_MAP:
        return None
    try:
        base = get_sim_asset_root(sim_name)
    except FileNotFoundError:
        return None
    if sim_idx + 1 >= len(posix_parts):
        return base
    rel_tail = Path(*posix_parts[sim_idx + 1 :])
    resolved = base / rel_tail
    if not resolved.exists() and sim_name == "myo_sim":
        rel_key = "/".join(posix_parts[sim_idx + 1 :])
        fallback = _MYO_SIM_BUNDLED_FALLBACKS.get(rel_key)
        if fallback is not None and fallback.exists():
            return fallback
    return resolved


def _rewrite_relative_package_path(raw: str) -> Path | None:
    """Map ../furniture_sim/... style paths to pip or local override roots."""
    norm = raw.replace("\\", "/").strip()
    for pref, sim_name in _REL_PACKAGE_PREFIXES:
        if norm.startswith(pref):
            tail = norm[len(pref) :].lstrip("/")
            try:
                root = get_sim_asset_root(sim_name)
            except FileNotFoundError:
                return None
            if not tail:
                return root
            resolved = root / tail
            if not resolved.exists() and sim_name == "myo_sim":
                # Some myo_sim pip fragments only ship under models/legacy/...
                legacy_resolved = root / "legacy" / tail
                if legacy_resolved.exists():
                    return legacy_resolved
                # A few files in the old submodule layout never made it into
                # the pip package at all; myosuite bundles compatibility copies.
                bundled_fallback = _MYO_SIM_BUNDLED_FALLBACKS.get(tail)
                if bundled_fallback is not None and bundled_fallback.exists():
                    return bundled_fallback
            return resolved
    return None


_PATCHED_INCLUDE_CACHE: dict[Path, Path] = {}
_PATCHED_BUNDLED_CACHE: dict[Path, Path] = {}


def _write_patched_xml(tree: ET.ElementTree, source: Path, prefix: str) -> Path:
    patch_dir = Path(tempfile.gettempdir()) / "myosuite_patched_xml"
    patch_dir.mkdir(parents=True, exist_ok=True)
    out_path = patch_dir / f"{prefix}_{source.stem}_{abs(hash(source))}{source.suffix}"
    tree.write(out_path, encoding="unicode")
    return out_path


def _patch_bundled_include(source: Path, old_meshdir: Path | None = None) -> Path:
    """Rewrite sim-package paths inside a locally-bundled include file."""
    source = source.resolve()
    if source in _PATCHED_BUNDLED_CACHE:
        return _PATCHED_BUNDLED_CACHE[source]

    try:
        tree = ET.parse(source)
    except (FileNotFoundError, ET.ParseError):
        _PATCHED_BUNDLED_CACHE[source] = source
        return source

    changed = False
    for elem in tree.getroot().iter():
        for attr in _PATH_ATTRIBUTES:
            raw = elem.get(attr)
            if not raw:
                continue
            candidate = Path(raw)
            if not candidate.is_absolute():
                local = (source.parent / candidate).resolve()
                if local.exists() and _MYOSUITE_ASSETS in local.parents:
                    patched = (
                        _patch_bundled_include(local, old_meshdir=old_meshdir)
                        if local.suffix == ".xml"
                        else local
                    )
                    elem.set(attr, str(patched))
                    changed = True
                    continue
            new = _try_rewrite(source, raw, old_meshdir=old_meshdir)
            if new is not None:
                elem.set(attr, new)
                changed = True
            elif attr == "file" and elem.tag in ("mesh", "texture"):
                # Old-meshdir-relative climb (e.g. "../../envs/myo/assets/
                # ground_brown.png") that no longer lands correctly once
                # meshdir is rewritten. Try the bare filename next to this
                # bundled fragment's own file.
                bare = (source.parent / candidate.name).resolve()
                if bare.exists():
                    elem.set(attr, str(bare))
                    changed = True

    if not changed:
        _PATCHED_BUNDLED_CACHE[source] = source
        return source

    out_path = _write_patched_xml(tree, source, "bundled")
    _PATCHED_BUNDLED_CACHE[source] = out_path
    return out_path


def _absolutize_include(source: Path) -> Path:
    """Return a copy of *source* XML with all relative asset paths absolutized.

    MuJoCo 3 resolves compiler meshdir/texturedir relative to the MAIN model
    file, not the include file. Any relative path in a pip-package include will
    therefore resolve incorrectly when the include is loaded from a different
    directory.

    Strategy:
    - Determine effective mesh/texture base dirs from compiler meshdir/texturedir
      (resolved relative to source.parent).
    - Absolutize every relative file= attribute on mesh/texture elements using
      the appropriate base dir.
    - Absolutize every other relative path attribute (include file=, meshdir,
      texturedir) relative to source.parent.

    Results are cached by source path. Returns the source path unchanged when
    no patches are needed.
    """
    if source in _PATCHED_INCLUDE_CACHE:
        return _PATCHED_INCLUDE_CACHE[source]

    try:
        tree = ET.parse(source)
    except (FileNotFoundError, ET.ParseError):
        _PATCHED_INCLUDE_CACHE[source] = source
        return source

    root = tree.getroot()
    changed = False

    # Determine compiler-declared mesh and texture base dirs.
    compiler_meshdir: Path | None = None
    compiler_texturedir: Path | None = None
    for compiler in root.iter("compiler"):
        for attr_name, store in (
            ("meshdir", "compiler_meshdir"),
            ("texturedir", "compiler_texturedir"),
        ):
            v = compiler.get(attr_name)
            if not v:
                continue
            c = Path(v)
            resolved = c if c.is_absolute() else (source.parent / c).resolve()
            if resolved.exists():
                if attr_name == "meshdir":
                    compiler_meshdir = resolved
                else:
                    compiler_texturedir = resolved

    # Fallback mesh base for include files in an assets/ subdirectory whose
    # meshes live one directory above (e.g. mpl_sim/assets/*.xml → mpl_sim/).
    if compiler_meshdir is None and source.parent.name == "assets":
        alt_mesh_base: Path | None = source.parent.parent
    else:
        alt_mesh_base = None

    # Absolutize all relative path attributes.
    for elem in root.iter():
        for attr in _PATH_ATTRIBUTES:
            raw = elem.get(attr)
            if not raw or raw.strip().lower().startswith("http"):
                continue
            c = Path(raw)
            if c.is_absolute():
                continue
            # Choose resolution base for mesh/texture file= attributes.
            if attr == "file" and elem.tag == "mesh":
                base = compiler_meshdir or alt_mesh_base or source.parent
            elif attr == "file" and elem.tag == "texture":
                base = compiler_texturedir or alt_mesh_base or source.parent
            else:
                base = source.parent
            resolved = (base / c).resolve()
            if resolved.exists():
                elem.set(attr, str(resolved))
                changed = True
            elif attr == "file":
                # Local checkout not present; try resolving via pip package.
                fallback = _rewrite_relative_package_path(raw)
                if fallback is None:
                    fallback = _resolve_legacy_submodule_path(raw)
                if (fallback is None or not fallback.exists()) and elem.tag in (
                    "mesh",
                    "texture",
                ):
                    # Some myo_sim legacy/* fragments declare bare mesh/texture
                    # filenames (no "meshes/" prefix), assuming they live in
                    # the package's shared models/meshes/ pool.
                    try:
                        pooled = get_sim_asset_root("myo_sim") / "meshes" / c.name
                    except FileNotFoundError:
                        pooled = None
                    if pooled is not None and pooled.exists():
                        fallback = pooled
                if fallback is not None and fallback.exists():
                    elem.set(attr, str(fallback))
                    changed = True

    if not changed:
        _PATCHED_INCLUDE_CACHE[source] = source
        return source

    out_path = _write_patched_xml(tree, source, source.stem)
    _PATCHED_INCLUDE_CACHE[source] = out_path
    return out_path


def _path_resolves(model_path: Path, raw: str) -> bool:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate.exists()
    return (model_path.parent / candidate).resolve().exists()


def _try_rewrite(
    model_path: Path, raw: str, old_meshdir: Path | None = None
) -> str | None:
    if not raw.strip() or raw.strip().lower().startswith("http"):
        return None
    if _path_resolves(model_path, raw):
        return None
    # If the old meshdir is known, try resolving the path against it (for local assets
    # that use paths relative to the OLD meshdir, which is no longer valid after rewrite).
    if old_meshdir is not None:
        candidate = (old_meshdir / raw).resolve()
        if candidate.exists():
            return str(candidate)
    for candidate in (
        _resolve_legacy_submodule_path(raw),
        _rewrite_relative_package_path(raw),
    ):
        if candidate is not None and candidate.exists():
            if candidate.suffix == ".xml":
                if _MYOSUITE_ASSETS in candidate.resolve().parents:
                    candidate = _patch_bundled_include(
                        candidate, old_meshdir=old_meshdir
                    )
                else:
                    candidate = _absolutize_include(candidate)
            return str(candidate)
    return None


def resolve_model_xml_path(model_path: str | Path) -> Path:
    """Return a model path with cross-package sim-asset paths rewritten if needed.

    Parses the XML, rewrites any include/file/meshdir/texturedir attributes that
    reference other sim-asset packages (myo_sim, YCB_sim, furniture_sim, etc.)
    via legacy submodule-relative or short package-relative paths, saves a
    patched temp file next to the source, and returns its path. Returns the
    original path unchanged when no rewrites are needed.
    """
    model_path = Path(model_path).resolve()
    try:
        tree = ET.parse(model_path)
    except (FileNotFoundError, ET.ParseError):
        return model_path

    changed = False
    old_meshdir: Path | None = None
    for elem in tree.getroot().iter():
        for attr in _PATH_ATTRIBUTES:
            raw = elem.get(attr)
            if not raw:
                continue
            new = _try_rewrite(model_path, raw, old_meshdir=old_meshdir)
            if new is not None:
                if attr == "meshdir":
                    # Capture old meshdir (resolved relative to model file) before overwriting.
                    # Don't check existence — the old dir may be a submodule not checked out,
                    # but paths relative to it (e.g. ../../envs/myo/assets/) may still resolve
                    # to existing files once Path.resolve() collapses the `..` segments.
                    _old = Path(raw)
                    if not _old.is_absolute():
                        _old = (model_path.parent / _old).resolve()
                    if old_meshdir is None:
                        old_meshdir = _old
                elem.set(attr, new)
                changed = True
            elif attr == "file" and elem.tag in {"mesh", "texture"}:
                candidate = Path(raw)
                if not candidate.is_absolute():
                    local = (model_path.parent / candidate).resolve()
                    if local.exists() and _MYOSUITE_ASSETS in local.parents:
                        elem.set(attr, str(local))
                        changed = True
                    else:
                        # The raw path may be an old-meshdir-relative climb
                        # (e.g. "../../envs/myo/assets/hand/dice.png") that no
                        # longer lands correctly once meshdir is rewritten to a
                        # pip package root. If the bare filename sits right
                        # next to the model file itself, use that instead.
                        bare = (model_path.parent / candidate.name).resolve()
                        if bare.exists():
                            elem.set(attr, str(bare))
                            changed = True
            elif attr == "file" and elem.tag == "include":
                candidate = Path(raw)
                if not candidate.is_absolute():
                    candidate = (model_path.parent / candidate).resolve()
                if (
                    candidate.suffix == ".xml"
                    and candidate.exists()
                    and _MYOSUITE_ASSETS in candidate.parents
                ):
                    patched = _patch_bundled_include(candidate, old_meshdir=old_meshdir)
                    if patched != candidate:
                        elem.set(attr, str(patched))
                        changed = True

    if not changed:
        return model_path

    fd, out_str = tempfile.mkstemp(
        suffix=".xml",
        prefix=".myosuite_resolved_",
        dir=str(model_path.parent),
    )
    try:
        os.close(fd)
        out_path = Path(out_str)
        tree.write(out_path, encoding="unicode")
    except Exception:
        try:
            os.unlink(out_str)
        except OSError:
            pass
        raise
    return out_path
