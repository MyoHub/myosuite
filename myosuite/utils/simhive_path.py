# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Resolve cross-package simhive asset paths in main model XML files.

The MyoSuite MJCF scene files reference assets from sibling pip packages
(ycb_sim, furniture_sim, mpl_sim) via paths like:

    ../../../../simhive/YCB_sim/includes/defaults_ycb.xml
    ../furniture_sim/common/textures/stone1.png

These paths assume the git-submodule layout. When assets are pip-installed
instead, the paths don't resolve. This module rewrites them to the absolute
pip-package root before MuJoCo loads the model.

Only the main model XML is rewritten; pip-package internal XML files are
correct as of v0.2.0+ (upstream path fixes applied).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
import tempfile

_SIMHIVE_PACKAGE_MAP: dict[str, tuple[str, str]] = {
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


def _workspace_simhive_root() -> Path:
    return Path(__file__).resolve().parents[1] / "simhive"


def _installed_package_root(module_name: str) -> Path | None:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    return Path(module.__file__).resolve().parent


def get_simhive_asset_root(sim_name: str) -> Path:
    """Return the root path for a simhive asset family (local submodule or pip)."""
    if sim_name not in _SIMHIVE_PACKAGE_MAP:
        raise KeyError(f"Unsupported simhive asset family: {sim_name}")
    module_name, local_dir = _SIMHIVE_PACKAGE_MAP[sim_name]
    local_root = _workspace_simhive_root() / local_dir
    if local_root.exists():
        return local_root
    pkg_root = _installed_package_root(module_name)
    if pkg_root is not None:
        models_dir = pkg_root / "models"
        return models_dir if models_dir.exists() else pkg_root
    if sim_name == "myo_sim":
        raise FileNotFoundError(
            "Unable to resolve myo_sim assets. "
            "Run: git submodule update --init myosuite/simhive/myo_sim"
        )
    raise FileNotFoundError(
        f"Unable to resolve simhive assets for {sim_name}. "
        f"Install via pip or place files under {local_root}."
    )


def _resolve_legacy_simhive_path(include_file: str) -> Path | None:
    """Map ../../../../simhive/X_sim/... paths to pip or local roots."""
    posix_parts = PurePosixPath(include_file).parts
    if "simhive" not in posix_parts:
        return None
    sim_idx = posix_parts.index("simhive") + 1
    if sim_idx >= len(posix_parts):
        return None
    sim_name = posix_parts[sim_idx]
    if sim_name not in _SIMHIVE_PACKAGE_MAP:
        return None
    try:
        base = get_simhive_asset_root(sim_name)
    except FileNotFoundError:
        return None
    if sim_idx + 1 >= len(posix_parts):
        return base
    rel_tail = Path(*posix_parts[sim_idx + 1 :])
    return base / rel_tail


def _rewrite_relative_package_path(raw: str) -> Path | None:
    """Map ../furniture_sim/... style paths to pip or local simhive roots."""
    norm = raw.replace("\\", "/").strip()
    for pref, sim_name in _REL_PACKAGE_PREFIXES:
        if norm.startswith(pref):
            tail = norm[len(pref) :].lstrip("/")
            try:
                root = get_simhive_asset_root(sim_name)
            except FileNotFoundError:
                return None
            return root / tail if tail else root
    return None


_PATCHED_INCLUDE_CACHE: dict[Path, Path] = {}


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

    if not changed:
        _PATCHED_INCLUDE_CACHE[source] = source
        return source

    patch_dir = Path(tempfile.gettempdir()) / "myosuite_patched_xml"
    patch_dir.mkdir(parents=True, exist_ok=True)
    out_path = patch_dir / f"{source.stem}_{abs(hash(source))}{source.suffix}"
    tree.write(out_path, encoding="unicode")
    _PATCHED_INCLUDE_CACHE[source] = out_path
    return out_path


def _path_resolves(model_path: Path, raw: str) -> bool:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate.exists()
    return (model_path.parent / candidate).resolve().exists()


def _try_rewrite(model_path: Path, raw: str) -> str | None:
    if not raw.strip() or raw.strip().lower().startswith("http"):
        return None
    if _path_resolves(model_path, raw):
        return None
    for candidate in (
        _resolve_legacy_simhive_path(raw),
        _rewrite_relative_package_path(raw),
    ):
        if candidate is not None and candidate.exists():
            if candidate.suffix == ".xml":
                candidate = _absolutize_include(candidate)
            return str(candidate)
    return None


def resolve_model_xml_path(model_path: str | Path) -> Path:
    """Return a model path with cross-package simhive paths rewritten if needed.

    Parses the XML, rewrites any include/file/meshdir/texturedir attributes that
    reference other simhive packages (YCB_sim, furniture_sim, etc.) via submodule-
    relative or sibling-directory paths, saves a patched temp file next to the
    source, and returns its path. Returns the original path unchanged when no
    rewrites are needed.
    """
    model_path = Path(model_path).resolve()
    try:
        tree = ET.parse(model_path)
    except (FileNotFoundError, ET.ParseError):
        return model_path

    changed = False
    for elem in tree.getroot().iter():
        for attr in _PATH_ATTRIBUTES:
            raw = elem.get(attr)
            if not raw:
                continue
            new = _try_rewrite(model_path, raw)
            if new is not None:
                elem.set(attr, new)
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
