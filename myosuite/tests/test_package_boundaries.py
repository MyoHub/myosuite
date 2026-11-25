# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Enforce internal subpackage ownership and the upward module import DAG."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess

import pytest

pytestmark = pytest.mark.tier1

_REPO = Path(__file__).resolve().parents[2]
_PACKAGE = _REPO / "myosuite"

# Active runtime subpackages in myosuite/.
_APPROVED_RUNTIME_SUBPACKAGES = frozenset(
    {
        "core",
        "physics",
        "terms",
        "scenes",
        "biomechanics",
        "integrations",
        "envs",
        "viz",
        "logger",
        "utils",
    }
)

# Out-of-wheel repository directories that runtime modules must never import.
_FORBIDDEN_EXTERNAL_IMPORTS = frozenset({"training", "tools", "scripts"})

# Allowed runtime imports between myosuite.<src> and myosuite.<dst> subpackages.
# "__root__" denotes myosuite/__init__.py and myosuite/version.py.
_ALLOWED_INTERNAL_IMPORTS: dict[str, frozenset[str]] = {
    "__root__": frozenset(
        {"__root__", "core", "envs", "logger", "scenes", "terms", "utils"}
    ),
    # Foundation & numerical helpers
    "core": frozenset({"__root__", "core", "physics"}),
    "physics": frozenset({"__root__", "physics"}),
    "scenes": frozenset({"__root__", "core", "scenes"}),
    "terms": frozenset({"__root__", "core", "physics", "terms"}),
    # Solvers & lightweight source/ecosystem adapters
    "biomechanics": frozenset({"__root__", "core", "physics", "biomechanics"}),
    # `integrations -> utils` remains allowed until `utils` policy helpers move into `policies/` (#938).
    "integrations": frozenset(
        {
            "__root__",
            "core",
            "terms",
            "physics",
            "biomechanics",
            "utils",
            "integrations",
        }
    ),
    # Simulation environments & visualization (`envs -> utils` remains allowed until
    # `utils` helpers move into `core/`, `policies/`, and `benchmarks/`).
    "envs": frozenset(
        {
            "__root__",
            "core",
            "scenes",
            "terms",
            "physics",
            "biomechanics",
            "integrations",
            "utils",
            "envs",
        }
    ),
    "viz": frozenset(
        {"__root__", "core", "scenes", "physics", "biomechanics", "envs", "viz"}
    ),
    # Rollout logging & utilities (scheduled for split into core/policies/benchmarks)
    "logger": frozenset({"__root__", "core", "physics", "logger"}),
    "utils": frozenset(
        {
            "__root__",
            "core",
            "terms",
            "physics",
            "biomechanics",
            "envs",
            "viz",
            "utils",
        }
    ),
}

# Pre-existing cross-layer runtime edges on dev (27 edges), burned down as migrations land.
# Do not add entries here; place new runtime logic in its canonical layer.
_LEGACY_CROSS_IMPORTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("biomechanics/retarget_instanthmr.py", "viz"),
        ("core/config.py", "envs"),
        ("core/model_builder.py", "envs"),
        ("core/model_recipes.py", "envs"),
        ("core/model_recipes.py", "integrations"),
        ("core/registry.py", "envs"),
        ("core/retargeting.py", "integrations"),
        ("envs/gymnasium_env.py", "viz"),
        ("envs/heightfields.py", "viz"),
        ("envs/multi_agent_modular_env.py", "viz"),
        ("integrations/musclemimic/fbvs_player.py", "envs"),
        ("integrations/musclemimic/fullbody_eval_cli.py", "envs"),
        ("integrations/musclemimic/fullbody_native_playback.py", "envs"),
        ("integrations/musclemimic/fullbody_native_playback.py", "viz"),
        ("integrations/musclemimic/mjlab_onnx_policy.py", "envs"),
        ("logger/examine_logs.py", "utils"),
        ("logger/grouped_datasets.py", "utils"),
        ("physics/inverse_kinematics.py", "biomechanics"),
        ("terms/multiplayer/chase_tag_vs_obs.py", "envs"),
        ("terms/multiplayer/chase_tag_vs_reward.py", "envs"),
        ("utils/checkpoint_manifest.py", "integrations"),
        ("utils/export_onnx.py", "integrations"),
        ("utils/onnx_checkpoint.py", "integrations"),
        ("utils/policy_utils.py", "logger"),
        ("viz/blender_render.py", "utils"),
        ("viz/mjlab_play.py", "integrations"),
        ("viz/mjlab_play.py", "utils"),
    }
)

_COMPATIBILITY_MODULE_SHIMS: tuple[tuple[str, str], ...] = (
    ("myosuite.scenes.library", "myosuite.core.builder.scene_library"),
    ("myosuite.scenes.scene_builder", "myosuite.core.builder.scene_builder"),
    ("myosuite.envs.myo.myoedits.model_editor", "myosuite.core.builder.model_editor"),
    ("myosuite.utils.spec_processing", "myosuite.core.builder.spec_processing"),
    ("myosuite.utils.mujoco_geom_utils", "myosuite.core.builder.mujoco_geom_utils"),
    ("myosuite.utils.mujoco_joint_utils", "myosuite.core.builder.mujoco_joint_utils"),
    ("myosuite.utils.asset_path_resolver", "myosuite.core.builder.asset_path_resolver"),
    ("myosuite.utils.colored_noise", "myosuite.physics.colored_noise"),
    ("myosuite.terms._array", "myosuite.core.terms._array"),
    ("myosuite.terms.base_action", "myosuite.core.terms.base_action"),
    ("myosuite.terms.base_event", "myosuite.core.terms.base_event"),
    ("myosuite.terms.base_obs", "myosuite.core.terms.base_obs"),
    ("myosuite.terms.base_reward", "myosuite.core.terms.base_reward"),
    ("myosuite.terms.base_termination", "myosuite.core.terms.base_termination"),
    ("myosuite.terms.mimic_obs", "myosuite.core.terms.mimic_obs"),
    ("myosuite.terms.mimic_reward", "myosuite.core.terms.mimic_reward"),
    ("myosuite.terms.mimic_track", "myosuite.core.terms.mimic_track"),
    (
        "myosuite.terms.opponent_relative_obs",
        "myosuite.core.terms.opponent_relative_obs",
    ),
    ("myosuite.terms.waypoint", "myosuite.core.terms.waypoint"),
    ("myosuite.terms.multiplayer.common", "myosuite.core.terms.multiplayer.common"),
    ("myosuite.terms.multiplayer.heading", "myosuite.core.terms.multiplayer.heading"),
)

_OPTIONAL_TOP_LEVEL_DEPS = frozenset({"jax", "mink", "torch", "onnxruntime"})


def _runtime_python_files() -> list[Path]:
    """Return runtime ``.py`` files inside ``myosuite/`` (excluding ``tests/`` and ``agents/``)."""
    proc = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "myosuite",
        ],
        cwd=_REPO,
        check=False,
        capture_output=True,
    )
    if proc.returncode == 0:
        candidates = [
            _REPO / raw
            for raw in proc.stdout.decode("utf-8").split("\0")
            if raw and raw.endswith(".py")
        ]
    else:
        candidates = [p for p in _PACKAGE.rglob("*.py") if "__pycache__" not in p.parts]
    paths: list[Path] = []
    for candidate in candidates:
        rel = candidate.relative_to(_REPO).as_posix()
        if rel.startswith("myosuite/tests/") or rel.startswith("myosuite/agents/"):
            continue
        if candidate.is_file():
            paths.append(candidate)
    return sorted(set(paths))


def _is_type_checking_guard(test: ast.expr) -> bool:
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if (
        isinstance(test, ast.Attribute)
        and test.attr == "TYPE_CHECKING"
        and isinstance(test.value, ast.Name)
        and test.value.id in {"typing", "t"}
    ):
        return True
    return False


def _resolve_relative_module(
    rel_parts: list[str], is_pkg: bool, name: str, level: int
) -> str:
    anchor_parts = ["myosuite", *rel_parts] if is_pkg else ["myosuite", *rel_parts[:-1]]
    keep = len(anchor_parts) - (level - 1)
    prefix = ".".join(anchor_parts[: max(keep, 0)])
    return f"{prefix}.{name}" if (prefix and name) else (name or prefix)


def _is_import_module_call(func: ast.expr) -> bool:
    if isinstance(func, ast.Name) and func.id == "import_module":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "import_module":
        return True
    return False


def _runtime_imports(py_file: Path) -> list[str]:
    """Extract runtime import targets (including literal ``import_module(...)`` calls), ignoring ``if TYPE_CHECKING:`` bodies."""
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(py_file))

    type_checking_nodes: set[ast.AST] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_type_checking_guard(node.test):
            for stmt in node.body:
                type_checking_nodes.update(ast.walk(stmt))

    rel_parts = list(py_file.relative_to(_PACKAGE).with_suffix("").parts)
    is_pkg = rel_parts[-1] == "__init__"
    if is_pkg:
        rel_parts = rel_parts[:-1]

    imports: list[str] = []
    for node in ast.walk(tree):
        if node in type_checking_nodes:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    imports.append(node.module)
                    if node.module == "myosuite" or node.module.startswith("myosuite."):
                        for alias in node.names:
                            if alias.name != "*":
                                imports.append(f"{node.module}.{alias.name}")
            else:
                base_mod = _resolve_relative_module(
                    rel_parts, is_pkg, node.module, node.level
                )
                if node.module:
                    imports.append(base_mod)
                for alias in node.names:
                    if alias.name != "*":
                        imports.append(
                            f"{base_mod}.{alias.name}" if base_mod else alias.name
                        )
        elif (
            isinstance(node, ast.Call)
            and _is_import_module_call(node.func)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            raw_mod = node.args[0].value
            pkg_arg: str | None = None
            if (
                len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                pkg_arg = node.args[1].value
            for kw in node.keywords:
                if (
                    kw.arg == "package"
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)
                ):
                    pkg_arg = kw.value.value
            if raw_mod.startswith("."):
                level = len(raw_mod) - len(raw_mod.lstrip("."))
                remainder = raw_mod[level:]
                if pkg_arg:
                    pkg_parts = pkg_arg.split(".")
                    keep = len(pkg_parts) - (level - 1)
                    prefix = ".".join(pkg_parts[: max(keep, 0)])
                    imports.append(
                        f"{prefix}.{remainder}"
                        if (prefix and remainder)
                        else (remainder or prefix)
                    )
                else:
                    imports.append(
                        _resolve_relative_module(rel_parts, is_pkg, remainder, level)
                    )
            else:
                imports.append(raw_mod)
    return imports


def test_approved_myosuite_subpackages() -> None:
    """Every tracked top-level subdirectory of ``myosuite/`` must be an approved subpackage."""
    found = {
        path.relative_to(_PACKAGE).parts[0]
        for path in _runtime_python_files()
        if len(path.relative_to(_PACKAGE).parts) > 1
    }
    unexpected = sorted(found - _APPROVED_RUNTIME_SUBPACKAGES)
    assert not unexpected, (
        f"Unapproved top-level myosuite/ subpackages: {unexpected}. "
        f"Approved: {sorted(_APPROVED_RUNTIME_SUBPACKAGES)}"
    )


def test_myosuite_never_imports_out_of_wheel_or_test_packages() -> None:
    """Runtime modules inside ``myosuite/`` must never import ``training``, ``tools``, ``scripts``, or ``tests``."""
    violations: list[str] = []
    for path in _runtime_python_files():
        rel = path.relative_to(_PACKAGE).as_posix()
        for mod in _runtime_imports(path):
            top = mod.split(".")[0]
            if top in _FORBIDDEN_EXTERNAL_IMPORTS:
                violations.append(f"{rel} -> {mod}")
            elif mod == "myosuite.tests" or mod.startswith("myosuite.tests."):
                violations.append(f"{rel} -> {mod}")
    assert not violations, (
        "Runtime myosuite modules must not import out-of-wheel or test packages:\n  "
        + "\n  ".join(violations)
    )


def test_myosuite_internal_import_dag() -> None:
    """Enforce the upward module import DAG across ``myosuite/``."""
    shim_map = dict(_COMPATIBILITY_MODULE_SHIMS)
    shim_imports: list[str] = []
    violations: set[tuple[str, str]] = set()
    for path in _runtime_python_files():
        rel = path.relative_to(_PACKAGE)
        rel_posix = rel.as_posix()
        src_sub = rel.parts[0] if len(rel.parts) > 1 else "__root__"
        allowed = _ALLOWED_INTERNAL_IMPORTS[src_sub]
        is_shim_or_facade = (
            f"myosuite.{rel_posix.removesuffix('.py').replace('/', '.')}" in shim_map
            or rel_posix
            in {
                "scenes/__init__.py",
                "terms/__init__.py",
                "terms/multiplayer/__init__.py",
            }
        )
        for mod in _runtime_imports(path):
            if not is_shim_or_facade and mod in shim_map:
                shim_imports.append(f"{rel_posix} -> {mod} (use {shim_map[mod]})")
            mod = shim_map.get(mod, mod)
            parts = mod.split(".")
            if parts[0] != "myosuite" or len(parts) < 2:
                continue
            dst_sub = (
                parts[1] if parts[1] in _APPROVED_RUNTIME_SUBPACKAGES else "__root__"
            )
            if dst_sub not in allowed:
                violations.add((rel.as_posix(), dst_sub))

    assert not shim_imports, (
        "Non-facade runtime modules must import canonical paths rather than compatibility shims:\n  "
        + "\n  ".join(sorted(shim_imports))
    )
    unexpected = sorted(violations - _LEGACY_CROSS_IMPORTS)
    assert not unexpected, (
        "Forbidden cross-subpackage runtime imports detected (see docs/wiki/repository-map.md):\n  "
        + "\n  ".join(f"{src} -> myosuite.{dst}" for src, dst in unexpected)
    )
    stale = sorted(_LEGACY_CROSS_IMPORTS - violations)
    assert not stale, (
        "Stale _LEGACY_CROSS_IMPORTS entries no longer occur — remove them so fixed edges cannot regress:\n  "
        + "\n  ".join(f"{src} -> myosuite.{dst}" for src, dst in stale)
    )


def test_runtime_imports_checks_type_checking_else_and_import_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_runtime_imports`` must check ``if TYPE_CHECKING: ... else:`` runtime branches and literal ``import_module`` calls."""
    pkg = tmp_path / "myosuite"
    (pkg / "core").mkdir(parents=True)
    sample = pkg / "core" / "sample.py"
    sample.write_text(
        "from typing import TYPE_CHECKING\n"
        "import importlib\n"
        "if TYPE_CHECKING:\n"
        "    import myosuite.viz\n"
        "else:\n"
        "    import myosuite.envs\n"
        "from myosuite import logger, make_env\n"
        "from myosuite.terms import mimic_track\n"
        "from ..utils import asset_path_resolver\n"
        "importlib.import_module('myosuite.integrations')\n"
        "importlib.import_module('.submod', package='myosuite.biomechanics')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("myosuite.tests.test_package_boundaries._PACKAGE", pkg)
    found = set(_runtime_imports(sample))
    assert "myosuite.viz" not in found
    assert "myosuite.envs" in found
    assert "myosuite.logger" in found
    assert "myosuite.make_env" in found
    assert "myosuite.terms.mimic_track" in found
    assert "myosuite.utils.asset_path_resolver" in found
    assert "myosuite.integrations" in found
    assert "myosuite.biomechanics.submod" in found


def test_compatibility_shims_are_pure_aliases() -> None:
    """Every migrated shim must resolve to the identical module object and define zero new classes/functions."""
    import importlib

    for legacy_mod, canonical_mod in _COMPATIBILITY_MODULE_SHIMS:
        shim_file = _REPO / f"{legacy_mod.replace('.', '/')}.py"
        assert shim_file.is_file(), f"Missing shim file: {shim_file}"
        tree = ast.parse(shim_file.read_text(encoding="utf-8"), filename=str(shim_file))
        defs = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        assert (
            not defs
        ), f"Compatibility shim {shim_file} must not define classes/functions: {defs}"
        try:
            legacy = importlib.import_module(legacy_mod)
            canonical = importlib.import_module(canonical_mod)
        except ModuleNotFoundError as exc:
            missing_top = (exc.name or "").split(".")[0]
            if missing_top in _OPTIONAL_TOP_LEVEL_DEPS:
                continue
            raise
        assert (
            legacy is canonical
        ), f"{legacy_mod} ({legacy!r}) is not identical to {canonical_mod} ({canonical!r})"

    for scene_file in (_PACKAGE / "scenes").rglob("*.py"):
        tree = ast.parse(
            scene_file.read_text(encoding="utf-8"), filename=str(scene_file)
        )
        defs = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        assert not defs, f"Closed facade {scene_file} must not define new logic: {defs}"


def test_scenes_package_binds_library_submodule_in_fresh_process() -> None:
    """Legacy package attribute access (``myosuite.scenes.library``, ``myosuite.terms.*``, ``myosuite.utils.*``) works in a fresh interpreter."""
    import sys

    code = (
        "import myosuite, myosuite.scenes as s; "
        "assert hasattr(s, 'library') and s.library.SCENES; "
        "assert myosuite.utils.spec_processing is not None; "
        "assert myosuite.utils.asset_path_resolver is not None; "
        "assert myosuite.terms.waypoint is not None; "
        "assert myosuite.terms.opponent_relative_obs is not None; "
        "assert myosuite.terms.multiplayer.heading is not None"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
