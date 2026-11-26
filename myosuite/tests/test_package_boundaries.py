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
        "motion",
        "policies",
        "biomechanics",
        "integrations",
        "envs",
        "viz",
        "benchmarks",
        "logger",
        "utils",
    }
)

# Newly created subpackages must never accumulate legacy import exemptions.
_ZERO_EXEMPTION_SUBPACKAGES = frozenset({"motion", "policies", "benchmarks"})

# Out-of-wheel repository directories that runtime modules must never import.
_FORBIDDEN_EXTERNAL_IMPORTS = frozenset({"training", "tools", "scripts"})

# Allowed runtime imports between myosuite.<src> and myosuite.<dst> subpackages.
# "__root__" denotes myosuite/__init__.py and myosuite/version.py.
_ALLOWED_INTERNAL_IMPORTS: dict[str, frozenset[str]] = {
    # Legacy compatibility subpackages on `__root__` support lazy `import myosuite; myosuite.<pkg>.*`.
    "__root__": frozenset(
        {"__root__", "core", "envs", "logger", "scenes", "terms", "utils"}
    ),
    # Foundation & numerical helpers
    "core": frozenset({"__root__", "core", "physics"}),
    "physics": frozenset({"__root__", "physics"}),
    "scenes": frozenset({"__root__", "core", "scenes"}),
    "terms": frozenset({"__root__", "core", "physics", "terms"}),
    "motion": frozenset({"__root__", "core", "physics", "motion"}),
    "policies": frozenset(
        {"__root__", "core", "terms", "physics", "motion", "biomechanics", "policies"}
    ),
    # Solvers & lightweight source/ecosystem adapters
    "biomechanics": frozenset(
        {"__root__", "core", "physics", "motion", "biomechanics"}
    ),
    "integrations": frozenset(
        {
            "__root__",
            "core",
            "terms",
            "physics",
            "motion",
            "policies",
            "biomechanics",
            "integrations",
        }
    ),
    # Simulation environments, visualization & evaluation suites
    "envs": frozenset(
        {
            "__root__",
            "core",
            "scenes",
            "terms",
            "physics",
            "motion",
            "policies",
            "biomechanics",
            "integrations",
            "envs",
        }
    ),
    "viz": frozenset(
        {
            "__root__",
            "core",
            "scenes",
            "physics",
            "motion",
            "policies",
            "biomechanics",
            "envs",
            "viz",
        }
    ),
    "benchmarks": frozenset(
        {
            "__root__",
            "core",
            "terms",
            "physics",
            "motion",
            "policies",
            "biomechanics",
            "integrations",
            "envs",
            "viz",
            "benchmarks",
        }
    ),
    # Closed backward-compatibility facade namespaces
    "logger": frozenset({"__root__", "core", "physics", "motion", "viz", "logger"}),
    "utils": frozenset(
        {
            "__root__",
            "core",
            "terms",
            "physics",
            "motion",
            "policies",
            "biomechanics",
            "envs",
            "viz",
            "benchmarks",
            "utils",
        }
    ),
}

# Lazy backward-compatibility re-exports on core/ preserving former public import paths until v4.0.
_COMPATIBILITY_REEXPORT_IMPORTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("core/__init__.py", "integrations"),
        ("core/__init__.py", "motion"),
        ("core/body_scaling.py", "integrations"),
    }
)

# Pre-existing cross-layer runtime edges on dev (16 edges), burned down as migrations land.
# Do not add entries here; place new runtime logic in its canonical layer.
_LEGACY_CROSS_IMPORTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("biomechanics/retarget_instanthmr.py", "viz"),
        ("core/config.py", "envs"),
        ("core/model_builder.py", "envs"),
        ("core/model_recipes.py", "envs"),
        ("core/model_recipes.py", "integrations"),
        ("core/registry.py", "envs"),
        ("envs/gymnasium_env.py", "viz"),
        ("envs/heightfields.py", "viz"),
        ("envs/multi_agent_modular_env.py", "viz"),
        ("integrations/musclemimic/fbvs_player.py", "envs"),
        ("integrations/musclemimic/fullbody_native_playback.py", "envs"),
        ("integrations/musclemimic/fullbody_native_playback.py", "viz"),
        ("integrations/musclemimic/mjlab_onnx_policy.py", "envs"),
        ("terms/multiplayer/chase_tag_vs_obs.py", "envs"),
        ("terms/multiplayer/chase_tag_vs_reward.py", "envs"),
        ("viz/mjlab_play.py", "integrations"),
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
    ("myosuite.terms.catalog", "myosuite.benchmarks.term_catalog"),
    ("myosuite.core.trajectory_io", "myosuite.motion.io"),
    ("myosuite.core.trajectory_yaw", "myosuite.motion.transforms"),
    ("myosuite.core.playback_contract", "myosuite.motion.playback"),
    ("myosuite.core.hf_dataset_cache", "myosuite.motion.hf_cache"),
    ("myosuite.core.retargeting", "myosuite.integrations.marker_retarget"),
    ("myosuite.logger.reference_motion", "myosuite.motion.reference"),
    ("myosuite.logger.reference_motion_jax", "myosuite.motion.reference_jax"),
    ("myosuite.logger.examine_logs", "myosuite.viz.examine_logs"),
    ("myosuite.logger.examine_reference", "myosuite.viz.examine_reference"),
    ("myosuite.logger.grouped_datasets", "myosuite.viz.grouped_datasets"),
    ("myosuite.logger.roboset_logger", "myosuite.viz.roboset_logger"),
    ("myosuite.biomechanics.opensim_motion", "myosuite.integrations.opensim_motion"),
    ("myosuite.biomechanics.opensim_grf", "myosuite.integrations.opensim_grf"),
    ("myosuite.biomechanics.smplh", "myosuite.integrations.smplh"),
    ("myosuite.biomechanics.retarget_smplh", "myosuite.integrations.retarget_smplh"),
    ("myosuite.biomechanics.mano", "myosuite.integrations.mano"),
    ("myosuite.biomechanics.gvhmr", "myosuite.integrations.gvhmr"),
    ("myosuite.biomechanics.chumpy_pickle", "myosuite.integrations.chumpy_pickle"),
    ("myosuite.utils.action_map", "myosuite.policies.action_map"),
    ("myosuite.utils.checkpoint_manifest", "myosuite.policies.checkpoint_manifest"),
    ("myosuite.utils.checkpoint_utils", "myosuite.policies.checkpoint_utils"),
    ("myosuite.utils.deployment_manifest", "myosuite.policies.deployment_manifest"),
    ("myosuite.utils.export_onnx", "myosuite.policies.export_onnx"),
    ("myosuite.utils.onnx_checkpoint", "myosuite.policies.onnx_checkpoint"),
    ("myosuite.utils.onnxruntime_utils", "myosuite.policies.onnxruntime_utils"),
    ("myosuite.utils.rslrl_policy", "myosuite.policies.rslrl_policy"),
    ("myosuite.integrations.musclemimic.actor_tree", "myosuite.policies.actor_tree"),
    (
        "myosuite.integrations.musclemimic.fullbody_checkpoint_io",
        "myosuite.policies.checkpoint_ref",
    ),
    (
        "myosuite.integrations.musclemimic.fullbody_eval_cli",
        "myosuite.benchmarks.musclemimic_fullbody_eval_cli",
    ),
    ("myosuite.utils.examine_env", "myosuite.benchmarks.examine_env"),
    ("myosuite.utils.examine_sim", "myosuite.benchmarks.examine_sim"),
    ("myosuite.utils.env_check", "myosuite.benchmarks.env_check"),
    ("myosuite.utils.paired_replay", "myosuite.benchmarks.paired_replay"),
    ("myosuite.utils.performance_card", "myosuite.benchmarks.performance_card"),
    ("myosuite.utils.push_metrics", "myosuite.benchmarks.push_metrics"),
    ("myosuite.utils.push_protocol", "myosuite.benchmarks.push_protocol"),
    ("myosuite.utils.push_recovery", "myosuite.benchmarks.push_recovery"),
    ("myosuite.utils.push_results", "myosuite.benchmarks.push_results"),
    ("myosuite.utils.push_scheduling", "myosuite.benchmarks.push_scheduling"),
    ("myosuite.utils.dict_utils", "myosuite.core.dict_utils"),
    ("myosuite.utils.import_utils", "myosuite.core.import_utils"),
    ("myosuite.utils.path_utils", "myosuite.core.path_utils"),
    ("myosuite.utils.prompt_utils", "myosuite.core.prompt_utils"),
    ("myosuite.utils.reach_workspace", "myosuite.core.reach_workspace"),
    ("myosuite.utils.step_timing", "myosuite.core.step_timing"),
    ("myosuite.utils.target_ranges", "myosuite.core.target_ranges"),
    ("myosuite.utils.tensor_utils", "myosuite.core.tensor_utils"),
    ("myosuite.utils.feature_cli", "myosuite.envs.feature_cli"),
    ("myosuite.utils.html", "myosuite.viz.html_embed"),
    ("myosuite.utils.path_plotting", "myosuite.viz.path_plotting"),
    ("myosuite.utils.policy_utils", "myosuite.viz.policy_playback"),
    ("myosuite.utils.video_io", "myosuite.viz.video_io"),
)

_NON_SHIM_FACADE_TARGETS: dict[str, tuple[str, ...]] = {
    "physics/inverse_kinematics.py": ("myosuite.biomechanics.inverse_kinematics",),
    "scenes/__init__.py": (
        "myosuite.scenes",
        "myosuite.core.builder.scene_library",
    ),
    "terms/__init__.py": ("myosuite.core.terms", "myosuite.terms"),
    "logger/__init__.py": (),
    "utils/__init__.py": ("myosuite.core.seeding",),
    "utils/gym.py": (),
}

_COMPATIBILITY_FACADE_FILES: frozenset[str] = frozenset(
    {
        *(
            f"{legacy.removeprefix('myosuite.').replace('.', '/')}.py"
            for legacy, _ in _COMPATIBILITY_MODULE_SHIMS
        ),
        *_NON_SHIM_FACADE_TARGETS,
    }
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


def _runtime_imports(
    py_file: Path, *, include_type_checking: bool = False
) -> list[str]:
    """Extract import targets (including literal ``import_module(...)`` calls)."""
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(py_file))

    type_checking_nodes: set[ast.AST] = set()
    if not include_type_checking:
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
                    if node.module == "myosuite":
                        imports.append(node.module)
                        for alias in node.names:
                            if alias.name != "*":
                                imports.append(f"myosuite.{alias.name}")
                    elif node.module.startswith("myosuite."):
                        if any(a.name == "*" for a in node.names):
                            imports.append(node.module)
                        mod_rel = Path(
                            *node.module.removeprefix("myosuite.").split(".")
                        )
                        for alias in node.names:
                            if alias.name == "*":
                                continue
                            sub_py = _PACKAGE / mod_rel / f"{alias.name}.py"
                            sub_pkg = _PACKAGE / mod_rel / alias.name / "__init__.py"
                            if sub_py.is_file() or sub_pkg.is_file():
                                imports.append(f"{node.module}.{alias.name}")
                            else:
                                imports.append(node.module)
                                imports.append(f"{node.module}.{alias.name}")
                    else:
                        imports.append(node.module)
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
        elif (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(t, ast.Name)
                and t.id in {"_LAZY_EXPORTS", "_COMPAT_SUBMODULES"}
                for t in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
            )
            and node.value is not None
        ):
            for sub in ast.walk(node.value):
                if (
                    isinstance(sub, ast.Constant)
                    and isinstance(sub.value, str)
                    and sub.value.startswith("myosuite.")
                ):
                    imports.append(sub.value)
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


def _matches_allowed_prefix(mod: str, prefixes: tuple[str, ...]) -> bool:
    return any(mod == p or mod.startswith(f"{p}.") for p in prefixes)


def test_myosuite_internal_import_dag() -> None:
    """Enforce the upward module import DAG across ``myosuite/``."""
    for rel_file, dst_sub in _LEGACY_CROSS_IMPORTS:
        src_sub = rel_file.split("/", 1)[0]
        assert src_sub not in _ZERO_EXEMPTION_SUBPACKAGES, (
            f"Subpackage {src_sub!r} must have zero outgoing legacy exemptions: "
            f"({rel_file!r}, {dst_sub!r})"
        )

    shim_map = dict(_COMPATIBILITY_MODULE_SHIMS)
    facade_allowed_prefixes: dict[str, tuple[str, ...]] = {
        **{
            f"{legacy.removeprefix('myosuite.').replace('.', '/')}.py": (canonical,)
            for legacy, canonical in _COMPATIBILITY_MODULE_SHIMS
        },
        **_NON_SHIM_FACADE_TARGETS,
    }
    assert set(facade_allowed_prefixes) == _COMPATIBILITY_FACADE_FILES

    facade_violations: list[str] = []
    shim_imports: list[str] = []
    violations: set[tuple[str, str]] = set()
    for path in _runtime_python_files():
        rel = path.relative_to(_PACKAGE)
        rel_posix = rel.as_posix()
        if rel_posix in _COMPATIBILITY_FACADE_FILES:
            allowed_prefixes = facade_allowed_prefixes[rel_posix]
            for mod in sorted(set(_runtime_imports(path, include_type_checking=True))):
                if mod == "myosuite" or mod.startswith("myosuite."):
                    if not _matches_allowed_prefix(mod, allowed_prefixes):
                        facade_violations.append(
                            f"{rel_posix} -> {mod} (allowed: {allowed_prefixes})"
                        )
            continue
        src_sub = rel.parts[0] if len(rel.parts) > 1 else "__root__"
        allowed = _ALLOWED_INTERNAL_IMPORTS[src_sub]
        for mod in _runtime_imports(path):
            if mod in shim_map:
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

    assert not facade_violations, (
        "Compatibility facade files may only import their mapped canonical target(s):\n  "
        + "\n  ".join(facade_violations)
    )
    assert not shim_imports, (
        "Non-facade runtime modules must import canonical paths rather than compatibility shims:\n  "
        + "\n  ".join(sorted(shim_imports))
    )
    known_exemptions = _LEGACY_CROSS_IMPORTS | _COMPATIBILITY_REEXPORT_IMPORTS
    unexpected = sorted(violations - known_exemptions)
    assert not unexpected, (
        "Forbidden cross-subpackage runtime imports detected (see docs/wiki/repository-map.md):\n  "
        + "\n  ".join(f"{src} -> myosuite.{dst}" for src, dst in unexpected)
    )
    stale = sorted(known_exemptions - violations)
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

    for closed_dir in ("scenes", "logger"):
        on_disk = {
            p.relative_to(_PACKAGE).as_posix()
            for p in (_PACKAGE / closed_dir).rglob("*.py")
        }
        unlisted = sorted(on_disk - _COMPATIBILITY_FACADE_FILES)
        assert (
            not unlisted
        ), f"Closed namespace myosuite/{closed_dir}/ has non-facade files: {unlisted}"

    utils_on_disk = {
        p.relative_to(_PACKAGE).as_posix() for p in (_PACKAGE / "utils").rglob("*.py")
    }
    utils_allowed = _COMPATIBILITY_FACADE_FILES | {
        "utils/sb3_callbacks.py",
        "utils/sb3_sweep.py",
    }
    utils_unlisted = sorted(utils_on_disk - utils_allowed)
    assert (
        not utils_unlisted
    ), f"Closed namespace myosuite/utils/ has unexpected files: {utils_unlisted}"

    for rel_facade in sorted(_COMPATIBILITY_FACADE_FILES):
        facade_file = _PACKAGE / rel_facade
        assert facade_file.is_file(), f"Missing facade file: {facade_file}"
        tree = ast.parse(
            facade_file.read_text(encoding="utf-8"), filename=str(facade_file)
        )
        defs = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef))
            or (isinstance(node, ast.FunctionDef) and node.name != "__getattr__")
        ]
        assert (
            not defs
        ), f"Compatibility facade {facade_file} must not define new logic: {defs}"


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
