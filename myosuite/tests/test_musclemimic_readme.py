# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The MuscleMimic README's code blocks import names that exist."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier1

_REPO_ROOT = Path(__file__).resolve().parents[2]
_README = (
    Path(__file__).resolve().parents[1] / "integrations" / "musclemimic" / "README.md"
)
_OPTIONAL_DEPS = frozenset(
    {"jax", "mjlab", "mujoco_warp", "onnxruntime", "torch", "warp"}
)


def _resolve_module_file(module_name: str) -> Path | None:
    rel = Path(*module_name.split("."))
    py_file = _REPO_ROOT / rel.with_suffix(".py")
    if py_file.is_file():
        return py_file
    init_file = _REPO_ROOT / rel / "__init__.py"
    if init_file.is_file():
        return init_file
    return None


def _defined_top_level_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _missing_imports_in_text(text: str) -> tuple[list[str], int]:
    missing: list[str] = []
    runtime_checked = 0
    for block in re.findall(r"```python\n(.*?)```", text, re.S):
        for node in ast.walk(ast.parse(block)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "myosuite"
            ):
                try:
                    module = importlib.import_module(node.module)
                except ModuleNotFoundError as exc:
                    dep = (exc.name or "").split(".")[0]
                    if (
                        dep not in _OPTIONAL_DEPS
                        or importlib.util.find_spec(dep) is not None
                    ):
                        raise
                    mod_file = _resolve_module_file(node.module)
                    if mod_file is None:
                        missing.extend(f"{node.module}.{a.name}" for a in node.names)
                        continue
                    defined = _defined_top_level_names(mod_file)
                    missing.extend(
                        f"{node.module}.{a.name}"
                        for a in node.names
                        if a.name not in defined
                    )
                else:
                    runtime_checked += 1
                    missing.extend(
                        f"{node.module}.{a.name}"
                        for a in node.names
                        if not hasattr(module, a.name)
                    )
    return missing, runtime_checked


def test_readme_blocks_parse_and_import_existing_names() -> None:
    text = _README.read_text(encoding="utf-8")
    missing, runtime_checked = _missing_imports_in_text(text)
    assert not missing
    assert runtime_checked > 0
    assert (
        "goal_params={}" not in text
    )  # raises: the adapter needs the mimic site names


def test_readme_import_check_handles_absent_optional_dep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module
    real_find_spec = importlib.util.find_spec

    def fake_import(name: str, package: str | None = None):
        if "mjlab" in name:
            raise ModuleNotFoundError("No module named 'mjlab'", name="mjlab")
        return real_import(name, package)

    def fake_find_spec(name: str, package: str | None = None):
        if name == "mjlab":
            return None
        return real_find_spec(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    missing, runtime_checked = _missing_imports_in_text(
        _README.read_text(encoding="utf-8")
    )
    assert not missing
    assert runtime_checked >= 8

    bad_snippet = (
        "```python\nfrom myosuite.integrations.musclemimic.mjlab_onnx_policy "
        "import NonExistentSymbol\n```"
    )
    bad_missing, _ = _missing_imports_in_text(bad_snippet)
    assert bad_missing == [
        "myosuite.integrations.musclemimic.mjlab_onnx_policy.NonExistentSymbol"
    ]
