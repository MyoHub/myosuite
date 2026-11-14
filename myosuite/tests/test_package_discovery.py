# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The wheel ships only the ``myosuite`` package, not the repo's other folders."""

from __future__ import annotations

from pathlib import Path

from packaging.specifiers import SpecifierSet
import pytest
from setuptools import find_namespace_packages

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = pytest.importorskip("tomli")

pytestmark = pytest.mark.tier1

_REPO = Path(__file__).resolve().parents[2]


def test_package_discovery_is_limited_to_myosuite() -> None:
    pyproject = _REPO / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("needs a source checkout")
    find = tomllib.loads(pyproject.read_text())["tool"]["setuptools"]["packages"][
        "find"
    ]
    # pyproject's `find` is namespace-aware, so docs/, scripts/ etc. would match.
    packages = find_namespace_packages(
        where=str(_REPO),
        include=find.get("include", ("*",)),
        exclude=find.get("exclude", ()),
    )
    stray = sorted({p.split(".")[0] for p in packages} - {"myosuite"})
    assert not stray, f"top-level packages besides myosuite: {stray}"
    assert "myosuite.agents" not in packages


def test_python_310_remains_installable_until_mujoco_4() -> None:
    pyproject = _REPO / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("needs a source checkout")
    project = tomllib.loads(pyproject.read_text())["project"]
    req_py = SpecifierSet(project["requires-python"])
    assert req_py.contains("3.10")
    assert req_py.contains("3.11")
    assert "Programming Language :: Python :: 3.10" in project["classifiers"]
    assert "Programming Language :: Python :: 3.11" in project["classifiers"]


def test_ci_and_docs_default_to_python_311_and_test_310_to_315() -> None:
    workflow = _REPO / ".github" / "workflows" / "python-app.yml"
    docs_wf = _REPO / ".github" / "workflows" / "docs.yml"
    rtd = _REPO / ".readthedocs.yaml"
    if not (workflow.is_file() and docs_wf.is_file() and rtd.is_file()):
        pytest.skip("needs a source checkout")
    text = workflow.read_text(encoding="utf-8")
    include_line = next(line for line in text.splitlines() if "include:" in line)
    pr_matrix, full_matrix = include_line.split("||", maxsplit=1)
    assert '"python-version":"3.11"' in pr_matrix
    assert '"python-version":"3.10"' not in pr_matrix
    for ver in ("3.10", "3.11", "3.14", "3.15"):
        assert f'"python-version":"{ver}"' in full_matrix
    assert "matrix.python-version == '3.11'" in text
    assert 'python-version: "3.11"' in docs_wf.read_text(encoding="utf-8")
    assert 'python: "3.11"' in rtd.read_text(encoding="utf-8")
