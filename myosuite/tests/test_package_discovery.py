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
from setuptools.config.pyprojecttoml import apply_configuration
from setuptools.dist import Distribution

from myosuite.tests.test_package_boundaries import _APPROVED_RUNTIME_SUBPACKAGES

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
    setuptools_cfg = tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"][
        "setuptools"
    ]
    find = setuptools_cfg["packages"]["find"]
    pkg_data = setuptools_cfg["package-data"]["myosuite"]
    excl_data = setuptools_cfg["exclude-package-data"]["myosuite"]

    assert setuptools_cfg.get("include-package-data") is False
    assert "myosuite.agents" in find["exclude"]
    assert "myosuite.tests" in find["exclude"]
    assert "myosuite.tests.*" in find["exclude"]
    assert "tests/**" in excl_data
    assert "tests/**" not in pkg_data

    # pyproject's `find` is namespace-aware, so docs/, scripts/ etc. would match.
    packages = find_namespace_packages(
        where=str(_REPO),
        include=find.get("include", ("*",)),
        exclude=find.get("exclude", ()),
    )
    stray = sorted({p.split(".")[0] for p in packages} - {"myosuite"})
    assert not stray, f"top-level packages besides myosuite: {stray}"
    assert "myosuite.agents" not in packages
    assert not any(
        p == "myosuite.tests" or p.startswith("myosuite.tests.") for p in packages
    )

    dist = apply_configuration(Distribution(), str(pyproject))
    discovered = set(dist.packages or ())
    assert "myosuite" in discovered
    assert any(pkg.startswith("myosuite.envs.") for pkg in discovered)
    assert any(pkg.startswith("myosuite.core") for pkg in discovered)
    assert any(".tasks" in pkg for pkg in discovered)
    assert any(".backends" in pkg for pkg in discovered)
    leaked = sorted(
        pkg
        for pkg in discovered
        if pkg == "myosuite.tests"
        or pkg.startswith("myosuite.tests.")
        or (
            pkg.startswith("myosuite.")
            and pkg.split(".")[1] not in _APPROVED_RUNTIME_SUBPACKAGES
        )
    )
    assert not leaked, f"Unapproved packages discovered for wheel: {leaked}"
    discovered_top = {
        pkg.split(".")[1]
        for pkg in discovered
        if pkg.startswith("myosuite.") and len(pkg.split(".")) >= 2
    }
    assert discovered_top == _APPROVED_RUNTIME_SUBPACKAGES, (
        f"Wheel discovery mismatch vs _APPROVED_RUNTIME_SUBPACKAGES: "
        f"{sorted(discovered_top)} != {sorted(_APPROVED_RUNTIME_SUBPACKAGES)}"
    )


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


def test_ci_and_docs_default_to_python_311_and_test_311_to_315() -> None:
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
    # CI starts at 3.11; 3.13 is the newest mjlab line and 3.15 the newest Python.
    for ver in ("3.11", "3.13", "3.15"):
        assert f'"python-version":"{ver}"' in full_matrix
    for ver in ("3.10", "3.14"):
        assert f'"python-version":"{ver}"' not in full_matrix
    assert "matrix.python-version == '3.11'" in text
    assert 'python-version: "3.11"' in docs_wf.read_text(encoding="utf-8")
    assert 'python: "3.11"' in rtd.read_text(encoding="utf-8")
