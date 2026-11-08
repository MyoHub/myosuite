# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The wheel ships only the ``myosuite`` package, not the repo's other folders."""

from __future__ import annotations

from pathlib import Path

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
