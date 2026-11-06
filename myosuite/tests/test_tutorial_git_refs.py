# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tutorials point Colab, downloads and installs at ``main`` or ``dev`` only.

Feature branches are deleted after their PR merges, which breaks every Colab
badge, raw download and ``pip install git+...@branch`` that names them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier1

_TUTORIALS = Path(__file__).resolve().parents[2] / "tutorials"
_ALLOWED = {"main", "dev"}
_REF_PATTERNS = (
    re.compile(r"github/MyoHub/myosuite/blob/([^/\s)\"']+)/"),
    re.compile(r"raw\.githubusercontent\.com/MyoHub/myosuite/([^/\s)\"']+)/"),
    re.compile(r"github\.com/MyoHub/myosuite\.git@([^\s\"'#]+)"),
    # Ref variables (f-string placeholders like ``@{GIT_REF}`` resolve to these).
    re.compile(
        r"\b(?:MYOSUITE_GIT_REF|MYOSUITE_REF|GIT_REF)\b[^=\n]*=\s*"
        r"(?:os\.environ\.get\([^,]+,\s*)?[\"']([^\"']+)"
    ),
)


def _refs(text: str) -> set[str]:
    found = {m.group(1) for pattern in _REF_PATTERNS for m in pattern.finditer(text)}
    return {ref for ref in found if not ref.startswith("{")}


def test_ref_patterns_catch_a_feature_branch() -> None:
    sample = (
        "colab.research.google.com/github/MyoHub/myosuite/blob/feat/x/tutorials/a.ipynb "
        "git+https://github.com/MyoHub/myosuite.git@fix/y "
        'GIT_REF = os.environ.get("MYOSUITE_GIT_REF", "feature/z") '
        'MYOSUITE_REF = "wip" pip install "git+https://github.com/MyoHub/myosuite.git@{MYOSUITE_REF}"'
    )
    assert _refs(sample) == {"feat", "fix/y", "feature/z", "wip"}


@pytest.mark.parametrize(
    "notebook", sorted(_TUTORIALS.glob("*.ipynb")), ids=lambda p: p.name
)
def test_tutorial_refs_are_main_or_dev(notebook: Path) -> None:
    if not _TUTORIALS.is_dir():
        pytest.skip("needs a source checkout")
    cells = json.loads(notebook.read_text())["cells"]
    text = "\n".join("".join(cell["source"]) for cell in cells)
    bad = _refs(text) - _ALLOWED
    assert not bad, f"{notebook.name} points at branches {sorted(bad)}; use main or dev"
