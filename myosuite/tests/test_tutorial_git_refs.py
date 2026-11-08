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
    # Ref variables (f-string placeholders like ``@{GIT_REF}`` resolve to these):
    # only an assignment at the start of a line, so ``print("GIT_REF =", ...)`` is ignored.
    re.compile(
        r"^[ \t]*(?:MYOSUITE_GIT_REF|MYOSUITE_REF|GIT_REF)[ \t]*=[ \t]*"
        r"(?:os\.environ\.get\([^,\n]+,[ \t]*)?[\"']([^\"'\n]+)",
        re.MULTILINE,
    ),
)


def _refs(text: str) -> set[str]:
    found = {m.group(1) for pattern in _REF_PATTERNS for m in pattern.finditer(text)}
    return {ref for ref in found if not ref.startswith("{")}


def test_ref_patterns_catch_a_feature_branch() -> None:
    sample = "\n".join(
        [
            "colab.research.google.com/github/MyoHub/myosuite/blob/feat/x/tutorials/a.ipynb",
            "git+https://github.com/MyoHub/myosuite.git@fix/y",
            'GIT_REF = os.environ.get("MYOSUITE_GIT_REF", "feature/z")',
            'MYOSUITE_REF = "wip"',
            'pip install "myosuite[musclemimic] @ git+https://github.com/MyoHub/myosuite.git@{MYOSUITE_REF}"',
            'print("GIT_REF =", GIT_REF)',
        ]
    )
    assert _refs(sample) == {"feat", "fix/y", "feature/z", "wip"}


def _text(path: Path) -> str:
    """Notebook cell sources (JSON-decoded), or the file itself for Markdown."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix != ".ipynb":
        return raw
    return "\n".join("".join(cell["source"]) for cell in json.loads(raw)["cells"])


@pytest.mark.parametrize(
    "path",
    sorted(_TUTORIALS.glob("*.ipynb")) + sorted(_TUTORIALS.glob("*.md")),
    ids=lambda p: p.name,
)
def test_tutorial_refs_are_main_or_dev(path: Path) -> None:
    if not _TUTORIALS.is_dir():
        pytest.skip("needs a source checkout")
    bad = _refs(_text(path)) - _ALLOWED
    assert not bad, f"{path.name} points at branches {sorted(bad)}; use main or dev"
