# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""CHANGELOG.md keeps its PR/issue link definitions at the end of each release, in increasing order."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier1

_CHANGELOG = Path(__file__).parents[2] / "CHANGELOG.md"
_DEF = re.compile(r"^\[#(\d+)\]:\s*\S+")
_CITE = re.compile(r"\[#(\d+)\](?![:(])")


def _sections() -> list[list[str]]:
    sections: list[list[str]] = [[]]
    for line in _CHANGELOG.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            sections.append([])
        sections[-1].append(line)
    return sections


def test_link_definitions_close_each_release_in_increasing_order() -> None:
    defined: list[int] = []
    for lines in _sections():
        text = [ln for ln in lines if ln.strip()]
        defs = [int(m.group(1)) for ln in text if (m := _DEF.match(ln))]
        if not defs:
            continue
        head = text[0] if text else ""
        tail = [
            int(_DEF.match(ln).group(1)) for ln in text[-len(defs) :] if _DEF.match(ln)
        ]
        assert (
            tail == defs
        ), f"{head}: link definitions must be the last lines of the release"
        assert defs == sorted(
            defs
        ), f"{head}: link definitions must be in increasing order"
        defined += defs
    assert len(defined) == len(set(defined)), "a link is defined twice"


def test_every_cited_link_is_defined() -> None:
    text = _CHANGELOG.read_text(encoding="utf-8")
    defined = {int(m.group(1)) for m in re.finditer(r"^\[#(\d+)\]:", text, re.M)}
    cited = {int(n) for n in _CITE.findall(text)}
    assert not cited - defined, f"cited without a definition: {sorted(cited - defined)}"
