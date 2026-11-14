# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The MuscleMimic README's code blocks import names that exist."""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier1

_README = (
    Path(__file__).resolve().parents[1] / "integrations" / "musclemimic" / "README.md"
)


def test_readme_blocks_parse_and_import_existing_names() -> None:
    text = _README.read_text(encoding="utf-8")
    missing = []
    for block in re.findall(r"```python\n(.*?)```", text, re.S):
        for node in ast.walk(ast.parse(block)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "myosuite"
            ):
                module = importlib.import_module(node.module)
                missing += [
                    f"{node.module}.{a.name}"
                    for a in node.names
                    if not hasattr(module, a.name)
                ]
    assert not missing
    assert (
        "goal_params={}" not in text
    )  # raises: the adapter needs the mimic site names
