# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tutorial 5.7 stays on make_env and the scored FBVs id."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier1

_NB = (
    Path(__file__).resolve().parents[2]
    / "tutorials"
    / "5.7_ChaseTag_Steered_Teacher.ipynb"
)


def _source() -> str:
    nb = json.loads(_NB.read_text())
    return "\n".join("".join(c.get("source", [])) for c in nb["cells"])


def test_tutorial_57_uses_make_env_and_scored_fbvs() -> None:
    text = _source()
    assert "from myosuite import make_env" in text
    assert "myoChallengeChaseTagFBVs-v0" in text
    assert "gym.make" not in text
    assert "MYOSUITE_MUSCLEMIMIC_MODELS_VERSION" in text
    assert "ChaseTagSteeredPair" in text
