# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tutorial 5.7 plays the steered myoSuperMimic teacher in the scored FBVs game through make_env."""

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


def _code() -> str:
    nb = json.loads(_NB.read_text())
    return "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )


def test_tutorial_57_plays_the_supermimic_teacher() -> None:
    text = _code()
    assert "from myosuite import make_env" in text
    assert "myoChallengeChaseTagFBVs-v0" in text
    assert "gym.make" not in text
    assert "w_gait_jpos=0.0, w_gait_jvel=0.0, w_steering=0.0" in text
    assert "SmPlayer" in text and "load_library" in text
    assert 'MIMIC_DATASET = os.environ.get("MYOSUITE_MIMIC_DATASET", "")' in text
    for old in (
        "ChaseTagSteeredPair",
        "MYOSUITE_MUSCLEMIMIC_MODELS_VERSION",
        "mm-10m-2",
        "amathislab",
        "integrations.musclemimic",
    ):
        assert old not in text, old
