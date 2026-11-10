# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Colab Run-all contract for tutorials 5.4 and 5.6 (myoSuperMimic walker on myofullbody)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier1

_TUTORIALS = Path(__file__).resolve().parents[2] / "tutorials"
_NB54 = _TUTORIALS / "5.4_MuscleMimic_Directional_Locomotion.ipynb"
_NB56 = _TUTORIALS / "5.6_ChaseTag_MyoChallenge.ipynb"
_MUSCLEMIMIC = (
    "amathislab",
    "mm-10m-2",
    "integrations.musclemimic",
    "musclemimic_models",
    "fbvs_player",
)


def _code(nb: dict) -> str:
    return "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )


@pytest.fixture(scope="module")
def nb54() -> dict:
    return json.loads(_NB54.read_text())


@pytest.fixture(scope="module")
def nb56() -> dict:
    return json.loads(_NB56.read_text())


def test_54_trains_and_loads_the_myofullbody_walker(nb54: dict) -> None:
    src = _code(nb54)
    assert "chasetag_teacher.directional" in src
    assert 'download_baseline_file("directional/walker_myofullbody.pt")' in src
    assert (
        'download_baseline_file(\n    "supermimic/tracker_myofullbody_kit_30min.pt"'
        in src
    )
    assert 'MIMIC_DATASET = os.environ.get("MYOSUITE_MIMIC_DATASET", "")' in src
    assert 'QUICK_MODE = os.environ.get("MYOSUITE_FULL_DAGGER", "0") != "1"' in src
    assert not [w for w in _MUSCLEMIMIC if w in src]
    meta = nb54.get("metadata", {})
    assert meta.get("accelerator") != "GPU"


def test_56_exports_the_walker_and_offers_the_student(nb56: dict) -> None:
    src = _code(nb56)
    assert "export_chasetag_onnx" in src
    assert 'download_baseline_file("directional/walker_myofullbody.pt")' in src
    assert 'download_baseline_file("onnx/chasetag_supermimic_selfplay.onnx")' in src
    assert (
        '"submit-onnx", str(ONNX_PATH)' in src and '"-t", "chasetag", "-p", "1"' in src
    )
    assert "w_gait_jpos=0.0, w_gait_jvel=0.0, w_steering=0.0" in src
    assert "gym.make" not in src
    assert not [w for w in _MUSCLEMIMIC if w in src]
