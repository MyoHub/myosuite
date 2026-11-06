# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Colab Run-all contract for tutorials 5.4 and 5.6."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier1

_TUTORIALS = Path(__file__).resolve().parents[2] / "tutorials"
_NB54 = _TUTORIALS / "5.4_MuscleMimic_Directional_Locomotion.ipynb"
_NB56 = _TUTORIALS / "5.6_ChaseTag_MyoChallenge.ipynb"


def _src(nb: dict) -> str:
    return "\n".join("".join(c.get("source", [])) for c in nb["cells"])


@pytest.fixture(scope="module")
def nb54() -> dict:
    return json.loads(_NB54.read_text())


@pytest.fixture(scope="module")
def nb56() -> dict:
    return json.loads(_NB56.read_text())


def test_54_colab_loads_heading_table(nb54: dict) -> None:
    src = _src(nb54)
    assert "policy_heading_table.pt" in src
    assert "load_directional_policy" in src
    assert 'MYOSUITE_COLLECT_TEACHER", "0"' in src
    assert "COLLECT_TEACHER = os.environ.get" in src
    assert "1.0.6" in src
    meta = nb54.get("metadata", {})
    assert meta.get("accelerator") != "GPU"
    assert "A100" not in json.dumps(meta.get("colab", {}))


def test_54_does_not_load_raw_state_dict_on_shipped_ckpt(nb54: dict) -> None:
    src = _src(nb54)
    assert "eval_policy.load_state_dict" not in src
    assert "ActorCritic(obs_dim=OBS_DIM, act_dim=ACT_DIM)" not in src or (
        "load_directional_policy" in src
    )


def test_56_export_handles_heading_table(nb56: dict) -> None:
    src = _src(nb56)
    assert 'get("kind") == "heading_table"' in src
    assert "policy_heading_table.pt" in src
    assert "HeadingTablePolicy" in src
    assert "obs[:, 544:546]" in src or "role" in src
