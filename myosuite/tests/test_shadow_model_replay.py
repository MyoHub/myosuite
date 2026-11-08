# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""The shadow-model replay script runs end to end and writes its log."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier1

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "shadow_model_replay.py"


def test_replay_script_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    argv = [
        str(_SCRIPT),
        "--env",
        "myoElbowPose1D6MRandom-v0",
        "--duration",
        "2",
        "--log-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(_SCRIPT), run_name="__main__")
    assert "solved 166/166" in capsys.readouterr().out
    assert (tmp_path / "shadow_model.csv").exists() and (
        tmp_path / "shadow_model.json"
    ).exists()
