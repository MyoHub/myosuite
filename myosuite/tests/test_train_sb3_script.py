# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Smoke test of the CPU training script ``scripts/train_sb3.py``."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_sb3.py"


def _load():
    spec = importlib.util.spec_from_file_location("train_sb3", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_sb3"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("algo", ["ppo", "sac"])
def test_train_sb3_saves_a_loadable_policy(tmp_path: Path, algo: str) -> None:
    train_sb3 = _load()
    out = tmp_path / "model"
    zip_path = train_sb3.main(
        [
            "myoFingerReachFixed-v0",
            "--algo",
            algo,
            "--timesteps",
            "300" if algo == "sac" else "256",
            "--eval-episodes",
            "1",
            "--out",
            str(out),
        ]
    )
    assert zip_path == tmp_path / "model.zip" and zip_path.is_file()
    model = train_sb3.ALGOS[algo].load(zip_path)
    assert model.action_space.shape[0] > 0
