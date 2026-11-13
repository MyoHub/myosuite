# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tutorial 3.3 assigns the reference trajectory to the model's joints by name."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from myosuite import make_env

pytestmark = pytest.mark.tier1

_TUTORIALS = Path(__file__).resolve().parents[2] / "tutorials"
_NB = _TUTORIALS / "3.3_Inverse_Dynamics.ipynb"
_CSV = _TUTORIALS / "files" / "3.3" / "trajectory.csv"


def test_tutorial_33_reads_the_trajectory_by_joint_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = json.loads(_NB.read_text(encoding="utf-8"))["cells"]
    (load_cell,) = (
        "".join(c["source"])
        for c in cells
        if c["cell_type"] == "code" and "read_csv" in "".join(c["source"])
    )
    env = make_env("myoHandPoseFixed-v0")
    try:
        model = env.unwrapped.model
        scope = {"pd": pd, "get_myohand_model": lambda: model}
        monkeypatch.chdir(_TUTORIALS)
        exec(load_cell.replace("FULL_ID = False", "FULL_ID = True"), scope)
        frame = pd.read_csv(_CSV)
        assert scope["traj"].shape == (len(frame), 1 + model.nq)
        for j in range(model.njnt):
            name = model.joint(j).name.removesuffix("_r")
            np.testing.assert_array_equal(
                scope["traj"][:, 1 + model.jnt_qposadr[j]], frame[name].to_numpy(), name
            )
    finally:
        env.close()
