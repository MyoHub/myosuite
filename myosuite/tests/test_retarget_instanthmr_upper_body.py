# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for upper-body and hand MHR70 retargeting onto fixed-base torso/arm models."""

from __future__ import annotations

from pathlib import Path

import mujoco
import myo_sim
import numpy as np
import pytest

from myosuite.biomechanics.retarget_instanthmr import (
    INSTANTHMR_MAPS,
    INSTANTHMR_MYOTORSO_ARMS,
    default_instanthmr_map,
    estimate_instanthmr_scales,
    instanthmr_source,
    solve_instanthmr_ik,
)

pytestmark = pytest.mark.tier1

_SAMPLE_NPZ = Path(__file__).parent / "data" / "instanthmr_mhr70_sample.npz"


def test_default_map_and_scaling_for_upper_body_models() -> None:
    spec = myo_sim.load_spec("myotorso_arms")
    model = spec.compile()

    imap = default_instanthmr_map(model)
    assert imap is INSTANTHMR_MYOTORSO_ARMS
    assert INSTANTHMR_MAPS["myotorso_arms"] is INSTANTHMR_MYOTORSO_ARMS
    assert INSTANTHMR_MAPS["myoarms"] is INSTANTHMR_MYOTORSO_ARMS

    scales = estimate_instanthmr_scales(_SAMPLE_NPZ, model, imap=imap)
    assert "pelvis" not in scales
    assert {"sacrum", "humerus_l", "humerus_r", "ulna_l", "ulna_r"} <= set(scales)
    assert all(0.5 <= v <= 2.0 for v in scales.values())

    _, lmap = instanthmr_source(model, imap)
    lm_names = {lm.source for lm in lmap.landmarks}
    assert {"L_IndexTip", "R_IndexTip", "L_ThumbTip", "R_ThumbTip"} <= lm_names

    res = solve_instanthmr_ik(_SAMPLE_NPZ, model=model, scale=False)
    assert res.qpos.shape == (len(res.targets.pos), model.nq)
    assert np.all(np.isfinite(res.qpos))
    assert np.all(np.isfinite(res.qvel))
    assert res.metrics["instanthmr_map"] == "instanthmr-mhr70->myotorso_arms"
    assert res.metrics["ground_shift_m"]["max"] == 0.0

    d = mujoco.MjData(model)
    d.qpos[:] = res.qpos[-1]
    mujoco.mj_forward(model, d)
    assert np.all(np.isfinite(d.xpos))
