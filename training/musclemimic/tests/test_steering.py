# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for the steered teacher's 8-bin compass."""

from __future__ import annotations

import numpy as np
import pytest

from musclemimic_training.steering import (
    COMPASS_LABELS,
    body_yaw,
    compass_index,
    compass_label,
    step_compass,
)
from myosuite.terms.multiplayer.heading import heading_yaw, role_heading_xy


def test_compass_bins_of_role_headings() -> None:
    me = np.array([0.0, 0.0])
    opp = np.array([2.0, 0.0])
    toward = role_heading_xy(me, opp, pursue=True)
    away = role_heading_xy(me, opp, pursue=False)
    assert body_yaw(toward) == pytest.approx(heading_yaw(toward) + np.pi / 2)
    assert compass_index(heading_yaw(toward)) == COMPASS_LABELS.index("E")
    assert compass_index(heading_yaw(away)) == COMPASS_LABELS.index("W")
    assert compass_label(compass_index(np.pi / 2)) == "N"


def test_step_compass_turns_at_most_one_bin() -> None:
    assert step_compass(0, 0) == 0
    assert step_compass(0, 1) == 1
    assert step_compass(0, 2) == 1
    assert step_compass(0, 7) == 7
    assert step_compass(0, 4) in (1, 7)
