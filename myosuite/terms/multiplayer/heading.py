# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Planar role heading for chase-tag observations and rewards (pursue / flee).

The scripted steering controller lives in
:mod:`myosuite.integrations.musclemimic.steering`.
"""

from __future__ import annotations

import math

import numpy as np


def role_heading_xy(
    self_xy: np.ndarray,
    opponent_xy: np.ndarray,
    *,
    pursue: bool,
) -> np.ndarray:
    """Unit XY heading toward the opponent (chaser) or away (runner)."""
    delta = (
        np.asarray(opponent_xy, dtype=np.float64)[:2]
        - np.asarray(self_xy, dtype=np.float64)[:2]
    )
    if not pursue:
        delta = -delta
    n = float(np.linalg.norm(delta))
    if n < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float64)
    return delta / n


def heading_yaw(heading_xy: np.ndarray) -> float:
    """World-frame yaw of a planar heading vector."""
    v = np.asarray(heading_xy, dtype=np.float64)
    return float(math.atan2(float(v[1]), float(v[0])))
