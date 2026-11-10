# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Scripted chase-tag steering: the 8-bin compass of the steered teacher.

Controller code for :mod:`~myosuite.integrations.musclemimic.steered_teacher`,
not a term function; the role and fence-aware headings are in
:mod:`myosuite.terms.multiplayer.heading`.
"""

from __future__ import annotations

import math

import numpy as np

from myosuite.terms.multiplayer.heading import heading_yaw

COMPASS_LABELS: tuple[str, ...] = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")
COMPASS_STEP = math.pi / 4.0
# MyoFullBody pelvis forward is +Y, so root yaw is heading + 90°.
PELVIS_FORWARD_OFFSET = math.pi / 2.0


def body_yaw(heading_xy: np.ndarray) -> float:
    """Root yaw that aims the pelvis forward along *heading_xy*."""
    return heading_yaw(heading_xy) + PELVIS_FORWARD_OFFSET


def compass_index(yaw: float) -> int:
    """Nearest 45° compass bin in ``[0, 7]`` (0 = east)."""
    return int(round(yaw / COMPASS_STEP)) % 8


def compass_label(idx: int) -> str:
    """Compass label for a bin index."""
    return COMPASS_LABELS[int(idx) % 8]


def step_compass(current: int, want: int) -> int:
    """Move at most one 45° bin toward *want* (shortest arc)."""
    cur = int(current) % 8
    d = (int(want) - cur + 4) % 8 - 4
    if d > 0:
        return (cur + 1) % 8
    if d < 0:
        return (cur - 1) % 8
    return cur
