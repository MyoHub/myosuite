# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Planar heading helpers for chase-tag (pursue / flee, 8-bin compass)."""

from __future__ import annotations

import math

import numpy as np

COMPASS_LABELS: tuple[str, ...] = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")
COMPASS_STEP = math.pi / 4.0
# Fence panels in chasetag_arena.xml sit at ±6 m.
ARENA_LIM_M = 6.0
WALL_MARGIN_M = 2.0
ARENA_PAD_M = 0.4


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


def wall_repulsion(
    xy: np.ndarray,
    *,
    arena_lim: float = ARENA_LIM_M,
    margin: float = WALL_MARGIN_M,
) -> np.ndarray:
    """Inward XY push when *xy* is within *margin* of a square fence."""
    p = np.asarray(xy, dtype=np.float64)[:2]
    v = np.zeros(2, dtype=np.float64)
    edge = float(arena_lim) - float(margin)
    for i in range(2):
        if p[i] > edge:
            v[i] -= (p[i] - edge) / max(float(margin), 1e-6)
        elif p[i] < -edge:
            v[i] += (-edge - p[i]) / max(float(margin), 1e-6)
    return v


def clamp_arena_xy(
    xy: np.ndarray,
    *,
    arena_lim: float = ARENA_LIM_M,
    pad: float = ARENA_PAD_M,
) -> np.ndarray:
    """Clip planar position to the interior of the chase-tag fence."""
    bound = float(arena_lim) - float(pad)
    out = np.asarray(xy, dtype=np.float64)[:2].copy()
    return np.clip(out, -bound, bound)


def _drop_outward(
    xy: np.ndarray,
    heading: np.ndarray,
    *,
    arena_lim: float,
    margin: float,
) -> np.ndarray:
    """Zero heading components that point through a nearby fence."""
    p = np.asarray(xy, dtype=np.float64)[:2]
    v = np.asarray(heading, dtype=np.float64)[:2].copy()
    edge = float(arena_lim) - float(margin)
    for i in range(2):
        if p[i] > edge:
            v[i] = min(v[i], 0.0)
        elif p[i] < -edge:
            v[i] = max(v[i], 0.0)
    return v


def aim_heading_xy(
    self_xy: np.ndarray,
    opponent_xy: np.ndarray,
    *,
    pursue: bool,
    arena_lim: float = ARENA_LIM_M,
    margin: float = WALL_MARGIN_M,
    wall_weight: float = 2.5,
) -> np.ndarray:
    """Role heading mixed with wall repulsion so walkers stay inside the arena.

    Outward components are stripped near a fence. If that cancels the heading,
    walk along the wall, then straight inward — never back through the fence.
    """
    role = role_heading_xy(self_xy, opponent_xy, pursue=pursue)
    v = role + wall_weight * wall_repulsion(self_xy, arena_lim=arena_lim, margin=margin)
    v = _drop_outward(self_xy, v, arena_lim=arena_lim, margin=margin)
    n = float(np.linalg.norm(v))
    if n > 1e-6:
        return v / n
    along = _drop_outward(self_xy, role, arena_lim=arena_lim, margin=margin)
    n_along = float(np.linalg.norm(along))
    if n_along > 1e-6:
        return along / n_along
    inward = wall_repulsion(self_xy, arena_lim=arena_lim, margin=margin)
    n_in = float(np.linalg.norm(inward))
    if n_in > 1e-6:
        return inward / n_in
    return role
