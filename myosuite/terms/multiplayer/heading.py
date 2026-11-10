# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Planar role heading for chase-tag observations, rewards and players (pursue / flee).

:func:`aim_heading_xy` adds the arena fence: the role heading mixed with a push off
nearby fence panels, so a steered player stays inside the ChaseTag arena.
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


ARENA_LIM_M = 6.0
"""Half-width (m) of the square ChaseTag fence (panels of ``chasetag_arena.xml`` at +-6 m)."""
WALL_MARGIN_M = 2.0
"""Distance (m) from the fence within which :func:`wall_repulsion` pushes inward."""
ARENA_PAD_M = 0.4
"""Clearance (m) :func:`clamp_arena_xy` keeps from the fence."""


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
    """Clip a planar position to the interior of the chase-tag fence."""
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
    walk along the wall, then straight inward, never back through the fence.

    Args:
        self_xy: The player's planar position.
        opponent_xy: The opponent's planar position.
        pursue: Toward the opponent (chaser) or away from it (runner).
        arena_lim: Half-width of the square fence (m).
        margin: Distance from the fence within which it repels (m).
        wall_weight: Weight of the repulsion against the unit role heading.

    Returns:
        Unit planar heading (float64, shape ``(2,)``).
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
