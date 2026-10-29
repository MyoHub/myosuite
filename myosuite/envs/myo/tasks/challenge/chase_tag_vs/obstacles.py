# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Discrete box obstacles for the two-agent chase-tag arena.

A fixed pool of box geoms is added to the ``MjSpec`` at build time. Each
reset enables a random subset, places them with rejection sampling, and
hides the rest under the floor (same trick as FLAT ChaseTag terrain).
"""

from __future__ import annotations

import mujoco
import numpy as np

N_OBSTACLE_SLOTS = 6
HIDDEN_Z = -10.0
OBSTACLE_RGBA = [0.55, 0.32, 0.18, 1.0]


def add_obstacle_geoms(spec: mujoco.MjSpec, n_slots: int = N_OBSTACLE_SLOTS) -> None:
    """Add hidden box geoms ``obstacle_0`` … ``obstacle_{n-1}`` to *spec*."""
    for i in range(n_slots):
        g = spec.worldbody.add_geom(name=f"obstacle_{i}")
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [0.3, 0.3, 0.6]
        g.pos = [0.0, 0.0, HIDDEN_Z]
        g.rgba = list(OBSTACLE_RGBA)
        g.contype = 0
        g.conaffinity = 0


def hide_obstacle(model: mujoco.MjModel, geom_id: int) -> None:
    """Disable collision and hide one obstacle slot under the floor."""
    model.geom_pos[geom_id] = [0.0, 0.0, HIDDEN_Z]
    model.geom_contype[geom_id] = 0
    model.geom_conaffinity[geom_id] = 0
    model.geom_rgba[geom_id, 3] = 0.0


def _hfield_z(heightfield, x: float, y: float) -> float:
    if heightfield is None:
        return 0.0
    ix, iy = heightfield.cart2map(np.asarray([[x]]), np.asarray([[y]]))
    elev = float(heightfield.hfield.data[int(ix.squeeze()), int(iy.squeeze())])
    return elev * float(heightfield.hfield.size[2])


def place_obstacles(
    model: mujoco.MjModel,
    rng: np.random.Generator,
    geom_ids: list[int],
    spawn_xy: np.ndarray,
    *,
    n_range: tuple[int, int],
    half_xy_range: tuple[float, float],
    height_range: tuple[float, float],
    min_agent_dist_m: float,
    min_separation_m: float,
    arena_half_extent_m: float,
    heightfield=None,
    max_attempts: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place a random subset of the obstacle pool; hide unused slots.

    Returns:
        ``(xy, height, active)`` each length ``len(geom_ids)``. Unused slots
        have ``xy=0``, ``height=0``, ``active=False``.
    """
    n_slots = len(geom_ids)
    xy_out = np.zeros((n_slots, 2), dtype=np.float64)
    height_out = np.zeros(n_slots, dtype=np.float64)
    active = np.zeros(n_slots, dtype=bool)
    if n_slots == 0:
        return xy_out, height_out, active

    n_lo, n_hi = n_range
    n_lo = max(0, min(n_lo, n_slots))
    n_hi = max(n_lo, min(n_hi, n_slots))
    n_active = int(rng.integers(n_lo, n_hi + 1)) if n_hi >= n_lo else 0

    spawn = np.asarray(spawn_xy, dtype=float).reshape(-1, 2)
    spawn_patches = None
    if heightfield is not None:
        spawn_patches = {heightfield.patch_index(p[0], p[1]) for p in spawn}

    placed: list[np.ndarray] = []
    for i, gid in enumerate(geom_ids):
        if i >= n_active:
            hide_obstacle(model, gid)
            continue
        found = False
        for _ in range(max_attempts):
            x = float(rng.uniform(-arena_half_extent_m, arena_half_extent_m))
            y = float(rng.uniform(-arena_half_extent_m, arena_half_extent_m))
            pt = np.array([x, y], dtype=float)
            if np.any(np.linalg.norm(spawn - pt, axis=1) < min_agent_dist_m):
                continue
            if placed and np.any(
                np.linalg.norm(np.stack(placed) - pt, axis=1) < min_separation_m
            ):
                continue
            if (
                spawn_patches is not None
                and heightfield.patch_index(x, y) in spawn_patches
            ):
                continue
            half_xy = float(rng.uniform(*half_xy_range))
            height = float(rng.uniform(*height_range))
            half_z = 0.5 * height
            z0 = _hfield_z(heightfield, x, y)
            model.geom_size[gid] = [half_xy, half_xy, half_z]
            model.geom_pos[gid] = [x, y, z0 + half_z]
            model.geom_contype[gid] = 1
            model.geom_conaffinity[gid] = 1
            model.geom_rgba[gid] = OBSTACLE_RGBA
            placed.append(pt)
            xy_out[i] = pt
            height_out[i] = height
            active[i] = True
            found = True
            break
        if not found:
            hide_obstacle(model, gid)
    return xy_out, height_out, active
