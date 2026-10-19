# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Split-level ADL apartment scene for TERRA full-body tracking demos."""

from __future__ import annotations

import mujoco
import numpy as np

TERRAIN_GROUP = 2
FURNITURE_GROUP = 3
FLOOR_HEIGHT = 2.38
STAIR_COUNT = 14
STAIR_RISE = FLOOR_HEIGHT / STAIR_COUNT
STAIR_RUN = 0.28
STAIR_X = 3.0
STAIR_Y0 = 0.65


def _box(
    spec: mujoco.MjSpec,
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
    *,
    group: int = FURNITURE_GROUP,
    collide: bool = True,
) -> None:
    # Rotate the architectural plan 180 degrees so entry motion starts along
    # TERRA's native -Y forward heading; this avoids an artificial spawn turn.
    world_pos = (-pos[0], -pos[1], pos[2])
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        group=group,
        pos=world_pos,
        size=size,
        contype=1 if collide else 0,
        conaffinity=1 if collide else 0,
        condim=3,
        friction=[0.8, 0.005, 0.0001],
        rgba=rgba,
    )


def adl_apartment_scene(spec: mujoco.MjSpec) -> None:
    """Add a furnished split-level home with unambiguous TERRA terrain rays."""
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            geom.group = TERRAIN_GROUP
            geom.material = ""
            geom.rgba = [0.72, 0.66, 0.56, 1.0]
            geom.friction = [0.8, 0.005, 0.0001]

    wall = (0.84, 0.82, 0.78, 1.0)
    wood = (0.42, 0.25, 0.14, 1.0)
    stone = (0.72, 0.72, 0.69, 1.0)
    fabric = (0.28, 0.34, 0.38, 1.0)
    mattress = (0.88, 0.86, 0.80, 1.0)

    _box(spec, "wall_left", (-2.5, 1.0, 1.25), (0.08, 6.7, 1.25), wall)
    _box(spec, "wall_right", (5.2, 1.0, 1.25), (0.08, 6.7, 1.25), wall)
    _box(spec, "wall_front_l", (-1.4, -5.6, 1.25), (1.1, 0.08, 1.25), wall)
    _box(spec, "wall_front_r", (3.9, -5.6, 1.25), (1.3, 0.08, 1.25), wall)

    _box(spec, "sofa", (-1.25, -2.2, 0.42), (0.55, 1.15, 0.42), fabric)
    _box(spec, "coffee_table", (0.15, -2.4, 0.22), (0.55, 0.38, 0.22), wood)
    _box(spec, "dining_table", (1.2, -3.8, 0.38), (0.8, 0.55, 0.38), wood)
    for i, (x, y) in enumerate(
        ((0.2, -3.8), (2.2, -3.8), (1.2, -3.0), (1.2, -4.6))
    ):
        _box(spec, f"dining_chair_{i}", (x, y, 0.42), (0.25, 0.25, 0.42), wood)

    _box(spec, "kitchen_back_counter", (4.45, -3.1, 0.46), (0.55, 2.0, 0.46), stone)
    _box(spec, "kitchen_island", (3.75, -4.45, 0.46), (0.55, 0.75, 0.46), stone)

    for i in range(STAIR_COUNT):
        height = (i + 1) * STAIR_RISE
        y = STAIR_Y0 + (i + 0.5) * STAIR_RUN
        _box(
            spec,
            f"stair_{i:02d}",
            (STAIR_X, y, height / 2),
            (0.62, STAIR_RUN / 2, height / 2),
            wood,
            group=TERRAIN_GROUP,
        )

    stair_end = STAIR_Y0 + STAIR_COUNT * STAIR_RUN
    _box(
        spec,
        "bedroom_floor",
        (0.9, stair_end + 1.5, FLOOR_HEIGHT / 2),
        (2.7, 1.5, FLOOR_HEIGHT / 2),
        (0.58, 0.43, 0.29, 1.0),
        group=TERRAIN_GROUP,
    )
    _box(
        spec,
        "upper_railing",
        (1.1, stair_end + 0.08, FLOOR_HEIGHT + 0.55),
        (1.25, 0.05, 0.55),
        wall,
    )
    _box(
        spec,
        "bed_frame",
        (-0.8, stair_end + 1.75, FLOOR_HEIGHT + 0.28),
        (0.9, 1.05, 0.28),
        wood,
    )
    _box(
        spec,
        "mattress",
        (-0.8, stair_end + 1.75, FLOOR_HEIGHT + 0.52),
        (0.86, 1.0, 0.18),
        mattress,
    )


def adl_route() -> np.ndarray:
    """Continuous route: entry, kitchen, stairs, bed, downstairs, sofa."""
    stair_end = STAIR_Y0 + STAIR_COUNT * STAIR_RUN
    stair_mid = STAIR_Y0 + 0.5 * STAIR_COUNT * STAIR_RUN
    canonical = np.array(
        [
            [0.0, -5.0],
            [0.0, -3.4],
            [-0.7, -1.0],
            [1.0, -1.0],
            [2.8, -2.0],
            [3.0, -3.0],
            [3.0, -2.0],
            [3.0, 0.35],
            [3.0, stair_mid],
            [3.0, stair_end + 0.25],
            [1.5, stair_end + 1.0],
            [0.25, stair_end + 1.35],
            [1.5, stair_end + 1.0],
            [3.0, stair_end + 0.25],
            [3.0, stair_mid],
            [3.0, 0.35],
            [1.2, -0.8],
            [-0.4, -1.4],
        ],
        dtype=float,
    )
    canonical[:, :2] *= -1
    return canonical
