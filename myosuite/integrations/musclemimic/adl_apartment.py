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
FLOOR_HEIGHT = 1.40
STAIR_COUNT = 14
STAIR_RISE = FLOOR_HEIGHT / STAIR_COUNT
STAIR_RUN = 0.28
STAIR_X = 3.0
STAIR_Y0 = 0.65


def _material(spec, name, rgb1, rgb2, texrepeat):
    """Create a high-contrast tiled procedural material without external assets."""
    spec.add_texture(
        name=f"{name}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=rgb1, rgb2=rgb2,
        width=256, height=256,
    )
    textures = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE)
    textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = f"{name}_tex"
    spec.add_material(
        name=f"{name}_mat", textures=textures, rgba=[1, 1, 1, 1],
        texrepeat=texrepeat, texuniform=True,
    )
    return f"{name}_mat"


def _box(spec, name, pos, size, rgba, *, group=FURNITURE_GROUP, collide=True, material=""):
    world_pos = (-pos[0], -pos[1], pos[2])
    geom = spec.worldbody.add_geom(
        name=name, type=mujoco.mjtGeom.mjGEOM_BOX, group=group, pos=world_pos,
        size=size, contype=1 if collide else 0, conaffinity=1 if collide else 0,
        condim=3, friction=[0.8, 0.005, 0.0001], rgba=rgba,
    )
    if material:
        geom.material = material


def adl_apartment_scene(spec: mujoco.MjSpec) -> None:
    """Add a furnished split-level home with unambiguous TERRA terrain rays."""
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            geom.group = TERRAIN_GROUP
            geom.material = ""
            geom.rgba = [0.72, 0.66, 0.56, 1.0]
            geom.friction = [0.8, 0.005, 0.0001]

    wood_mat = _material(spec, "adl_wood", [0.22, 0.08, 0.025], [0.68, 0.38, 0.12], [5, 2])
    stone_mat = _material(spec, "adl_stone", [0.38, 0.40, 0.42], [0.84, 0.84, 0.80], [4, 4])
    fabric_mat = _material(spec, "adl_fabric", [0.10, 0.18, 0.24], [0.38, 0.55, 0.62], [8, 8])
    mattress_mat = _material(spec, "adl_mattress", [0.68, 0.67, 0.61], [0.98, 0.96, 0.88], [10, 10])

    wall = (0.84, 0.82, 0.78, 1.0)
    white = (1.0, 1.0, 1.0, 1.0)

    _box(spec, "wall_left", (-2.5, 1.0, 1.25), (0.08, 6.7, 1.25), wall)
    _box(spec, "wall_right", (5.2, 1.0, 1.25), (0.08, 6.7, 1.25), wall)
    _box(spec, "wall_front_l", (-1.4, -5.6, 1.25), (1.1, 0.08, 1.25), wall)
    _box(spec, "wall_front_r", (3.9, -5.6, 1.25), (1.3, 0.08, 1.25), wall)
    _box(spec, "sofa", (-1.25, -2.2, 0.42), (0.55, 1.15, 0.42), white, material=fabric_mat)
    _box(spec, "coffee_table", (-1.5, -0.2, 0.22), (0.55, 0.38, 0.22), white, material=wood_mat)
    _box(spec, "dining_table", (1.2, -3.8, 0.38), (0.8, 0.55, 0.38), white, material=wood_mat)
    for i, (x, y) in enumerate(((-1.5, -3.8), (2.2, -3.8), (1.2, -3.0), (1.2, -4.6))):
        _box(spec, f"dining_chair_{i}", (x, y, 0.42), (0.25, 0.25, 0.42), white, material=wood_mat)
    _box(spec, "kitchen_back_counter", (4.45, -3.1, 0.46), (0.55, 2.0, 0.46), white, material=stone_mat)
    _box(spec, "kitchen_island", (3.75, -4.45, 0.46), (0.55, 0.75, 0.46), white, material=stone_mat)

    for i in range(STAIR_COUNT):
        height = (i + 1) * STAIR_RISE
        y = STAIR_Y0 + (i + 0.5) * STAIR_RUN
        _box(spec, f"stair_{i:02d}", (STAIR_X, y, height / 2),
             (0.62, STAIR_RUN / 2, height / 2), white,
             group=TERRAIN_GROUP, material=wood_mat)

    stair_end = STAIR_Y0 + STAIR_COUNT * STAIR_RUN
    _box(spec, "bedroom_floor", (0.9, stair_end + 1.5, FLOOR_HEIGHT / 2),
         (2.7, 1.5, FLOOR_HEIGHT / 2), white, group=TERRAIN_GROUP, material=wood_mat)
    _box(spec, "upper_railing", (1.1, stair_end + 0.08, FLOOR_HEIGHT + 0.55),
         (1.25, 0.05, 0.55), wall)
    _box(spec, "bed_frame", (-0.8, stair_end + 1.75, FLOOR_HEIGHT + 0.28),
         (0.9, 1.05, 0.28), white, material=wood_mat)
    _box(spec, "mattress", (-0.8, stair_end + 1.75, FLOOR_HEIGHT + 0.52),
         (0.86, 1.0, 0.18), white, material=mattress_mat)


def adl_route() -> np.ndarray:
    """Clear aisle from entry through kitchen, stairs, bedroom and back."""
    stair_end = STAIR_Y0 + STAIR_COUNT * STAIR_RUN
    stair_mid = STAIR_Y0 + 0.5 * STAIR_COUNT * STAIR_RUN
    return np.array([
        [0.0, 5.0], [0.0, 3.4], [0.0, 1.0],
        [-0.7, 0.7], [-1.4, 0.0], [-2.2, -0.35], [-3.0, -0.35],
        [-3.0, -stair_mid], [-3.0, -(stair_end + 0.25)],
        [-1.5, -(stair_end + 1.0)], [-1.0, -(stair_end + 1.35)],
        [-1.5, -(stair_end + 1.0)], [-3.0, -(stair_end + 0.25)],
        [-3.0, -stair_mid], [-3.0, -0.35], [-2.2, -0.35],
        [-1.4, 0.0], [-0.7, 0.7], [0.0, 1.0],
    ], dtype=float)
