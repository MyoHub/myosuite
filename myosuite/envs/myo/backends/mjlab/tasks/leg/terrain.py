# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Height-field terrains of the CPU ``LegTerrainEnvV0`` for the mjlab leg twins.

The CPU env rewrites the terrain at every reset. MuJoCo Warp shares one height field
between all envs, so the twin bakes one terrain (fixed seed for ``rough``) into the
model instead; the ``hilly`` and ``stairs`` CPU registrations use their fixed variant
and are reproduced exactly.
"""

from __future__ import annotations

import mujoco
import numpy as np

_TERRAIN_SEED = 0  # the CPU env draws a new rough terrain at every reset
_N = 10_000  # height-field samples (100 x 100)


def terrain_heights(
    terrain: str, variant: str | None, rng: np.random.Generator
) -> np.ndarray:
    """Height-field elevations (metres, flat ``(10000,)``) as ``LegTerrainEnvV0.reset``.

    Args:
        terrain: ``"rough"``, ``"hilly"`` or ``"stairs"``.
        variant: ``"fixed"`` for the fixed hilly/stairs parameters, else sampled.
        rng: Random generator for the rough terrain and non-fixed variants.

    Returns:
        The elevations the CPU env writes into ``model.hfield_data``.
    """
    if terrain == "rough":
        rough = rng.uniform(low=-0.5, high=0.5, size=(_N,))
        normalized = (rough - rough.min()) / (rough.max() - rough.min())
        return normalized * 0.08 - 0.02
    if terrain == "hilly":
        flat_length, frequency = 3000, 3
        scalar = 0.63 if variant == "fixed" else float(rng.uniform(0.53, 0.73))
        combined = np.concatenate(
            (
                -2 * np.ones(flat_length),
                -2
                + 0.5
                * (
                    np.sin(
                        np.linspace(0, frequency * np.pi, _N - flat_length) + np.pi / 2
                    )
                    - 1
                ),
            )
        )
        normalized = (combined - combined.min()) / (combined.max() - combined.min())
        return np.flip(normalized.reshape(100, 100) * scalar, [0, 1]).reshape(_N)
    if terrain == "stairs":
        num_stairs, stair_height = 12, 0.1
        flat = 5200 - (_N - 5200) % num_stairs
        stairs_width = (_N - flat) // num_stairs
        scalar = 2.5 if variant == "fixed" else float(rng.uniform(1.5, 3.5))
        parts = [
            np.full((int(stairs_width // 100), 100), -2 + stair_height * j)
            for j in range(num_stairs)
        ]
        data = np.concatenate([np.full((int(flat // 100), 100), -2)] + parts, axis=0)
        normalized = (data + 2) / (2 + stair_height * num_stairs)
        return np.flip(normalized.reshape(100, 100) * scalar, [0, 1]).reshape(_N)
    raise ValueError(f"Unknown terrain {terrain!r}")


def add_terrain(spec: mujoco.MjSpec, terrain: str, variant: str | None) -> None:
    """Bake the terrain into the entity spec, visible and colliding as after a CPU reset.

    MuJoCo normalises height-field data to ``[0, 1]`` at compile time, so the elevation
    range goes into the field's z size and its minimum into the geom's z position.

    Args:
        spec: Entity spec of the leg model (with its ``terrain`` height field).
        terrain: ``"rough"``, ``"hilly"`` or ``"stairs"``.
        variant: CPU ``variant`` of the registration.
    """
    heights = terrain_heights(terrain, variant, np.random.default_rng(_TERRAIN_SEED))
    field = spec.hfield("terrain")
    field.userdata = heights
    rx, ry, _, base = field.size
    field.size = [rx, ry, float(heights.max() - heights.min()), base]
    geom = spec.geom("terrain")
    geom.pos = np.array([0.0, 0.0, float(heights.min())])
    geom.rgba[3] = 1.0
    geom.contype = geom.conaffinity = 1
