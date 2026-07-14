# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Shared geometry data and sampling for SAR reorient environments.

Provides 8-object, 100-object, in-distribution, and out-of-distribution variants.
MuJoCo geom types: 3=capsule, 4=ellipsoid, 5=cylinder, 6=box.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Geometry variants live in the sibling JSON asset reorient_sar_geometries.json
# to keep this module readable (the ID/OOD variants have 1000 entries each).
# JSON stringifies the integer object indices, so they are restored to int on
# load. Each variant is {shape: {index: [[size_3d], [rgba_4d]]}}.
# ---------------------------------------------------------------------------

import json
from pathlib import Path

_GEOMETRIES_PATH = Path(__file__).with_suffix(".json")
_VARIANT_NAMES = ("GEOMETRIES_8", "GEOMETRIES_ID", "GEOMETRIES_OOD", "GEOMETRIES_100")


def _load_geometries() -> dict[str, dict]:
    """Load all geometry variants from the JSON asset, restoring int indices."""
    raw = json.loads(_GEOMETRIES_PATH.read_text())
    return {
        name: {
            shape: {int(k): v for k, v in inner.items()}
            for shape, inner in raw[name].items()
        }
        for name in _VARIANT_NAMES
    }


GEOMETRIES_8, GEOMETRIES_ID, GEOMETRIES_OOD, GEOMETRIES_100 = (
    _load_geometries().values()
)


def _sample_from_variant(
    rng: np.random.Generator,
    variant: dict[str, dict[int, list]],
    override_color_8: bool = False,
) -> tuple[int, list, list, np.ndarray, np.ndarray, np.ndarray]:
    """Sample geom_type, size, color, top_pos, bot_pos, desired_euler from a variant dict."""
    geom_type = int(rng.choice([3, 4, 5, 6]))
    if geom_type == 3:
        _name, d = "caps", variant["caps"]
    elif geom_type == 4:
        _name, d = "ellips", variant["ellips"]
    elif geom_type == 5:
        _name, d = "cyl", variant["cyl"]
    else:
        _name, d = "box", variant["box"]
    ind = rng.integers(0, len(d))
    size, color = d[ind][0], list(d[ind][1])
    if geom_type == 3:
        top_pos = np.array([0.0, 0.0, 1.3 * size[1]])
        bot_pos = np.array([0.0, 0.0, -1.3 * size[1]])
    elif geom_type == 4:
        top_pos = np.array([0.0, 0.0, size[2]])
        bot_pos = np.array([0.0, 0.0, -size[2]])
    elif geom_type == 5:
        top_pos = np.array([0.0, 0.0, size[1]])
        bot_pos = np.array([0.0, 0.0, -size[1]])
    else:
        top_pos = np.array([0.0, 0.0, size[2]])
        bot_pos = np.array([0.0, 0.0, -size[2]])
    if override_color_8:
        color = [1.0, 0.9, 0.0, 1.0]
    desired_euler = np.zeros(3)
    desired_euler[0] = float(rng.uniform(low=-1, high=1))
    desired_euler[1] = float(rng.uniform(low=-0.8, high=1.2))
    return geom_type, size, color, top_pos, bot_pos, desired_euler


def sample_geometry_8(
    rng: np.random.Generator,
) -> tuple[int, list, list, np.ndarray, np.ndarray, np.ndarray]:
    """Sample object geometry for 8-object variant.

    Returns:
        geom_type, size, color, top_pos, bot_pos, desired_euler.
    """
    return _sample_from_variant(rng, GEOMETRIES_8, override_color_8=True)


def sample_geometry_100(
    rng: np.random.Generator,
) -> tuple[int, list, list, np.ndarray, np.ndarray, np.ndarray]:
    """Sample object geometry for 100-object variant.

    Returns:
        geom_type, size, color, top_pos, bot_pos, desired_euler.
    """
    return _sample_from_variant(rng, GEOMETRIES_100, override_color_8=False)


def sample_geometry_from_variant(
    rng: np.random.Generator,
    variant: dict[str, dict[int, list]],
    override_color: list[float] | None = None,
    two_draws: bool = False,
) -> tuple[int, list, list, np.ndarray, np.ndarray, np.ndarray]:
    """Sample geometry from a variant dict (ID/OOD style).

    Same return as _sample_from_variant. When two_draws is True, size and color
    are sampled from two independent indices (legacy InDistribution/OutofDistribution).
    override_color, when set, replaces the sampled color (e.g. ID green, OOD red).
    """
    geom_type = int(rng.choice([3, 4, 5, 6]))
    if geom_type == 3:
        _name, d = "caps", variant["caps"]
    elif geom_type == 4:
        _name, d = "ellips", variant["ellips"]
    elif geom_type == 5:
        _name, d = "cyl", variant["cyl"]
    else:
        _name, d = "box", variant["box"]
    if two_draws:
        ind_size = int(rng.integers(0, len(d)))
        ind_color = int(rng.integers(0, len(d)))
        size = list(d[ind_size][0])
        color = list(d[ind_color][1])
    else:
        ind = int(rng.integers(0, len(d)))
        size, color = list(d[ind][0]), list(d[ind][1])
    if geom_type == 3:
        top_pos = np.array([0.0, 0.0, 1.3 * size[1]])
        bot_pos = np.array([0.0, 0.0, -1.3 * size[1]])
    elif geom_type == 4:
        top_pos = np.array([0.0, 0.0, size[2]])
        bot_pos = np.array([0.0, 0.0, -size[2]])
    elif geom_type == 5:
        top_pos = np.array([0.0, 0.0, size[1]])
        bot_pos = np.array([0.0, 0.0, -size[1]])
    else:
        top_pos = np.array([0.0, 0.0, size[2]])
        bot_pos = np.array([0.0, 0.0, -size[2]])
    if override_color is not None:
        color = list(override_color)
    desired_euler = np.zeros(3)
    desired_euler[0] = float(rng.uniform(low=-1, high=1))
    desired_euler[1] = float(rng.uniform(low=-0.8, high=1.2))
    return geom_type, size, color, top_pos, bot_pos, desired_euler


# Legacy fixed colors, scaled from 0-255 RGB to 0-1.
_ID_COLOR = [38 / 255, 194 / 255, 129 / 255, 255 / 255]
_OOD_COLOR = [128 / 255, 0 / 255, 0 / 255, 255 / 255]


def sample_geometry_id(
    rng: np.random.Generator,
) -> tuple[int, list, list, np.ndarray, np.ndarray, np.ndarray]:
    """Sample object geometry for the in-distribution test variant.

    Returns:
        geom_type, size, color, top_pos, bot_pos, desired_euler.
    """
    return sample_geometry_from_variant(
        rng, GEOMETRIES_ID, override_color=_ID_COLOR, two_draws=True
    )


def sample_geometry_ood(
    rng: np.random.Generator,
) -> tuple[int, list, list, np.ndarray, np.ndarray, np.ndarray]:
    """Sample object geometry for the out-of-distribution test variant.

    Returns:
        geom_type, size, color, top_pos, bot_pos, desired_euler.
    """
    return sample_geometry_from_variant(
        rng, GEOMETRIES_OOD, override_color=_OOD_COLOR, two_draws=True
    )
