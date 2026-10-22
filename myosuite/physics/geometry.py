# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Array-backend-neutral primitive support geometry."""

from __future__ import annotations

from typing import Any


def primitive_support_extent(shape: str, size: Any, direction: Any, xp: Any) -> Any:
    """Return a centred primitive's extent along a local unit direction.

    Args:
        shape: Sphere, capsule, cylinder, box or ellipsoid.
        size: MuJoCo-style shape dimensions, final axis three.
        direction: Unit direction in the geom frame, final axis three.
        xp: NumPy or Torch array module; leading axes broadcast.
    """
    if shape == "sphere":
        return xp.ones_like(direction[..., 0]) * size[..., 0]
    if shape == "capsule":
        return size[..., 0] + size[..., 1] * xp.abs(direction[..., 2])
    if shape == "cylinder":
        return size[..., 1] * xp.abs(direction[..., 2]) + size[..., 0] * xp.sqrt(
            xp.sum(direction[..., :2] ** 2, axis=-1)
        )
    if shape == "box":
        return xp.sum(xp.abs(direction) * size, axis=-1)
    if shape == "ellipsoid":
        return xp.sqrt(xp.sum((direction * size) ** 2, axis=-1))
    raise NotImplementedError(f"Unsupported primitive support shape {shape!r}.")
