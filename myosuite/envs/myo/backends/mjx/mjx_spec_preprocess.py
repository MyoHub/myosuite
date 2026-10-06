# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Shared :class:`mujoco.MjSpec` preprocessing for MJX environments.

Call :func:`preprocess_mjx_spec` after loading or building a spec and before
``compile()`` so JAX/XLA and Warp backends both see consistent collision rules.
"""

from __future__ import annotations

import mujoco


def preprocess_mjx_spec(spec: mujoco.MjSpec, impl: str | None = None) -> mujoco.MjSpec:
    """Normalize collision rules before compiling an MJCF spec for MJX.

    - Sets every geom ``margin`` to ``0.0`` so the JAX/XLA backend can compile
      models that pair planes, meshes, heightfields, etc.
    - When *impl* is not ``\"warp\"``, clears contacts on **cylinder** and
      **ellipsoid** geoms. The JAX/XLA stack does not implement every collision
      primitive that Warp supports; disabling these pairs avoids compile or
      runtime failures while capsule-based collision (for example on the leg
      walk model) remains active.

    When *impl* is ``\"warp\"``, cylinder and ellipsoid contacts are left
    unchanged (margins are still zeroed).

    Args:
        spec: Mutable spec loaded from XML or assembled in memory.
        impl: Backend string passed to :func:`mujoco.mjx.put_model`
            (``None``, ``\"jax\"``, or ``\"warp\"``).

    Returns:
        The same *spec* instance, mutated in place.
    """
    use_jax_collision_rules = impl != "warp"
    for geom in spec.geoms:
        geom.margin = 0.0
        if use_jax_collision_rules and geom.type in (
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        ):
            geom.conaffinity = 0
            geom.contype = 0
    return spec


__all__ = ["preprocess_mjx_spec"]
