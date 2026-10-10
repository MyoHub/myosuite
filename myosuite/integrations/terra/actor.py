# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The MyoFullBody actor TERRA-4B was trained and evaluated on.

TERRA pins ``musclemimic_models`` 1.0.5, applies two muscle-wrap corrections
(``musclemimic.utils.model_fixes``) and evaluates in native MuJoCo with the MJX
solver settings (4 iterations, 8 line-search iterations, no Euler damping).
"""

from __future__ import annotations

from collections.abc import Callable

import mujoco

from myosuite.integrations.musclemimic.fullbody_model import (
    build_mimic_fullbody_spec,
    default_mimic_fullbody_config,
)

TERRA_MODELS_VERSION = "1.0.5"
# Terrain geoms (and the floor) are in this group: the heightmap rays see only it.
TERRAIN_GROUP = 2
_BICLONG_SIDESITE_Y = 0.020
_GASLAT_WRAP_RADIUS = 0.052


def apply_terra_model_fixes(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """TERRA's wrap corrections (``apply_myofullbody_model_fixes``), in place."""
    for name in (
        "BIClong_ellipsoid_BIClong_2_sidesite",
        "BIClong_ellipsoid_BIClong_2_sidesite_left",
    ):
        site = spec.site(name)
        if site is not None:
            site.pos[1] = _BICLONG_SIDESITE_Y
    for name in ("GasLat_at_shank_r_wrap", "GasLat_at_shank_l_wrap"):
        geom = spec.geom(name)
        if geom is not None:
            geom.size[0] = _GASLAT_WRAP_RADIUS
    return spec


def terra_fullbody_spec(
    scene_fn: Callable[[mujoco.MjSpec], object] | None = None,
) -> mujoco.MjSpec:
    """Edited TERRA actor spec, plus an optional scene edit (terrain in :data:`TERRAIN_GROUP`).

    Raises:
        ImportError: Without ``musclemimic_models`` (the native fallback model is
            not the actor TERRA was trained on).
    """
    try:
        import musclemimic_models  # noqa: F401, PLC0415
    except ImportError as err:
        raise ImportError(
            "TERRA needs the musclemimic_models MyoFullBody: pip install 'myosuite[musclemimic]'"
        ) from err
    cfg = default_mimic_fullbody_config()
    cfg.model_version = TERRA_MODELS_VERSION
    spec, _ = build_mimic_fullbody_spec(cfg)
    apply_terra_model_fixes(spec)
    spec.option.timestep = float(cfg.sim_dt)
    spec.option.iterations = 4
    spec.option.ls_iterations = 8
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    if scene_fn is not None:
        scene_fn(spec)
    return spec
