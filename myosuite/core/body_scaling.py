# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Compatibility facade for myo_sim's copied-spec subject scaling."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Any

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from myosuite.core.retargeting import retarget

Scales = Mapping[str, Any] | str | os.PathLike[str]


def _library(module: str = "scaling") -> ModuleType:
    try:
        return importlib.import_module(f"myo_sim.{module}")
    except ModuleNotFoundError as exc:
        if exc.name != f"myo_sim.{module}":
            raise
        raise ImportError(
            "Subject scaling requires myo_sim's scaling API (myo_sim PR #149). "
            "Install a myo_sim build containing that API; published 0.2.5 lacks it."
        ) from exc


def scale_spec(
    spec: mujoco.MjSpec, scales: Scales, *, total_mass: float | None = None
) -> mujoco.MjSpec:
    """Return a scaled copy, preserving the source spec and element order.

    Args:
        spec: Nominal source model.
        scales: Body factors, point correspondences, or an OpenSim scale file.
        total_mass: Optional target mass of the selected subject trees, in kg.

    Returns:
        The independent specification returned by ``myo_sim.scale_spec``.
    """
    return _library().scale_spec(spec, scales, total_mass=total_mass)


def scale_model(
    spec: mujoco.MjSpec, scales: Scales, *, total_mass: float | None = None
) -> mujoco.MjModel:
    """Compile a scaled copy using myo_sim's scaling rules.

    Args:
        spec: Nominal source model, which remains unchanged.
        scales: Body factors, point correspondences, or an OpenSim scale file.
        total_mass: Optional target subject mass in kg.

    Returns:
        The compiled scaled model.
    """
    return scale_spec(spec, scales, total_mass=total_mass).compile()


def _plain_scales(scales: Scales) -> dict[str, Any] | str:
    if isinstance(scales, (str, os.PathLike)):
        return os.fspath(scales)
    return {
        body: [np.asarray(v, dtype=float).tolist() for v in value]
        if isinstance(value, (tuple, list))
        and len(value) == 2
        and np.ndim(value[0]) == 2
        else np.asarray(value, dtype=float).tolist()
        for body, value in scales.items()
    }


def _model_digest(model: mujoco.MjModel) -> str:
    return _library().model_digest(model)


def _geom_bottom(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> float:
    return float(_library()._geom_bottom(model, data, index))


def _factors(scale: float | ArrayLike) -> np.ndarray:
    return _library()._factors(scale)


def _body_scales(
    spec: mujoco.MjSpec,
    scales: Mapping[str, float | ArrayLike],
    segments: Mapping[str, tuple[Sequence[str], Sequence[int]]] | None,
) -> dict[str, np.ndarray]:
    if segments is None:
        return _library().scale_report(spec, scales).factors
    factors = {name: _library()._factors(value) for name, value in scales.items()}
    resolved = _library("opensim_scales")._body_factors(spec, factors, segments)
    return {
        name: value
        for name, value in resolved.items()
        if not np.array_equal(value, np.ones(3))
    }


def __getattr__(name: str) -> Any:
    if name == "_scale_spec":
        return scale_spec
    if name in {
        "_factors",
        "_fit_points",
        "_frame_rot",
        "_geom_bottom",
        "_quat",
        "_rot",
    }:
        return getattr(_library(), name)
    raise AttributeError(name)


__all__ = ["retarget", "scale_model", "scale_spec"]
