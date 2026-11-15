# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""
Muscle condition helpers: cumulative fatigue, sarcopenia, reafferentation.

These are pure transform functions that modify MuJoCo MjSpec or MjModel
to represent physiological muscle conditions. They are used by ModelBuilder
as transform callbacks and by term functions via EnvAccessor.

The 3CC-r fatigue model lives in :mod:`myo_sim.physics.fatigue`; its names are
re-exported here so existing imports keep working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from myo_sim.physics.fatigue import (  # noqa: F401
    _UNIFORM_PARAMS,
    FATIGUE_REST_THRESHOLD,
    MUSCLE_FATIGUE_PARAMS,
    CumulativeFatigue,
    TorchFatigueState,
    _muscle_group,
    _per_muscle_params,
    three_cc_r_step,
    torch_three_cc_r_step,
)

if TYPE_CHECKING:
    import mujoco


# ---------------------------------------------------------------------------
# Sarcopenia helpers
# ---------------------------------------------------------------------------


def _peak_force(model: mujoco.MjModel) -> np.ndarray:
    """Per-actuator peak force ``gainprm[:, 2]`` as MuJoCo applies it.

    A negative muscle peak force (``force="-1"``) means automatic: MuJoCo
    uses ``gainprm[3] / acc0`` at runtime, so scaling the stored value would
    leave the force unchanged.
    """
    import mujoco as _mujoco  # noqa: PLC0415

    force = np.array(model.actuator_gainprm[:, 2], dtype=np.float64)
    auto = (force < 0) & (model.actuator_gaintype == _mujoco.mjtGain.mjGAIN_MUSCLE)
    if np.any(auto):
        force[auto] = model.actuator_gainprm[auto, 3] / np.maximum(
            model.actuator_acc0[auto], _mujoco.mjMINVAL
        )
    return force


def apply_sarcopenia_to_spec(
    spec: mujoco.MjSpec, force_scale: float = 0.5
) -> mujoco.MjSpec:
    """Scale muscle peak forces to simulate sarcopenia (age-related muscle loss).

    Spec-level equivalent of :func:`apply_sarcopenia_to_model`: scales the
    *compiled* peak force ``gainprm[2]`` of every actuator. Muscles declared
    with ``force="-1"`` keep ``gainprm[2] = -1`` after compilation and get
    their peak force ``scale / acc0`` from the compiled model, so the spec is
    compiled once to resolve the peak forces, which are then written back
    explicitly (scaled).

    Args:
        spec: MuJoCo model spec to modify in-place.
        force_scale: Fraction of original peak force to retain (0–1).

    Returns:
        The modified spec (same object, modified in-place).

    Example:
        >>> import mujoco
        >>> spec = mujoco.MjSpec.from_file("elbow.xml")
        >>> apply_sarcopenia_to_spec(spec, force_scale=0.5)
    """
    peak_force = _peak_force(spec.compile())
    for actuator, force in zip(spec.actuators, peak_force, strict=True):
        actuator.gainprm[2] = float(force) * force_scale
    return spec


def apply_sarcopenia_to_model(model: mujoco.MjModel, force_scale: float = 0.5) -> None:
    """Scale muscle peak forces on a compiled MjModel in-place.

    Multiplies ``actuator_gainprm[:, 2]`` (the maximum isometric force Fmax)
    by *force_scale* for every actuator, as the original
    ``BaseV0.initializeConditions`` does. Muscles declared with
    ``force="-1"`` are first resolved to their peak force ``scale / acc0``.

    Args:
        model: Compiled MuJoCo model to modify.
        force_scale: Fraction of original peak force to retain (0–1).
    """
    model.actuator_gainprm[:, 2] = _peak_force(model) * force_scale
