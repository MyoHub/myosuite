# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Compatibility facade for myo_sim's muscle strength transforms and MVC calibration."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType

import mujoco
import numpy as np

from myosuite.core.muscle_conditions import _peak_force as _peak_force


@dataclass(frozen=True)
class MaxTorqueMeasurement:
    """One measured maximal voluntary joint torque.

    Attributes:
        joint: Name of the measured joint (the master of any couplings).
        qpos: ``(nq,)`` posture of the trial; coupled joints are projected.
        torque: Gravity-corrected maximal torque [N m], positive.
        direction: ``+1`` for torque along the joint's positive axis, ``-1``
            for the opposite direction (e.g. extension of a flexion joint).
    """

    joint: str
    qpos: np.ndarray
    torque: float
    direction: float = 1.0


@dataclass(frozen=True)
class StrengthCalibration:
    """Fitted strength factors and the torques they reproduce.

    Attributes:
        factors: ``(nu,)`` strength factor per actuator (1 for non-muscles,
            ``prior`` for muscles no measurement involves).
        involved: ``(nu,)`` muscles whose force acts on some measurement.
        measured: ``(m,)`` measured torques [N m].
        nominal: ``(m,)`` maximal torques of the unscaled model [N m].
        predicted: ``(m,)`` maximal torques with ``factors`` applied [N m].
    """

    factors: np.ndarray
    involved: np.ndarray
    measured: np.ndarray
    nominal: np.ndarray
    predicted: np.ndarray


def _library(*, need_mvc: bool = False) -> ModuleType:
    try:
        mod = importlib.import_module("myo_sim.physics.muscle_strength")
    except ModuleNotFoundError as exc:
        if exc.name != "myo_sim.physics.muscle_strength":
            raise
        prs = (
            "MyoHub/myo_sim#186 and MyoHub/myo_sim#195"
            if need_mvc
            else "MyoHub/myo_sim#186"
        )
        raise ImportError(
            f"Muscle strength requires a myo-sim build containing {prs} "
            "(myo_sim.physics.muscle_strength). Install that development build "
            "or a release containing it; 0.2.5 lacks this API."
        ) from exc
    if need_mvc and not hasattr(mod, "calibrate_muscle_strength"):
        raise ImportError(
            "MVC torque probing and strength calibration require a myo-sim build "
            "containing MyoHub/myo_sim#195 (myo_sim.physics.muscle_strength). "
            "Install that development build or a release containing it."
        )
    return mod


def scale_muscle_strength(model: mujoco.MjModel, factors: float | np.ndarray) -> None:
    """Multiply each muscle's peak active and passive force in place.

    Args:
        model: Compiled model.
        factors: Positive finite scalar, or ``(nu,)`` per-actuator factors.

    Raises:
        ValueError: If a muscle's factor is not positive and finite.
        ImportError: If the installed myo-sim lacks ``myo_sim.physics.muscle_strength``.
    """
    _library().scale_muscle_strength(model, factors)


def scale_muscle_strength_spec(
    spec: mujoco.MjSpec, factors: float | np.ndarray
) -> mujoco.MjSpec:
    """Multiply each muscle's peak active and passive force on an ``MjSpec``.

    Args:
        spec: Model specification (modified in place and returned).
        factors: Positive finite scalar, or ``(nu,)`` per-actuator factors.

    Returns:
        ``spec``.

    Raises:
        ValueError: If a muscle's factor is not positive and finite.
        ImportError: If the installed myo-sim lacks ``myo_sim.physics.muscle_strength``.
    """
    return _library().scale_muscle_strength_spec(spec, factors)


def strength_factor_from_body_mass(mass: float, reference_mass: float) -> float:
    """Strength factor of a subject relative to the model's reference subject.

    Args:
        mass: Subject body mass [kg].
        reference_mass: Body mass of the model's reference subject [kg].

    Returns:
        ``mass / reference_mass``.

    Raises:
        ValueError: If either mass is not positive and finite.
        ImportError: If the installed myo-sim lacks the MVC calibration API.
    """
    return float(
        _library(need_mvc=True).strength_factor_from_body_mass(mass, reference_mass)
    )


def max_torque_terms(
    model: mujoco.MjModel, measurement: MaxTorqueMeasurement
) -> tuple[np.ndarray, np.ndarray, float]:
    """Each actuator's active and passive torque, and the joints' passive torque, in a trial.

    Args:
        model: Compiled model.
        measurement: The trial (joint, posture, direction).

    Returns:
        ``(active, passive, joints)`` torques [N m].

    Raises:
        ValueError: If the model or trial violates the steady MVC contract.
        ImportError: If the installed myo-sim lacks the MVC calibration API.
    """
    mod = _library(need_mvc=True)
    trial = mod.MaxTorqueMeasurement(
        joint=measurement.joint,
        qpos=measurement.qpos,
        torque=measurement.torque,
        direction=measurement.direction,
    )
    active, passive, joints = mod.max_torque_terms(model, trial)
    return active, passive, float(joints)


def calibrate_muscle_strength(
    model: mujoco.MjModel,
    measurements: Sequence[MaxTorqueMeasurement],
    *,
    prior: float = 1.0,
    regularization: float = 1e-3,
    bounds: tuple[float, float] = (0.25, 4.0),
) -> StrengthCalibration:
    """Fit per-muscle strength factors to measured maximal torques.

    Args:
        model: Compiled model (not modified).
        measurements: Trials; at least one.
        prior: Factor of muscles no trial involves, and target of regularization.
        regularization: Weight of the prior against relative torque errors.
        bounds: Strictly ordered positive lower and upper bounds of each factor.

    Returns:
        Fitted factors and measured, nominal and predicted torques.

    Raises:
        ValueError: If trials, hyperparameters or model force paths are invalid.
        ImportError: If the installed myo-sim lacks the MVC calibration API.
    """
    mod = _library(need_mvc=True)
    trials = [
        mod.MaxTorqueMeasurement(
            joint=m.joint, qpos=m.qpos, torque=m.torque, direction=m.direction
        )
        for m in measurements
    ]
    res = mod.calibrate_muscle_strength(
        model,
        trials,
        prior=prior,
        regularization=regularization,
        bounds=bounds,
    )
    return StrengthCalibration(
        factors=res.factors,
        involved=res.involved,
        measured=res.measured,
        nominal=res.nominal,
        predicted=res.predicted,
    )


__all__ = [
    "MaxTorqueMeasurement",
    "StrengthCalibration",
    "calibrate_muscle_strength",
    "max_torque_terms",
    "scale_muscle_strength",
    "scale_muscle_strength_spec",
    "strength_factor_from_body_mass",
]
