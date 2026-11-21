# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Compatibility facade for ``myo_sim.physics.hill_muscle``."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover
    if TYPE_CHECKING:
        import mujoco


def _library() -> ModuleType:
    try:
        return importlib.import_module("myo_sim.physics.hill_muscle")
    except ModuleNotFoundError as exc:
        if exc.name != "myo_sim.physics.hill_muscle":
            raise
        raise ImportError(
            "Hill muscle curves and joint-torque spec stripping require a myo-sim "
            "build containing MyoHub/myo_sim#200 (myo_sim.physics.hill_muscle). "
            "Install that development build or a release containing it; 0.2.5 lacks this API."
        ) from exc


@dataclass(frozen=True)
class MuscleParams:
    """Per-muscle parameters extracted from a compiled ``mujoco.MjModel``.

    Attributes:
        tau_act: ``(nu,)`` activation time constants [s].
        tau_deact: ``(nu,)`` deactivation time constants [s].
        tausmooth: ``(nu,)`` smoothing widths.
        act_limited: ``(nu,)`` boolean mask for activation clamping.
        act_range: ``(nu, 2)`` lower and upper activation bounds.
        length_range: ``(nu, 2)`` actuator operating length range [m].
        acc0: ``(nu,)`` joint-space acceleration norms at ``qpos0``.
        gain_prm: ``(nu, 9)`` ``mjGAIN_MUSCLE`` parameters.
        bias_prm: ``(nu, 9)`` ``mjBIAS_MUSCLE`` parameters.
    """

    tau_act: np.ndarray
    tau_deact: np.ndarray
    tausmooth: np.ndarray
    act_limited: np.ndarray
    act_range: np.ndarray
    length_range: np.ndarray
    acc0: np.ndarray
    gain_prm: np.ndarray
    bias_prm: np.ndarray

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> MuscleParams:
        """Extract Hill muscle parameters from ``model``.

        Args:
            model: Compiled model whose actuators are Hill muscles.

        Returns:
            Extracted :class:`MuscleParams`.

        Raises:
            ValueError: If not all model actuators are Hill muscles.
            ImportError: If the installed ``myo-sim`` lacks ``myo_sim.physics.hill_muscle``.
        """
        res = _library().MuscleParams.from_model(model)
        return cls(
            tau_act=res.tau_act,
            tau_deact=res.tau_deact,
            tausmooth=res.tausmooth,
            act_limited=res.act_limited,
            act_range=res.act_range,
            length_range=res.length_range,
            acc0=res.acc0,
            gain_prm=res.gain_prm,
            bias_prm=res.bias_prm,
        )


def muscle_activation_dot(ctrl: Any, act: Any, params: MuscleParams) -> Any:
    """Compute Millard ``mjDYN_MUSCLE`` activation derivative ``da/dt``.

    Args:
        ctrl: Muscle excitation input (NumPy array or PyTorch tensor).
        act: Current muscle activation state.
        params: Per-muscle parameters.

    Returns:
        Time derivative ``da/dt`` matching ``mju_muscleDynamics``.

    Raises:
        ImportError: If the installed ``myo-sim`` lacks ``myo_sim.physics.hill_muscle``.
    """
    return _library().muscle_activation_dot(ctrl, act, params)


def step_muscle_activation(
    act: Any,
    ctrl: Any,
    params: MuscleParams,
    dt: float,
    substeps: int = 1,
) -> tuple[Any, Any]:
    """Advance ``act`` over ``substeps`` explicit Euler steps.

    Args:
        act: Initial muscle activation state.
        ctrl: Muscle excitation input.
        params: Per-muscle parameters.
        dt: Timestep per substep [s].
        substeps: Number of Euler integration substeps.

    Returns:
        ``(next_act, mean_act)`` across ``substeps``.

    Raises:
        ImportError: If the installed ``myo-sim`` lacks ``myo_sim.physics.hill_muscle``.
    """
    return _library().step_muscle_activation(act, ctrl, params, dt, substeps=substeps)


def muscle_gain(length: Any, velocity: Any, params: MuscleParams) -> Any:
    """Compute active Hill gain ``-F0 * f_L(L) * f_V(V)`` matching ``mju_muscleGain``.

    Args:
        length: Actuator length [m].
        velocity: Actuator velocity [m/s].
        params: Per-muscle parameters.

    Returns:
        Active muscle gain (negative = pulling).

    Raises:
        ImportError: If the installed ``myo-sim`` lacks ``myo_sim.physics.hill_muscle``.
    """
    return _library().muscle_gain(length, velocity, params)


def muscle_bias(length: Any, params: MuscleParams) -> Any:
    """Compute passive Hill bias ``-F0 * f_P(L)`` matching ``mju_muscleBias``.

    Args:
        length: Actuator length [m].
        params: Per-muscle parameters.

    Returns:
        Passive muscle force (negative = pulling).

    Raises:
        ImportError: If the installed ``myo-sim`` lacks ``myo_sim.physics.hill_muscle``.
    """
    return _library().muscle_bias(length, params)


def muscle_actuator_force(
    length: Any, velocity: Any, act: Any, params: MuscleParams
) -> Any:
    """Compute total scalar actuator force ``gain(L, V) * act + bias(L)``.

    Args:
        length: Actuator length [m].
        velocity: Actuator velocity [m/s].
        act: Muscle activation state.
        params: Per-muscle parameters.

    Returns:
        Total scalar actuator force (negative = pulling).

    Raises:
        ImportError: If the installed ``myo-sim`` lacks ``myo_sim.physics.hill_muscle``.
    """
    return _library().muscle_actuator_force(length, velocity, act, params)


def dense_actuator_moment(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Unpack the dense ``(nu, nv)`` moment-arm Jacobian from ``data``.

    Args:
        model: Compiled model.
        data: Simulation data after forward kinematics/transmission.

    Returns:
        Dense ``(nu, nv)`` moment-arm matrix.

    Raises:
        ImportError: If the installed ``myo-sim`` lacks ``myo_sim.physics.hill_muscle``.
    """
    return _library().dense_actuator_moment(model, data)


class HillMuscleModel:
    """Evaluator for MuJoCo Hill muscle generalized forces via ``myo_sim``."""

    def __init__(
        self,
        model: mujoco.MjModel | None = None,
        *,
        params: MuscleParams | None = None,
        nv: int | None = None,
        dt: float = 0.002,
    ) -> None:
        """Initialize from either a compiled ``MjModel`` or ``(params, nv)``.

        Args:
            model: Compiled musculoskeletal model.
            params: Pre-extracted muscle parameters when ``model`` is ``None``.
            nv: Generalized velocity dimension when ``model`` is ``None``.
            dt: Simulation timestep [s] when ``model`` is ``None``.

        Raises:
            ValueError: If neither ``model`` nor ``(params, nv)`` is provided.
            ImportError: If the installed ``myo-sim`` lacks ``myo_sim.physics.hill_muscle``.
        """
        self._impl = _library().HillMuscleModel(model, params=params, nv=nv, dt=dt)
        impl_p = self._impl.params
        self.params: MuscleParams = (
            params
            if isinstance(params, MuscleParams)
            else MuscleParams(
                tau_act=impl_p.tau_act,
                tau_deact=impl_p.tau_deact,
                tausmooth=impl_p.tausmooth,
                act_limited=impl_p.act_limited,
                act_range=impl_p.act_range,
                length_range=impl_p.length_range,
                acc0=impl_p.acc0,
                gain_prm=impl_p.gain_prm,
                bias_prm=impl_p.bias_prm,
            )
        )
        self.nu: int = int(self._impl.nu)
        self.nv: int = int(self._impl.nv)
        self.dt: float = float(self._impl.dt)

    def generalized_forces(
        self, length: Any, velocity: Any, act: Any, moment_arm: Any
    ) -> Any:
        """Compute ``qfrc_actuator = moment_arm.T @ force`` for single or batched inputs.

        Args:
            length: Actuator lengths ``(..., nu)``.
            velocity: Actuator velocities ``(..., nu)``.
            act: Muscle activations ``(..., nu)``.
            moment_arm: Moment-arm Jacobian ``(..., nu, nv)``.

        Returns:
            Generalized joint torques ``(..., nv)``.
        """
        return self._impl.generalized_forces(length, velocity, act, moment_arm)


def strip_muscles_for_joint_torque_sim(
    spec: mujoco.MjSpec,
    *,
    add_joint_motors: bool = True,
) -> mujoco.MjSpec:
    """Return a tendon- and muscle-free copy of ``spec`` for joint-torque rigid sims.

    Args:
        spec: Source musculoskeletal ``MjSpec``.
        add_joint_motors: Add 1-to-1 ``mjTRN_JOINT`` unit torque motors on independent
            1-DOF joints.

    Returns:
        Independent ``MjSpec`` copy with tendons and muscles removed.

    Raises:
        ImportError: If the installed ``myo-sim`` lacks ``myo_sim.physics.hill_muscle``.
    """
    return _library().strip_muscles_for_joint_torque_sim(
        spec, add_joint_motors=add_joint_motors
    )


__all__ = [
    "HillMuscleModel",
    "MuscleParams",
    "dense_actuator_moment",
    "muscle_activation_dot",
    "muscle_actuator_force",
    "muscle_bias",
    "muscle_gain",
    "step_muscle_activation",
    "strip_muscles_for_joint_torque_sim",
]
