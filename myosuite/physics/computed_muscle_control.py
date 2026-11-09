# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Computed muscle control (CMC): excitations that make a forward simulation track a motion.

Every control step from ``t`` to ``t + T`` (Thelen, Anderson & Delp 2003):

1. ``qacc* = qacc_ref + kv (qvel_ref - qvel) + kp (qpos_ref - qpos)`` with the
   reference at ``t + T`` and the simulated state at ``t``: muscle forces reach
   their targets over the step, so the targets are the accelerations wanted at its end.
2. :class:`~myosuite.physics.static_optimization.StaticOptimizer` solves the state
   for ``qacc*`` with the simulation's ``qfrc_applied`` / ``xfrc_applied`` as known
   loads and each muscle bounded by what its activation dynamics can reach within
   the step (as OpenSim bounds excitations); unbounded, fast changes go to muscles
   that cannot follow. Under MuJoCo's Euler integrator, which integrates joint
   damping ``D`` implicitly, the muscles also supply ``h D qacc*``.
3. Inverting MuJoCo's activation dynamics gives excitations whose mean activation
   over the step's physics steps is the optimised one (the optimised impulse).
4. The simulation advances with them and, as in OpenSim, with the reserves applied.

MuJoCo enforces couplings, limits and contacts in the simulation; feedback absorbs
what the optimization does not model. Estimated contact forces
(``contact_forces=True``) are not applied, the simulation's own contacts act::

    kin = clip_kinematics(model, qpos, dt, lowpass_hz=6.0)
    res = computed_muscle_control(model, kin.qpos, kin.qvel, kin.qacc, kin.dt)
    res.excitation, res.activation, res.tracking_error, res.optimization.reserve
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import mujoco
import numpy as np

from myosuite.physics.static_optimization import (
    StaticOptimizationConfig,
    StaticOptimizationResult,
    StaticOptimizer,
)


@dataclass(frozen=True)
class ComputedMuscleControlConfig:
    """Feedback gains and the static optimization of every step.

    Attributes:
        kp: Position gain [1/s^2], OpenSim's default. Light distal segments (the
            wrist) track closer with larger gains.
        kv: Velocity gain [1/s]; ``None`` is ``2 sqrt(kp)``, a critically damped error.
        apply_reserves: Apply the reserve and residual actuators in the simulation, as
            OpenSim does. Without them the simulation falls behind wherever the
            muscles cannot supply the load.
        optimization: Static-optimization settings.
    """

    kp: float = 100.0
    kv: float | None = None
    apply_reserves: bool = True
    optimization: StaticOptimizationConfig = field(
        default_factory=StaticOptimizationConfig
    )


@dataclass(frozen=True)
class ComputedMuscleControlResult:
    """One control step (no leading axis) or every step of a clip (leading axis).

    Attributes:
        excitation: ``(..., nu)`` controls written to ``data.ctrl``: muscle
            excitations, the optimised controls of actuators without activation state.
        activation: ``(..., nu)`` simulated activations at the end of the step
            (controls for actuators without activation state).
        qpos: ``(..., nq)`` simulated positions at the end of the step.
        qvel: ``(..., nv)`` simulated velocities at the end of the step.
        tracking_error: ``(..., nv)`` reference minus simulated position at the end
            of the step.
        qacc_desired: ``(..., nv)`` desired accelerations ``qacc*``.
        optimization: Static optimization of the step: the mean activations it aims
            at, reserves, constraint and contact forces.
    """

    excitation: np.ndarray
    activation: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    tracking_error: np.ndarray
    qacc_desired: np.ndarray
    optimization: StaticOptimizationResult


class ComputedMuscleControl:
    """Computed muscle control of one model, step by step."""

    def __init__(
        self, model: mujoco.MjModel, config: ComputedMuscleControlConfig | None = None
    ) -> None:
        """Prepare the static optimization of *model*.

        Args:
            model: Compiled model of the forward simulation; it is not modified.
            config: Gains and settings (defaults if ``None``).

        Raises:
            ValueError: For activation dynamics other than MuJoCo's muscle, or what
                the static optimizer rejects.
        """
        self.model, self.config = model, config or ComputedMuscleControlConfig()
        stateful = model.actuator_actadr >= 0
        other = stateful & (model.actuator_dyntype != mujoco.mjtDyn.mjDYN_MUSCLE)
        if other.any():
            names = [model.actuator(i).name for i in np.flatnonzero(other)]
            raise ValueError(f"Only muscle activation dynamics are inverted: {names}")
        self.optimizer = StaticOptimizer(model, self.config.optimization)
        self._muscles = np.flatnonzero(stateful)
        self._actadr = model.actuator_actadr[self._muscles]
        limited = model.actuator_ctrllimited != 0
        self._ctrl_lo = np.where(limited, model.actuator_ctrlrange[:, 0], -np.inf)
        self._ctrl_hi = np.where(limited, model.actuator_ctrlrange[:, 1], np.inf)
        self._ctrl_lo[self._muscles] = np.maximum(self._ctrl_lo[self._muscles], 0.0)
        self._ctrl_hi[self._muscles] = np.minimum(self._ctrl_hi[self._muscles], 1.0)
        euler = model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER and not (
            model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        )
        self._implicit_damping = model.opt.timestep * model.dof_damping * euler
        self.kp = self.config.kp
        self.kv = 2.0 * np.sqrt(self.kp) if self.config.kv is None else self.config.kv

    def reset(
        self,
        data: mujoco.MjData,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
    ) -> StaticOptimizationResult:
        """Start tracking at a reference state, the muscles in equilibrium with it.

        Puts *data* at *qpos* and *qvel* with the controls and activations of the
        state's static optimization for *qacc* (``qfrc_applied`` and
        ``xfrc_applied`` are known loads, as in :meth:`step`) and forwards it.

        Args:
            data: Forward simulation of the model; its applied loads are kept.
            qpos: ``(nq,)`` reference positions.
            qvel: ``(nv,)`` reference velocities.
            qacc: ``(nv,)`` reference accelerations.

        Returns:
            The static optimization of the state.
        """
        data.qpos[:], data.qvel[:] = qpos, qvel
        mujoco.mj_forward(self.model, data)
        res = self.optimizer.solve(qpos, qvel, qacc, self._known_loads(data))
        data.ctrl[:] = res.activation
        data.act[self._actadr] = res.activation[self._muscles]
        mujoco.mj_forward(self.model, data)
        return res

    def _known_loads(self, data: mujoco.MjData) -> np.ndarray:
        """``qfrc_applied`` plus ``xfrc_applied`` as generalized forces."""
        loads = data.qfrc_applied.copy()
        for body in np.flatnonzero(np.any(data.xfrc_applied != 0, axis=1)):
            wrench = data.xfrc_applied[body]
            mujoco.mj_applyFT(
                self.model,
                data,
                wrench[:3],
                wrench[3:],
                data.xipos[body],
                int(body),
                loads,
            )
        return loads

    def step(
        self,
        data: mujoco.MjData,
        qpos_ref: np.ndarray,
        qvel_ref: np.ndarray,
        qacc_ref: np.ndarray,
        *,
        n_substeps: int = 1,
        max_activation: np.ndarray | None = None,
    ) -> ComputedMuscleControlResult:
        """Advance *data* by one control step towards the reference.

        Args:
            data: Forward simulation of the model, forwarded (``mj_forward``) at its
                state as this method leaves it; advanced in place. Its
                ``qfrc_applied`` and ``xfrc_applied`` are known loads and are kept.
            qpos_ref: ``(nq,)`` reference positions at the end of the step,
                ``data.time + n_substeps * timestep``, on the joint couplings.
            qvel_ref: ``(nv,)`` reference velocities at the end of the step.
            qacc_ref: ``(nv,)`` reference accelerations at the end of the step.
            n_substeps: Physics steps per control step.
            max_activation: Optional ``(nu,)`` activation caps for this step (e.g.
                ``1 - MF`` of fatigued muscles).

        Returns:
            The step's excitations, simulated state and static optimization.
        """
        m, mus = self.model, self._muscles
        qpos_ref = np.asarray(qpos_ref, dtype=np.float64)
        error = np.empty(m.nv)
        mujoco.mj_differentiatePos(m, error, 1.0, data.qpos, qpos_ref)
        qacc = qacc_ref + self.kv * (qvel_ref - data.qvel) + self.kp * error

        applied = data.qfrc_applied.copy()
        external = self._known_loads(data) - self._implicit_damping * qacc

        # Mean activations reachable over the step with the extreme excitations.
        current = data.act[self._actadr]
        lo, hi = self._ctrl_lo[mus], self._ctrl_hi[mus]
        low = np.full(m.nu, -np.inf)
        low[mus] = lo + (current - lo) * self._decay(current, lo, n_substeps)
        high = np.full(m.nu, np.inf)
        if max_activation is not None:
            high[:] = max_activation
        reach = hi + (current - hi) * self._decay(current, hi, n_substeps)
        high[mus] = np.minimum(high[mus], reach)
        res = self.optimizer.solve(data.qpos, data.qvel, qacc, external, high, low)

        target = res.activation[mus]
        decay = self._decay(current, target, n_substeps)
        excitation = res.activation.copy()
        excitation[mus] = np.where(
            decay < 1.0,
            (target - current * decay) / np.maximum(1.0 - decay, 1e-300),
            target,
        )
        excitation = np.clip(excitation, self._ctrl_lo, self._ctrl_hi)
        data.ctrl[:] = excitation
        if self.config.apply_reserves:
            data.qfrc_applied[:] = applied + res.reserve
        for _ in range(n_substeps):
            mujoco.mj_step(m, data)
        data.qfrc_applied[:] = applied
        mujoco.mj_forward(m, data)

        mujoco.mj_differentiatePos(m, error, 1.0, data.qpos, qpos_ref)
        activation = excitation.copy()
        activation[mus] = data.act[self._actadr]
        return ComputedMuscleControlResult(
            excitation=excitation,
            activation=activation,
            qpos=data.qpos.copy(),
            qvel=data.qvel.copy(),
            tracking_error=error,
            qacc_desired=qacc,
            optimization=res,
        )

    def _decay(
        self, current: np.ndarray, control: np.ndarray, n_substeps: int
    ) -> np.ndarray:
        """``c`` per muscle such that the mean activation over the step is ``u + (a_0 - u) c``.

        MuJoCo integrates ``da/dt = (u - a) / tau`` with Euler steps of length ``h``,
        so ``a_k = u + (a_0 - u) d^k`` with ``d = 1 - h / tau`` and
        ``c = mean(d^1 .. d^n)``. ``tau`` is held at its value for the current
        activation (``mju_muscleDynamics``): exact for one physics step without
        ``tausmooth``, otherwise the feedback absorbs the difference.
        """
        m, h = self.model, self.model.opt.timestep
        decay = np.ones_like(current)
        powers = np.arange(1, n_substeps + 1)
        for k in np.flatnonzero(control != current):
            prm = m.actuator_dynprm[self._muscles[k], :3]
            rate = mujoco.mju_muscleDynamics(control[k], current[k], prm)
            if rate != 0.0:
                d = max(1.0 - h * rate / (control[k] - current[k]), 0.0)
                decay[k] = np.mean(d**powers)
        return decay


def _stack(steps: list[Any]) -> Any:
    """Stack per-step dataclasses (nested ones included) along a leading axis."""
    stacked = {}
    for f in fields(steps[0]):
        values = [getattr(s, f.name) for s in steps]
        stacked[f.name] = (
            _stack(values) if is_dataclass(values[0]) else np.stack(values)
        )
    return type(steps[0])(**stacked)


def computed_muscle_control(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    qacc: np.ndarray,
    dt: float,
    *,
    external_qfrc: np.ndarray | None = None,
    config: ComputedMuscleControlConfig | None = None,
) -> ComputedMuscleControlResult:
    """Track a motion clip with computed muscle control.

    The simulation starts at the first frame with the activations of its static
    optimization and takes one control step per frame interval.

    Args:
        model: Compiled model; ``dt`` must be a whole number of its physics steps.
        qpos: ``(T, nq)`` reference positions.
        qvel: ``(T, nv)`` reference velocities.
        qacc: ``(T, nv)`` reference accelerations, e.g. from
            :func:`~myosuite.physics.inverse_dynamics.clip_kinematics`.
        dt: Frame period [s].
        external_qfrc: Optional ``(T, nv)`` known external generalized forces,
            applied in the simulation (``qfrc_applied``) and known to the
            optimization.
        config: Gains and settings.

    Returns:
        The ``T - 1`` control steps stacked along a leading axis.

    Raises:
        ValueError: If ``dt`` is not a whole number of physics steps.
    """
    n_substeps = round(dt / model.opt.timestep)
    if n_substeps < 1 or abs(n_substeps * model.opt.timestep - dt) > 1e-9 * dt:
        raise ValueError(
            f"dt={dt} must be a whole number of physics steps "
            f"({model.opt.timestep}); resample the clip or change the timestep"
        )
    cmc = ComputedMuscleControl(model, config)
    data = mujoco.MjData(model)
    loads = np.zeros((len(qpos), model.nv)) if external_qfrc is None else external_qfrc
    data.qfrc_applied[:] = loads[0]
    cmc.reset(data, qpos[0], qvel[0], qacc[0])
    steps = []
    for t in range(1, len(qpos)):
        data.qfrc_applied[:] = loads[t]
        steps.append(cmc.step(data, qpos[t], qvel[t], qacc[t], n_substeps=n_substeps))
    return _stack(steps)
