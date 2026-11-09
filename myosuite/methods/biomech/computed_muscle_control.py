# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Computed muscle control (CMC): excitations that make a forward simulation track a motion.

Every control step from ``t`` to ``t + T`` (Thelen, Anderson & Delp 2003):

1. ``qacc* = qacc_ref + kv (qvel_ref - qvel) + kp (qpos_ref - qpos)``: the errors at
   ``t`` and, as feedforward, the reference's mean acceleration over the step.
2. :class:`~myosuite.methods.biomech.static_optimization.StaticOptimizer` solves the state
   for ``qacc*``. The simulation's ``qfrc_applied`` / ``xfrc_applied`` are known
   loads; each actuator is bounded by the activations its dynamics can reach
   within the step (as OpenSim bounds excitations); velocity-dependent forces
   (force-velocity, joint damping) are taken at the velocity that delivers
   ``qacc*`` over MuJoCo's Euler steps, which matters for light, stiffly damped
   segments such as the foot and the wrist.
3. Excitations from inverting MuJoCo's own activation dynamics (any ``dyntype``,
   ``tausmooth`` included): the mean activation over the step's physics steps is
   the optimised one.
4. The simulation advances with them and, as in OpenSim, with the reserves and
   residuals applied. It has no contacts: the optimization's estimated contact
   forces (``contact_forces=True``) are applied instead, as OpenSim applies
   measured ground reactions. MuJoCo enforces the couplings and joint limits.

::

    kin = clip_kinematics(model, qpos, dt, lowpass_hz=6.0)
    config = ComputedMuscleControlConfig(
        optimization=StaticOptimizationConfig(contact_forces=True)  # standing, walking
    )
    res = computed_muscle_control(model, kin.qpos, kin.qvel, kin.qacc, kin.dt, config=config)
    res.excitation, res.activation, res.tracking_error, res.optimization.reserve
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import mujoco
import numpy as np

from myosuite.methods.biomech.static_optimization import (
    StaticOptimizationConfig,
    StaticOptimizationResult,
    StaticOptimizer,
)


@dataclass(frozen=True)
class ComputedMuscleControlConfig:
    """Feedback gains and the static optimization of every step.

    Attributes:
        kp: Position gain [1/s^2]. OpenSim uses 100; on MyoSuite's elbow and arm 400
            tracked 2-2.5 times closer at 50-250 Hz control. Larger gains track smooth
            references closer still and pass a noisy reference's noise to the
            excitations.
        kv: Velocity gain [1/s]; ``None`` is ``2 sqrt(kp)``, a critically damped error.
        apply_reserves: Apply the reserve and residual actuators in the simulation, as
            OpenSim does. Without them the simulation falls behind wherever the
            muscles cannot supply the load.
        optimization: Static-optimization settings.
    """

    kp: float = 400.0
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
        tracking_error: ``(..., nv)`` reference minus simulated position at the start
            of the step, the error the feedback corrects.
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
            ValueError: For an actuator with activation state and no ``ctrlrange``
                (other than a muscle), or what the static optimizer rejects.
        """
        self.model, self.config = model, config or ComputedMuscleControlConfig()
        # No contacts in the simulation: the estimated ones are applied (as OpenSim
        # applies measured ground reactions to a model without contacts).
        self.sim_model = copy.deepcopy(model)
        self.sim_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.optimizer = StaticOptimizer(model, self.config.optimization)
        self._stateful = idx = np.flatnonzero(model.actuator_actadr >= 0)
        self._actadr = model.actuator_actadr[idx]
        limited = model.actuator_ctrllimited != 0
        self._ctrl_lo = np.where(limited, model.actuator_ctrlrange[:, 0], -np.inf)
        self._ctrl_hi = np.where(limited, model.actuator_ctrlrange[:, 1], np.inf)
        muscle = model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self._ctrl_lo[muscle] = np.maximum(self._ctrl_lo[muscle], 0.0)
        self._ctrl_hi[muscle] = np.minimum(self._ctrl_hi[muscle], 1.0)
        unbounded = ~(np.isfinite(self._ctrl_lo[idx]) & np.isfinite(self._ctrl_hi[idx]))
        if unbounded.any():
            names = [model.actuator(int(i)).name for i in idx[unbounded]]
            raise ValueError(
                f"Actuators with activation state need a ctrlrange: {names}"
            )
        # Activation update per unit act_dot, and its clamp, as MuJoCo integrates it.
        h, tau = model.opt.timestep, np.maximum(model.actuator_dynprm[idx, 0], 1e-15)
        exact = model.actuator_dyntype[idx] == mujoco.mjtDyn.mjDYN_FILTEREXACT
        self._act_step = np.where(exact, tau * -np.expm1(-h / tau), h)
        act_limited = model.actuator_actlimited[idx] != 0
        self._act_lo = np.where(act_limited, model.actuator_actrange[idx, 0], -np.inf)
        self._act_hi = np.where(act_limited, model.actuator_actrange[idx, 1], np.inf)
        self._probe = mujoco.MjData(self.sim_model)
        mujoco.mj_forward(self.sim_model, self._probe)
        euler = model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER and not (
            model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        )
        # Joint damping Euler integrates implicitly, and what it leaves explicit.
        self._damping_implicit = model.dof_damping * euler
        self._damping_explicit = model.dof_damping * (not euler)
        self._moment = np.zeros((model.nu, model.nv))
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
        mujoco.mj_forward(self.sim_model, data)
        res = self.optimizer.solve(qpos, qvel, qacc, self._known_loads(data))
        data.ctrl[:] = res.activation
        data.act[self._actadr] = res.activation[self._stateful]
        mujoco.mj_forward(self.sim_model, data)
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
            qpos_ref: ``(nq,)`` reference positions at ``data.time``, on the joint
                couplings.
            qvel_ref: ``(nv,)`` reference velocities at ``data.time``.
            qacc_ref: ``(nv,)`` mean reference acceleration over the step, e.g. the
                mean of the accelerations at its start and end.
            n_substeps: Physics steps per control step.
            max_activation: Optional ``(nu,)`` activation caps for this step (e.g.
                ``1 - MF`` of fatigued muscles).

        Returns:
            The step's excitations, simulated state and static optimization.
        """
        m, idx = self.sim_model, self._stateful
        qpos_ref = np.asarray(qpos_ref, dtype=np.float64)
        error = np.empty(m.nv)
        mujoco.mj_differentiatePos(m, error, 1.0, data.qpos, qpos_ref)
        qacc = qacc_ref + self.kv * (qvel_ref - data.qvel) + self.kp * error

        applied = data.qfrc_applied.copy()
        external = self._known_loads(data)

        # Mean activations reachable over the step with the extreme controls.
        current = data.act[self._actadr]
        lo, hi = self._ctrl_lo[idx], self._ctrl_hi[idx]
        reach = [self._mean_activation(current, u, n_substeps) for u in (lo, hi)]
        low, high = np.full(m.nu, -np.inf), np.full(m.nu, np.inf)
        low[idx], high[idx] = reach
        if max_activation is not None:
            high = np.minimum(high, max_activation)
        # Velocity-dependent forces at the velocity that delivers qacc* over the step.
        qvel = data.qvel + self._velocity_fraction(data, n_substeps) * qacc * (
            n_substeps * m.opt.timestep
        )
        res = self.optimizer.solve(data.qpos, qvel, qacc, external, high, low)

        excitation = res.activation.copy()
        excitation[idx] = self._invert(current, res.activation[idx], n_substeps, reach)
        excitation = np.clip(excitation, self._ctrl_lo, self._ctrl_hi)
        data.ctrl[:] = excitation
        if self.config.apply_reserves:
            data.qfrc_applied += res.reserve
        data.qfrc_applied += res.contact_qfrc
        for _ in range(n_substeps):
            mujoco.mj_step(m, data)
        data.qfrc_applied[:] = applied
        mujoco.mj_forward(m, data)

        activation = excitation.copy()
        activation[idx] = data.act[self._actadr]
        return ComputedMuscleControlResult(
            excitation=excitation,
            activation=activation,
            qpos=data.qpos.copy(),
            qvel=data.qvel.copy(),
            tracking_error=error,
            qacc_desired=qacc,
            optimization=res,
        )

    def _velocity_fraction(self, data: mujoco.MjData, n_substeps: int) -> np.ndarray:
        """Per dof, the fraction ``c`` of the step's velocity change ``T qacc*`` at which
        the optimization evaluates velocity-dependent forces.

        For one dof with inertia ``I``, actuator damping ``B`` (force-velocity, which
        MuJoCo's Euler step treats explicitly) and joint damping ``D`` (implicit), ``n``
        physics steps of length ``h`` give ``v_n`` exactly when the forces are taken at
        ``v_0 + c T qacc*`` with ``c = 1 / (1 - rho^n) - 1 / x``, ``rho = (1 - h B / I)
        / (1 + h D / I)`` and ``x = n h (B + D) / I``. ``c`` is 0 for explicit damping
        at one step, 1 for implicit damping or a stiff dof, and 1/2 for a smooth one.
        ``B`` is the diagonal of ``R^T (dF/dv) R`` at the current activations, ``I``
        that of the inverse mass matrix (dofs taken one at a time).
        """
        m, d, h = self.sim_model, self._probe, self.sim_model.opt.timestep
        d.qpos[:], d.qvel[:], d.act[:] = data.qpos, data.qvel, data.act
        d.ctrl[:] = data.ctrl
        mujoco.mj_fwdPosition(m, d)
        mujoco.mj_fwdVelocity(m, d)
        velocity, force = d.actuator_velocity.copy(), []
        for dv in (1e-6, -1e-6):
            d.actuator_velocity[:] = velocity + dv
            mujoco.mj_fwdActuation(m, d)
            force.append(d.actuator_force.copy())
        mujoco.mju_sparse2dense(
            self._moment,
            d.actuator_moment,
            d.moment_rownnz,
            d.moment_rowadr,
            d.moment_colind,
        )
        damping = -(self._moment**2).T @ ((force[0] - force[1]) / 2e-6)
        explicit = np.maximum(damping, 0.0) + self._damping_explicit
        inv_mass = np.empty((m.nv, m.nv))
        mujoco.mj_solveM(m, d, inv_mass, np.eye(m.nv))
        b = h * explicit * np.diag(inv_mass)
        g = h * self._damping_implicit * np.diag(inv_mass)
        n, x = n_substeps, n_substeps * (b + g)
        with np.errstate(divide="ignore", invalid="ignore"):
            exact = 1.0 / (1.0 - ((1.0 - b) / (1.0 + g)) ** n) - 1.0 / x
            smooth = (n - 1) / (2 * n) + np.where(b + g > 0, g / (b + g), 0.0) / n
        return np.clip(np.where(x > 1e-4, exact, smooth), 0.0, 1.0)

    def _mean_activation(
        self, start: np.ndarray, control: np.ndarray, n_substeps: int
    ) -> np.ndarray:
        """Mean activation after each of *n_substeps* physics steps at constant *control*.

        Integrates MuJoCo's own ``act_dot`` (``mj_fwdActuation``) for every actuator
        with activation state at once, as the simulation does: Euler steps (exact
        for ``filterexact``), clamped to ``actrange``.
        """
        m, d, adr = self.sim_model, self._probe, self._actadr
        d.ctrl[self._stateful] = control
        act, total = start.copy(), np.zeros_like(start)
        for _ in range(n_substeps):
            d.act[adr] = act
            mujoco.mj_fwdActuation(m, d)
            act = np.clip(
                act + self._act_step * d.act_dot[adr], self._act_lo, self._act_hi
            )
            total += act
        return total / n_substeps

    def _invert(
        self,
        start: np.ndarray,
        target: np.ndarray,
        n_substeps: int,
        reach: list[np.ndarray],
    ) -> np.ndarray:
        """Controls whose mean activation over the step is *target*.

        The mean activation is continuous and non-decreasing in the control (flat
        where ``actrange`` clamps it), so a safeguarded secant iteration finds it
        from the control bounds, whose mean activations are *reach*: secant steps,
        bisection where they leave the bracket. Targets out of reach get the bound.
        """
        a, b = self._ctrl_lo[self._stateful], self._ctrl_hi[self._stateful]
        fa, fb = reach[0] - target, reach[1] - target
        u = np.where(fa >= 0, a, b)
        active = (fa < 0) & (fb > 0)
        x0, f0, x1, f1 = a, fa, b, fb
        for _ in range(60):
            if not active.any():
                break
            # Secant on the last two iterates, bisection when it leaves the bracket.
            with np.errstate(divide="ignore", invalid="ignore"):
                c = x1 - f1 * (x1 - x0) / (f1 - f0)
            c = np.where(np.isfinite(c) & (c > a) & (c < b), c, 0.5 * (a + b))
            c = np.where(active, c, u)
            fc = self._mean_activation(start, c, n_substeps) - target
            right = active & (fc > 0)
            left = active & (fc <= 0)
            a, fa = np.where(left, c, a), np.where(left, fc, fa)
            b, fb = np.where(right, c, b), np.where(right, fc, fb)
            x0, f0, x1, f1 = x1, f1, c, fc
            u = np.where(active, c, u)
            active &= (np.abs(fc) > 1e-13) & (b - a > 1e-15)
        return u


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
            :func:`~myosuite.methods.biomech.inverse_dynamics.clip_kinematics`.
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
    for t in range(len(qpos) - 1):
        data.qfrc_applied[:] = loads[t]
        mean_qacc = 0.5 * (qacc[t] + qacc[t + 1])
        steps.append(cmc.step(data, qpos[t], qvel[t], mean_qacc, n_substeps=n_substeps))
    return _stack(steps)
