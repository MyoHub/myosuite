# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Static optimization: actuator activations that reproduce a motion's inverse dynamics.

At every frame the activations ``a`` minimise ``sum(a**2) + sum((r / F_opt)**2)``
subject to::

    AM^T (g * a + b) + r + J_eq^T lam = tau

``tau`` is the inverse dynamics (:mod:`myosuite.physics.inverse_dynamics`), ``AM``
the actuator moment arms, and ``g`` and ``b`` each actuator's force per unit
activation and its activation-independent (passive) force, taken from MuJoCo's own
actuator model at the frame's state: for muscles, force-length-velocity with a
rigid tendon. ``r`` are reserve actuators on every dof, penalised by their optimal
force as in OpenSim's static optimization; on a free joint they are the residual
actuators that carry the wrench ground reactions would supply (not penalised: no
actuator acts on a free root, so they equal its rows of ``tau``). ``lam`` are the
forces of the model's equality constraints (joint couplings), free multipliers.
Joint limits and contacts are not modelled: their loads appear in the reserves.

:class:`StaticOptimizer` solves one frame at a time and keeps its QP (OSQP)
warm, so the same code serves offline clips and a live stream::

    so = StaticOptimizer(model)
    frame = so.solve(qpos, qvel, qacc)  # one state
    result = static_optimization(model, kin.qpos, kin.qvel, kin.qacc)  # a clip

Activations are a minimum-effort estimate: co-contraction and activation dynamics
are not predicted.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields

import mujoco
import numpy as np
import osqp
from scipy import sparse

from myosuite.physics.inverse_dynamics import required_qfrc

# Constraint forces are optimised in units of this force [N m or N].
_CONSTRAINT_SCALE = 100.0
# Cost of a unit constraint force: regularises without biasing the activations.
_CONSTRAINT_WEIGHT = 1e-8


@dataclass(frozen=True)
class StaticOptimizationConfig:
    """Weights, bounds and solver tolerances.

    Attributes:
        reserve_optimal_force: Force of a reserve actuator at unit activation
            [N m or N]. Small values make reserves a last resort.
        min_activation: Lower bound of muscle activations.
        tolerance: OSQP's absolute and relative tolerance.
        max_iter: OSQP's iteration limit per frame.
    """

    reserve_optimal_force: float = 1.0
    min_activation: float = 0.0
    tolerance: float = 1e-7
    max_iter: int = 20000


@dataclass(frozen=True)
class StaticOptimizationResult:
    """Solution of one frame (no leading axis) or of a clip (leading axis ``T``).

    Attributes:
        activation: ``(..., nu)`` activations; controls for actuators without
            activation dynamics.
        actuator_force: ``(..., nu)`` actuator forces in MuJoCo's convention
            (muscles pull with negative values) [N].
        reserve: ``(..., nv)`` generalized force of reserve and residual actuators.
        constraint_force: ``(..., nv)`` generalized force of equality constraints.
        joint_torque: ``(..., nv)`` inverse dynamics ``tau``, the sum of the actuator,
            reserve and constraint generalized forces.
        status: ``(...)`` OSQP status, ``"solved"`` when converged.
    """

    activation: np.ndarray
    actuator_force: np.ndarray
    reserve: np.ndarray
    constraint_force: np.ndarray
    joint_torque: np.ndarray
    status: np.ndarray


class StaticOptimizer:
    """Static optimization of one model, frame by frame."""

    def __init__(
        self, model: mujoco.MjModel, config: StaticOptimizationConfig | None = None
    ) -> None:
        """Prepare a working copy of *model* and the QP bounds.

        Args:
            model: Compiled model; it is not modified.
            config: Weights and tolerances (defaults if ``None``).

        Raises:
            ValueError: If an actuator's force is clamped (``forcelimited``,
                joint ``actuatorfrcrange``) or has more than one activation.
        """
        cfg = config or StaticOptimizationConfig()
        m = copy.deepcopy(model)
        unsupported = [
            m.actuator(i).name
            for i in range(m.nu)
            if m.actuator_forcelimited[i] or m.actuator_actnum[i] > 1
        ] + [m.joint(j).name for j in np.flatnonzero(m.jnt_actfrclimited)]
        if unsupported:
            raise ValueError(
                f"Force clamps and multi-state actuators are not supported: {unsupported}"
            )
        # Dense Jacobians, the activation (not the next one) drives the force, no ctrl
        # clamping, equality rows built, contacts skipped.
        m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        m.actuator_actearly[:] = 0
        flags = mujoco.mjtDisableBit
        m.opt.disableflags = (
            (m.opt.disableflags & ~int(flags.mjDSBL_CONSTRAINT))
            | int(flags.mjDSBL_CONTACT)
            | int(flags.mjDSBL_CLAMPCTRL)
        )
        self.model, self.data, self.config = m, mujoco.MjData(m), cfg

        self._stateful = m.actuator_actadr >= 0
        lo, hi = np.full(m.nu, -np.inf), np.full(m.nu, np.inf)
        act_lim = self._stateful & (m.actuator_actlimited != 0)
        ctrl_lim = ~self._stateful & (m.actuator_ctrllimited != 0)
        lo[act_lim], hi[act_lim] = m.actuator_actrange[act_lim].T
        lo[ctrl_lim], hi[ctrl_lim] = m.actuator_ctrlrange[ctrl_lim].T
        muscle = m.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        lo[muscle], hi[muscle] = cfg.min_activation, 1.0
        self._lo, self._hi = lo, hi
        bounded = np.isfinite(lo) & np.isfinite(hi)
        self._probe = (np.where(bounded, lo, 0.0), np.where(bounded, hi, 1.0))

        self._optimal_force = np.full(m.nv, cfg.reserve_optimal_force)
        # Free-joint residuals are fixed by their rows (no actuator acts on a free
        # root); leaving them out of the cost keeps the large ground reaction from
        # swamping the solver's relative tolerance.
        self._reserve_weight = np.ones(m.nv)
        for j in np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
            self._reserve_weight[m.jnt_dofadr[j] : m.jnt_dofadr[j] + 6] = 0.0
        self._solver: osqp.OSQP | None = None
        self._pattern: np.ndarray | None = None

    def _actuator_force(self, value: np.ndarray) -> np.ndarray:
        """Actuator forces at the current state with activations/controls *value*."""
        m, d = self.model, self.data
        d.act[m.actuator_actadr[self._stateful]] = value[self._stateful]
        d.ctrl[~self._stateful] = value[~self._stateful]
        mujoco.mj_fwdActuation(m, d)
        return d.actuator_force.copy()

    def _solve_qp(
        self, A: np.ndarray, lower: np.ndarray, upper: np.ndarray, n_eq: int
    ) -> tuple[np.ndarray, str]:
        """Minimise the cost subject to ``lower <= A x <= upper``, warm-started.

        OSQP's matrix keeps the union of all sparsity patterns seen so far, so most
        frames only update values; a new nonzero rebuilds the problem.
        """
        pattern = A != 0
        if self._pattern is not None and self._pattern.shape == pattern.shape:
            if not (pattern & ~self._pattern).any():
                self._solver.update(Ax=A.T[self._pattern.T], l=lower, u=upper)
                res = self._solver.solve(raise_error=False)
                return res.x, res.info.status
            pattern |= self._pattern
        self._pattern = pattern
        cols, rows = np.nonzero(pattern.T)  # CSC order
        indptr = np.r_[0, np.cumsum(np.bincount(cols, minlength=A.shape[1]))]
        A_csc = sparse.csc_matrix((A[rows, cols], rows, indptr), shape=A.shape)
        weight = np.r_[
            np.ones(self.model.nu),
            self._reserve_weight,
            np.full(n_eq, _CONSTRAINT_WEIGHT),
        ]
        self._solver = osqp.OSQP()
        self._solver.setup(
            sparse.diags(2.0 * weight, format="csc"),
            np.zeros(A.shape[1]),
            A_csc,
            lower,
            upper,
            verbose=False,
            eps_abs=self.config.tolerance,
            eps_rel=self.config.tolerance,
            max_iter=self.config.max_iter,
            polishing=True,
            warm_starting=True,
        )
        res = self._solver.solve(raise_error=False)
        return res.x, res.info.status

    def solve(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        qacc: np.ndarray,
        external_qfrc: np.ndarray | None = None,
    ) -> StaticOptimizationResult:
        """Activations for one state.

        Args:
            qpos: ``(nq,)`` positions.
            qvel: ``(nv,)`` velocities.
            qacc: ``(nv,)`` accelerations.
            external_qfrc: Optional ``(nv,)`` known external generalized forces.

        Returns:
            The solution of this frame.
        """
        m, d = self.model, self.data
        d.qpos[:], d.qvel[:] = qpos, qvel
        mujoco.mj_fwdPosition(m, d)
        mujoco.mj_fwdVelocity(m, d)
        tau = required_qfrc(m, d, qacc)
        if external_qfrc is not None:
            tau -= external_qfrc

        f0, f1 = (self._actuator_force(p) for p in self._probe)
        gain = (f1 - f0) / (self._probe[1] - self._probe[0])
        bias = f0 - gain * self._probe[0]
        moment = np.zeros((m.nu, m.nv))
        mujoco.mju_sparse2dense(
            moment, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind
        )
        efc_J = d.efc_J[: d.nefc * m.nv].reshape(d.nefc, m.nv)
        J_eq = efc_J[d.efc_type[: d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY]
        n_eq = len(J_eq)

        A = np.zeros((m.nv + m.nu, m.nu + m.nv + n_eq))
        A[: m.nv, : m.nu] = moment.T * gain
        A[: m.nv, m.nu : m.nu + m.nv] = np.diag(self._optimal_force)
        A[: m.nv, m.nu + m.nv :] = J_eq.T * _CONSTRAINT_SCALE
        A[m.nv :, : m.nu] = np.eye(m.nu)
        rhs = tau - moment.T @ bias
        x, status = self._solve_qp(A, np.r_[rhs, self._lo], np.r_[rhs, self._hi], n_eq)

        a = np.clip(x[: m.nu], self._lo, self._hi)
        return StaticOptimizationResult(
            activation=a,
            actuator_force=gain * a + bias,
            reserve=self._optimal_force * x[m.nu : m.nu + m.nv],
            constraint_force=J_eq.T @ (_CONSTRAINT_SCALE * x[m.nu + m.nv :]),
            joint_torque=tau,
            status=np.asarray(status),
        )


def static_optimization(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    qacc: np.ndarray,
    *,
    external_qfrc: np.ndarray | None = None,
    config: StaticOptimizationConfig | None = None,
) -> StaticOptimizationResult:
    """Static optimization of every frame of a motion.

    Args:
        model: Compiled model.
        qpos: ``(T, nq)`` positions.
        qvel: ``(T, nv)`` velocities.
        qacc: ``(T, nv)`` accelerations, e.g. from
            :func:`~myosuite.physics.inverse_dynamics.clip_kinematics`.
        external_qfrc: Optional ``(T, nv)`` known external generalized forces.
        config: Weights and tolerances.

    Returns:
        The per-frame solutions stacked along a leading axis.
    """
    so = StaticOptimizer(model, config)
    frames = [
        so.solve(
            qpos[t],
            qvel[t],
            qacc[t],
            None if external_qfrc is None else external_qfrc[t],
        )
        for t in range(len(qpos))
    ]
    return StaticOptimizationResult(
        **{
            f.name: np.stack([getattr(fr, f.name) for fr in frames])
            for f in fields(StaticOptimizationResult)
        }
    )
