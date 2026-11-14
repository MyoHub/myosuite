# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Static optimization: actuator activations that reproduce a motion's inverse dynamics.

At every frame the activations ``a`` minimise ``sum(a**2) + sum((r / F_opt)**2)``
subject to::

    AM^T (g * a + b) + r + J_eq^T lam = tau

``tau`` is the inverse dynamics (:mod:`myosuite.biomechanics.inverse_dynamics`), ``AM``
the actuator moment arms, and ``g`` and ``b`` each actuator's force per unit
activation and its activation-independent (passive) force, taken from MuJoCo's own
actuator model at the frame's state: for muscles, force-length-velocity with a
rigid tendon. ``r`` are reserve actuators on every dof, penalised by their optimal
force as in OpenSim's static optimization; on a free joint they are the residual
actuators that carry the wrench ground reactions would supply (not penalised: no
actuator acts on a free root, so they equal its rows of ``tau``). ``lam`` are the
forces of the model's equality constraints (joint couplings), free multipliers.
Joint limits are not modelled: their loads appear in the reserves.

A reserve takes a share of a dof's load that grows with its optimal force
relative to the muscles spanning the dof, so by default each dof's optimal force
is a fixed fraction (``reserve_fraction``) of that dof's peak actuator torque: 1-4 N m
at MyoLeg's hip, knee and ankle (OpenSim's usual reserves are 1 N m) and about
5 mN m at MyoHand's fingertips. A fixed value cannot serve both: 1 N m carries
most of a hand motion's load, and 1 mN m makes the QP too ill-conditioned for
OSQP. Reserves that still carry a large share mark frames the muscles cannot
produce: activations at their bounds, poses outside the joint ranges, a weak model.

With ``contact_forces=True`` the optimization also estimates ground reactions
for motions without force plates: ``+ J_c^T f`` with ``f >= 0`` on the edges of
MuJoCo's friction pyramid at every contact with a world-fixed body (floor, chair,
table) within ``contact_distance``, penalised by ``(f / F_contact)**2``. A
contact's capacity fades linearly to zero across that distance, so forces do not
jump as a foot lands or lifts off. Residuals are then penalised too, so contacts
carry what they can and residuals flag what they cannot (pulling, too little
friction, a missed contact).

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
from dataclasses import dataclass, fields, replace

import mujoco
import numpy as np
import osqp
from scipy import sparse

from myosuite.core.muscle_conditions import _peak_force
from myosuite.biomechanics.inverse_dynamics import required_qfrc

# Constraint forces are optimised in units of this force [N m or N].
_CONSTRAINT_SCALE = 100.0
# Cost of a unit constraint force: regularises without biasing the activations.
_CONSTRAINT_WEIGHT = 1e-8
_CONTACT_ROWS = (
    mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS,
    mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL,
)


@dataclass(frozen=True)
class StaticOptimizationConfig:
    """Weights, bounds and solver tolerances.

    Attributes:
        reserve_optimal_force: Force of the reserve actuators at unit activation
            [N m or N]: one value for every dof, as OpenSim's reserves, or one per
            dof. ``None`` scales it per dof (see ``reserve_fraction``). Smaller
            values make reserves a later resort.
        reserve_fraction: With ``reserve_optimal_force=None``, each dof's reserve
            optimal force as a fraction of its peak actuator torque
            ``sum_i |R_ij| F_i`` at ``qpos0`` (moment arms ``R``, muscle peak
            isometric forces or actuator forces at full control ``F``); dofs no
            actuator spans get 1 N m (or N).
        min_activation: Lower bound of muscle activations.
        tolerance: OSQP's absolute and relative tolerance.
        max_iter: OSQP's iteration limit per frame.
        contact_forces: Estimate contact forces with world-fixed bodies.
        contact_distance: Gap up to which a contact can push [m]. It must cover
            the error of the clip's foot heights: 1 cm suits clips solved on this
            model (IK, retargeting onto it); unscaled clips can leave a stance
            foot several centimetres above the floor, and then no contact pushes.
            Raise it with care: human toe clearance in swing is about 1-2 cm, so a
            reach of several centimetres lets a swinging foot push too. Scaling
            the model to the subject, or shifting the clip's root to put the
            stance feet on the floor, is the better fix.
        contact_optimal_force: ``F_contact`` per pyramid edge [N]; ``None`` is
            the body weight.
        residual_optimal_force: Optimal force of the free-joint residuals when
            contacts are estimated [N or N m]; ``None`` is a tenth of the body
            weight. Without contacts residuals are not penalised.
    """

    reserve_optimal_force: float | tuple[float, ...] | None = None
    reserve_fraction: float = 0.005
    min_activation: float = 0.0
    tolerance: float = 1e-7
    max_iter: int = 20000
    contact_forces: bool = False
    contact_distance: float = 0.01
    contact_optimal_force: float | None = None
    residual_optimal_force: float | None = None


@dataclass(frozen=True)
class StaticOptimizationResult:
    """Solution of one frame (no leading axis) or of a clip (leading axis ``T``).

    Attributes:
        activation: ``(..., nu)`` activations; controls for actuators without
            activation dynamics.
        actuator_force: ``(..., nu)`` actuator forces in MuJoCo's convention
            (muscles pull with negative values) [N].
        reserve: ``(..., nv)`` generalized force of reserve and residual actuators.
        constraint_force: ``(..., nv)`` generalized force of equality constraints
            and contacts.
        joint_torque: ``(..., nv)`` inverse dynamics ``tau``, the sum of the actuator,
            reserve and constraint generalized forces.
        status: ``(...)`` OSQP status, ``"solved"`` when converged.
        contact_force: ``(..., nbody, 3)`` world-frame contact force on each body
            (zero without ``contact_forces``); body 0 holds the reaction on the world.
        contact_qfrc: ``(..., nv)`` generalized force of those contact forces, the
            contact part of ``constraint_force``.
    """

    activation: np.ndarray
    actuator_force: np.ndarray
    reserve: np.ndarray
    constraint_force: np.ndarray
    joint_torque: np.ndarray
    status: np.ndarray
    contact_force: np.ndarray
    contact_qfrc: np.ndarray


class StaticOptimizer:
    """Static optimization of one model, frame by frame.

    Attributes:
        model: The working copy of the model.
        data: Its data, at the last solved state.
        config: The settings as solved: the model-derived defaults (per-dof
            reserves, contact and residual optimal forces) filled in, so passing
            it on reproduces this optimizer's weights on another model.
    """

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
        # clamping, equality rows built; contacts within reach on a friction pyramid.
        m.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        m.actuator_actearly[:] = 0
        flags = mujoco.mjtDisableBit
        m.opt.disableflags = (
            m.opt.disableflags
            & ~int(flags.mjDSBL_CONSTRAINT)
            & ~int(flags.mjDSBL_CONTACT)
        ) | int(flags.mjDSBL_CLAMPCTRL)
        if cfg.contact_forces:
            m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
            m.geom_margin[:] = np.maximum(m.geom_margin, cfg.contact_distance)
            m.geom_gap[:] = 0.0
            m.pair_margin[:] = np.maximum(m.pair_margin, cfg.contact_distance)
            m.pair_gap[:] = 0.0
        else:
            m.opt.disableflags |= int(flags.mjDSBL_CONTACT)
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

        body_weight = float(
            m.body_mass[m.body_weldid != 0].sum() * np.linalg.norm(m.opt.gravity)
        )
        default_force = body_weight if body_weight > 0.0 else 1.0
        self._contact_force = (
            cfg.contact_optimal_force
            if cfg.contact_optimal_force is not None
            else default_force
        )
        residual = (
            cfg.residual_optimal_force
            if cfg.residual_optimal_force is not None
            else 0.1 * default_force
        )
        if cfg.reserve_optimal_force is None:
            peak = self.peak_actuator_qfrc()
            spanned = peak > 1e-9 * peak.max(initial=0.0)
            self._optimal_force = np.where(spanned, cfg.reserve_fraction * peak, 1.0)
        else:
            self._optimal_force = np.broadcast_to(
                np.asarray(cfg.reserve_optimal_force, dtype=np.float64), (m.nv,)
            ).copy()
        # Without contacts, free-joint residuals are fixed by their rows (no actuator
        # acts on a free root); leaving them out of the cost keeps the large ground
        # reaction from swamping the solver's relative tolerance.
        self._reserve_weight = np.ones(m.nv)
        for j in np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
            root = slice(m.jnt_dofadr[j], m.jnt_dofadr[j] + 6)
            if cfg.contact_forces:
                self._optimal_force[root] = residual
            else:
                self._reserve_weight[root] = 0.0
        # The settings as solved, with the model-derived defaults filled in.
        self.config = replace(
            cfg,
            reserve_optimal_force=tuple(self._optimal_force.tolist()),
            contact_optimal_force=self._contact_force,
            residual_optimal_force=residual,
        )
        self._solver: osqp.OSQP | None = None
        self._pattern: np.ndarray | None = None

    def peak_actuator_qfrc(self) -> np.ndarray:
        """Per dof, ``sum_i |R_ij| F_i`` at ``qpos0`` and rest [N m or N].

        ``R`` are the moment arms; ``F`` each muscle's peak isometric force and,
        for other actuators, the force at the larger-magnitude end of the
        activation or control range. This scales the default reserves.

        Returns:
            ``(nv,)`` peak generalized force the actuators can apply at each dof
            if all pulled the same way.
        """
        m, d = self.model, self.data
        d.qpos[:], d.qvel[:] = m.qpos0, 0.0
        mujoco.mj_fwdPosition(m, d)
        mujoco.mj_fwdVelocity(m, d)
        lo, hi = self._probe
        gain = (self._actuator_force(hi) - self._actuator_force(lo)) / (hi - lo)
        force = np.abs(gain) * np.maximum(np.abs(lo), np.abs(hi))
        muscle = m.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE
        force[muscle] = np.abs(_peak_force(m)[muscle])
        moment = np.zeros((m.nu, m.nv))
        mujoco.mju_sparse2dense(
            moment, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind
        )
        return np.abs(moment).T @ force

    def _actuator_force(self, value: np.ndarray) -> np.ndarray:
        """Actuator forces at the current state with activations/controls *value*."""
        m, d = self.model, self.data
        d.act[m.actuator_actadr[self._stateful]] = value[self._stateful]
        d.ctrl[~self._stateful] = value[~self._stateful]
        mujoco.mj_fwdActuation(m, d)
        return d.actuator_force.copy()

    def _solve_qp(
        self, A: np.ndarray, lower: np.ndarray, upper: np.ndarray, weight: np.ndarray
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
        max_activation: np.ndarray | None = None,
        min_activation: np.ndarray | None = None,
    ) -> StaticOptimizationResult:
        """Activations for one state.

        Args:
            qpos: ``(nq,)`` positions.
            qvel: ``(nv,)`` velocities.
            qacc: ``(nv,)`` accelerations.
            external_qfrc: Optional ``(nv,)`` known external generalized forces.
            max_activation: Optional ``(nu,)`` upper activation bounds for this
                frame (e.g. ``1 - MF`` of fatigued muscles), within the
                actuators' own ranges.
            min_activation: Optional ``(nu,)`` lower activation bounds for this
                frame (e.g. the activations reachable under activation
                dynamics), within the actuators' ranges and the upper bounds.

        Returns:
            The solution of this frame.
        """
        lo, hi = self._lo, self._hi
        if max_activation is not None:
            hi = np.maximum(np.minimum(hi, max_activation), lo)
        if min_activation is not None:
            lo = np.minimum(np.maximum(lo, min_activation), hi)
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
        rows, capacity = self._contact_rows()
        nu, nv, n_eq, n_c = m.nu, m.nv, len(J_eq), len(rows)

        # Variables: activations, reserves, equality forces, contact edge forces
        # in units of each edge's capacity.
        A = np.zeros((nv + nu + n_c, nu + nv + n_eq + n_c))
        A[:nv, :nu] = moment.T * gain
        A[:nv, nu : nu + nv] = np.diag(self._optimal_force)
        A[:nv, nu + nv : nu + nv + n_eq] = J_eq.T * _CONSTRAINT_SCALE
        A[:nv, nu + nv + n_eq :] = efc_J[rows].T * capacity
        A[nv : nv + nu, :nu] = np.eye(nu)
        A[nv + nu :, nu + nv + n_eq :] = np.eye(n_c)
        rhs = tau - moment.T @ bias
        lower = np.r_[rhs, lo, np.zeros(n_c)]
        upper = np.r_[rhs, hi, np.full(n_c, np.inf)]
        weight = np.r_[
            np.ones(nu),
            self._reserve_weight,
            np.full(n_eq, _CONSTRAINT_WEIGHT),
            np.ones(n_c),
        ]
        x, status = self._solve_qp(A, lower, upper, weight)
        if x is None or x.dtype == object or not np.all(np.isfinite(x)):
            zero_a = np.clip(np.zeros(nu), lo, hi)
            return StaticOptimizationResult(
                activation=zero_a,
                actuator_force=gain * zero_a + bias,
                reserve=np.zeros(nv),
                constraint_force=np.zeros(nv),
                joint_torque=tau,
                status=np.asarray(status),
                contact_force=np.zeros((m.nbody, 3)),
                contact_qfrc=np.zeros(nv),
            )

        a = np.clip(x[:nu], lo, hi)
        f = capacity * np.maximum(x[nu + nv + n_eq :], 0.0)
        contact_qfrc = efc_J[rows].T @ f
        return StaticOptimizationResult(
            activation=a,
            actuator_force=gain * a + bias,
            reserve=self._optimal_force * x[nu : nu + nv],
            constraint_force=J_eq.T @ (_CONSTRAINT_SCALE * x[nu + nv : nu + nv + n_eq])
            + contact_qfrc,
            joint_torque=tau,
            status=np.asarray(status),
            contact_force=self._body_contact_forces(rows, f),
            contact_qfrc=contact_qfrc,
        )

    def _contact_rows(self) -> tuple[np.ndarray, np.ndarray]:
        """Pyramid rows of contacts with world-fixed bodies and each row's capacity [N]."""
        m, d = self.model, self.data
        if not self.config.contact_forces:
            return np.zeros(0, dtype=int), np.zeros(0)
        rows = np.flatnonzero(np.isin(d.efc_type[: d.nefc], _CONTACT_ROWS))
        con = d.efc_id[rows]
        weld = m.body_weldid[m.geom_bodyid]
        ground = (weld[d.contact.geom1[con]] == 0) | (weld[d.contact.geom2[con]] == 0)
        rows, con = rows[ground], con[ground]
        reach = np.maximum(d.contact.dist[con], 0.0) / self.config.contact_distance
        return rows, self._contact_force * np.clip(1.0 - reach, 1e-3, 1.0)

    def _body_contact_forces(self, rows: np.ndarray, f: np.ndarray) -> np.ndarray:
        """World-frame contact force on each body from the pyramid edge forces *f*."""
        m, d = self.model, self.data
        out = np.zeros((m.nbody, 3))
        if not len(rows):
            return out
        d.efc_force[: d.nefc] = 0.0
        d.efc_force[rows] = f
        wrench = np.zeros(6)
        for c in np.unique(d.efc_id[rows]):
            mujoco.mj_contactForce(m, d, int(c), wrench)
            # Force of geom1 on geom2 in the contact frame, whose rows are its axes.
            force = d.contact.frame[c].reshape(3, 3).T @ wrench[:3]
            out[m.geom_bodyid[d.contact.geom2[c]]] += force
            out[m.geom_bodyid[d.contact.geom1[c]]] -= force
        return out


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
            :func:`~myosuite.biomechanics.inverse_dynamics.clip_kinematics`.
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
