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

With a ``grip`` (:class:`GripContactConfig`) the hand holds a handle, a mocap
body that follows the hand: its contacts with the hand are unknowns too, on
MuJoCo's friction pyramid. With a ``loaded_joint`` the handle transmits the
external generalized force on that joint's coordinate (a dynamometer's torque):
``s^T J_c^T f = s^T e``, ``s`` the coordinate's motion (couplings included) and
``e`` the ``external_qfrc``, whose part on that coordinate is then dropped. The
grip force follows from equilibrium and minimum effort: muscles that also flex
the fingers (FDS, FDP) can help the wrist, as the handle takes their finger torques.

Activations are a minimum-effort estimate: co-contraction and activation dynamics
are not predicted.
"""

from __future__ import annotations

import copy
import warnings
from collections import Counter
from dataclasses import dataclass, fields, replace
from typing import Any, Literal, get_args

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
OnFailure = Literal["warn", "raise", "nan"]


class StaticOptimizationWarning(UserWarning):
    """OSQP did not solve some frames of a static optimization."""


class StaticOptimizationError(RuntimeError):
    """OSQP did not solve a frame of a static optimization (``on_failure="raise"``)."""


@dataclass(frozen=True)
class GripContactConfig:
    """A handle held in the hand, with the hand-handle contact forces as unknowns.

    Attributes:
        handle_body: Mocap body carrying the handle geoms
            (:class:`myosuite.biomechanics.grip.GripHandle`).
        hand_body: Body the handle is held in; the handle follows its pose.
        handle_pos: Handle position in the hand body's frame [m].
        handle_quat: Handle orientation in the hand body's frame (w, x, y, z).
        loaded_joint: Hinge or slide joint whose part of ``external_qfrc`` the
            handle transmits (a dynamometer axis aligned with that joint);
            ``None`` adds the contacts without a load.
        contact_distance: Gap up to which a hand geom can push on the handle [m].
        optimal_force: ``F_contact`` per pyramid edge [N], whose forces cost
            ``(f / F_contact)**2``; ``None`` is ten times the weight of the
            moving bodies (at least 1 N), so grip forces cost next to nothing
            against muscle effort.
    """

    handle_body: str
    hand_body: str
    handle_pos: tuple[float, float, float]
    handle_quat: tuple[float, float, float, float]
    loaded_joint: str | None = None
    contact_distance: float = 0.005
    optimal_force: float | None = None


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
        on_failure: What to do with a frame OSQP did not solve (any status but
            ``"solved"``, e.g. ``"maximum iterations reached"``): ``"warn"``
            keeps its unconverged solution and warns
            (:func:`static_optimization` once per clip, with the failed frame
            count), ``"raise"`` raises :class:`StaticOptimizationError`, and
            ``"nan"`` returns NaN activations and forces for it (``joint_torque``,
            ``status`` and solver diagnostics stay).
        supported_dofs: Dof addresses whose load an external support carries
            (a strap, a brace, a dynamometer). Their reserves cost nothing, as
            free-joint residuals do without contacts, so the muscles crossing
            them need not cancel their own torques there with antagonists.
        grip: A handle held in the hand, whose contact forces are unknowns
            (``None``: no handle).
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
    on_failure: OnFailure = "warn"
    supported_dofs: tuple[int, ...] = ()
    grip: GripContactConfig | None = None


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
        iterations: ``(...)`` OSQP iteration count.
        primal_residual: ``(...)`` OSQP primal residual of the QP.
        dual_residual: ``(...)`` OSQP dual residual of the QP.
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
    iterations: np.ndarray
    primal_residual: np.ndarray
    dual_residual: np.ndarray


def _near_optimal(
    matrix: sparse.csc_matrix,
    lower: np.ndarray,
    upper: np.ndarray,
    weight: np.ndarray,
    primal: np.ndarray,
    dual: np.ndarray,
    tolerance: float,
) -> bool:
    """Check the previous solution against the rebuilt QP's absolute tolerance."""
    if not (np.isfinite(primal).all() and np.isfinite(dual).all()):
        return False
    activity = matrix @ primal
    active = np.abs(dual) > tolerance
    bounds = np.where(dual[active] > 0, upper[active], lower[active])
    return bool(
        np.all(activity >= lower - tolerance)
        and np.all(activity <= upper + tolerance)
        and np.all(np.abs(activity[active] - bounds) <= tolerance)
        and np.all(np.abs(2.0 * weight * primal + matrix.T @ dual) <= tolerance)
    )


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
                joint ``actuatorfrcrange``) or has more than one activation, or
                ``on_failure`` is not one of ``"warn"``, ``"raise"``, ``"nan"``.
        """
        self._initialize(copy.deepcopy(model), config)

    @classmethod
    def _from_owned_model(
        cls, model: mujoco.MjModel, config: StaticOptimizationConfig | None = None
    ) -> StaticOptimizer:
        """Take ownership of a private model that no caller will reuse."""
        optimizer = cls.__new__(cls)
        optimizer._initialize(model, config)
        return optimizer

    def _initialize(
        self, m: mujoco.MjModel, config: StaticOptimizationConfig | None
    ) -> None:
        """Prepare the model owned by this optimizer."""
        cfg = config or StaticOptimizationConfig()
        if cfg.on_failure not in get_args(OnFailure):
            raise ValueError(
                f"on_failure must be one of {get_args(OnFailure)}: {cfg.on_failure!r}"
            )
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
        if cfg.contact_forces or cfg.grip is not None:
            m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        if cfg.contact_forces:
            m.geom_margin[:] = np.maximum(m.geom_margin, cfg.contact_distance)
            m.geom_gap[:] = 0.0
            m.pair_margin[:] = np.maximum(m.pair_margin, cfg.contact_distance)
            m.pair_gap[:] = 0.0
        elif cfg.grip is None:
            m.opt.disableflags |= int(flags.mjDSBL_CONTACT)
        self._grip = self._prepare_grip(m, cfg.grip)
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
        supported = np.asarray(cfg.supported_dofs, dtype=int)
        self._optimal_force[supported] = 1.0
        self._reserve_weight[supported] = 0.0
        # The settings as solved, with the model-derived defaults filled in.
        self.config = replace(
            cfg,
            reserve_optimal_force=tuple(self._optimal_force.tolist()),
            contact_optimal_force=self._contact_force,
            residual_optimal_force=residual,
        )
        self._solver: osqp.OSQP | None = None
        self._pattern: np.ndarray | None = None
        self._warm_start: tuple[np.ndarray, np.ndarray] | None = None

    @staticmethod
    def _prepare_grip(
        m: mujoco.MjModel, grip: GripContactConfig | None
    ) -> dict[str, Any] | None:
        """Ids of the handle, check it is a mocap body, give its geoms their reach."""
        if grip is None:
            return None
        handle = m.body(grip.handle_body).id
        if m.body_mocapid[handle] < 0:
            raise ValueError(f"grip handle {grip.handle_body!r} is not a mocap body")
        geoms = np.flatnonzero(m.geom_bodyid == handle)
        m.geom_margin[geoms] = grip.contact_distance
        m.geom_gap[geoms] = 0.0
        dof = -1
        if grip.loaded_joint is not None:
            j = m.joint(grip.loaded_joint).id
            if int(m.jnt_type[j]) not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                raise ValueError(f"loaded joint {grip.loaded_joint!r} is not 1-dof")
            dof = int(m.jnt_dofadr[j])
        # Edge forces are solved in units of the body weight (at least 1 N).
        scale = max(
            float(
                m.body_mass[m.body_weldid != 0].sum() * np.linalg.norm(m.opt.gravity)
            ),
            1.0,
        )
        return {
            "handle": handle,
            "mocap": int(m.body_mocapid[handle]),
            "hand": m.body(grip.hand_body).id,
            "dof": dof,
            "force": scale,
            "weight": (scale / (grip.optimal_force or 10.0 * scale)) ** 2,
        }

    def _place_handle(self) -> None:
        """Move the handle to its pose in the hand (after the kinematics)."""
        g, cfg, d = self._grip, self.config.grip, self.data
        rot = d.xmat[g["hand"]].reshape(3, 3)
        d.mocap_pos[g["mocap"]] = d.xpos[g["hand"]] + rot @ np.asarray(cfg.handle_pos)
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, d.xquat[g["hand"]], np.asarray(cfg.handle_quat))
        d.mocap_quat[g["mocap"]] = quat

    def peak_actuator_qfrc(self) -> np.ndarray:
        """Per dof, ``sum_i |R_ij| F_i`` at ``qpos0`` and rest [N m or N].

        ``R`` are the moment arms; ``F`` each muscle's peak isometric force and,
        for other actuators, the largest absolute force at either end of the
        activation or control range, including bias. This scales the default reserves.

        Returns:
            ``(nv,)`` peak generalized force the actuators can apply at each dof
            if all pulled the same way.
        """
        m, d = self.model, self.data
        d.qpos[:], d.qvel[:] = m.qpos0, 0.0
        mujoco.mj_fwdPosition(m, d)
        mujoco.mj_fwdVelocity(m, d)
        lo, hi = self._probe
        f_hi, f_lo = self._actuator_force(hi), self._actuator_force(lo)
        force = np.maximum(np.abs(f_hi), np.abs(f_lo))
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
    ) -> tuple[np.ndarray | None, Any]:
        """Minimise the cost subject to ``lower <= A x <= upper``, warm-started.

        OSQP's matrix keeps the union of all sparsity patterns seen so far, so most
        frames only update values. A rebuild can reuse converged primal and
        dual values if they still approximately satisfy the new problem.
        """
        pattern = A != 0
        previous_pattern = self._pattern
        same_shape = previous_pattern is not None and previous_pattern.shape == A.shape
        if same_shape and not (pattern & ~previous_pattern).any():
            self._solver.update(Ax=A.T[previous_pattern.T], l=lower, u=upper)
        else:
            if same_shape:
                pattern |= previous_pattern
            cols, rows = np.nonzero(pattern.T)
            indptr = np.r_[0, np.cumsum(np.bincount(cols, minlength=A.shape[1]))]
            A_csc = sparse.csc_matrix((A[rows, cols], rows, indptr), shape=A.shape)
            solver = osqp.OSQP()
            solver.setup(
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
            if same_shape and self._warm_start is not None:
                primal, dual = self._warm_start
                if _near_optimal(
                    A_csc, lower, upper, weight, primal, dual, self.config.tolerance
                ):
                    solver.warm_start(x=primal, y=dual)
            self._solver, self._pattern = solver, pattern
        res = self._solver.solve(raise_error=False)
        self._warm_start = (
            (res.x, res.y)
            if res.info.status == "solved" and res.x is not None and res.y is not None
            else None
        )
        return res.x, res.info

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

        Raises:
            StaticOptimizationError: If OSQP did not solve it and
                ``on_failure="raise"``.

        Warns:
            StaticOptimizationWarning: If OSQP did not solve it and
                ``on_failure="warn"``.
        """
        lo, hi = self._lo, self._hi
        if max_activation is not None:
            hi = np.maximum(np.minimum(hi, max_activation), lo)
        if min_activation is not None:
            lo = np.minimum(np.maximum(lo, min_activation), hi)
        m, d = self.model, self.data
        d.qpos[:], d.qvel[:] = qpos, qvel
        if self._grip is not None:
            mujoco.mj_kinematics(m, d)
            self._place_handle()
        mujoco.mj_fwdPosition(m, d)
        mujoco.mj_fwdVelocity(m, d)
        tau = required_qfrc(m, d, qacc)
        efc_J = d.efc_J[: d.nefc * m.nv].reshape(d.nefc, m.nv)
        J_eq = efc_J[d.efc_type[: d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY]
        rows, capacity, n_grip = self._contact_rows()
        # The handle transmits the external force on the loaded coordinate.
        load_row = None
        if n_grip and self._grip["dof"] >= 0 and external_qfrc is not None:
            dof = self._grip["dof"]
            lhs = np.vstack([J_eq, np.eye(m.nv)[dof]])
            s = np.linalg.lstsq(lhs, np.eye(len(lhs))[-1], rcond=None)[0]
            load = float(s @ external_qfrc)
            external_qfrc = np.array(external_qfrc, dtype=np.float64)
            external_qfrc[dof] -= load
            load_row = (s @ efc_J[rows[:n_grip]].T * capacity[:n_grip], load)
        if external_qfrc is not None:
            tau -= external_qfrc

        f0, f1 = (self._actuator_force(p) for p in self._probe)
        gain = (f1 - f0) / (self._probe[1] - self._probe[0])
        bias = f0 - gain * self._probe[0]
        moment = np.zeros((m.nu, m.nv))
        mujoco.mju_sparse2dense(
            moment, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind
        )
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
        if load_row is not None:
            row = np.zeros((1, A.shape[1]))
            row[0, nu + nv + n_eq : nu + nv + n_eq + n_grip] = load_row[0]
            A = np.vstack([A, row])
            lower, upper = np.r_[lower, load_row[1]], np.r_[upper, load_row[1]]
        weight = np.r_[
            np.ones(nu),
            self._reserve_weight,
            np.full(n_eq, _CONSTRAINT_WEIGHT),
            np.full(n_grip, self._grip["weight"] if n_grip else 0.0),
            np.ones(n_c - n_grip),
        ]
        x, solver = self._solve_qp(A, lower, upper, weight)
        status = solver if isinstance(solver, str) else solver.status
        iterations = 0 if isinstance(solver, str) else solver.iter
        primal_residual = np.nan if isinstance(solver, str) else solver.prim_res
        dual_residual = np.nan if isinstance(solver, str) else solver.dual_res
        if x is None or x.dtype == object or not np.all(np.isfinite(x)):
            zero_a = np.clip(np.zeros(nu), lo, hi)
            failed = StaticOptimizationResult(
                activation=zero_a,
                actuator_force=gain * zero_a + bias,
                reserve=np.zeros(nv),
                constraint_force=np.zeros(nv),
                joint_torque=tau,
                status=np.asarray(status),
                contact_force=np.zeros((m.nbody, 3)),
                contact_qfrc=np.zeros(nv),
                iterations=np.asarray(iterations),
                primal_residual=np.asarray(primal_residual),
                dual_residual=np.asarray(dual_residual),
            )
            return self._apply_failure_policy(failed)

        a = np.clip(x[:nu], lo, hi)
        f = capacity * np.maximum(x[nu + nv + n_eq :], 0.0)
        contact_qfrc = efc_J[rows].T @ f
        result = StaticOptimizationResult(
            activation=a,
            actuator_force=gain * a + bias,
            reserve=self._optimal_force * x[nu : nu + nv],
            constraint_force=J_eq.T @ (_CONSTRAINT_SCALE * x[nu + nv : nu + nv + n_eq])
            + contact_qfrc,
            joint_torque=tau,
            status=np.asarray(status),
            contact_force=self._body_contact_forces(rows, f),
            contact_qfrc=contact_qfrc,
            iterations=np.asarray(iterations),
            primal_residual=np.asarray(primal_residual),
            dual_residual=np.asarray(dual_residual),
        )
        return self._apply_failure_policy(result)

    def _apply_failure_policy(
        self, result: StaticOptimizationResult
    ) -> StaticOptimizationResult:
        """*result* as ``config.on_failure`` says, if OSQP did not solve its frame."""
        status = str(result.status)
        if status == "solved":
            return result
        if self.config.on_failure == "raise":
            raise StaticOptimizationError(f"OSQP did not solve the frame: {status}")
        if self.config.on_failure == "warn":
            warnings.warn(
                f"OSQP did not solve the frame ({status}); its activations are not optimal",
                StaticOptimizationWarning,
                stacklevel=3,
            )
            return result
        kept = (
            "joint_torque",
            "status",
            "iterations",
            "primal_residual",
            "dual_residual",
        )
        return replace(
            result,
            **{
                f.name: np.full(np.shape(getattr(result, f.name)), np.nan)
                for f in fields(result)
                if f.name not in kept
            },
        )

    def _contact_rows(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Pyramid rows of the handle's and then the ground's contacts with the body.

        Returns the rows, each row's capacity [N] and the number of handle rows.
        """
        m, d = self.model, self.data
        rows = np.flatnonzero(np.isin(d.efc_type[: d.nefc], _CONTACT_ROWS))
        con = d.efc_id[rows]
        body1 = m.geom_bodyid[d.contact.geom1[con]]
        body2 = m.geom_bodyid[d.contact.geom2[con]]
        weld1, weld2 = m.body_weldid[body1], m.body_weldid[body2]
        out_rows, out_cap = [np.zeros(0, dtype=int)], [np.zeros(0)]
        n_grip = 0
        if self._grip is not None:
            h, g = self._grip["handle"], self.config.grip
            held = ((body1 == h) & (weld2 != 0)) | ((body2 == h) & (weld1 != 0))
            reach = np.maximum(d.contact.dist[con[held]], 0.0) / g.contact_distance
            out_rows.append(rows[held])
            out_cap.append(self._grip["force"] * np.clip(1.0 - reach, 1e-3, 1.0))
            n_grip = int(held.sum())
        if self.config.contact_forces:
            ground = (weld1 == 0) | (weld2 == 0)
            reach = np.maximum(d.contact.dist[con[ground]], 0.0)
            reach = reach / self.config.contact_distance
            out_rows.append(rows[ground])
            out_cap.append(self._contact_force * np.clip(1.0 - reach, 1e-3, 1.0))
        return np.concatenate(out_rows), np.concatenate(out_cap), n_grip

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
        config: Weights, tolerances and ``on_failure``.

    Returns:
        The per-frame solutions stacked along a leading axis.

    Raises:
        StaticOptimizationError: At the first frame OSQP did not solve, with
            ``on_failure="raise"``.

    Warns:
        StaticOptimizationWarning: Once, with the number of frames OSQP did not
            solve and their statuses, with ``on_failure="warn"``.
    """
    return _solve_clip(
        StaticOptimizer(model, config), qpos, qvel, qacc, external_qfrc, stacklevel=3
    )


def _solve_clip(
    so: StaticOptimizer,
    qpos: np.ndarray,
    qvel: np.ndarray,
    qacc: np.ndarray,
    external_qfrc: np.ndarray | None = None,
    *,
    stacklevel: int = 2,
) -> StaticOptimizationResult:
    """Solve a clip on an existing optimizer, including private sampled models."""
    frames = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", StaticOptimizationWarning)
        for t in range(len(qpos)):
            ext = None if external_qfrc is None else external_qfrc[t]
            try:
                frames.append(so.solve(qpos[t], qvel[t], qacc[t], ext))
            except StaticOptimizationError as exc:
                raise StaticOptimizationError(
                    f"frame {t} of {len(qpos)}: {exc}"
                ) from exc
    result = StaticOptimizationResult(
        **{
            f.name: np.stack([getattr(fr, f.name) for fr in frames])
            for f in fields(StaticOptimizationResult)
        }
    )
    failed = result.status != "solved"
    if failed.any() and so.config.on_failure == "warn":
        warnings.warn(
            f"OSQP did not solve {int(failed.sum())} of {len(failed)} frames "
            f"({dict(Counter(result.status[failed].tolist()))}); their activations "
            "are not optimal (StaticOptimizationConfig: max_iter, tolerance, on_failure)",
            StaticOptimizationWarning,
            stacklevel=stacklevel,
        )
    return result
