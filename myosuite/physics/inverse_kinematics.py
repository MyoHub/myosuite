# Copyright (c) MyoSuite Authors. All rights reserved.
# Copyright 2017-2018 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Inverse kinematics on MuJoCo models.

:func:`solve_ik` places one or more sites at target positions and/or
orientations on a raw ``MjModel``. It fits the selected joints by damped least
squares on MuJoCo's analytic site and equality-constraint Jacobians.

:func:`qpos_from_site_pose` and :func:`nullspace_method` are the earlier
single-site solver, adapted from dm_control for RoboHive's physics class. The
former is deprecated in favour of :func:`solve_ik`.
"""

from __future__ import annotations

import collections
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mujoco
import numpy as np
from absl import logging
from numpy.typing import ArrayLike

_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_BALL = int(mujoco.mjtJoint.mjJNT_BALL)
_EQ_JOINT = int(mujoco.mjtEq.mjEQ_JOINT)
_EQ_TENDON = int(mujoco.mjtEq.mjEQ_TENDON)
_EQ_FLEX = int(mujoco.mjtEq.mjEQ_FLEX)
_EQUALITY = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)

#: A site target: a position ``(3,)``, or a ``(position, quaternion)`` pair in
#: which either entry may be ``None``. Quaternions are ``wxyz`` in the world frame.
SiteTarget = ArrayLike | tuple[ArrayLike | None, ArrayLike | None]


class IKError(ValueError):
    """Raised by :func:`solve_ik` when the targets cannot be reached.

    Attributes:
        qpos: Closest pose found, shape ``(nq,)``.
        error: Largest remaining error: a target distance in metres, a target
            angle in radians or an equality-constraint violation.
    """

    def __init__(self, message: str, qpos: np.ndarray, error: float) -> None:
        super().__init__(message)
        self.qpos = qpos
        self.error = error


def solve_ik(
    model: mujoco.MjModel,
    targets: Mapping[str, SiteTarget],
    *,
    joints: Sequence[str] | None = None,
    qpos: ArrayLike | None = None,
    strict: bool = True,
    retry: bool = True,
    tol: float = 1e-4,
    rot_weight: float = 1.0,
    equality_weight: float = 100.0,
    max_evals: int = 200,
) -> np.ndarray:
    """Return joint positions that place sites at target positions and poses.

    The selected joints are fitted by damped least squares (Levenberg-Marquardt)
    on analytic Jacobians: ``mj_jacSite`` for the site targets and MuJoCo's
    constraint Jacobian for the active equality constraints. Each step is a
    tangent-space velocity applied with ``mj_integratePos``, so ball and free
    joints move on their quaternion manifold. Unselected joints keep their
    seed value.

    * Active equality constraints (``model.eq_active0``) are residuals with
      weight *equality_weight*, so a coupled joint stays tied to its joint.
      Joints that follow a selected joint through a joint equality are
      selected with it. A first pass weighs these residuals like the targets,
      which avoids getting stuck with coupled joints at their limits.
    * Limited hinge and slide coordinates are clipped to ``jnt_range``; a
      coordinate held at a limit by the descent direction is frozen for that
      step.
    * A limited ball joint is solved without its cone limit, which is not a
      scalar bound: the result may leave the cone.
    * The fit is local. A seed with straight, limit-bound joints (such as
      ``qpos0`` of the arm models) can stall, so by default an unreached
      target is solved again from the middle of the joint ranges and the
      closer pose is kept.

    Example::

        qpos = solve_ik(model, {"S_grasp": [0.2, 0.1, 0.4]})
        qpos = solve_ik(model, {"S_grasp": (pos, quat)}, joints=["elbow_flexion"])

    Args:
        model: Compiled model; it is not modified.
        targets: Site name to target: a world position ``(3,)``, or a
            ``(position, quaternion)`` pair with a ``wxyz`` world orientation.
            Either entry of the pair may be ``None``.
        joints: Joints that may move. ``None`` selects every joint.
        qpos: Seed pose, shape ``(nq,)``. Defaults to ``model.qpos0``.
        strict: Raise :class:`IKError` when a target error or equality
            violation stays above *tol*. With ``False`` the closest pose found
            is returned instead.
        retry: Solve again from mid-range joint values when *tol* is not met.
            Turn it off to track a motion from the previous frame's pose.
        tol: Reach tolerance, in metres for positions and radians for
            orientations and angular constraints.
        rot_weight: Weight of orientation residuals (radians) relative to
            position residuals (metres).
        equality_weight: Weight of equality-constraint residuals.
        max_evals: Maximum number of residual evaluations.

    Returns:
        The solved ``qpos``, shape ``(nq,)``.

    Raises:
        IKError: If *strict* and the targets cannot be reached.
        KeyError: If a site or joint name is not in the model.
        ValueError: If a target, seed or option is invalid.
    """
    for name, value in (
        ("tol", tol),
        ("rot_weight", rot_weight),
        ("equality_weight", equality_weight),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value}.")
    if max_evals < 1:
        raise ValueError(f"max_evals must be at least 1, got {max_evals}.")
    problem = _IKProblem(model, _joint_ids(model, joints), rot_weight, equality_weight)
    parsed, seed = _parse_targets(model, targets), _seed(model, qpos)
    result = problem.solve(parsed, seed, max_evals)
    error, label = problem.worst_error()
    middle = problem.mid_range(seed)
    if retry and error > tol and not np.array_equal(middle, seed):
        first, first_cost = (result, error, label), problem.cost()
        result = problem.solve(parsed, middle, max_evals)
        error, label = problem.worst_error()
        if first_cost <= problem.cost():  # keep the closer of the two poses
            result, error, label = first
    if strict and error > tol:
        raise IKError(
            f"solve_ik could not reach the targets: {label} is off by {error:.3g} "
            f"(tol {tol:g}). Pass strict=False for the closest pose, which is also "
            "on this error's .qpos.",
            result,
            error,
        )
    return result


@dataclass(frozen=True)
class _Targets:
    """Parsed site targets: positions and unit world quaternions (wxyz)."""

    pos_sites: np.ndarray
    pos: np.ndarray
    rot_sites: np.ndarray
    quat: np.ndarray


def _parse_targets(
    model: mujoco.MjModel, targets: Mapping[str, SiteTarget]
) -> _Targets:
    if not targets:
        raise ValueError("solve_ik needs at least one site target.")
    pos_sites, pos, rot_sites, quat = [], [], [], []
    for name, value in targets.items():
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site < 0:
            raise KeyError(f"No site named {name!r} in the model.")
        pair = isinstance(value, tuple | list) and len(value) == 2
        target_pos, target_quat = value if pair else (value, None)
        if target_pos is None and target_quat is None:
            raise ValueError(f"Target {name!r} has no position and no orientation.")
        if target_pos is not None:
            p = np.asarray(target_pos, dtype=float)
            if p.shape != (3,) or not np.all(np.isfinite(p)):
                raise ValueError(
                    f"Position target of {name!r} must be a finite 3-vector."
                )
            pos_sites.append(site)
            pos.append(p)
        if target_quat is not None:
            q = np.asarray(target_quat, dtype=float)
            norm = float(np.linalg.norm(q)) if q.shape == (4,) else 0.0
            if not np.isfinite(norm) or norm == 0:
                raise ValueError(
                    f"Orientation target of {name!r} must be a finite nonzero wxyz quaternion."
                )
            rot_sites.append(site)
            quat.append(q / norm)
    return _Targets(
        np.array(pos_sites, dtype=int),
        np.array(pos, dtype=float).reshape(-1, 3),
        np.array(rot_sites, dtype=int),
        np.array(quat, dtype=float).reshape(-1, 4),
    )


def _joint_ids(model: mujoco.MjModel, joints: Sequence[str] | None) -> list[int]:
    """Selected joint ids, plus joints that follow them through joint equalities."""
    if joints is None:
        return list(range(model.njnt))
    names = [joints] if isinstance(joints, str) else list(joints)
    if not names or len(set(names)) != len(names):
        raise ValueError("joints must be a nonempty list of unique joint names.")
    selected = set()
    for name in names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise KeyError(f"No joint named {name!r} in the model.")
        selected.add(joint)
    # A joint equality ties joint1 to a polynomial of joint2: joint1 follows.
    couplings = [
        (int(model.eq_obj1id[e]), int(model.eq_obj2id[e]))
        for e in range(model.neq)
        if model.eq_type[e] == _EQ_JOINT and model.eq_active0[e]
    ]
    grown = True
    while grown:
        grown = False
        for follower, leader in couplings:
            if leader in selected and follower not in selected:
                selected.add(follower)
                grown = True
    return sorted(selected)


def _seed(model: mujoco.MjModel, qpos: ArrayLike | None) -> np.ndarray:
    seed = np.array(model.qpos0 if qpos is None else qpos, dtype=float, copy=True)
    if seed.shape != (model.nq,) or not np.all(np.isfinite(seed)):
        raise ValueError(f"qpos must be finite with shape ({model.nq},).")
    mujoco.mj_normalizeQuat(model, seed)
    return seed


def _damped_step(jac: np.ndarray, residual: np.ndarray, damping: float) -> np.ndarray:
    """Levenberg step ``argmin_s |J s + r|^2 + damping |s|^2``, in the smaller space."""
    rows, cols = jac.shape
    if rows < cols:
        return -jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(rows), residual)
    return -np.linalg.solve(jac.T @ jac + damping * np.eye(cols), jac.T @ residual)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _right_jacobian(phi: np.ndarray) -> np.ndarray:
    """SO(3) right Jacobian: ``q exp(phi + e) = q exp(phi) exp(J_r(phi) e)``."""
    theta = float(np.linalg.norm(phi))
    if theta < 1e-2:  # Taylor series, avoiding cancellation
        t2 = theta * theta
        a = 0.5 - t2 / 24.0 + t2 * t2 / 720.0
        b = 1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0
    else:
        a = (1.0 - np.cos(theta)) / theta**2
        b = (theta - np.sin(theta)) / theta**3
    k = _skew(phi)
    return np.eye(3) - a * k + b * (k @ k)


def _left_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    """Inverse SO(3) left Jacobian: ``log(exp(w) exp(phi)) = phi + J_l^-1(phi) w``."""
    theta = float(np.linalg.norm(phi))
    if theta < 1e-2:
        t2 = theta * theta
        c = 1.0 / 12.0 + t2 / 720.0 + t2 * t2 / 30240.0
    else:
        c = 1.0 / theta**2 - 1.0 / (2.0 * theta * np.tan(theta / 2.0))
    k = _skew(phi)
    return np.eye(3) - 0.5 * k + c * (k @ k)


class _IKProblem:
    """Residuals and Jacobian of site targets over a tangent step from a seed pose.

    The unknown ``x`` holds the selected DOFs of a velocity that
    ``mj_integratePos`` applies for unit time. Hinge and slide entries are
    coordinate offsets; ball and free rotations are local rotation vectors.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_ids: Sequence[int],
        rot_weight: float,
        equality_weight: float,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.rot_weight = rot_weight
        self.equality_weight = equality_weight
        self._weight = equality_weight  # weight of the current pass
        dofs: list[int] = []
        self._rot_cols: list[int] = []  # first column of each rotation-vector block
        self._limited: list[
            tuple[int, int, float, float]
        ] = []  # column, qpos adr, range
        for joint in joint_ids:
            kind, dof = int(model.jnt_type[joint]), int(model.jnt_dofadr[joint])
            if kind in (_FREE, _BALL):
                self._rot_cols.append(len(dofs) + (3 if kind == _FREE else 0))
                dofs.extend(range(dof, dof + (6 if kind == _FREE else 3)))
                continue
            if model.jnt_limited[joint]:
                low, high = (float(v) for v in model.jnt_range[joint])
                if high <= low:  # a locked joint cannot move
                    continue
                self._limited.append(
                    (len(dofs), int(model.jnt_qposadr[joint]), low, high)
                )
            dofs.append(dof)
        if not dofs:
            raise ValueError("No movable joint is selected.")
        self._dofs = np.array(dofs, dtype=int)
        active = [
            int(model.eq_type[e]) for e in range(model.neq) if model.eq_active0[e]
        ]
        disabled = model.opt.disableflags & (
            mujoco.mjtDisableBit.mjDSBL_EQUALITY
            | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
        )
        self._has_equality = bool(active) and not disabled
        self._needs_tendon = _EQ_TENDON in active
        self._needs_flex = _EQ_FLEX in active
        self._velocity = np.zeros(model.nv)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._x: np.ndarray | None = None
        self._eq_rows = np.zeros(0, dtype=int)
        self.evals = 0  # residual evaluations of the last solve

    def prepare(
        self, targets: _Targets, seed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Set the targets and the seed (clipped into range); return the step bounds."""
        self.targets = targets
        self._conj = targets.quat * np.array([1.0, -1.0, -1.0, -1.0])
        n = len(self._dofs)
        lower, upper = np.full(n, -np.inf), np.full(n, np.inf)
        seed = seed.copy()
        for column, address, low, high in self._limited:
            seed[address] = np.clip(seed[address], low, high)
            lower[column], upper[column] = low - seed[address], high - seed[address]
        self._seed, self._x = seed, None
        self._forward(np.zeros(n))
        self._eq_rows = self._equality_rows()
        return lower, upper

    def mid_range(self, seed: np.ndarray) -> np.ndarray:
        """*seed* with each selected limited hinge or slide at the middle of its range."""
        middle = seed.copy()
        for _, address, low, high in self._limited:
            middle[address] = 0.5 * (low + high)
        return middle

    def cost(self) -> float:
        """Weighted squared residual at the last pose."""
        residual = self.residual(
            self._x if self._x is not None else np.zeros(len(self._dofs))
        )
        return float(residual @ residual)

    def solve(self, targets: _Targets, seed: np.ndarray, max_evals: int) -> np.ndarray:
        """Fit *targets* from *seed* within *max_evals* evaluations; return qpos.

        With equality rows, a first pass weighs them like the targets (weight
        1) and a second pass applies the full weight. The soft pass escapes
        minima where a stiff coupling pins joints at their limits; the second
        keeps the coupling tight when the targets are out of reach.
        """
        lower, upper = self.prepare(targets, seed)
        x = np.zeros(len(lower))
        self.evals = 0
        if len(self._eq_rows) and self.equality_weight > 1.0:
            self._weight = 1.0
            x = self._levenberg_marquardt(x, lower, upper, max_evals // 2)
        self._weight = self.equality_weight
        x = self._levenberg_marquardt(x, lower, upper, max_evals)
        self._forward(x)
        return self.data.qpos.copy()

    def _levenberg_marquardt(
        self, x: np.ndarray, lower: np.ndarray, upper: np.ndarray, max_evals: int
    ) -> np.ndarray:
        """Projected Levenberg-Marquardt from *x* until ``self.evals == max_evals``.

        SciPy's bounded ``least_squares`` (trf, dogbox) was tried first: on
        MyoArm, whose 38 joints are all limited, it often stalled at the
        evaluation budget, while this loop converges in about ten evaluations.
        """
        n = len(x)
        residual = self.residual(x)
        self.evals += 1
        cost, damping = float(residual @ residual), 1e-3
        while self.evals < max_evals and cost > 1e-24:
            jac = self.jacobian(x)
            grad = jac.T @ residual
            # Freeze coordinates that the descent direction pushes past a limit.
            free = ~(((x <= lower) & (grad > 0)) | ((x >= upper) & (grad < 0)))
            if not np.any(free) or np.max(np.abs(grad[free])) <= 1e-15:
                break
            jac_free, accepted = jac[:, free], False
            while self.evals < max_evals and damping < 1e12:
                step = np.zeros(n)
                step[free] = _damped_step(jac_free, residual, damping)
                trial = np.clip(x + step, lower, upper)
                trial_residual = self.residual(trial)
                self.evals += 1
                trial_cost = float(trial_residual @ trial_residual)
                if trial_cost < cost:
                    accepted = True
                    break
                damping *= 4.0
            if not accepted:
                break
            converged = cost - trial_cost <= 1e-10 * cost
            x, residual, cost = trial, trial_residual, trial_cost
            damping = max(damping / 3.0, 1e-12)
            if converged:
                break
        return x

    def _forward(self, x: np.ndarray) -> None:
        """Pose the model at ``seed + x``: kinematics, and constraints if needed."""
        if self._x is not None and np.array_equal(x, self._x):
            return
        model, data = self.model, self.data
        data.qpos[:] = self._seed
        self._velocity[self._dofs] = x
        mujoco.mj_integratePos(model, data.qpos, self._velocity, 1.0)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        if self._has_equality:
            if self._needs_tendon:
                mujoco.mj_tendon(model, data)
            if self._needs_flex:
                mujoco.mj_flex(model, data)
            mujoco.mj_makeConstraint(model, data)
        self._x = x.copy()

    def _efc_jacobian(self, rows: np.ndarray) -> np.ndarray:
        """Dense constraint Jacobian rows (``efc_J`` may be stored sparse)."""
        data, nv = self.data, self.model.nv
        if not mujoco.mj_isSparse(self.model):
            return data.efc_J[: data.nefc * nv].reshape(data.nefc, nv)[rows]
        dense = np.zeros((len(rows), nv))
        for i, row in enumerate(rows):
            start = data.efc_J_rowadr[row]
            stop = start + data.efc_J_rownnz[row]
            dense[i, data.efc_J_colind[start:stop]] = data.efc_J[start:stop]
        return dense

    def _equality_rows(self) -> np.ndarray:
        """Equality rows that the selected DOFs change (the others are constant)."""
        if not self._has_equality:
            return np.zeros(0, dtype=int)
        data = self.data
        rows = np.flatnonzero(data.efc_type[: data.nefc] == _EQUALITY)
        moving = np.any(self._efc_jacobian(rows)[:, self._dofs] != 0, axis=1)
        return rows[moving]

    def _rot_errors(self) -> np.ndarray:
        """World rotation vectors ``log(R_site R_target^T)`` of orientation targets."""
        sites = self.targets.rot_sites
        errors = np.zeros((len(sites), 3))
        quat, diff = np.zeros(4), np.zeros(4)
        for i, site in enumerate(sites):
            mujoco.mju_mat2Quat(quat, self.data.site_xmat[site])
            mujoco.mju_mulQuat(diff, quat, self._conj[i])
            mujoco.mju_quat2Vel(errors[i], diff, 1.0)
        return errors

    def residual(self, x: np.ndarray) -> np.ndarray:
        """Stacked site positions, weighted orientations and equality violations."""
        self._forward(x)
        targets, data = self.targets, self.data
        return np.concatenate(
            (
                (data.site_xpos[targets.pos_sites] - targets.pos).ravel(),
                self.rot_weight * self._rot_errors().ravel(),
                self._weight * data.efc_pos[self._eq_rows],
            )
        )

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Analytic Jacobian of :meth:`residual` with respect to ``x``."""
        self._forward(x)
        model, data, targets, dofs = self.model, self.data, self.targets, self._dofs
        rows = (
            3 * len(targets.pos_sites) + 3 * len(targets.rot_sites) + len(self._eq_rows)
        )
        jac = np.empty((rows, len(dofs)))
        row = 0
        for site in targets.pos_sites:
            mujoco.mj_jacSite(model, data, self._jacp, None, int(site))
            jac[row : row + 3] = self._jacp[:, dofs]
            row += 3
        for site, error in zip(targets.rot_sites, self._rot_errors()):
            mujoco.mj_jacSite(model, data, None, self._jacr, int(site))
            jac[row : row + 3] = self.rot_weight * (
                _left_jacobian_inv(error) @ self._jacr[:, dofs]
            )
            row += 3
        if len(self._eq_rows):
            jac[row:] = self._weight * self._efc_jacobian(self._eq_rows)[:, dofs]
        # mj_jacSite and efc_J map a velocity at the current pose; chain the
        # rotation-vector steps of ball and free joints through J_r.
        for column in self._rot_cols:
            block = slice(column, column + 3)
            jac[:, block] = jac[:, block] @ _right_jacobian(x[block])
        return jac

    def worst_error(self) -> tuple[float, str]:
        """Largest unweighted error at the last pose, and what it belongs to."""
        model, targets, data = self.model, self.targets, self.data
        distance = np.linalg.norm(
            data.site_xpos[targets.pos_sites] - targets.pos, axis=1
        )
        errors = [
            (float(e), f"site {model.site(int(s)).name!r} position")
            for e, s in zip(distance, targets.pos_sites)
        ]
        angle = np.linalg.norm(self._rot_errors(), axis=1)
        errors += [
            (float(e), f"site {model.site(int(s)).name!r} orientation")
            for e, s in zip(angle, targets.rot_sites)
        ]
        if len(self._eq_rows):
            violation = np.abs(data.efc_pos[self._eq_rows])
            worst = int(np.argmax(violation))
            equality = int(data.efc_id[self._eq_rows[worst]])
            label = model.equality(equality).name or f"#{equality}"
            errors.append((float(violation[worst]), f"equality {label!r}"))
        return max(errors)


_INVALID_JOINT_NAMES_TYPE = (
    "`joint_names` must be either None, a list, a tuple, or a numpy array; got {}."
)
_REQUIRE_TARGET_POS_OR_QUAT = (
    "At least one of `target_pos` or `target_quat` must be specified."
)

IKResult = collections.namedtuple("IKResult", ["qpos", "err_norm", "steps", "success"])


def qpos_from_site_pose(
    physics,
    site_name,
    target_pos=None,
    target_quat=None,
    joint_names=None,
    tol=1e-14,
    rot_weight=1.0,
    regularization_threshold=0.1,
    regularization_strength=3e-2,
    max_update_norm=2.0,
    progress_thresh=20.0,
    max_steps=500,
    inplace=False,
):
    """Find joint positions that satisfy a target site position and/or rotation.

    .. deprecated::
        Use :func:`solve_ik` on a raw ``MjModel``:
        ``solve_ik(model, {site_name: (target_pos, target_quat)}, joints=...)``.
        It fits several sites at once, respects joint ranges and keeps
        equality-coupled joints tied.

    Args:
      physics: A `mujoco.Physics` instance.
      site_name: A string specifying the name of the target site.
      target_pos: A (3,) numpy array specifying the desired Cartesian position of
        the site, or None if the position should be unconstrained (default).
        One or both of `target_pos` or `target_quat` must be specified.
      target_quat: A (4,) numpy array specifying the desired orientation of the
        site as a quaternion, or None if the orientation should be unconstrained
        (default). One or both of `target_pos` or `target_quat` must be specified.
      joint_names: (optional) A list, tuple or numpy array specifying the names of
        one or more joints that can be manipulated in order to achieve the target
        site pose. If None (default), all joints may be manipulated.
      tol: (optional) Precision goal for `qpos` (the maximum value of `err_norm`
        in the stopping criterion).
      rot_weight: (optional) Determines the weight given to rotational error
        relative to translational error.
      regularization_threshold: (optional) L2 regularization will be used when
        inverting the Jacobian whilst `err_norm` is greater than this value.
      regularization_strength: (optional) Coefficient of the quadratic penalty
        on joint movements.
      max_update_norm: (optional) The maximum L2 norm of the update applied to
        the joint positions on each iteration. The update vector will be scaled
        such that its magnitude never exceeds this value.
      progress_thresh: (optional) If `err_norm` divided by the magnitude of the
        joint position update is greater than this value then the optimization
        will terminate prematurely. This is a useful heuristic to avoid getting
        stuck in local minima.
      max_steps: (optional) The maximum number of iterations to perform.
      inplace: (optional) If True, `physics.data` will be modified in place.
        Default value is False, i.e. a copy of `physics.data` will be made.

    Returns:
      An `IKResult` namedtuple with the following fields:
        qpos: An (nq,) numpy array of joint positions.
        err_norm: A float, the weighted sum of L2 norms for the residual
          translational and rotational errors.
        steps: An int, the number of iterations that were performed.
        success: Boolean, True if we converged on a solution within `max_steps`,
          False otherwise.

    Raises:
      ValueError: If both `target_pos` and `target_quat` are None, or if
        `joint_names` has an invalid type.
    """
    warnings.warn(
        "qpos_from_site_pose is deprecated and will be removed in a future release; "
        "use myosuite.physics.inverse_kinematics.solve_ik(model, "
        "{site_name: (target_pos, target_quat)}) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    dtype = physics.data.qpos.dtype
    model = physics.get_handle(physics.model)
    data = physics.get_handle(physics.data)
    mjlib = physics.get_mjlib()

    if target_pos is not None and target_quat is not None:
        jac = np.empty((6, model.nv), dtype=dtype)
        err = np.empty(6, dtype=dtype)
        jac_pos, jac_rot = jac[:3], jac[3:]
        err_pos, err_rot = err[:3], err[3:]
    else:
        jac = np.zeros((3, model.nv), dtype=dtype)
        err = np.zeros(3, dtype=dtype)
        if target_pos is not None:
            jac_pos, jac_rot = jac, None
            err_pos, err_rot = err, None
        elif target_quat is not None:
            jac_pos, jac_rot = None, jac
            err_pos, err_rot = None, err
        else:
            raise ValueError(_REQUIRE_TARGET_POS_OR_QUAT)

    update_nv = np.zeros(model.nv, dtype=dtype)

    if target_quat is not None:
        site_xquat = np.empty(4, dtype=dtype)
        neg_site_xquat = np.empty(4, dtype=dtype)
        err_rot_quat = np.empty(4, dtype=dtype)

    if not inplace:
        # physics = physics.copy(share_model=True)
        old_state = physics.get_state()

    # Ensure that the Cartesian position of the site is up to date.
    mjlib.mj_fwdPosition(model, data)

    # Convert site name to index.
    #   site_id = physics.model.name2id(site_name, 'site')
    site_id = physics.model.site_name2id(site_name)

    # These are views onto the underlying MuJoCo buffers. mj_fwdPosition will
    # update them in place, so we can avoid indexing overhead in the main loop.
    site_xpos = physics.data.site_xpos[site_id]
    site_xmat = physics.data.site_xmat[site_id]
    #   site_xpos = physics.named.data.site_xpos[site_name]
    #   site_xmat = physics.named.data.site_xmat[site_name]

    # This is an index into the rows of `update` and the columns of `jac`
    # that selects DOFs associated with joints that we are allowed to manipulate.
    if joint_names is None:
        dof_indices = slice(None)  # Update all DOFs.
    elif isinstance(joint_names, (list, np.ndarray, tuple)):
        if isinstance(joint_names, tuple):
            joint_names = list(joint_names)
        # Find the indices of the DOFs belonging to each named joint. Note that
        # these are not necessarily the same as the joint IDs, since a single joint
        # may have >1 DOF (e.g. ball joints).
        indexer = physics.named.model.dof_jntid.axes.row
        # `dof_jntid` is an `(nv,)` array indexed by joint name. We use its row
        # indexer to map each joint name to the indices of its corresponding DOFs.
        dof_indices = indexer.convert_key_item(joint_names)
    else:
        raise ValueError(_INVALID_JOINT_NAMES_TYPE.format(type(joint_names)))

    steps = 0
    success = False

    for steps in range(max_steps):
        err_norm = 0.0

        if target_pos is not None:
            # Translational error.
            err_pos[:] = target_pos - site_xpos
            err_norm += np.linalg.norm(err_pos)
        if target_quat is not None:
            # Rotational error.
            mjlib.mju_mat2Quat(site_xquat, site_xmat)
            mjlib.mju_negQuat(neg_site_xquat, site_xquat)
            mjlib.mju_mulQuat(err_rot_quat, target_quat, neg_site_xquat)
            mjlib.mju_quat2Vel(err_rot, err_rot_quat, 1)
            err_norm += np.linalg.norm(err_rot) * rot_weight

        if err_norm < tol:
            logging.debug("Converged after %i steps: err_norm=%3g", steps, err_norm)
            success = True
            break
        else:
            # TODO(b/112141670): Generalize this to other entities besides sites.
            mjlib.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
            jac_joints = jac[:, dof_indices]

            # TODO(b/112141592): This does not take joint limits into consideration.
            reg_strength = (
                regularization_strength if err_norm > regularization_threshold else 0.0
            )
            update_joints = nullspace_method(
                jac_joints, err, regularization_strength=reg_strength
            )

            update_norm = np.linalg.norm(update_joints)

            # Check whether we are still making enough progress, and halt if not.
            progress_criterion = err_norm / update_norm
            if progress_criterion > progress_thresh:
                logging.debug(
                    "Step %2i: err_norm / update_norm (%3g) > "
                    "tolerance (%3g). Halting due to insufficient progress",
                    steps,
                    progress_criterion,
                    progress_thresh,
                )
                break

            if update_norm > max_update_norm:
                update_joints *= max_update_norm / update_norm

            # Write the entries for the specified joints into the full `update_nv`
            # vector.
            update_nv[dof_indices] = update_joints

            # Update `physics.qpos`, taking quaternions into account.
            mjlib.mj_integratePos(model, physics.data.qpos, update_nv, 1)

            # Compute the new Cartesian position of the site.
            mjlib.mj_fwdPosition(model, data)

            logging.debug(
                "Step %2i: err_norm=%-10.3g update_norm=%-10.3g",
                steps,
                err_norm,
                update_norm,
            )

    if not success and steps == max_steps - 1:
        logging.warning(
            "Failed to converge after %i steps: err_norm=%3g", steps, err_norm
        )

    if not inplace:
        # Our temporary copy of physics.data is about to go out of scope, and when
        # it does the underlying mjData pointer will be freed and physics.data.qpos
        # will be a view onto a block of deallocated memory. We therefore need to
        # make a copy of physics.data.qpos while physics.data is still alive.
        qpos = physics.data.qpos.copy()
        physics.set_state(**old_state)
        physics.forward()
    else:
        # If we're modifying physics.data in place then it's fine to return a view.
        qpos = physics.data.qpos

    return IKResult(qpos=qpos, err_norm=err_norm, steps=steps, success=success)


def nullspace_method(jac_joints, delta, regularization_strength=0.0):
    """Calculates the joint velocities to achieve a specified end effector delta.

    Args:
      jac_joints: The Jacobian of the end effector with respect to the joints. A
        numpy array of shape `(ndelta, nv)`, where `ndelta` is the size of `delta`
        and `nv` is the number of degrees of freedom.
      delta: The desired end-effector delta. A numpy array of shape `(3,)` or
        `(6,)` containing either position deltas, rotation deltas, or both.
      regularization_strength: (optional) Coefficient of the quadratic penalty
        on joint movements. Default is zero, i.e. no regularization.

    Returns:
      An `(nv,)` numpy array of joint velocities.

    Reference:
      Buss, S. R. S. (2004). Introduction to inverse kinematics with jacobian
      transpose, pseudoinverse and damped least squares methods.
      https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
    """
    hess_approx = jac_joints.T.dot(jac_joints)
    joint_delta = jac_joints.T.dot(delta)
    if regularization_strength > 0:
        # L2 regularization
        hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
        return np.linalg.solve(hess_approx, joint_delta)
    else:
        return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]
