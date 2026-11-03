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
orientations on a raw ``MjModel``. It iterates the differential IK of `mink
<https://github.com/kevinzakka/mink>`_: each step is a quadratic program on
MuJoCo's analytic Jacobians, with the joint ranges as bounds.

:func:`qpos_from_site_pose` and :func:`nullspace_method` are the earlier
single-site solver, adapted from dm_control for RoboHive's physics class. The
former is deprecated in favour of :func:`solve_ik`.
"""

from __future__ import annotations

import collections
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mink
import mujoco
import numpy as np
from absl import logging
from numpy.typing import ArrayLike

_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_BALL = int(mujoco.mjtJoint.mjJNT_BALL)
_EQ_JOINT = int(mujoco.mjtEq.mjEQ_JOINT)
_EQ_TENDON = int(mujoco.mjtEq.mjEQ_TENDON)
# Equality types that mink's EqualityConstraintTask regulates (flex is not one).
_REGULATED = {
    int(t)
    for t in (
        mujoco.mjtEq.mjEQ_CONNECT,
        mujoco.mjtEq.mjEQ_WELD,
        mujoco.mjtEq.mjEQ_JOINT,
        mujoco.mjtEq.mjEQ_TENDON,
    )
}
_EQUALITY = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
_SOLVER = "daqp"  # the QP backend installed with mink

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
    max_iters: int = 100,
) -> np.ndarray:
    """Return joint positions that place sites at target positions and poses.

    Each iteration is one step of mink's differential IK: a quadratic program
    on the linearised site errors (``FrameTask``) and equality violations
    (``EqualityConstraintTask``), bounded by the joint ranges
    (``ConfigurationLimit``), with unselected joints held (``DofFreezingTask``).
    The step is applied with ``mj_integratePos``, so ball and free joints move
    on their quaternion manifold. A step that does not lower the weighted
    error is solved again with more Levenberg-Marquardt damping, and
    iteration ends when the error stops decreasing.

    * Active equality constraints (``model.eq_active0``) are costs with weight
      *equality_weight*, so a coupled joint stays tied to its joint. Joints that
      follow a selected joint through a joint equality are selected with it. A
      first pass weighs these costs like the targets, which avoids getting
      stuck with coupled joints at their limits. Flex equalities are ignored.
    * Selected hinge and slide coordinates stay in ``jnt_range``; a seed outside
      it is clipped. mink 1.3 and later bound a limited ball joint's cone to
      first order, so the result may leave the cone slightly; earlier releases
      leave it unbounded.
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
        rot_weight: Weight of orientation errors (radians) relative to
            position errors (metres).
        equality_weight: Weight of equality-constraint violations.
        max_iters: Maximum number of IK steps per pass.

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
    if max_iters < 1:
        raise ValueError(f"max_iters must be at least 1, got {max_iters}.")
    seed = _seed(model, qpos)
    problem = _IKProblem(
        model,
        _joint_ids(model, joints),
        _parse_targets(model, targets),
        seed,
        rot_weight,
        equality_weight,
    )
    result = problem.solve(seed, max_iters)
    error, label = problem.worst_error()
    middle = problem.mid_range(seed)
    if retry and error > tol and not np.array_equal(middle, seed):
        first, first_cost = (result, error, label), problem.cost()
        result = problem.solve(middle, max_iters)
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
class _Target:
    """A parsed site target: a world position and/or a unit world quaternion (wxyz)."""

    site: int
    pos: np.ndarray | None
    quat: np.ndarray | None


def _parse_targets(
    model: mujoco.MjModel, targets: Mapping[str, SiteTarget]
) -> list[_Target]:
    if not targets:
        raise ValueError("solve_ik needs at least one site target.")
    parsed = []
    for name, value in targets.items():
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site < 0:
            raise KeyError(f"No site named {name!r} in the model.")
        pair = isinstance(value, tuple | list) and len(value) == 2
        target_pos, target_quat = value if pair else (value, None)
        if target_pos is None and target_quat is None:
            raise ValueError(f"Target {name!r} has no position and no orientation.")
        pos = quat = None
        if target_pos is not None:
            pos = np.asarray(target_pos, dtype=float)
            if pos.shape != (3,) or not np.all(np.isfinite(pos)):
                raise ValueError(
                    f"Position target of {name!r} must be a finite 3-vector."
                )
        if target_quat is not None:
            q = np.asarray(target_quat, dtype=float)
            norm = float(np.linalg.norm(q)) if q.shape == (4,) else 0.0
            if not np.isfinite(norm) or norm == 0:
                raise ValueError(
                    f"Orientation target of {name!r} must be a finite nonzero wxyz quaternion."
                )
            quat = q / norm
        parsed.append(_Target(site, pos, quat))
    return parsed


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


def _site_quat(data: mujoco.MjData, site: int) -> np.ndarray:
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site])
    return quat


def _angle(qa: np.ndarray, qb: np.ndarray) -> float:
    """Rotation angle between two unit quaternions (atan2 keeps small angles exact)."""
    conj, rel = np.empty(4), np.empty(4)
    mujoco.mju_negQuat(conj, qb)
    mujoco.mju_mulQuat(rel, qa, conj)
    return 2.0 * float(np.arctan2(np.linalg.norm(rel[1:]), abs(rel[0])))


def _out_of_range(model: mujoco.MjModel, qpos: np.ndarray, joint: int) -> bool:
    """Whether a limited joint's ``qpos`` is outside its range (a ball joint's cone)."""
    kind, address = int(model.jnt_type[joint]), int(model.jnt_qposadr[joint])
    low, high = model.jnt_range[joint]
    if kind == _FREE:
        return False
    if kind == _BALL:
        quat = qpos[address : address + 4]
        return 2.0 * np.arctan2(np.linalg.norm(quat[1:]), abs(quat[0])) > high
    return not low <= qpos[address] <= high


def _dense_rows(
    model: mujoco.MjModel, data: mujoco.MjData, rows: np.ndarray
) -> np.ndarray:
    """Dense constraint Jacobian rows (``efc_J`` may be stored sparse)."""
    if not mujoco.mj_isSparse(model):
        return data.efc_J[: data.nefc * model.nv].reshape(data.nefc, model.nv)[rows]
    dense = np.zeros((len(rows), model.nv))
    for i, row in enumerate(rows):
        start = data.efc_J_rowadr[row]
        stop = start + data.efc_J_rownnz[row]
        dense[i, data.efc_J_colind[start:stop]] = data.efc_J[start:stop]
    return dense


class _IKProblem:
    """The mink tasks, bounds and held joints of one :func:`solve_ik` call."""

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_ids: Sequence[int],
        targets: Sequence[_Target],
        seed: np.ndarray,
        rot_weight: float,
        equality_weight: float,
    ) -> None:
        moving: set[int] = set()
        dofs: list[int] = []
        self._limited: list[tuple[int, float, float]] = []  # qpos adr, range
        for joint in joint_ids:
            kind, dof = int(model.jnt_type[joint]), int(model.jnt_dofadr[joint])
            if kind not in (_FREE, _BALL) and model.jnt_limited[joint]:
                low, high = (float(v) for v in model.jnt_range[joint])
                if high <= low:  # a locked joint cannot move
                    continue
                self._limited.append((int(model.jnt_qposadr[joint]), low, high))
            moving.add(joint)
            dofs.extend(range(dof, dof + {_FREE: 6, _BALL: 3}.get(kind, 1)))
        if not dofs:
            raise ValueError("No movable joint is selected.")
        self._dofs = np.array(dofs, dtype=int)
        self._address = np.array([a for a, _, _ in self._limited], dtype=int)
        self._low = np.array([low for _, low, _ in self._limited])
        self._high = np.array([high for _, _, high in self._limited])
        # mink bounds every limited joint, and a held joint whose seed is out of
        # range would make the QP infeasible. mink reads the bounds when its
        # objects are built, so such joints are unbounded only meanwhile.
        held = [
            j
            for j in range(model.njnt)
            if model.jnt_limited[j]
            and j not in moving
            and _out_of_range(model, seed, j)
        ]
        self.model = model
        limited = model.jnt_limited.copy()
        try:
            model.jnt_limited[held] = 0
            self.configuration = mink.Configuration(model)
            self.limits = [mink.ConfigurationLimit(model, gain=1.0)]
        finally:
            model.jnt_limited[:] = limited
        frozen = sorted(set(range(model.nv)) - set(dofs))
        self._frozen = np.array(frozen, dtype=int)
        self.constraints = (
            [mink.DofFreezingTask(model, dof_indices=frozen)] if frozen else []
        )
        self.targets = list(targets)
        self.frames = [
            mink.FrameTask(
                model.site(t.site).name,
                "site",
                position_cost=1.0 if t.pos is not None else 0.0,
                orientation_cost=rot_weight if t.quat is not None else 0.0,
            )
            for t in self.targets
        ]
        self.equality_weight = equality_weight
        self.equality: mink.EqualityConstraintTask | None = None
        self._equality_ids: list[int] = []
        disabled = model.opt.disableflags & (
            mujoco.mjtDisableBit.mjDSBL_EQUALITY
            | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
        )
        # mink before 1.4 leaves tendon lengths stale in Configuration.update.
        self._tendon_equality = not disabled and any(
            model.eq_type[e] == _EQ_TENDON and model.eq_active0[e]
            for e in range(model.neq)
        )

    def solve(self, seed: np.ndarray, max_iters: int) -> np.ndarray:
        """Fit the targets from *seed* (clipped into range); return qpos.

        With equality costs, a first pass weighs them like the targets (weight
        1) and a second pass applies the full weight. The soft pass escapes
        minima where a stiff coupling pins joints at their limits; the second
        keeps the coupling tight when the targets are out of reach.
        """
        q = seed.copy()
        q[self._address] = np.clip(q[self._address], self._low, self._high)
        self._update(q)
        self.equality = self._equality_task()
        if self.equality is not None and self.equality_weight > 1.0:
            self.equality.set_cost(1.0)
            self._descend(max(1, max_iters // 2))
        if self.equality is not None:
            self.equality.set_cost(self.equality_weight)
        self._descend(max_iters)
        return self.configuration.q

    def mid_range(self, seed: np.ndarray) -> np.ndarray:
        """*seed* with each selected limited hinge or slide at the middle of its range."""
        middle = seed.copy()
        for address, low, high in self._limited:
            middle[address] = 0.5 * (low + high)
        return middle

    def cost(self) -> float:
        """Weighted squared error of the mink tasks at the current pose."""
        self._anchor()
        total = 0.0
        for task in self._tasks():
            error = task.compute_error(self.configuration)
            weighted = task.cost * error
            total += float(weighted @ weighted)
        return total

    def worst_error(self) -> tuple[float, str]:
        """Largest unweighted error at the current pose, and what it belongs to."""
        model, data = self.model, self.configuration.data
        errors = []
        for target in self.targets:
            name = model.site(target.site).name
            if target.pos is not None:
                distance = np.linalg.norm(data.site_xpos[target.site] - target.pos)
                errors.append((float(distance), f"site {name!r} position"))
            if target.quat is not None:
                angle = _angle(_site_quat(data, target.site), target.quat)
                errors.append((angle, f"site {name!r} orientation"))
        if self._equality_ids:
            rows = np.flatnonzero(
                (data.efc_type[: data.nefc] == _EQUALITY)
                & np.isin(data.efc_id[: data.nefc], self._equality_ids)
            )
            violation = np.abs(data.efc_pos[rows])
            worst = int(np.argmax(violation))
            equality = int(data.efc_id[rows[worst]])
            label = model.equality(equality).name or f"#{equality}"
            errors.append((float(violation[worst]), f"equality {label!r}"))
        return max(errors)

    def _tasks(self) -> list[mink.Task]:
        return self.frames + ([self.equality] if self.equality is not None else [])

    def _update(self, q: np.ndarray) -> None:
        self.configuration.update(q)
        if self._tendon_equality:
            mujoco.mj_tendon(self.model, self.configuration.data)
            mujoco.mj_makeConstraint(self.model, self.configuration.data)

    def _anchor(self) -> None:
        """Set the frame targets; an absent part copies the site's current one.

        A position target keeps the site's orientation and an orientation
        target its position, so the free part has no error.
        """
        data = self.configuration.data
        for task, target in zip(self.frames, self.targets):
            pos = target.pos if target.pos is not None else data.site_xpos[target.site]
            quat = (
                target.quat
                if target.quat is not None
                else _site_quat(data, target.site)
            )
            task.set_target(mink.SE3(wxyz_xyz=np.concatenate([quat, pos])))

    def _equality_task(self) -> mink.EqualityConstraintTask | None:
        """A cost on the active equality constraints that the selected DOFs change."""
        model, data = self.model, self.configuration.data
        rows = np.flatnonzero(data.efc_type[: data.nefc] == _EQUALITY)
        moving = np.any(_dense_rows(model, data, rows)[:, self._dofs] != 0, axis=1)
        self._equality_ids = sorted(
            {
                int(data.efc_id[row])
                for row in rows[moving]
                if model.eq_type[data.efc_id[row]] in _REGULATED
            }
        )
        if not self._equality_ids:
            return None
        return mink.EqualityConstraintTask(
            model, cost=self.equality_weight, equalities=self._equality_ids
        )

    def _descend(self, max_iters: int) -> None:
        """Take up to *max_iters* mink steps, adapting their Levenberg-Marquardt damping.

        A step that does not lower the cost is solved again with four times the
        damping, which turns it towards the gradient; an accepted step divides
        the damping by three. Descent ends when the cost vanishes or a step
        shrinks below 1e-9 (radians or metres): a relative cost change would
        stop slow fits to out-of-reach targets early.
        """
        model, damping = self.model, 1e-3
        cost = self.cost()
        for _ in range(max_iters):
            if cost <= 1e-24 or damping >= 1e12:
                break
            try:
                velocity = mink.solve_ik(
                    self.configuration,
                    self._tasks(),
                    1.0,
                    _SOLVER,
                    damping=damping,
                    limits=self.limits,
                    constraints=self.constraints,
                )
            except mink.NoSolutionFound:  # the QP failed: keep the closest pose
                break
            velocity[self._frozen] = 0.0  # held exactly, not up to QP round-off
            q = self.configuration.q
            trial = q.copy()
            mujoco.mj_integratePos(model, trial, velocity, 1.0)
            # mink keeps the ranges up to rounding; clipping keeps them exactly.
            trial[self._address] = np.clip(trial[self._address], self._low, self._high)
            self._update(trial)
            trial_cost = self.cost()
            converged = np.max(np.abs(velocity)) <= 1e-9
            if trial_cost >= cost:
                self._update(q)
                self._anchor()
                if converged:
                    break
                damping *= 4.0
                continue
            cost, damping = trial_cost, max(damping / 3.0, 1e-12)
            if converged:
                break


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
