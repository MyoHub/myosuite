# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Local geometric observability of landmark positions, without posture priors.

Rank and conditioning describe infinitesimal motion at one pose, not statistical
uncertainty, tracking accuracy or global uniqueness. Omitted independent DOFs are
free nuisance variables unless explicitly declared fixed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from myosuite.biomechanics.retarget import Landmark
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings


@dataclass(frozen=True)
class DofInfo:
    """A full-model tangent DOF, its joint/axis label and native unit."""

    index: int
    name: str
    joint: str
    axis: str
    unit: str


def independent_dofs(model: mujoco.MjModel) -> tuple[DofInfo, ...]:
    """Describe tangent DOFs not driven or locked by active joint equalities.

    Args:
        model: Compiled model, unchanged.

    Returns:
        DOFs in model order. Free-root translations use world axes, while ball
        and free-root angular velocities use MuJoCo's local tangent axes.
    """
    c = JointCouplings.from_model(model)
    driven = set(c.dep_dof.tolist() + c.locked_dof.tolist())
    result = []
    for j in range(model.njnt):
        kind, start = int(model.jnt_type[j]), int(model.jnt_dofadr[j])
        joint = model.joint(j).name or f"joint[{j}]"
        axes = (
            ("tx", "ty", "tz", "rx", "ry", "rz")
            if kind == 0
            else ("rx", "ry", "rz")
            if kind == 1
            else ("",)
        )
        for offset, axis in enumerate(axes):
            index = start + offset
            if index not in driven:
                unit = "m" if axis.startswith("t") or kind == 2 else "rad"
                result.append(
                    DofInfo(
                        index, f"{joint}/{axis}" if axis else joint, joint, axis, unit
                    )
                )
    return tuple(result)


@dataclass(frozen=True)
class ObservabilityResult:
    """Target observability after allowing free nuisance motion.

    ``jacobian`` is the native position derivative in m per target DOF unit.
    ``effective_jacobian`` is dimensionless: landmark weights, characteristic
    DOF scales and landmark_scale_m are applied, then nuisance motion projected
    out. Nullspace rows and their per-coordinate loadings use these scaled target
    coordinates. Loadings indicate ambiguity, not probabilities or variances.
    """

    landmark_names: tuple[str, ...]
    target_dofs: tuple[DofInfo, ...]
    nuisance_dofs: tuple[DofInfo, ...]
    fixed_dofs: tuple[DofInfo, ...]
    target_scales: np.ndarray
    nuisance_scales: np.ndarray
    landmark_scale_m: float
    jacobian: np.ndarray
    effective_jacobian: np.ndarray
    singular_values: np.ndarray
    rank: int
    nuisance_rank: int
    rank_threshold: float
    condition_number: float
    nullspace: np.ndarray
    nullspace_loadings: np.ndarray
    ambiguous_dofs: tuple[str, ...]
    unconstrained_dofs: tuple[str, ...]
    ambiguous_groups: tuple[tuple[str, ...], ...]

    def record(self, *, include_matrices: bool = False) -> dict:
        """Return strict JSON-compatible diagnostics; deficient condition is null."""
        count = len(self.target_dofs)
        result = {
            "status": "empty_scope"
            if not count
            else "full_rank"
            if self.rank == count
            else "rank_deficient",
            "rank": self.rank,
            "target_count": count,
            "nuisance_count": len(self.nuisance_dofs),
            "fixed_count": len(self.fixed_dofs),
            "nuisance_rank": self.nuisance_rank,
            "rank_threshold": self.rank_threshold,
            "condition_number": self.condition_number
            if np.isfinite(self.condition_number)
            else None,
            "landmark_names": self.landmark_names,
            "landmark_scale_m": self.landmark_scale_m,
            "singular_values": self.singular_values.tolist(),
            "nullspace_loadings": self.nullspace_loadings.tolist(),
            "ambiguous_dofs": self.ambiguous_dofs,
            "unconstrained_dofs": self.unconstrained_dofs,
            "ambiguous_groups": self.ambiguous_groups,
        }
        for name in ("target_dofs", "nuisance_dofs", "fixed_dofs"):
            result[name] = [asdict(d) for d in getattr(self, name)]
        for name in ("target_scales", "nuisance_scales"):
            result[name] = getattr(self, name).tolist()
        if include_matrices:
            for name in ("jacobian", "effective_jacobian", "nullspace"):
                result[name] = getattr(self, name).tolist()
        return result


def _selection(values: Sequence[int], allowed: set[int], name: str) -> list[int]:
    if any(
        isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
        for i in values
    ):
        raise ValueError(f"{name} must contain integer independent DOF indices")
    result = [int(i) for i in values]
    if len(set(result)) != len(result) or not set(result) <= allowed:
        raise ValueError(f"{name} must contain distinct independent DOF indices")
    return result


def _groups(projector: np.ndarray, names: tuple[str, ...], tol: float) -> tuple:
    pending = set(np.flatnonzero(np.sqrt(np.maximum(projector.diagonal(), 0)) > tol))
    groups = []
    while pending:
        component, frontier = [], [min(pending)]
        while frontier:
            index = frontier.pop()
            if index not in pending:
                continue
            pending.remove(index)
            component.append(index)
            frontier.extend(i for i in pending if abs(projector[index, i]) > tol)
        groups.append(tuple(names[i] for i in sorted(component)))
    return tuple(groups)


def landmark_observability(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    landmarks: Sequence[Landmark],
    *,
    target_dofs: Sequence[int] | None = None,
    fixed_dofs: Sequence[int] = (),
    dof_scales: Mapping[int, float] | None = None,
    translation_scale_m: float = 0.1,
    rotation_scale_rad: float = 1.0,
    landmark_scale_m: float = 1.0,
    rtol: float = 1e-6,
    atol: float = 1e-12,
    ambiguity_tol: float = 1e-6,
) -> ObservabilityResult:
    """Analyse a coupled landmark position Jacobian at one posture.

    Args:
        model: Compiled model, unchanged. Joint equality couplings are projected.
        qpos: Finite ``(nq,)`` pose; nonzero quaternions are normalized on a copy.
        landmarks: Existing retarget landmarks, using sites or body-local offsets.
            Named sites must exist. Weights are nonnegative relative multipliers.
        target_dofs: Full-model independent tangent indices to assess; None selects
            all independent DOFs except those explicitly fixed. Others are free
            nuisance DOFs, not silently fixed assumptions.
        fixed_dofs: Independent tangent indices explicitly assumed known/fixed.
        dof_scales: Positive characteristic physical increments by full DOF index,
            including any nuisance overrides (metres or radians, not precision).
        translation_scale_m: Default characteristic translation increment [m].
        rotation_scale_rad: Default characteristic angular increment [rad].
        landmark_scale_m: Position normalization [m].
        rtol, atol: Rank cutoff is max(atol, rtol * full scaled Jacobian norm).
        ambiguity_tol: Null-projector loading/group threshold in (0, 1).

    Returns:
        SVD diagnostics of target motion after nuisance projection. Priors, joint
        limits, orientations, temporal information and sensor noise are excluded.
    """
    limits = [translation_scale_m, rotation_scale_rad, landmark_scale_m]
    if not np.all(np.isfinite(limits)) or min(limits) <= 0:
        raise ValueError("characteristic scales must be positive and finite")
    if (
        not np.all(np.isfinite([rtol, atol, ambiguity_tol]))
        or min(rtol, atol) < 0
        or not 0 < ambiguity_tol < 1
    ):
        raise ValueError("invalid rank or ambiguity tolerances")
    q = np.array(qpos, dtype=float, copy=True)
    if q.shape != (model.nq,) or not np.isfinite(q).all():
        raise ValueError("qpos must be finite with shape (model.nq,)")
    for j in range(model.njnt):
        kind, address = int(model.jnt_type[j]), int(model.jnt_qposadr[j])
        if kind in (0, 1):
            start = address + (3 if kind == 0 else 0)
            if np.linalg.norm(q[start : start + 4]) < 1e-12:
                raise ValueError("pose quaternions must be nonzero")
    mujoco.mj_normalizeQuat(model, q)
    infos = independent_dofs(model)
    by_index = {d.index: d for d in infos}
    allowed = set(by_index)
    fixed = _selection(fixed_dofs, allowed, "fixed_dofs")
    target = (
        [d.index for d in infos if d.index not in fixed]
        if target_dofs is None
        else _selection(target_dofs, allowed, "target_dofs")
    )
    if set(target) & set(fixed):
        raise ValueError("target_dofs and fixed_dofs must be disjoint")
    nuisance = [
        d.index for d in infos if d.index not in target and d.index not in fixed
    ]
    overrides = {} if dof_scales is None else dict(dof_scales)
    _selection(list(overrides), allowed, "dof_scales")
    scales = {
        d.index: float(
            overrides.get(
                d.index, translation_scale_m if d.unit == "m" else rotation_scale_rad
            )
        )
        for d in infos
    }
    if not all(np.isfinite(s) and s > 0 for s in scales.values()):
        raise ValueError("DOF scales must be positive and finite")
    c = JointCouplings.from_model(model)
    project_joint_couplings(c, q)
    velocity = np.eye(model.nv)[[d.index for d in infos]]
    poses = np.broadcast_to(q, (len(infos), model.nq)).copy()
    project_joint_couplings(c, poses, velocity)
    basis = velocity.T
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    names, rows, weights = [], [], []
    for lm in landmarks:
        if (
            not isinstance(lm.source, str)
            or not lm.source
            or lm.source in names
            or not np.isfinite(lm.weight)
            or lm.weight < 0
        ):
            raise ValueError(
                "landmarks need unique names and finite nonnegative weights"
            )
        if lm.site is not None:
            site = model.site(lm.site)
            body, offset = int(site.bodyid[0]), site.pos
        else:
            body, offset = model.body(lm.body).id, np.asarray(lm.pos, dtype=float)
        if np.shape(offset) != (3,) or not np.isfinite(offset).all():
            raise ValueError("landmark offsets must be finite three-vectors")
        point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ offset
        jac = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jac, None, point, body)
        rows.append(jac @ basis)
        weights.extend([float(lm.weight)] * 3)
        names.append(lm.source)
    native = np.vstack(rows) if rows else np.zeros((0, len(infos)))
    columns = {d.index: i for i, d in enumerate(infos)}
    tc, nc = [columns[i] for i in target], [columns[i] for i in nuisance]
    ts, ns = (
        np.array([scales[i] for i in target]),
        np.array([scales[i] for i in nuisance]),
    )
    factor = np.array(weights)[:, None] / landmark_scale_m
    a, n = native[:, tc] * ts * factor, native[:, nc] * ns * factor
    combined = np.column_stack((a, n))
    if not np.isfinite(combined).all():
        raise ValueError("scaled landmark Jacobian is not finite")
    full_s = np.linalg.svd(combined, compute_uv=False)
    threshold = max(float(atol), float(rtol) * (float(full_s[0]) if len(full_s) else 0))
    u, s, _ = np.linalg.svd(n, full_matrices=False)
    nr = int(np.count_nonzero(s > threshold))
    effective = a - u[:, :nr] @ (u[:, :nr].T @ a)
    _, singular, vt = np.linalg.svd(
        effective, full_matrices=len(effective) < len(target)
    )
    rank = int(np.count_nonzero(singular > threshold))
    nullspace = vt[rank:]
    projector = nullspace.T @ nullspace
    loadings = np.sqrt(np.maximum(projector.diagonal(), 0))
    labels = tuple(by_index[i].name for i in target)
    condition = (
        1.0
        if not target
        else float("inf")
        if rank < len(target)
        else float(singular[0] / singular[-1])
    )
    return ObservabilityResult(
        tuple(names),
        tuple(by_index[i] for i in target),
        tuple(by_index[i] for i in nuisance),
        tuple(by_index[i] for i in fixed),
        ts,
        ns,
        float(landmark_scale_m),
        native[:, tc],
        effective,
        np.pad(singular, (0, len(target) - len(singular))),
        rank,
        nr,
        threshold,
        condition,
        nullspace,
        loadings,
        tuple(name for name, value in zip(labels, loadings) if value > ambiguity_tol),
        tuple(
            name
            for name, value in zip(labels, np.linalg.norm(effective, axis=0))
            if value <= threshold
        ),
        _groups(projector, labels, ambiguity_tol),
    )
