# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Local geometric and multi-camera observability of musculoskeletal landmarks.

Rank and conditioning describe infinitesimal motion at one pose, not statistical
uncertainty, tracking accuracy or global uniqueness. Omitted independent DOFs are
free nuisance variables unless explicitly declared fixed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from myosuite.biomechanics.camera import PinholeCamera
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
    """Describe tangent DOFs not driven or locked by active joint equalities."""
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
    """Target observability after allowing free nuisance motion."""

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


@dataclass(frozen=True)
class CameraObservabilityResult:
    """Multi-camera pixel-ray and self-occlusion observability at one posture."""

    observability: ObservabilityResult
    camera_ids: tuple[str, ...]
    visibility: np.ndarray
    views_per_landmark: np.ndarray
    active_landmarks: tuple[str, ...]
    pixel_scale_px: float

    def record(self, *, include_matrices: bool = False) -> dict:
        """Return strict JSON-compatible camera and DOF observability diagnostics."""
        out = self.observability.record(include_matrices=include_matrices)
        out.update(
            {
                "camera_ids": self.camera_ids,
                "visibility": self.visibility.tolist(),
                "views_per_landmark": self.views_per_landmark.tolist(),
                "active_landmarks": self.active_landmarks,
                "pixel_scale_px": self.pixel_scale_px,
            }
        )
        return out


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


def _prepare_state_and_landmarks(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    landmarks: Sequence[Landmark],
) -> tuple[
    mujoco.MjData,
    tuple[DofInfo, ...],
    np.ndarray,
    list[str],
    list[int],
    list[np.ndarray],
    list[np.ndarray],
    list[float],
]:
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

    names: list[str] = []
    bodies: list[int] = []
    points: list[np.ndarray] = []
    jacs: list[np.ndarray] = []
    lm_weights: list[float] = []
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
        names.append(lm.source)
        bodies.append(int(body))
        points.append(point)
        jacs.append(jac @ basis)
        lm_weights.append(float(lm.weight))
    return data, infos, basis, names, bodies, points, jacs, lm_weights


def _solve_svd_observability(
    infos: tuple[DofInfo, ...],
    landmark_names: Sequence[str],
    native_rows: np.ndarray,
    row_weights: np.ndarray,
    *,
    target_dofs: Sequence[int] | None,
    fixed_dofs: Sequence[int],
    dof_scales: Mapping[int, float] | None,
    translation_scale_m: float,
    rotation_scale_rad: float,
    obs_scale: float,
    rtol: float,
    atol: float,
    ambiguity_tol: float,
) -> ObservabilityResult:
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

    columns = {d.index: i for i, d in enumerate(infos)}
    tc, nc = [columns[i] for i in target], [columns[i] for i in nuisance]
    ts = np.array([scales[i] for i in target])
    ns = np.array([scales[i] for i in nuisance])
    factor = row_weights[:, None] / obs_scale
    a, n = native_rows[:, tc] * ts * factor, native_rows[:, nc] * ns * factor
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
        tuple(landmark_names),
        tuple(by_index[i] for i in target),
        tuple(by_index[i] for i in nuisance),
        tuple(by_index[i] for i in fixed),
        ts,
        ns,
        float(obs_scale),
        native_rows[:, tc],
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
    """Analyse a coupled 3D landmark position Jacobian at one posture."""
    limits = [translation_scale_m, rotation_scale_rad, landmark_scale_m]
    if not np.all(np.isfinite(limits)) or min(limits) <= 0:
        raise ValueError("characteristic scales must be positive and finite")
    if (
        not np.all(np.isfinite([rtol, atol, ambiguity_tol]))
        or min(rtol, atol) < 0
        or not 0 < ambiguity_tol < 1
    ):
        raise ValueError("invalid rank or ambiguity tolerances")
    _, infos, _, names, _, _, jacs, lm_weights = _prepare_state_and_landmarks(
        model, qpos, landmarks
    )
    native = np.vstack(jacs) if jacs else np.zeros((0, len(infos)))
    weights = np.repeat(np.array(lm_weights, dtype=float), 3)
    return _solve_svd_observability(
        infos,
        names,
        native,
        weights,
        target_dofs=target_dofs,
        fixed_dofs=fixed_dofs,
        dof_scales=dof_scales,
        translation_scale_m=translation_scale_m,
        rotation_scale_rad=rotation_scale_rad,
        obs_scale=landmark_scale_m,
        rtol=rtol,
        atol=atol,
        ambiguity_tol=ambiguity_tol,
    )


def _ray_occluded(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    origin: np.ndarray,
    target: np.ndarray,
    landmark_body: int,
    geomgroup: np.ndarray | None,
    margin_m: float,
) -> bool:
    vec = target - origin
    total = float(np.linalg.norm(vec))
    if total <= margin_m:
        return False
    direction = vec / total
    weld_self = int(model.body_weldid[landmark_body])
    travelled = 0.0
    gid = np.array([-1], dtype=np.int32)
    for _ in range(8):
        start = origin + travelled * direction
        dist = float(
            mujoco.mj_ray(
                model, data, start, direction, geomgroup, 1, landmark_body, gid
            )
        )
        if dist < 0 or gid[0] < 0 or travelled + dist >= total - margin_m:
            return False
        hit_weld = int(model.body_weldid[model.geom_bodyid[int(gid[0])]])
        if hit_weld != weld_self:
            return True
        travelled += dist + 1e-4
    return False


def camera_landmark_observability(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    landmarks: Sequence[Landmark],
    cameras: Sequence[PinholeCamera],
    *,
    image_sizes: Sequence[tuple[int, int] | None] | None = None,
    check_occlusion: bool = True,
    occlusion_geomgroup: Sequence[int] | None = None,
    occlusion_margin_m: float = 2e-3,
    min_views: int = 1,
    min_depth_m: float = 1e-3,
    target_dofs: Sequence[int] | None = None,
    fixed_dofs: Sequence[int] = (),
    dof_scales: Mapping[int, float] | None = None,
    translation_scale_m: float = 0.1,
    rotation_scale_rad: float = 1.0,
    pixel_scale_px: float = 1.0,
    rtol: float = 1e-6,
    atol: float = 1e-12,
    ambiguity_tol: float = 1e-6,
) -> CameraObservabilityResult:
    """Analyse coupled musculoskeletal DOF observability through calibrated cameras.

    Projects each landmark's coupled 3D Jacobian onto the 2D image planes of all
    cameras that have an unoccluded, in-frustum view, inspired by markerless/marker
    rig design in ``lab-camera-optimizer`` (Delaplace et al., 2026).

    Args:
        model: Compiled model, unchanged.
        qpos: Finite ``(nq,)`` pose.
        landmarks: Landmarks evaluated on ``model``.
        cameras: One or more calibrated pinhole cameras.
        image_sizes: Optional ``(width, height)`` bounds per camera; falls back to
            ``camera.image_size`` when ``None``.
        check_occlusion: Raycast with ``mujoco.mj_ray`` to drop self-occluded views.
        occlusion_geomgroup: Optional 6-element ``mjNGROUP`` mask for ``mj_ray``.
        occlusion_margin_m: Ray termination margin near the landmark surface [m].
        min_views: Minimum cameras required for a landmark to contribute (`1` for
            monocular/pixel-ray equations, `2` for stereo triangulation coverage).
        min_depth_m: Minimum optical-axis depth ``z_c`` [m].
        target_dofs: Independent DOF indices to assess; ``None`` assesses all non-fixed.
        fixed_dofs: Independent DOF indices assumed known/fixed.
        dof_scales: Characteristic DOF increments by index.
        translation_scale_m: Default translational increment [m].
        rotation_scale_rad: Default rotational increment [rad].
        pixel_scale_px: Characteristic image-plane normalization [px].
        rtol, atol: Singular-value rank tolerances.
        ambiguity_tol: Nullspace loading threshold in ``(0, 1)``.

    Returns:
        Pixel-space :class:`ObservabilityResult` plus per-camera visibility matrices.
    """
    if not cameras:
        raise ValueError("cameras must contain at least one PinholeCamera")
    limits = [translation_scale_m, rotation_scale_rad, pixel_scale_px, min_depth_m]
    if (
        not np.all(np.isfinite(limits))
        or min(limits) <= 0
        or min_views < 1
        or occlusion_margin_m < 0
    ):
        raise ValueError("invalid scales, min_views, min_depth_m or occlusion_margin_m")
    if (
        not np.all(np.isfinite([rtol, atol, ambiguity_tol]))
        or min(rtol, atol) < 0
        or not 0 < ambiguity_tol < 1
    ):
        raise ValueError("invalid rank or ambiguity tolerances")
    if image_sizes is not None and len(image_sizes) != len(cameras):
        raise ValueError("image_sizes must match len(cameras)")

    ggroup = None
    if occlusion_geomgroup is not None:
        ggroup = np.asarray(occlusion_geomgroup, dtype=np.uint8)
        if ggroup.shape != (mujoco.mjNGROUP,):
            raise ValueError(f"occlusion_geomgroup must have length {mujoco.mjNGROUP}")

    data, infos, _, names, bodies, points, jacs, lm_weights = (
        _prepare_state_and_landmarks(model, qpos, landmarks)
    )
    n_cam, n_lm = len(cameras), len(names)
    visibility = np.zeros((n_cam, n_lm), dtype=bool)
    cam_jacs: dict[tuple[int, int], np.ndarray] = {}

    for c_idx, cam in enumerate(cameras):
        size = (
            getattr(cam, "image_size", None)
            if image_sizes is None
            else image_sizes[c_idx]
        )
        k = cam.intrinsic
        fx, skew, cx, fy, cy = k[0, 0], k[0, 1], k[0, 2], k[1, 1], k[1, 2]
        origin = cam.center
        for l_idx in range(n_lm):
            pt = points[l_idx]
            xc = cam.rotation @ pt + cam.translation
            if xc[2] <= min_depth_m:
                continue
            inv_z = 1.0 / xc[2]
            u = (fx * xc[0] + skew * xc[1]) * inv_z + cx
            v = fy * xc[1] * inv_z + cy
            if size is not None and not (0.0 <= u <= size[0] and 0.0 <= v <= size[1]):
                continue
            if check_occlusion and _ray_occluded(
                model, data, origin, pt, bodies[l_idx], ggroup, occlusion_margin_m
            ):
                continue
            visibility[c_idx, l_idx] = True
            inv_z2 = inv_z * inv_z
            j_proj = (
                np.array(
                    [
                        [
                            fx * inv_z,
                            skew * inv_z,
                            -(fx * xc[0] + skew * xc[1]) * inv_z2,
                        ],
                        [0.0, fy * inv_z, -fy * xc[1] * inv_z2],
                    ]
                )
                @ cam.rotation
            )
            cam_jacs[c_idx, l_idx] = j_proj @ jacs[l_idx]

    views = visibility.sum(axis=0)
    rows: list[np.ndarray] = []
    row_weights: list[float] = []
    active: list[str] = []
    for l_idx, name in enumerate(names):
        if views[l_idx] < min_views:
            continue
        active.append(name)
        for c_idx in np.flatnonzero(visibility[:, l_idx]):
            rows.append(cam_jacs[int(c_idx), l_idx])
            row_weights.extend([lm_weights[l_idx], lm_weights[l_idx]])

    native = np.vstack(rows) if rows else np.zeros((0, len(infos)))
    obs = _solve_svd_observability(
        infos,
        names,
        native,
        np.array(row_weights, dtype=float),
        target_dofs=target_dofs,
        fixed_dofs=fixed_dofs,
        dof_scales=dof_scales,
        translation_scale_m=translation_scale_m,
        rotation_scale_rad=rotation_scale_rad,
        obs_scale=pixel_scale_px,
        rtol=rtol,
        atol=atol,
        ambiguity_tol=ambiguity_tol,
    )
    return CameraObservabilityResult(
        obs,
        tuple(c.calibration_id for c in cameras),
        visibility,
        views,
        tuple(active),
        float(pixel_scale_px),
    )
