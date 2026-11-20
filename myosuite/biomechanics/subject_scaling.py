# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Bilevel multi-frame anthropometric scaling and coupled LandmarkIK.

Adapts the bilevel optimization pattern of AddBiomechanics (Werling et al., *PLOS ONE*,
2023): an outer bounded nonlinear least-squares solver optimizes constant per-segment
scales while an inner loop alternates coupled :class:`~.retarget.LandmarkIK` and
closed-form Ridge estimation of body-local virtual marker offsets across diverse frames.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import re

import mujoco
import numpy as np
from scipy.optimize import least_squares

from myosuite.biomechanics.retarget import (
    Landmark,
    LandmarkIK,
    LandmarkMap,
    RefitConfig,
)
from myosuite.biomechanics.retarget_instanthmr import scale_spec_from_instanthmr
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings


@dataclass(frozen=True)
class BilevelScaleConfig:
    """Configuration for bilevel multi-frame subject scaling and marker registration."""

    scale_bounds: tuple[float, float] = (0.65, 1.50)
    bilateral_symmetry: bool = True
    max_calibration_frames: int = 16
    optimize_marker_offsets: bool = True
    fixed_landmarks: tuple[str, ...] = ()
    offset_reg_weight: float = 2.0
    scale_reg_weight: float = 1e-2
    outer_max_nfev: int = 80
    inner_alternations: int = 3
    ik_config: RefitConfig = field(
        default_factory=lambda: RefitConfig(smooth_weight=0.01, max_iters=30)
    )


@dataclass(frozen=True)
class BilevelScaleResult:
    """Subject-scaled model, registered landmark map, and full-clip coupled IK poses."""

    scales: dict[str, float]
    scaled_spec: mujoco.MjSpec
    scaled_model: mujoco.MjModel
    landmark_map: LandmarkMap
    marker_offsets: dict[str, np.ndarray]
    qpos: np.ndarray
    initial_rmse_m: float
    final_rmse_m: float
    calibration_frame_indices: np.ndarray
    outer_evaluations: int


def select_diverse_frames(positions: np.ndarray, max_frames: int = 16) -> np.ndarray:
    """Select up to *max_frames* maximally diverse frames via farthest-point sampling."""
    pts = np.asarray(positions, dtype=np.float64)
    if pts.ndim != 3 or pts.shape[2] != 3 or len(pts) == 0:
        raise ValueError(f"Expected (T >= 1, L, 3) positions, got {pts.shape}")
    if max_frames < 1:
        raise ValueError(f"max_frames must be >= 1, got {max_frames}")
    valid = np.flatnonzero(np.isfinite(pts).all(axis=(1, 2)))
    if len(valid) == 0:
        valid = np.flatnonzero(np.isfinite(pts).any(axis=(1, 2)))
    if len(valid) == 0:
        raise ValueError("positions contains no finite landmark coordinates")
    if len(valid) <= max_frames:
        return valid.astype(np.int64)

    sub = np.nan_to_num(pts[valid] - np.nanmean(pts[valid], axis=1, keepdims=True))
    flat = sub.reshape(len(valid), -1)
    med = np.median(flat, axis=0)
    first = int(np.argmin(np.linalg.norm(flat - med, axis=1)))
    chosen = [first]
    min_d2 = np.sum((flat - flat[first]) ** 2, axis=1)
    for _ in range(max_frames - 1):
        nxt = int(np.argmax(min_d2))
        if min_d2[nxt] <= 1e-15:
            break
        chosen.append(nxt)
        min_d2 = np.minimum(min_d2, np.sum((flat - flat[nxt]) ** 2, axis=1))
    return np.sort(valid[chosen]).astype(np.int64)


def default_scale_groups(
    spec: mujoco.MjSpec, *, bilateral_symmetry: bool = True
) -> tuple[tuple[str, ...], ...]:
    """Infer scalable segment body groups (pairing ``_l``/``_r`` limbs when symmetric)."""
    body_names = [b.name for b in spec.bodies if b.name and b.name != "world"]
    name_set = set(body_names)
    known = (
        ("pelvis",),
        ("sacrum",),
        ("torso",),
        ("clavicle_l", "clavicle_r"),
        ("femur_l", "femur_r"),
        ("tibia_l", "tibia_r"),
        ("talus_l", "talus_r"),
        ("humerus_l", "humerus_r"),
        ("ulna_l", "ulna_r"),
        ("lunate_l", "lunate_r"),
    )
    groups: list[tuple[str, ...]] = []
    covered: set[str] = set()
    for grp in known:
        present = tuple(b for b in grp if b in name_set)
        if present:
            if bilateral_symmetry or len(present) == 1:
                groups.append(present)
            else:
                groups.extend((b,) for b in present)
            covered.update(present)
    if groups:
        return tuple(groups)

    # Generic fallback for arbitrary articulated models (arm, elbow, custom rigs).
    for name in body_names:
        if name in covered:
            continue
        if bilateral_symmetry:
            m = re.match(r"^(.*)_(l|r)$", name)
            if m:
                base = m.group(1)
                pair = tuple(b for b in (f"{base}_l", f"{base}_r") if b in name_set)
                groups.append(pair)
                covered.update(pair)
                continue
        groups.append((name,))
        covered.add(name)
    return tuple(groups)


def _body_scale_map(
    spec: mujoco.MjSpec, scalar_scales: dict[str, float]
) -> dict[str, float]:
    out: dict[str, float] = {}
    roots = [
        b
        for b in spec.worldbody.bodies
        if any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in b.joints)
    ] or list(spec.worldbody.bodies)

    def walk(body: mujoco.MjsBody, s_par: float) -> None:
        s_b = scalar_scales.get(body.name, s_par)
        out[body.name] = s_b
        for child in body.bodies:
            walk(child, s_b)

    for rb in roots:
        walk(rb, scalar_scales.get(rb.name, 1.0))
    return out


def _scaled_base_lmap(
    model: mujoco.MjModel, lmap: LandmarkMap, body_scales: dict[str, float]
) -> LandmarkMap:
    site_names = {model.site(i).name for i in range(model.nsite)}
    lms: list[Landmark] = []
    for lm in lmap.landmarks:
        if lm.site is not None and lm.site in site_names:
            st = model.site(lm.site)
            b_name = model.body(int(st.bodyid[0])).name
            base_off = tuple(float(x) for x in st.pos)
            lms.append(Landmark(lm.source, b_name, None, base_off, lm.weight))
        else:
            s_b = body_scales.get(lm.body, 1.0)
            base_off = tuple(float(s_b * x) for x in lm.pos)
            lms.append(Landmark(lm.source, lm.body, None, base_off, lm.weight))
    return LandmarkMap(
        name=lmap.name,
        version=lmap.version,
        landmarks=tuple(lms),
        orientations=lmap.orientations,
        feet=lmap.feet,
        twist_pairs=lmap.twist_pairs,
        twist_reference=lmap.twist_reference,
        coupling_term=lmap.coupling_term,
        period_joint=lmap.period_joint,
        posture=lmap.posture,
    )


def fit_marker_offsets(
    model: mujoco.MjModel,
    base_lmap: LandmarkMap,
    qpos: np.ndarray,
    target_pos: np.ndarray,
    *,
    offset_reg_weight: float = 2.0,
    fixed_landmarks: Sequence[str] = (),
) -> tuple[np.ndarray, LandmarkMap]:
    """Closed-form Ridge update of body-local marker offsets across *qpos* frames."""
    d = mujoco.MjData(model)
    n_lm = len(base_lmap.landmarks)
    body_ids = np.array([model.body(lm.body).id for lm in base_lmap.landmarks], int)
    base_off = np.array([lm.pos for lm in base_lmap.landmarks], dtype=np.float64)
    weights = np.array([lm.weight for lm in base_lmap.landmarks], dtype=np.float64)
    fixed = set(fixed_landmarks)

    local_sum = np.zeros((n_lm, 3), dtype=np.float64)
    counts = np.zeros(n_lm, dtype=np.float64)
    for t in range(len(qpos)):
        d.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, d)
        rot_t = d.xmat[body_ids].reshape(n_lm, 3, 3)
        diff = target_pos[t] - d.xpos[body_ids]
        ok = np.isfinite(diff).all(axis=1)
        if ok.any():
            rel = np.einsum("lji,lj->li", rot_t[ok], diff[ok]) - base_off[ok]
            local_sum[ok] += rel
            counts[ok] += 1.0

    deltas = np.zeros((n_lm, 3), dtype=np.float64)
    for i, lm in enumerate(base_lmap.landmarks):
        if lm.source in fixed or weights[i] <= 0.0 or counts[i] == 0.0:
            continue
        w2 = weights[i] * weights[i]
        gain = w2 / (counts[i] * w2 + max(offset_reg_weight, 0.0))
        deltas[i] = gain * local_sum[i]

    updated = tuple(
        Landmark(
            lm.source,
            lm.body,
            None,
            tuple(float(x) for x in (base_off[i] + deltas[i])),
            lm.weight,
        )
        for i, lm in enumerate(base_lmap.landmarks)
    )
    return deltas, LandmarkMap(
        name=base_lmap.name,
        version=base_lmap.version,
        landmarks=updated,
        orientations=base_lmap.orientations,
        feet=base_lmap.feet,
        twist_pairs=base_lmap.twist_pairs,
        twist_reference=base_lmap.twist_reference,
        coupling_term=base_lmap.coupling_term,
        period_joint=base_lmap.period_joint,
        posture=base_lmap.posture,
    )


def _fill_nan_targets(pos: np.ndarray) -> np.ndarray:
    if np.isfinite(pos).all():
        return pos
    out = pos.copy()
    t = np.arange(len(pos), dtype=np.float64)
    for lm_i in range(pos.shape[1]):
        ok = np.isfinite(pos[:, lm_i]).all(axis=1)
        if not ok.any():
            raise ValueError(f"Landmark column {lm_i} has no finite frames")
        if not ok.all():
            for c in range(3):
                out[:, lm_i, c] = np.interp(t, t[ok], pos[ok, lm_i, c])
    return out


def _default_seed(
    model: mujoco.MjModel, first_pos: np.ndarray, lmap: LandmarkMap
) -> np.ndarray:
    q = model.qpos0.copy()
    project_joint_couplings(JointCouplings.from_model(model), q)
    free_jnts = [
        j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if free_jnts:
        d = mujoco.MjData(model)
        d.qpos[:] = q
        mujoco.mj_kinematics(model, d)
        b_id = model.body(lmap.landmarks[0].body).id
        adr = int(model.jnt_qposadr[free_jnts[0]])
        q[adr : adr + 3] += first_pos[0] - d.xpos[b_id]
    return q


def fit_subject_scale_and_ik(
    spec: mujoco.MjSpec,
    target_pos: np.ndarray,
    lmap: LandmarkMap,
    *,
    target_quat: np.ndarray | None = None,
    scale_groups: Sequence[Sequence[str]] | None = None,
    body_mass: float | None = None,
    seed_qpos: np.ndarray | None = None,
    config: BilevelScaleConfig | None = None,
) -> BilevelScaleResult:
    """Jointly optimize per-segment scales, marker offsets, and coupled IK trajectories."""
    cfg = config or BilevelScaleConfig()
    pos_raw = np.asarray(target_pos, dtype=np.float64)
    n_lm = len(lmap.landmarks)
    if pos_raw.ndim != 3 or pos_raw.shape[1:] != (n_lm, 3) or len(pos_raw) < 1:
        raise ValueError(
            f"target_pos must have shape (T >= 1, {n_lm}, 3), got {pos_raw.shape}"
        )
    if cfg.scale_bounds[0] <= 0.0 or cfg.scale_bounds[0] >= cfg.scale_bounds[1]:
        raise ValueError(f"Invalid scale_bounds: {cfg.scale_bounds}")

    pos_filled = _fill_nan_targets(pos_raw)
    n_ori = len(lmap.orientations)
    quat_all = (
        np.zeros((len(pos_raw), 0, 4), dtype=np.float64)
        if target_quat is None and n_ori == 0
        else np.asarray(target_quat, dtype=np.float64)
    )
    if quat_all.shape != (len(pos_raw), n_ori, 4):
        raise ValueError(
            f"target_quat must have shape ({len(pos_raw)}, {n_ori}, 4), got {quat_all.shape}"
        )

    groups = (
        tuple(tuple(g) for g in scale_groups)
        if scale_groups is not None
        else default_scale_groups(spec, bilateral_symmetry=cfg.bilateral_symmetry)
    )
    if not groups:
        raise ValueError("No scalable body groups found on spec")

    cal_idx = select_diverse_frames(pos_raw, cfg.max_calibration_frames)
    pos_cal = pos_filled[cal_idx]
    pos_cal_raw = pos_raw[cal_idx]
    quat_cal = quat_all[cal_idx]
    valid_cal = np.isfinite(pos_cal_raw).all(axis=-1)
    lm_w = np.array([lm.weight for lm in lmap.landmarks], dtype=np.float64)

    # Evaluate baseline unscaled coupled-IK RMSE before bilevel optimization.
    m_nom = spec.copy().compile()
    seed0 = (
        _default_seed(m_nom, pos_filled[0], lmap)
        if seed_qpos is None
        else np.asarray(seed_qpos, dtype=np.float64).copy()
    )
    ik_nom = LandmarkIK(m_nom, lmap, cfg.ik_config)
    q_nom, _ = ik_nom.fit(pos_filled, quat_all, seed0)
    err_nom = np.stack([ik_nom.landmarks(q) - p for q, p in zip(q_nom, pos_raw)])
    initial_rmse = float(np.sqrt(np.nanmean(np.sum(err_nom**2, axis=-1))))

    cal_seeds = q_nom[cal_idx].copy()
    cal_ik_cfg = RefitConfig(
        smooth_weight=0.0,
        damping=cfg.ik_config.damping,
        max_iters=cfg.ik_config.max_iters,
        step_tol=cfg.ik_config.step_tol,
    )
    eval_count = 0

    def evaluate_scales(
        theta: np.ndarray,
    ) -> tuple[
        mujoco.MjSpec,
        mujoco.MjModel,
        LandmarkMap,
        np.ndarray,
        np.ndarray,
        dict[str, float],
    ]:
        scale_dict = {b: float(s) for s, grp in zip(theta, groups) for b in grp}
        s_spec = scale_spec_from_instanthmr(spec, scale_dict, body_mass=body_mass)
        s_model = s_spec.compile()
        b_scales = _body_scale_map(spec, scale_dict)
        base_map = _scaled_base_lmap(s_model, lmap, b_scales)
        cur_map = base_map
        deltas = np.zeros((n_lm, 3), dtype=np.float64)
        n_alt = max(1, cfg.inner_alternations if cfg.optimize_marker_offsets else 1)
        q_cal = cal_seeds.copy()
        for _ in range(n_alt):
            ik = LandmarkIK(s_model, cur_map, cal_ik_cfg)
            for k in range(len(cal_idx)):
                q_cal[k], _ = ik.solve(q_cal[k], pos_cal[k], quat_cal[k], None)
            if cfg.optimize_marker_offsets:
                deltas, cur_map = fit_marker_offsets(
                    s_model,
                    base_map,
                    q_cal,
                    pos_cal_raw,
                    offset_reg_weight=cfg.offset_reg_weight,
                    fixed_landmarks=cfg.fixed_landmarks,
                )
        return s_spec, s_model, cur_map, deltas, q_cal, scale_dict

    def outer_residual(theta: np.ndarray) -> np.ndarray:
        nonlocal eval_count
        eval_count += 1
        _, s_model, cur_map, deltas, q_cal, _ = evaluate_scales(theta)
        ik = LandmarkIK(s_model, cur_map, cfg.ik_config)
        pred = np.stack([ik.landmarks(q) for q in q_cal])
        diff = np.where(
            valid_cal[..., None], (pred - pos_cal) * lm_w[None, :, None], 0.0
        )
        parts = [diff.ravel()]
        if cfg.optimize_marker_offsets and cfg.offset_reg_weight > 0.0:
            parts.append(np.sqrt(cfg.offset_reg_weight) * deltas.ravel())
        if cfg.scale_reg_weight > 0.0:
            parts.append(np.sqrt(cfg.scale_reg_weight) * (theta - 1.0))
        return np.concatenate(parts)

    x0 = np.ones(len(groups), dtype=np.float64)
    opt = least_squares(
        outer_residual,
        x0,
        bounds=(cfg.scale_bounds[0], cfg.scale_bounds[1]),
        method="trf",
        diff_step=1e-3,
        ftol=1e-8,
        xtol=1e-6,
        max_nfev=cfg.outer_max_nfev,
    )

    s_spec, s_model, final_map, deltas, _, scale_dict = evaluate_scales(opt.x)
    # Write calibrated marker offsets back to named sites on scaled_spec when applicable.
    if cfg.optimize_marker_offsets:
        orig_sites = {
            lm.source: lm.site for lm in lmap.landmarks if lm.site is not None
        }
        spec_sites = {st.name: st for b in s_spec.bodies for st in b.sites}
        for i, lm_final in enumerate(final_map.landmarks):
            st_name = orig_sites.get(lm_final.source)
            if st_name is not None and st_name in spec_sites:
                spec_sites[st_name].pos = np.asarray(lm_final.pos, dtype=np.float64)
        s_model = s_spec.compile()

    ik_final = LandmarkIK(s_model, final_map, cfg.ik_config)
    final_seed = (
        _default_seed(s_model, pos_filled[0], final_map)
        if seed_qpos is None
        else seed0.copy()
    )
    qpos_full, _ = ik_final.fit(pos_filled, quat_all, final_seed)
    err_final = np.stack(
        [ik_final.landmarks(q) - p for q, p in zip(qpos_full, pos_raw)]
    )
    final_rmse = float(np.sqrt(np.nanmean(np.sum(err_final**2, axis=-1))))

    offset_dict = {
        lm.source: deltas[i].copy() for i, lm in enumerate(final_map.landmarks)
    }
    return BilevelScaleResult(
        scales=scale_dict,
        scaled_spec=s_spec,
        scaled_model=s_model,
        landmark_map=final_map,
        marker_offsets=offset_dict,
        qpos=qpos_full,
        initial_rmse_m=initial_rmse,
        final_rmse_m=final_rmse,
        calibration_frame_indices=cal_idx,
        outer_evaluations=eval_count,
    )


__all__ = [
    "BilevelScaleConfig",
    "BilevelScaleResult",
    "default_scale_groups",
    "fit_marker_offsets",
    "fit_subject_scale_and_ik",
    "select_diverse_frames",
]
