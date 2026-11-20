# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Eye-head coordination and cervical/upper-limb ergonomic assessment metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# McAtamney & Corlett (1993) RULA Lookup Table A: shape (6, 3, 4, 2)
# Axes: (upper_arm 1..6, lower_arm 1..3, wrist 1..4, wrist_twist 1..2)
_RULA_TABLE_A = np.array(
    [
        [
            [[1, 2], [2, 2], [2, 3], [3, 3]],
            [[2, 2], [2, 2], [3, 3], [3, 3]],
            [[2, 3], [3, 3], [3, 3], [4, 4]],
        ],
        [
            [[2, 3], [3, 3], [3, 4], [4, 4]],
            [[3, 3], [3, 3], [3, 4], [4, 4]],
            [[3, 4], [4, 4], [4, 4], [5, 5]],
        ],
        [
            [[3, 3], [4, 4], [4, 4], [5, 5]],
            [[3, 4], [4, 4], [4, 4], [5, 5]],
            [[4, 4], [4, 4], [4, 5], [5, 5]],
        ],
        [
            [[4, 4], [4, 4], [4, 5], [5, 5]],
            [[4, 4], [4, 4], [4, 5], [5, 5]],
            [[4, 4], [4, 5], [5, 5], [6, 6]],
        ],
        [
            [[5, 5], [5, 5], [5, 6], [6, 7]],
            [[5, 6], [6, 6], [6, 7], [7, 7]],
            [[6, 6], [6, 7], [7, 7], [7, 8]],
        ],
        [
            [[7, 7], [7, 7], [7, 8], [8, 9]],
            [[8, 8], [8, 8], [8, 9], [9, 9]],
            [[9, 9], [9, 9], [9, 9], [9, 9]],
        ],
    ],
    dtype=np.int32,
)

# RULA Lookup Table B: shape (6, 6, 2) -> (neck 1..6, trunk 1..6, legs 1..2)
_RULA_TABLE_B = np.array(
    [
        [[1, 3], [2, 3], [3, 4], [5, 5], [6, 6], [7, 7]],
        [[2, 3], [2, 3], [4, 5], [5, 5], [6, 7], [7, 7]],
        [[3, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 7]],
        [[5, 5], [5, 6], [6, 7], [7, 7], [7, 7], [8, 8]],
        [[7, 7], [7, 7], [7, 8], [8, 8], [8, 8], [8, 8]],
        [[8, 8], [8, 8], [8, 8], [8, 9], [9, 9], [9, 9]],
    ],
    dtype=np.int32,
)

# RULA Grand Score Table C: shape (8, 7) -> (score_c 1..8+, score_d 1..7+)
_RULA_TABLE_C = np.array(
    [
        [1, 2, 3, 3, 4, 5, 5],
        [2, 2, 3, 4, 4, 5, 5],
        [3, 3, 3, 4, 4, 5, 6],
        [3, 3, 3, 4, 5, 6, 6],
        [4, 4, 4, 5, 6, 7, 7],
        [4, 4, 5, 6, 6, 7, 7],
        [5, 5, 6, 6, 7, 7, 7],
        [5, 5, 6, 7, 7, 7, 7],
    ],
    dtype=np.int32,
)


def _validate_dt(dt: float) -> float:
    dt_f = float(dt)
    if not (math.isfinite(dt_f) and dt_f > 0.0):
        raise ValueError(f"dt must be finite and positive, got {dt}")
    return dt_f


def _net_angular_excursion_deg(traj: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(traj, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size < 2:
            return 0.0, 1.0
        delta = float(arr[-1] - arr[0])
        return abs(delta), (1.0 if delta >= 0.0 else -1.0)
    if arr.ndim == 2 and arr.shape[1] == 3:
        if arr.shape[0] < 2:
            return 0.0, 1.0
        v0 = arr[0] / max(float(np.linalg.norm(arr[0])), 1e-12)
        v1 = arr[-1] / max(float(np.linalg.norm(arr[-1])), 1e-12)
        cos_ang = float(np.clip(np.dot(v0, v1), -1.0, 1.0))
        return math.degrees(math.acos(cos_ang)), 1.0
    raise ValueError(
        f"Expected 1D angle array (T,) or 3D direction vectors (T, 3), got {arr.shape}"
    )


def head_contribution_ratio(
    gaze_trajectory: np.ndarray,
    head_trajectory: np.ndarray,
) -> float:
    """Compute head-movement contribution fraction ``Delta H / Delta G`` during a gaze shift.

    Accepts either 1D angle traces ``(T,)`` [deg] or 3D unit direction vectors ``(T, 3)``
    (Freedman, 2008; Vercher et al., 1994).

    Args:
        gaze_trajectory: World-space gaze angles ``(T,)`` [deg] or unit vectors ``(T, 3)``.
        head_trajectory: World-space head angles ``(T,)`` [deg] or unit vectors ``(T, 3)``.

    Returns:
        Head contribution ratio (``0.0`` when gaze excursion is negligible).
    """
    dg, sg = _net_angular_excursion_deg(gaze_trajectory)
    dh, sh = _net_angular_excursion_deg(head_trajectory)
    if dg <= 1e-6:
        return 0.0
    if np.asarray(gaze_trajectory).ndim == 1 and np.asarray(head_trajectory).ndim == 1:
        return float((sh * dh) / (sg * dg))
    return float(dh / dg)


def eye_head_onset_latency(
    gaze_speed_deg_s: np.ndarray,
    head_speed_deg_s: np.ndarray,
    dt: float,
    *,
    threshold_ratio: float = 0.10,
) -> float:
    """Compute signed head-versus-gaze movement onset latency ``t_head - t_gaze`` [s].

    Positive values indicate the eye saccade initiates before head rotation; negative
    values indicate anticipatory head movement precedes gaze shift onset.

    Args:
        gaze_speed_deg_s: Gaze angular speed profile ``(T,)`` [deg/s].
        head_speed_deg_s: Head angular speed profile ``(T,)`` [deg/s].
        dt: Sampling interval [s].
        threshold_ratio: Fraction of peak speed defining movement onset in ``(0, 1)``.

    Returns:
        Signed onset latency ``t_head_onset - t_gaze_onset`` in seconds.
    """
    dt_f = _validate_dt(dt)
    if not (0.0 < float(threshold_ratio) < 1.0):
        raise ValueError("threshold_ratio must lie in (0, 1)")
    vg = np.abs(np.asarray(gaze_speed_deg_s, dtype=np.float64).ravel())
    vh = np.abs(np.asarray(head_speed_deg_s, dtype=np.float64).ravel())
    if vg.shape != vh.shape or vg.size < 2:
        raise ValueError(
            "gaze_speed_deg_s and head_speed_deg_s must share shape (T>=2,)"
        )

    pg, ph = float(np.max(vg)), float(np.max(vh))
    if pg <= 1e-8 or ph <= 1e-8:
        return 0.0

    idx_g = np.flatnonzero(vg >= float(threshold_ratio) * pg)
    idx_h = np.flatnonzero(vh >= float(threshold_ratio) * ph)
    if idx_g.size == 0 or idx_h.size == 0:
        return 0.0
    return float(int(idx_h[0]) - int(idx_g[0])) * dt_f


def eye_head_coupling_lag(
    gaze_speed_deg_s: np.ndarray,
    head_speed_deg_s: np.ndarray,
    dt: float,
    *,
    max_lag_s: float = 0.5,
) -> tuple[float, float]:
    """Compute peak normalized cross-correlation lag ``(lag_s, peak_corr)``.

    A positive ``lag_s`` means ``head_speed_deg_s`` lags behind ``gaze_speed_deg_s``.

    Args:
        gaze_speed_deg_s: Gaze speed profile ``(T,)``.
        head_speed_deg_s: Head speed profile ``(T,)``.
        dt: Sampling interval [s].
        max_lag_s: Maximum lag search window [s] (positive).

    Returns:
        Tuple ``(lag_s, peak_correlation)`` with ``peak_correlation`` in ``[-1, 1]``.
    """
    dt_f = _validate_dt(dt)
    if max_lag_s <= 0.0 or not math.isfinite(max_lag_s):
        raise ValueError("max_lag_s must be finite and positive")
    vg = np.asarray(gaze_speed_deg_s, dtype=np.float64).ravel()
    vh = np.asarray(head_speed_deg_s, dtype=np.float64).ravel()
    if vg.shape != vh.shape or vg.size < 3:
        raise ValueError("Speed profiles must share 1D shape with T >= 3")

    g0 = vg - np.mean(vg)
    h0 = vh - np.mean(vh)
    denom = float(np.linalg.norm(g0) * np.linalg.norm(h0))
    if denom <= 1e-12:
        return 0.0, 0.0

    # np.correlate(h0, g0, "full") peaks at index (N - 1 + k) when h0[t + k] aligns with g0[t]
    corr = np.correlate(h0, g0, mode="full") / denom
    n = vg.size
    lags = np.arange(-(n - 1), n, dtype=np.int64)
    max_steps = max(1, int(round(float(max_lag_s) / dt_f)))
    mask = np.abs(lags) <= max_steps
    sub_corr = corr[mask]
    sub_lags = lags[mask]
    best = int(np.argmax(sub_corr))
    return float(sub_lags[best]) * dt_f, float(np.clip(sub_corr[best], -1.0, 1.0))


def vor_gain(
    eye_in_head_vel: np.ndarray,
    head_vel: np.ndarray,
    *,
    min_head_speed: float = 5.0,
) -> float:
    """Compute compensatory vestibulo-ocular reflex gain ``-Cov(e_dot, h_dot) / Var(h_dot)``.

    Ideal gaze stabilization during head motion satisfies ``eye_in_head_vel == -head_vel``,
    yielding a VOR gain of ``1.0``.

    Args:
        eye_in_head_vel: Eye-in-head angular velocity ``(T,)`` or ``(T, D)`` [deg/s or rad/s].
        head_vel: Head-in-world angular velocity ``(T,)`` or ``(T, D)`` in matching units.
        min_head_speed: Speed threshold filtering quiescent samples.

    Returns:
        Estimated scalar VOR gain.
    """
    e = np.asarray(eye_in_head_vel, dtype=np.float64)
    h = np.asarray(head_vel, dtype=np.float64)
    if e.shape != h.shape or e.size == 0:
        raise ValueError(
            "eye_in_head_vel and head_vel must be non-empty with equal shape"
        )
    if e.ndim == 1:
        active = np.abs(h) >= float(min_head_speed)
        if not np.any(active):
            return 0.0
        ea, ha = e[active], h[active]
        denom = float(np.sum(ha * ha))
        return 0.0 if denom <= 1e-12 else float(-np.sum(ea * ha) / denom)

    speed = np.linalg.norm(h, axis=-1)
    active = speed >= float(min_head_speed)
    if not np.any(active):
        return 0.0
    ea, ha = e[active], h[active]
    denom = float(np.sum(ha * ha))
    return 0.0 if denom <= 1e-12 else float(-np.sum(ea * ha) / denom)


def craniovertebral_angle_deg(
    c7_pos: np.ndarray | Sequence[float],
    tragus_pos: np.ndarray | Sequence[float],
    *,
    up_axis: Sequence[float] = (0.0, 0.0, 1.0),
) -> np.ndarray | float:
    """Compute craniovertebral angle (CVA) [deg] between the C7-tragus line and horizontal.

    Lower angles (typically ``< 48 deg``) indicate increased forward-head posture.

    Args:
        c7_pos: C7 spinous process 3D position ``(3,)`` or ``(T, 3)`` [m].
        tragus_pos: Ear tragus 3D position ``(3,)`` or ``(T, 3)`` [m].
        up_axis: World vertical axis ``(3,)``.

    Returns:
        Scalar float when inputs are ``(3,)``, or ``(T,)`` ndarray for trajectories.
    """
    c7 = np.asarray(c7_pos, dtype=np.float64)
    tr = np.asarray(tragus_pos, dtype=np.float64)
    if c7.shape != tr.shape or c7.shape[-1] != 3 or c7.ndim not in (1, 2):
        raise ValueError(
            "c7_pos and tragus_pos must have matching shape (3,) or (T, 3)"
        )
    up = np.asarray(up_axis, dtype=np.float64)
    up_norm = float(np.linalg.norm(up))
    if up.shape != (3,) or up_norm <= 1e-12:
        raise ValueError("up_axis must be a non-zero 3-vector")
    up_u = up / up_norm

    vec = tr - c7
    vert = np.sum(vec * up_u, axis=-1)
    horiz_vec = vec - vert[..., None] * up_u
    horiz = np.linalg.norm(horiz_vec, axis=-1)
    angles = np.degrees(np.arctan2(vert, np.maximum(horiz, 1e-12)))
    return float(angles) if c7.ndim == 1 else angles


def cervical_flexion_moment_nm(
    head_com_pos: np.ndarray | Sequence[float],
    c7_pos: np.ndarray | Sequence[float],
    *,
    head_mass_kg: float = 4.5,
    gravity: Sequence[float] = (0.0, 0.0, -9.81),
    base_lin_acc: np.ndarray | Sequence[float] | None = None,
) -> np.ndarray | float:
    """Compute gravitational and optional non-inertial cervical moment magnitude [N*m].

    Evaluates ``|| (r_head - r_c7) x (m_head * (g - a_base)) ||`` about the C7/T1
    cervical pivot (Hansraj, 2014).

    Args:
        head_com_pos: Head center-of-mass position ``(3,)`` or ``(T, 3)`` [m].
        c7_pos: C7/T1 pivot position ``(3,)`` or ``(T, 3)`` [m].
        head_mass_kg: Head-and-neck segment mass [kg] (positive).
        gravity: World gravity vector ``(3,)`` [m/s^2].
        base_lin_acc: Optional support-frame linear acceleration ``(3,)`` or ``(T, 3)`` [m/s^2].

    Returns:
        Scalar moment magnitude [N*m] for ``(3,)`` inputs, or ``(T,)`` ndarray.
    """
    if not (math.isfinite(head_mass_kg) and head_mass_kg > 0.0):
        raise ValueError("head_mass_kg must be finite and positive")
    head = np.asarray(head_com_pos, dtype=np.float64)
    c7 = np.asarray(c7_pos, dtype=np.float64)
    if head.shape != c7.shape or head.shape[-1] != 3 or head.ndim not in (1, 2):
        raise ValueError("head_com_pos and c7_pos must share shape (3,) or (T, 3)")
    g_vec = np.asarray(gravity, dtype=np.float64)
    if g_vec.shape != (3,):
        raise ValueError("gravity must be a 3-vector")

    eff_acc = g_vec
    if base_lin_acc is not None:
        a_base = np.asarray(base_lin_acc, dtype=np.float64)
        eff_acc = g_vec - a_base

    lever = head - c7
    force = float(head_mass_kg) * eff_acc
    torque = np.cross(lever, force)
    mag = np.linalg.norm(torque, axis=-1)
    return float(mag) if head.ndim == 1 else mag


@dataclass(frozen=True)
class RULAResult:
    """Time-resolved and summary Rapid Upper Limb Assessment (RULA) scores."""

    grand_scores: np.ndarray
    wrist_arm_scores: np.ndarray
    neck_trunk_scores: np.ndarray
    mean_grand_score: float
    p95_grand_score: float
    action_level: int


def _broadcast_1d(
    value: float | bool | np.ndarray | Sequence[float | bool], length: int, dtype: Any
) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        return np.full(length, arr.item(), dtype=dtype)
    arr1 = arr.ravel()
    if arr1.size != length:
        raise ValueError(
            f"Input length {arr1.size} does not match trajectory T={length}"
        )
    return arr1


def upper_limb_rula_score(
    upper_arm_flex_deg: float | np.ndarray | Sequence[float],
    lower_arm_flex_deg: float | np.ndarray | Sequence[float],
    wrist_flex_deg: float | np.ndarray | Sequence[float],
    neck_flex_deg: float | np.ndarray | Sequence[float],
    trunk_flex_deg: float | np.ndarray | Sequence[float],
    *,
    shoulder_raised: bool | np.ndarray | Sequence[bool] = False,
    upper_arm_abducted: bool | np.ndarray | Sequence[bool] = False,
    arm_supported: bool | np.ndarray | Sequence[bool] = False,
    midline_or_out_to_side: bool | np.ndarray | Sequence[bool] = False,
    wrist_deviated: bool | np.ndarray | Sequence[bool] = False,
    wrist_twisted_end_range: bool | np.ndarray | Sequence[bool] = False,
    neck_twisted: bool | np.ndarray | Sequence[bool] = False,
    neck_side_bent: bool | np.ndarray | Sequence[bool] = False,
    trunk_twisted: bool | np.ndarray | Sequence[bool] = False,
    trunk_side_bent: bool | np.ndarray | Sequence[bool] = False,
    legs_supported: bool | np.ndarray | Sequence[bool] = True,
    static_posture: bool = False,
    force_load_score: int = 0,
) -> RULAResult:
    """Compute McAtamney & Corlett (1993) RULA postural risk scores ``[1..7]``.

    Args:
        upper_arm_flex_deg: Sagittal shoulder flexion ``[-20..180]`` [deg] (negative = extension).
        lower_arm_flex_deg: Elbow flexion ``[0..180]`` [deg] (optimal ``[60, 100]``).
        wrist_flex_deg: Wrist flexion/extension excursion from neutral [deg].
        neck_flex_deg: Cervical flexion [deg] (negative ``< -2`` = neck extension).
        trunk_flex_deg: Trunk forward flexion [deg].
        shoulder_raised: ``+1`` when shoulder is elevated.
        upper_arm_abducted: ``+1`` when upper arm is abducted.
        arm_supported: ``-1`` when arm is supported or person leans.
        midline_or_out_to_side: ``+1`` when forearm crosses midline or works out to side.
        wrist_deviated: ``+1`` for radial/ulnar wrist deviation.
        wrist_twisted_end_range: ``True`` for end-range pronation/supination (twist score 2).
        neck_twisted: ``+1`` for axial neck rotation.
        neck_side_bent: ``+1`` for lateral neck bending.
        trunk_twisted: ``+1`` for axial trunk twist.
        trunk_side_bent: ``+1`` for lateral trunk bending.
        legs_supported: ``True`` when seated/standing evenly balanced (leg score 1, else 2).
        static_posture: ``+1`` muscle-use score when posture is held statically (> 1 min).
        force_load_score: External load/force score in ``{0, 1, 2, 3}``.

    Returns:
        :class:`RULAResult` with frame-level arrays and summary Grand Score / Action Level.
    """
    if force_load_score not in (0, 1, 2, 3):
        raise ValueError("force_load_score must be in {0, 1, 2, 3}")

    ua_raw = np.asarray(upper_arm_flex_deg, dtype=np.float64)
    n = 1 if ua_raw.ndim == 0 else ua_raw.ravel().size
    if n < 1:
        raise ValueError("Trajectories must be non-empty")

    ua = _broadcast_1d(upper_arm_flex_deg, n, np.float64)
    la = _broadcast_1d(lower_arm_flex_deg, n, np.float64)
    wr = np.abs(_broadcast_1d(wrist_flex_deg, n, np.float64))
    nk = _broadcast_1d(neck_flex_deg, n, np.float64)
    tr = _broadcast_1d(trunk_flex_deg, n, np.float64)

    # Step 1: Upper arm score (1..6)
    ua_score = np.ones(n, dtype=np.int32)
    ua_score = np.where((ua < -20.0) | ((ua > 20.0) & (ua <= 45.0)), 2, ua_score)
    ua_score = np.where((ua > 45.0) & (ua <= 90.0), 3, ua_score)
    ua_score = np.where(ua > 90.0, 4, ua_score)
    ua_score += _broadcast_1d(shoulder_raised, n, bool).astype(np.int32)
    ua_score += _broadcast_1d(upper_arm_abducted, n, bool).astype(np.int32)
    ua_score -= _broadcast_1d(arm_supported, n, bool).astype(np.int32)
    ua_score = np.clip(ua_score, 1, 6)

    # Step 2: Lower arm score (1..3)
    la_score = np.where((la >= 60.0) & (la <= 100.0), 1, 2).astype(np.int32)
    la_score += _broadcast_1d(midline_or_out_to_side, n, bool).astype(np.int32)
    la_score = np.clip(la_score, 1, 3)

    # Step 3 & 4: Wrist (1..4) & Wrist twist (1..2)
    wr_score = np.ones(n, dtype=np.int32)
    wr_score = np.where((wr > 1.0) & (wr <= 15.0), 2, wr_score)
    wr_score = np.where(wr > 15.0, 3, wr_score)
    wr_score += _broadcast_1d(wrist_deviated, n, bool).astype(np.int32)
    wr_score = np.clip(wr_score, 1, 4)

    wt_score = np.where(_broadcast_1d(wrist_twisted_end_range, n, bool), 2, 1).astype(
        np.int32
    )

    score_a = _RULA_TABLE_A[ua_score - 1, la_score - 1, wr_score - 1, wt_score - 1]
    muscle_add = 1 if static_posture else 0
    wrist_arm = score_a + muscle_add + int(force_load_score)

    # Step 9: Neck score (1..6)
    nk_score = np.ones(n, dtype=np.int32)
    nk_score = np.where((nk > 10.0) & (nk <= 20.0), 2, nk_score)
    nk_score = np.where(nk > 20.0, 3, nk_score)
    nk_score = np.where(nk < -2.0, 4, nk_score)
    nk_score += _broadcast_1d(neck_twisted, n, bool).astype(np.int32)
    nk_score += _broadcast_1d(neck_side_bent, n, bool).astype(np.int32)
    nk_score = np.clip(nk_score, 1, 6)

    # Step 10 & 11: Trunk (1..6) & Legs (1..2)
    tr_score = np.ones(n, dtype=np.int32)
    tr_score = np.where((tr > 2.0) & (tr <= 20.0), 2, tr_score)
    tr_score = np.where((tr > 20.0) & (tr <= 60.0), 3, tr_score)
    tr_score = np.where(tr > 60.0, 4, tr_score)
    tr_score += _broadcast_1d(trunk_twisted, n, bool).astype(np.int32)
    tr_score += _broadcast_1d(trunk_side_bent, n, bool).astype(np.int32)
    tr_score = np.clip(tr_score, 1, 6)

    leg_score = np.where(_broadcast_1d(legs_supported, n, bool), 1, 2).astype(np.int32)
    score_b = _RULA_TABLE_B[nk_score - 1, tr_score - 1, leg_score - 1]
    neck_trunk = score_b + muscle_add + int(force_load_score)

    c_idx = np.clip(wrist_arm, 1, 8) - 1
    d_idx = np.clip(neck_trunk, 1, 7) - 1
    grand = _RULA_TABLE_C[c_idx, d_idx]

    mean_g = float(np.mean(grand))
    p95_g = float(np.percentile(grand, 95))
    peak_round = int(round(p95_g))
    if peak_round <= 2:
        action = 1
    elif peak_round <= 4:
        action = 2
    elif peak_round <= 6:
        action = 3
    else:
        action = 4

    return RULAResult(
        grand_scores=grand,
        wrist_arm_scores=wrist_arm,
        neck_trunk_scores=neck_trunk,
        mean_grand_score=mean_g,
        p95_grand_score=p95_g,
        action_level=action,
    )


__all__ = [
    "RULAResult",
    "cervical_flexion_moment_nm",
    "craniovertebral_angle_deg",
    "eye_head_coupling_lag",
    "eye_head_onset_latency",
    "head_contribution_ratio",
    "upper_limb_rula_score",
    "vor_gain",
]
