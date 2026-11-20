# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Force-plate-free gait event detection from 3D foot and pelvis landmarks.

Implements a cubic smoothing B-spline sagittal foot-excursion detector inspired by
O-GEST (Hatamzadeh et al., *Journal of Biomechanics*, 2025; ``flodelaplace/O-GEST``,
MIT), generalized across overground and treadmill walking as well as heel-strike and
equinus/forefoot-strike gait patterns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.signal import find_peaks


@dataclass(frozen=True)
class GaitEventConfig:
    """Parameters for B-spline force-plate-free gait event detection.

    Attributes:
        smooth_lam: Optional smoothing parameter ``lam`` for cubic B-spline fitting;
            when ``None``, scales automatically with ``fps`` as ``1e-4 / fps``.
        min_stride_s: Minimum time (s) between consecutive same-side strikes or offs.
        min_prominence_m: Minimum sagittal excursion prominence (m) required for a gait
            cycle peak/trough (suppresses static standing jitter).
        contact_height_m: Fallback sole clearance threshold (m) used when a clip has no
            walking strides (static standing or aerial jump).
        contact_speed_mps: Fallback horizontal speed threshold (m/s) when no strides
            are detected.
    """

    smooth_lam: float | None = None
    min_stride_s: float = 0.45
    min_prominence_m: float = 0.04
    contact_height_m: float = 0.035
    contact_speed_mps: float = 0.50


@dataclass(frozen=True)
class GaitEventsResult:
    """Detected bilateral foot-strike and foot-off events and stance contact mask.

    Attributes:
        left_strikes: Integer frame indices of left foot-strike events.
        right_strikes: Integer frame indices of right foot-strike events.
        left_offs: Integer frame indices of left foot-off events.
        right_offs: Integer frame indices of right foot-off events.
        left_strike_times: Sub-frame refined left foot-strike timestamps (s).
        right_strike_times: Sub-frame refined right foot-strike timestamps (s).
        left_off_times: Sub-frame refined left foot-off timestamps (s).
        right_off_times: Sub-frame refined right foot-off timestamps (s).
        stance_mask: ``(T, 2)`` boolean ``[left, right]`` stance contact mask.
    """

    left_strikes: np.ndarray
    right_strikes: np.ndarray
    left_offs: np.ndarray
    right_offs: np.ndarray
    left_strike_times: np.ndarray
    right_strike_times: np.ndarray
    left_off_times: np.ndarray
    right_off_times: np.ndarray
    stance_mask: np.ndarray


def _progression_axes(
    pelvis: np.ndarray,
    left_heel: np.ndarray,
    left_toe: np.ndarray,
    right_heel: np.ndarray,
    right_toe: np.ndarray,
    time: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Per-frame horizontal unit progression vector ``(T, 3)`` (overground or treadmill)."""
    n = len(pelvis)
    disp_xy = pelvis[-1, :2] - pelvis[0, :2]
    path_span = float(
        np.linalg.norm(pelvis[:, :2].max(axis=0) - pelvis[:, :2].min(axis=0))
    )
    if path_span >= 0.25 and n >= 6:
        vx = make_smoothing_spline(time, pelvis[:, 0], lam=lam * 10.0)(time, nu=1)
        vy = make_smoothing_spline(time, pelvis[:, 1], lam=lam * 10.0)(time, nu=1)
        vel = np.stack([vx, vy, np.zeros(n)], axis=-1)
        norms = np.linalg.norm(vel[:, :2], axis=-1, keepdims=True)
        fallback = np.array([disp_xy[0], disp_xy[1], 0.0], dtype=np.float64)
        fn = float(np.linalg.norm(fallback[:2]))
        fallback = fallback / fn if fn > 1e-8 else np.array([1.0, 0.0, 0.0])
        return np.where(norms > 0.08, vel / np.maximum(norms, 1e-9), fallback[None, :])

    foot_vec = 0.5 * ((left_toe - left_heel) + (right_toe - right_heel))
    med_xy = np.nanmedian(foot_vec[:, :2], axis=0)
    norm = float(np.linalg.norm(med_xy))
    u = (
        np.array([med_xy[0] / norm, med_xy[1] / norm, 0.0], dtype=np.float64)
        if norm > 1e-8
        else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    )
    return np.broadcast_to(u[None, :], (n, 3)).copy()


def _subframe_extremum_time(
    spline: object, time: np.ndarray, idx: np.ndarray, dt: float
) -> np.ndarray:
    """Refine integer peak/trough indices using quadratic Newton step on the B-spline."""
    if len(idx) == 0:
        return np.zeros(0, dtype=np.float64)
    t0 = time[idx]
    d1 = spline(t0, nu=1)  # type: ignore[operator]
    d2 = spline(t0, nu=2)  # type: ignore[operator]
    step = np.where(np.abs(d2) > 1e-9, -d1 / np.where(np.abs(d2) > 1e-9, d2, 1.0), 0.0)
    step = np.clip(step, -dt, dt)
    return np.clip(t0 + step, float(time[0]), float(time[-1]))


def contacts_from_gait_events(
    n_frames: int,
    strikes: np.ndarray,
    offs: np.ndarray,
    *,
    initial_excursion_vel: float = 0.0,
) -> np.ndarray:
    """Construct a ``(T,)`` boolean stance mask from single-limb strike and off indices."""
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    s_idx = np.sort(np.asarray(strikes, dtype=np.int64))
    o_idx = np.sort(np.asarray(offs, dtype=np.int64))
    events: list[tuple[int, int]] = [
        *((int(i), 1) for i in s_idx),
        *((int(i), 0) for i in o_idx),
    ]
    if not events:
        return np.zeros(n_frames, dtype=bool)
    events.sort(key=lambda item: (item[0], -item[1]))

    first_frame, first_kind = events[0]
    if first_frame == 0:
        in_stance = first_kind == 1
    elif first_kind == 0:
        # First event is foot-off -> limb started in stance.
        in_stance = True
    elif first_kind == 1:
        # First event is foot-strike -> limb was swinging forward into the strike.
        in_stance = False
    else:
        in_stance = initial_excursion_vel <= 0.0

    mask = np.zeros(n_frames, dtype=bool)
    cursor = 0
    for frame, kind in events:
        frame_c = int(np.clip(frame, 0, n_frames - 1))
        if frame_c > cursor:
            mask[cursor:frame_c] = in_stance
        in_stance = kind == 1
        mask[frame_c] = in_stance
        cursor = frame_c + 1
    if cursor < n_frames:
        mask[cursor:] = in_stance
    return mask


def _detect_single_limb(
    heel: np.ndarray,
    toe: np.ndarray,
    pelvis: np.ndarray,
    prog: np.ndarray,
    time: np.ndarray,
    fps: float,
    cfg: GaitEventConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    n = len(time)
    dt = 1.0 / fps
    if n < 6:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        return empty_i, empty_i, empty_f, empty_f, None

    lam = (1e-4 / fps) if cfg.smooth_lam is None else float(cfg.smooth_lam)
    s_heel = np.sum((heel - pelvis) * prog, axis=-1)
    s_toe = np.sum((toe - pelvis) * prog, axis=-1)
    # Centering heel and toe before taking their anterior envelope detects both
    # healthy heel-strike and clinical equinus/forefoot-strike transitions.
    s_ant_raw = np.maximum(s_heel - np.median(s_heel), s_toe - np.median(s_toe))
    sp_ant = make_smoothing_spline(time, s_ant_raw, lam=lam)
    sp_toe = make_smoothing_spline(time, s_toe, lam=lam)

    s_ant = sp_ant(time)
    s_toe_s = sp_toe(time)
    min_dist = max(2, int(np.round(cfg.min_stride_s * fps)))

    strikes, _ = find_peaks(s_ant, distance=min_dist, prominence=cfg.min_prominence_m)
    offs, _ = find_peaks(-s_toe_s, distance=min_dist, prominence=cfg.min_prominence_m)
    strikes = strikes.astype(np.int64)
    offs = offs.astype(np.int64)
    if len(strikes) == 0 and len(offs) == 0:
        return strikes, offs, np.zeros(0), np.zeros(0), None

    strike_t = _subframe_extremum_time(sp_ant, time, strikes, dt)
    off_t = _subframe_extremum_time(sp_toe, time, offs, dt)
    v0 = float(sp_ant(time[0], nu=1))
    stance = contacts_from_gait_events(n, strikes, offs, initial_excursion_vel=v0)
    return strikes, offs, strike_t, off_t, stance


def detect_gait_events(
    pelvis: np.ndarray,
    left_heel: np.ndarray,
    left_toe: np.ndarray,
    right_heel: np.ndarray,
    right_toe: np.ndarray,
    fps: float,
    config: GaitEventConfig | None = None,
) -> GaitEventsResult:
    """Detect bilateral foot-strike and foot-off events from 3D landmarks (O-GEST).

    Args:
        pelvis: ``(T, 3)`` pelvis / sacrum midpoint trajectory (m).
        left_heel: ``(T, 3)`` left heel landmark trajectory (m).
        left_toe: ``(T, 3)`` left forefoot / toe landmark trajectory (m).
        right_heel: ``(T, 3)`` right heel landmark trajectory (m).
        right_toe: ``(T, 3)`` right forefoot / toe landmark trajectory (m).
        fps: Sampling frequency (Hz).
        config: Optional :class:`GaitEventConfig`.

    Returns:
        :class:`GaitEventsResult` with integer event frames, sub-frame event times,
        and a ``(T, 2)`` stance contact mask.
    """
    cfg = config or GaitEventConfig()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be finite and positive, got {fps}")
    arrays = [
        np.asarray(a, dtype=np.float64)
        for a in (pelvis, left_heel, left_toe, right_heel, right_toe)
    ]
    n = len(arrays[0])
    if n < 2 or any(a.shape != (n, 3) for a in arrays):
        raise ValueError("All landmark inputs must have matching shape (T >= 2, 3)")
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("Landmark inputs to detect_gait_events must be finite")
    pel, lh, lt, rh, rt = arrays
    time = np.arange(n, dtype=np.float64) / fps
    lam = (1e-4 / fps) if cfg.smooth_lam is None else float(cfg.smooth_lam)
    prog = _progression_axes(pel, lh, lt, rh, rt, time, lam)

    ls, lo, lst, lot, l_mask = _detect_single_limb(lh, lt, pel, prog, time, fps, cfg)
    rs, ro, rst, rot, r_mask = _detect_single_limb(rh, rt, pel, prog, time, fps, cfg)

    # Fallback per limb when no cyclic gait events occur (e.g. quiet standing or jump).
    if l_mask is None or r_mask is None:
        sole = np.stack([lh, lt, rh, rt], axis=1)
        floor = np.percentile(sole[..., 2], 2, axis=0)
        h = sole[..., 2] - floor
        v = np.linalg.norm(np.gradient(sole[..., :2], 1.0 / fps, axis=0), axis=-1)
        near = (h < cfg.contact_height_m) & (v < cfg.contact_speed_mps)
        if l_mask is None:
            l_mask = near[:, :2].any(axis=1)
        if r_mask is None:
            r_mask = near[:, 2:].any(axis=1)

    stance = np.stack([l_mask, r_mask], axis=1)
    return GaitEventsResult(
        left_strikes=ls,
        right_strikes=rs,
        left_offs=lo,
        right_offs=ro,
        left_strike_times=lst,
        right_strike_times=rst,
        left_off_times=lot,
        right_off_times=rot,
        stance_mask=stance,
    )


__all__ = [
    "GaitEventConfig",
    "GaitEventsResult",
    "contacts_from_gait_events",
    "detect_gait_events",
]
