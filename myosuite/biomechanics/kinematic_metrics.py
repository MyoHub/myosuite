# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Kinematic movement-quality metrics and reference-cohort distance comparison.

Pure NumPy/SciPy functions on recorded end-effector trajectories ``(T, D)`` and
speed profiles ``(T,)``, complementing the muscle-effort metrics in
:mod:`myosuite.biomechanics.effort`:

- :func:`dimensionless_squared_jerk`: scale-free jerk smoothness (Hogan & Sternad, 2009).
- :func:`spectral_arc_length`: SPARC frequency-domain smoothness (Balasubramanian et al., 2015).
- :func:`velocity_bell_symmetry`: time-to-peak-speed fraction over the active window.
- :func:`two_thirds_power_law_r2`: curvature-velocity two-thirds power-law fit (Lacquaniti et al., 1983).
- :func:`fitts_index_of_difficulty` and :func:`fitts_throughput`: speed-accuracy index and throughput.
- :func:`dtw_alignment` and :func:`dtw_distance`: multidimensional Dynamic Time Warping with Sakoe-Chiba band (Sakoe & Chiba, 1978; Berndt & Clifford, 1994).
- :func:`energy_distance` and :func:`compare_against_reference_cohorts`: multivariate Energy Distance
  (Székely & Rizzo, 2013), 1D Wasserstein distances, and split-half inter-subject reference ceiling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.stats import wasserstein_distance


def _as_trajectory_2d(positions: np.ndarray) -> np.ndarray:
    arr = np.asarray(positions, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"positions must be 1D or 2D (T, D), got shape {arr.shape}")
    return arr


def _validate_dt(dt: float) -> float:
    dt_f = float(dt)
    if dt_f <= 0.0 or not np.isfinite(dt_f):
        raise ValueError(f"dt must be finite and positive, got {dt}")
    return dt_f


def dimensionless_squared_jerk(positions: np.ndarray, dt: float) -> float:
    """Compute dimensionless squared jerk (DSJ) of an end-effector trajectory.

    Uses the scale-free formulation of Hogan & Sternad (2009):
    ``0.5 * (T^5 / A^2) * integral ||d^3 x / dt^3||^2 dt``, where ``T`` is
    movement duration and ``A`` is trajectory amplitude. An analytic minimum-jerk
    point-to-point reach approaches ``360`` as ``dt -> 0``.

    Args:
        positions: End-effector trajectory of shape ``(T, D)`` or ``(T,)``.
        dt: Sampling interval in seconds.

    Returns:
        Non-negative dimensionless squared jerk scalar.
    """
    pos = _as_trajectory_2d(positions)
    dt_f = _validate_dt(dt)
    n_steps = pos.shape[0]
    if n_steps < 4:
        return 0.0

    disp = float(np.linalg.norm(pos[-1] - pos[0]))
    span = float(np.max(np.linalg.norm(pos - pos[0], axis=-1)))
    amplitude = max(disp, span)
    if amplitude < 1e-8:
        return 0.0

    duration = float(n_steps - 1) * dt_f
    jerk = np.diff(pos, n=3, axis=0) / (dt_f**3)
    integral_sq_jerk = float(np.sum(np.sum(np.square(jerk), axis=-1)) * dt_f)
    return float(0.5 * (duration**5 / (amplitude**2)) * integral_sq_jerk)


def spectral_arc_length(
    speeds: np.ndarray,
    dt: float,
    *,
    pad_level: int = 4,
    fc: float = 10.0,
    amp_th: float = 0.05,
) -> float:
    """Compute SPARC (Spectral Arc Length) movement smoothness (Balasubramanian et al., 2015).

    Values are non-positive (``<= 0``); values closer to ``0`` (e.g. ``-1.4`` to
    ``-1.8``) correspond to smooth single-peaked reaches, whereas submovements or
    oscillations produce more negative values.

    Args:
        speeds: Speed profile of shape ``(T,)``.
        dt: Sampling interval in seconds.
        pad_level: Zero-padding power-of-two increment for FFT resolution.
        fc: Maximum cutoff frequency in Hz.
        amp_th: Normalized magnitude threshold defining the adaptive cutoff.

    Returns:
        SPARC smoothness scalar (``<= 0.0``).
    """
    v = np.asarray(speeds, dtype=np.float64).ravel()
    dt_f = _validate_dt(dt)
    if v.size < 4:
        return 0.0
    v_peak = float(np.max(np.abs(v)))
    if v_peak <= 1e-9:
        return 0.0

    nfft = int(2 ** (np.ceil(np.log2(v.size)) + max(0, int(pad_level))))
    freqs = np.fft.rfftfreq(nfft, d=dt_f)
    spectrum = np.abs(np.fft.rfft(v, n=nfft))
    norm_ref = max(float(spectrum[0]), float(np.max(spectrum)), 1e-12)
    spectrum_norm = spectrum / norm_ref

    below_fc = np.flatnonzero(freqs <= float(fc))
    if below_fc.size < 2:
        below_fc = np.arange(min(2, freqs.size))
    above_th = np.flatnonzero(spectrum_norm[below_fc] >= float(amp_th))
    cutoff_idx = int(below_fc[above_th[-1]]) if above_th.size > 0 else int(below_fc[-1])
    cutoff_idx = max(cutoff_idx, 1)

    fc_eff = max(float(freqs[cutoff_idx]), 1e-12)
    df_norm = np.diff(freqs[: cutoff_idx + 1]) / fc_eff
    dv_norm = np.diff(spectrum_norm[: cutoff_idx + 1])
    arc_len = float(np.sum(np.sqrt(np.square(df_norm) + np.square(dv_norm))))
    return -arc_len


def velocity_bell_symmetry(
    speeds: np.ndarray, *, threshold_ratio: float = 0.05
) -> float:
    """Compute time-to-peak-velocity ratio across the active movement window.

    Args:
        speeds: Speed profile of shape ``(T,)``.
        threshold_ratio: Fraction of peak speed used to segment onset and offset.

    Returns:
        Normalized peak-velocity timing in ``[0.0, 1.0]``.
    """
    v = np.asarray(speeds, dtype=np.float64).ravel()
    if v.size < 2:
        return 0.5
    peak = float(np.max(v))
    if peak <= 1e-9:
        return 0.5

    active = np.flatnonzero(v >= float(threshold_ratio) * peak)
    start_idx = int(active[0]) if active.size > 0 else 0
    end_idx = int(active[-1]) if active.size > 0 else v.size - 1
    if end_idx <= start_idx:
        return 0.5

    seg = v[start_idx : end_idx + 1]
    peak_rel = int(np.argmax(seg))
    return float(peak_rel / float(end_idx - start_idx))


def two_thirds_power_law_r2(positions: np.ndarray, dt: float) -> float:
    """Evaluate conformity to the Lacquaniti et al. (1983) two-thirds power law.

    Evaluates curved-trajectory speed scaling ``v ~ kappa^(-1/3)`` by computing
    ``R^2`` of ``log(v)`` against the theoretical ``-1/3 * log(kappa)`` slope on
    curved movement segments, clamped to ``[0, 1]``.

    Args:
        positions: Trajectory array of shape ``(T, 2)`` or ``(T, 3)``.
        dt: Sampling interval in seconds.

    Returns:
        Power-law conformity score in ``[0.0, 1.0]``.
    """
    pos = _as_trajectory_2d(positions)
    dt_f = _validate_dt(dt)
    if pos.shape[0] < 6 or pos.shape[1] < 2:
        return 0.0

    if pos.shape[1] == 2:
        pos3 = np.column_stack([pos, np.zeros(pos.shape[0], dtype=np.float64)])
    else:
        pos3 = pos[:, :3]

    vel = np.gradient(pos3, dt_f, axis=0)
    acc = np.gradient(vel, dt_f, axis=0)

    vel_in = vel[2:-2]
    acc_in = acc[2:-2]
    speed = np.linalg.norm(vel_in, axis=-1)
    v_max = float(np.max(speed)) if speed.size > 0 else 0.0
    if v_max <= 1e-8:
        return 0.0

    cross_norm = np.linalg.norm(np.cross(vel_in, acc_in), axis=-1)
    curvature = cross_norm / np.maximum(speed**3, 1e-12)

    valid = (speed >= 0.10 * v_max) & (curvature >= 1e-2) & np.isfinite(curvature)
    if int(np.sum(valid)) < 5:
        return 0.0

    log_k = np.log(curvature[valid])
    log_v = np.log(speed[valid])
    if float(np.std(log_k)) < 1e-4:
        return 0.0

    ss_tot = float(np.sum(np.square(log_v - np.mean(log_v))))
    if ss_tot <= 1e-12:
        return 0.0

    pred_log_v = np.mean(log_v) - (1.0 / 3.0) * (log_k - np.mean(log_k))
    ss_res = float(np.sum(np.square(log_v - pred_log_v)))
    return float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))


def movement_time(
    speeds: np.ndarray, dt: float, *, threshold_ratio: float = 0.05
) -> float:
    """Compute active movement duration [s] between onset and offset thresholds.

    Args:
        speeds: Speed profile of shape ``(T,)``.
        dt: Sampling interval in seconds.
        threshold_ratio: Fraction of peak speed defining movement onset/offset.

    Returns:
        Active movement duration in seconds.
    """
    v = np.asarray(speeds, dtype=np.float64).ravel()
    dt_f = _validate_dt(dt)
    if v.size < 2:
        return dt_f
    peak = float(np.max(v))
    if peak <= 1e-9:
        return float(v.size - 1) * dt_f
    active = np.flatnonzero(v >= float(threshold_ratio) * peak)
    if active.size < 2:
        return dt_f
    return float(max(1, int(active[-1] - active[0])) * dt_f)


def endpoint_error(positions: np.ndarray, target_position: np.ndarray) -> float:
    """Compute Euclidean distance [m] between final end-effector position and target.

    Args:
        positions: Trajectory of shape ``(T, D)`` or final position ``(D,)``.
        target_position: Target position of shape ``(D,)``.

    Returns:
        Terminal Euclidean distance in meters.
    """
    pos = np.asarray(positions, dtype=np.float64)
    tgt = np.asarray(target_position, dtype=np.float64).ravel()
    final_pos = pos[-1].ravel() if pos.ndim >= 2 else pos.ravel()
    return float(np.linalg.norm(final_pos - tgt))


def fitts_index_of_difficulty(distance: float, target_width: float) -> float:
    """Compute Shannon formulation Fitts index of difficulty ``log2(D / W + 1)`` [bits].

    Args:
        distance: Movement amplitude ``D`` (must be positive).
        target_width: Effective target width ``W`` (must be positive).

    Returns:
        Index of difficulty in bits (MacKenzie, 1992).
    """
    d_val = float(distance)
    w_val = float(target_width)
    if not np.isfinite(d_val) or d_val <= 0.0:
        raise ValueError(f"distance must be positive and finite, got {distance}")
    if not np.isfinite(w_val) or w_val <= 0.0:
        raise ValueError(
            f"target_width must be positive and finite, got {target_width}"
        )
    return float(np.log2(d_val / w_val + 1.0))


def fitts_throughput(
    distance: float, target_width: float, movement_duration: float
) -> float:
    """Compute Fitts throughput ``ID / MT`` [bits/s].

    Args:
        distance: Movement amplitude ``D`` [m].
        target_width: Target width ``W`` [m].
        movement_duration: Movement time ``MT`` [s] (must be positive).

    Returns:
        Throughput in bits per second.
    """
    mt = float(movement_duration)
    if not np.isfinite(mt) or mt <= 0.0:
        raise ValueError(
            f"movement_duration must be positive and finite, got {movement_duration}"
        )
    return float(fitts_index_of_difficulty(distance, target_width) / mt)


@dataclass(frozen=True)
class KinematicTrialSummary:
    """Summary of kinematic trajectory quality metrics for a recorded trial."""

    dimensionless_squared_jerk: float
    spectral_arc_length: float
    velocity_bell_symmetry: float
    two_thirds_power_law_r2: float
    movement_time: float
    endpoint_error: float
    fitts_id: float
    fitts_throughput: float

    FEATURE_NAMES: ClassVar[tuple[str, ...]] = (
        "dimensionless_squared_jerk",
        "spectral_arc_length",
        "velocity_bell_symmetry",
        "two_thirds_power_law_r2",
        "movement_time",
        "endpoint_error",
        "fitts_throughput",
    )

    def to_feature_vector(self) -> np.ndarray:
        """Pack trial metrics into a 1D ``float64`` feature vector."""
        return np.array(
            [
                self.dimensionless_squared_jerk,
                self.spectral_arc_length,
                self.velocity_bell_symmetry,
                self.two_thirds_power_law_r2,
                self.movement_time,
                self.endpoint_error,
                self.fitts_throughput,
            ],
            dtype=np.float64,
        )


def summarize_kinematic_trial(
    positions: np.ndarray,
    dt: float,
    target_position: np.ndarray,
    *,
    target_width: float = 0.05,
) -> KinematicTrialSummary:
    """Compute a :class:`KinematicTrialSummary` for a single recorded trajectory.

    Args:
        positions: End-effector trajectory of shape ``(T, D)``.
        dt: Sampling interval in seconds.
        target_position: Goal position of shape ``(D,)``.
        target_width: Effective target tolerance width ``W`` [m].

    Returns:
        Populated :class:`KinematicTrialSummary`.
    """
    pos = _as_trajectory_2d(positions)
    dt_f = _validate_dt(dt)
    vel = np.gradient(pos, dt_f, axis=0) if pos.shape[0] >= 2 else np.zeros_like(pos)
    speeds = np.linalg.norm(vel, axis=-1)

    mt = movement_time(speeds, dt_f)
    tgt = np.asarray(target_position, dtype=np.float64).ravel()
    dist = max(float(np.linalg.norm(tgt - pos[0])), 1e-3)
    fid = fitts_index_of_difficulty(dist, target_width)
    ftp = fitts_throughput(dist, target_width, mt)
    return KinematicTrialSummary(
        dimensionless_squared_jerk=dimensionless_squared_jerk(pos, dt_f),
        spectral_arc_length=spectral_arc_length(speeds, dt_f),
        velocity_bell_symmetry=velocity_bell_symmetry(speeds),
        two_thirds_power_law_r2=two_thirds_power_law_r2(pos, dt_f),
        movement_time=mt,
        endpoint_error=endpoint_error(pos, tgt),
        fitts_id=fid,
        fitts_throughput=ftp,
    )


@dataclass(frozen=True)
class CohortComparison:
    """Distributional comparison against multi-subject reference cohorts.

    Attributes:
        raw_distance: Multivariate Energy Distance between standardized rollout
            features and split reference half-cohorts.
        reference_split_half_distance: Monte Carlo mean split-half Energy Distance
            between disjoint halves of the reference cohorts.
        ceiling_normalized_score: Ceiling-normalized similarity ``2 * c / (c + d)``
            where ``1.0`` indicates rollout features match reference cohorts as
            closely as disjoint halves of the reference cohorts match each other.
        per_feature_wasserstein: 1D Wasserstein distance per standardized feature.
        reference_split_half_std: Standard deviation of split-half Energy Distances.
    """

    raw_distance: float
    reference_split_half_distance: float
    ceiling_normalized_score: float
    per_feature_wasserstein: np.ndarray
    reference_split_half_std: float


def energy_distance(sample_x: np.ndarray, sample_y: np.ndarray) -> float:
    """Compute multivariate Energy Distance (Székely & Rizzo, 2013) between two samples.

    Args:
        sample_x: Array of shape ``(N, D)``.
        sample_y: Array of shape ``(M, D)``.

    Returns:
        Non-negative Energy Distance ``sqrt(max(0, 2 E||X-Y|| - E||X-X'|| - E||Y-Y'||))``.
    """
    x = np.asarray(sample_x, dtype=np.float64)
    y = np.asarray(sample_y, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("Both samples must contain at least one observation.")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Feature dimension mismatch: {x.shape[1]} vs {y.shape[1]}")

    d_xy = float(np.mean(np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)))
    d_xx = float(np.mean(np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)))
    d_yy = float(np.mean(np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1)))
    e2 = max(0.0, 2.0 * d_xy - d_xx - d_yy)
    return float(np.sqrt(e2))


def compare_against_reference_cohorts(
    rollout_features: np.ndarray,
    reference_cohorts: Sequence[np.ndarray],
    *,
    n_splits: int = 100,
    rng: int | np.random.Generator | None = 0,
) -> CohortComparison:
    """Compare rollout trial features against multi-subject reference cohorts.

    Args:
        rollout_features: Feature matrix of shape ``(N_rollouts, D_features)``.
        reference_cohorts: Sequence of ``P >= 2`` per-subject feature arrays,
            each of shape ``(N_i, D_features)``.
        n_splits: Number of Monte Carlo half-splits (``>= 1``).
        rng: Integer seed or ``np.random.Generator`` for reproducible splits.

    Returns:
        :class:`CohortComparison` with Energy Distance, split-half ceiling,
        ceiling-normalized similarity, and 1D Wasserstein distances.
    """
    model_arr = np.asarray(rollout_features, dtype=np.float64)
    if model_arr.ndim == 1:
        model_arr = model_arr[None, :]
    if model_arr.ndim != 2 or model_arr.shape[0] == 0:
        raise ValueError(
            f"rollout_features must be non-empty 2D (N, D), got shape {model_arr.shape}"
        )

    if len(reference_cohorts) < 2:
        raise ValueError(
            "reference_cohorts requires at least 2 cohorts to compute a split-half "
            f"reference ceiling, got {len(reference_cohorts)}."
        )
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")

    n_features = model_arr.shape[1]
    cohorts: list[np.ndarray] = []
    for idx, part in enumerate(reference_cohorts):
        p_arr = np.asarray(part, dtype=np.float64)
        if p_arr.ndim == 1:
            p_arr = p_arr[None, :]
        if p_arr.ndim != 2 or p_arr.shape[0] == 0 or p_arr.shape[1] != n_features:
            raise ValueError(
                f"Cohort {idx} array must have shape (N_i, {n_features}), got {p_arr.shape}"
            )
        cohorts.append(p_arr)

    pooled_ref = np.vstack(cohorts)
    mean = np.mean(pooled_ref, axis=0)
    std = np.std(pooled_ref, axis=0)
    scale = np.where(std > 1e-8, std, 1.0)

    z_model = (model_arr - mean) / scale
    z_parts = [(p - mean) / scale for p in cohorts]
    z_ref = (pooled_ref - mean) / scale

    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    n_parts = len(z_parts)
    half = n_parts // 2

    split_distances = np.empty(int(n_splits), dtype=np.float64)
    model_half_distances = np.empty(int(n_splits), dtype=np.float64)
    for s_idx in range(int(n_splits)):
        perm = gen.permutation(n_parts)
        half_a = np.vstack([z_parts[i] for i in perm[:half]])
        half_b = np.vstack([z_parts[i] for i in perm[half:]])
        split_distances[s_idx] = energy_distance(half_a, half_b)
        model_half_distances[s_idx] = 0.5 * (
            energy_distance(z_model, half_a) + energy_distance(z_model, half_b)
        )

    raw_dist = float(np.mean(model_half_distances))
    ceiling_dist = float(np.mean(split_distances))
    ceiling_std = float(np.std(split_distances))

    per_feat_w1 = np.array(
        [
            float(wasserstein_distance(z_model[:, f], z_ref[:, f]))
            for f in range(n_features)
        ],
        dtype=np.float64,
    )

    if ceiling_dist <= 1e-12 and raw_dist <= 1e-12:
        norm_score = 1.0
    elif ceiling_dist <= 1e-12:
        norm_score = 0.0
    else:
        norm_score = float((2.0 * ceiling_dist) / (ceiling_dist + raw_dist))

    return CohortComparison(
        raw_distance=raw_dist,
        reference_split_half_distance=ceiling_dist,
        ceiling_normalized_score=norm_score,
        per_feature_wasserstein=per_feat_w1,
        reference_split_half_std=ceiling_std,
    )


@dataclass(frozen=True)
class DTWAlignment:
    """Dynamic Time Warping alignment between two trajectories.

    Attributes:
        total_cost: Accumulated Euclidean cost along the optimal warping path.
        normalized_distance: Mean per-step Euclidean distance ``total_cost / len(path)``.
        path: Integer index pairs ``(K, 2)`` monotonically mapping ``seq_x`` to ``seq_y``.
    """

    total_cost: float
    normalized_distance: float
    path: np.ndarray


def dtw_alignment(
    seq_x: np.ndarray,
    seq_y: np.ndarray,
    *,
    window: int | None = None,
    zero_start: bool = False,
) -> DTWAlignment:
    """Align two 1D or 2D trajectories via Dynamic Time Warping (Sakoe & Chiba, 1978).

    Args:
        seq_x: First trajectory of shape ``(T1, D)`` or ``(T1,)``.
        seq_y: Second trajectory of shape ``(T2, D)`` or ``(T2,)``.
        window: Optional non-negative Sakoe-Chiba band half-width in steps
            (widened to ``abs(T1 - T2)`` if smaller so endpoints remain reachable).
        zero_start: Translate each trajectory by its initial sample before comparison.

    Returns:
        :class:`DTWAlignment` with total cost, path-length-normalized distance, and path.
    """
    x = _as_trajectory_2d(seq_x)
    y = _as_trajectory_2d(seq_y)
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("DTW trajectories must be non-empty.")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Trajectory dimension mismatch: {x.shape[1]} vs {y.shape[1]}")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("DTW trajectories must contain finite values.")

    if zero_start:
        x = x - x[:1]
        y = y - y[:1]

    n, m = x.shape[0], y.shape[0]
    if window is None:
        band = max(n, m)
    else:
        if int(window) < 0:
            raise ValueError(f"window must be non-negative or None, got {window}")
        band = max(int(window), abs(n - m))

    local_cost = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        j_lo = max(1, i - band)
        j_hi = min(m, i + band)
        row_c = local_cost[i - 1]
        prev_row = dp[i - 1]
        cur_row = dp[i]
        for j in range(j_lo, j_hi + 1):
            best_prev = prev_row[j - 1]
            if prev_row[j] < best_prev:
                best_prev = prev_row[j]
            if cur_row[j - 1] < best_prev:
                best_prev = cur_row[j - 1]
            cur_row[j] = row_c[j - 1] + best_prev

    i, j = n, m
    rev_path: list[tuple[int, int]] = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            c_diag = dp[i - 1, j - 1]
            c_up = dp[i - 1, j]
            c_left = dp[i, j - 1]
            if c_diag <= c_up and c_diag <= c_left:
                i -= 1
                j -= 1
            elif c_up <= c_left:
                i -= 1
            else:
                j -= 1
        rev_path.append((i - 1, j - 1))

    path_arr = np.asarray(rev_path[::-1], dtype=np.int64)
    total = float(dp[n, m])
    return DTWAlignment(
        total_cost=total,
        normalized_distance=total / float(len(path_arr)),
        path=path_arr,
    )


def dtw_distance(
    seq_x: np.ndarray,
    seq_y: np.ndarray,
    *,
    window: int | None = None,
    normalize: bool = True,
    zero_start: bool = False,
) -> float:
    """Compute scalar Dynamic Time Warping distance between two trajectories."""
    res = dtw_alignment(seq_x, seq_y, window=window, zero_start=zero_start)
    return res.normalized_distance if normalize else res.total_cost


__all__ = [
    "CohortComparison",
    "DTWAlignment",
    "KinematicTrialSummary",
    "compare_against_reference_cohorts",
    "dimensionless_squared_jerk",
    "dtw_alignment",
    "dtw_distance",
    "endpoint_error",
    "energy_distance",
    "fitts_index_of_difficulty",
    "fitts_throughput",
    "movement_time",
    "spectral_arc_length",
    "summarize_kinematic_trial",
    "two_thirds_power_law_r2",
    "velocity_bell_symmetry",
]
