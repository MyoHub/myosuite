# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Movement-quality metrics for point-to-point movements (HCI and motor control).

Pure NumPy/SciPy functions over trajectories sampled at a fixed interval ``dt``
(seconds); positions are ``(N,)`` (one coordinate) or ``(N, d)`` arrays:

* minimum-jerk reference (Flash & Hogan 1985);
* kinematics: speed, path length, straightness (path-length ratio);
* segmentation: movement onset/offset and movement time, target entry and dwell;
* velocity-profile shape: peak speed, time-to-peak ratio, speed peaks;
* smoothness: dimensionless jerk, log dimensionless jerk (LDLJ) and SPARC;
* Fitts' law: index of difficulty, effective width/distance/ID, throughput, regression;
* the two-thirds power law of curved movements.

:func:`point_to_point_metrics` combines them for one trial (an end-effector or a
joint-space trajectory and its target); ``ReachEnvV0.get_metrics`` and
``PoseEnvV0.get_metrics`` apply it to rollout paths.

References:
    Ackermann, M. & van den Bogert, A. J. (2010). Optimality principles for model-based
        prediction of human gait. J. Biomech. 43(6), 1055-1060.
    Balasubramanian, S., Melendez-Calderon, A. & Burdet, E. (2012). A robust and
        sensitive metric for quantifying movement smoothness. IEEE TBME 59(8), 2126-2136.
    Balasubramanian, S., Melendez-Calderon, A., Roby-Brami, A. & Burdet, E. (2015). On
        the analysis of movement smoothness. J. NeuroEng. Rehabil. 12, 112. Reference
        code: github.com/siva82kb/SPARC (``scripts/smoothness.py``).
    Fitts, P. M. (1954). The information capacity of the human motor system in
        controlling the amplitude of movement. J. Exp. Psychol. 47(6), 381-391.
    Flash, T. & Hogan, N. (1985). The coordination of arm movements: an experimentally
        confirmed mathematical model. J. Neurosci. 5(7), 1688-1703.
    Hogan, N. & Sternad, D. (2009). Sensitivity of smoothness measures to movement
        duration, amplitude, and arrests. J. Mot. Behav. 41(6), 529-534.
    ISO 9241-411 (2012). Ergonomics of human-system interaction, Part 411: Evaluation
        methods for the design of physical input devices.
    Lacquaniti, F., Terzuolo, C. & Viviani, P. (1983). The law relating the kinematic and
        figural aspects of drawing movements. Acta Psychol. 54, 115-130.
    MacKenzie, I. S. (1992). Fitts' law as a research and design tool in human-computer
        interaction. Hum.-Comput. Interact. 7, 91-139.
    MacKenzie, I. S. (2018). Fitts' law. In Norman & Kirakowski (Eds.), Handbook of
        Human-Computer Interaction, 349-370. Wiley.
    MacKenzie, I. S., Kauppinen, T. & Silfverberg, M. (2001). Accuracy measures for
        evaluating computer pointing devices. Proc. CHI 2001, 9-16.
    Nagasaki, H. (1989). Asymmetric velocity and acceleration profiles of human arm
        movements. Exp. Brain Res. 74, 319-326.
    Rohrer, B. et al. (2002). Movement smoothness changes during stroke recovery.
        J. Neurosci. 22(18), 8297-8304.
    Schot, W. D., Brenner, E. & Smeets, J. B. J. (2010). Robust movement segmentation by
        combining multiple sources of information. J. Neurosci. Methods 187, 147-155.
    Schwarz, A., Kanzler, C. M., Lambercy, O., Luft, A. R. & Veerbeek, J. M. (2019).
        Systematic review on kinematic assessments of upper limb movements after stroke.
        Stroke 50(3), 718-727.
    Soukoreff, R. W. & MacKenzie, I. S. (2004). Towards a standard for pointing device
        evaluation, perspectives on 27 years of Fitts' law research in HCI. Int. J.
        Hum.-Comput. Stud. 61(6), 751-789.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import simpson
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import linregress

#: Effective-width factor: sqrt(2*pi*e) = 4.1327 rounded as published, so that
#: log2(We) is the entropy of normally distributed endpoints (Soukoreff & MacKenzie
#: 2004; ISO 9241-411; MacKenzie 2018, Fig. 17.7).
EFFECTIVE_WIDTH_FACTOR = 4.133

#: Onset/offset threshold as a fraction of peak speed. 5% is a common convention for
#: speed-threshold segmentation (see Schot, Brenner & Smeets 2010 for its pitfalls).
ONSET_FRACTION = 0.05

#: SPARC defaults of Balasubramanian et al. (2015) and their reference code: zero
#: padding level, maximum cutoff frequency (Hz) and adaptive amplitude threshold.
SPARC_PADLEVEL = 4
SPARC_MAX_CUTOFF_HZ = 10.0
SPARC_AMPLITUDE_THRESHOLD = 0.05

#: Peak speed of a minimum-jerk movement in units of D/T: 30 * 0.5^2 * 0.5^2.
MIN_JERK_PEAK_SPEED_FACTOR = 1.875

#: Savitzky-Golay smoothing parameters ``(window_length, polyorder)``.
SavgolParams = tuple[int, int]


class MinimumJerk(NamedTuple):
    """Minimum-jerk trajectory samples (:func:`minimum_jerk`)."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


@dataclass(frozen=True)
class MovementBounds:
    """Movement onset and offset of a speed profile.

    Attributes:
        onset_index: First sample with speed at or above the threshold.
        offset_index: Last sample with speed at or above the threshold.
        onset_time: Threshold crossing before ``onset_index`` (s, linearly interpolated).
        offset_time: Threshold crossing after ``offset_index`` (s, linearly interpolated).
    """

    onset_index: int
    offset_index: int
    onset_time: float
    offset_time: float

    @property
    def movement_time(self) -> float:
        """Offset minus onset time (s)."""
        return self.offset_time - self.onset_time


@dataclass(frozen=True)
class TargetAcquisition:
    """Target entry and acquisition of one trial (:func:`target_acquisition`).

    Attributes:
        entry_time: Time of the first sample inside the target (s); NaN if never inside.
        acquisition_time: Time of the entry that starts the stable dwell (s); NaN if the
            target is never acquired.
        entries: Number of separate entries into the target; ``entries - 1`` is the
            target re-entry count TRE of MacKenzie, Kauppinen & Silfverberg (2001).
    """

    entry_time: float
    acquisition_time: float
    entries: int


@dataclass(frozen=True)
class EffectiveParameters:
    """Effective Fitts' law parameters of one condition (:func:`effective_parameters`).

    Attributes:
        distance: Effective distance De, the mean movement amplitude along the task axis.
        width: Effective width We = 4.133 * SD of the endpoint deviations.
        index_of_difficulty: Effective index of difficulty IDe = log2(De / We + 1) (bits).
    """

    distance: float
    width: float
    index_of_difficulty: float


@dataclass(frozen=True)
class FittsRegression:
    """Least-squares fit ``MT = a + b * ID`` (:func:`fitts_regression`).

    Attributes:
        intercept: ``a`` (s).
        slope: ``b`` (s/bit).
        r_squared: Coefficient of determination.
        p_value: Two-sided p-value of the slope (Wald test).
        stderr: Standard error of the slope.
    """

    intercept: float
    slope: float
    r_squared: float
    p_value: float
    stderr: float


@dataclass(frozen=True)
class PowerLawFit:
    """Speed-curvature power law ``v = K * kappa**beta`` (:func:`two_thirds_power_law`).

    Attributes:
        exponent: ``beta``; -1/3 for the two-thirds power law.
        gain: ``K`` (velocity gain factor).
        r_squared: Coefficient of determination of the log-log fit.
    """

    exponent: float
    gain: float
    r_squared: float


# ── Helpers ──────────────────────────────────────────────────────────────────


def _check_dt(dt: float) -> None:
    if not dt > 0:
        raise ValueError(f"dt must be positive, got {dt}")


def _as_2d(positions: ArrayLike) -> np.ndarray:
    """``(N,)`` -> ``(N, 1)``; ``(N, d)`` unchanged."""
    x = np.asarray(positions, dtype=float)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim != 2:
        raise ValueError(f"positions must be (N,) or (N, d), got shape {x.shape}")
    return x


def _finite_mean(values: ArrayLike) -> float:
    """Mean of the finite values; NaN if there are none."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else math.nan


# ── Minimum-jerk reference ───────────────────────────────────────────────────


def minimum_jerk(
    start: ArrayLike, end: ArrayLike, duration: float, t: ArrayLike
) -> MinimumJerk:
    """Minimum-jerk point-to-point trajectory (Flash & Hogan 1985).

    The unconstrained point-to-point solution of Flash & Hogan (1985), which minimises
    the integrated squared jerk with zero velocity and acceleration at both ends:
    ``x(t) = x0 + (xf - x0) * (10 tau^3 - 15 tau^4 + 6 tau^5)`` with ``tau = t / T``.
    Peak speed is ``1.875 * |xf - x0| / T`` at ``tau = 0.5``. Times outside ``[0, T]``
    hold the start/end position.

    Args:
        start: Start position, scalar or ``(d,)``.
        end: End position, same shape as ``start``.
        duration: Movement duration ``T`` (s).
        t: Sample times ``(N,)`` (s).

    Returns:
        Position, velocity and acceleration, each ``(N,)`` for a scalar start or
        ``(N, d)``.
    """
    if not duration > 0:
        raise ValueError(f"duration must be positive, got {duration}")
    start = np.asarray(start, dtype=float)
    displacement = np.asarray(end, dtype=float) - start
    tau = np.clip(np.asarray(t, dtype=float) / duration, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
    dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / duration**2
    return MinimumJerk(
        position=start + np.multiply.outer(s, displacement),
        velocity=np.multiply.outer(ds, displacement),
        acceleration=np.multiply.outer(dds, displacement),
    )


# ── Kinematics ───────────────────────────────────────────────────────────────


def derivative(
    x: ArrayLike, dt: float, order: int = 1, savgol: SavgolParams | None = None
) -> np.ndarray:
    """Time derivative of a uniformly sampled signal along axis 0.

    Without smoothing, the samples are interpolated by a quintic B-spline
    (``scipy.interpolate.make_interp_spline``, ``k = 5``, not-a-knot ends) and the
    spline is differentiated: exact for polynomials up to degree 5 (so for
    minimum-jerk movements, at any sampling rate and at the ends, where repeated
    finite differences are biased) and accurate for smooth, noise-free (simulated)
    data. With ``savgol``, ``scipy.signal.savgol_filter`` fits and differentiates local
    polynomials (Savitzky-Golay smoothing); use it for noisy (measured or
    noise-injected) data.

    Args:
        x: Samples ``(N, ...)``, ``N >= order + 1``.
        dt: Sample interval (s).
        order: Derivative order (1 velocity, 2 acceleration, 3 jerk).
        savgol: Optional ``(window_length, polyorder)``; ``polyorder >= order``.

    Returns:
        The derivative, same shape as ``x``.
    """
    _check_dt(dt)
    y = np.asarray(x, dtype=float)
    if savgol is not None:
        window, polyorder = savgol
        return savgol_filter(y, window, polyorder, deriv=order, delta=dt, axis=0)
    n = len(y)
    if n < order + 1:
        raise ValueError(
            f"a derivative of order {order} needs {order + 1} samples, got {n}"
        )
    t = np.arange(n) * dt
    spline = make_interp_spline(t, y, k=min(5, n - 1), axis=0)
    return spline.derivative(order)(t)


def speed(
    positions: ArrayLike, dt: float, savgol: SavgolParams | None = None
) -> np.ndarray:
    """Tangential speed ``||dx/dt||`` of a trajectory.

    Args:
        positions: ``(N,)`` or ``(N, d)`` positions.
        dt: Sample interval (s).
        savgol: Optional Savitzky-Golay smoothing, see :func:`derivative`.

    Returns:
        Speed ``(N,)`` (position units per second).
    """
    return np.linalg.norm(derivative(_as_2d(positions), dt, 1, savgol), axis=-1)


def path_length(positions: ArrayLike) -> float:
    """Length of the sampled path (sum of the segment lengths).

    Args:
        positions: ``(N,)`` or ``(N, d)`` positions.

    Returns:
        Path length (position units).
    """
    return float(np.linalg.norm(np.diff(_as_2d(positions), axis=0), axis=-1).sum())


def straightness(positions: ArrayLike) -> float:
    """Path-length ratio: path length over the start-to-end distance.

    1 for a straight path, larger for curved or corrected paths (the "path length
    ratio" or "index of curvature" of the kinematic-assessment literature; Schwarz et
    al. 2019).

    Args:
        positions: ``(N,)`` or ``(N, d)`` positions.

    Returns:
        The ratio (>= 1); NaN if start and end coincide.
    """
    x = _as_2d(positions)
    chord = float(np.linalg.norm(x[-1] - x[0]))
    return path_length(x) / chord if chord > 0 else math.nan


# ── Segmentation ─────────────────────────────────────────────────────────────


def movement_bounds(
    speed_profile: ArrayLike, dt: float, fraction: float = ONSET_FRACTION
) -> MovementBounds | None:
    """Movement onset and offset by a speed threshold.

    Convention: the threshold is ``fraction * peak speed`` (default 5%; Schot, Brenner
    & Smeets 2010 review this family of criteria). Onset is the first and offset the
    last sample at or above it, so the movement time includes corrective submovements
    (the total movement, not only the primary submovement). Crossing times are
    linearly interpolated between samples; a movement already above threshold at the
    first (last) sample starts (ends) there.

    Args:
        speed_profile: Speed ``(N,)``.
        dt: Sample interval (s).
        fraction: Threshold as a fraction of peak speed, in ``(0, 1)``.

    Returns:
        The bounds, or ``None`` for a profile without movement (peak speed 0).
    """
    _check_dt(dt)
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    s = np.asarray(speed_profile, dtype=float)
    peak = float(s.max()) if s.size else 0.0
    if not (np.isfinite(peak) and peak > 0):
        return None
    threshold = fraction * peak
    above = np.flatnonzero(s >= threshold)
    i0, i1 = int(above[0]), int(above[-1])

    def crossing(a: int, b: int) -> float:
        # Linear interpolation of the threshold crossing between samples a and b.
        return (a + (threshold - s[a]) / (s[b] - s[a])) * dt

    onset = crossing(i0 - 1, i0) if i0 > 0 else 0.0
    offset = crossing(i1, i1 + 1) if i1 < len(s) - 1 else i1 * dt
    return MovementBounds(i0, i1, float(onset), float(offset))


def target_acquisition(
    inside: ArrayLike, dt: float, dwell_time: float | None = None
) -> TargetAcquisition:
    """Target entry, acquisition and entry count of one trial.

    Sample ``i`` is at time ``i * dt``. With ``dwell_time=None`` the target is acquired
    at the start of the final stay inside, if that stay lasts to the last sample
    ("reached and held"). With a ``dwell_time`` it is acquired at the first entry
    after which the trajectory stays inside for at least ``dwell_time``
    (``ceil(dwell_time / dt)`` samples), as in dwell-based selection; selection then
    completes ``dwell_time`` later.

    Args:
        inside: Boolean ``(N,)``, e.g. ``distance < radius``.
        dt: Sample interval (s).
        dwell_time: Required stay inside (s), or ``None``.

    Returns:
        Entry time, acquisition time and number of entries.
    """
    _check_dt(dt)
    mask = np.asarray(inside, dtype=bool)
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if starts.size == 0:
        return TargetAcquisition(math.nan, math.nan, 0)
    if dwell_time is None:
        acquired = starts[-1] * dt if mask[-1] else math.nan
    else:
        needed = max(1, math.ceil(dwell_time / dt - 1e-9))
        long_stays = starts[(ends - starts) >= needed]
        acquired = long_stays[0] * dt if long_stays.size else math.nan
    return TargetAcquisition(float(starts[0] * dt), float(acquired), int(starts.size))


# ── Velocity-profile shape ───────────────────────────────────────────────────


def time_to_peak_ratio(
    speed_profile: ArrayLike, dt: float, bounds: MovementBounds
) -> float:
    """Share of the movement time spent accelerating (velocity-profile symmetry).

    ``(t_peak - t_onset) / (t_offset - t_onset)``: 0.5 for a symmetric bell-shaped
    profile such as minimum jerk, below 0.5 for a long deceleration phase typical of
    aimed movements (Nagasaki 1989).

    Args:
        speed_profile: Speed ``(N,)``.
        dt: Sample interval (s).
        bounds: Movement bounds of the profile (:func:`movement_bounds`).

    Returns:
        The ratio in ``[0, 1]``; NaN for a zero movement time.
    """
    s = np.asarray(speed_profile, dtype=float)
    i_peak = bounds.onset_index + int(
        np.argmax(s[bounds.onset_index : bounds.offset_index + 1])
    )
    duration = bounds.movement_time
    if duration <= 0:
        return math.nan
    return float(np.clip((i_peak * dt - bounds.onset_time) / duration, 0.0, 1.0))


def count_speed_peaks(
    speed_profile: ArrayLike,
    height_fraction: float = ONSET_FRACTION,
    prominence_fraction: float = ONSET_FRACTION,
) -> int:
    """Number of speed peaks, a count of submovements (the peaks metric of Rohrer et al. 2002).

    Peaks are found with ``scipy.signal.find_peaks``; a peak must reach
    ``height_fraction * peak speed`` and stand out by ``prominence_fraction * peak
    speed`` so that numerical ripple is not counted. The profile is padded with zero
    speed (rest) so a peak at the first or last sample counts.

    Args:
        speed_profile: Speed ``(N,)``.
        height_fraction: Minimum peak height relative to peak speed.
        prominence_fraction: Minimum peak prominence relative to peak speed.

    Returns:
        Number of peaks (0 for no movement).
    """
    s = np.pad(np.asarray(speed_profile, dtype=float), 1)
    peak = float(s.max())
    if not peak > 0:
        return 0
    peaks, _ = find_peaks(
        s, height=height_fraction * peak, prominence=prominence_fraction * peak
    )
    return int(peaks.size)


# ── Smoothness ───────────────────────────────────────────────────────────────


def dimensionless_jerk(
    positions: ArrayLike,
    dt: float,
    *,
    normalization: Literal["peak_speed", "amplitude"] = "peak_speed",
    savgol: SavgolParams | None = None,
    segment: tuple[int, int] | None = None,
) -> float:
    """Dimensionless squared jerk of a movement (>= 0, larger is less smooth).

    ``J = integral ||d^3x/dt^3||^2 dt`` over the movement, made dimensionless by its
    duration ``T`` and either

    * ``"peak_speed"``: ``DJ = T^3 / v_peak^2 * J`` (Balasubramanian et al. 2012, 2015;
      the "LDLJ-V" form of their reference code, which carries a minus sign), or
    * ``"amplitude"``: ``DJ = T^5 / A^2 * J`` with ``A`` the start-to-end distance
      (Hogan & Sternad 2009).

    A minimum-jerk movement gives exactly ``720`` (amplitude; ``int_0^1 (60 - 360 tau +
    360 tau^2)^2 dtau``) and ``720 / 1.875^2 = 204.8`` (peak speed). The jerk is that of
    the position vector (Flash & Hogan 1985), which equals the second derivative of
    the speed for straight movements. Derivatives are taken over the whole input so
    that a ``segment`` has no edge effects; ``T = (i1 - i0) * dt`` (the reference code
    uses ``N * dt``) and the integral uses Simpson's rule.

    Args:
        positions: ``(N,)`` or ``(N, d)`` positions.
        dt: Sample interval (s).
        normalization: ``"peak_speed"`` or ``"amplitude"``.
        savgol: Optional Savitzky-Golay smoothing (``polyorder >= 3``).
        segment: Inclusive sample range ``(i0, i1)`` of the movement; default all.

    Returns:
        The dimensionless jerk; NaN if the segment is shorter than 3 samples or the
        normaliser is zero.
    """
    x = _as_2d(positions)
    i0, i1 = (0, len(x) - 1) if segment is None else segment
    if i1 - i0 < 2 or len(x) < 4:  # a third derivative needs 4 samples
        return math.nan
    window = slice(i0, i1 + 1)
    jerk = derivative(x, dt, 3, savgol)[window]
    duration = (i1 - i0) * dt
    jerk_integral = float(simpson(np.sum(jerk**2, axis=-1), dx=dt))
    if normalization == "peak_speed":
        v_peak = float(
            np.linalg.norm(derivative(x, dt, 1, savgol)[window], axis=-1).max()
        )
        scale = duration**3 / v_peak**2 if v_peak > 0 else math.nan
    elif normalization == "amplitude":
        amplitude = float(np.linalg.norm(x[i1] - x[i0]))
        scale = duration**5 / amplitude**2 if amplitude > 0 else math.nan
    else:
        raise ValueError(f"unknown normalization {normalization!r}")
    return jerk_integral * scale


def log_dimensionless_jerk(
    positions: ArrayLike,
    dt: float,
    *,
    normalization: Literal["peak_speed", "amplitude"] = "peak_speed",
    savgol: SavgolParams | None = None,
    segment: tuple[int, int] | None = None,
) -> float:
    """Log dimensionless jerk ``LDLJ = -ln(DJ)`` (Balasubramanian et al. 2012, 2015).

    Higher (less negative) is smoother; minimum jerk gives ``-ln(204.8) = -5.322``
    (peak speed) or ``-ln(720) = -6.579`` (amplitude). See :func:`dimensionless_jerk`.

    Args:
        positions: ``(N,)`` or ``(N, d)`` positions.
        dt: Sample interval (s).
        normalization: ``"peak_speed"`` or ``"amplitude"``.
        savgol: Optional Savitzky-Golay smoothing (``polyorder >= 3``).
        segment: Inclusive sample range ``(i0, i1)`` of the movement; default all.

    Returns:
        LDLJ; NaN where :func:`dimensionless_jerk` is undefined or zero.
    """
    dj = dimensionless_jerk(
        positions, dt, normalization=normalization, savgol=savgol, segment=segment
    )
    return -math.log(dj) if dj > 0 else math.nan


def sparc(
    speed_profile: ArrayLike,
    dt: float,
    *,
    padlevel: int = SPARC_PADLEVEL,
    max_cutoff: float = SPARC_MAX_CUTOFF_HZ,
    amplitude_threshold: float = SPARC_AMPLITUDE_THRESHOLD,
) -> float:
    """Spectral arc length SPARC of a speed profile (Balasubramanian et al. 2015).

    The negative arc length of the DC-normalised Fourier magnitude spectrum of the
    zero-padded speed profile, from 0 Hz to the adaptive cutoff (the highest
    frequency below ``max_cutoff`` whose magnitude is at least
    ``amplitude_threshold``), with frequencies normalised by that cutoff. Higher (less
    negative) is smoother. Identical to the authors' reference ``sparc`` for
    ``max_cutoff`` up to the Nyquist frequency; pass the movement segment only.

    Args:
        speed_profile: Speed ``(N,)`` of the movement (onset to offset).
        dt: Sample interval (s).
        padlevel: Zero padding to ``2^(ceil(log2 N) + padlevel)`` samples.
        max_cutoff: Maximum cutoff frequency (Hz).
        amplitude_threshold: Normalised magnitude threshold of the adaptive cutoff.

    Returns:
        SPARC; NaN for fewer than 2 samples or a zero profile.
    """
    _check_dt(dt)
    s = np.asarray(speed_profile, dtype=float)
    if s.size < 2:
        return math.nan
    nfft = int(2 ** (math.ceil(math.log2(s.size)) + padlevel))
    freqs = np.fft.rfftfreq(nfft, d=dt)
    magnitude = np.abs(np.fft.rfft(s, nfft))
    if not magnitude.max() > 0:
        return math.nan
    magnitude /= magnitude.max()  # = V(0) for a non-negative speed profile
    keep = freqs <= max_cutoff
    freqs, magnitude = freqs[keep], magnitude[keep]
    above = np.flatnonzero(magnitude >= amplitude_threshold)
    band = slice(above[0], above[-1] + 1)
    freqs, magnitude = freqs[band], magnitude[band]
    if freqs.size < 2:
        return math.nan
    df = np.diff(freqs) / (freqs[-1] - freqs[0])
    return float(-np.sum(np.sqrt(df**2 + np.diff(magnitude) ** 2)))


# ── Fitts' law ───────────────────────────────────────────────────────────────


def index_of_difficulty(distance: ArrayLike, width: ArrayLike) -> np.ndarray:
    """Shannon index of difficulty ``ID = log2(D / W + 1)`` in bits.

    MacKenzie (1992); MacKenzie (2018), Eq. 17.6. Fitts (1954) used ``log2(2D / W)``.

    Args:
        distance: Movement distance(s) ``D``.
        width: Target width(s) ``W`` (same units).

    Returns:
        ID (bits), broadcast over the inputs.
    """
    return np.log2(
        np.asarray(distance, dtype=float) / np.asarray(width, dtype=float) + 1.0
    )


def task_axis_projection(
    starts: ArrayLike, targets: ArrayLike, endpoints: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Movement amplitudes and endpoint deviations along each trial's task axis.

    With ``u`` the unit vector from start to target, the endpoint deviation is
    ``dx = (endpoint - target) . u`` (signed overshoot) and the effective amplitude
    ``ae = |target - start| + dx`` (MacKenzie 2018: ``dx = (c^2 - b^2 - a^2) / 2a``,
    the same projection; Soukoreff & MacKenzie 2004; ISO 9241-411).

    Args:
        starts: Start positions, ``(n,)`` (1-D task) or ``(n, d)``.
        targets: Target centres, broadcastable to ``starts`` (e.g. ``(1, d)``).
        endpoints: Selection/end points, broadcastable to ``starts``.

    Returns:
        ``(ae, dx)``, each ``(n,)``.
    """
    s, t, e = np.broadcast_arrays(
        *(np.asarray(a, dtype=float) for a in (starts, targets, endpoints))
    )
    if s.ndim == 1:
        s, t, e = s[:, None], t[:, None], e[:, None]
    axis = t - s
    amplitude = np.linalg.norm(axis, axis=-1)
    if np.any(amplitude == 0):
        raise ValueError("every trial needs a start different from its target")
    dx = np.sum((e - t) * axis, axis=-1) / amplitude
    return amplitude + dx, dx


def effective_width(deviations: ArrayLike) -> float:
    """Effective width ``We = 4.133 * SD`` of endpoint deviations along the task axis.

    Soukoreff & MacKenzie (2004); ISO 9241-411; the sample SD (``ddof=1``) is used.

    Args:
        deviations: Endpoint deviations ``dx`` of one condition (``(n,)``, n >= 2).

    Returns:
        We (position units).
    """
    dx = np.asarray(deviations, dtype=float)
    if dx.size < 2:
        raise ValueError("effective width needs at least 2 trials")
    return EFFECTIVE_WIDTH_FACTOR * float(np.std(dx, ddof=1))


def effective_parameters(
    starts: ArrayLike, targets: ArrayLike, endpoints: ArrayLike
) -> EffectiveParameters:
    """Effective distance, width and index of difficulty of one D x W condition.

    ``De = mean(ae)``, ``We = 4.133 * SD(dx)``, ``IDe = log2(De / We + 1)``
    (Soukoreff & MacKenzie 2004; ISO 9241-411; MacKenzie 2018).

    Args:
        starts: Start positions, ``(n,)`` or ``(n, d)``.
        targets: Target centres, broadcastable to ``starts``.
        endpoints: Selection/end points, broadcastable to ``starts``.

    Returns:
        De, We and IDe.
    """
    ae, dx = task_axis_projection(starts, targets, endpoints)
    de = float(ae.mean())
    we = effective_width(dx)
    return EffectiveParameters(de, we, float(index_of_difficulty(de, we)))


def throughput(effective_ids: ArrayLike, movement_times: ArrayLike) -> float:
    """Throughput ``TP = mean(IDe / MT)`` in bits/s, the mean over conditions.

    Each condition's (or sequence's) ``IDe / MT`` is averaged: the "mean of means"
    of Soukoreff & MacKenzie (2004) and ISO 9241-411 (MacKenzie 2018, Eq. 17.10).
    Pass a ``(participants, conditions)`` array for the grand mean.

    Args:
        effective_ids: IDe per condition (bits).
        movement_times: Mean movement time per condition (s), same shape.

    Returns:
        Throughput (bits/s).
    """
    ide = np.asarray(effective_ids, dtype=float)
    mt = np.asarray(movement_times, dtype=float)
    if ide.shape != mt.shape:
        raise ValueError(f"shape mismatch: {ide.shape} vs {mt.shape}")
    return float(np.mean(ide / mt))


def fitts_regression(
    index_of_difficulty_values: ArrayLike, movement_times: ArrayLike
) -> FittsRegression:
    """Fitts' law regression ``MT = a + b * ID`` (Fitts 1954; MacKenzie 1992).

    Ordinary least squares via ``scipy.stats.linregress``.

    Args:
        index_of_difficulty_values: ID or IDe per condition (bits).
        movement_times: Movement time per condition (s).

    Returns:
        Intercept, slope, R^2, p-value and slope standard error.
    """
    fit = linregress(
        np.asarray(index_of_difficulty_values, dtype=float),
        np.asarray(movement_times, dtype=float),
    )
    return FittsRegression(
        intercept=float(fit.intercept),
        slope=float(fit.slope),
        r_squared=float(fit.rvalue**2),
        p_value=float(fit.pvalue),
        stderr=float(fit.stderr),
    )


# ── Two-thirds power law ─────────────────────────────────────────────────────


def two_thirds_power_law(
    positions: ArrayLike,
    dt: float,
    *,
    savgol: SavgolParams | None = None,
    min_speed_fraction: float = ONSET_FRACTION,
) -> PowerLawFit:
    """Speed-curvature exponent of a curved movement (Lacquaniti et al. 1983).

    The two-thirds power law ``A = K C^(2/3)`` (angular speed ``A``, curvature ``C``)
    is equivalent to ``v = K kappa^(-1/3)`` for the tangential speed ``v``; ``beta``
    is fitted by ``scipy.stats.linregress`` of ``ln v`` on ``ln kappa``. Curvature is
    ``kappa = sqrt(|v|^2 |a|^2 - (v . a)^2) / |v|^3`` (any dimension). Samples slower
    than ``min_speed_fraction`` of peak speed or with zero curvature are excluded.

    Args:
        positions: ``(N, d)`` positions with ``d >= 2``.
        dt: Sample interval (s).
        savgol: Optional Savitzky-Golay smoothing (``polyorder >= 2``).
        min_speed_fraction: Speed cut-off relative to peak speed.

    Returns:
        Exponent ``beta`` (-1/3 for the law), gain ``K`` and R^2.
    """
    x = _as_2d(positions)
    if x.shape[1] < 2:
        raise ValueError("the power law needs at least 2-D positions")
    vel = derivative(x, dt, 1, savgol)
    acc = derivative(x, dt, 2, savgol)
    v = np.linalg.norm(vel, axis=-1)
    cross_sq = np.clip(
        v**2 * np.sum(acc**2, axis=-1) - np.sum(vel * acc, axis=-1) ** 2, 0.0, None
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.sqrt(cross_sq) / v**3
    keep = (v > min_speed_fraction * v.max()) & (kappa > 0) & np.isfinite(kappa)
    if keep.sum() < 3:
        raise ValueError("too few curved, moving samples for the power-law fit")
    fit = linregress(np.log(kappa[keep]), np.log(v[keep]))
    return PowerLawFit(
        float(fit.slope), float(np.exp(fit.intercept)), float(fit.rvalue**2)
    )


# ── Trial summaries ──────────────────────────────────────────────────────────


def _point_kinematics(
    x: np.ndarray, dt: float, onset_fraction: float, savgol: SavgolParams | None
) -> dict[str, float]:
    """Speed-profile and smoothness metrics of one point's trajectory ``(N, d)``."""
    metrics = {
        "movement_time": math.nan,
        "peak_speed": math.nan,
        "time_to_peak_ratio": math.nan,
        "speed_peaks": math.nan,
        "ldlj": math.nan,
        "sparc": math.nan,
        "straightness": math.nan,
    }
    if len(x) < 2:  # e.g. an episode that terminated at its first step
        return metrics
    v = speed(x, dt, savgol)
    bounds = movement_bounds(v, dt, onset_fraction)
    metrics.update(peak_speed=float(v.max()), speed_peaks=0.0)
    if bounds is None:
        return metrics
    i0, i1 = bounds.onset_index, bounds.offset_index
    metrics.update(
        movement_time=bounds.movement_time,
        time_to_peak_ratio=time_to_peak_ratio(v, dt, bounds),
        speed_peaks=float(
            count_speed_peaks(v[i0 : i1 + 1], onset_fraction, onset_fraction)
        ),
        ldlj=log_dimensionless_jerk(x, dt, savgol=savgol, segment=(i0, i1)),
        sparc=sparc(v[i0 : i1 + 1], dt),
        straightness=straightness(x[i0 : i1 + 1]),
    )
    return metrics


def point_to_point_metrics(
    positions: ArrayLike,
    targets: ArrayLike,
    dt: float,
    radius: float,
    *,
    activations: ArrayLike | None = None,
    onset_fraction: float = ONSET_FRACTION,
    dwell_time: float | None = None,
    savgol: SavgolParams | None = None,
) -> dict[str, float]:
    """Accuracy, timing, kinematic, smoothness and effort metrics of one trial.

    Accuracy and timing use the distance over all coordinates,
    ``||targets - positions||`` (the reach/pose "solved" distance), and
    ``inside = distance < radius``. Speed-profile and smoothness metrics are computed
    per point (e.g. per fingertip) on its movement segment (onset to offset) and
    averaged over points.

    Keys: ``success`` (inside at the last sample), ``final_error``,
    ``time_to_target`` and ``time_to_acquire`` (:func:`target_acquisition`, s from the
    first sample), ``target_entries``, ``movement_time`` (s, :func:`movement_bounds`),
    ``peak_speed``, ``time_to_peak_ratio``, ``speed_peaks``, ``ldlj`` (peak-speed
    normalisation), ``sparc``, ``straightness`` and ``effort``: the mean squared
    activation over samples and muscles, the cost of squared-activation effort models
    (e.g. Ackermann & van den Bogert 2010); NaN without ``activations``. Undefined
    values (never inside, no movement) are NaN.

    Args:
        positions: ``(N, d)`` for one point or ``(N, k, d)`` for ``k`` points.
        targets: Target positions broadcastable to ``positions`` (fixed or per sample).
        dt: Sample interval (s).
        radius: Target radius on the combined distance.
        activations: Optional muscle activations ``(N, n_muscles)``.
        onset_fraction: Speed threshold of :func:`movement_bounds`.
        dwell_time: Dwell of :func:`target_acquisition` (``None``: reached and held).
        savgol: Optional Savitzky-Golay smoothing, see :func:`derivative`.

    Returns:
        Metric name to value.
    """
    p = np.asarray(positions, dtype=float)
    if p.ndim not in (2, 3):
        raise ValueError(f"positions must be (N, d) or (N, k, d), got {p.shape}")
    tgt = np.broadcast_to(np.asarray(targets, dtype=float), p.shape)
    if p.ndim == 2:
        p, tgt = p[:, None, :], tgt[:, None, :]
    n = len(p)
    distance = np.linalg.norm((tgt - p).reshape(n, -1), axis=-1)
    acquisition = target_acquisition(distance < radius, dt, dwell_time)
    points = [
        _point_kinematics(p[:, j], dt, onset_fraction, savgol)
        for j in range(p.shape[1])
    ]
    metrics = {
        "success": float(distance[-1] < radius),
        "final_error": float(distance[-1]),
        "time_to_target": acquisition.entry_time,
        "time_to_acquire": acquisition.acquisition_time,
        "target_entries": float(acquisition.entries),
    }
    for key in points[0]:
        metrics[key] = _finite_mean([m[key] for m in points])
    act = None if activations is None else np.asarray(activations, dtype=float)
    metrics["effort"] = (
        float(np.mean(act**2)) if act is not None and act.size else math.nan
    )
    return metrics


def mean_metrics(per_trial: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """Average per-trial metrics key by key, ignoring undefined (NaN) values.

    E.g. ``time_to_target`` is the mean over the trials that entered the target.

    Args:
        per_trial: One metric dict per trial (same keys).

    Returns:
        Mean of the finite values per key (NaN if a key has none).
    """
    trials = list(per_trial)
    if not trials:
        raise ValueError("no trials to average")
    return {key: _finite_mean([t[key] for t in trials]) for key in trials[0]}
