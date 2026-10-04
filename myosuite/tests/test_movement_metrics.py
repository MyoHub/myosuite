"""Validation of the movement-quality metrics against analytic and published values.

Analytic minimum-jerk properties (Flash & Hogan 1985), the reference SPARC/LDLJ code
of Balasubramanian et al. (2015), Fitts' law quantities with known answers, and the
two-thirds power law on a harmonic ellipse. No RL; synthetic trajectories only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid, quad

from myosuite.utils import movement_metrics as mm
from myosuite.utils.path_utils import path_obs_series

pytestmark = pytest.mark.tier1

# Minimum-jerk onset at 5% of peak speed: 30 tau^2 (1 - tau)^2 = 0.05 * 1.875.
_TAU_ONSET = (1 - math.sqrt(1 - 4 * math.sqrt(0.05 * 1.875 / 30))) / 2


def _min_jerk_samples(n: int = 1001, duration: float = 1.0, distance: float = 0.3):
    t = np.linspace(0.0, duration, n)
    return t, t[1] - t[0], mm.minimum_jerk(0.0, distance, duration, t)


# ── Minimum jerk ─────────────────────────────────────────────────────────────


def test_minimum_jerk_boundary_conditions_and_peak():
    start, end = np.array([0.1, -0.2, 0.3]), np.array([0.4, 0.2, 0.3])
    t = np.array([-0.1, 0.0, 0.25, 0.5, 1.0, 1.2])
    mj = mm.minimum_jerk(start, end, 0.5, t)
    np.testing.assert_allclose(mj.position[[0, 1]], [start, start])
    np.testing.assert_allclose(mj.position[[4, 5]], [end, end])
    np.testing.assert_allclose(mj.velocity[[1, 4]], 0.0, atol=1e-12)
    np.testing.assert_allclose(mj.acceleration[[1, 4]], 0.0, atol=1e-12)
    np.testing.assert_allclose(mj.position[2], (start + end) / 2)  # symmetric
    peak = np.linalg.norm(mj.velocity[2])
    assert peak == pytest.approx(1.875 * np.linalg.norm(end - start) / 0.5)


def test_minimum_jerk_sampled_peak_speed_and_symmetry():
    _, dt, mj = _min_jerk_samples()
    v = mm.speed(mj.position, dt)
    assert v.max() == pytest.approx(mm.MIN_JERK_PEAK_SPEED_FACTOR * 0.3 / 1.0, rel=1e-9)
    bounds = mm.movement_bounds(v, dt)
    assert mm.time_to_peak_ratio(v, dt, bounds) == pytest.approx(0.5, abs=1e-9)


def test_minimum_jerk_squared_jerk_constant_is_720():
    integral, _ = quad(lambda tau: (60 - 360 * tau + 360 * tau**2) ** 2, 0, 1)
    assert integral == pytest.approx(720.0, rel=1e-12)


@pytest.mark.parametrize("n", [101, 1001])
def test_minimum_jerk_dimensionless_jerk_and_ldlj(n):
    _, dt, mj = _min_jerk_samples(n)
    rel = 1e-6 if n == 1001 else 1e-5
    dj_amp = mm.dimensionless_jerk(mj.position, dt, normalization="amplitude")
    dj_peak = mm.dimensionless_jerk(mj.position, dt, normalization="peak_speed")
    assert dj_amp == pytest.approx(720.0, rel=rel)
    assert dj_peak == pytest.approx(720.0 / 1.875**2, rel=rel)  # 204.8
    assert mm.log_dimensionless_jerk(mj.position, dt) == pytest.approx(
        -math.log(204.8), abs=rel
    )
    assert mm.log_dimensionless_jerk(
        mj.position, dt, normalization="amplitude"
    ) == pytest.approx(-math.log(720.0), abs=rel)


def test_dimensionless_jerk_is_scale_and_direction_invariant():
    values = []
    for distance, duration, direction in [
        (0.05, 0.4, [1.0, 0.0, 0.0]),
        (2.0, 3.0, [0.3, -0.5, 0.8]),
    ]:
        t = np.linspace(0.0, duration, 401)
        end = distance * np.asarray(direction) / np.linalg.norm(direction)
        mj = mm.minimum_jerk(np.zeros(3), end, duration, t)
        values.append(mm.dimensionless_jerk(mj.position, t[1]))
    assert values[0] == pytest.approx(values[1], rel=1e-6)
    assert values[0] == pytest.approx(204.8, rel=1e-5)


def test_savgol_derivatives_are_exact_for_minimum_jerk():
    _, dt, mj = _min_jerk_samples(201)
    dj = mm.dimensionless_jerk(
        mj.position, dt, normalization="amplitude", savgol=(11, 5)
    )
    assert dj == pytest.approx(720.0, rel=1e-5)


# ── Reference implementation (Balasubramanian et al. 2015) ───────────────────


def test_sparc_and_ldlj_match_reference_code():
    # Doctests of github.com/siva82kb/SPARC scripts/smoothness.py:
    # Gaussian speed profile, fs = 100 Hz -> SPARC -1.41403, LDLJ -5.81636.
    t = np.arange(-1, 1, 0.01)
    move = np.exp(-5 * t**2)
    assert mm.sparc(move, 0.01) == pytest.approx(-1.41403, abs=1e-5)
    position = cumulative_trapezoid(move, dx=0.01, initial=0.0)
    ldlj = mm.log_dimensionless_jerk(position, 0.01)
    # The reference uses T = N dt; this module T = (N - 1) dt.
    n = len(move)
    assert ldlj - 3 * math.log(n / (n - 1)) == pytest.approx(-5.81636, abs=1e-3)


def test_smoothness_drops_with_a_second_submovement():
    t = np.arange(0.0, 2.0, 0.01)
    one = mm.minimum_jerk(0.0, 0.1, 1.0, t).position
    two = one + mm.minimum_jerk(0.0, 0.1, 1.0, t - 0.7).position
    v_one, v_two = mm.speed(one, 0.01), mm.speed(two, 0.01)
    assert mm.count_speed_peaks(v_one) == 1
    assert mm.count_speed_peaks(v_two) == 2
    assert mm.log_dimensionless_jerk(two, 0.01) < mm.log_dimensionless_jerk(one, 0.01)
    assert mm.sparc(v_two, 0.01) < mm.sparc(v_one, 0.01)


def test_sparc_is_amplitude_invariant():
    _, dt, mj = _min_jerk_samples(501)
    v = mm.speed(mj.position, dt)
    assert mm.sparc(3.7 * v, dt) == pytest.approx(mm.sparc(v, dt), abs=1e-12)


# ── Kinematics and segmentation ──────────────────────────────────────────────


def test_onset_offset_and_movement_time_noise_free():
    _, dt, mj = _min_jerk_samples()
    bounds = mm.movement_bounds(mm.speed(mj.position, dt), dt)
    assert bounds.onset_time == pytest.approx(_TAU_ONSET, abs=1e-5)
    assert bounds.offset_time == pytest.approx(1 - _TAU_ONSET, abs=1e-5)
    assert bounds.movement_time == pytest.approx(1 - 2 * _TAU_ONSET, abs=2e-5)


def test_speed_and_onset_detection_on_noisy_minimum_jerk():
    rng = np.random.default_rng(0)
    dt, start_time, duration = 0.01, 0.25, 1.0
    t = np.arange(0.0, 1.5 + 1e-9, dt)
    end = np.array([0.2, 0.1, 0.0])
    mj = mm.minimum_jerk(np.zeros(3), end, duration, t - start_time)
    noisy = mj.position + rng.normal(0.0, 1e-4, mj.position.shape)
    v = mm.speed(noisy, dt, savgol=(21, 3))
    bounds = mm.movement_bounds(v, dt)
    assert bounds.onset_time == pytest.approx(start_time + _TAU_ONSET, abs=5e-3)
    assert bounds.offset_time == pytest.approx(
        start_time + duration * (1 - _TAU_ONSET), abs=5e-3
    )
    assert v.max() == pytest.approx(1.875 * np.linalg.norm(end) / duration, rel=0.01)


def test_movement_bounds_without_movement():
    assert mm.movement_bounds(np.zeros(10), 0.01) is None


def test_straightness_and_path_length():
    s = np.linspace(0.0, 1.0, 50) ** 2  # non-uniform speed along a line
    line = np.outer(s, [0.3, -0.4, 0.0])
    assert mm.straightness(line) == pytest.approx(1.0, abs=1e-12)
    assert mm.path_length(line) == pytest.approx(0.5)
    theta = np.linspace(0.0, np.pi, 2001)
    semicircle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    assert mm.straightness(semicircle) == pytest.approx(np.pi / 2, rel=1e-6)


def test_target_acquisition_entries_and_dwell():
    inside = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1], dtype=bool)
    acq = mm.target_acquisition(inside, 0.1)
    assert acq.entry_time == pytest.approx(0.2)
    assert acq.acquisition_time == pytest.approx(1.0)  # final stay, held to the end
    assert acq.entries == 3  # two re-entries (TRE = 2)
    dwell = mm.target_acquisition(inside, 0.1, dwell_time=0.3)
    assert dwell.acquisition_time == pytest.approx(0.5)  # first stay of >= 3 samples
    missed = mm.target_acquisition(inside[:-2], 0.1)
    assert math.isnan(missed.acquisition_time) and missed.entries == 2
    never = mm.target_acquisition(np.zeros(5, dtype=bool), 0.1)
    assert math.isnan(never.entry_time) and never.entries == 0


# ── Fitts' law ───────────────────────────────────────────────────────────────


def test_index_of_difficulty_shannon_form():
    np.testing.assert_allclose(mm.index_of_difficulty([3.0, 7.0], 1.0), [2.0, 3.0])


def test_effective_width_of_gaussian_endpoints():
    rng = np.random.default_rng(1)
    n, sd, distance, overshoot = 20000, 0.01, 0.2, 0.002
    angle = rng.uniform(0.0, 2 * np.pi, n)  # multi-directional task axes
    axis = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    normal = np.stack([-axis[:, 1], axis[:, 0]], axis=1)
    starts = rng.normal(0.0, 0.05, (n, 2))
    targets = starts + distance * axis
    along = rng.normal(overshoot, sd, n)
    across = rng.normal(0.0, 5 * sd, n)  # off-axis scatter must not count
    endpoints = targets + along[:, None] * axis + across[:, None] * normal
    eff = mm.effective_parameters(starts, targets, endpoints)
    assert eff.width == pytest.approx(4.133 * sd, rel=0.02)
    assert eff.distance == pytest.approx(distance + overshoot, abs=3 * sd / np.sqrt(n))
    assert eff.index_of_difficulty == pytest.approx(
        math.log2(eff.distance / eff.width + 1)
    )


def test_effective_parameters_hand_computed():
    # 1-D task: start 0, target 10, endpoints 9, 10, 11 -> dx = -1, 0, 1 (SD 1).
    eff = mm.effective_parameters(0.0, 10.0, np.array([9.0, 10.0, 11.0]))
    assert eff.distance == pytest.approx(10.0)
    assert eff.width == pytest.approx(4.133)
    assert eff.index_of_difficulty == pytest.approx(math.log2(10.0 / 4.133 + 1.0))


def test_throughput_hand_computed():
    # IDe / MT = 4, 5, 5 bits/s -> TP = 14 / 3.
    assert mm.throughput([2.0, 3.0, 4.0], [0.5, 0.6, 0.8]) == pytest.approx(14 / 3)


def test_fitts_regression_recovers_known_line():
    ids = mm.index_of_difficulty(
        np.repeat([0.1, 0.2, 0.4], 3), np.tile([0.01, 0.02, 0.04], 3)
    )
    a, b = 0.12, 0.165
    exact = mm.fitts_regression(ids, a + b * ids)
    assert exact.intercept == pytest.approx(a)
    assert exact.slope == pytest.approx(b)
    assert exact.r_squared == pytest.approx(1.0)
    rng = np.random.default_rng(2)
    ids_rep = np.repeat(ids, 200)
    noisy = mm.fitts_regression(
        ids_rep, a + b * ids_rep + rng.normal(0, 0.02, ids_rep.size)
    )
    assert noisy.intercept == pytest.approx(a, abs=0.01)
    assert noisy.slope == pytest.approx(b, abs=0.003)
    assert 0.9 < noisy.r_squared < 1.0


# ── Two-thirds power law ─────────────────────────────────────────────────────


def test_two_thirds_power_law_on_harmonic_ellipse():
    # x = a cos(wt), y = b sin(wt) obeys v = w (ab)^(1/3) kappa^(-1/3) exactly.
    w, a, b = 2 * np.pi, 0.2, 0.1
    t = np.linspace(0.0, 1.0, 2001)
    xy = np.stack([a * np.cos(w * t), b * np.sin(w * t)], axis=1)
    fit = mm.two_thirds_power_law(xy, t[1])
    assert fit.exponent == pytest.approx(-1.0 / 3.0, abs=1e-6)
    assert fit.gain == pytest.approx(w * (a * b) ** (1 / 3), rel=1e-5)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)


# ── Trial summary and path helper ────────────────────────────────────────────


def test_point_to_point_metrics_on_two_point_minimum_jerk():
    dt, duration = 0.01, 0.8
    t = np.arange(0.0, 1.2 + 1e-9, dt)
    starts = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    targets = starts + np.array([[0.2, 0.0, 0.0], [0.0, 0.15, 0.0]])
    positions = np.stack(
        [mm.minimum_jerk(s, g, duration, t).position for s, g in zip(starts, targets)],
        axis=1,
    )
    radius = 0.01
    m = mm.point_to_point_metrics(
        positions, targets, dt, radius, activations=np.full((len(t), 4), 0.5)
    )
    distance = np.linalg.norm((targets - positions).reshape(len(t), -1), axis=-1)
    assert m["success"] == 1.0 and m["final_error"] == pytest.approx(0.0, abs=1e-12)
    assert m["time_to_target"] == pytest.approx(
        np.flatnonzero(distance < radius)[0] * dt
    )
    assert m["target_entries"] == 1.0
    assert m["movement_time"] == pytest.approx(
        duration * (1 - 2 * _TAU_ONSET), abs=1e-3
    )
    assert m["peak_speed"] == pytest.approx(1.875 * 0.175 / duration, rel=1e-3)
    assert m["time_to_peak_ratio"] == pytest.approx(0.5, abs=dt / duration)
    assert m["speed_peaks"] == 1.0
    assert m["straightness"] == pytest.approx(1.0, abs=1e-9)
    assert m["effort"] == pytest.approx(0.25)
    # The 5% segment excludes the smoothest tails, so LDLJ is a little above -5.322.
    assert -5.4 < m["ldlj"] < -4.0
    assert math.isnan(
        mm.point_to_point_metrics(positions, targets, dt, radius)["effort"]
    )


@pytest.mark.parametrize("n", [1, 2, 3])
def test_point_to_point_metrics_on_very_short_trials(n):
    # Episodes can terminate after a step or two; their smoothness is undefined.
    positions = np.linspace(0.0, 0.01, n)[:, None] * [1.0, 0.0, 0.0]
    m = mm.point_to_point_metrics(positions, [0.1, 0.0, 0.0], 0.02, 0.01)
    assert math.isnan(m["ldlj"])
    assert m["final_error"] == pytest.approx(0.1 - positions[-1, 0])


def test_mean_metrics_skips_undefined_values():
    out = mm.mean_metrics(
        [{"a": 1.0, "b": math.nan}, {"a": 3.0, "b": 2.0}, {"a": 2.0, "b": math.nan}]
    )
    assert out == {"a": 2.0, "b": 2.0}
    assert math.isnan(mm.mean_metrics([{"a": math.nan}])["a"])


def _trace_like_path(n_steps: int, repeat_terminal: bool, transform=None):
    """A path laid out like examine_policy's trace (reset obs + post-step infos)."""
    tip = np.arange(n_steps + 1, dtype=float)[:, None] * [0.01, 0.0, 0.0]
    err = 0.5 - tip
    flat = np.concatenate([tip, err], axis=1).astype(np.float32)
    infos = {"obs_dict": {"tip_pos": tip[1:], "reach_err": err[1:]}}
    if repeat_terminal:
        infos = {
            "obs_dict": {
                k: np.concatenate([v, v[-1:]]) for k, v in infos["obs_dict"].items()
            }
        }
    observations = flat if transform is None else transform(flat)
    return {"observations": observations, "env_infos": infos}, tip


@pytest.mark.parametrize("repeat_terminal", [True, False])
def test_path_obs_series_prepends_reset_and_drops_repeated_record(repeat_terminal):
    path, tip = _trace_like_path(6, repeat_terminal)
    series = path_obs_series(path, ("tip_pos", "reach_err", "act"))
    assert set(series) == {"tip_pos", "reach_err"}
    np.testing.assert_allclose(series["tip_pos"], tip, atol=1e-7)


def test_path_obs_series_skips_transformed_observations():
    path, tip = _trace_like_path(6, True, transform=lambda o: (o - o.mean()) / o.std())
    series = path_obs_series(path, ("tip_pos",))
    np.testing.assert_allclose(series["tip_pos"], tip[1:])
