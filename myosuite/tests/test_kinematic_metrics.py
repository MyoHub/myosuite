# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for kinematic trajectory smoothness, power-law, Fitts, and cohort metrics."""

from __future__ import annotations

import numpy as np
import pytest

from myosuite.biomechanics.kinematic_metrics import (
    CohortComparison,
    KinematicTrialSummary,
    compare_against_reference_cohorts,
    dimensionless_squared_jerk,
    endpoint_error,
    energy_distance,
    fitts_index_of_difficulty,
    fitts_throughput,
    movement_time,
    spectral_arc_length,
    summarize_kinematic_trial,
    two_thirds_power_law_r2,
    velocity_bell_symmetry,
)

pytestmark = pytest.mark.tier1


def _min_jerk_reach(
    amplitude: float = 0.35, duration: float = 0.8, dt: float = 0.002
) -> np.ndarray:
    t = np.arange(0.0, duration + 0.5 * dt, dt)
    tau = np.clip(t / duration, 0.0, 1.0)
    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    return np.column_stack([amplitude * s, 0.5 * amplitude * s, np.zeros_like(s)])


def test_minimum_jerk_dsj_sparc_and_symmetry_invariants() -> None:
    dt = 0.0005
    reach_a = _min_jerk_reach(amplitude=0.25, duration=0.6, dt=dt)
    reach_b = _min_jerk_reach(amplitude=0.75, duration=1.4, dt=dt)

    dsj_a = dimensionless_squared_jerk(reach_a, dt)
    dsj_b = dimensionless_squared_jerk(reach_b, dt)
    assert dsj_a == pytest.approx(360.0, rel=1.5e-2)
    assert dsj_b == pytest.approx(dsj_a, rel=1e-2)

    speeds_a = np.linalg.norm(np.gradient(reach_a, dt, axis=0), axis=-1)
    sparc_smooth = spectral_arc_length(speeds_a, dt)
    assert -2.0 < sparc_smooth < -1.0
    assert velocity_bell_symmetry(speeds_a) == pytest.approx(0.5, abs=0.02)

    t = np.arange(reach_a.shape[0]) * dt
    jerky = reach_a.copy()
    jerky[:, 0] += 0.025 * np.sin(2.0 * np.pi * 7.5 * t)
    speeds_jerky = np.linalg.norm(np.gradient(jerky, dt, axis=0), axis=-1)
    assert dimensionless_squared_jerk(jerky, dt) > 5.0 * dsj_a
    assert spectral_arc_length(speeds_jerky, dt) < sparc_smooth - 0.5


def test_two_thirds_power_law_distinguishes_biological_from_constant_speed_ellipse() -> (
    None
):
    dt = 0.005
    a, b = 0.25, 0.07
    # Construct parameter u(t) so tangential speed scales as kappa^(-1/3).
    u_grid = np.linspace(0.0, 2.0 * np.pi, 4000)
    dx_du = -a * np.sin(u_grid)
    dy_du = b * np.cos(u_grid)
    ds_du = np.hypot(dx_du, dy_du)
    kappa = (a * b) / (ds_du**3)
    target_speed = 0.35 * (kappa ** (-1.0 / 3.0))
    du_dt = target_speed / ds_du
    t_of_u = np.concatenate(
        [[0.0], np.cumsum(0.5 * (1.0 / du_dt[:-1] + 1.0 / du_dt[1:]) * np.diff(u_grid))]
    )
    t_samples = np.arange(0.0, float(t_of_u[-1]), dt)
    u_bio = np.interp(t_samples, t_of_u, u_grid)
    pos_bio = np.column_stack([a * np.cos(u_bio), b * np.sin(u_bio)])

    # Constant tangential speed along the same ellipse violates v ~ kappa^(-1/3).
    s_of_u = np.concatenate(
        [[0.0], np.cumsum(0.5 * (ds_du[:-1] + ds_du[1:]) * np.diff(u_grid))]
    )
    s_samples = np.linspace(0.0, float(s_of_u[-1]), t_samples.size)
    u_const = np.interp(s_samples, s_of_u, u_grid)
    pos_const = np.column_stack([a * np.cos(u_const), b * np.sin(u_const)])

    r2_bio = two_thirds_power_law_r2(pos_bio, dt)
    r2_const = two_thirds_power_law_r2(pos_const, dt)
    assert r2_bio > 0.92
    assert r2_const < 0.25


def test_fitts_shannon_throughput_and_trial_summary() -> None:
    assert fitts_index_of_difficulty(0.30, 0.10) == pytest.approx(2.0)
    assert fitts_throughput(0.30, 0.10, 0.50) == pytest.approx(4.0)

    dt = 0.005
    reach = _min_jerk_reach(amplitude=0.30, duration=0.60, dt=dt)
    target = reach[-1].copy()
    summary = summarize_kinematic_trial(reach, dt, target, target_width=0.04)
    assert isinstance(summary, KinematicTrialSummary)
    vec = summary.to_feature_vector()
    assert vec.shape == (len(KinematicTrialSummary.FEATURE_NAMES),)
    assert endpoint_error(reach, target) == pytest.approx(0.0, abs=1e-12)
    assert (
        movement_time(np.linalg.norm(np.gradient(reach, dt, axis=0), axis=-1), dt) > 0.4
    )

    with pytest.raises(ValueError):
        fitts_index_of_difficulty(0.0, 0.05)
    with pytest.raises(ValueError):
        fitts_throughput(0.3, 0.05, -0.1)


def test_compare_against_reference_cohorts_split_half_ceiling() -> None:
    rng = np.random.default_rng(42)
    cohorts = [rng.normal(loc=0.05 * i, scale=1.0, size=(25, 5)) for i in range(-3, 3)]
    in_dist = rng.normal(loc=0.0, scale=1.0, size=(30, 5))
    out_dist = rng.normal(loc=6.0, scale=1.5, size=(30, 5))

    comp_in = compare_against_reference_cohorts(in_dist, cohorts, n_splits=60, rng=7)
    comp_in_repeat = compare_against_reference_cohorts(
        in_dist, cohorts, n_splits=60, rng=7
    )
    comp_out = compare_against_reference_cohorts(out_dist, cohorts, n_splits=60, rng=7)

    assert isinstance(comp_in, CohortComparison)
    assert comp_in.raw_distance == pytest.approx(comp_in_repeat.raw_distance)
    assert comp_in.ceiling_normalized_score > 0.70
    assert comp_out.ceiling_normalized_score < 0.30
    assert np.all(comp_out.per_feature_wasserstein > comp_in.per_feature_wasserstein)
    assert energy_distance(in_dist, in_dist) == pytest.approx(0.0, abs=1e-12)

    with pytest.raises(ValueError, match="at least 2 cohorts"):
        compare_against_reference_cohorts(in_dist, [cohorts[0]])
