# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Discrete Fréchet alignment and multi-quantity rollout comparison."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from myosuite.biomechanics.kinematic_metrics import (
    EntityTrajectoryMetrics,
    FrechetAlignment,
    compare_rollout_trajectories,
    discrete_frechet_alignment,
    discrete_frechet_distance,
    energy_distance,
)

pytestmark = pytest.mark.tier1


def _naive_frechet_distance(x: np.ndarray, y: np.ndarray) -> float:
    d = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
    t1, t2 = d.shape
    ca = np.empty((t1, t2), dtype=np.float64)
    ca[0, 0] = d[0, 0]
    for i in range(1, t1):
        ca[i, 0] = max(ca[i - 1, 0], d[i, 0])
    for j in range(1, t2):
        ca[0, j] = max(ca[0, j - 1], d[0, j])
    for i in range(1, t1):
        for j in range(1, t2):
            ca[i, j] = max(d[i, j], min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]))
    return float(ca[-1, -1])


def _assert_valid_coupling(
    alignment: FrechetAlignment, x: np.ndarray, y: np.ndarray
) -> None:
    coupling = alignment.coupling
    assert coupling.ndim == 2
    assert coupling.shape[1] == 2
    assert coupling.dtype == np.int64
    assert tuple(coupling[0]) == (0, 0)
    assert tuple(coupling[-1]) == (x.shape[0] - 1, y.shape[0] - 1)

    if coupling.shape[0] > 1:
        steps = np.diff(coupling, axis=0)
        allowed = {(1, 0), (0, 1), (1, 1)}
        assert {tuple(row) for row in steps}.issubset(allowed)

    x_2d = x[:, None] if x.ndim == 1 else x
    y_2d = y[:, None] if y.ndim == 1 else y
    paired_dists = np.linalg.norm(x_2d[coupling[:, 0]] - y_2d[coupling[:, 1]], axis=-1)
    assert float(np.max(paired_dists)) == pytest.approx(alignment.distance, abs=1e-12)


def test_identical_trajectories_zero_distance_and_diagonal_coupling() -> None:
    t = np.linspace(0.0, 1.0, 40)
    traj_3d = np.column_stack([np.cos(2.0 * t), np.sin(2.0 * t), 0.3 * t])
    # Include a stationary plateau to verify diagonal tie-breaking preference.
    traj_3d[15:22] = traj_3d[15]

    align = discrete_frechet_alignment(traj_3d, traj_3d.copy())
    assert isinstance(align, FrechetAlignment)
    assert align.distance == 0.0
    assert discrete_frechet_distance(traj_3d, traj_3d) == 0.0
    expected_diag = np.column_stack(
        [np.arange(traj_3d.shape[0]), np.arange(traj_3d.shape[0])]
    )
    assert np.array_equal(align.coupling, expected_diag)
    _assert_valid_coupling(align, traj_3d, traj_3d)

    traj_1d = np.sin(np.linspace(0.0, np.pi, 18))
    align_1d = discrete_frechet_alignment(traj_1d, traj_1d)
    assert align_1d.distance == 0.0
    assert align_1d.coupling.shape == (18, 2)

    single = np.array([[1.2, -0.4, 0.8]])
    align_single = discrete_frechet_alignment(single, single)
    assert align_single.distance == 0.0
    assert np.array_equal(align_single.coupling, np.array([[0, 0]], dtype=np.int64))

    with pytest.raises(FrozenInstanceError):
        align.distance = 1.0  # type: ignore[misc]


def test_agreement_with_reference_dp_and_temporal_order_sensitivity() -> None:
    rng = np.random.default_rng(17)
    for t1, t2, dim in [(1, 1, 2), (1, 9, 3), (8, 1, 3), (12, 19, 1), (25, 17, 6)]:
        x = rng.normal(size=(t1, dim))
        y = rng.normal(size=(t2, dim))
        align = discrete_frechet_alignment(x, y)
        dist = discrete_frechet_distance(x, y)
        expected = _naive_frechet_distance(x, y)

        assert dist == align.distance
        assert dist == pytest.approx(expected, abs=1e-12)
        _assert_valid_coupling(align, x, y)

    # Reversed traversal occupies identical spatial points (Hausdorff = 0)
    # but incurs full diameter bottleneck under monotonic Fréchet coupling.
    forward = np.column_stack([np.linspace(0.0, 2.0, 11), np.zeros(11)])
    backward = forward[::-1].copy()
    assert discrete_frechet_distance(forward, backward) == pytest.approx(2.0)


def test_time_warped_polyline_stays_small_under_frechet_vs_interpolated_euclidean() -> (
    None
):
    u_ref = np.linspace(0.0, 1.0, 120)
    u_warped = np.linspace(0.0, 1.0, 185) ** 3

    def _arc(u: np.ndarray) -> np.ndarray:
        return np.column_stack([np.cos(np.pi * u), np.sin(np.pi * u), 0.5 * u])

    ref_curve = _arc(u_ref)
    warped_curve = _arc(u_warped)

    summary = compare_rollout_trajectories(
        {"ee_pos": warped_curve}, {"ee_pos": ref_curve}
    )["ee_pos"]

    assert np.isnan(summary.euclidean_mean)
    assert summary.discrete_frechet < 0.03
    assert summary.interpolated_euclidean_mean > 0.70
    assert summary.interpolated_euclidean_mean > 25.0 * summary.discrete_frechet

    # Exact discrete vertex stuttering yields zero Discrete Fréchet distance.
    repeats = np.array([1, 4, 1, 1, 5, 2, 1, 3, 1, 4])
    base_poly = _arc(np.linspace(0.0, 1.0, repeats.size))
    stuttered_poly = np.repeat(base_poly, repeats, axis=0)
    stuttered_align = discrete_frechet_alignment(stuttered_poly, base_poly)
    assert stuttered_align.distance == pytest.approx(0.0, abs=1e-12)
    _assert_valid_coupling(stuttered_align, stuttered_poly, base_poly)


def test_symmetry_triangle_inequality_and_zero_start_translation() -> None:
    rng = np.random.default_rng(99)
    for _ in range(10):
        x = rng.normal(size=(14, 4))
        y = rng.normal(size=(21, 4))
        z = rng.normal(size=(17, 4))

        d_xy = discrete_frechet_distance(x, y)
        d_yx = discrete_frechet_distance(y, x)
        d_yz = discrete_frechet_distance(y, z)
        d_xz = discrete_frechet_distance(x, z)

        assert d_xy == pytest.approx(d_yx, abs=1e-13)
        assert d_xz <= d_xy + d_yz + 1e-12

    # When Y is an exact discrete reparameterization of X (d_F(X, Y) == 0),
    # d_F(X, Z) == d_F(Y, Z) <= max(d_F(X, Y), d_F(Y, Z)).
    x_base = rng.normal(size=(8, 3))
    y_reparam = np.repeat(x_base, [2, 1, 3, 1, 2, 1, 4, 1], axis=0)
    z_other = rng.normal(size=(11, 3))
    assert discrete_frechet_distance(x_base, y_reparam) == pytest.approx(0.0, abs=1e-12)
    assert discrete_frechet_distance(x_base, z_other) == pytest.approx(
        discrete_frechet_distance(y_reparam, z_other), abs=1e-12
    )

    offset = np.array([3.0, -4.0, 12.0])
    shifted = x_base + offset
    assert discrete_frechet_distance(shifted, x_base) == pytest.approx(13.0)
    align_zero = discrete_frechet_alignment(shifted, x_base, zero_start=True)
    assert align_zero.distance == pytest.approx(0.0, abs=1e-12)
    assert discrete_frechet_distance(shifted, x_base, zero_start=True) == pytest.approx(
        0.0, abs=1e-12
    )
    assert align_zero.coupling.shape == (x_base.shape[0], 2)


def test_input_validation_errors() -> None:
    valid = np.ones((6, 3))

    for bad in [
        np.empty((0,)),
        np.empty((0, 3)),
        np.empty((6, 0)),
        np.array(1.0),
        np.ones((2, 3, 4)),
    ]:
        with pytest.raises(ValueError):
            discrete_frechet_distance(bad, valid)
        with pytest.raises(ValueError):
            discrete_frechet_alignment(valid, bad)

    with pytest.raises(ValueError, match="dimension mismatch"):
        discrete_frechet_distance(np.ones((6, 3)), np.ones((6, 2)))
    with pytest.raises(ValueError, match="dimension mismatch"):
        discrete_frechet_alignment(np.ones((6,)), np.ones((6, 2)))

    for non_finite in [np.nan, np.inf, -np.inf]:
        corrupted = valid.copy()
        corrupted[2, 1] = non_finite
        with pytest.raises(ValueError, match="finite"):
            discrete_frechet_distance(corrupted, valid, zero_start=True)
        with pytest.raises(ValueError, match="finite"):
            discrete_frechet_alignment(valid, corrupted)

    with pytest.raises(ValueError, match="non-empty"):
        compare_rollout_trajectories({}, {})
    with pytest.raises(ValueError, match="keys must match"):
        compare_rollout_trajectories(
            {"qpos": valid, "qvel": valid},
            {"qpos": valid, "site_rpos": valid},
        )
    with pytest.raises(ValueError, match="site_rpos"):
        compare_rollout_trajectories(
            {"site_rpos": np.ones((6, 3))},
            {"site_rpos": np.ones((6, 4))},
        )


def test_compare_rollout_trajectories_equal_and_unequal_lengths() -> None:
    t_eq = np.linspace(0.0, 1.0, 60)
    ref_eq = {
        "qpos": np.column_stack([np.sin(2.0 * np.pi * t_eq), 0.5 * np.cos(t_eq)]),
        "site_rpos": np.column_stack([t_eq, 0.2 * t_eq**2, np.zeros_like(t_eq)]),
        "qvel": 2.0 * np.pi * np.cos(2.0 * np.pi * t_eq),
    }
    rollout_eq = {
        "qpos": ref_eq["qpos"] + np.array([0.1, 0.0]),
        "site_rpos": ref_eq["site_rpos"] + np.array([5.0, -3.0, 1.5]),
        "qvel": ref_eq["qvel"].copy(),
    }

    custom = {
        "terminal_gap": lambda a, b: float(np.linalg.norm(a[-1] - b[-1])),
        "energy": energy_distance,
    }
    res_eq = compare_rollout_trajectories(
        rollout_eq,
        ref_eq,
        zero_start_keys=("site_rpos",),
        custom_measures=custom,
    )

    assert list(res_eq.keys()) == ["qpos", "site_rpos", "qvel"]
    assert isinstance(res_eq["qpos"], EntityTrajectoryMetrics)
    assert res_eq["qpos"].euclidean_mean == pytest.approx(0.1)
    assert res_eq["qpos"].interpolated_euclidean_mean == res_eq["qpos"].euclidean_mean
    assert res_eq["qpos"].discrete_frechet == pytest.approx(0.1)
    assert res_eq["qpos"].extra["terminal_gap"] == pytest.approx(0.1)

    # Zero-started site_rpos cancels pure spatial translation across all measures.
    assert res_eq["site_rpos"].euclidean_mean == pytest.approx(0.0, abs=1e-12)
    assert res_eq["site_rpos"].interpolated_euclidean_mean == pytest.approx(
        0.0, abs=1e-12
    )
    assert res_eq["site_rpos"].discrete_frechet == pytest.approx(0.0, abs=1e-12)
    assert res_eq["site_rpos"].extra["terminal_gap"] == pytest.approx(0.0, abs=1e-12)
    assert res_eq["site_rpos"].extra["energy"] == pytest.approx(0.0, abs=1e-7)

    assert res_eq["qvel"].discrete_frechet == pytest.approx(0.0, abs=1e-12)

    # Unequal horizon lengths (T1 = 95 vs T2 = 60) + single-sample rollout edge case.
    t_roll = np.linspace(0.0, 1.0, 95)
    rollout_uneq = {
        "qpos": np.column_stack([np.sin(2.0 * np.pi * t_roll), 0.5 * np.cos(t_roll)]),
        "site_rpos": np.array([[2.0, 2.0, 2.0]]),
        "qvel": 2.0 * np.pi * np.cos(2.0 * np.pi * t_roll),
    }
    res_uneq = compare_rollout_trajectories(
        rollout_uneq,
        ref_eq,
        zero_start_keys=("site_rpos",),
        custom_measures={"terminal_gap": custom["terminal_gap"]},
    )
    for key in ("qpos", "site_rpos", "qvel"):
        assert np.isnan(res_uneq[key].euclidean_mean)
        assert np.isfinite(res_uneq[key].interpolated_euclidean_mean)
        assert np.isfinite(res_uneq[key].discrete_frechet)

    assert res_uneq["qpos"].interpolated_euclidean_mean < 5e-4
    assert res_uneq["qpos"].discrete_frechet < 0.06
    # Single-sample zero-started rollout vs reference starting at origin:
    expected_rpos_norms = np.linalg.norm(
        ref_eq["site_rpos"] - ref_eq["site_rpos"][:1], axis=-1
    )
    assert res_uneq["site_rpos"].interpolated_euclidean_mean == pytest.approx(
        float(np.mean(expected_rpos_norms))
    )
    assert res_uneq["site_rpos"].discrete_frechet == pytest.approx(
        float(np.max(expected_rpos_norms))
    )
