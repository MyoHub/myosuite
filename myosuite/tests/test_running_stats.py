# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for Welford online running stats (myosuite/physics/running_stats.py)."""

from __future__ import annotations

import numpy as np

from myosuite.physics.running_stats import RunningMeanStd, normalize, update

RNG = np.random.default_rng(7)
DIM = 32


# ---------------------------------------------------------------------------
# Welford convergence
# ---------------------------------------------------------------------------


def test_mean_converges():
    stats = RunningMeanStd.zeros(DIM)
    for _ in range(200):
        batch = RNG.standard_normal((50, DIM))
        stats = update(stats, batch)
    assert np.allclose(stats.mean, 0.0, atol=0.05), f"mean off: {stats.mean[:4]}"


def test_var_converges():
    stats = RunningMeanStd.zeros(DIM)
    for _ in range(200):
        batch = RNG.standard_normal((50, DIM))
        stats = update(stats, batch)
    assert np.allclose(stats.var, 1.0, atol=0.1), f"var off: {stats.var[:4]}"


def test_count_accumulates():
    stats = RunningMeanStd.zeros(DIM)
    stats = update(stats, RNG.standard_normal((10, DIM)))
    stats = update(stats, RNG.standard_normal((20, DIM)))
    assert stats.count == 30


def test_single_sample_update():
    stats = RunningMeanStd.zeros(DIM)
    sample = RNG.standard_normal(DIM)
    stats = update(stats, sample)
    assert stats.count == 1
    assert np.allclose(stats.mean, sample, atol=1e-12)


# ---------------------------------------------------------------------------
# Normalise output
# ---------------------------------------------------------------------------


def test_normalize_shape():
    stats = RunningMeanStd.zeros(DIM)
    for _ in range(50):
        stats = update(stats, RNG.standard_normal((100, DIM)))
    obs = RNG.standard_normal((10, DIM)).astype(np.float32)
    normed = normalize(stats, obs)
    assert normed.shape == obs.shape
    assert normed.dtype == np.float32


def test_normalize_clip():
    stats = RunningMeanStd.zeros(DIM)
    # Huge obs should be clipped to ±10
    obs = np.full((1, DIM), 1000.0, dtype=np.float32)
    normed = normalize(stats, obs, clip=10.0)
    assert np.all(np.abs(normed) <= 10.0 + 1e-6)


def test_normalize_after_convergence():
    stats = RunningMeanStd.zeros(DIM)
    data = RNG.standard_normal((10000, DIM))
    for i in range(0, 10000, 200):
        stats = update(stats, data[i : i + 200])
    # Fresh standard-normal sample: after normalisation std should be ≈ 1
    test_batch = RNG.standard_normal((500, DIM)).astype(np.float32)
    normed = normalize(stats, test_batch)
    assert np.allclose(normed.std(axis=0), 1.0, atol=0.1)
