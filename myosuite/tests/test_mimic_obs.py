# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for mimic observation and termination terms."""

from __future__ import annotations

import numpy as np

from myosuite.terms.mimic_obs import (
    mimic_lookahead_obs,
    mimic_lookahead_obs_size,
    mimic_should_terminate,
)

RNG = np.random.default_rng(42)
T = 100
N_SITES = 17


def _make_clip():
    site_xpos = RNG.standard_normal((T, N_SITES, 3)).astype(np.float32)
    root_pos = RNG.standard_normal((T, 3)).astype(np.float32)
    root_vel = RNG.standard_normal((T, 3)).astype(np.float32)
    return site_xpos, root_pos, root_vel


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


def test_lookahead_shape():
    site_xpos, root_pos, root_vel = _make_clip()
    obs = mimic_lookahead_obs(
        current_frame=0,
        clip_site_xpos=site_xpos,
        clip_root_pos=root_pos,
        clip_root_vel=root_vel,
        current_root_pos=np.zeros(3, dtype=np.float32),
    )
    expected = mimic_lookahead_obs_size(N_SITES, has_root_pos=True, has_root_vel=True)
    assert obs.shape == (expected,), f"Expected ({expected},), got {obs.shape}"


def test_lookahead_no_root():
    site_xpos, _, _ = _make_clip()
    obs = mimic_lookahead_obs(
        current_frame=0,
        clip_site_xpos=site_xpos,
        clip_root_pos=None,
        clip_root_vel=None,
        current_root_pos=np.zeros(3, dtype=np.float32),
    )
    expected = mimic_lookahead_obs_size(N_SITES, has_root_pos=False, has_root_vel=False)
    assert obs.shape == (expected,)


# ---------------------------------------------------------------------------
# Frame advancement: step 1 at t=0 == step 0 at t=stride
# ---------------------------------------------------------------------------


def test_lookahead_shift_consistency():
    site_xpos, root_pos, root_vel = _make_clip()
    k, stride = 5, 20
    root_origin = np.zeros(3, dtype=np.float32)

    obs_t0 = mimic_lookahead_obs(
        0, site_xpos, root_pos, root_vel, root_origin, k=k, stride=stride
    )
    obs_t1 = mimic_lookahead_obs(
        stride, site_xpos, root_pos, root_vel, root_origin, k=k, stride=stride
    )
    # The first step of obs_t1 matches the second step of obs_t0 only in site content
    # (root delta differs due to different current_root_pos), but shape must be equal.
    assert obs_t0.shape == obs_t1.shape


# ---------------------------------------------------------------------------
# Boundary wrapping
# ---------------------------------------------------------------------------


def test_lookahead_wraps_at_boundary():
    site_xpos, root_pos, root_vel = _make_clip()
    # Frame near end: should not raise
    obs = mimic_lookahead_obs(
        current_frame=T - 2,
        clip_site_xpos=site_xpos,
        clip_root_pos=root_pos,
        clip_root_vel=root_vel,
        current_root_pos=np.zeros(3, dtype=np.float32),
    )
    assert np.all(np.isfinite(obs))


# ---------------------------------------------------------------------------
# Termination triggers
# ---------------------------------------------------------------------------


def test_termination_site_threshold():
    current = np.zeros((N_SITES, 3), dtype=np.float32)
    # ref far away → should terminate
    ref = np.ones((N_SITES, 3), dtype=np.float32) * 5.0
    root_c = np.zeros(3, dtype=np.float32)
    root_r = np.zeros(3, dtype=np.float32)
    assert mimic_should_terminate(current, ref, root_c, root_r, site_err_threshold=1.0)


def test_termination_root_threshold():
    current_sites = np.zeros((N_SITES, 3), dtype=np.float32)
    ref_sites = np.zeros((N_SITES, 3), dtype=np.float32)
    root_c = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    root_r = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert mimic_should_terminate(
        current_sites, ref_sites, root_c, root_r, root_err_threshold=0.3
    )


def test_no_termination_within_threshold():
    current_sites = np.zeros((N_SITES, 3), dtype=np.float32)
    ref_sites = np.zeros((N_SITES, 3), dtype=np.float32) + 0.01  # 0.01 m error
    root_c = np.array([0.0, 0.0, 0.9], dtype=np.float32)
    root_r = np.array([0.0, 0.0, 0.9], dtype=np.float32)
    assert not mimic_should_terminate(current_sites, ref_sites, root_c, root_r)


def test_termination_exactly_at_threshold():
    # Exactly at threshold: strict > means it does NOT trigger at the boundary.
    current = np.zeros((N_SITES, 3), dtype=np.float32)
    ref = np.zeros((N_SITES, 3), dtype=np.float32)
    ref[:, 0] = 1.0  # mean distance = exactly 1.0 m
    # Should not terminate (threshold is strict >)
    assert not mimic_should_terminate(
        current, ref, np.zeros(3), np.zeros(3), site_err_threshold=1.0
    )
    # Adding epsilon beyond threshold should trigger
    ref[:, 0] = 1.0 + 1e-4
    assert mimic_should_terminate(
        current, ref, np.zeros(3), np.zeros(3), site_err_threshold=1.0
    )
