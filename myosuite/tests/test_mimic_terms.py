# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for DeepMimic reward terms (myosuite/terms/mimic_reward.py)."""

from __future__ import annotations

import numpy as np
import pytest

from myosuite.terms.mimic_reward import (
    DEFAULT_WEIGHTS,
    mimic_composite_reward,
    mimic_joint_pos_reward,
    mimic_joint_vel_reward,
    mimic_root_orient_reward,
    mimic_root_pos_reward,
    mimic_root_vel_reward,
    mimic_site_tracking_reward,
)

RNG = np.random.default_rng(0)
N_SITES = 17
NQ = 69
NV = 68


def _rand_sites():
    return RNG.standard_normal((N_SITES, 3)).astype(np.float32)


def _rand_qpos():
    return RNG.standard_normal(NQ).astype(np.float32)


def _rand_quat():
    q = RNG.standard_normal(4).astype(np.float32)
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# Individual terms: range in (0, 1]
# ---------------------------------------------------------------------------


def test_site_tracking_range():
    r = mimic_site_tracking_reward(np, _rand_sites(), _rand_sites() + 0.1)
    assert 0.0 < float(r) <= 1.0


def test_joint_pos_range():
    r = mimic_joint_pos_reward(np, _rand_qpos(), _rand_qpos() + 0.5)
    assert 0.0 < float(r) <= 1.0


def test_joint_vel_range():
    r = mimic_joint_vel_reward(np, _rand_qpos(), _rand_qpos() + 1.0)
    assert 0.0 < float(r) <= 1.0


def test_root_pos_range():
    r = mimic_root_pos_reward(
        np,
        np.array([0.0, 0.0, 0.9], dtype=np.float32),
        np.array([0.1, 0.1, 0.9], dtype=np.float32),
    )
    assert 0.0 < float(r) <= 1.0


def test_root_vel_range():
    r = mimic_root_vel_reward(
        np,
        np.array([0.5, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert 0.0 < float(r) <= 1.0


def test_root_orient_range():
    r = mimic_root_orient_reward(np, _rand_quat(), _rand_quat())
    assert 0.0 < float(r) <= 1.0


# ---------------------------------------------------------------------------
# Perfect tracking → reward ≈ 1.0
# ---------------------------------------------------------------------------


def test_site_perfect_tracking():
    sites = _rand_sites()
    assert abs(float(mimic_site_tracking_reward(np, sites, sites)) - 1.0) < 1e-6


def test_joint_pos_perfect():
    q = _rand_qpos()
    assert abs(float(mimic_joint_pos_reward(np, q, q)) - 1.0) < 1e-6


def test_root_orient_perfect():
    quat = _rand_quat()
    assert abs(float(mimic_root_orient_reward(np, quat, quat)) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Composite reward
# ---------------------------------------------------------------------------


def test_composite_reward_range():
    sites = _rand_sites()
    qpos = _rand_qpos()
    qvel = _rand_qpos()[:NV]
    root_pos = np.array([0.0, 0.0, 0.9], dtype=np.float32)
    root_vel = np.zeros(3, dtype=np.float32)
    root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    result = mimic_composite_reward(
        np,
        sites,
        sites + 0.05,
        qpos,
        qpos + 0.1,
        qvel,
        qvel + 0.2,
        root_pos,
        root_pos + np.array([0.05, 0.0, 0.0]),
        root_vel,
        root_vel,
        root_quat,
        root_quat,
    )
    dense = float(result["dense"])
    assert 0.0 < dense <= 1.0
    assert "solved" in result
    assert "done" in result


def test_composite_perfect_tracking():
    sites = _rand_sites()
    qpos = _rand_qpos()
    qvel = _rand_qpos()[:NV]
    root_pos = np.array([0.0, 0.0, 0.9], dtype=np.float32)
    root_vel = np.zeros(3, dtype=np.float32)
    root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    result = mimic_composite_reward(
        np,
        sites,
        sites,
        qpos,
        qpos,
        qvel,
        qvel,
        root_pos,
        root_pos,
        root_vel,
        root_vel,
        root_quat,
        root_quat,
    )
    # max possible = sum of weights
    max_r = (
        DEFAULT_WEIGHTS.site
        + DEFAULT_WEIGHTS.joint_pos
        + DEFAULT_WEIGHTS.joint_vel
        + DEFAULT_WEIGHTS.root_pos
        + DEFAULT_WEIGHTS.root_vel
        + DEFAULT_WEIGHTS.root_orient
    )
    assert abs(float(result["dense"]) - max_r) < 1e-5
    assert result["solved"] is True


def test_composite_reward_skips_missing_optional_terms():
    sites = _rand_sites()

    result = mimic_composite_reward(
        np,
        sites,
        sites,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )

    assert abs(float(result["dense"]) - DEFAULT_WEIGHTS.site) < 1e-6
    assert result["joint_pos"] is None
    assert result["joint_vel"] is None
    assert result["root_pos"] is None
    assert result["root_vel"] is None
    assert result["root_orient"] is None


# ---------------------------------------------------------------------------
# Monotonicity: larger error → lower reward
# ---------------------------------------------------------------------------


def test_site_reward_monotone():
    sites = _rand_sites()
    r_small = float(mimic_site_tracking_reward(np, sites, sites + 0.01))
    r_large = float(mimic_site_tracking_reward(np, sites, sites + 0.5))
    assert r_small > r_large


# ---------------------------------------------------------------------------
# Optional: numpy / torch numerical parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("torch"),
    reason="torch not installed",
)
def test_numpy_torch_parity():
    import torch

    sites_np = _rand_sites()
    ref_np = sites_np + 0.1
    sites_t = torch.from_numpy(sites_np)
    ref_t = torch.from_numpy(ref_np)

    import torch as xp_t

    r_np = float(mimic_site_tracking_reward(np, sites_np, ref_np))
    r_t = float(mimic_site_tracking_reward(xp_t, sites_t, ref_t))
    assert abs(r_np - r_t) < 1e-5
