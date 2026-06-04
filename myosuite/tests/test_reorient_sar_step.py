# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for SAR reorient: geometry sampling and env reset/step contracts.

Verifies each step of the reorient pipeline: shared geometry module outputs,
reset() return shape and types, step() return shape and types, and info/rwd_dict.
"""

from __future__ import annotations

import numpy as np
import pytest

import myosuite

from myosuite.envs.myo.tasks.basic.arm.reorient_sar_geometries import (
    GEOMETRIES_8,
    GEOMETRIES_100,
    sample_geometry_8,
    sample_geometry_100,
)
from myosuite.utils import gym

myosuite.register_all_envs()


pytestmark = pytest.mark.tier2


# -----------------------------------------------------------------------------
# Geometry module
# -----------------------------------------------------------------------------


def test_sample_geometry_8_returns_correct_structure() -> None:
    """sample_geometry_8 returns (geom_type, size, color, top_pos, bot_pos, desired_euler)."""
    rng = np.random.default_rng(42)
    out = sample_geometry_8(rng)
    assert len(out) == 6
    geom_type, size, color, top_pos, bot_pos, desired_euler = out
    assert geom_type in (3, 4, 5, 6)
    assert isinstance(size, list) and len(size) == 3
    assert isinstance(color, list) and len(color) == 4
    assert np.shape(top_pos) == (3,)
    assert np.shape(bot_pos) == (3,)
    assert np.shape(desired_euler) == (3,)
    # 8-object variant overrides color to yellow
    assert color == [1.0, 0.9, 0.0, 1.0]


def test_sample_geometry_8_deterministic() -> None:
    """Same seed produces same geometry sample."""
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    out1 = sample_geometry_8(rng1)
    out2 = sample_geometry_8(rng2)
    assert out1[0] == out2[0]
    assert out1[1] == out2[1]
    assert out1[2] == out2[2]
    np.testing.assert_array_almost_equal(out1[3], out2[3])
    np.testing.assert_array_almost_equal(out1[4], out2[4])
    np.testing.assert_array_almost_equal(out1[5], out2[5])


def test_sample_geometry_100_returns_correct_structure() -> None:
    """sample_geometry_100 returns same structure; color not overridden."""
    rng = np.random.default_rng(7)
    out = sample_geometry_100(rng)
    assert len(out) == 6
    geom_type, size, color, top_pos, bot_pos, desired_euler = out
    assert geom_type in (3, 4, 5, 6)
    assert isinstance(size, list) and len(size) == 3
    assert isinstance(color, list) and len(color) == 4
    assert np.shape(top_pos) == (3,)
    assert np.shape(bot_pos) == (3,)
    assert np.shape(desired_euler) == (3,)
    # 100-object keeps sampled color (not forced to yellow)
    assert color != [1.0, 0.9, 0.0, 1.0] or geom_type == 3  # may coincide for one geom


def test_sample_geometry_100_deterministic() -> None:
    """Same seed produces same geometry sample for 100 variant."""
    rng1 = np.random.default_rng(456)
    rng2 = np.random.default_rng(456)
    out1 = sample_geometry_100(rng1)
    out2 = sample_geometry_100(rng2)
    assert out1[0] == out2[0]
    assert out1[1] == out2[1]
    np.testing.assert_array_almost_equal(out1[3], out2[3])
    np.testing.assert_array_almost_equal(out1[4], out2[4])
    np.testing.assert_array_almost_equal(out1[5], out2[5])


def test_geometries_8_and_100_have_four_shapes() -> None:
    """GEOMETRIES_8 and GEOMETRIES_100 each have ellips, box, caps, cyl."""
    for name, variant in [("8", GEOMETRIES_8), ("100", GEOMETRIES_100)]:
        assert set(variant.keys()) == {"ellips", "box", "caps", "cyl"}, name
        for shape_name, entries in variant.items():
            assert isinstance(entries, dict), f"{name} {shape_name}"
            assert len(entries) >= 1, f"{name} {shape_name} empty"


# -----------------------------------------------------------------------------
# Env: reset
# -----------------------------------------------------------------------------


def test_reorient8_reset_returns_obs_and_info() -> None:
    """myoHandReorient8-v0 reset(seed=...) returns (obs, info); obs in space and finite."""
    env = gym.make("myoHandReorient8-v0")
    try:
        obs, info = env.reset(seed=0)
        assert isinstance(obs, np.ndarray), "obs should be ndarray"
        assert obs.ndim == 1 and obs.shape[0] > 0, "obs should be 1-D non-empty"
        assert np.all(np.isfinite(obs)), "obs should be finite"
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs), "obs should be in observation_space"
        assert isinstance(info, dict), "info should be dict"
    finally:
        env.close()


def test_reorient8_reset_deterministic() -> None:
    """Same seed gives same initial obs."""
    env = gym.make("myoHandReorient8-v0")
    try:
        obs1, _ = env.reset(seed=99)
        obs2, _ = env.reset(seed=99)
        np.testing.assert_allclose(
            obs1, obs2, atol=1e-9, err_msg="reset(seed=99) should be deterministic"
        )
    finally:
        env.close()


# -----------------------------------------------------------------------------
# Env: step
# -----------------------------------------------------------------------------


def test_reorient8_step_returns_five_tuple() -> None:
    """step(action) returns (obs, rwd, terminated, truncated, info) with correct types."""
    env = gym.make("myoHandReorient8-v0")
    try:
        env.reset(seed=1)
        action = env.action_space.sample()
        obs, rwd, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray) and obs.ndim == 1
        assert isinstance(rwd, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert np.all(np.isfinite(obs)), "obs finite"
        assert np.isfinite(rwd), "rwd finite"
    finally:
        env.close()


def test_reorient8_step_info_contains_rwd_dict() -> None:
    """step() info contains rwd_dict with expected reward keys."""
    env = gym.make("myoHandReorient8-v0")
    try:
        env.reset(seed=2)
        _, _, _, _, info = env.step(env.action_space.sample())
        rwd_dict = info.get("rwd_dict")
        assert rwd_dict is not None, "info should have rwd_dict"
        expected = {
            "pos_align",
            "rot_align",
            "act_reg",
            "drop",
            "bonus",
            "dense",
            "done",
            "solved",
            "sparse",
        }
        assert expected.issubset(
            rwd_dict.keys()
        ), f"rwd_dict missing keys: {expected - set(rwd_dict)}"
        assert "dense" in rwd_dict
        assert isinstance(rwd_dict["dense"], (float, np.floating))
        assert isinstance(rwd_dict["done"], (bool, np.bool_))
    finally:
        env.close()


def test_reorient8_multiple_steps_finite() -> None:
    """Multiple steps in one episode yield finite obs and reward."""
    env = gym.make("myoHandReorient8-v0")
    try:
        env.reset(seed=3)
        for _ in range(10):
            obs, rwd, terminated, truncated, info = env.step(env.action_space.sample())
            assert np.all(np.isfinite(obs))
            assert np.isfinite(rwd)
            if terminated or truncated:
                break
    finally:
        env.close()


def test_reorient8_step_obs_in_space() -> None:
    """Each step observation lies within observation_space (after clipping)."""
    env = gym.make("myoHandReorient8-v0")
    try:
        env.reset(seed=4)
        for _ in range(5):
            obs, _, _, _, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(
                obs
            ), f"obs not in space: min={obs.min()}, max={obs.max()}"
    finally:
        env.close()


def test_reorient100_reset_and_step_same_contract() -> None:
    """myoHandReorient100-v0 reset and step return same types/shapes as 8-object."""
    env = gym.make("myoHandReorient100-v0")
    try:
        obs, info = env.reset(seed=5)
        assert obs.ndim == 1 and obs.shape[0] > 0 and isinstance(info, dict)
        obs, rwd, term, trunc, info = env.step(env.action_space.sample())
        assert isinstance(rwd, (float, np.floating)) and isinstance(term, bool)
        assert "rwd_dict" in info
    finally:
        env.close()


def test_reorient8_parity_baseline_replay_step_by_step() -> None:
    """Replay myoHandReorient8-v0 baseline: each step matches stored obs, rwd, terminated."""
    from pathlib import Path

    import pickle

    # Same policy as test_parity.NONDETERMINISTIC_PARITY_ENVS: long replays diverge on CI.
    pytest.skip(
        "myoHandReorient8-v0 baseline replay is non-deterministic on some runners; "
        "see test_parity.NONDETERMINISTIC_PARITY_ENVS."
    )

    baseline_dir = Path(__file__).parent / "parity_baselines"
    pkl_path = baseline_dir / "myoHandReorient8-v0.pkl"
    if not pkl_path.exists():
        pytest.skip("No parity baseline for myoHandReorient8-v0")

    with open(pkl_path, "rb") as fh:
        records = pickle.load(fh)

    env = gym.make("myoHandReorient8-v0")
    try:
        env.reset()
        current_episode_np_state = None
        for i, rec in enumerate(records):
            if rec["episode_np_state"] is not current_episode_np_state:
                current_episode_np_state = rec["episode_np_state"]
                env.unwrapped.np_random.bit_generator.state = current_episode_np_state
                env.reset()

            obs, rwd, terminated, truncated, _ = env.step(rec["action"])

            np.testing.assert_allclose(
                obs, rec["obs"], atol=1.0, err_msg=f"step {i}: obs mismatch"
            )
            assert abs(rwd - rec["rwd"]) < 1e-2, f"step {i}: rwd mismatch"
            assert terminated == rec["terminated"], f"step {i}: terminated mismatch"
    finally:
        env.close()
