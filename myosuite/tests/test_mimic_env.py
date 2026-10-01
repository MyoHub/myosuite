# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for MuscleMimicClipEnvV0 — clip-following CPU gymnasium env."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

_CLIP_REPO_ID = "amathislab/musclemimic-retargeted"
_CLIP_FILENAME = "MyoFullBody/gmr/KIT/167/walking_medium06_poses.npz"


def _resolve_clip_path() -> pathlib.Path | None:
    """Resolve the default mimic clip path in a machine-agnostic way."""
    try:
        from huggingface_hub import hf_hub_download

        return pathlib.Path(
            hf_hub_download(
                repo_id=_CLIP_REPO_ID,
                filename=_CLIP_FILENAME,
                repo_type="dataset",
            )
        )
    except ImportError:
        return None
    except RuntimeError:
        return None
    except OSError:
        return None


_HF_CLIP = _resolve_clip_path()
_CLIP_AVAILABLE = _HF_CLIP is not None and _HF_CLIP.is_file()

pytestmark = pytest.mark.skipif(
    not _CLIP_AVAILABLE,
    reason="HuggingFace motion clip not in cache (run musclemimic-setup-demo-cache)",
)


@pytest.fixture(scope="module")
def env():
    from myosuite.envs.myo.tasks.mimic.clip_env import MuscleMimicClipEnvV0

    e = MuscleMimicClipEnvV0(clip_path=_HF_CLIP, seed=42, use_obs_normalizer=False)
    yield e


def test_obs_shape(env):
    obs, _ = env.reset()
    assert obs.ndim == 1
    assert obs.shape == env.observation_space.shape


def test_obs_finite(env):
    obs, _ = env.reset()
    assert np.all(np.isfinite(obs)), "reset obs contains non-finite values"


def test_step_returns_5tuple(env):
    env.reset()
    action = env.action_space.sample()
    result = env.step(action)
    assert len(result) == 5, "step() must return 5-tuple"


def test_reward_range(env):
    obs, _ = env.reset()
    # First step from clip init: reward should be reasonable (>0.5)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, rew, _, _, _ = env.step(action)
    assert 0.0 < rew <= 1.01, f"reward out of expected range: {rew}"


def test_perfect_init_reward_high(env):
    """Starting exactly from clip state should give high reward."""
    obs, _ = env.reset()
    # With zero control the first step stays close to clip → high reward
    _, rew, _, _, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert rew > 0.5, f"Expected high reward from clip init, got {rew:.4f}"


def test_reward_degrades_with_zero_ctrl(env):
    """Zero control (body falls) should reduce reward over time."""
    env.reset()
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    rewards = []
    for _ in range(20):
        _, rew, term, _, _ = env.step(zero)
        rewards.append(rew)
        if term:
            break
    # Later rewards should be lower on average than early rewards
    assert np.mean(rewards[:3]) > np.mean(
        rewards[-3:]
    ), f"Reward should degrade; got {rewards}"


def test_early_termination_fires(env):
    """Zero control should eventually trigger early termination."""
    env.reset()
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    terminated_ever = False
    for _ in range(500):
        _, _, term, _, _ = env.step(zero)
        if term:
            terminated_ever = True
            break
    assert (
        terminated_ever
    ), "Expected early termination with zero control over 500 steps"


def test_reset_different_frames(env):
    """Multiple resets should sample different start frames."""
    frames = set()
    for _ in range(10):
        env.reset()
        frames.add(env._frame_offset)
    assert len(frames) > 1, "Expected different start frames across resets"


def test_action_space_bounds(env):
    env.reset()
    low, high = env.action_space.low, env.action_space.high
    action = env.action_space.sample()
    assert np.all(action >= low) and np.all(action <= high)
