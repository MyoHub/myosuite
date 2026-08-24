# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Smoke tests for MuscleMimicFullbodyDirectionalEnv.

Requires:
  - musclemimic_models (MJCF assets)
  - huggingface_hub + internet access (gait clip download)
  - amathislab/musclemimic-retargeted dataset on HF Hub
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("musclemimic_models", reason="musclemimic_models package required")
pytest.importorskip("huggingface_hub", reason="huggingface_hub required")


@pytest.fixture(scope="module")
def directional_env():
    from myosuite.envs.myo.tasks.mimic.cpu import MuscleMimicFullbodyDirectionalEnv

    env = MuscleMimicFullbodyDirectionalEnv(seed=0)
    yield env
    env.close()


class TestMuscleMimicFullbodyDirectionalEnv:
    def test_observation_space_shape(self, directional_env):
        obs, _ = directional_env.reset()
        assert obs.shape == directional_env.observation_space.shape
        assert directional_env.observation_space.contains(obs)

    def test_obs_contains_heading(self, directional_env):
        obs, _ = directional_env.reset()
        # Last 2 elements are the heading unit vector — must have unit norm
        heading = obs[-2:]
        np.testing.assert_allclose(np.linalg.norm(heading), 1.0, atol=1e-5)

    def test_step_returns_5tuple(self, directional_env):
        directional_env.reset()
        action = directional_env.action_space.sample()
        result = directional_env.step(action)
        assert len(result) == 5

    def test_reward_is_finite(self, directional_env):
        directional_env.reset()
        for _ in range(10):
            action = directional_env.action_space.sample()
            obs, rwd, terminated, truncated, info = directional_env.step(action)
            assert np.isfinite(rwd), f"reward was {rwd}"
            assert np.all(np.isfinite(obs))
            if terminated or truncated:
                directional_env.reset()
                break

    def test_reset_randomises_heading(self, directional_env):
        headings = []
        for _ in range(5):
            obs, _ = directional_env.reset()
            headings.append(obs[-2:].copy())
        headings = np.stack(headings)
        assert not np.allclose(
            headings[0], headings[1:]
        ), "heading should be randomised across resets"

    def test_heading_reward_term_in_info(self, directional_env):
        directional_env.reset()
        action = directional_env.action_space.sample()
        _, _, _, _, info = directional_env.step(action)
        assert "rwd_dict" in info or "heading" in str(
            info
        ), f"expected reward breakdown in info, got: {info.keys()}"

    def test_gym_make_registration(self):
        import gymnasium as gym
        import myosuite  # noqa: F401 — triggers registration

        env = gym.make("myoFullBodyDirectional-v0")
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
        env.close()
