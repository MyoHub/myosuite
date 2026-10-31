# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Smoke tests for myoFullBodyDirectional-v0 (heading only, no gait clip)."""

from __future__ import annotations

import numpy as np
import pytest
from myosuite import make_env

pytest.importorskip("musclemimic_models", reason="musclemimic_models package required")


@pytest.fixture(scope="module")
def directional_env():
    from myosuite.envs.myo.tasks.mimic.cpu import MuscleMimicFullbodyDirectionalEnv

    env = MuscleMimicFullbodyDirectionalEnv(seed=0)
    yield env
    env.close()


def _heading_slice(env) -> slice:
    """Locate heading_cmd in the flattened obs from obs_dict key sizes."""
    obs_dict = env.get_obs_dict(env._accessor)
    offset = 0
    for key, value in obs_dict.items():
        size = np.atleast_1d(value).ravel().shape[0]
        if key == "heading_cmd":
            return slice(offset, offset + size)
        offset += size
    raise KeyError("heading_cmd not found in obs_dict")


class TestMuscleMimicFullbodyDirectionalEnv:
    def test_observation_space_shape(self, directional_env):
        obs, _ = directional_env.reset()
        assert obs.shape == directional_env.observation_space.shape
        assert directional_env.observation_space.contains(obs)

    def test_obs_contains_heading(self, directional_env):
        obs, _ = directional_env.reset()
        heading = obs[_heading_slice(directional_env)]
        np.testing.assert_allclose(np.linalg.norm(heading), 1.0, atol=1e-5)

    def test_step_returns_5tuple(self, directional_env):
        directional_env.reset()
        action = directional_env.action_space.sample()
        result = directional_env.step(action)
        assert len(result) == 5

    def test_reward_is_finite_and_has_no_gait_terms(self, directional_env):
        directional_env.reset()
        action = directional_env.action_space.sample()
        obs, rwd, terminated, truncated, info = directional_env.step(action)
        assert np.isfinite(rwd)
        assert np.all(np.isfinite(obs))
        rwd_dict = info.get("rwd_dict", info)
        assert "heading" in rwd_dict or "heading" in str(info)
        assert "gait_jpos" not in rwd_dict
        assert "gait_jvel" not in rwd_dict
        del terminated, truncated

    def test_reset_randomises_heading(self, directional_env):
        heading_slice = _heading_slice(directional_env)
        headings = []
        for _ in range(5):
            obs, _ = directional_env.reset()
            headings.append(obs[heading_slice].copy())
        headings = np.stack(headings)
        assert not np.allclose(headings[0], headings[1:])

    def test_gym_make_registration(self):
        env = make_env("myoFullBodyDirectional-v0")
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
        env.close()

    def test_legacy_gait_kwargs_are_ignored(self):
        from myosuite.envs.myo.tasks.mimic.cpu import MuscleMimicFullbodyDirectionalEnv

        env = MuscleMimicFullbodyDirectionalEnv(
            seed=0, w_jpos=0.15, w_jvel=0.05, gait_clip_repo="unused"
        )
        env.reset()
        env.close()
