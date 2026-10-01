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
    from huggingface_hub.errors import HfHubHTTPError

    from myosuite.envs.myo.tasks.mimic.cpu import MuscleMimicFullbodyDirectionalEnv

    try:
        env = MuscleMimicFullbodyDirectionalEnv(seed=0)
    except HfHubHTTPError as exc:
        # amathislab/musclemimic-retargeted is a gated HF dataset; skip rather
        # than error when there's no HF_TOKEN for an account that has
        # accepted its license (e.g. in CI).
        pytest.skip(f"gated HF dataset unavailable ({exc})")
    yield env
    env.close()


def _heading_slice(env) -> slice:
    """Locate heading_cmd's span in the flattened obs from obs_dict's own
    key sizes, rather than assuming a fixed offset like ``obs[-2:]``.

    ``_obs_dict_to_vec`` concatenates in dict insertion order:
    qpos_local, qvel_local, act, root_vel_xy, heading_cmd, orientation(6).
    heading_cmd is NOT last — the trailing 6 elements are orientation
    (roll, pitch, wx, wy, wz, vz), whose norm has no reason to be 1.
    """
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
        heading_slice = _heading_slice(directional_env)
        headings = []
        for _ in range(5):
            obs, _ = directional_env.reset()
            headings.append(obs[heading_slice].copy())
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
        from huggingface_hub.errors import HfHubHTTPError

        import myosuite  # noqa: F401 — triggers registration

        try:
            env = gym.make("myoFullBodyDirectional-v0")
        except HfHubHTTPError as exc:
            pytest.skip(f"gated HF dataset unavailable ({exc})")
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
        env.close()
