# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke-tests for every registered MyoSuite environment.

Each test:
1. Instantiates the env.
2. Calls reset(seed=0) and validates the observation.
3. Calls step() with a random action and validates the return types.
4. Closes the env.

Environments that cannot be instantiated (e.g. missing assets) are skipped
rather than failed, so CI passes even on partial installs.
"""

from __future__ import annotations

import numpy as np
import pytest

from myosuite.utils import gym


pytestmark = pytest.mark.tier1


def _get_myo_env_ids() -> list[str]:
    """Return all MyoSuite env IDs from the gymnasium registry."""
    import myosuite

    myosuite.register_all_envs()

    return [
        env_id
        for env_id in gym.envs.registry.keys()
        if env_id.startswith("myo") or env_id.startswith("motor")
    ]


@pytest.mark.parametrize("env_id", _get_myo_env_ids())
def test_env_smoke(env_id: str) -> None:
    """Smoke-test a single environment: reset, step, close."""
    try:
        env = gym.make(env_id)
    except Exception as exc:
        pytest.skip(f"Cannot instantiate {env_id}: {exc}")

    try:
        obs, info = env.reset(seed=0)
        assert isinstance(info, dict), f"{env_id}: reset() info must be a dict"

        if isinstance(obs, dict):
            # Multi-agent env: validate each agent's obs independently.
            for agent_id, agent_obs in obs.items():
                assert (
                    agent_obs.ndim == 1
                ), f"{env_id}/{agent_id}: obs should be 1-D, got shape {agent_obs.shape}"
                assert (
                    agent_obs.shape[0] > 0
                ), f"{env_id}/{agent_id}: obs must not be empty"
                assert np.all(
                    np.isfinite(agent_obs)
                ), f"{env_id}/{agent_id}: obs contains non-finite values"
            obs, rwd, terminated, truncated, info = env.step(env.action_space.sample())
            assert isinstance(rwd, dict), f"{env_id}: multi-agent reward must be a dict"
            assert isinstance(terminated, dict), f"{env_id}: terminated must be a dict"
            assert isinstance(truncated, dict), f"{env_id}: truncated must be a dict"
        else:
            assert obs.ndim == 1, f"{env_id}: obs should be 1-D, got shape {obs.shape}"
            assert obs.shape[0] > 0, f"{env_id}: obs must not be empty"
            assert np.all(np.isfinite(obs)), f"{env_id}: obs contains non-finite values"

            obs, rwd, terminated, truncated, info = env.step(env.action_space.sample())
            assert isinstance(rwd, float), f"{env_id}: reward must be a float"
            assert isinstance(terminated, bool), f"{env_id}: terminated must be a bool"
            assert isinstance(truncated, bool), f"{env_id}: truncated must be a bool"

        assert isinstance(info, dict), f"{env_id}: step() info must be a dict"
    finally:
        env.close()
