# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Sweep every CPU/Gymnasium-registered MyoSuite env id for reset/step sanity.

CLAUDE.md's pre-commit checklist names this file, but no test currently
swept the full registry (per-task tests like test_boxing_registry.py and
test_saber_env.py only cover their own task). This file is the
registry-wide complement: cheap reset()/step() smoke checks across every
id in ``myosuite.myosuite_env_suite`` (the CPU/mujoco-py-backed suite;
mjlab/mjx envs are registered separately and covered by their own
backend-specific tests).
"""

from __future__ import annotations

import numpy as np
import pytest

from myosuite.utils import gym

pytestmark = pytest.mark.tier1

# NOTE: do not rely on myosuite.myosuite_env_suite here. register_all_envs()
# computes each suite by diffing the gym registry against its state *before*
# that call; a second call elsewhere (e.g. test_parity.py imports myosuite
# and calls register_all_envs() again at module scope) diffs against an
# already-fully-registered registry and clobbers the suite globals to empty.
# Scanning the live gym registry directly is robust to call order.
_ALL_ENV_IDS = sorted(
    env_id for env_id in gym.envs.registry if env_id.startswith(("myo", "motor"))
)


@pytest.mark.parametrize("env_id", _ALL_ENV_IDS)
def test_env_resets_and_steps(env_id: str) -> None:
    """Every registered env must reset and take one random action without error."""
    import gymnasium as gym

    def _assert_finite(value: object) -> None:
        # ModularMultiAgentTaskEnv returns per-agent dicts (CLAUDE.md exception);
        # everything else returns a flat array-like.
        if isinstance(value, dict):
            for v in value.values():
                _assert_finite(v)
            return
        assert np.all(np.isfinite(np.asarray(value, dtype=np.float64)))

    env = gym.make(env_id)
    try:
        obs, info = env.reset(seed=0)
        assert obs is not None
        _assert_finite(obs)

        action = env.action_space.sample()
        obs, rwd, terminated, truncated, info = env.step(action)
        _assert_finite(obs)
        _assert_finite(rwd)
        assert isinstance(terminated, bool | np.bool_) or isinstance(terminated, dict)
        assert isinstance(truncated, bool | np.bool_) or isinstance(truncated, dict)
    finally:
        env.close()


def test_registry_is_non_empty() -> None:
    """Guard against register_all_envs() silently registering nothing."""
    assert len(_ALL_ENV_IDS) > 50
