# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Sweep every CPU/Gymnasium-registered MyoSuite env id for reset/step sanity.

Cheap ``reset()``/``step()`` smoke checks across every ID in
``myosuite.myosuite_env_suite``. mjlab/MJX envs are registered separately.
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


def _skip_if_hf_gated(env_id: str, exc: Exception) -> None:
    """Skip (rather than fail) an env that needs a gated HuggingFace dataset.

    Some envs (e.g. ChaseTagFBVs, FullBodyDirectional) download a reference
    gait clip from a gated HF dataset (amathislab/musclemimic-retargeted) on
    first reset/step. CI has no HF_TOKEN for an account that has accepted the
    dataset's license, so this is an access-control limitation, not a code
    bug -- treat it the same as an unavailable optional dependency.
    """
    try:
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        return
    if isinstance(exc, HfHubHTTPError):
        pytest.skip(f"{env_id}: gated HF dataset unavailable ({exc})")


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

    try:
        env = gym.make(env_id)
    except Exception as exc:
        _skip_if_hf_gated(env_id, exc)
        raise

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
    except Exception as exc:
        _skip_if_hf_gated(env_id, exc)
        raise
    finally:
        env.close()


def test_registry_is_non_empty() -> None:
    """Guard against register_all_envs() silently registering nothing."""
    assert len(_ALL_ENV_IDS) > 50
