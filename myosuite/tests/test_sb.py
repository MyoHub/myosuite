# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gymnasium as gym
import pytest
from huggingface_hub.errors import GatedRepoError

import myosuite
from myosuite import make_env
from myosuite.envs.multi_agent_modular_env import (
    ModularMultiAgentSingleAgentWrapper,
    ModularMultiAgentTaskEnv,
)

# SB3 imports matplotlib at module import time; a NumPy 2 / older matplotlib
# binary mismatch raises AttributeError: _ARRAY_API not found. Defer failure to
# test runtime so pytest collection succeeds in mixed conda/pip environments.
try:
    from stable_baselines3 import PPO
except (ImportError, AttributeError) as exc:
    PPO = None  # type: ignore[misc, assignment]
    _PPO_IMPORT_ERROR: BaseException | None = exc
else:
    _PPO_IMPORT_ERROR = None


pytestmark = [pytest.mark.tier3, pytest.mark.slow]


def _single_agent_view(env: gym.Env) -> gym.Env:
    """Return ``env``, or one player's view of a two-agent game.

    SB3 rejects the per-agent ``Dict`` spaces of a ``ModularMultiAgentTaskEnv``;
    the wrapper trains ``agent_0`` against an opponent taking random actions.
    """
    if isinstance(env.unwrapped, ModularMultiAgentTaskEnv):
        return ModularMultiAgentSingleAgentWrapper(env)
    return env


@pytest.mark.parametrize("env_name", myosuite.myosuite_myochal_suite)
def test_sb3_smoke_for_myochal_envs(env_name: str) -> None:
    """Very short SB3 PPO smoke test on one MyoChallenge env."""
    if _PPO_IMPORT_ERROR is not None or PPO is None:
        pytest.skip(
            "stable_baselines3 is missing or incompatible with the current "
            f"NumPy/Matplotlib build (install myosuite[rl] with compatible wheels): {_PPO_IMPORT_ERROR}"
        )
    assert PPO is not None
    try:
        with _single_agent_view(make_env(env_name)) as env:
            # Force CPU device so this smoke test remains lightweight even when
            # CUDA is available in the environment.
            # Use n_steps=2 and batch_size=2 so learn(2) actually runs ~2 steps;
            # default n_steps=2048 would collect 2048 steps per env before first update.
            model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                device="cpu",
                n_steps=2,
                batch_size=2,
            )
            model.learn(total_timesteps=2)
    except GatedRepoError as exc:
        # Gait reference clips (e.g. ChaseTagFBVs' reward) come from a gated HF
        # dataset; CI has no token for an account that accepted its license.
        pytest.skip(f"{env_name}: gated HF dataset unavailable ({exc})")
