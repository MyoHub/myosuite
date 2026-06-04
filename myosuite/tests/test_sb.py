# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

from myosuite.utils import gym
import myosuite

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


def test_sb3_smoke_for_myochal_envs() -> None:
    """Very short SB3 PPO smoke test across MyoChallenge envs."""
    if _PPO_IMPORT_ERROR is not None or PPO is None:
        pytest.skip(
            "stable_baselines3 is missing or incompatible with the current "
            f"NumPy/Matplotlib build (install myosuite[rl] with compatible wheels): {_PPO_IMPORT_ERROR}"
        )
    assert PPO is not None
    myosuite.register_all_envs()
    suite = list(myosuite.myosuite_myochal_suite)
    total = len(suite)
    for i, env_name in enumerate(suite, 1):
        print(f"  [{i}/{total}] {env_name} ...", flush=True)
        env = gym.make(env_name)
        try:
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
        finally:
            env.close()
    print(f"  Done: {total} envs.", flush=True)
