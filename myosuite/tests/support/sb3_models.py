# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tiny Stable-Baselines3 models trained behind ``VecNormalize``, for policy tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import myosuite
from myosuite.utils import gym

ENV_ID = "myoElbowPose1D6MRandom-v0"


def vec_normalized_model(algo_name: str, env_id: str = ENV_ID) -> tuple[Any, Any]:
    """A fresh SB3 model behind a ``VecNormalize`` with spread, non-trivial statistics.

    Args:
        algo_name: ``"PPO"``, ``"SAC"`` or ``"TD3"``.
        env_id: Env the model acts in.

    Returns:
        ``(model, vec_normalize)``; close the ``VecNormalize`` when done.
    """
    sb3 = pytest.importorskip("stable_baselines3")
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    myosuite.register_all_envs()
    venv = VecNormalize(
        DummyVecEnv([lambda: gym.make(env_id)]), norm_reward=False, clip_obs=3.0
    )
    rng = np.random.default_rng(0)
    obs_dim = venv.observation_space.shape[0]
    venv.obs_rms.mean = rng.normal(0.0, 1.0, obs_dim)
    venv.obs_rms.var = rng.uniform(1e-3, 2.0, obs_dim)
    if algo_name == "PPO":
        kwargs: dict[str, Any] = {"n_steps": 16, "batch_size": 16}
    else:
        kwargs = {"buffer_size": 16}
    model = getattr(sb3, algo_name)("MlpPolicy", venv, seed=0, device="cpu", **kwargs)
    if algo_name == "PPO":
        # SB3 initializes the PPO action head near zero (gain 0.01); scale it so the
        # action visibly depends on the observation, as a trained policy's does.
        with torch.no_grad():
            model.policy.action_net.weight.mul_(50.0)
    return model, venv


def raw_observations(venv: Any) -> np.ndarray:
    """Raw float32 env observations, some far enough out to hit ``clip_obs``."""
    env = venv.envs[0]
    obs = np.stack([env.reset(seed=s)[0] for s in range(16)]).astype(np.float32)
    noise = np.random.default_rng(1).normal(0.0, 2.0, obs.shape).astype(np.float32)
    return np.concatenate([obs, obs + noise])
