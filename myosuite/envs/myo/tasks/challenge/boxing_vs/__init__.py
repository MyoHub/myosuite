# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Two-agent competitive boxing environment package.

Usage::

    from myosuite.envs.myo.tasks.challenge.boxing_vs import BoxingVsEnv, BoxingVsConfig

    env = BoxingVsEnv(config=BoxingVsConfig(max_episode_steps=300))
    obs, info = env.reset(seed=0)

Or via Gymnasium (after ``register_all_envs()``)::

    import gymnasium as gym
    env = gym.make("myoChallengeBoxingVs-v0")
"""

from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_config import BoxingVsConfig
from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_env import (
    BoxingVsEnv,
    BoxingVsSingleAgentWrapper,
)
from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_model import (
    BoxingVsModelMeta,
)
from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_task_config import (
    BoxingVsTaskConfig,
)

__all__ = [
    "BoxingVsConfig",
    "BoxingVsEnv",
    "BoxingVsModelMeta",
    "BoxingVsSingleAgentWrapper",
    "BoxingVsTaskConfig",
]
