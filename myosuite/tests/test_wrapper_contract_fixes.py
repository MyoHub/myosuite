# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for wrapper attribute forwarding and seeded opponent sampling."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from myosuite import make_env
from myosuite.envs.multi_agent_modular_env import ModularMultiAgentSingleAgentWrapper
from myosuite.envs.wrappers import DictObservationWrapper, PerturbationWrapper

pytestmark = pytest.mark.tier1


def test_single_agent_opponent_wrapper_seed_reproducibility() -> None:
    class _DummyMultiEnv(gym.Env):
        def __init__(self) -> None:
            super().__init__()
            self._agents = ("agent_0", "agent_1")
            box = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
            self.observation_space = gym.spaces.Dict({"agent_0": box, "agent_1": box})
            self.action_space = gym.spaces.Dict({"agent_0": box, "agent_1": box})
            self.last_opp_actions: list[np.ndarray] = []

        def _opponent(self, agent_id: str) -> str:
            return "agent_1" if agent_id == "agent_0" else "agent_0"

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            z = np.zeros(3, dtype=np.float32)
            return {"agent_0": z, "agent_1": z}, {}

        def step(self, actions):
            self.last_opp_actions.append(actions["agent_1"].copy())
            z = np.zeros(3, dtype=np.float32)
            return (
                {"agent_0": z, "agent_1": z},
                {"agent_0": 0.0, "agent_1": 0.0},
                {"agent_0": False, "agent_1": False},
                {"agent_0": False, "agent_1": False},
                {},
            )

    base = _DummyMultiEnv()
    wrapped = ModularMultiAgentSingleAgentWrapper(base)
    wrapped.reset(seed=42)
    for _ in range(3):
        wrapped.step(np.zeros(3, dtype=np.float32))
    first = [a.copy() for a in base.last_opp_actions]

    base.last_opp_actions.clear()
    wrapped.reset(seed=42)
    for _ in range(3):
        wrapped.step(np.zeros(3, dtype=np.float32))
    second = [a.copy() for a in base.last_opp_actions]

    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_dict_and_perturbation_wrappers_forward_attrs_and_filter_keys() -> None:
    dict_env = DictObservationWrapper(
        make_env("myoChallengeBimanual-v0"), filter_obs_keys=True
    )
    try:
        assert callable(dict_env.mj_render)
        assert dict_env.model is dict_env.unwrapped.model
        obs, _ = dict_env.reset(seed=0)
        assert list(obs.keys()) == list(dict_env.unwrapped.obs_keys)
    finally:
        dict_env.close()

    pert_env = PerturbationWrapper(make_env("myoElbowPose1D6MRandom-v0"))
    try:
        assert callable(pert_env.mj_render)
        assert pert_env.model is pert_env.unwrapped.model
    finally:
        pert_env.close()
