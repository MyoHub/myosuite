# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""PerturbationWrapper and DictObservationWrapper."""

from __future__ import annotations

from typing import Any

import gymnasium
import mujoco
import numpy as np
import pytest

from myosuite.envs.wrappers import DictObservationWrapper, PerturbationWrapper
from myosuite import make_env

pytestmark = pytest.mark.tier1

_TWO_BODIES = """
<mujoco>
  <worldbody>
    <body name="a"><freejoint/><geom size="0.1"/></body>
    <body name="b" pos="1 0 0"><freejoint/><geom size="0.1"/></body>
  </worldbody>
</mujoco>
"""


class _TwoBodyEnv(gymnasium.Env):
    """Minimal env exposing model/data; records the xfrc_applied of every step."""

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(_TWO_BODIES)
        self.data = mujoco.MjData(self.model)
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, (1,))
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,))
        self.applied: list[np.ndarray] = []

    def reset(self, *, seed: int | None = None, options: Any = None) -> Any:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action: Any) -> Any:
        self.applied.append(self.data.xfrc_applied.copy())
        mujoco.mj_step(self.model, self.data)
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


def test_perturbations_on_one_body_add_up() -> None:
    """Force + torque on one body are both applied (the torque used to replace
    the force), and a window closing does not cancel the others."""
    env = PerturbationWrapper(_TwoBodyEnv())
    env.add_perturbation({"body": "a", "force": [0, 5, 0], "start": 0, "end": 2})
    env.add_perturbation({"body": "a", "torque": [0, 0, 0.3], "start": 0, "end": 4})
    env.add_perturbation({"body": "a", "force": [1, 0, 0], "start": 1, "end": 3})
    env.add_perturbation({"body": "b", "force": [0, 0, 2], "start": 1})
    env.reset(seed=0)
    for _ in range(5):
        env.step(env.action_space.sample())

    base = env.unwrapped
    applied = np.asarray(base.applied)
    np.testing.assert_allclose(
        applied[:, base.model.body("a").id],
        [
            [0, 5, 0, 0, 0, 0.3],
            [1, 5, 0, 0, 0, 0.3],
            [1, 0, 0, 0, 0, 0.3],
            [0, 0, 0, 0, 0, 0.3],
            [0, 0, 0, 0, 0, 0],
        ],
    )
    np.testing.assert_allclose(applied[:, base.model.body("b").id, 2], [0, 2, 2, 2, 2])


def test_cleared_perturbations_stop_being_applied() -> None:
    env = PerturbationWrapper(_TwoBodyEnv())
    env.add_perturbation({"body": "a", "force": [0, 5, 0], "start": 0})
    env.reset(seed=0)
    env.step(env.action_space.sample())
    env.clear_perturbations()
    env.step(env.action_space.sample())
    np.testing.assert_allclose(env.unwrapped.applied[-1], 0.0)


def test_dict_observation_wrapper_space_and_reset_obs() -> None:
    """The Dict space exists before the first reset and reset returns the dict
    observation of the reset state (it returned {} and kept the flat Box)."""

    env_id = "myoElbowPose1D6MRandom-v0"
    flat_env = make_env(env_id)
    flat_reset, _ = flat_env.reset(seed=0)
    flat_env.close()
    env = DictObservationWrapper(make_env(env_id))
    assert isinstance(env.observation_space, gymnasium.spaces.Dict)

    obs, _ = env.reset(seed=0)
    assert obs and env.observation_space.contains(obs)
    np.testing.assert_allclose(
        np.concatenate([v.ravel() for v in obs.values()]), flat_reset, rtol=1e-6
    )
    obs, *_, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert obs.keys() == info["obs_dict"].keys()
    env.close()
