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


def test_body_frame_off_com_push_matches_native_force_application() -> None:
    base = _TwoBodyEnv()
    base.model = mujoco.MjModel.from_xml_string(
        _TWO_BODIES.replace('<geom size="0.1"/>', '<geom size="0.1" pos=".1 0 0"/>', 1)
    )
    base.data = mujoco.MjData(base.model)
    env = PerturbationWrapper(base)
    env.reset(seed=0)
    force = np.array([4.0, 0, 0])
    env.add_perturbation(
        {
            "body": "a",
            "force": force,
            "torque": [0, 0, 1],
            "point": [0, 1, 0],
            "frame": "body",
            "start": 0,
            "end": 2,
        }
    )
    force[:] = 999  # The registered protocol owns its vectors.
    body = base.model.body("a").id
    for yaw in (np.pi / 2, np.pi):
        base.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(base.model, base.data)
        rotation = base.data.xmat[body].reshape(3, 3)
        world_force = rotation @ np.array([4.0, 0, 0])
        world_point = base.data.xpos[body] + rotation @ np.array([0, 1.0, 0])
        native = np.zeros(base.model.nv)
        mujoco.mj_applyFT(
            base.model,
            base.data,
            world_force,
            np.array([0.0, 0.0, 1.0]),
            world_point,
            body,
            native,
        )
        com = base.data.xipos[body].copy()
        reference = mujoco.MjData(base.model)
        mujoco.mj_copyData(reference, base.model, base.data)
        env.step(np.zeros(1))
        wrench = base.applied[-1][body]
        equivalent = np.zeros(base.model.nv)
        mujoco.mj_applyFT(
            base.model, reference, wrench[:3], wrench[3:], com, body, equivalent
        )
        np.testing.assert_allclose(wrench[:3], world_force, atol=1e-14)
        np.testing.assert_allclose(equivalent, native, atol=1e-12)
    env.step(np.zeros(1))
    np.testing.assert_array_equal(base.applied[-1][body], np.zeros(6))


@pytest.mark.parametrize(
    "extra",
    [
        {"frame": "camera"},
        {"force": [1, 2]},
        {"torque": [0, np.nan, 0]},
        {"point": [np.inf, 0, 0]},
    ],
)
def test_invalid_force_contract_is_not_registered(extra: dict) -> None:
    env = PerturbationWrapper(_TwoBodyEnv())
    with pytest.raises(ValueError):
        env.add_perturbation({"body": "a", "start": 0, **extra})
    assert env._perturbations == []


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
