# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Exact native replay includes task state, wrapper clocks and controller history."""

import copy
import random

import gymnasium as gym
import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.envs.wrappers import FatigueWrapper, MotorNoiseWrapper
from myosuite.utils.paired_replay import ReplayPolicy, ReplaySnapshot, ReplayStateOwner

pytestmark = pytest.mark.tier1


class RandomEnv(gym.Env):
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string("""<mujoco><option gravity="0 0 0"/>
          <worldbody><body><joint name="slide" type="slide"/><geom size=".1" mass="1"/></body></worldbody>
          <actuator><general joint="slide" dyntype="filter" dynprm=".02"/></actuator></mujoco>""")
        self.data = mujoco.MjData(self.model)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.steps = 0
        self.history = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.steps, self.history = 0, []
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        self.history.append(action.copy())
        self.data.ctrl[:] = (
            action
            + self.np_random.normal(0, 0.01)
            + random.random() * 0.01
            + np.random.random() * 0.01
        )
        mujoco.mj_step(self.model, self.data, 2)
        mujoco.mj_forward(self.model, self.data)
        obs = np.r_[self.data.qpos, self.data.qvel, self.data.act].astype(np.float32)
        return obs, float(self.data.qpos[0]), False, False, {}


class HistoryController:
    def __init__(self) -> None:
        self.history = []
        self.rng = np.random.default_rng(7)

    def action(self, obs):
        self.history.append(float(obs[0]))
        return np.array(
            [self.rng.uniform(0.1, 0.3) + 0.01 * len(self.history)], np.float32
        )

    def get_state(self):
        return {
            "history": self.history.copy(),
            "rng": copy.deepcopy(self.rng.bit_generator.state),
        }

    def set_state(self, state):
        self.history = state["history"].copy()
        self.rng.bit_generator.state = state["rng"]


def test_stateful_native_replay_restores_every_runtime_layer() -> None:
    env = gym.wrappers.TimeLimit(RandomEnv(), max_episode_steps=5)
    obs, _ = env.reset(seed=3)
    controller = HistoryController()
    policy = ReplayPolicy(controller.action, controller.get_state, controller.set_state)
    obs, *_ = env.step(policy.action(obs))
    snapshot = ReplaySnapshot.capture(env, policy)
    integration = snapshot.integration_state()
    trajectories = []
    for _ in range(3):
        snapshot.restore()
        current = obs.copy()
        rows = []
        for _ in range(4):
            action = policy.action(current)
            current, reward, terminated, truncated, _ = env.step(action)
            rows.append((current.copy(), action.copy(), reward, terminated, truncated))
        trajectories.append(rows)
    for rows in trajectories[1:]:
        for actual, expected in zip(rows, trajectories[0]):
            for a, e in zip(actual, expected):
                np.testing.assert_array_equal(a, e)
    assert trajectories[0][-1][-1]
    snapshot.restore()
    assert env.unwrapped.steps == 1 and len(controller.history) == 1
    np.testing.assert_array_equal(snapshot.integration_state(), integration)


def test_snapshot_does_not_alias_returned_policy_state() -> None:
    env = RandomEnv()
    env.reset(seed=0)
    controller = HistoryController()
    snapshot = ReplaySnapshot.capture(
        env, ReplayPolicy(controller.action, controller.get_state, controller.set_state)
    )
    exported = snapshot.policy_state
    exported["history"].append(999.0)
    controller.history.append(42.0)
    snapshot.restore()
    assert controller.history == []


@pytest.mark.parametrize("env_id", ["myoLegWalk-v0", "myoLegDirectionalForward-v0"])
def test_existing_walking_env_restores_native_and_task_state(env_id: str) -> None:
    env = make_env(env_id)
    try:
        obs, _ = env.reset(seed=4)
        policy = ReplayPolicy.stateless(
            lambda obs: np.zeros(env.action_space.shape, np.float32)
        )
        snapshot = ReplaySnapshot.capture(env, policy)
        expected = env.step(policy.action(obs))
        qpos = env.unwrapped.data.qpos.copy()
        snapshot.restore()
        actual = env.step(policy.action(obs))
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(env.unwrapped.data.qpos, qpos)
        assert actual[1:4] == expected[1:4]
    finally:
        env.close()


def test_changed_model_is_rejected_before_restore() -> None:
    env = RandomEnv()
    env.reset(seed=0)
    snapshot = ReplaySnapshot.capture(
        env, ReplayPolicy.stateless(lambda obs: np.zeros(1))
    )
    env.model.body_mass[1] *= 2
    env.steps = 9
    with pytest.raises(ValueError, match="model changed"):
        snapshot.restore()
    assert env.steps == 9


def test_loaded_torch_rng_restores_when_available() -> None:
    torch = pytest.importorskip("torch")
    env = RandomEnv()
    env.reset(seed=0)
    snapshot = ReplaySnapshot.capture(
        env, ReplayPolicy.stateless(lambda obs: np.zeros(1))
    )
    expected = torch.rand(5)
    snapshot.restore()
    torch.testing.assert_close(torch.rand(5), expected, rtol=0, atol=0)


def test_shared_state_across_layers_keeps_aliases() -> None:
    env = gym.wrappers.TimeLimit(RandomEnv(), max_episode_steps=10)
    env.reset(seed=0)
    env.shared = env.unwrapped.shared = {"history": [1]}
    snapshot = ReplaySnapshot.capture(
        env, ReplayPolicy.stateless(lambda obs: np.zeros(1))
    )
    env.shared["history"].append(2)
    snapshot.restore()
    assert env.shared is env.unwrapped.shared
    assert env.shared == {"history": [1]}


@pytest.mark.parametrize("muscles", [None, [0, 2, 4]])
def test_fatigue_noise_closure_restores_original_owner_in_place(muscles) -> None:
    env = MotorNoiseWrapper(
        FatigueWrapper(make_env("myoElbowPose1D6MFixed-v0"), muscles=muscles),
        {"constant_std": 0.01},
    )
    try:
        obs, _ = env.reset(seed=4)
        fatigue = env.env.muscle_fatigue
        policy = ReplayPolicy.stateless(
            lambda obs: np.full(env.action_space.shape, 0.6, np.float32)
        )
        snapshot = ReplaySnapshot.capture(env, policy)
        expected = []
        for _ in range(8):
            current, reward, *_ = env.step(policy.action(obs))
            expected.append(
                (current.copy(), reward, copy.deepcopy(fatigue.state_dict()))
            )
        snapshot.restore()
        assert env.env.muscle_fatigue is fatigue
        assert env.unwrapped.muscle_fatigue is fatigue
        for current_expected, reward_expected, fatigue_expected in expected:
            current, reward, *_ = env.step(policy.action(obs))
            np.testing.assert_array_equal(current, current_expected)
            assert reward == reward_expected
            assert fatigue.state_dict() == fatigue_expected
    finally:
        env.close()


def test_custom_mutable_closure_requires_and_restores_explicit_owner() -> None:
    env = make_env("myoElbowPose1D6MFixed-v0")
    history = []

    def stage(host, ctrl):
        history.append(float(ctrl.sum()))
        return ctrl

    try:
        env.unwrapped.add_ctrl_stage("history", stage)
        env.reset(seed=0)
        policy = ReplayPolicy.stateless(
            lambda obs: np.full(env.action_space.shape, 0.5, np.float32)
        )
        with pytest.raises(ValueError, match="ReplayStateOwner"):
            ReplaySnapshot.capture(env, policy)
        owner = ReplayStateOwner(
            history,
            lambda: list(history),
            lambda state: history.__setitem__(slice(None), state),
        )
        snapshot = ReplaySnapshot.capture(env, policy, state_owners=(owner,))
        env.step(policy.action(None))
        assert history
        snapshot.restore()
        assert history == []
    finally:
        env.close()
