# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The TaskConfig elbow pose tasks observe the target they are rewarded for."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import myosuite
from myosuite.core.config import ObsSpec
from myosuite.envs.modular_env import ModularTaskEnv
from myosuite.utils import gym

myosuite.register_all_envs()

pytestmark = pytest.mark.tier1

_RANDOM_IDS = (
    "myoElbowPoseTaskRandom-v0",
    "myoSarcElbowPoseTaskRandom-v0",
    "myoFatiElbowPoseTaskRandom-v0",
)
_FIXED_IDS = (
    "myoElbowPoseTaskFixed-v0",
    "myoSarcElbowPoseTaskFixed-v0",
    "myoFatiElbowPoseTaskFixed-v0",
)
_OLD_KEYS = ["joint_pos", "joint_vel", "muscle_act"]


@pytest.mark.parametrize("env_id", _RANDOM_IDS)
def test_random_target_changes_the_observation(env_id: str) -> None:
    env = gym.make(env_id)
    first, _ = env.reset(seed=123)
    second, _ = env.reset(seed=124)
    assert not np.array_equal(first, second)


@pytest.mark.parametrize("env_id", _RANDOM_IDS + _FIXED_IDS)
def test_observation_ends_with_pose_error(env_id: str) -> None:
    env = gym.make(env_id)
    unwrapped = env.unwrapped
    obs, _ = env.reset(seed=0)
    # joint_pos (1) + joint_vel (1) + muscle_act (6) + pose_error (1)
    assert env.observation_space.shape == obs.shape == (9,)
    error = unwrapped._task_state["target_angles"] - unwrapped.data.qpos
    np.testing.assert_allclose(obs[-unwrapped.model.nq :], error, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("env_id", ["myoElbowPoseTaskRandom-v0", _FIXED_IDS[0]])
def test_pose_error_only_appends_to_the_old_observation(env_id: str) -> None:
    task = gym.spec(env_id).kwargs["task_config"]
    new = ModularTaskEnv(task)
    old = ModularTaskEnv(dataclasses.replace(task, obs=ObsSpec(keys=_OLD_KEYS)))
    obs_new, _ = new.reset(seed=7)
    obs_old, _ = old.reset(seed=7)
    np.testing.assert_array_equal(obs_new[:-1], obs_old)
    rng = np.random.default_rng(0)
    for _ in range(10):
        action = rng.uniform(new.action_space.low, new.action_space.high)
        obs_new, rwd_new, *_ = new.step(action)
        obs_old, rwd_old, *_ = old.step(action)
        assert rwd_new == rwd_old
        np.testing.assert_array_equal(obs_new[:-1], obs_old)


@pytest.mark.parametrize("env_id", _RANDOM_IDS + _FIXED_IDS)
def test_joint_velocity_is_scaled_by_the_control_timestep(env_id: str) -> None:
    """The observed velocity is ``qvel * ctrl_dt``, as in the legacy PoseEnvV0 twins."""
    env = gym.make(env_id)
    unwrapped = env.unwrapped
    unwrapped.reset(seed=0)
    unwrapped.data.qvel[:] = 0.8  # a known joint velocity
    accessor = unwrapped._accessor
    obs_dict = unwrapped._get_obs_dict(accessor)
    np.testing.assert_allclose(obs_dict["joint_vel"], 0.8 * accessor.dt(), rtol=1e-6)
