# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import pickle
import unittest

import pytest
import numpy as np
from flatten_dict import flatten

from myosuite.utils import gym


pytestmark = pytest.mark.tier2


def assert_close(prm1, prm2, atol=1e-05, rtol=1e-08):
    if prm1 is None and prm2 is None:
        return True
    elif isinstance(prm1, dict) and isinstance(prm2, dict):
        prm1_dict = flatten(prm1)
        prm2_dict = flatten(prm2)
        for key in prm1_dict.keys():
            assert_close(prm1_dict[key], prm2_dict[key], atol=atol, rtol=rtol)
    else:
        np.testing.assert_allclose(prm1, prm2, atol=atol, rtol=rtol)


class TestEnvs(unittest.TestCase):
    def check_envs(self, module_name, env_names, lite=False, input_seed=1234):
        print("\n=================================", flush=True)
        print("Testing module:: ", module_name)
        for env_name in env_names:
            print("Testing env: ", env_name, flush=True)
            self.check_env(env_name, input_seed)

    def check_env(self, environment_id, input_seed):
        # If requested, skip tests for envs that requires encoder downloading
        ROBOHIVE_TEST = os.getenv("ROBOHIVE_TEST")
        if ROBOHIVE_TEST == "LITE":
            if (
                "r3m" in environment_id
                or "rrl" in environment_id
                or "vc1" in environment_id
            ):
                return

        # test init
        env1w = gym.make(environment_id, seed=input_seed)
        env1 = env1w.unwrapped
        # seed() first so get_input_seed() is set (Gymnasium envs set it in seed(), not __init__)
        env1.seed(input_seed)
        assert env1.get_input_seed() == input_seed
        reset_obs1, *_ = env1.reset()

        # Support both legacy (mj_model/mj_data) and Gymnasium (model/data)
        mj_model = getattr(env1, "mj_model", getattr(env1, "model", None))
        mj_data = getattr(env1, "mj_data", getattr(env1, "data", None))
        nu = getattr(mj_model, "nu", env1.action_space.shape[0])

        # step
        u = 0.01 * np.random.uniform(low=0, high=1, size=nu)  # small controls
        step_out = env1.step(u.copy())
        obs1, rwd1, done1, *rest = step_out
        infos1 = rest[-1] if rest else {}
        infos1 = copy.deepcopy(infos1)  # info points to internal variables.
        proprio1_t, proprio1_vec, proprio1_dict = env1.get_proprioception()
        extero1 = env1.get_exteroception()
        assert len(obs1) > 0
        # test dicts (get_obs_dict accepts (model, data) for both legacy and Gymnasium)
        assert mj_model is not None and mj_data is not None
        assert len(infos1) > 0
        obs_dict1 = env1.get_obs_dict(mj_model, mj_data)
        assert len(obs_dict1) > 0
        rwd_dict1 = env1.get_reward_dict(obs_dict1)
        assert len(rwd_dict1) > 0
        # reset env
        reset_data = env1.reset()
        self.check_reset(reset_data)

        # serialize / deserialize env ------------
        env2w = pickle.loads(pickle.dumps(env1w))
        env2 = env2w.unwrapped
        # reseed so get_input_seed() is set (deserialized Gymnasium envs may not have it)
        env2.seed(input_seed)
        assert env2.get_input_seed() == input_seed
        assert env1.get_input_seed() == env2.get_input_seed(), {
            env1.get_input_seed(),
            env2.get_input_seed(),
        }
        # check input output spaces
        assert env1.action_space == env2.action_space, (
            env1.action_space,
            env2.action_space,
        )
        assert env1.observation_space == env2.observation_space, (
            env1.observation_space,
            env2.observation_space,
        )

        # reset and compare with env1
        reset_obs2, *_ = env2.reset()
        assert_close(reset_obs1, reset_obs2)

        # step
        obs2, rwd2, done2, *_, infos2 = env2.step(u)
        infos2 = copy.deepcopy(infos2)
        proprio2_t, proprio2_vec, proprio2_dict = env2.get_proprioception()
        extero2 = env2.get_exteroception()

        assert_close(obs1, obs2)
        assert_close(
            proprio1_vec, proprio2_vec
        )  # , f"Difference in Proprio: {proprio1_vec-proprio2_vec}"
        assert_close(
            extero1, extero2, atol=2, rtol=0.04
        )  # , f"Difference in Extero {extero1}, {extero2}"
        assert_close(rwd1, rwd2)  # , "Difference in Rewards"
        assert done1 == done2, (done1, done2)
        assert len(infos1) == len(infos2), (infos1, infos2)
        assert_close(infos1, infos2)
        # reset
        env2.reset()

        del env1
        del env2

    def check_reset(self, reset_data):
        assert (
            isinstance(reset_data, tuple) and len(reset_data) == 2
        ), "Reset should return a tuple of length 2"
        assert isinstance(
            reset_data[1], dict
        ), "second element returned should be a dict"

    def check_old_envs(self, module_name, env_names, lite=False, seed=1234):
        print("\nTesting module:: ", module_name)
        for env_name in env_names:
            print("Testing env: ", env_name)
            # test init
            env = gym.make(env_name)
            env.seed(seed)

            # test reset
            env.env.reset()
            # test obs vec
            env.env.get_obs()

            if not lite:
                # test obs dict
                obs_dict = env.env.get_obs_dict(env.env.sim)
                # test rewards
                env.env.get_reward_dict(obs_dict)

                # test vector => dict upgrade
                # print(env.env.get_obs() - env.env.get_obs_vec())
                # assert (env.env.get_obs() == env.env.get_obs_vec()).all(), "check vectorized computations"

            # test env infos
            env.env.get_env_infos()

            # test step (everything together)
            observation, _reward, done, _info = env.env.step(
                np.zeros(env.env.mj_model.nu)
            )
            del env


if __name__ == "__main__":
    unittest.main()
