""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import unittest

import gym
import numpy as np
import pickle

class TestEnvs(unittest.TestCase):

    def check_envs(self, module_name, env_names, lite=False, input_seed=1234):
        print("\nTesting module:: ", module_name)
        for env_name in env_names:
            print("Testing env: ", env_name)
            self.check_env(env_name, input_seed)


    def check_env(self, environment_id, input_seed):
        # test init
        env1 = gym.make(environment_id, seed=input_seed)
        assert env1.get_input_seed() == input_seed
        # test reset
        env1.env.reset()

        # step
        u = 0.01*np.random.uniform(low=0, high=1, size=env1.env.sim.model.nu) # small controls
        obs1, rwd1, done1, infos1 = env1.env.step(u.copy())
        assert len(obs1>0)
        # assert len(rwd1>0)
        # test dicts
        assert len(infos1) > 0
        obs_dict1 = env1.get_obs_dict(env1.env.sim)
        assert len(obs_dict1) > 0
        rwd_dict1 = env1.get_reward_dict(obs_dict1)
        assert len(rwd_dict1) > 0
        # reset env
        env1.reset()

        # serialize / deserialize env ------------
        env2 = pickle.loads(pickle.dumps(env1))
        # test reset
        env2.reset()
        # test seed
        assert env2.get_input_seed() == input_seed
        assert env1.get_input_seed() == env2.get_input_seed(), {env1.get_input_seed(), env2.get_input_seed()}
        # check input output spaces
        assert env1.action_space == env2.action_space, (env1.action_space, env2.action_space)
        assert env1.observation_space == env2.observation_space, (env1.observation_space, env2.observation_space)
        # step
        obs2, rwd2, done2, infos2 = env2.env.step(u)
        assert (obs1==obs2).all(), (obs1, obs2)
        assert (rwd1==rwd2).all(), (rwd1, rwd2)
        assert (done1==done2), (done1, done2)
        assert len(infos1)==len(infos2), (infos1, infos2)
        # reset
        env2.reset()

        del(env1)
        del(env2)

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
            obs = env.env.get_obs()

            if not lite:
                # test obs dict
                obs_dict = env.env.get_obs_dict(env.env.sim)
                # test rewards
                rwd = env.env.get_reward_dict(obs_dict)

                # test vector => dict upgrade
                # print(env.env.get_obs() - env.env.get_obs_vec())
                # assert (env.env.get_obs() == env.env.get_obs_vec()).all(), "check vectorized computations"

            # test env infos
            infos = env.env.get_env_infos()

            # test step (everything together)
            observation, _reward, done, _info = env.env.step(np.zeros(env.env.sim.model.nu))
            del(env)


if __name__ == '__main__':
    unittest.main()