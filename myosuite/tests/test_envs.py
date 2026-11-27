# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import pytest

from myosuite import make_env
from myosuite.benchmarks.env_check import (
    check_reset_tuple,
    check_single_env,
    is_multi_agent_env_spec,
)
from myosuite.core.hf_io import is_gated_or_transient_hf_error
from myosuite.utils import gym


pytestmark = pytest.mark.tier2


class TestEnvs(unittest.TestCase):
    def check_envs(self, module_name, env_names, lite=False, input_seed=1234):
        print("\n=================================", flush=True)
        print("Testing module:: ", module_name)
        assert env_names, f"{module_name}: no envs to test"
        from myosuite.envs.myo.tasks.mimic.tracking import MissingMotionData

        for env_name in env_names:
            if is_multi_agent_env_spec(gym.registry[env_name]):
                print("Skipping multi-agent env: ", env_name, flush=True)
                continue
            print("Testing env: ", env_name, flush=True)
            try:
                self.check_env(env_name, input_seed)
            except MissingMotionData as exc:
                print(f"Skipping {env_name}: motion clips unavailable ({exc})")
            except Exception as exc:
                if is_gated_or_transient_hf_error(exc):
                    print(f"Skipping {env_name}: gated/transient HF error ({exc})")
                else:
                    raise

    def check_env(self, environment_id, input_seed):
        check_single_env(environment_id, input_seed, respect_lite_env=True)

    def check_reset(self, reset_data):
        check_reset_tuple(reset_data)

    def check_old_envs(self, module_name, env_names, lite=False, seed=1234):
        print("\nTesting module:: ", module_name)
        for env_name in env_names:
            print("Testing env: ", env_name)
            env = make_env(env_name)
            env.seed(seed)
            env.env.reset()
            env.env.get_obs()
            if not lite:
                obs_dict = env.env.get_obs_dict(env.env.sim)
                env.env.get_reward_dict(obs_dict)
            env.env.get_env_infos()
            env.env.step(np.zeros(env.env.mj_model.nu))
            del env


if __name__ == "__main__":
    unittest.main()
