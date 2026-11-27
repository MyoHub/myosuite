# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for Bimanual side-effect-free obs/reward dicts and drop time shape."""

from __future__ import annotations

import numpy as np
import pytest

from myosuite import make_env

pytestmark = pytest.mark.tier1


def test_bimanual_pure_obs_reward_and_drop_time_shape() -> None:
    env = make_env("myoChallengeBimanual-v0").unwrapped
    try:
        env.reset(seed=0)
        hist_len = len(env.touch_history)
        goal_touch = env.goal_touch
        for _ in range(3):
            obs_d = env.get_obs_dict()
            _ = env.get_reward_dict(obs_d)
        assert len(env.touch_history) == hist_len
        assert env.goal_touch == goal_touch

        custom_obs = env.get_obs_dict()
        custom_obs["obj_pos"] = np.array([0.0, 0.0, 0.1])
        rwd_d = env.get_reward_dict(custom_obs)
        assert rwd_d["done"] == 1
        assert isinstance(custom_obs["time"], np.ndarray)
        assert custom_obs["time"].shape == (1,)
    finally:
        env.close()
