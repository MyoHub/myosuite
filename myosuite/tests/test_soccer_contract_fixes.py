# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for Soccer sparse goal reward, spawn distance, and goalkeeper fallback."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite import make_env

pytestmark = pytest.mark.tier1


def test_soccer_sparse_reward_spawn_distance_and_policy_fallback() -> None:
    env = make_env("myoChallengeSoccerP2-v0", min_agent_spawn_distance=2.5).unwrapped
    try:
        obs, _ = env.reset(seed=7)
        assert obs.shape == env.observation_space.shape
        ball_xy = env.data.body("soccer_ball").xpos[:2]
        root_xy = env.data.joint("root").qpos[:2]
        assert float(np.linalg.norm(root_xy - ball_xy)) >= 2.5 - 1e-6

        env.data.joint("root").qpos[2] = 0.05
        mujoco.mj_forward(env.model, env.data)
        obs_dict = env.get_obs_dict()
        rwd_dict = env.get_reward_dict(obs_dict)
        assert rwd_dict["done"] is True
        assert rwd_dict["solved"] == 0.0
        assert rwd_dict["sparse"] == 0.0

        ball_jnt = int(env.model.body_jntadr[env.soccer_ball_id])
        ball_adr = int(env.model.jnt_qposadr[ball_jnt])
        env.data.qpos[ball_adr : ball_adr + 3] = [
            env.GOAL_X_POS + 0.1,
            0.0,
            0.2,
        ]
        env.data.joint("root").qpos[2] = 0.92
        mujoco.mj_forward(env.model, env.data)
        rwd_goal = env.get_reward_dict(env.get_obs_dict())
        assert rwd_goal["solved"] == 1.0
        assert rwd_goal["sparse"] == 1.0

        gk = env.goalkeeper
        gk.goalkeeper_probabilities = (0.1, 0.1, 0.1)
        orig_rng = gk.rng
        gk.rng = type("_Rng", (), {"uniform": lambda self: 0.99})()
        gk.sample_goalkeeper_policy()
        assert gk.goalkeeper_policy == "stationary"
        gk.rng = orig_rng
    finally:
        env.close()
