# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for TableTennis pure obs/reward, ball_qvel, and multi-rally relaunch."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.envs.myo.tasks.challenge.tabletennis import PingpongContactLabels

pytestmark = pytest.mark.tier1


def test_tabletennis_pure_obs_reward_and_fresh_relaunch_state() -> None:
    env = make_env(
        "myoChallengeTableTennisP0-v0", ball_qvel=True, rally_count=2
    ).unwrapped
    try:
        env.reset(seed=3)
        assert not np.allclose(
            env.data.qvel[env.ball_dofadr : env.ball_dofadr + 3],
            env.start_vel.ravel(),
        )

        traj_len = len(env.contact_trajectory)
        for _ in range(3):
            obs_d = env.get_obs_dict()
            _ = env.get_reward_dict(obs_d)
        assert len(env.contact_trajectory) == traj_len
        assert env.cur_rally == 0

        own = PingpongContactLabels.OWN
        pad = PingpongContactLabels.PADDLE
        opp = PingpongContactLabels.OPPONENT
        env.contact_trajectory = [{own}, set(), {pad}, set(), {opp}]
        _, _, terminated, _, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        assert not terminated
        assert env.cur_rally == 1
        ref = mujoco.MjData(env.model)
        mujoco.mj_copyData(ref, env.model, env.data)
        mujoco.mj_forward(env.model, ref)
        np.testing.assert_array_equal(env.data.site_xpos, ref.site_xpos)
        np.testing.assert_allclose(
            info["obs_dict"]["ball_pos"],
            env.data.site_xpos[env.id_info.ball_sid],
        )
    finally:
        env.close()
