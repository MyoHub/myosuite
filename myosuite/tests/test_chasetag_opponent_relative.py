# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""``opponent_relative`` of myoChallengeChaseTagFBP2-v0 holds a Cartesian velocity.

``rel_vel`` is the opponent's world-frame velocity minus the agent's root
velocity; the opponent's velocity is its displacement over the step.
"""

from __future__ import annotations

import numpy as np
import pytest

from myosuite import make_env

pytestmark = pytest.mark.tier1


def test_relative_velocity_is_the_opponent_displacement() -> None:
    env = make_env("myoChallengeChaseTagFBP2-v0")
    unwrapped = env.unwrapped
    opponent = unwrapped.opponent
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    try:
        info = None
        env.reset(seed=0)
        opponent.opponent_policy = "random"  # moves and turns
        opponent.set_opponent_pose([3.0, 3.0, 0.5])  # clear of agent and walls
        for _ in range(5):
            before = opponent.get_opponent_pose()[:2]
            _, _, terminated, _, info = env.step(action)
            assert not terminated
            rel_vel = info["obs_dict"]["opponent_relative"][3:6]
            opponent_vel = rel_vel + unwrapped.data.qvel[:3].astype(np.float32)
            moved = opponent.get_opponent_pose()[:2] - before
            expected = np.r_[moved / unwrapped.dt, 0.0]
            np.testing.assert_allclose(opponent_vel, expected, atol=1e-4)
        assert np.linalg.norm(opponent_vel[:2]) > 0.1
        env.reset(seed=0)  # no motion before the first step
        rel_vel = unwrapped._get_obs_dict(unwrapped._accessor)["opponent_relative"]
        np.testing.assert_allclose(
            rel_vel[3:6], -unwrapped.data.qvel[:3].astype(np.float32), atol=1e-6
        )
    finally:
        env.close()
