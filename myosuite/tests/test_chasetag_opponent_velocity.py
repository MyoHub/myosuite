# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The observed ChaseTag ``opponent_vel`` is the velocity the opponent moves with.

``move_opponent`` caps the ``[lin_vel, rot_vel]`` command at 2 m/s and 2 rad/s.
In EVADE the opponent chases with a turn rate proportional to the distance and,
in P2eval, a speed drawn from (1, 5) m/s, so both caps are active.
"""

from __future__ import annotations

import numpy as np
import pytest

from myosuite import make_env

pytestmark = pytest.mark.tier1


@pytest.mark.parametrize(
    "env_id", ["myoChallengeChaseTagP2-v0", "myoChallengeChaseTagP2eval-v0"]
)
def test_observed_opponent_velocity_is_the_applied_one(env_id: str) -> None:
    env = make_env(env_id, task_choice="EVADE")
    unwrapped = env.unwrapped
    opponent = unwrapped.opponent
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    try:
        speeds = []
        for seed in range(6):
            env.reset(seed=seed)
            assert opponent.opponent_policy == "chase_player"
            # 3 m from the agent towards the arena centre, clear of the walls.
            agent = unwrapped.data.body("pelvis").xpos[:2].copy()
            towards_centre = -agent / max(np.linalg.norm(agent), 1e-9)
            opponent.set_opponent_pose([*(agent + 3.0 * towards_centre), 0.4])
            before = opponent.get_opponent_pose()
            info = env.step(action)[4]
            after = opponent.get_opponent_pose()
            turn = np.angle(np.exp(1j * (after[2] - before[2])))
            step = np.array([np.linalg.norm(after[:2] - before[:2]), turn])
            applied = step / unwrapped.dt
            observed = info["obs_dict"]["opponent_vel"]
            np.testing.assert_allclose(observed, applied, atol=1e-6)
            speeds.append(opponent.chase_velocity)
        assert np.all(np.abs(observed) <= 2.0)
    finally:
        env.close()
    if env_id.endswith("P2eval-v0"):
        assert max(speeds) > 2.0  # the speed cap was exercised
