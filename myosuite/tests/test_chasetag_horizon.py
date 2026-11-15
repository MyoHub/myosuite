# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""ChaseTag outcomes at the time limit: EVADE is won, CHASE is not.

The episode clock is the control-step count. The summed ``data.time`` after
the last step is 20.00000000000146 s (20 s envs) or 10.000000000000009 s
(10 s envs), so a float comparison against ``maxTime`` is not a horizon rule.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from myosuite import make_env

pytestmark = pytest.mark.tier1


def _rollout_to_horizon(env_id: str, role: str) -> tuple[Any, list[dict], tuple]:
    """Play *role* until the episode ends: frozen far opponent, agent never falls."""
    env = make_env(env_id, task_choice=role)
    unwrapped = env.unwrapped
    env.reset(seed=0)
    unwrapped.opponent.opponent_policy = "stationary"
    far_corner = -5.0 * np.where(unwrapped.data.qpos[:2] >= 0, 1.0, -1.0)
    unwrapped.opponent.set_opponent_pose([*far_corner, 0.0])
    unwrapped._get_fallen_condition = lambda: 0
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    infos: list[dict] = []
    terminated = truncated = False
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(action)
        infos.append(info)
    assert len(infos) == env.spec.max_episode_steps
    assert unwrapped.maxTime == pytest.approx(len(infos) * unwrapped.dt)
    env.close()
    return unwrapped, infos, (reward, terminated, truncated)


def _metrics(unwrapped: Any, infos: list[dict]) -> dict[str, float]:
    env_infos = {
        "obs_dict": {
            k: np.array([i["obs_dict"][k] for i in infos]) for k in ("time", "task")
        },
        "rwd_dict": {
            k: np.array([i["rwd_dict"][k] for i in infos])
            for k in ("sparse", "solved", "lose")
        },
    }
    return unwrapped.get_metrics([{"env_infos": env_infos}])


@pytest.mark.parametrize(
    "env_id", ["myoChallengeChaseTagP2-v0", "myoChallengeChaseTagFBP2-v0"]
)
def test_evade_is_won_on_the_last_step(env_id: str) -> None:
    unwrapped, infos, (reward, terminated, _) = _rollout_to_horizon(env_id, "EVADE")
    last = infos[-1]
    assert not any(i["solved"] or i["lose"] for i in infos[:-1])
    assert last["solved"] == 1 and last["lose"] == 0
    assert terminated
    assert last["sparse"] == 1.0
    assert reward == pytest.approx(unwrapped.rwd_keys_wt["solved"], abs=1.0)
    metrics = _metrics(unwrapped, infos)
    assert metrics["points"] == 1.0
    assert metrics["score"] == 1.0
    assert metrics["times"] == pytest.approx(unwrapped.maxTime)


def test_chase_time_out_is_not_a_win() -> None:
    """A CHASE time-out is truncated: no tag, no score, the full ``maxTime``."""
    unwrapped, infos, (reward, terminated, truncated) = _rollout_to_horizon(
        "myoChallengeChaseTagP1-v0", "CHASE"
    )
    assert truncated and not terminated
    assert not any(i["solved"] or i["lose"] for i in infos)
    assert abs(reward) < 1.0
    metrics = _metrics(unwrapped, infos)
    assert metrics["points"] == 0.0
    assert metrics["score"] == 0.0
    assert metrics["times"] == pytest.approx(unwrapped.maxTime)
