# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Challenge ``get_metrics`` keeps the legacy scoring of lost episodes.

Legacy ``run_track_v0`` and ``chasetag_v0`` (CHASE) reported ``maxTime`` as the
episode time of a lost episode, so falling early cannot earn a short time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import myosuite  # noqa: F401
from myosuite.utils import gym

pytestmark = pytest.mark.tier1


def _lost_episode(env_id: str, **kwargs: Any) -> tuple[Any, dict, dict]:
    """Roll out zero actions (the model falls) and stack the infos into a path."""
    env = gym.make(env_id, **kwargs)
    env.reset(seed=0)
    infos = []
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape, np.float32)
        )
        infos.append(info)
    assert terminated and not info["rwd_dict"]["solved"], "the episode must be lost"
    env_infos = {
        group: {k: np.array([i[group][k] for i in infos]) for k in info[group]}
        for group in ("obs_dict", "rwd_dict")
    }
    return env.unwrapped, {"env_infos": env_infos}, info


def test_run_track_lost_episode_reports_max_time() -> None:
    env, path, info = _lost_episode("myoChallengeOslRunFixed-v0")
    sim_time = float(np.ravel(info["obs_dict"]["time"])[-1])
    assert sim_time < env.maxTime  # the observation keeps the simulated time
    assert env.get_metrics([path])["time"] == pytest.approx(env.maxTime)


@pytest.mark.parametrize("task", ["CHASE", "EVADE"])
def test_chasetag_lost_episode_time(task: str) -> None:
    env, path, info = _lost_episode("myoChallengeChaseTagP1-v0", task_choice=task)
    sim_time = float(np.ravel(info["obs_dict"]["time"])[-1])
    assert sim_time < env.maxTime
    # A lost CHASE scores maxTime; an EVADE episode scores the time survived.
    expected = env.maxTime if task == "CHASE" else round(sim_time, 2)
    assert env.get_metrics([path])["times"] == pytest.approx(expected)
