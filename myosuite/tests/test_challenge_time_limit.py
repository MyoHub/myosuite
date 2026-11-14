# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Soccer and Bimanual report their time limit as a truncation, not a termination.

Learners that bootstrap on ``truncated`` must not see the time-out as a real
end of the task (5-tuple contract).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from myosuite import make_env

pytestmark = pytest.mark.tier1


def _run_to_the_end(env: Any) -> tuple[list[dict], bool, bool]:
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    infos: list[dict] = []
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(action)
        infos.append(info)
    return infos, terminated, truncated


def test_soccer_time_out_is_a_truncation_after_max_time() -> None:
    env = make_env("myoChallengeSoccerP1-v0")
    unwrapped = env.unwrapped
    try:
        env.reset(seed=0)
        unwrapped._get_fallen_condition = lambda: 0  # an agent that never falls
        infos, terminated, truncated = _run_to_the_end(env)
        assert truncated and not terminated
        assert len(infos) == env.spec.max_episode_steps
        assert len(infos) * unwrapped.dt == pytest.approx(unwrapped.max_time)
        assert not any(i["rwd_dict"]["done"] for i in infos)
        path = {
            "env_infos": {
                g: {k: np.array([i[g][k] for i in infos]) for k in infos[0][g]}
                for g in ("obs_dict", "rwd_dict")
            }
        }
        metrics = unwrapped.get_metrics([path])
        assert metrics["score"] == 0.0 and metrics["points"] == 0.0
        assert metrics["times"] == pytest.approx(unwrapped.max_time)
    finally:
        env.close()


def test_soccer_truncates_itself_after_max_time_sec() -> None:
    """Without ``TimeLimit`` the env truncates after ``max_time_sec / dt`` steps."""
    env = make_env("myoChallengeSoccerP1-v0", max_time_sec=2.0).unwrapped
    try:
        env.reset(seed=0)
        env._get_fallen_condition = lambda: 0
        infos, terminated, truncated = _run_to_the_end(env)
        assert truncated and not terminated
        assert len(infos) == round(2.0 / env.dt)
    finally:
        env.close()


def test_bimanual_time_out_is_a_truncation() -> None:
    env = make_env("myoChallengeBimanual-v0")
    try:
        env.reset(seed=0)
        infos, terminated, truncated = _run_to_the_end(env)
        assert truncated and not terminated
        assert len(infos) == env.spec.max_episode_steps
        assert not np.any(infos[-1]["rwd_dict"]["done"])
    finally:
        env.close()
