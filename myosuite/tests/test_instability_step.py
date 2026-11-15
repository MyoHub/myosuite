# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""A step whose state MuJoCo auto-reset is flagged and never scored as a success.

``mj_step`` answers a bad ``qvel`` with ``mjWARN_BADQVEL`` and resets the data
to ``qpos0``. The reward dict of that step describes the reset state: in
ChaseTag the agent and the opponent are then both at the origin, a tag.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from myosuite import make_env
from myosuite.envs.wrappers import MjInstabilityTerminationWrapper

pytestmark = pytest.mark.tier1


def _diverged_step(env: gym.Env) -> tuple[float, bool, bool, dict[str, Any]]:
    """Reset, take one regular step, then one step from a NaN velocity."""
    env.reset(seed=0)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    assert not env.step(action)[2]
    env.unwrapped.data.qvel[:] = np.nan
    _, reward, terminated, truncated, info = env.step(action)
    return reward, terminated, truncated, info


@pytest.mark.parametrize(
    "env_id",
    [
        "myoChallengeChaseTagP1-v0",  # dev: reward +1002.8, solved, score 0.9995
        "myoTorsoPoseFixed-v0",  # dev: reward +8.0, solved
        "myoElbowPose1D6MRandom-v0",  # negative reward: stays as computed
    ],
)
def test_diverged_step_is_flagged_and_not_scored(env_id: str) -> None:
    env = make_env(env_id)
    try:
        reward, terminated, truncated, info = _diverged_step(env)
        assert terminated and not truncated
        assert reward <= 0.0
        assert info["rwd_dict"]["dense"] == reward
        assert not np.any(info["solved"])
        assert not np.any(info["rwd_dict"]["solved"])
        assert np.all(np.asarray(info["rwd_dict"]["sparse"]) <= 0.0)
        assert info["mj_unstable"] is True

        env.reset(seed=0)  # mj_resetData clears the warning counters
        _, _, terminated, _, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        assert not terminated and info["mj_unstable"] is False
    finally:
        env.close()


def test_chasetag_metrics_score_a_diverged_episode_as_lost() -> None:
    env = make_env("myoChallengeChaseTagP1-v0")
    try:
        _, _, _, info = _diverged_step(env)
        unwrapped = env.unwrapped
        env_infos = {
            group: {k: np.array([v]) for k, v in info[group].items()}
            for group in ("obs_dict", "rwd_dict")
        }
        metrics = unwrapped.get_metrics([{"env_infos": env_infos}])
        assert metrics["points"] == 0.0
        assert metrics["score"] == 0.0
        assert metrics["times"] == pytest.approx(unwrapped.maxTime)
    finally:
        env.close()


def test_env_failure_penalty_is_kept() -> None:
    """``WaypointEnv`` already fails a diverged step with its own penalty."""
    env = make_env("myoFullBodyWaypoint-v0")
    try:
        reward, terminated, _, info = _diverged_step(env)
        penalty = env.unwrapped.task.reward.failure
        assert penalty > 0
        assert terminated
        assert reward == pytest.approx(-penalty, abs=0.1)
        assert info["mj_unstable"] is True
    finally:
        env.close()


def test_unwrapped_env_voids_every_step_after_a_divergence() -> None:
    """``examine_policy`` steps the unwrapped env: no termination, no success."""
    env = make_env("myoTorsoPoseFixed-v0").unwrapped
    try:
        reward, _, _, info = _diverged_step(env)
        assert reward <= 0.0 and not np.any(info["solved"])
        assert info["mj_unstable"] is True
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, reward, _, _, info = env.step(action)
        assert reward <= 0.0 and not np.any(info["solved"])
        assert info["mj_unstable"] is True
    finally:
        env.close()


def test_opt_out_keeps_the_env_reward() -> None:
    env = make_env("myoTorsoPoseFixed-v0")
    try:
        env.unwrapped.mj_instability_termination = False
        reward, terminated, _, info = _diverged_step(env)
        assert not terminated
        assert reward > 0.0 and np.all(info["solved"])
        assert info["mj_unstable"] is False
    finally:
        env.close()


class _DivergedEnv(gym.Env):
    """An env whose step does not end in ``MyoGymnasiumEnv._finalize_step``."""

    mj_instability_termination = True
    action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
    observation_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

    def _check_mj_instability_termination(self) -> bool:
        return True

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        return np.zeros(1, np.float32), 5.0, False, False, {}


def test_wrapper_handles_an_env_without_the_flag() -> None:
    env = MjInstabilityTerminationWrapper(_DivergedEnv())
    _, reward, terminated, truncated, info = env.step(np.zeros(1, np.float32))
    assert terminated and not truncated
    assert reward == 0.0
    assert info["mj_unstable"] is True
