# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for ChaseTag dense reward learnability."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.tier1


def test_chasetag_default_weights_include_alive() -> None:
    from myosuite.envs.myo.tasks.challenge.chasetag import ChaseTagEnv

    weights = ChaseTagEnv.DEFAULT_RWD_KEYS_AND_WEIGHTS
    assert "alive" in weights
    assert weights["alive"] == pytest.approx(0.5)
    assert weights["distance"] == pytest.approx(-0.5)
    assert weights["solved"] == pytest.approx(1000.0)
    assert weights["lose"] == pytest.approx(-100.0)


def test_chasetag_distance_is_potential_based() -> None:
    """``distance`` must be Δrange, not absolute range (anti-survival bug)."""
    import myosuite

    myosuite.register_all_envs()
    from myosuite.utils import gym

    env = gym.make("myoChallengeChaseTagP1-v0")
    try:
        env.reset(seed=0)
        unwrapped = env.unwrapped
        obs_dict = unwrapped._get_obs_dict(unwrapped._accessor)
        unwrapped._prev_distance = 2.0
        r1 = unwrapped.get_reward_dict(obs_dict)
        abs1 = float(r1["distance_abs"])
        # Force a known previous absolute distance and recompute.
        unwrapped._prev_distance = abs1 + 0.25
        r2 = unwrapped.get_reward_dict(obs_dict)
        assert float(r2["distance"]) == pytest.approx(-0.25, abs=1e-6)
        assert float(r2["alive"]) == pytest.approx(1.0)
        dense = float(r2["dense"])
        # Potential approach + alive must dominate lose-free steps.
        expected = (-0.5) * float(r2["distance"]) + 0.5 * float(r2["alive"])
        assert dense == pytest.approx(expected, abs=1e-6)
    finally:
        env.close()


def test_chasetag_evade_flips_distance_potential() -> None:
    """On EVADE, ``distance`` rewards fleeing (negated Δ range)."""
    import myosuite
    from myosuite.envs.myo.tasks.challenge.chasetag import Task

    myosuite.register_all_envs()
    from myosuite.utils import gym

    env = gym.make("myoChallengeChaseTagP1-v0")
    try:
        env.reset(seed=0)
        unwrapped = env.unwrapped
        unwrapped.current_task = Task.EVADE
        obs_dict = unwrapped._get_obs_dict(unwrapped._accessor)
        abs1 = float(
            np.linalg.norm(
                obs_dict["model_root_pos"][..., :2] - obs_dict["opponent_pose"][..., :2]
            )
        )
        unwrapped._prev_distance = abs1 + 0.25  # curr closer than prev → raw Δ=-0.25
        r = unwrapped.get_reward_dict(obs_dict)
        # Negated Δ is +0.25; weight -0.1 punishes closing in during EVADE.
        assert float(r["distance"]) == pytest.approx(0.25, abs=1e-6)
    finally:
        env.close()


def test_chasetag_alive_requires_upright() -> None:
    """Fallen postures must not earn ``alive`` (EVADE does not end on fall)."""
    import mujoco
    import myosuite

    myosuite.register_all_envs()
    from myosuite.utils import gym

    env = gym.make("myoChallengeChaseTagP1-v0")
    try:
        env.reset(seed=0)
        unwrapped = env.unwrapped
        unwrapped.data.qpos[2] = 0.05
        mujoco.mj_forward(unwrapped.model, unwrapped.data)
        assert unwrapped._get_fallen_condition() == 1
        obs_dict = unwrapped._get_obs_dict(unwrapped._accessor)
        unwrapped._prev_distance = float(
            np.linalg.norm(
                obs_dict["model_root_pos"][..., :2] - obs_dict["opponent_pose"][..., :2]
            )
        )
        rwd = unwrapped.get_reward_dict(obs_dict)
        assert float(rwd["alive"]) == pytest.approx(0.0)
    finally:
        env.close()


def test_chasetag_evade_fall_is_lose() -> None:
    """EVADE must terminate on fall (challenge / docs contract)."""
    import mujoco
    import myosuite
    from myosuite.envs.myo.tasks.challenge.chasetag import Task

    myosuite.register_all_envs()
    from myosuite.utils import gym

    env = gym.make("myoChallengeChaseTagP1-v0")
    try:
        env.reset(seed=0)
        unwrapped = env.unwrapped
        unwrapped.current_task = Task.EVADE
        unwrapped.data.qpos[2] = 0.05
        mujoco.mj_forward(unwrapped.model, unwrapped.data)
        obs_dict = unwrapped._get_obs_dict(unwrapped._accessor)
        assert unwrapped._lose_condition(obs_dict) == 1
        assert unwrapped._get_done(obs_dict) == 1
    finally:
        env.close()


def test_chasetag_alive_keeps_tag_better_than_timeout() -> None:
    """``alive`` ≤ 0.5 so an early tag still beats surviving a CHASE timeout."""
    from myosuite.envs.myo.tasks.challenge.chasetag import ChaseTagEnv

    w = ChaseTagEnv.DEFAULT_RWD_KEYS_AND_WEIGHTS
    alive_w = float(w["alive"])
    lose_w = float(w["lose"])
    solved_w = float(w["solved"])
    assert alive_w <= 0.5 + 1e-9
    timeout_return = alive_w * 2000 + lose_w  # CHASE timeout = lose, not solved
    early_tag_return = alive_w * 100 + solved_w  # tag: solved, no lose
    assert early_tag_return > timeout_return


def test_chasetag_surviving_beats_immediate_fall_return() -> None:
    """Longer upright rollouts must accumulate more return than instant fall."""
    import myosuite

    myosuite.register_all_envs()
    from myosuite.utils import gym

    env = gym.make("myoChallengeChaseTagP1-v0")
    try:
        env.reset(seed=1)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        returns = []
        for horizon in (5, 40):
            env.reset(seed=1)
            total = 0.0
            for _ in range(horizon):
                _obs, reward, terminated, truncated, _info = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
            returns.append(total)
        assert returns[1] > returns[0] + 5.0
    finally:
        env.close()
