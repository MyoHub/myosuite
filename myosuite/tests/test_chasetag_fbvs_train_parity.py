# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The single-learner train env's observation is byte-identical to FBVs agent_0.

See https://github.com/MyoHub/myosuite/issues/576: FBP2 is 537-d and is not
the submission contract. ``myoChallengeChaseTagFBVsTrain-v0`` must match
``myoChallengeChaseTagFBVs-v0`` ``agent_0`` from the same state.
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.tests.support.optional_deps import require_musclemimic_models

pytestmark = pytest.mark.tier2

TRAIN_ID = "myoChallengeChaseTagFBVsTrain-v0"
FBVS_ID = "myoChallengeChaseTagFBVs-v0"


def _make(env_id: str):
    require_musclemimic_models()
    try:
        return make_env(env_id)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ChaseTag full-body env unavailable ({exc})")


def test_train_env_is_registered() -> None:
    require_musclemimic_models()
    assert gym.spec(TRAIN_ID) is not None


def test_train_obs_dim_is_fbvs_agent_dim() -> None:
    from myosuite.terms.multiplayer.chase_tag_vs_obs import FBVS_AGENT_OBS_DIM

    env = _make(TRAIN_ID)
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (FBVS_AGENT_OBS_DIM,)
        assert env.observation_space.contains(obs)
        assert env.action_space.shape == (354,)
        assert "task" in info
    finally:
        env.close()


def test_learner_obs_matches_fbvs_agent_0_from_the_same_state() -> None:
    train = _make(TRAIN_ID)
    fbvs = _make(FBVS_ID)
    try:
        train.reset(seed=1)
        inner = train.unwrapped.unwrapped_vs
        raw_vs = fbvs.unwrapped
        assert inner.model.nq == raw_vs.model.nq
        raw_vs.data.qpos[:] = inner.data.qpos
        raw_vs.data.qvel[:] = inner.data.qvel
        if inner.model.na:
            raw_vs.data.act[:] = inner.data.act
        mujoco.mj_forward(raw_vs.model, raw_vs.data)
        raw_vs._config.current_task = inner._config.current_task

        expected = raw_vs._config.get_obs(
            raw_vs.model, raw_vs.data, raw_vs._meta, "agent_0"
        )
        got = train.unwrapped._learner_obs()
        np.testing.assert_array_equal(got, expected)

        action = np.full(train.action_space.shape, 0.05, dtype=np.float32)
        train.step(action)
        raw_vs.data.qpos[:] = inner.data.qpos
        raw_vs.data.qvel[:] = inner.data.qvel
        if inner.model.na:
            raw_vs.data.act[:] = inner.data.act
        mujoco.mj_forward(raw_vs.model, raw_vs.data)
        raw_vs._config.current_task = inner._config.current_task
        expected = raw_vs._config.get_obs(
            raw_vs.model, raw_vs.data, raw_vs._meta, "agent_0"
        )
        got = train.unwrapped._learner_obs()
        np.testing.assert_array_equal(got, expected)
    finally:
        train.close()
        fbvs.close()


def test_train_env_runs_a_few_steps() -> None:
    env = _make(TRAIN_ID)
    try:
        env.reset(seed=0)
        action = np.full(env.action_space.shape, 0.05, dtype=np.float32)
        for _ in range(8):
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(obs).all()
            assert np.isfinite(reward)
            assert obs.shape == env.observation_space.shape
            if terminated or truncated:
                env.reset()
    finally:
        env.close()
