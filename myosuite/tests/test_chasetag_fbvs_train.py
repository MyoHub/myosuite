# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""``myoChallengeChaseTagFBVsTrain-v0`` is ``myoChallengeChaseTagFBVs-v0`` seen from ``agent_0``.

For the same actions of both players the CPU train env and FBVs give
bit-identical observations (both players), rewards, terminations and scores,
so a policy trained on the train env is an FBVs player (a 546 -> 354
MyoChallenge submission) as-is.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from myosuite import make_env
from myosuite.core.config import EnvConfig
from myosuite.envs.multi_agent_modular_env import ModularMultiAgentTaskEnv
from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_task_config import (
    ChaseTagVsFullbodySteeredTaskConfig,
)
from myosuite.envs.myo.tasks.challenge.chase_tag_vs.train_env import (
    CHASETAG_VS_OBS_DIM,
    LEARNER,
    OPPONENT,
    load_opponent_policy,
)
from myosuite.envs.wrappers import FatigueWrapper
from myosuite.tests.support.optional_deps import require_musclemimic_models

pytestmark = pytest.mark.tier2

ENV_ID = "myoChallengeChaseTagFBVsTrain-v0"
FBVS_ID = "myoChallengeChaseTagFBVs-v0"
N_ACT = 354


@pytest.fixture(scope="module", autouse=True)
def _models() -> None:
    require_musclemimic_models()


class _Replay:
    """Opponent policy replaying a fixed action sequence (records its inputs)."""

    def __init__(self, actions: np.ndarray) -> None:
        self.actions, self.seen = actions, []

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        self.seen.append(obs.copy())
        return self.actions[len(self.seen) - 1]


def _rollout_both(
    train: gym.Env, fbvs: gym.Env, seed: int, n_steps: int
) -> tuple[int, dict]:
    """Step both envs with the same actions; assert bit-identity every step."""
    rng = np.random.default_rng(seed)
    acts = rng.uniform(0.0, 0.4, (n_steps, 2, N_ACT)).astype(np.float32)
    acts[:, :, :3] = 1.3  # out of range: clipped for physics, raw in the reward
    replay = _Replay(acts[:, 1])
    train.unwrapped.set_opponent_policy(replay)
    obs, info = train.reset(seed=seed)
    obs_f, info_f = fbvs.reset(seed=seed)
    assert info["task"] == info_f["task"]
    np.testing.assert_array_equal(obs, obs_f[LEARNER])
    for t in range(n_steps):
        opp_obs = train.unwrapped.opponent_obs.copy()
        obs, rew, term, trunc, info = train.step(acts[t, 0])
        obs_f, rew_f, term_f, trunc_f, info_f = fbvs.step(
            {LEARNER: acts[t, 0], OPPONENT: acts[t, 1]}
        )
        np.testing.assert_array_equal(replay.seen[t], opp_obs)
        np.testing.assert_array_equal(obs, obs_f[LEARNER], err_msg=str(t))
        np.testing.assert_array_equal(
            train.unwrapped.opponent_obs, obs_f[OPPONENT], err_msg=str(t)
        )
        assert rew == rew_f[LEARNER], t
        assert term == term_f[LEARNER] and trunc == trunc_f[LEARNER], t
        assert info["score"] == info_f["score"] and info["tagged"] == info_f["tagged"]
        assert info["learner_fell"] == info_f["fell"][LEARNER]
        assert info["opponent_fell"] == info_f["fell"][OPPONENT]
        if term:
            return t + 1, info
    return n_steps, info


@pytest.mark.parametrize("seed", [0, 1])
def test_matches_fbvs_agent_0(seed: int) -> None:
    """Observation, opponent observation, reward and termination = FBVs, bit for bit."""
    train, fbvs = make_env(ENV_ID), make_env(FBVS_ID)
    try:
        assert train.observation_space.shape == (CHASETAG_VS_OBS_DIM,)
        assert train.observation_space.dtype == np.float32
        assert np.isinf(train.observation_space.low).all()
        assert train.action_space == fbvs.unwrapped.action_space[LEARNER]
        assert train.unwrapped.dt == pytest.approx(0.01)
        assert train.spec.max_episode_steps == fbvs.unwrapped.max_episode_steps
        steps, info = _rollout_both(train, fbvs, seed, 400)
        assert steps < 400 and (info["learner_fell"] or info["opponent_fell"])
    finally:
        train.close()
        fbvs.close()


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_tag_ends_the_episode_with_the_fbvs_score(seed: int) -> None:
    """Players spawned within the tag radius: the first step tags, as in FBVs."""
    # The standing pose puts each pelvis 0.1 m behind its spawn point: 0.4 m apart.
    config = ChaseTagVsFullbodySteeredTaskConfig(agent_separation_m=0.2)
    train = make_env(ENV_ID, task_config=config)
    fbvs = ModularMultiAgentTaskEnv(
        ChaseTagVsFullbodySteeredTaskConfig(agent_separation_m=0.2)
    )
    try:
        steps, info = _rollout_both(train, fbvs, seed, 3)
        assert steps == 1 and info["tagged"]
        chaser = info["task"] == "CHASE"
        assert info["solved"] == chaser and info["chaser"] == chaser
        assert info["score"] == pytest.approx(1 - 0.01 / 20 if chaser else 0.01 / 20)
    finally:
        train.close()
        fbvs.close()


def test_horizon_truncates_and_a_surviving_runner_wins() -> None:
    """Step 2000 (20 s) truncates; as runner the learner then scores 1."""
    env = make_env(ENV_ID)
    try:
        for seed in range(20):
            _, info = env.reset(seed=seed)
            if info["task"] == "EVADE":
                break
        wrapper = env
        while not isinstance(wrapper, gym.wrappers.TimeLimit):
            wrapper = wrapper.env
        wrapper._elapsed_steps = env.unwrapped.steps = 1999
        _, _, term, trunc, info = env.step(np.zeros(N_ACT, dtype=np.float32))
        assert trunc and not term
        assert info["solved"] and info["score"] == 1.0
    finally:
        env.close()


def test_reset_is_reproducible_and_draws_both_roles() -> None:
    env = make_env(ENV_ID)
    try:
        roles = set()
        for seed in range(8):
            obs_a, info_a = env.reset(seed=seed)
            obs_b, info_b = env.reset(seed=seed)
            np.testing.assert_array_equal(obs_a, obs_b)
            assert info_a == info_b
            roles.add(info_a["task"])
            role = obs_a[-2:].tolist()
            assert role == ([1.0, 0.0] if info_a["task"] == "CHASE" else [0.0, 1.0])
            assert env.unwrapped.opponent_obs[-2:].tolist() == role[::-1]
        assert roles == {"CHASE", "EVADE"}
    finally:
        env.close()


def test_opponent_policies(tmp_path) -> None:
    """Idle opponent, a callable, an ONNX submission; anything else is rejected."""
    torch = pytest.importorskip("torch")
    env = make_env(ENV_ID).unwrapped
    try:
        env.reset(seed=0)
        env.step(np.zeros(N_ACT, dtype=np.float32))
        np.testing.assert_array_equal(env.data.ctrl[env._act[OPPONENT]], 0.0)

        net = torch.nn.Sequential(torch.nn.Linear(CHASETAG_VS_OBS_DIM, N_ACT))
        torch.manual_seed(0)
        path = tmp_path / "player.onnx"
        torch.onnx.export(
            net, torch.zeros(1, CHASETAG_VS_OBS_DIM), str(path), dynamo=False
        )
        policy = load_opponent_policy(path, N_ACT)
        obs = env.opponent_obs
        expected = net(torch.as_tensor(obs)[None]).detach().numpy()[0]
        np.testing.assert_allclose(policy(obs), expected, atol=1e-5)
        env.set_opponent_policy(path)
        env.step(np.zeros(N_ACT, dtype=np.float32))
        np.testing.assert_allclose(
            env.data.ctrl[env._act[OPPONENT]], np.clip(expected, 0, 1), atol=1e-5
        )
        with pytest.raises(ValueError, match=".onnx or .pt"):
            load_opponent_policy(tmp_path / "player.zip", N_ACT)
    finally:
        env.close()


def test_muscle_condition_features_are_rejected() -> None:
    """Two bodies share the model: fatigue and the other conditions are not supported."""
    with pytest.raises(TypeError):
        make_env(EnvConfig(ENV_ID, features=(FatigueWrapper,)))
