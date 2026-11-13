# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for CPU environment contracts, resets, and wrappers."""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
import pytest

import myosuite
from myosuite import make_env
from myosuite.envs.heightfields import ChaseTagField, TerrainTypes
from myosuite.envs.multi_agent_modular_env import ModularMultiAgentSingleAgentWrapper
from myosuite.envs.myo.tasks.challenge.tabletennis import PingpongContactLabels
from myosuite.envs.wrappers import DictObservationWrapper, PerturbationWrapper

pytestmark = pytest.mark.tier1


def test_myoedits_suite_populates_arm_reach_ids() -> None:
    suites = myosuite.register_all_envs()
    expected = sorted(
        [
            "myoArmReachFixed-v0",
            "myoArmReachRandom-v0",
            "myoFatiArmReachFixed-v0",
            "myoFatiArmReachRandom-v0",
            "myoSarcArmReachFixed-v0",
            "myoSarcArmReachRandom-v0",
        ]
    )
    assert suites["myoedits"] == expected
    assert myosuite.myosuite_myoedits_suite == expected


def test_heightfield_size_and_chasetag_flatten_agent_patch_axes() -> None:
    env = make_env("myoChallengeChaseTagP2-v0").unwrapped
    try:
        hf: ChaseTagField = env.heightfield
        assert hf.size == hf.nrow * hf.ncol

        hf.hfield.data[:, :] = 1.0
        # Place agent in top-right quadrant (x=+4.5 -> col patch 2, y=-4.5 -> row patch 0)
        qpos = np.zeros(7, dtype=np.float64)
        qpos[0] = 4.5
        qpos[1] = -4.5
        hf.flatten_agent_patch(qpos)
        ps = hf.patch_size
        np.testing.assert_array_equal(hf.hfield.data[0:ps, 2 * ps : 3 * ps], 0.0)
        assert float(hf.hfield.data[2 * ps : 3 * ps, 0:ps].sum()) > 0.0
        assert TerrainTypes.FLAT is not None
    finally:
        env.close()


def test_chasetag_repeller_equilibrium_fallback_and_single_keyframe_clamp() -> None:
    env = make_env("myoChallengeChaseTagP2eval-v0").unwrapped
    try:
        opp = env.opponent
        opp.opponent_probabilities = (0.1, 0.2, 0.2, 0.1)  # sum < 1.0
        opp.rng = np.random.default_rng(0)
        # Force draw >= 0.6 by mocking uniform once
        orig_rng = opp.rng

        class _HighRng:
            def uniform(self, low=0.0, high=1.0, size=None):
                if low == 0.0 and high == 1.0 and size is None:
                    return 0.99
                return orig_rng.uniform(low=low, high=high, size=size)

        opp.rng = _HighRng()
        opp.sample_opponent_policy()
        assert opp.opponent_policy == "repeller"
        opp.rng = orig_rng

        # Place agent and opponent at exact same XY near origin so escape_xpos == 0
        env.reset(seed=0)
        agent_xy = opp.get_agent_pos().copy()
        opp.set_opponent_pose([float(agent_xy[0]), float(agent_xy[1]), 0.0])
        vel = opp.repeller_stochastic()
        assert vel.shape == (2,)
        assert np.all(np.isfinite(vel))
    finally:
        env.close()

    fb_env = make_env("myoChallengeChaseTagFBP2-v0").unwrapped
    try:
        assert fb_env.model.nkey == 1
        fb_env.reset_type = "init"
        qpos_init, _ = fb_env._get_reset_state()
        assert qpos_init.shape == (fb_env.model.nq,)
        fb_env.reset_type = "random"
        qpos_rand, _ = fb_env._get_reset_state()
        assert qpos_rand.shape == (fb_env.model.nq,)
    finally:
        fb_env.close()


def test_soccer_sparse_reward_spawn_distance_and_policy_fallback() -> None:
    env = make_env("myoChallengeSoccerP2-v0", min_agent_spawn_distance=2.5).unwrapped
    try:
        obs, _ = env.reset(seed=7)
        assert obs.shape == env.observation_space.shape
        ball_xy = env.data.body("soccer_ball").xpos[:2]
        root_xy = env.data.joint("root").qpos[:2]
        assert float(np.linalg.norm(root_xy - ball_xy)) >= 2.5 - 1e-6

        # Force fall without scoring: sparse must be 0.0, not 1.0
        env.data.joint("root").qpos[2] = 0.05
        mujoco.mj_forward(env.model, env.data)
        obs_dict = env.get_obs_dict()
        rwd_dict = env.get_reward_dict(obs_dict)
        assert rwd_dict["done"] is True
        assert rwd_dict["solved"] == 0.0
        assert rwd_dict["sparse"] == 0.0

        # Force ball inside goal: sparse and solved must both be 1.0
        ball_jnt = int(env.model.body_jntadr[env.soccer_ball_id])
        ball_adr = int(env.model.jnt_qposadr[ball_jnt])
        env.data.qpos[ball_adr : ball_adr + 3] = [
            env.GOAL_X_POS + 0.1,
            0.0,
            0.2,
        ]
        env.data.joint("root").qpos[2] = 0.92
        mujoco.mj_forward(env.model, env.data)
        rwd_goal = env.get_reward_dict(env.get_obs_dict())
        assert rwd_goal["solved"] == 1.0
        assert rwd_goal["sparse"] == 1.0

        # Goalkeeper policy fallback when sum(probabilities) < 1.0
        gk = env.goalkeeper
        gk.goalkeeper_probabilities = (0.1, 0.1, 0.1)
        orig_rng = gk.rng
        gk.rng = type("_Rng", (), {"uniform": lambda self: 0.99})()
        gk.sample_goalkeeper_policy()
        assert gk.goalkeeper_policy == "stationary"
        gk.rng = orig_rng
    finally:
        env.close()


def test_bimanual_pure_obs_reward_and_drop_time_shape() -> None:
    env = make_env("myoChallengeBimanual-v0").unwrapped
    try:
        env.reset(seed=0)
        hist_len = len(env.touch_history)
        goal_touch = env.goal_touch
        for _ in range(3):
            obs_d = env.get_obs_dict()
            _ = env.get_reward_dict(obs_d)
        assert len(env.touch_history) == hist_len
        assert env.goal_touch == goal_touch

        custom_obs = env.get_obs_dict()
        custom_obs["obj_pos"] = np.array([0.0, 0.0, 0.1])
        rwd_d = env.get_reward_dict(custom_obs)
        assert rwd_d["done"] == 1
        assert isinstance(custom_obs["time"], np.ndarray)
        assert custom_obs["time"].shape == (1,)
    finally:
        env.close()


def test_tabletennis_pure_obs_reward_and_fresh_relaunch_state() -> None:
    env = make_env(
        "myoChallengeTableTennisP0-v0", ball_qvel=True, rally_count=2
    ).unwrapped
    try:
        env.reset(seed=3)
        # ball_qvel=True without ball_xyz_range still computes cal_ball_qvel
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

        # Simulate rally win on step 1 -> relaunch must leave derived site_xpos & obs fresh
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


def test_runtrack_instance_isolation_of_actuator_params() -> None:
    env1 = make_env("myoChallengeOslRunFixed-v0").unwrapped
    env2 = make_env("myoChallengeOslRunFixed-v0").unwrapped
    try:
        assert env1.ACTUATOR_PARAM is not env2.ACTUATOR_PARAM
        assert env1.OSL_PARAM_LIST is not env2.OSL_PARAM_LIST
        env1.ACTUATOR_PARAM["custom"] = {"id": -1, "Fmax": 0.0}
        assert "custom" not in env2.ACTUATOR_PARAM
    finally:
        env1.close()
        env2.close()


def test_leg_reach_preserves_freejoint_position_under_qpos_generation() -> None:
    env = make_env("myoLegStandRandom-v0").unwrapped
    try:
        env.reset(seed=0)
        env._init_qpos[0] = 1.25
        qpos = env._generate_qpos()
        assert qpos[0] == pytest.approx(1.25)
    finally:
        env.close()


def test_single_agent_opponent_wrapper_seed_reproducibility() -> None:
    class _DummyMultiEnv(gym.Env):
        def __init__(self) -> None:
            super().__init__()
            self._agents = ("agent_0", "agent_1")
            box = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
            self.observation_space = gym.spaces.Dict({"agent_0": box, "agent_1": box})
            self.action_space = gym.spaces.Dict({"agent_0": box, "agent_1": box})
            self.last_opp_actions: list[np.ndarray] = []

        def _opponent(self, agent_id: str) -> str:
            return "agent_1" if agent_id == "agent_0" else "agent_0"

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            z = np.zeros(3, dtype=np.float32)
            return {"agent_0": z, "agent_1": z}, {}

        def step(self, actions):
            self.last_opp_actions.append(actions["agent_1"].copy())
            z = np.zeros(3, dtype=np.float32)
            return (
                {"agent_0": z, "agent_1": z},
                {"agent_0": 0.0, "agent_1": 0.0},
                {"agent_0": False, "agent_1": False},
                {"agent_0": False, "agent_1": False},
                {},
            )

    base = _DummyMultiEnv()
    wrapped = ModularMultiAgentSingleAgentWrapper(base)
    wrapped.reset(seed=42)
    for _ in range(3):
        wrapped.step(np.zeros(3, dtype=np.float32))
    first = [a.copy() for a in base.last_opp_actions]

    base.last_opp_actions.clear()
    wrapped.reset(seed=42)
    for _ in range(3):
        wrapped.step(np.zeros(3, dtype=np.float32))
    second = [a.copy() for a in base.last_opp_actions]

    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_dict_and_perturbation_wrappers_forward_attrs_and_filter_keys() -> None:
    dict_env = DictObservationWrapper(
        make_env("myoChallengeBimanual-v0"), filter_obs_keys=True
    )
    try:
        assert callable(dict_env.mj_render)
        assert dict_env.model is dict_env.unwrapped.model
        obs, _ = dict_env.reset(seed=0)
        assert list(obs.keys()) == list(dict_env.unwrapped.obs_keys)
    finally:
        dict_env.close()

    pert_env = PerturbationWrapper(make_env("myoElbowPose1D6MRandom-v0"))
    try:
        assert callable(pert_env.mj_render)
        assert pert_env.model is pert_env.unwrapped.model
    finally:
        pert_env.close()
