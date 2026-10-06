# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""CPU <-> mjlab parity of ``myoChallengeChaseTagFBP2-v0``.

The CPU ``ChaseTagEnv`` and the mjlab task registered under the same id are
one task: the 537-dim chasetag_obs observation, action space, control step,
horizon, physics options, opponent, rewards and terminations. As in
``test_mjlab_cpu_twins``, the mjlab env is synced to the CPU state (agent,
opponent, previous distance, episode step) and the two are compared one step
at a time; open-loop rollouts diverge (float32 Warp vs float64 MuJoCo).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.tier2

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

import myosuite  # noqa: E402, F401
from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (  # noqa: E402
    _powerlaw_psd_gaussian,
)
from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import (  # noqa: E402
    mujoco_cfg_from_model,
)
from myosuite.envs.myo.backends.mjlab.tasks.mdp import write_cpu_state  # noqa: E402
from myosuite.envs.myo.tasks.mimic.chasetag_obs import (  # noqa: E402
    CHASETAG_OBS_DIM,
    CHASETAG_OBS_KEYS,
)
from myosuite import make_env  # noqa: E402

ENV_ID = "myoChallengeChaseTagFBP2-v0"
ENTITY = "chasetag_agent"
GROUP = "policy"
_POLICY_IDS = {"static_stationary": 0, "stationary": 1, "random": 2}

# One control step from the same state (float32 Warp vs float64 MuJoCo). Most
# coordinates agree to ~1e-5 (median); the largest differences sit in the light
# forearm/shoulder rotation dofs (pro_sup, shoulder_rot; cf. the arm tendon-wrap
# note in test_mjlab_cpu_twins): measured up to 0.27 rad/s in qvel, 0.1 in the
# root angular velocity, 2e-3 in qpos, 6e-5 in reward. Blocks without physics
# (heading_cmd, role) must match exactly.
_STEP_ATOL = {
    "qpos_local": 1e-2,
    "qvel_local": 0.6,
    "act": 1e-5,
    "root_vel_body": 2e-2,
    "heading_cmd": 0.0,
    "orientation": 0.25,
    "opponent_relative": 2e-2,
    "role": 0.0,
}
_STEP_MEDIAN_ATOL = 1e-3  # a block that is merely bounded (e.g. zeros) fails this
# Blocks whose median float32/solver noise between MuJoCo versions exceeds the default.
_STEP_MEDIAN_ATOL_BY_KEY = {"orientation": 2e-3}
_STEP_REW_ATOL = 1e-3
# Same state, no physics: term functions must agree to float32 precision.
_TERM_ATOL = 1e-5


@pytest.fixture(scope="module")
def pair() -> tuple[gym.Env, ManagerBasedRlEnv]:
    """The TimeLimit-wrapped CPU env and a 2-env mjlab env (env 0 is synced)."""
    import myosuite.envs.myo.backends.mjlab  # noqa: F401, PLC0415 (registers twins)

    cpu_env = make_env(ENV_ID)
    cpu_env.reset(seed=0)
    cfg = load_env_cfg(ENV_ID)
    cfg.scene.num_envs = 2
    mj = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    mj.reset()
    yield cpu_env, mj
    cpu_env.close()
    mj.close()


def _opponent(mj: ManagerBasedRlEnv):
    return mj.event_manager.get_term_cfg("reset_opponent").func


def _distance_term(mj: ManagerBasedRlEnv):
    return mj.reward_manager.get_term_cfg("distance").func


def _t(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=torch.float32)


def _sync(cpu, mj: ManagerBasedRlEnv, prev_distance: float | None = None) -> None:
    """Copy the CPU state into mjlab env 0 (agent, opponent, distance, step)."""
    write_cpu_state(
        mj,
        ENTITY,
        torch.zeros(1, dtype=torch.long),
        _t(cpu.data.qpos)[None],
        _t(cpu.data.qvel)[None],
    )
    mj.sim.data.act[0] = _t(cpu.data.act)
    opponent, cpu_opponent = _opponent(mj), cpu.opponent
    opponent.pose[0] = _t(cpu_opponent.get_opponent_pose())
    opponent.vel[0] = _t(cpu_opponent.opponent_vel)
    opponent.policy[0] = _POLICY_IDS[cpu_opponent.opponent_policy]
    opponent.noise[0] = _t(cpu_opponent.noise_process.buffer)
    opponent.noise_idx[0] = cpu_opponent.noise_process.idx
    opponent._write_mocap(mj)
    prev = cpu._prev_distance if prev_distance is None else prev_distance
    _distance_term(mj)._prev[0] = float("nan") if prev is None else prev
    mj.episode_length_buf[0] = cpu.steps
    mj.sim.forward()


def _cpu_obs(cpu) -> np.ndarray:
    return cpu._obs_dict_to_vec(cpu._get_obs_dict(cpu._accessor)).astype(np.float32)


def _assert_obs_close(cpu, got: np.ndarray, expected: np.ndarray) -> None:
    """Per-block comparison of one-step observations (see ``_STEP_ATOL``)."""
    obs_dict = cpu._get_obs_dict(cpu._accessor)
    start = 0
    for key in CHASETAG_OBS_KEYS:
        size = np.atleast_1d(obs_dict[key]).size
        diff = np.abs(got[start : start + size] - expected[start : start + size])
        assert diff.max() <= _STEP_ATOL[key], (key, diff.max())
        median_atol = _STEP_MEDIAN_ATOL_BY_KEY.get(key, _STEP_MEDIAN_ATOL)
        assert np.median(diff) <= median_atol, (key, np.median(diff))
        start += size
    assert start == CHASETAG_OBS_DIM


def test_spaces_timing_physics_and_reward_terms_match(pair) -> None:
    """Obs/action spaces, control step, horizon, physics options and reward weights."""
    cpu_env, mj = pair
    cpu, cfg = cpu_env.unwrapped, mj.cfg

    assert list(cpu.obs_keys) == list(CHASETAG_OBS_KEYS)
    assert mj.observation_manager.active_terms[GROUP] == list(CHASETAG_OBS_KEYS)
    assert cpu.observation_space.shape == (CHASETAG_OBS_DIM,)
    assert mj.observation_manager.group_obs_dim[GROUP] == (CHASETAG_OBS_DIM,)
    assert np.isinf(cpu.observation_space.low).all()  # unclipped, like mjlab
    assert np.isinf(cpu.observation_space.high).all()

    assert cpu.action_space.shape == (mj.action_manager.total_action_dim,)
    np.testing.assert_array_equal(cpu.action_space.low, -1.0)
    np.testing.assert_array_equal(cpu.action_space.high, 1.0)
    assert mj.action_manager.get_term("muscles").cfg.action_mode == "direct"

    assert cpu.frame_skip == cfg.decimation == 5
    assert cpu.dt == pytest.approx(mj.step_dt) == pytest.approx(0.01)
    assert cpu_env.spec.max_episode_steps == mj.max_episode_length == 2000
    assert cpu_env.spec.max_episode_steps * cpu.dt == pytest.approx(20.0)
    # CHASE also loses at data.time >= maxTime; the float64 time after the
    # last step is still below it, so the horizon is a truncation on both.
    time = 0.0
    for _ in range(cpu_env.spec.max_episode_steps * cpu.frame_skip):
        time += cpu.model.opt.timestep
    assert time < cpu.maxTime

    assert cfg.sim.mujoco == mujoco_cfg_from_model(cpu.model)
    for field in (
        "timestep",
        "integrator",
        "solver",
        "cone",
        "jacobian",
        "iterations",
        "ls_iterations",
        "ccd_iterations",
        "tolerance",
        "ls_tolerance",
        "impratio",
        "disableflags",
        "enableflags",
    ):
        assert getattr(mj.sim.mj_model.opt, field) == getattr(cpu.model.opt, field)

    terrain = [
        i
        for i in range(mj.sim.mj_model.ngeom)
        if mj.sim.mj_model.geom(i).name.endswith("terrain")
    ]
    assert cpu.terrain == "FLAT"
    np.testing.assert_array_equal(
        mj.sim.mj_model.geom_pos[terrain[0]],
        cpu.model.geom_pos[cpu.model.geom("terrain").id],
    )

    assert cfg.scale_rewards_by_dt is False
    assert {k: float(w) for k, w in cpu.rwd_keys_wt.items()} == {
        k: term.weight for k, term in cfg.rewards.items()
    }


def test_reset_matches_cpu_reset(pair) -> None:
    """Keyframe reset (body-frame root angular velocity) and the opponent spawn."""
    cpu_env, mj = pair
    cpu = cpu_env.unwrapped
    mj.reset()
    np.testing.assert_allclose(
        mj.sim.data.qpos[0].numpy(), cpu.model.key_qpos[0], atol=1e-6
    )
    np.testing.assert_allclose(
        mj.sim.data.qvel[0].numpy(), cpu.model.key_qvel[0], atol=1e-6
    )
    assert not mj.sim.data.act.any()

    opponent = _opponent(mj)
    reset_opponent = mj.event_manager.get_term_cfg("reset_opponent")
    torch.manual_seed(0)
    n_draws = 1000
    policies, poses = [], []
    for _ in range(n_draws):
        opponent(mj, torch.arange(2), **reset_opponent.params)
        policies.append(opponent.policy.clone())
        poses.append(opponent.pose.clone())
    policy, pose = torch.cat(policies), torch.cat(poses)
    freq = torch.bincount(policy, minlength=3).double() / policy.numel()
    np.testing.assert_allclose(freq.numpy(), (0.1, 0.45, 0.45), atol=0.05)
    agent_xy = mj.sim.data.qpos[:, :2].repeat(n_draws, 1)
    spawned = policy != 0
    distance = torch.linalg.norm(pose[:, :2] - agent_xy, dim=1)
    assert (distance[spawned] >= 2.0).all()  # min_spawn_distance
    assert (pose[spawned, :2].abs() <= 5.0).all()
    np.testing.assert_array_equal(
        pose[~spawned].numpy(), [[0.0, -5.0, 0.0]] * int((~spawned).sum())
    )


def test_terms_match_cpu_on_cpu_states(pair) -> None:
    """Same state, no physics: the obs, reward and termination terms are the CPU ones.

    The CPU env runs a random-opponent episode; after each CPU step its state is
    written into mjlab and the mjlab managers evaluate their terms on it.
    """
    cpu_env, mj = pair
    cpu = cpu_env.unwrapped
    cpu.reset(seed=1)
    cpu.opponent.opponent_policy = "random"
    _sync(cpu, mj)
    np.testing.assert_allclose(
        mj.observation_manager.compute_group(GROUP)[0].numpy(),
        _cpu_obs(cpu),
        atol=_TERM_ATOL,
    )
    rng = np.random.default_rng(1)
    for _ in range(300):
        prev = cpu._prev_distance
        action = rng.uniform(-0.4, 0.4, cpu.action_space.shape).astype(np.float32)
        cpu_obs, cpu_rew, cpu_term, _, cpu_info = cpu.step(action)
        _sync(cpu, mj, prev_distance=prev)
        np.testing.assert_allclose(
            mj.observation_manager.compute_group(GROUP)[0].numpy(),
            cpu_obs,
            atol=_TERM_ATOL,
        )
        mj_rew = mj.reward_manager.compute(dt=mj.step_dt)[0]
        np.testing.assert_allclose(float(mj_rew), cpu_rew, atol=_TERM_ATOL, rtol=1e-6)
        mj.termination_manager.compute()
        assert bool(mj.termination_manager.terminated[0]) == cpu_term
        if cpu_term:
            break
    assert cpu_term, "the episode never ended (no fall within 300 steps)"


def test_one_step_parity(pair) -> None:
    """Same state + same action -> same obs (all 537 dims), reward and termination."""
    cpu_env, mj = pair
    cpu = cpu_env.unwrapped
    cpu.reset(seed=2)
    cpu.opponent.opponent_policy = "random"
    rng = np.random.default_rng(2)
    for _ in range(300):
        _sync(cpu, mj)
        action = rng.uniform(-0.4, 0.4, cpu.action_space.shape).astype(np.float32)
        cpu_obs, cpu_rew, cpu_term, _, cpu_info = cpu.step(action)
        mj_obs, mj_rew, mj_term, mj_trunc, _ = mj.step(
            torch.as_tensor(np.stack([action, action]))
        )
        assert not bool(mj_trunc[0])
        assert bool(mj_term[0]) == cpu_term
        np.testing.assert_allclose(float(mj_rew[0]), cpu_rew, atol=_STEP_REW_ATOL)
        if cpu_term:
            assert cpu_info["lose"]
            break
        _assert_obs_close(cpu, mj_obs[GROUP][0].numpy(), cpu_obs)
    assert cpu_term, "the episode never ended (no fall within 300 steps)"


@pytest.mark.parametrize("case", ["tag", "out_of_bounds"])
def test_termination_parity(pair, case: str) -> None:
    """Tagging (win) and leaving the arena (lose) end the episode on both."""
    cpu_env, mj = pair
    cpu = cpu_env.unwrapped
    cpu.reset(seed=3)
    cpu.opponent.opponent_policy = "stationary"
    if case == "tag":
        cpu.opponent.set_opponent_pose([cpu.data.qpos[0] + 0.3, cpu.data.qpos[1], 0.0])
    else:
        cpu.data.qpos[0] = 6.6
    _sync(cpu, mj)
    action = np.zeros(cpu.action_space.shape, dtype=np.float32)
    _, cpu_rew, cpu_term, _, cpu_info = cpu.step(action)
    _, mj_rew, mj_term, _, _ = mj.step(torch.as_tensor(np.stack([action, action])))
    assert cpu_term and bool(mj_term[0])
    assert bool(cpu_info["solved"]) == (case == "tag")
    assert bool(cpu_info["lose"]) == (case == "out_of_bounds")
    np.testing.assert_allclose(float(mj_rew[0]), cpu_rew, atol=_STEP_REW_ATOL)
    success = dict(mj.metrics_manager.get_active_iterable_terms(0))["success"][0]
    assert bool(success) == bool(cpu_info["solved"])


def test_horizon_is_a_truncation_on_both(pair) -> None:
    """The 2000th step (20 s) truncates without the CHASE time-limit penalty."""
    cpu_env, mj = pair
    cpu = cpu_env.unwrapped
    cpu_env.reset(seed=4)
    cpu.opponent.opponent_policy = "stationary"
    steps = cpu_env.spec.max_episode_steps - 1
    wrapper = cpu_env
    while not isinstance(wrapper, gym.wrappers.TimeLimit):
        wrapper = wrapper.env
    wrapper._elapsed_steps = steps
    cpu.steps = steps
    for _ in range(steps * cpu.frame_skip):  # the float64 sum MuJoCo would hold
        cpu.data.time += cpu.model.opt.timestep
    _sync(cpu, mj)
    action = np.zeros(cpu.action_space.shape, dtype=np.float32)
    _, cpu_rew, cpu_term, cpu_trunc, cpu_info = cpu_env.step(action)
    _, mj_rew, mj_term, mj_trunc, _ = mj.step(
        torch.as_tensor(np.stack([action, action]))
    )
    assert cpu_trunc and bool(mj_trunc[0])
    assert not cpu_term and not bool(mj_term[0])
    assert not cpu_info["lose"]
    np.testing.assert_allclose(float(mj_rew[0]), cpu_rew, atol=_STEP_REW_ATOL)


def test_distance_delta_restarts_each_episode(pair) -> None:
    """The first step of an episode scores no distance change, as on CPU."""
    cpu_env, mj = pair
    cpu = cpu_env.unwrapped
    cpu.reset(seed=5)
    _, _, _, _, info = cpu.step(np.zeros(cpu.action_space.shape, dtype=np.float32))
    assert info["distance"] == 0.0

    term = _distance_term(mj)
    term._prev[:] = torch.tensor([1.0, 2.0])
    term.reset(torch.tensor([1]))  # partial reset forgets env 1 only
    assert term._prev[0] == 1.0 and torch.isnan(term._prev[1])

    mj.reset()
    zero = torch.zeros(2, mj.action_manager.total_action_dim)
    mj.step(zero)
    opponent = _opponent(mj)
    opponent.policy[:] = 1  # stationary
    opponent.pose[0, :2] = mj.sim.data.qpos[0, :2]  # env 0 tags on its next step
    _, _, terminated, _, _ = mj.step(zero)
    assert bool(terminated[0])  # tagged -> auto-reset, opponent respawns >= 2 m away
    assert torch.isnan(term._prev[0])
    mj.step(zero)
    values = dict(mj.reward_manager.get_active_iterable_terms(0))
    assert values["distance"][0] == 0.0


@pytest.mark.parametrize("samples", [2000, 1999])
def test_opponent_noise_matches_colorednoise(samples: int) -> None:
    """The torch colored-noise generator is colorednoise's for the same normal draws."""
    import colorednoise  # noqa: PLC0415

    shape = (3, 2, samples)
    expected = colorednoise.powerlaw_psd_gaussian(
        2, shape, random_state=np.random.default_rng(7)
    )
    rng = np.random.default_rng(7)  # the real, then the imaginary parts
    freq_shape = (*shape[:-1], samples // 2 + 1)
    normals = np.stack(
        [rng.standard_normal(freq_shape), rng.standard_normal(freq_shape)]
    )
    got = _powerlaw_psd_gaussian(2.0, shape, "cpu", normals=torch.as_tensor(normals))
    np.testing.assert_allclose(got.numpy(), expected, atol=1e-5)
