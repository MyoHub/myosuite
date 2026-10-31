# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""CPU <-> mjlab parity of ``myoChallengeChaseTagFBVsTrain-v0``.

The CPU ``ChaseTagVsTrainEnv`` and the mjlab task registered under the same id
are one task: both players' 546-d FBVs observations, the action, control step,
horizon, physics, reset, reward, terminations and score. As in
``test_chasetag_fbp2_parity``, mjlab env 0 is synced to the CPU state (both
players, activations, role, episode step, learner action) and the two are
compared one step at a time; open-loop rollouts diverge (float32 Warp vs
float64 MuJoCo).
"""

from __future__ import annotations

import dataclasses

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.tier2

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

from myosuite import make_env  # noqa: E402
from myosuite.envs.myo.backends.mjlab.tasks.chasetag_vs import mdp as ct  # noqa: E402
from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import (  # noqa: E402
    mujoco_cfg_from_model,
)
from myosuite.envs.myo.backends.mjlab.tasks.mdp import write_cpu_state  # noqa: E402
from myosuite.envs.myo.backends.mjlab.tasks.mdp.self_play import (  # noqa: E402
    SelfPlayCfg,
    install_self_play,
)
from myosuite.envs.myo.tasks.challenge.chase_tag_vs.train_env import (  # noqa: E402
    CHASETAG_VS_OBS_DIM,
    CHASETAG_VS_OBS_KEYS,
    LEARNER,
    OPPONENT,
)
from myosuite.tests.support.optional_deps import (  # noqa: E402
    require_musclemimic_models,
)

ENV_ID = "myoChallengeChaseTagFBVsTrain-v0"
AGENTS = (LEARNER, OPPONENT)  # CPU agent of each mjlab player (ct.PLAYERS)
N_ACT = 354
BLOCK = dict(zip(CHASETAG_VS_OBS_KEYS, (89, 88, 354, 6, 7, 2), strict=True))

# One control step from the same state (float32 Warp vs float64 MuJoCo); the
# bounds were measured on these rollouts with ~3x headroom. Blocks without
# physics (role) must match exactly.
_STEP_ATOL = {
    "qpos": 1e-2,
    "qvel": 0.6,
    "act": 1e-5,
    "pelvis": 2e-2,
    "opponent_relative": 2e-2,
    "role": 0.0,
}
# Medians: per step (a contact one backend resolves differently moves many dofs
# at once: measured up to ~5e-3) and over the episode (measured < 1e-5). A
# block that is merely bounded (e.g. zeros, or the other player's) fails these.
_STEP_MEDIAN_ATOL = 1e-2
_EPISODE_MEDIAN_ATOL = 1e-4
_STEP_REW_ATOL = 1e-3
# Same state, no physics: term functions must agree to float32 precision.
_TERM_ATOL = 1e-5


@pytest.fixture(scope="module")
def pair() -> tuple[gym.Env, ManagerBasedRlEnv]:
    """The CPU env and a 2-env mjlab env (env 0 is synced)."""
    require_musclemimic_models()
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


def _t(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=torch.float32)


def _term(mj: ManagerBasedRlEnv) -> ct.ChaseTagVsAction:
    return mj.action_manager.get_term("muscles")


def _sync(cpu, mj: ManagerBasedRlEnv) -> None:
    """Copy the CPU state into mjlab env 0 (players, act, role, step, action, obs)."""
    ids = torch.zeros(1, dtype=torch.long)
    match = ct.match(mj)
    for player, (name, agent) in enumerate(zip(ct.PLAYERS, AGENTS, strict=True)):
        qs, vs = cpu.player_slices(agent)
        write_cpu_state(
            mj, name, ids, _t(cpu.data.qpos[qs])[None], _t(cpu.data.qvel[vs])[None]
        )
        mj.sim.data.act[0, match.act_cols[player]] = _t(cpu.data.act[cpu._act[agent]])
    match.chase[0] = cpu.task_config.current_task == "CHASE"
    mj.episode_length_buf[0] = cpu.steps
    _term(mj)._raw_actions[0] = _t(cpu._action)
    mj.sim.forward()
    # compute() returns the observations cached this step: recompute each group.
    om = mj.observation_manager
    mj.obs_buf = {group: om.compute_group(group) for group in ("policy", "opponent")}


def _cpu_obs(cpu) -> np.ndarray:
    return cpu._obs_dict_to_vec(cpu._get_obs_dict(cpu._accessor)).astype(np.float32)


def _assert_obs_close(
    got: np.ndarray, expected: np.ndarray, medians: dict[str, list[float]]
) -> None:
    """Per-block comparison of one-step observations (see ``_STEP_ATOL``).

    Appends each block's median difference to *medians* for the episode check.
    """
    start = 0
    for key, size in BLOCK.items():
        diff = np.abs(got[start : start + size] - expected[start : start + size])
        assert diff.max() <= _STEP_ATOL[key], (key, diff.max())
        assert np.median(diff) <= _STEP_MEDIAN_ATOL, (key, np.median(diff))
        medians.setdefault(key, []).append(float(np.median(diff)))
        start += size
    assert start == CHASETAG_VS_OBS_DIM


class _Opponent(torch.nn.Module):
    """A fixed policy playing the opponent on both backends."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.net = torch.nn.Linear(CHASETAG_VS_OBS_DIM, N_ACT)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return 0.3 * torch.sigmoid(1e-3 * self.net(obs))

    def numpy(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self(_t(obs)[None])[0].numpy()


def _set_opponent(cpu, mj: ManagerBasedRlEnv, policy: _Opponent | None) -> None:
    cpu.set_opponent_policy(None if policy is None else policy.numpy)
    pool = _term(mj).opponent_pool
    pool.policies.clear()
    if policy is not None:
        pool.add(policy)
    pool.slot[:] = 0


def test_spaces_timing_and_physics_match(pair) -> None:
    """Obs/action spaces, control step, horizon, physics options and the model."""
    cpu_env, mj = pair
    cpu, cfg = cpu_env.unwrapped, mj.cfg
    for group in ("policy", "opponent"):
        assert mj.observation_manager.active_terms[group] == list(CHASETAG_VS_OBS_KEYS)
        assert mj.observation_manager.group_obs_dim[group] == (CHASETAG_VS_OBS_DIM,)
    assert cpu.obs_keys == list(CHASETAG_VS_OBS_KEYS)
    assert cpu.observation_space.shape == (CHASETAG_VS_OBS_DIM,)
    assert mj.action_manager.total_action_dim == N_ACT == cpu.action_space.shape[0]
    np.testing.assert_array_equal(cpu.action_space.low, 0.0)
    np.testing.assert_array_equal(cpu.action_space.high, 1.0)

    assert cpu.frame_skip == cfg.decimation == 5
    assert cpu.dt == pytest.approx(mj.step_dt) == pytest.approx(0.01)
    assert cpu_env.spec.max_episode_steps == mj.max_episode_length == 2000
    assert cfg.sim.mujoco == mujoco_cfg_from_model(cpu.model)
    assert cfg.scale_rewards_by_dt is False

    # Same bodies, joints, muscles and collision geometry per player.
    m_cpu, m_mj = cpu.model, mj.sim.mj_model
    for player, (name, agent) in enumerate(zip(ct.PLAYERS, AGENTS, strict=True)):
        entity = mj.scene[name]
        act_cpu = np.asarray(cpu._act[agent])
        act_mj = entity.indexing.ctrl_ids.cpu().numpy()
        for field in ("actuator_gainprm", "actuator_biasprm", "actuator_dynprm"):
            np.testing.assert_allclose(
                getattr(m_mj, field)[act_mj], getattr(m_cpu, field)[act_cpu], rtol=1e-6
            )
        body_cpu = [
            b
            for b in range(m_cpu.nbody)
            if m_cpu.body(b).name.startswith(("a0_", "a1_")[player])
        ]
        body_mj = entity.indexing.body_ids.cpu().numpy()
        np.testing.assert_allclose(m_mj.body_mass[body_mj], m_cpu.body_mass[body_cpu])
        geom_cpu = [
            g for g in range(m_cpu.ngeom) if m_cpu.geom_bodyid[g] in set(body_cpu)
        ]
        geom_mj = [g for g in range(m_mj.ngeom) if m_mj.geom_bodyid[g] in set(body_mj)]
        for field in ("geom_size", "geom_contype", "geom_conaffinity", "geom_friction"):
            np.testing.assert_allclose(
                getattr(m_mj, field)[geom_mj], getattr(m_cpu, field)[geom_cpu]
            )
    world = {
        m_mj.geom(g).name: g
        for g in range(m_mj.ngeom)
        if m_mj.geom_bodyid[g] == 0 and m_mj.geom_contype[g]
    }
    world_cpu = {
        m_cpu.geom(g).name: g
        for g in range(m_cpu.ngeom)
        if m_cpu.geom_bodyid[g] == 0 and m_cpu.geom_contype[g]
    }
    assert set(world) == set(world_cpu)  # floor and fence
    for gname, g in world.items():
        for field in ("geom_size", "geom_pos", "geom_quat", "geom_friction"):
            np.testing.assert_allclose(
                getattr(m_mj, field)[g],
                getattr(m_cpu, field)[world_cpu[gname]],
                atol=1e-7,
            )


def test_reset_matches_cpu_reset(pair) -> None:
    """Both players in the FBVs standing pose at rest, roles drawn 50/50."""
    cpu_env, mj = pair
    cpu = cpu_env.unwrapped
    mj.reset()
    for name, agent in zip(ct.PLAYERS, AGENTS, strict=True):
        qs, vs = cpu.player_slices(agent)
        idx = mj.scene[name].indexing
        q = torch.sort(torch.cat([idx.free_joint_q_adr, idx.joint_q_adr]).long())[0]
        v = torch.sort(torch.cat([idx.free_joint_v_adr, idx.joint_v_adr]).long())[0]
        np.testing.assert_allclose(
            mj.sim.data.qpos[0, q].numpy(), cpu.meta.standing_qpos[qs], atol=1e-6
        )
        assert not mj.sim.data.qvel[0, v].any()
    assert not mj.sim.data.act.any()
    cpu.reset(seed=0)
    _sync(cpu, mj)
    np.testing.assert_allclose(
        mj.obs_buf["policy"][0].numpy(), _cpu_obs(cpu), atol=_TERM_ATOL
    )

    match = ct.match(mj)
    torch.manual_seed(0)
    draws = []
    for _ in range(500):
        match(mj, torch.arange(2))
        draws.append(match.chase.clone())
    assert abs(torch.cat(draws).float().mean().item() - 0.5) < 0.05


def test_terms_match_cpu_on_cpu_states(pair) -> None:
    """Same state, no physics: both observation groups, the reward and termination.

    The CPU env plays an episode against a fixed opponent; after each CPU step
    its state is written into mjlab and the managers evaluate their terms on it.
    """
    cpu_env, mj = pair
    cpu = cpu_env.unwrapped
    opponent = _Opponent()
    _set_opponent(cpu, mj, opponent)
    rng = np.random.default_rng(1)
    for seed in (1, 2):
        cpu.reset(seed=seed)
        for _ in range(300):
            action = rng.uniform(0.0, 0.4, N_ACT).astype(np.float32)
            action[:3] = 1.3  # clipped for physics, raw in the reward
            cpu_obs, cpu_rew, cpu_term, _, cpu_info = cpu.step(action)
            _sync(cpu, mj)
            np.testing.assert_allclose(
                mj.obs_buf["policy"][0].numpy(), cpu_obs, atol=_TERM_ATOL
            )
            np.testing.assert_allclose(
                mj.obs_buf["opponent"][0].numpy(), cpu.opponent_obs, atol=_TERM_ATOL
            )
            mj_rew = mj.reward_manager.compute(dt=mj.step_dt)[0]
            np.testing.assert_allclose(float(mj_rew), cpu_rew, atol=_TERM_ATOL)
            mj.termination_manager.compute()
            assert bool(mj.termination_manager.terminated[0]) == cpu_term
            if cpu_term:
                break
        assert cpu_term, "the episode never ended (no fall within 300 steps)"
    _set_opponent(cpu, mj, None)


def test_one_step_parity(pair) -> None:
    """Same state + same actions (both players) -> same obs, reward, termination."""
    cpu_env, mj = pair
    cpu = cpu_env.unwrapped
    opponent = _Opponent()
    _set_opponent(cpu, mj, opponent)
    cpu.reset(seed=3)
    rng = np.random.default_rng(3)
    medians: dict[str, list[float]] = {}
    for _ in range(300):
        _sync(cpu, mj)
        action = rng.uniform(0.0, 0.4, N_ACT).astype(np.float32)
        cpu_obs, cpu_rew, cpu_term, _, _ = cpu.step(action)
        mj_obs, mj_rew, mj_term, mj_trunc, _ = mj.step(
            torch.as_tensor(np.stack([action, action]))
        )
        assert not bool(mj_trunc[0])
        assert bool(mj_term[0]) == cpu_term
        np.testing.assert_allclose(float(mj_rew[0]), cpu_rew, atol=_STEP_REW_ATOL)
        if cpu_term:
            break
        np.testing.assert_allclose(  # (zeroed by the auto-reset on the last step)
            _term(mj).opponent_ctrl[0].numpy(),
            cpu.data.ctrl[cpu._act[OPPONENT]],
            atol=1e-5,
        )
        _assert_obs_close(mj_obs["policy"][0].numpy(), cpu_obs, medians)
        _assert_obs_close(mj_obs["opponent"][0].numpy(), cpu.opponent_obs, medians)
    assert cpu_term, "the episode never ended (no fall within 300 steps)"
    for key, values in medians.items():
        assert np.median(values) <= _EPISODE_MEDIAN_ATOL, (key, np.median(values))
    _set_opponent(cpu, mj, None)


def _metrics(mj: ManagerBasedRlEnv) -> dict[str, float]:
    return {k: float(v[0]) for k, v in mj.metrics_manager.get_active_iterable_terms(0)}


@pytest.mark.parametrize("case", ["tag", "learner_fell", "opponent_fell"])
def test_termination_and_score_parity(pair, case: str) -> None:
    """A tag or either player's fall ends the episode on both, with the same score."""
    cpu_env, mj = pair
    cpu = cpu_env.unwrapped
    for seed in (4, 5):  # one CHASE and one EVADE episode
        cpu.reset(seed=seed)
        q_learner, _ = cpu.player_slices(LEARNER)
        q_opp, _ = cpu.player_slices(OPPONENT)
        if case == "tag":
            cpu.data.qpos[q_opp.start + 1] = cpu.data.qpos[q_learner.start + 1] - 0.3
        else:
            q = q_learner if case == "learner_fell" else q_opp
            cpu.data.qpos[q.start + 2] = 0.5
        _sync(cpu, mj)
        action = np.zeros(N_ACT, dtype=np.float32)
        _, cpu_rew, cpu_term, _, info = cpu.step(action)
        _, mj_rew, mj_term, _, _ = mj.step(torch.as_tensor(np.stack([action, action])))
        assert cpu_term and bool(mj_term[0])
        assert info["tagged"] == (case == "tag")
        np.testing.assert_allclose(float(mj_rew[0]), cpu_rew, atol=_STEP_REW_ATOL)
        metrics = _metrics(mj)
        assert bool(metrics["success"]) == info["solved"]
        assert metrics["score"] == pytest.approx(info["score"], abs=1e-6)
        assert bool(metrics["tagged"]) == info["tagged"]
        assert bool(metrics["learner_fell"]) == info["learner_fell"]
        assert bool(metrics["opponent_fell"]) == info["opponent_fell"]


def test_horizon_is_a_truncation_on_both(pair) -> None:
    """The 2000th step (20 s) truncates; the score is FBVs's on both."""
    cpu_env, mj = pair
    cpu = cpu_env.unwrapped
    wrapper = cpu_env
    while not isinstance(wrapper, gym.wrappers.TimeLimit):
        wrapper = wrapper.env
    for seed in (4, 5):
        cpu_env.reset(seed=seed)
        wrapper._elapsed_steps = cpu.steps = 1999
        for _ in range(1999 * cpu.frame_skip):  # the float64 sum MuJoCo would hold
            cpu.data.time += cpu.model.opt.timestep  # (the gait-clip frame)
        _sync(cpu, mj)
        action = np.zeros(N_ACT, dtype=np.float32)
        _, cpu_rew, cpu_term, cpu_trunc, info = cpu_env.step(action)
        _, mj_rew, mj_term, mj_trunc, _ = mj.step(
            torch.as_tensor(np.stack([action, action]))
        )
        assert cpu_trunc and bool(mj_trunc[0])
        assert not cpu_term and not bool(mj_term[0])
        np.testing.assert_allclose(float(mj_rew[0]), cpu_rew, atol=_STEP_REW_ATOL)
        metrics = _metrics(mj)
        assert bool(metrics["success"]) == info["solved"]
        assert metrics["score"] == pytest.approx(info["score"], abs=1e-6)


def test_opponent_modes_and_submission_export(tmp_path) -> None:
    """Self-play snapshots, a fixed checkpoint opponent, and the 546 -> 354 ONNX export.

    Two PPO iterations with a snapshot each fill the pool; the saved checkpoint
    then plays the opponent of a ``"policy"`` env, and its ONNX export plays
    both players of ``myoChallengeChaseTagFBVs-v0``.
    """
    require_musclemimic_models()
    import myosuite.envs.myo.backends.mjlab  # noqa: F401, PLC0415
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: PLC0415
    from mjlab.tasks.registry import load_rl_cfg  # noqa: PLC0415

    from myosuite.utils.export_onnx import export_rslrl_to_onnx  # noqa: PLC0415
    from myosuite.utils.rslrl_policy import load_rslrl_policy  # noqa: PLC0415

    cfg = load_env_cfg(ENV_ID)
    cfg.scene.num_envs = 4
    cfg.actions["muscles"].self_play = SelfPlayCfg(snapshot_every=1, pool_size=8)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cpu"))
    agent = load_rl_cfg(ENV_ID)
    agent.num_steps_per_env = 4
    agent.logger = "tensorboard"
    runner = MjlabOnPolicyRunner(env, dataclasses.asdict(agent), str(tmp_path), "cpu")
    try:
        pool = install_self_play(runner)
        assert pool is not None and len(pool) == 1  # the initial policy
        runner.learn(num_learning_iterations=2)
        assert len(pool) == 3
        assert _term(env.unwrapped).opponent_ctrl.abs().sum() > 0
        checkpoint = tmp_path / "model.pt"
        runner.save(str(checkpoint))
    finally:
        env.close()

    cfg = load_env_cfg(ENV_ID)
    cfg.scene.num_envs = 2
    cfg.actions["muscles"].opponent = "policy"
    cfg.actions["muscles"].opponent_checkpoint = str(checkpoint)
    mj = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        obs, _ = mj.reset()
        mj.step(torch.zeros(2, N_ACT))
        policy = load_rslrl_policy(checkpoint, N_ACT)
        with torch.no_grad():
            expected = torch.clamp(policy(obs["opponent"]), 0.0, 1.0)
        np.testing.assert_allclose(
            _term(mj).opponent_ctrl.numpy(), expected.numpy(), atol=1e-5
        )
    finally:
        mj.close()

    onnx_path = tmp_path / "player.onnx"
    export_rslrl_to_onnx(
        checkpoint, onnx_path, obs_dim=CHASETAG_VS_OBS_DIM, act_dim=N_ACT
    )
    fbvs = make_env("myoChallengeChaseTagFBVs-v0")
    train = make_env(ENV_ID, opponent_policy=onnx_path)
    try:
        from myosuite.envs.myo.tasks.challenge.chase_tag_vs.train_env import (  # noqa: PLC0415
            load_opponent_policy,
        )

        player = load_opponent_policy(onnx_path, N_ACT)
        obs, _ = fbvs.reset(seed=0)
        for _ in range(5):
            obs, rew, _, _, _ = fbvs.step({a: player(obs[a]) for a in AGENTS})
            assert all(np.isfinite(r) for r in rew.values())
        train.reset(seed=0)
        train.step(np.zeros(N_ACT, dtype=np.float32))
    finally:
        fbvs.close()
        train.close()


def test_self_play_steps_without_host_syncs(pair) -> None:
    """Opponent snapshots, terms and an auto-reset run without host syncs."""
    from myosuite.tests.support.host_sync import HostSyncCounter  # noqa: PLC0415

    cpu_env, mj = pair
    _set_opponent(cpu_env.unwrapped, mj, _Opponent())
    mj.reset()
    mj.episode_length_buf[0] = mj.max_episode_length - 1
    gen = torch.Generator().manual_seed(0)
    resets = []
    for _ in range(3):
        action = 0.3 * torch.rand(mj.num_envs, N_ACT, generator=gen)
        with HostSyncCounter(package_only=True) as syncs:
            mj.step(action)
        assert syncs.total == 0, syncs.report()
        resets.append(bool(mj.reset_buf.any()))
    assert any(resets)
    assert _term(mj).opponent_ctrl.abs().sum() > 0
    _set_opponent(cpu_env.unwrapped, mj, None)
