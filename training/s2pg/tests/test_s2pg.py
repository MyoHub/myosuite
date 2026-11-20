# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Unit and smoke tests for myosuite_s2pg."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

from myosuite_s2pg import (
    VALID_RNN_TYPES,
    ActorCriticS2PG,
    IndependentJointDist,
    RNNType,
    RunningMeanStd,
    S2PGConfig,
    S2PGPPOTrainer,
    StatefulS2PGExportWrapper,
    baseline_feedforward_ppo,
    compute_gae,
    export_onnx_stateful,
    export_torchscript,
    run_s2pg_benchmark,
    stored_hidden_dim,
)
from myosuite_s2pg.train_mjlab import (
    MjlabTensorVecEnv,
    MjlabTrainArgs,
    run_s2pg_on_mjlab_env,
)


class _DeterministicBoundaryVecEnv:
    """Controlled 2-env tensor environment triggering terminated and truncated boundaries."""

    def __init__(self, obs_dim: int = 5, action_dim: int = 3) -> None:
        self.num_envs = 2
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_episode_steps = 10
        self.device = torch.device("cpu")
        self._step_idx = 0

    def reset(self, seed: int | None = None) -> torch.Tensor:
        del seed
        self._step_idx = 0
        return torch.full((self.num_envs, self.obs_dim), 0.25, dtype=torch.float32)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        step_now = self._step_idx
        self._step_idx += 1

        terminated = torch.tensor([step_now == 1, False], dtype=torch.bool)
        truncated = torch.tensor([False, step_now == 2], dtype=torch.bool)
        rewards = actions.sum(dim=-1) * 0.1 + float(step_now + 1)
        next_obs = torch.full(
            (self.num_envs, self.obs_dim),
            float(step_now + 1) * 0.5,
            dtype=torch.float32,
        )
        completed: list[dict[str, float]] = []
        if bool((terminated | truncated).any()):
            completed.append({"return": 1.0, "length": 2.0, "success": 1.0})
        return (
            next_obs,
            rewards,
            terminated,
            truncated,
            {"completed_episodes": completed},
        )

    def close(self) -> None:
        pass


class _FakeMetricsManager:
    active_terms = ("success",)

    def get_active_iterable_terms(self, env_idx: int) -> list[tuple[str, list[float]]]:
        return [("success", [float(env_idx % 2 == 0)])]


class _FakeActionManager:
    def __init__(self, total_action_dim: int) -> None:
        self.total_action_dim = total_action_dim


class _FakeMjlabEnv:
    """Pure-PyTorch mock matching ManagerBasedRlEnv's tensor API."""

    def __init__(
        self, num_envs: int = 4, obs_dim: int = 9, action_dim: int = 6
    ) -> None:
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.device = "cpu"
        self.max_episode_length = 8
        self.action_manager = _FakeActionManager(action_dim)
        self.metrics_manager = _FakeMetricsManager()
        self._t = 0

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        self._t = 0
        obs = torch.randn(self.num_envs, self.obs_dim, dtype=torch.float32)
        return {"actor": obs}, {}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        assert isinstance(actions, torch.Tensor)
        self._t += 1
        obs = torch.tanh(torch.randn(self.num_envs, self.obs_dim, dtype=torch.float32))
        rew = -actions.square().mean(dim=-1)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = torch.full((self.num_envs,), self._t % 4 == 0, dtype=torch.bool)
        return {"actor": obs}, rew, terminated, truncated, {}

    def close(self) -> None:
        pass


def test_stored_hidden_dim() -> None:
    assert stored_hidden_dim(32, "gru") == 32
    assert stored_hidden_dim(32, "vanilla") == 32
    assert stored_hidden_dim(32, "lstm") == 64

    with pytest.raises(ValueError, match="hidden_state_dim"):
        stored_hidden_dim(0, "gru")
    with pytest.raises(ValueError, match="Unsupported rnn_type"):
        stored_hidden_dim(16, "transformer")


def test_independent_joint_dist_additivity_and_shapes() -> None:
    torch.manual_seed(42)
    batch_size, act_dim, lat_dim = 7, 6, 10
    mean_a = torch.randn(batch_size, act_dim)
    std_a = torch.full((batch_size, act_dim), 0.4)
    mean_z = torch.randn(batch_size, lat_dim)
    std_z = torch.full((batch_size, lat_dim), 0.15)

    dist = IndependentJointDist(mean_a, std_a, mean_z, std_z)
    assert dist.mode.shape == (batch_size, act_dim + lat_dim)
    assert torch.allclose(dist.mode, torch.cat([mean_a, mean_z], dim=-1))

    sampled = dist.sample()
    assert sampled.shape == (batch_size, act_dim + lat_dim)
    a_part, z_part = dist.split(sampled)
    assert a_part.shape == (batch_size, act_dim)
    assert z_part.shape == (batch_size, lat_dim)

    joint_logp = dist.log_prob(sampled)
    expected_logp = dist.dist_a.log_prob(a_part) + dist.dist_z.log_prob(z_part)
    assert joint_logp.shape == (batch_size,)
    assert torch.allclose(joint_logp, expected_logp, atol=1e-6)

    joint_ent = dist.entropy()
    expected_ent = dist.dist_a.entropy() + dist.dist_z.entropy()
    assert joint_ent.shape == (batch_size,)
    assert torch.allclose(joint_ent, expected_ent, atol=1e-6)


@pytest.mark.parametrize("rnn_type", VALID_RNN_TYPES)
def test_actor_critic_s2pg_rnn_types_and_ema(rnn_type: RNNType) -> None:
    torch.manual_seed(0)
    obs_dim, act_dim, hidden_dim = 9, 6, 16
    lat_dim = stored_hidden_dim(hidden_dim, rnn_type)

    net_raw = ActorCriticS2PG(
        obs_dim=obs_dim,
        action_dim=act_dim,
        hidden_state_dim=hidden_dim,
        rnn_type=rnn_type,
        ema_coef=1.0,
        init_std_a=0.5,
        init_std_z=0.1,
    )
    assert torch.allclose(
        net_raw.log_std_a, torch.full((act_dim,), math.log(0.5)), atol=1e-6
    )
    assert torch.allclose(
        net_raw.log_std_z, torch.full((lat_dim,), math.log(0.1)), atol=1e-6
    )

    obs = torch.randn(5, obs_dim)
    hidden = torch.randn(5, lat_dim)
    prev_act = torch.randn(5, act_dim)

    dist, value = net_raw.get_dist_and_value(obs, hidden, prev_act)
    assert dist.mode.shape == (5, act_dim + lat_dim)
    assert value.shape == (5,)

    obs_3d = torch.randn(2, 3, obs_dim)
    hid_3d = torch.randn(2, 3, lat_dim)
    act_3d = torch.randn(2, 3, act_dim)
    mean_a_3d, mean_z_3d, val_3d = net_raw(obs_3d, hid_3d, act_3d)
    assert mean_a_3d.shape == (2, 3, act_dim)
    assert mean_z_3d.shape == (2, 3, lat_dim)
    assert val_3d.shape == (2, 3)

    net_frozen = ActorCriticS2PG(
        obs_dim=obs_dim,
        action_dim=act_dim,
        hidden_state_dim=hidden_dim,
        rnn_type=rnn_type,
        ema_coef=0.0,
    )
    net_frozen.load_state_dict(net_raw.state_dict())
    _, z_frozen = net_frozen.actor_means(obs, hidden, prev_act)
    assert torch.allclose(z_frozen, hidden, atol=1e-6)

    ema_alpha = 0.35
    net_ema = ActorCriticS2PG(
        obs_dim=obs_dim,
        action_dim=act_dim,
        hidden_state_dim=hidden_dim,
        rnn_type=rnn_type,
        ema_coef=ema_alpha,
    )
    net_ema.load_state_dict(net_raw.state_dict())
    _, raw_z = net_raw.actor_means(obs, hidden, prev_act)
    _, smoothed_z = net_ema.actor_means(obs, hidden, prev_act)
    expected_z = (1.0 - ema_alpha) * hidden + ema_alpha * raw_z
    assert torch.allclose(smoothed_z, expected_z, atol=1e-6)


def test_running_mean_std_welford_accuracy() -> None:
    rms = RunningMeanStd(dim=4, init_count=0.0, clip=None)
    x1 = torch.randn(32, 4) * 3.0 + 2.5
    x2 = torch.randn(48, 4) * 1.5 - 1.0
    rms.update(x1)
    rms.update(x2)
    full = torch.cat([x1, x2], dim=0)
    assert torch.allclose(rms.mean, full.mean(dim=0), atol=1e-5)
    assert torch.allclose(rms.var, full.var(dim=0, unbiased=False), atol=1e-5)


def test_episode_boundary_zeroing_and_gae_isolation() -> None:
    env = _DeterministicBoundaryVecEnv(obs_dim=5, action_dim=3)
    cfg = S2PGConfig(
        hidden_state_dim=8,
        rnn_type="lstm",
        num_steps_per_env=4,
        num_epochs=2,
        num_minibatches=2,
        seed=7,
    )
    trainer = S2PGPPOTrainer(vec_env=env, cfg=cfg)
    batch = trainer.collect_rollout()

    assert torch.equal(batch.hidden_states[0], torch.zeros_like(batch.hidden_states[0]))
    assert torch.equal(batch.prev_actions[0], torch.zeros_like(batch.prev_actions[0]))

    assert bool(batch.dones[1, 0]) and not bool(batch.dones[1, 1])
    assert batch.next_hidden_states[1, 0].abs().sum() > 0.0
    with torch.no_grad():
        dist_step1 = trainer.policy.get_dist(
            batch.obs[1], batch.hidden_states[1], batch.prev_actions[1]
        )
        recomputed_logp = dist_step1.log_prob(
            torch.cat([batch.actions[1], batch.next_hidden_states[1]], dim=-1)
        )
    assert torch.allclose(recomputed_logp, batch.log_probs[1], atol=1e-6)

    assert torch.equal(
        batch.hidden_states[2, 0], torch.zeros_like(batch.hidden_states[2, 0])
    )
    assert torch.equal(
        batch.prev_actions[2, 0], torch.zeros_like(batch.prev_actions[2, 0])
    )
    assert torch.allclose(batch.hidden_states[2, 1], batch.next_hidden_states[1, 1])
    assert torch.allclose(batch.prev_actions[2, 1], batch.actions[1, 1])

    assert not bool(batch.dones[2, 0]) and bool(batch.dones[2, 1])
    assert torch.equal(
        batch.hidden_states[3, 1], torch.zeros_like(batch.hidden_states[3, 1])
    )
    assert torch.equal(
        batch.prev_actions[3, 1], torch.zeros_like(batch.prev_actions[3, 1])
    )
    assert torch.allclose(batch.hidden_states[3, 0], batch.next_hidden_states[2, 0])

    rewards = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    values = torch.tensor([[0.5], [0.8]], dtype=torch.float32)
    dones = torch.tensor([[True], [False]], dtype=torch.bool)
    last_val = torch.tensor([10.0], dtype=torch.float32)
    adv, ret = compute_gae(
        rewards, values, dones, last_val, gamma=0.99, gae_lambda=0.95
    )
    assert torch.isclose(adv[0, 0], torch.tensor(1.0 - 0.5), atol=1e-6)
    assert torch.isclose(ret[0, 0], torch.tensor(1.0), atol=1e-6)


@pytest.mark.parametrize("rnn_type", VALID_RNN_TYPES)
def test_s2pg_trainer_update_and_checkpoint(rnn_type: RNNType, tmp_path: Path) -> None:
    env = _DeterministicBoundaryVecEnv(obs_dim=6, action_dim=4)
    cfg = S2PGConfig(
        hidden_state_dim=12,
        rnn_type=rnn_type,
        num_steps_per_env=8,
        num_epochs=3,
        num_minibatches=2,
        lr=1e-3,
        seed=3,
    )
    trainer = S2PGPPOTrainer(vec_env=env, cfg=cfg)
    before = {k: v.detach().clone() for k, v in trainer.policy.named_parameters()}

    _, metrics = trainer.train_step(progress=0.1)
    for key, val in metrics.items():
        assert math.isfinite(val), f"Non-finite metric {key}={val}"

    changed = [
        name
        for name, param in trainer.policy.named_parameters()
        if not torch.equal(param.detach(), before[name])
    ]
    assert "action_mean_head.weight" in changed
    assert "critic_head.weight" in changed
    assert "log_std_a" in changed
    assert "log_std_z" in changed

    ckpt_file = tmp_path / "s2pg_test_ckpt.pt"
    trainer.save_checkpoint(ckpt_file)
    restored = S2PGPPOTrainer(vec_env=env, cfg=cfg)
    restored.load_checkpoint(ckpt_file)
    assert restored.env_steps == trainer.env_steps
    for name, param in trainer.policy.named_parameters():
        assert torch.allclose(param, restored.policy.state_dict()[name])


def test_baseline_feedforward_ppo_and_mjlab_adapter() -> None:
    fake_mjlab = _FakeMjlabEnv(num_envs=4, obs_dim=9, action_dim=6)
    vec_env = MjlabTensorVecEnv(fake_mjlab)
    cfg = S2PGConfig(num_steps_per_env=6, num_epochs=2, num_minibatches=2, seed=11)
    ff_trainer = baseline_feedforward_ppo(vec_env=vec_env, cfg=cfg)
    _, ff_metrics = ff_trainer.train_step(progress=0.0)
    assert all(math.isfinite(v) for v in ff_metrics.values())

    raw_mjlab = _FakeMjlabEnv(num_envs=4, obs_dim=9, action_dim=6)
    args = MjlabTrainArgs(
        env_id="myoElbowPose1D6MRandom-v0",
        num_envs=4,
        max_iterations=2,
        save_interval=0,
        export_ts=False,
        s2pg=S2PGConfig(num_steps_per_env=4, num_epochs=2, num_minibatches=2, seed=5),
    )
    summary = run_s2pg_on_mjlab_env(raw_mjlab, args, log_dir=None)
    assert summary["env_steps"] == 32
    assert summary["final_mean_success"] is not None


@pytest.mark.parametrize("rnn_type", VALID_RNN_TYPES)
def test_stateful_export_torchscript_and_onnx(
    rnn_type: RNNType, tmp_path: Path
) -> None:
    torch.manual_seed(9)
    policy = ActorCriticS2PG(
        obs_dim=9,
        action_dim=6,
        hidden_state_dim=10,
        rnn_type=rnn_type,
        ema_coef=0.8,
    )
    policy.obs_rms.update(torch.randn(64, 9) * 2.0 + 1.0)
    policy.eval()

    obs = torch.randn(3, 9)
    hidden, prev_act = policy.initial_state(3)
    hidden = hidden + 0.2
    prev_act = prev_act - 0.1

    with torch.no_grad():
        expected_act, expected_next_h = policy.deterministic_act(obs, hidden, prev_act)
        wrapper = StatefulS2PGExportWrapper(policy)
        wrap_act, wrap_next_h = wrapper(obs, hidden, prev_act)
    assert torch.allclose(wrap_act, expected_act, atol=1e-6)
    assert torch.allclose(wrap_next_h, expected_next_h, atol=1e-6)

    ts_path = tmp_path / f"stateful_{rnn_type}.pt"
    export_torchscript(policy, output=ts_path)
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        loaded_ts = torch.jit.load(str(ts_path))
        ts_act, ts_next_h = loaded_ts(obs, hidden, prev_act)
    assert torch.allclose(ts_act, expected_act, atol=1e-6)
    assert torch.allclose(ts_next_h, expected_next_h, atol=1e-6)

    try:
        import onnx  # noqa: F401
    except ImportError:
        return
    onnx_path = tmp_path / f"stateful_{rnn_type}.onnx"
    export_onnx_stateful(policy, output=onnx_path)
    assert onnx_path.is_file()


def test_smoke_benchmark_and_fatigue_flag(tmp_path: Path) -> None:
    out_json = tmp_path / "bench_smoke.json"
    results = run_s2pg_benchmark(
        total_steps=512,
        eval_interval=256,
        seeds=(0,),
        n_envs=2,
        n_eval_episodes=2,
        out_json=out_json,
        verbose=False,
    )
    assert out_json.is_file()
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["metadata"]["fatigue"] is False

    for env_id in ("myoElbowPose1D6MRandom-v0", "myoHandPoseRandom-v0"):
        assert env_id in results["environments"]
        for method in ("PPO", "S2PG-PPO"):
            agg = results["environments"][env_id][method]["aggregate"]
            assert math.isfinite(agg["final_return_mean"])
            assert 0.0 <= agg["final_success_mean"] <= 1.0
            assert agg["train_wall_clock_s_mean"] > 0.0
            assert agg["peak_tracemalloc_mb_max"] > 0.0
            assert len(agg["learning_curve"]) == 3

    fatigue_results = run_s2pg_benchmark(
        env_ids=("myoElbowPose1D6MRandom-v0",),
        total_steps=256,
        eval_interval=256,
        seeds=(0,),
        n_envs=2,
        n_eval_episodes=1,
        fatigue=True,
        verbose=False,
    )
    assert fatigue_results["metadata"]["fatigue"] is True
    assert fatigue_results["metadata"]["task_modification"] is not None
