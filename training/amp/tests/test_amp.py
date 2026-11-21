# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Unit and smoke tests for myosuite_amp."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from myosuite import make_env
from myosuite_amp import (
    AMPConfig,
    AMPTrainer,
    BCConfig,
    CpuVecEnv,
    ExpertDataset,
    MotionDiscriminator,
    RunningMeanStd,
    collect_expert_dataset,
    export_policy_onnx,
    export_policy_torchscript,
    run_amp_benchmark,
    train_bc,
)
from myosuite_amp.train_mjlab import MjlabVecEnvAdapter, load_motion_npz

ENV_ID = "myoElbowPose1D6MRandom-v0"


def test_running_mean_std_welford_accuracy() -> None:
    rms = RunningMeanStd(shape=6, eps=0.0)
    rms.count.fill_(0.0)
    torch.manual_seed(0)
    x1 = torch.randn(32, 6) * 3.0 + 2.5
    x2 = torch.randn(48, 6) * 1.5 - 1.0
    rms.update(x1)
    rms.update(x2)
    full = torch.cat([x1, x2], dim=0)
    np.testing.assert_allclose(rms.mean.numpy(), full.mean(dim=0).numpy(), atol=1e-5)
    np.testing.assert_allclose(
        rms.var.numpy(), full.var(dim=0, unbiased=False).numpy(), atol=1e-5
    )
    normed = rms.normalize(full, clip=5.0)
    assert float(normed.abs().max()) <= 5.0


def test_amp_discriminator_loss_grad_penalty_and_reward_bounds() -> None:
    torch.manual_seed(1)
    obs_dim = 9
    disc = MotionDiscriminator(
        obs_dim=obs_dim,
        mode="amp",
        hidden_dims=(64, 64),
        grad_penalty_weight=5.0,
        logit_reg_weight=0.05,
    )
    opt = torch.optim.Adam(disc.parameters(), lr=3e-3)

    exp_obs = torch.ones(64, obs_dim) * 1.5 + 0.1 * torch.randn(64, obs_dim)
    exp_next = exp_obs + 0.2
    pol_obs = -torch.ones(64, obs_dim) * 1.5 + 0.1 * torch.randn(64, obs_dim)
    pol_next = pol_obs - 0.2

    loss0, stats0 = disc.compute_loss(exp_obs, exp_next, pol_obs, pol_next)
    assert np.isfinite(stats0.total_loss)
    assert stats0.grad_penalty > 0.0
    assert stats0.weighted_grad_penalty == pytest.approx(
        5.0 * stats0.grad_penalty, rel=1e-5
    )
    assert stats0.logit_reg_loss > 0.0

    for _ in range(25):
        loss, stats = disc.compute_loss(exp_obs, exp_next, pol_obs, pol_next)
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert stats.gan_loss < stats0.gan_loss
    assert stats.expert_logit_mean > stats.policy_logit_mean

    extreme_obs = torch.linspace(-50.0, 50.0, 128 * obs_dim).reshape(128, obs_dim)
    extreme_next = extreme_obs * -1.5
    rewards = disc.compute_style_reward(extreme_obs, extreme_next)
    assert rewards.shape == (128,)
    assert float(rewards.min().item()) >= 0.0
    assert float(rewards.max().item()) <= 1.0

    exp_rew = disc.compute_style_reward(exp_obs, exp_next).mean()
    pol_rew = disc.compute_style_reward(pol_obs, pol_next).mean()
    assert float(exp_rew) > float(pol_rew)


def test_gail_discriminator_and_feature_slice() -> None:
    torch.manual_seed(2)
    obs_dim = 9
    disc = MotionDiscriminator(
        obs_dim=obs_dim,
        mode="gail",
        feature_indices=(0, 1, 3, 4),
        hidden_dims=(64, 64),
        grad_penalty_weight=5.0,
    )
    assert disc.transition_dim == 8
    opt = torch.optim.Adam(disc.parameters(), lr=3e-3)

    exp_obs = torch.zeros(64, obs_dim)
    exp_obs[:, [0, 1, 3, 4]] = 2.0
    exp_next = exp_obs + 0.5
    pol_obs = torch.zeros(64, obs_dim)
    pol_obs[:, [0, 1, 3, 4]] = -2.0
    pol_next = pol_obs - 0.5

    loss0, stats0 = disc.compute_loss(exp_obs, exp_next, pol_obs, pol_next)
    assert stats0.grad_penalty > 0.0
    for _ in range(20):
        loss, stats = disc.compute_loss(exp_obs, exp_next, pol_obs, pol_next)
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert stats.gan_loss < float(loss0.detach().item())
    r_exp = disc.compute_style_reward(exp_obs, exp_next)
    r_pol = disc.compute_style_reward(pol_obs, pol_next)
    assert float(r_exp.min()) >= 0.0
    assert float(r_pol.min()) >= 0.0
    assert float(r_exp.mean()) > float(r_pol.mean())

    with pytest.raises(ValueError, match="out of bounds"):
        MotionDiscriminator(obs_dim=obs_dim, feature_indices=(0, 9))


def test_reward_blending_contract() -> None:
    env_r = torch.tensor([-2.0, 0.0, 4.0])
    style_r = torch.tensor([0.2, 0.8, 1.0])
    blended = MotionDiscriminator.combine_rewards(
        env_r, style_r, proportion_env_reward=0.5
    )
    np.testing.assert_allclose(blended.numpy(), np.array([-0.9, 0.4, 2.5]), atol=1e-6)
    np.testing.assert_allclose(
        MotionDiscriminator.combine_rewards(env_r, style_r, 1.0).numpy(),
        env_r.numpy(),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        MotionDiscriminator.combine_rewards(env_r, style_r, 0.0).numpy(),
        style_r.numpy(),
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="proportion_env_reward"):
        MotionDiscriminator.combine_rewards(env_r, style_r, 1.5)


@pytest.mark.tier2
def test_amp_trainer_step_blending_and_reproducibility() -> None:
    ds = collect_expert_dataset(
        ENV_ID, n_transitions=128, seed=3, expert_steps=128, n_envs=2
    )
    motion_ds = ds.without_actions()
    assert motion_ds.actions is None

    def _single_run() -> tuple[float, float, float, float]:
        vec = CpuVecEnv(ENV_ID, num_envs=2, device="cpu")
        try:
            cfg = AMPConfig(
                mode="amp",
                use_discriminator=True,
                proportion_env_reward=0.5,
                rollout_steps=32,
                ppo_epochs=2,
                num_minibatches=2,
                disc_updates_per_iter=2,
                disc_batch_size=32,
                seed=11,
            )
            trainer = AMPTrainer(vec, cfg, expert_transitions=motion_ds)
            step_stats = trainer.collect_and_update()
            return (
                step_stats.mean_env_reward,
                step_stats.mean_style_reward,
                step_stats.mean_blended_reward,
                step_stats.disc_grad_penalty,
            )
        finally:
            vec.close()

    env_r1, style_r1, blend_r1, gp1 = _single_run()
    env_r2, style_r2, blend_r2, gp2 = _single_run()
    assert 0.0 <= style_r1 <= 1.0
    assert blend_r1 == pytest.approx(0.5 * style_r1 + 0.5 * env_r1, abs=1e-5)
    assert gp1 > 0.0
    assert (env_r1, style_r1, blend_r1, gp1) == pytest.approx(
        (env_r2, style_r2, blend_r2, gp2), abs=1e-6
    )


def test_behavior_cloning_reduces_mse() -> None:
    rng = np.random.default_rng(7)
    obs = rng.standard_normal((256, 9)).astype(np.float32)
    w_true = rng.uniform(-0.4, 0.4, size=(9, 6)).astype(np.float32)
    actions = np.tanh(obs @ w_true).astype(np.float32)

    actor, losses = train_bc(
        obs,
        actions,
        cfg=BCConfig(epochs=25, batch_size=64, learning_rate=3e-3, seed=5),
    )
    assert len(losses) == 25
    assert losses[-1] < 0.25 * losses[0]
    pred = actor.predict_np(obs[:8])
    assert pred.shape == (8, 6)
    assert float(np.max(np.abs(pred))) <= 1.0


def test_mjlab_vec_env_adapter_and_npz_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    ds = ExpertDataset(
        obs=rng.standard_normal((64, 9)).astype(np.float32),
        next_obs=rng.standard_normal((64, 9)).astype(np.float32),
        actions=rng.uniform(-1.0, 1.0, size=(64, 6)).astype(np.float32),
        rewards=np.zeros(64, dtype=np.float32),
        dones=np.zeros(64, dtype=bool),
        env_id=ENV_ID,
    )
    npz_path = ds.save_npz(tmp_path / "demo.npz")
    exp_obs, exp_next = load_motion_npz(npz_path, device="cpu")
    assert exp_obs.shape == (64, 9)
    assert exp_next.shape == (64, 9)

    class _FakeMjlabWrapper:
        def __init__(self) -> None:
            self.num_envs = 4
            self.num_actions = 6
            self.device = "cpu"
            self.unwrapped = self
            self._step = 0

        def get_observations(self) -> dict[str, torch.Tensor]:
            return {"actor": torch.zeros(self.num_envs, 9)}

        def step(
            self, actions: torch.Tensor
        ) -> tuple[
            dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
        ]:
            self._step += 1
            next_obs = {"actor": actions.mean(dim=-1, keepdim=True).repeat(1, 9)}
            rew = -actions.square().mean(dim=-1)
            dones = torch.tensor([self._step % 4 == 0, False, False, False])
            extras = {"time_outs": dones.clone()}
            return next_obs, rew, dones, extras

        def close(self) -> None:
            pass

    adapter = MjlabVecEnvAdapter(_FakeMjlabWrapper(), clip_actions=1.0)
    trainer = AMPTrainer(
        adapter,
        AMPConfig(
            mode="gail",
            rollout_steps=8,
            ppo_epochs=2,
            num_minibatches=2,
            disc_updates_per_iter=2,
            disc_batch_size=16,
            seed=0,
        ),
        expert_transitions=(exp_obs, exp_next),
    )
    history = trainer.train(total_steps=32)
    adapter.close()
    assert history[-1].env_steps == 32
    assert np.isfinite(history[-1].disc_loss)


@pytest.mark.tier2
def test_torchscript_and_onnx_policy_export(tmp_path: Path) -> None:
    vec = CpuVecEnv(ENV_ID, num_envs=2, device="cpu")
    try:
        trainer = AMPTrainer(
            vec,
            AMPConfig(use_discriminator=False, rollout_steps=16, ppo_epochs=1, seed=0),
        )
        trainer.collect_and_update()
        actor = trainer.actor_critic
    finally:
        vec.close()

    ts_path = export_policy_torchscript(actor, tmp_path / "policy_ts.pt")
    assert ts_path.exists()
    loaded_ts = torch.jit.load(str(ts_path))

    env = make_env(ENV_ID)
    try:
        obs, _ = env.reset(seed=0)
        obs_t = torch.as_tensor(obs[None, :], dtype=torch.float32)
        with torch.no_grad():
            expected = actor.act_deterministic(obs_t).numpy()
            actual_ts = loaded_ts(obs_t).numpy()
        np.testing.assert_allclose(actual_ts, expected, atol=1e-5)
        next_obs, rew, _, _, _ = env.step(actual_ts.squeeze(0))
        assert next_obs.shape == obs.shape
        assert np.isfinite(float(rew))
    finally:
        env.close()

    onnx_path = tmp_path / "policy.onnx"
    if (
        importlib.util.find_spec("onnx") is not None
        and importlib.util.find_spec("onnxruntime") is not None
    ):
        import onnxruntime as ort

        export_policy_onnx(actor, onnx_path)
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(["action"], {"obs": obs_t.numpy()})[0]
        np.testing.assert_allclose(ort_out, expected, atol=1e-5)
    else:
        with pytest.raises(ImportError, match="onnx"):
            export_policy_onnx(actor, onnx_path)


@pytest.mark.tier2
def test_smoke_run_amp_benchmark(tmp_path: Path) -> None:
    out_json = tmp_path / "amp_smoke.json"
    res = run_amp_benchmark(
        env_ids=(ENV_ID,),
        total_steps=512,
        eval_interval=256,
        seeds=(0,),
        n_envs=2,
        n_eval_episodes=2,
        expert_steps=512,
        bc_epochs=6,
        out_json=out_json,
    )
    assert out_json.exists()
    parsed = json.loads(out_json.read_text(encoding="utf-8"))
    env_block = parsed["environments"][ENV_ID]
    assert env_block["expert_setup"]["amp_gail_actions_stripped"] is True
    assert set(env_block["arms"].keys()) == {"PPO", "BC", "GAIL", "AMP"}

    for online_arm in ("PPO", "GAIL", "AMP"):
        arm_data = env_block["arms"][online_arm]
        assert arm_data["env_step_budget"] == 512
        assert [c["env_steps"] for c in arm_data["curve"]] == [0, 256, 512]
        assert arm_data["wall_time_mean_s"] > 0.0
        assert arm_data["peak_memory_mean_mb"] > 0.0

    bc_data = env_block["arms"]["BC"]
    assert bc_data["env_step_budget"] == 0
    assert bc_data["uses_expert_actions"] is True
    assert env_block["arms"]["AMP"]["uses_expert_actions"] is False
    assert env_block["arms"]["GAIL"]["uses_expert_actions"] is False
    assert ENV_ID in res["markdown_table"]
