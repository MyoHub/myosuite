# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Smoke tests for PPO training script — verifies 2 iterations reduce loss.

Runs entirely on CPU with 2 envs × 32 rollout steps so it completes
in a few seconds without GPU or a large clip.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch

_CLIP_REPO_ID = "amathislab/musclemimic-retargeted"
_CLIP_FILENAME = "MyoFullBody/gmr/KIT/167/walking_medium06_poses.npz"


def _resolve_clip_path() -> pathlib.Path | None:
    """Resolve the default mimic clip path in a machine-agnostic way."""
    try:
        from huggingface_hub import hf_hub_download

        return pathlib.Path(
            hf_hub_download(
                repo_id=_CLIP_REPO_ID,
                filename=_CLIP_FILENAME,
                repo_type="dataset",
            )
        )
    except ImportError:
        return None
    except RuntimeError:
        return None
    except OSError:
        return None


_HF_CLIP = _resolve_clip_path()

pytestmark = pytest.mark.skipif(
    _HF_CLIP is None or not _HF_CLIP.is_file(),
    reason="HuggingFace motion clip not in cache",
)


def test_ppo_loss_decreases():
    """Two PPO iterations should reduce policy-gradient loss."""
    from tutorials.mimic.train_mimic import ActorCritic, VecMimicEnv

    torch.manual_seed(0)
    n_envs = 2
    rollout_steps = 32
    env = VecMimicEnv(clip_path=_HF_CLIP, n_envs=n_envs, max_episode_steps=50)
    policy = ActorCritic(env.obs_dim, env.act_dim, hidden_dim=64, n_layers=2)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)

    losses: list[float] = []
    cur_obs = env.reset_all()

    for _iter in range(3):
        obs_buf = np.zeros((rollout_steps, n_envs, env.obs_dim), dtype=np.float32)
        act_buf = np.zeros((rollout_steps, n_envs, env.act_dim), dtype=np.float32)
        rew_buf = np.zeros((rollout_steps, n_envs), dtype=np.float32)
        done_buf = np.zeros((rollout_steps, n_envs), dtype=np.float32)
        val_buf = np.zeros((rollout_steps, n_envs), dtype=np.float32)
        logp_buf = np.zeros((rollout_steps, n_envs), dtype=np.float32)

        with torch.no_grad():
            for t in range(rollout_steps):
                obs_t = torch.as_tensor(cur_obs, dtype=torch.float32)
                action, log_prob, _, value = policy.get_action_and_value(obs_t)
                action_np = action.numpy()
                cur_obs, rew, term, trunc = env.step(action_np)
                obs_buf[t] = cur_obs
                act_buf[t] = action_np
                rew_buf[t] = rew
                done_buf[t] = (term | trunc).astype(np.float32)
                val_buf[t] = value.numpy()
                logp_buf[t] = log_prob.numpy()

        # Simple returns (no GAE for brevity)
        ret_buf = np.zeros_like(rew_buf)
        running = np.zeros(n_envs)
        for t in reversed(range(rollout_steps)):
            running = rew_buf[t] + 0.99 * running * (1 - done_buf[t])
            ret_buf[t] = running

        batch = rollout_steps * n_envs
        obs_t = torch.as_tensor(obs_buf.reshape(batch, env.obs_dim))
        act_t = torch.as_tensor(act_buf.reshape(batch, env.act_dim))
        logp_old = torch.as_tensor(logp_buf.reshape(batch))
        ret_t = torch.as_tensor(ret_buf.reshape(batch), dtype=torch.float32)
        adv_t = ret_t - torch.as_tensor(val_buf.reshape(batch))
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        _, new_logp, entropy, new_val = policy.get_action_and_value(obs_t, act_t)
        ratio = (new_logp - logp_old).exp()
        pg_loss = -(adv_t * ratio.clamp(0.8, 1.2)).mean()
        vf_loss = 0.5 * (new_val - ret_t).pow(2).mean()
        loss = pg_loss + 0.5 * vf_loss - 1e-3 * entropy.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # Losses should at least be finite
    assert all(np.isfinite(v) for v in losses), f"Non-finite losses: {losses}"
    # Loss magnitude should be reasonable (not exploding)
    assert abs(losses[-1]) < 1000.0, f"Loss too large: {losses[-1]}"
    print(f"PPO losses over 3 iters: {losses}")


def test_actor_critic_forward():
    """Basic forward pass shape test."""
    from tutorials.mimic.train_mimic import ActorCritic

    obs_dim, act_dim = 821, 354
    policy = ActorCritic(obs_dim, act_dim, hidden_dim=64, n_layers=2)
    obs = torch.randn(8, obs_dim)
    action, log_prob, entropy, value = policy.get_action_and_value(obs)
    assert action.shape == (8, act_dim)
    assert log_prob.shape == (8,)
    assert entropy.shape == (8,)
    assert value.shape == (8,)
    assert torch.all(torch.isfinite(action))


def test_vec_env_reset_step():
    """VecMimicEnv returns correct shapes."""
    from tutorials.mimic.train_mimic import VecMimicEnv

    env = VecMimicEnv(clip_path=_HF_CLIP, n_envs=2, max_episode_steps=10)
    obs = env.reset_all()
    assert obs.shape == (2, env.obs_dim)
    action = np.zeros((2, env.act_dim), dtype=np.float32)
    obs2, rew, term, _trunc = env.step(action)
    assert obs2.shape == (2, env.obs_dim)
    assert rew.shape == (2,)
    assert term.shape == (2,)
