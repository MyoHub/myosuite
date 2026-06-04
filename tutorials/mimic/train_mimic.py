#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""PPO training for MuscleMimic clip-following (CPU / GPU, single process).

Implements a self-contained PPO with:
  - Actor-critic MLP (configurable depth/width, SiLU + LayerNorm)
  - GAE advantage estimation (λ=0.95, γ=0.99)
  - PPO clip loss (ε=0.2)
  - Entropy bonus
  - DeepMimic composite reward (via MuscleMimicClipEnvV0)

Runs on CPU when no CUDA is available; set CUDA_VISIBLE_DEVICES for GPU.

Usage::

    # Quick test (2k steps):
    python tutorials/mimic/train_mimic.py --total_steps 2000 --n_envs 4

    # Full training (2B steps recommended for convergence — use GPU):
    python tutorials/mimic/train_mimic.py \\
        --clip /path/to/walking_medium06_poses.npz \\
        --total_steps 2_000_000_000 \\
        --n_envs 512 \\
        --hidden_dim 1024 --n_layers 16 \\
        --device cuda

    # mjlab backend (when mjlab is installed):
    python tutorials/mimic/train_mimic.py --backend mjlab --n_envs 8192

Environment vars:
    MIMIC_CLIP: default clip path (overridden by --clip)
    WANDB_DISABLED: set to 1 to skip wandb logging

Clip metadata:
    Full-width clips keep the historical ``qpos``/``qvel`` layout.
    Partial clips are also supported when the NPZ stores either
    ``qpos_names``/``qvel_names`` or shared ``joint_names`` metadata listing
    the model joints covered by those arrays.  In that case the baseline only
    applies qpos/qvel tracking losses on the named joint subset.
"""

from __future__ import annotations

import argparse
import importlib
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_CLIP_REPO_ID = "amathislab/musclemimic-retargeted"
_CLIP_FILENAME = "MyoFullBody/gmr/KIT/167/walking_medium06_poses.npz"


def _default_clip() -> Path:
    """Resolve default retargeted clip path in a machine-agnostic way."""
    env_raw = os.environ.get("MIMIC_CLIP", "").strip()
    if env_raw:
        env = Path(env_raw).expanduser()
        if env.is_file():
            return env
    hf_hub_download = importlib.import_module("huggingface_hub").hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=_CLIP_REPO_ID,
            filename=_CLIP_FILENAME,
            repo_type="dataset",
        )
    )


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------


class _LayerNormSiLUBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim, eps=1e-5)
        self.act = nn.SiLU()
        nn.init.orthogonal_(self.fc.weight, gain=2.0**0.5)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.ln(self.fc(x)))


class ActorCritic(nn.Module):
    """Shared-trunk MLP actor-critic for continuous muscle control.

    Args:
        obs_dim: Observation dimension.
        act_dim: Action dimension.
        hidden_dim: Width of each hidden layer.
        n_layers: Number of hidden layers.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(obs_dim, hidden_dim), nn.SiLU()]
        nn.init.orthogonal_(layers[0].weight, 2.0**0.5)
        nn.init.zeros_(layers[0].bias)
        for _ in range(n_layers - 1):
            layers.append(_LayerNormSiLUBlock(hidden_dim))
        self.trunk = nn.Sequential(*layers)

        self.actor_mean = nn.Linear(hidden_dim, act_dim)
        nn.init.orthogonal_(self.actor_mean.weight, 0.01)
        nn.init.zeros_(self.actor_mean.bias)

        self.log_std = nn.Parameter(torch.full((act_dim,), 3.0))  # init std ≈ 3 (paper)

        self.critic = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(self.critic.weight, 1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        return self.actor_mean(features), self.critic(features).squeeze(-1)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        mean = self.actor_mean(features)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(features).squeeze(-1)
        return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# Vectorised environment wrapper (sequential multi-env on CPU)
# ---------------------------------------------------------------------------


class VecMimicEnv:
    """Minimal synchronous vectorised wrapper over MuscleMimicClipEnvV0.

    Args:
        clip_path: Path to retargeted motion clip.
        n_envs: Number of parallel environments.
        max_episode_steps: Episode truncation length.
        kwargs: Extra arguments forwarded to each env.
    """

    def __init__(
        self,
        clip_path: Path,
        n_envs: int = 4,
        max_episode_steps: int = 300,
        **kwargs: Any,
    ) -> None:
        from myosuite.envs.myo.tasks.mimic.clip_env import MuscleMimicClipEnvV0

        self.n_envs = n_envs
        self.max_episode_steps = max_episode_steps
        self._envs = [
            MuscleMimicClipEnvV0(clip_path=clip_path, seed=i, **kwargs)
            for i in range(n_envs)
        ]
        self._steps = np.zeros(n_envs, dtype=np.int32)
        obs, _ = zip(*[e.reset() for e in self._envs])
        self._last_obs = np.stack(obs)
        self.obs_dim = self._last_obs.shape[1]
        self.act_dim = self._envs[0].action_space.shape[0]
        self.act_low = self._envs[0].action_space.low
        self.act_high = self._envs[0].action_space.high

    def reset_all(self) -> np.ndarray:
        obs, _ = zip(*[e.reset() for e in self._envs])
        self._last_obs = np.stack(obs)
        self._steps[:] = 0
        return self._last_obs.copy()

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        actions = np.clip(actions, self.act_low, self.act_high)
        obs_list, rews, terms, truncs = [], [], [], []
        for i, (env, a) in enumerate(zip(self._envs, actions)):
            o, r, term, trunc, _ = env.step(a)
            self._steps[i] += 1
            trunc = trunc or (self._steps[i] >= self.max_episode_steps)
            done = term or trunc
            if done:
                o, _ = env.reset()
                self._steps[i] = 0
            obs_list.append(o)
            rews.append(r)
            terms.append(term)
            truncs.append(trunc)
        self._last_obs = np.stack(obs_list)
        return (
            self._last_obs.copy(),
            np.array(rews, dtype=np.float32),
            np.array(terms, dtype=bool),
            np.array(truncs, dtype=bool),
        )


# ---------------------------------------------------------------------------
# PPO training loop
# ---------------------------------------------------------------------------


def train(
    clip_path: Path,
    total_steps: int = 10_000,
    n_envs: int = 4,
    rollout_steps: int = 128,
    hidden_dim: int = 256,
    n_layers: int = 4,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    n_epochs: int = 4,
    n_minibatches: int = 4,
    entropy_coef: float = 1e-3,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    device: str = "cpu",
    save_path: str | None = None,
    log_interval: int = 10,
) -> ActorCritic:
    """Run PPO training and return the trained actor-critic.

    Args:
        clip_path: Path to retargeted MotionClip .npz file.
        total_steps: Total environment timesteps to train for.
        n_envs: Number of parallel environments.
        rollout_steps: Steps collected per env per PPO iteration.
        hidden_dim: Hidden layer width.
        n_layers: Number of hidden layers.
        lr: Initial learning rate (linearly annealed to 0).
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        clip_eps: PPO clip epsilon.
        n_epochs: PPO update epochs per iteration.
        n_minibatches: Number of minibatches per epoch.
        entropy_coef: Entropy bonus coefficient.
        vf_coef: Value function loss coefficient.
        max_grad_norm: Gradient clip norm.
        device: torch device string.
        save_path: If given, save checkpoint every log_interval iterations.
        log_interval: Log every N iterations.

    Returns:
        Trained :class:`ActorCritic` model.
    """
    dev = torch.device(device)
    print(f"Device: {dev}")

    env = VecMimicEnv(clip_path=clip_path, n_envs=n_envs, max_episode_steps=300)
    policy = ActorCritic(env.obs_dim, env.act_dim, hidden_dim, n_layers).to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    total_iterations = total_steps // (n_envs * rollout_steps)
    batch_size = n_envs * rollout_steps
    minibatch_size = batch_size // n_minibatches

    print(
        f"Training: {total_steps:,} steps, {total_iterations} iterations, "
        f"{n_envs} envs × {rollout_steps} steps/iter = {batch_size} batch"
    )

    # Pre-allocate rollout buffers
    obs_buf = np.zeros((rollout_steps, n_envs, env.obs_dim), dtype=np.float32)
    act_buf = np.zeros((rollout_steps, n_envs, env.act_dim), dtype=np.float32)
    rew_buf = np.zeros((rollout_steps, n_envs), dtype=np.float32)
    done_buf = np.zeros((rollout_steps, n_envs), dtype=np.float32)
    val_buf = np.zeros((rollout_steps, n_envs), dtype=np.float32)
    logp_buf = np.zeros((rollout_steps, n_envs), dtype=np.float32)

    cur_obs = env.reset_all()
    ep_rewards: list[float] = []
    ep_reward_acc = np.zeros(n_envs, dtype=np.float32)
    t0 = time.monotonic()
    global_step = 0

    for iteration in range(1, total_iterations + 1):
        # Linear LR annealing
        frac = 1.0 - (iteration - 1) / total_iterations
        for g in opt.param_groups:
            g["lr"] = lr * frac

        # ---- Rollout ----
        policy.eval()
        with torch.no_grad():
            for t in range(rollout_steps):
                obs_t = torch.as_tensor(cur_obs, dtype=torch.float32, device=dev)
                action, log_prob, _, value = policy.get_action_and_value(obs_t)
                action_np = action.cpu().numpy()
                next_obs, rew, term, trunc = env.step(action_np)
                done = term | trunc

                obs_buf[t] = cur_obs
                act_buf[t] = action_np
                rew_buf[t] = rew
                done_buf[t] = done.astype(np.float32)
                val_buf[t] = value.cpu().numpy()
                logp_buf[t] = log_prob.cpu().numpy()

                ep_reward_acc += rew
                for i in range(n_envs):
                    if done[i]:
                        ep_rewards.append(float(ep_reward_acc[i]))
                        ep_reward_acc[i] = 0.0

                cur_obs = next_obs
                global_step += n_envs

            # Bootstrap value for last obs
            last_val = policy(
                torch.as_tensor(cur_obs, dtype=torch.float32, device=dev)
            )[1]
            last_val = last_val.cpu().numpy()

        # ---- GAE advantage ----
        adv_buf = np.zeros_like(rew_buf)
        last_gae = np.zeros(n_envs, dtype=np.float32)
        for t in reversed(range(rollout_steps)):
            next_val = val_buf[t + 1] if t < rollout_steps - 1 else last_val
            delta = rew_buf[t] + gamma * next_val * (1 - done_buf[t]) - val_buf[t]
            last_gae = delta + gamma * gae_lambda * (1 - done_buf[t]) * last_gae
            adv_buf[t] = last_gae
        ret_buf = adv_buf + val_buf

        # ---- PPO update ----
        obs_t = torch.as_tensor(obs_buf.reshape(batch_size, env.obs_dim), device=dev)
        act_t = torch.as_tensor(act_buf.reshape(batch_size, env.act_dim), device=dev)
        logp_old = torch.as_tensor(logp_buf.reshape(batch_size), device=dev)
        ret_t = torch.as_tensor(
            ret_buf.reshape(batch_size), dtype=torch.float32, device=dev
        )
        adv_t = torch.as_tensor(
            adv_buf.reshape(batch_size), dtype=torch.float32, device=dev
        )
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        policy.train()
        pg_losses, vf_losses, ent_losses = [], [], []
        for _ in range(n_epochs):
            perm = torch.randperm(batch_size, device=dev)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = perm[start : start + minibatch_size]
                _, new_logp, entropy, new_val = policy.get_action_and_value(
                    obs_t[mb_idx], act_t[mb_idx]
                )
                ratio = (new_logp - logp_old[mb_idx]).exp()
                pg_loss1 = -adv_t[mb_idx] * ratio
                pg_loss2 = -adv_t[mb_idx] * ratio.clamp(1 - clip_eps, 1 + clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                vf_loss = 0.5 * (new_val - ret_t[mb_idx]).pow(2).mean()
                ent_loss = -entropy.mean()
                loss = pg_loss + vf_coef * vf_loss + entropy_coef * ent_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                opt.step()
                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())

        # ---- Logging ----
        if iteration % log_interval == 0 or iteration == total_iterations:
            elapsed = time.monotonic() - t0
            ep_rew_mean = np.mean(ep_rewards[-100:]) if ep_rewards else float("nan")
            sps = global_step / elapsed
            print(
                f"iter={iteration:6d}  steps={global_step:10,d}  "
                f"ep_rew={ep_rew_mean:.4f}  "
                f"pg={np.mean(pg_losses):.4f}  "
                f"vf={np.mean(vf_losses):.4f}  "
                f"ent={-np.mean(ent_losses):.4f}  "
                f"sps={sps:.0f}"
            )

        if save_path and iteration % (log_interval * 10) == 0:
            torch.save(policy.state_dict(), save_path)
            print(f"  → checkpoint saved to {save_path}")

    if save_path:
        torch.save(policy.state_dict(), save_path)
        print(f"Final checkpoint saved to {save_path}")

    return policy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MuscleMimic PPO training")
    p.add_argument("--clip", type=Path, default=_default_clip())
    p.add_argument("--total_steps", type=int, default=10_000)
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--rollout_steps", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n_epochs", type=int, default=4)
    p.add_argument("--n_minibatches", type=int, default=4)
    p.add_argument("--entropy_coef", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save", type=str, default="mimic_policy.pt")
    p.add_argument("--log_interval", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    train(
        clip_path=args.clip,
        total_steps=args.total_steps,
        n_envs=args.n_envs,
        rollout_steps=args.rollout_steps,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        lr=args.lr,
        n_epochs=args.n_epochs,
        n_minibatches=args.n_minibatches,
        entropy_coef=args.entropy_coef,
        device=args.device,
        save_path=args.save,
        log_interval=args.log_interval,
    )
