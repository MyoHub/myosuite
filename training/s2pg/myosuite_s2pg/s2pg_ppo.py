# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
#
# Joint action-and-latent PPO update adapted from LocoMuJoCo
# (https://github.com/robfiras/loco-mujoco,
# loco_mujoco/algorithms/experimental/s2pg_ppo_jax.py), Copyright (c) 2024
# Al-Hafez, MIT License (see LICENSE-loco-mujoco.txt), based on:
# Firas Al-Hafez, Guoping Zhao, Jan Peters, and Davide Tateo,
# "Time-Efficient Reinforcement Learning with Stochastic Stateful Policies",
# ICLR 2024. Reimplemented in PyTorch for MyoSuite.
"""S2PG-PPO and matched feedforward PPO trainers for MyoSuite environments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from myosuite import make_env
from myosuite.core.config import EnvConfig
from torch import nn

from myosuite_s2pg.networks import (
    ActorCriticFeedforward,
    ActorCriticS2PG,
    RNNType,
)


class VectorEnvProtocol(Protocol):
    """Minimal tensor-based vectorized environment interface consumed by trainers."""

    num_envs: int
    obs_dim: int
    action_dim: int
    max_episode_steps: int
    device: torch.device

    def reset(self, seed: int | None = None) -> torch.Tensor:
        """Reset all environments and return initial observations ``(num_envs, obs_dim)``."""
        ...

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step all environments, auto-resetting done environments before returning ``obs``."""
        ...

    def close(self) -> None:
        """Release underlying environment resources."""
        ...


class CpuVectorEnv:
    """Deterministic CPU vectorized wrapper built via :func:`myosuite.make_env`."""

    def __init__(
        self,
        env_id: str,
        num_envs: int,
        seed: int = 0,
        features: tuple[Any, ...] = (),
        device: str | torch.device = "cpu",
    ) -> None:
        """Build ``num_envs`` independent MyoSuite CPU environments.

        Args:
            env_id: Registered MyoSuite environment identifier.
            num_envs: Number of parallel environment instances.
            seed: Base seed controlling all initial and auto-reset episode seeds.
            features: Optional ``EnvConfig.features`` specs (e.g. ``(FatigueWrapper,)``).
            device: Torch device for returned tensors.
        """
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")
        self.env_id = env_id
        self.num_envs = num_envs
        self.base_seed = int(seed)
        self.features = features
        self.device = torch.device(device)
        self._rng = np.random.default_rng(self.base_seed)

        env_cfg = EnvConfig(env_id=env_id, features=features)
        self.envs = [make_env(env_cfg) for _ in range(num_envs)]

        obs_space = self.envs[0].observation_space
        act_space = self.envs[0].action_space
        assert obs_space.shape is not None and act_space.shape is not None
        self.obs_dim = int(obs_space.shape[0])
        self.action_dim = int(act_space.shape[0])
        spec = getattr(self.envs[0], "spec", None)
        self.max_episode_steps = int(getattr(spec, "max_episode_steps", None) or 200)

        self._ep_returns = np.zeros(num_envs, dtype=np.float64)
        self._ep_lengths = np.zeros(num_envs, dtype=np.int64)

    def _next_seed(self) -> int:
        return int(self._rng.integers(0, 2**31 - 1))

    def reset(self, seed: int | None = None) -> torch.Tensor:
        """Reset all environments deterministically and return ``(num_envs, obs_dim)``."""
        if seed is not None:
            self.base_seed = int(seed)
            self._rng = np.random.default_rng(self.base_seed)
        obs_list = [env.reset(seed=self._next_seed())[0] for env in self.envs]
        self._ep_returns.fill(0.0)
        self._ep_lengths.fill(0)
        obs_np = np.asarray(obs_list, dtype=np.float32)
        return torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step all environments, auto-resetting completed episodes."""
        act_np = actions.detach().cpu().numpy().astype(np.float32, copy=False)
        next_obs = np.empty((self.num_envs, self.obs_dim), dtype=np.float32)
        rewards = np.empty(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        completed: list[dict[str, float]] = []

        for idx, env in enumerate(self.envs):
            obs_i, rew_i, term_i, trunc_i, info_i = env.step(act_np[idx])
            rew_f = float(rew_i)
            self._ep_returns[idx] += rew_f
            self._ep_lengths[idx] += 1
            rewards[idx] = rew_f
            terminated[idx] = bool(term_i)
            truncated[idx] = bool(trunc_i)

            if term_i or trunc_i:
                ep_record: dict[str, float] = {
                    "return": float(self._ep_returns[idx]),
                    "length": float(self._ep_lengths[idx]),
                }
                if "solved" in info_i:
                    ep_record["success"] = float(bool(info_i["solved"]))
                completed.append(ep_record)
                self._ep_returns[idx] = 0.0
                self._ep_lengths[idx] = 0
                obs_i, _ = env.reset(seed=self._next_seed())
            next_obs[idx] = obs_i

        return (
            torch.as_tensor(next_obs, dtype=torch.float32, device=self.device),
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(terminated, dtype=torch.bool, device=self.device),
            torch.as_tensor(truncated, dtype=torch.bool, device=self.device),
            {"completed_episodes": completed},
        )

    def close(self) -> None:
        """Close all underlying CPU environments."""
        for env in self.envs:
            env.close()


@dataclass
class S2PGConfig:
    """Hyperparameters shared between :class:`S2PGPPOTrainer` and :class:`FeedforwardPPOTrainer`."""

    hidden_state_dim: int = 64
    rnn_type: RNNType = "gru"
    ema_coef: float = 1.0
    pre_rnn_obs_dims: tuple[int, ...] = (128,)
    pre_rnn_act_dims: tuple[int, ...] = (32,)
    post_rnn_obs_dims: tuple[int, ...] = (64,)
    actor_hidden_dims: tuple[int, ...] = (128,)
    ff_actor_hidden_dims: tuple[int, ...] = (128, 128)
    critic_hidden_dims: tuple[int, ...] = (256, 256)
    vanilla_rnn_dims: tuple[int, ...] = (128,)
    activation: str = "tanh"
    init_std_a: float = 0.5
    init_std_z: float = 0.1
    normalize_obs: bool = True

    lr: float = 3e-4
    adam_eps: float = 1e-5
    anneal_lr: bool = True
    num_steps_per_env: int = 128
    num_epochs: int = 8
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    clip_actions: float | None = 1.0
    seed: int = 0


@dataclass
class S2PGRolloutBatch:
    """Single-step transition rollout buffer for S2PG-PPO without BPTT sequences."""

    obs: torch.Tensor
    hidden_states: torch.Tensor
    prev_actions: torch.Tensor
    actions: torch.Tensor
    next_hidden_states: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    completed_episodes: list[dict[str, float]]


@dataclass
class FeedforwardRolloutBatch:
    """Rollout buffer for the matched stateless PPO baseline."""

    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    completed_episodes: list[dict[str, float]]


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation bootstrapping across non-terminal steps.

    Args:
        rewards: Step rewards ``(T, N)``.
        values: Critic predictions ``(T, N)``.
        dones: Episode boundary flags ``terminated | truncated`` at step ``t`` of shape ``(T, N)``.
        last_value: Bootstrap value ``V(s_T)`` after step ``T - 1`` of shape ``(N,)``.
        gamma: Discount factor.
        gae_lambda: GAE exponential trace parameter.

    Returns:
        Tuple ``(advantages, returns)`` of shape ``(T, N)``.
    """
    horizon, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(num_envs, dtype=rewards.dtype, device=rewards.device)

    for step in range(horizon - 1, -1, -1):
        next_val = last_value if step == horizon - 1 else values[step + 1]
        non_terminal = 1.0 - dones[step].to(dtype=rewards.dtype)
        delta = rewards[step] + gamma * next_val * non_terminal - values[step]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[step] = gae

    returns = advantages + values
    return advantages, returns


def _explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    var_y = torch.var(y_true, unbiased=False)
    if float(var_y) < 1e-12:
        return 0.0
    return float(1.0 - torch.var(y_true - y_pred, unbiased=False) / var_y)


class S2PGPPOTrainer:
    """PPO trainer optimizing the joint action and stochastic hidden-state transition kernel."""

    def __init__(
        self,
        vec_env: VectorEnvProtocol,
        cfg: S2PGConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize the S2PG-PPO policy, optimizer, and per-env recurrent carry.

        Args:
            vec_env: Vectorized environment implementing :class:`VectorEnvProtocol`.
            cfg: Trainer and network configuration.
            device: Torch device (defaults to ``vec_env.device``).
        """
        self.vec_env = vec_env
        self.cfg = cfg or S2PGConfig()
        self.device = (
            torch.device(device) if device is not None else self.vec_env.device
        )

        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)

        self.policy = ActorCriticS2PG(
            obs_dim=vec_env.obs_dim,
            action_dim=vec_env.action_dim,
            hidden_state_dim=self.cfg.hidden_state_dim,
            rnn_type=self.cfg.rnn_type,
            ema_coef=self.cfg.ema_coef,
            pre_rnn_obs_dims=self.cfg.pre_rnn_obs_dims,
            pre_rnn_act_dims=self.cfg.pre_rnn_act_dims,
            post_rnn_obs_dims=self.cfg.post_rnn_obs_dims,
            actor_hidden_dims=self.cfg.actor_hidden_dims,
            critic_hidden_dims=self.cfg.critic_hidden_dims,
            vanilla_rnn_dims=self.cfg.vanilla_rnn_dims,
            activation=self.cfg.activation,
            init_std_a=self.cfg.init_std_a,
            init_std_z=self.cfg.init_std_z,
            normalize_obs=self.cfg.normalize_obs,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.cfg.lr, eps=self.cfg.adam_eps
        )
        self._perm_gen = torch.Generator(device="cpu")
        self._perm_gen.manual_seed(self.cfg.seed + 10_007)

        self._obs: torch.Tensor | None = None
        self._hidden_state: torch.Tensor | None = None
        self._prev_action: torch.Tensor | None = None
        self.env_steps = 0

    @property
    def current_hidden_state(self) -> torch.Tensor | None:
        """Return the live per-env hidden state carry ``z_t``."""
        return self._hidden_state

    @property
    def current_prev_action(self) -> torch.Tensor | None:
        """Return the live per-env previous action ``a_{t-1}``."""
        return self._prev_action

    def reset_rollout_state(self, seed: int | None = None) -> None:
        """Reset the vectorized environment and zero ``(hidden_state, prev_action)``."""
        reset_seed = self.cfg.seed if seed is None else seed
        self._obs = self.vec_env.reset(seed=reset_seed).to(
            device=self.device, dtype=torch.float32
        )
        self._hidden_state, self._prev_action = self.policy.initial_state(
            self.vec_env.num_envs, device=self.device
        )

    def collect_rollout(self) -> S2PGRolloutBatch:
        """Collect ``num_steps_per_env`` transitions per env and compute GAE targets."""
        if self._obs is None or self._hidden_state is None or self._prev_action is None:
            self.reset_rollout_state()
        assert (
            self._obs is not None
            and self._hidden_state is not None
            and self._prev_action is not None
        )

        horizon = self.cfg.num_steps_per_env
        num_envs = self.vec_env.num_envs
        obs_dim = self.vec_env.obs_dim
        act_dim = self.vec_env.action_dim
        lat_dim = self.policy.latent_dim

        buf_obs = torch.empty(
            (horizon, num_envs, obs_dim), dtype=torch.float32, device=self.device
        )
        buf_hidden = torch.empty(
            (horizon, num_envs, lat_dim), dtype=torch.float32, device=self.device
        )
        buf_prev_act = torch.empty(
            (horizon, num_envs, act_dim), dtype=torch.float32, device=self.device
        )
        buf_act = torch.empty(
            (horizon, num_envs, act_dim), dtype=torch.float32, device=self.device
        )
        buf_next_hidden = torch.empty(
            (horizon, num_envs, lat_dim), dtype=torch.float32, device=self.device
        )
        buf_log_probs = torch.empty(
            (horizon, num_envs), dtype=torch.float32, device=self.device
        )
        buf_rewards = torch.empty(
            (horizon, num_envs), dtype=torch.float32, device=self.device
        )
        buf_dones = torch.empty(
            (horizon, num_envs), dtype=torch.bool, device=self.device
        )
        buf_values = torch.empty(
            (horizon, num_envs), dtype=torch.float32, device=self.device
        )
        completed_episodes: list[dict[str, float]] = []

        obs = self._obs
        hidden_state = self._hidden_state
        prev_action = self._prev_action

        self.policy.eval()
        with torch.no_grad():
            for step in range(horizon):
                buf_obs[step] = obs
                buf_hidden[step] = hidden_state
                buf_prev_act[step] = prev_action

                dist, value = self.policy.get_dist_and_value(
                    obs, hidden_state, prev_action
                )
                joint_az = dist.sample()
                log_prob = dist.log_prob(joint_az)
                action, next_hidden = dist.split(joint_az)

                buf_act[step] = action
                buf_next_hidden[step] = next_hidden
                buf_log_probs[step] = log_prob
                buf_values[step] = value

                env_action = (
                    torch.clamp(action, -self.cfg.clip_actions, self.cfg.clip_actions)
                    if self.cfg.clip_actions is not None
                    else action
                )
                next_obs, reward, terminated, truncated, info = self.vec_env.step(
                    env_action
                )
                done = terminated | truncated

                buf_rewards[step] = reward
                buf_dones[step] = done
                if "completed_episodes" in info:
                    completed_episodes.extend(info["completed_episodes"])

                not_done = (~done).unsqueeze(-1).to(dtype=torch.float32)
                hidden_state = next_hidden * not_done
                prev_action = action * not_done
                obs = next_obs.to(device=self.device, dtype=torch.float32)

            last_value = self.policy.get_value(obs, hidden_state, prev_action)

        self._obs = obs
        self._hidden_state = hidden_state
        self._prev_action = prev_action
        self.env_steps += horizon * num_envs

        advantages, returns = compute_gae(
            rewards=buf_rewards,
            values=buf_values,
            dones=buf_dones,
            last_value=last_value,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )
        return S2PGRolloutBatch(
            obs=buf_obs,
            hidden_states=buf_hidden,
            prev_actions=buf_prev_act,
            actions=buf_act,
            next_hidden_states=buf_next_hidden,
            log_probs=buf_log_probs,
            rewards=buf_rewards,
            dones=buf_dones,
            values=buf_values,
            advantages=advantages,
            returns=returns,
            completed_episodes=completed_episodes,
        )

    def update(
        self, batch: S2PGRolloutBatch, progress: float = 0.0
    ) -> dict[str, float]:
        """Run flat shuffled minibatch PPO epochs over the joint ``(a_t, z_{t+1})`` kernel.

        Args:
            batch: Collected :class:`S2PGRolloutBatch`.
            progress: Training fraction in ``[0, 1]`` used when ``anneal_lr=True``.

        Returns:
            Dictionary of averaged scalar optimization diagnostics.
        """
        self.policy.train()
        if self.cfg.anneal_lr:
            lr_now = self.cfg.lr * max(0.0, 1.0 - float(progress))
            for group in self.optimizer.param_groups:
                group["lr"] = lr_now

        batch_size = batch.obs.shape[0] * batch.obs.shape[1]
        num_mb = max(1, min(self.cfg.num_minibatches, batch_size))
        mb_size = batch_size // num_mb

        flat_obs = batch.obs.reshape(batch_size, -1)
        flat_hidden = batch.hidden_states.reshape(batch_size, -1)
        flat_prev_act = batch.prev_actions.reshape(batch_size, -1)
        flat_actions = batch.actions.reshape(batch_size, -1)
        flat_next_hidden = batch.next_hidden_states.reshape(batch_size, -1)
        flat_joint_az = torch.cat([flat_actions, flat_next_hidden], dim=-1)
        flat_old_logp = batch.log_probs.reshape(batch_size)
        flat_adv = batch.advantages.reshape(batch_size)
        flat_returns = batch.returns.reshape(batch_size)
        flat_old_val = batch.values.reshape(batch_size)

        pg_losses: list[float] = []
        v_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        clip_fracs: list[float] = []
        grad_norms: list[float] = []

        for _ in range(self.cfg.num_epochs):
            perm = torch.randperm(batch_size, generator=self._perm_gen).to(self.device)
            for start in range(0, num_mb * mb_size, mb_size):
                mb_idx = perm[start : start + mb_size]

                dist, new_val = self.policy.get_dist_and_value(
                    flat_obs[mb_idx],
                    flat_hidden[mb_idx],
                    flat_prev_act[mb_idx],
                )
                new_logp = dist.log_prob(flat_joint_az[mb_idx])
                entropy = dist.entropy().mean()

                log_ratio = new_logp - flat_old_logp[mb_idx]
                ratio = torch.exp(log_ratio)

                mb_adv = flat_adv[mb_idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(unbiased=False) + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef
                )
                pg_loss = torch.maximum(pg_loss1, pg_loss2).mean()

                mb_ret = flat_returns[mb_idx]
                if self.cfg.clip_vloss:
                    mb_old_v = flat_old_val[mb_idx]
                    v_clipped = mb_old_v + torch.clamp(
                        new_val - mb_old_v, -self.cfg.clip_coef, self.cfg.clip_coef
                    )
                    v_loss = (
                        0.5
                        * torch.maximum(
                            (new_val - mb_ret).square(),
                            (v_clipped - mb_ret).square(),
                        ).mean()
                    )
                else:
                    v_loss = 0.5 * (new_val - mb_ret).square().mean()

                loss = pg_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = float(((ratio - 1.0) - log_ratio).mean())
                    clip_frac = float(
                        ((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean()
                    )

                pg_losses.append(float(pg_loss.detach()))
                v_losses.append(float(v_loss.detach()))
                entropies.append(float(entropy.detach()))
                approx_kls.append(approx_kl)
                clip_fracs.append(clip_frac)
                grad_norms.append(float(grad_norm))

        if self.policy.normalize_obs:
            self.policy.obs_rms.update(batch.obs)

        return {
            "policy_loss": float(np.mean(pg_losses)),
            "value_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_fraction": float(np.mean(clip_fracs)),
            "explained_variance": _explained_variance(flat_old_val, flat_returns),
            "grad_norm": float(np.mean(grad_norms)),
        }

    def train_step(
        self, progress: float = 0.0
    ) -> tuple[S2PGRolloutBatch, dict[str, float]]:
        """Collect one rollout and perform one S2PG-PPO update."""
        batch = self.collect_rollout()
        metrics = self.update(batch, progress=progress)
        return batch, metrics

    def save_checkpoint(
        self, path: Path | str, extra: dict[str, Any] | None = None
    ) -> Path:
        """Save policy, optimizer, config, and step count to ``path``."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "algorithm": "s2pg_ppo",
            "config": asdict(self.cfg),
            "obs_dim": self.policy.obs_dim,
            "action_dim": self.policy.action_dim,
            "env_steps": self.env_steps,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, str(out_path))
        return out_path

    def load_checkpoint(self, path: Path | str) -> dict[str, Any]:
        """Load policy and optimizer states from ``path``."""
        ckpt = torch.load(str(Path(path)), map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.env_steps = int(ckpt.get("env_steps", 0))
        return ckpt


class FeedforwardPPOTrainer:
    """Stateless feedforward PPO trainer sharing identical hyperparameters with :class:`S2PGPPOTrainer`."""

    def __init__(
        self,
        vec_env: VectorEnvProtocol,
        cfg: S2PGConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize the matched feedforward PPO baseline trainer."""
        self.vec_env = vec_env
        self.cfg = cfg or S2PGConfig()
        self.device = (
            torch.device(device) if device is not None else self.vec_env.device
        )

        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)

        self.policy = ActorCriticFeedforward(
            obs_dim=vec_env.obs_dim,
            action_dim=vec_env.action_dim,
            actor_hidden_dims=self.cfg.ff_actor_hidden_dims,
            critic_hidden_dims=self.cfg.critic_hidden_dims,
            activation=self.cfg.activation,
            init_std_a=self.cfg.init_std_a,
            normalize_obs=self.cfg.normalize_obs,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.cfg.lr, eps=self.cfg.adam_eps
        )
        self._perm_gen = torch.Generator(device="cpu")
        self._perm_gen.manual_seed(self.cfg.seed + 10_007)

        self._obs: torch.Tensor | None = None
        self.env_steps = 0

    def reset_rollout_state(self, seed: int | None = None) -> None:
        """Reset the vectorized environment."""
        reset_seed = self.cfg.seed if seed is None else seed
        self._obs = self.vec_env.reset(seed=reset_seed).to(
            device=self.device, dtype=torch.float32
        )

    def collect_rollout(self) -> FeedforwardRolloutBatch:
        """Collect ``num_steps_per_env`` stateless transitions and compute GAE targets."""
        if self._obs is None:
            self.reset_rollout_state()
        assert self._obs is not None

        horizon = self.cfg.num_steps_per_env
        num_envs = self.vec_env.num_envs
        obs_dim = self.vec_env.obs_dim
        act_dim = self.vec_env.action_dim

        buf_obs = torch.empty(
            (horizon, num_envs, obs_dim), dtype=torch.float32, device=self.device
        )
        buf_act = torch.empty(
            (horizon, num_envs, act_dim), dtype=torch.float32, device=self.device
        )
        buf_log_probs = torch.empty(
            (horizon, num_envs), dtype=torch.float32, device=self.device
        )
        buf_rewards = torch.empty(
            (horizon, num_envs), dtype=torch.float32, device=self.device
        )
        buf_dones = torch.empty(
            (horizon, num_envs), dtype=torch.bool, device=self.device
        )
        buf_values = torch.empty(
            (horizon, num_envs), dtype=torch.float32, device=self.device
        )
        completed_episodes: list[dict[str, float]] = []

        obs = self._obs
        self.policy.eval()
        with torch.no_grad():
            for step in range(horizon):
                buf_obs[step] = obs
                dist, value = self.policy.get_dist_and_value(obs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                buf_act[step] = action
                buf_log_probs[step] = log_prob
                buf_values[step] = value

                env_action = (
                    torch.clamp(action, -self.cfg.clip_actions, self.cfg.clip_actions)
                    if self.cfg.clip_actions is not None
                    else action
                )
                next_obs, reward, terminated, truncated, info = self.vec_env.step(
                    env_action
                )
                buf_rewards[step] = reward
                buf_dones[step] = terminated | truncated
                if "completed_episodes" in info:
                    completed_episodes.extend(info["completed_episodes"])
                obs = next_obs.to(device=self.device, dtype=torch.float32)

            last_value = self.policy.get_value(obs)

        self._obs = obs
        self.env_steps += horizon * num_envs

        advantages, returns = compute_gae(
            rewards=buf_rewards,
            values=buf_values,
            dones=buf_dones,
            last_value=last_value,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )
        return FeedforwardRolloutBatch(
            obs=buf_obs,
            actions=buf_act,
            log_probs=buf_log_probs,
            rewards=buf_rewards,
            dones=buf_dones,
            values=buf_values,
            advantages=advantages,
            returns=returns,
            completed_episodes=completed_episodes,
        )

    def update(
        self, batch: FeedforwardRolloutBatch, progress: float = 0.0
    ) -> dict[str, float]:
        """Run PPO epochs with settings identical to :class:`S2PGPPOTrainer`."""
        self.policy.train()
        if self.cfg.anneal_lr:
            lr_now = self.cfg.lr * max(0.0, 1.0 - float(progress))
            for group in self.optimizer.param_groups:
                group["lr"] = lr_now

        batch_size = batch.obs.shape[0] * batch.obs.shape[1]
        num_mb = max(1, min(self.cfg.num_minibatches, batch_size))
        mb_size = batch_size // num_mb

        flat_obs = batch.obs.reshape(batch_size, -1)
        flat_actions = batch.actions.reshape(batch_size, -1)
        flat_old_logp = batch.log_probs.reshape(batch_size)
        flat_adv = batch.advantages.reshape(batch_size)
        flat_returns = batch.returns.reshape(batch_size)
        flat_old_val = batch.values.reshape(batch_size)

        pg_losses: list[float] = []
        v_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        clip_fracs: list[float] = []
        grad_norms: list[float] = []

        for _ in range(self.cfg.num_epochs):
            perm = torch.randperm(batch_size, generator=self._perm_gen).to(self.device)
            for start in range(0, num_mb * mb_size, mb_size):
                mb_idx = perm[start : start + mb_size]

                dist, new_val = self.policy.get_dist_and_value(flat_obs[mb_idx])
                new_logp = dist.log_prob(flat_actions[mb_idx])
                entropy = dist.entropy().mean()

                log_ratio = new_logp - flat_old_logp[mb_idx]
                ratio = torch.exp(log_ratio)

                mb_adv = flat_adv[mb_idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(unbiased=False) + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef
                )
                pg_loss = torch.maximum(pg_loss1, pg_loss2).mean()

                mb_ret = flat_returns[mb_idx]
                if self.cfg.clip_vloss:
                    mb_old_v = flat_old_val[mb_idx]
                    v_clipped = mb_old_v + torch.clamp(
                        new_val - mb_old_v, -self.cfg.clip_coef, self.cfg.clip_coef
                    )
                    v_loss = (
                        0.5
                        * torch.maximum(
                            (new_val - mb_ret).square(),
                            (v_clipped - mb_ret).square(),
                        ).mean()
                    )
                else:
                    v_loss = 0.5 * (new_val - mb_ret).square().mean()

                loss = pg_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = float(((ratio - 1.0) - log_ratio).mean())
                    clip_frac = float(
                        ((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean()
                    )

                pg_losses.append(float(pg_loss.detach()))
                v_losses.append(float(v_loss.detach()))
                entropies.append(float(entropy.detach()))
                approx_kls.append(approx_kl)
                clip_fracs.append(clip_frac)
                grad_norms.append(float(grad_norm))

        if self.policy.normalize_obs:
            self.policy.obs_rms.update(batch.obs)

        return {
            "policy_loss": float(np.mean(pg_losses)),
            "value_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_fraction": float(np.mean(clip_fracs)),
            "explained_variance": _explained_variance(flat_old_val, flat_returns),
            "grad_norm": float(np.mean(grad_norms)),
        }

    def train_step(
        self, progress: float = 0.0
    ) -> tuple[FeedforwardRolloutBatch, dict[str, float]]:
        """Collect one rollout and perform one stateless PPO update."""
        batch = self.collect_rollout()
        metrics = self.update(batch, progress=progress)
        return batch, metrics


def baseline_feedforward_ppo(
    vec_env: VectorEnvProtocol,
    cfg: S2PGConfig | None = None,
    device: str | torch.device | None = None,
) -> FeedforwardPPOTrainer:
    """Construct the apple-to-apple :class:`FeedforwardPPOTrainer` baseline."""
    return FeedforwardPPOTrainer(vec_env=vec_env, cfg=cfg, device=device)
