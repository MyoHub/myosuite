# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""PPO with Adversarial Motion Priors (AMP), GAIL, Behaviour Cloning, and export utilities."""

from __future__ import annotations

import copy
import importlib.util
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from myosuite import make_env
from myosuite_amp.discriminator import (
    DiscriminatorMode,
    MotionDiscriminator,
    RunningMeanStd,
)


@dataclass
class VecStepBatch:
    """Batched single-step transition output from a vectorized environment."""

    obs: torch.Tensor
    transition_next_obs: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    valid_transition: torch.Tensor


class VecEnvBackend(Protocol):
    """Protocol implemented by CPU and mjlab GPU vectorized environment adapters."""

    num_envs: int
    obs_dim: int
    act_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    device: torch.device

    def reset(self, seed: int | None = None) -> torch.Tensor:
        """Reset all environments and return initial observations ``(num_envs, obs_dim)``."""
        ...

    def step(self, actions: torch.Tensor) -> VecStepBatch:
        """Step all environments with ``actions`` ``(num_envs, act_dim)``."""
        ...

    def close(self) -> None:
        """Release environment resources."""
        ...


class CpuVecEnv:
    """Synchronous CPU vectorized wrapper around ``myosuite.make_env``.

    Captures the true pre-reset terminal observation ``transition_next_obs`` on
    every episode boundary before auto-resetting individual environments, avoiding
    both Gymnasium 1.x next-step dummy transitions and cross-reset state teleports.
    """

    def __init__(
        self, env_id: str, num_envs: int = 8, device: str | torch.device = "cpu"
    ) -> None:
        """Create ``num_envs`` instances of ``env_id``.

        Args:
            env_id: Registered MyoSuite environment identifier.
            num_envs: Number of parallel CPU environments.
            device: Torch device for returned tensors.
        """
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")
        self.env_id = env_id
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self._envs = [make_env(env_id) for _ in range(self.num_envs)]
        obs_space = self._envs[0].observation_space
        act_space = self._envs[0].action_space
        assert obs_space.shape is not None and act_space.shape is not None
        self.obs_dim = int(obs_space.shape[0])
        self.act_dim = int(act_space.shape[0])
        self.action_low = np.asarray(act_space.low, dtype=np.float32)
        self.action_high = np.asarray(act_space.high, dtype=np.float32)
        self._obs_np = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        self._next_obs_np = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        self._rew_np = np.zeros(self.num_envs, dtype=np.float32)
        self._term_np = np.zeros(self.num_envs, dtype=bool)
        self._trunc_np = np.zeros(self.num_envs, dtype=bool)
        self._reset_counter = 0
        self._base_seed = 0

    def reset(self, seed: int | None = None) -> torch.Tensor:
        """Reset all sub-environments with deterministic per-env seeds."""
        if seed is not None:
            self._base_seed = int(seed)
            self._reset_counter = 0
        for i, env in enumerate(self._envs):
            ep_seed = self._base_seed + i * 100_003 + self._reset_counter
            obs_i, _ = env.reset(seed=ep_seed)
            self._obs_np[i] = np.asarray(obs_i, dtype=np.float32)
        self._reset_counter += 1
        return torch.as_tensor(
            self._obs_np.copy(), dtype=torch.float32, device=self.device
        )

    def step(self, actions: torch.Tensor) -> VecStepBatch:
        """Step all sub-environments and auto-reset finished episodes."""
        act_np = np.clip(
            actions.detach().cpu().numpy().astype(np.float32, copy=False),
            self.action_low,
            self.action_high,
        )
        for i, env in enumerate(self._envs):
            next_obs, rew, term, trunc, _ = env.step(act_np[i])
            next_obs_arr = np.asarray(next_obs, dtype=np.float32)
            self._next_obs_np[i] = next_obs_arr
            self._rew_np[i] = float(rew)
            self._term_np[i] = bool(term)
            self._trunc_np[i] = bool(trunc)
            if term or trunc:
                ep_seed = self._base_seed + i * 100_003 + self._reset_counter
                self._reset_counter += 1
                reset_obs, _ = env.reset(seed=ep_seed)
                self._obs_np[i] = np.asarray(reset_obs, dtype=np.float32)
            else:
                self._obs_np[i] = next_obs_arr

        return VecStepBatch(
            obs=torch.as_tensor(
                self._obs_np.copy(), dtype=torch.float32, device=self.device
            ),
            transition_next_obs=torch.as_tensor(
                self._next_obs_np.copy(), dtype=torch.float32, device=self.device
            ),
            rewards=torch.as_tensor(
                self._rew_np.copy(), dtype=torch.float32, device=self.device
            ),
            terminated=torch.as_tensor(
                self._term_np.copy(), dtype=torch.bool, device=self.device
            ),
            truncated=torch.as_tensor(
                self._trunc_np.copy(), dtype=torch.bool, device=self.device
            ),
            valid_transition=torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            ),
        )

    def close(self) -> None:
        """Close all underlying environments."""
        for env in self._envs:
            env.close()


def _build_mlp(in_dim: int, hidden_dims: tuple[int, ...]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    cur = in_dim
    for width in hidden_dims:
        linear = nn.Linear(cur, int(width))
        nn.init.orthogonal_(linear.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        layers.append(nn.ELU())
        cur = int(width)
    return nn.Sequential(*layers), cur


class ActorCritic(nn.Module):
    """Gaussian policy and value network with running observation normalization."""

    action_low: torch.Tensor
    action_high: torch.Tensor

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        actor_hidden_dims: tuple[int, ...] = (128, 128),
        critic_hidden_dims: tuple[int, ...] = (128, 128),
        init_std: float = 0.5,
        std_range: tuple[float, float] = (0.05, 1.0),
        normalize_obs: bool = True,
        action_low: np.ndarray | float = -1.0,
        action_high: np.ndarray | float = 1.0,
    ) -> None:
        """Initialize actor-critic networks and observation normalizer.

        Args:
            obs_dim: Observation vector dimension.
            act_dim: Continuous action vector dimension.
            actor_hidden_dims: Hidden widths of the policy MLP.
            critic_hidden_dims: Hidden widths of the value MLP.
            init_std: Initial Gaussian exploration standard deviation.
            std_range: ``(min_std, max_std)`` bounds applied to policy std.
            normalize_obs: Standardize observations with ``RunningMeanStd``.
            action_low: Lower bound of environment actions.
            action_high: Upper bound of environment actions.
        """
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.normalize_obs_enabled = bool(normalize_obs)
        self.min_log_std = math.log(float(std_range[0]))
        self.max_log_std = math.log(float(std_range[1]))

        self.obs_rms = RunningMeanStd(self.obs_dim)
        low_t = torch.broadcast_to(
            torch.as_tensor(action_low, dtype=torch.float32), (self.act_dim,)
        ).clone()
        high_t = torch.broadcast_to(
            torch.as_tensor(action_high, dtype=torch.float32), (self.act_dim,)
        ).clone()
        self.register_buffer("action_low", low_t)
        self.register_buffer("action_high", high_t)

        self.actor_trunk, actor_last = _build_mlp(self.obs_dim, actor_hidden_dims)
        self.actor_mean = nn.Linear(actor_last, self.act_dim)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)

        self.critic_trunk, critic_last = _build_mlp(self.obs_dim, critic_hidden_dims)
        self.critic_head = nn.Linear(critic_last, 1)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

        clamped_init = min(
            max(float(init_std), float(std_range[0])), float(std_range[1])
        )
        self.log_std = nn.Parameter(torch.full((self.act_dim,), math.log(clamped_init)))

    def _norm(self, obs: torch.Tensor) -> torch.Tensor:
        if self.normalize_obs_enabled:
            return self.obs_rms.normalize(obs, clip=10.0)
        return obs

    def _distribution(self, norm_obs: torch.Tensor) -> Normal:
        mean = self.actor_mean(self.actor_trunk(norm_obs))
        std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std).exp()
        return Normal(mean, std.expand_as(mean))

    def act_and_value(
        self, obs: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample stochastic policy actions and evaluate value estimates.

        Args:
            obs: Raw observations ``(B, obs_dim)``.
            generator: Optional seeded ``torch.Generator`` for reproducible sampling.

        Returns:
            Tuple ``(raw_action, clipped_env_action, log_prob, value)``.
        """
        norm_obs = self._norm(obs)
        dist = self._distribution(norm_obs)
        if generator is None:
            noise = torch.randn_like(dist.loc)
        else:
            noise = torch.randn(
                dist.loc.shape,
                dtype=dist.loc.dtype,
                device="cpu",
                generator=generator,
            ).to(dist.loc.device)
        raw_action = dist.loc + dist.scale * noise
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        env_action = torch.clamp(raw_action, self.action_low, self.action_high)
        value = self.critic_head(self.critic_trunk(norm_obs)).squeeze(-1)
        return raw_action, env_action, log_prob, value

    def evaluate_actions(
        self, obs: torch.Tensor, raw_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log-probabilities, entropy, and state values for PPO updates.

        Args:
            obs: Raw observations ``(B, obs_dim)``.
            raw_actions: Unclipped Gaussian actions ``(B, act_dim)`` from rollout.

        Returns:
            Tuple ``(log_prob, entropy, value)``.
        """
        norm_obs = self._norm(obs)
        dist = self._distribution(norm_obs)
        log_prob = dist.log_prob(raw_actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic_head(self.critic_trunk(norm_obs)).squeeze(-1)
        return log_prob, entropy, value

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """Evaluate critic ``V(s)`` on raw observations."""
        norm_obs = self._norm(obs)
        return self.critic_head(self.critic_trunk(norm_obs)).squeeze(-1)

    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute deterministic mean action clamped to ``[action_low, action_high]``."""
        norm_obs = self._norm(obs)
        mean = self.actor_mean(self.actor_trunk(norm_obs))
        return torch.clamp(mean, self.action_low, self.action_high)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic inference forward pass ``obs -> action``."""
        return self.act_deterministic(obs)

    @torch.no_grad()
    def predict_np(self, obs: np.ndarray) -> np.ndarray:
        """Run deterministic policy inference on a single or batched NumPy observation."""
        obs_arr = np.asarray(obs, dtype=np.float32)
        single = obs_arr.ndim == 1
        batch = obs_arr[None, :] if single else obs_arr
        tensor = torch.as_tensor(
            batch, dtype=torch.float32, device=self.action_low.device
        )
        act = self.act_deterministic(tensor).cpu().numpy()
        return act[0] if single else act


@dataclass(frozen=True)
class ExpertDataset:
    """Demonstration dataset holding transitions and optional expert actions."""

    obs: np.ndarray
    next_obs: np.ndarray
    actions: np.ndarray | None
    rewards: np.ndarray
    dones: np.ndarray
    env_id: str = ""
    teacher_mean_return: float = 0.0
    teacher_solved_rate: float = 0.0
    teacher_train_steps: int = 0
    generation_wall_time_s: float = 0.0

    @property
    def num_transitions(self) -> int:
        """Number of transitions ``(s_t, s_{t+1})`` stored in the dataset."""
        return int(self.obs.shape[0])

    def without_actions(self) -> ExpertDataset:
        """Return a view with ``actions=None`` for state-only motion priors (AMP/GAIL)."""
        return ExpertDataset(
            obs=self.obs,
            next_obs=self.next_obs,
            actions=None,
            rewards=self.rewards,
            dones=self.dones,
            env_id=self.env_id,
            teacher_mean_return=self.teacher_mean_return,
            teacher_solved_rate=self.teacher_solved_rate,
            teacher_train_steps=self.teacher_train_steps,
            generation_wall_time_s=self.generation_wall_time_s,
        )

    def save_npz(self, path: str | Path) -> Path:
        """Persist dataset arrays to a compressed ``.npz`` archive."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "obs": np.asarray(self.obs, dtype=np.float32),
            "next_obs": np.asarray(self.next_obs, dtype=np.float32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=bool),
            "env_id": np.asarray(self.env_id),
        }
        if self.actions is not None:
            payload["actions"] = np.asarray(self.actions, dtype=np.float32)
        np.savez_compressed(out, **payload)
        return out

    @staticmethod
    def load_npz(path: str | Path) -> ExpertDataset:
        """Load an :class:`ExpertDataset` from an ``.npz`` archive."""
        with np.load(Path(path), allow_pickle=False) as data:
            obs = np.asarray(data["obs"], dtype=np.float32)
            next_obs = np.asarray(data["next_obs"], dtype=np.float32)
            actions = (
                np.asarray(data["actions"], dtype=np.float32)
                if "actions" in data.files
                else None
            )
            rewards = (
                np.asarray(data["rewards"], dtype=np.float32)
                if "rewards" in data.files
                else np.zeros(obs.shape[0], dtype=np.float32)
            )
            dones = (
                np.asarray(data["dones"], dtype=bool)
                if "dones" in data.files
                else np.zeros(obs.shape[0], dtype=bool)
            )
            env_id = str(data["env_id"].item()) if "env_id" in data.files else ""
        return ExpertDataset(
            obs=obs,
            next_obs=next_obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            env_id=env_id,
        )


class TransitionReplayBuffer:
    """FIFO state-transition replay buffer stabilizing AMP/GAIL discriminator updates."""

    def __init__(self, capacity: int, obs_dim: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.device = device
        self.obs = torch.zeros(
            (self.capacity, self.obs_dim), dtype=torch.float32, device=device
        )
        self.next_obs = torch.zeros(
            (self.capacity, self.obs_dim), dtype=torch.float32, device=device
        )
        self.size = 0
        self.ptr = 0

    @torch.no_grad()
    def add(self, obs: torch.Tensor, next_obs: torch.Tensor) -> None:
        """Insert a batch of policy transitions ``(obs, next_obs)``."""
        n = int(obs.shape[0])
        if n == 0:
            return
        if n >= self.capacity:
            self.obs.copy_(obs[-self.capacity :].to(self.device))
            self.next_obs.copy_(next_obs[-self.capacity :].to(self.device))
            self.size = self.capacity
            self.ptr = 0
            return
        end = self.ptr + n
        if end <= self.capacity:
            self.obs[self.ptr : end].copy_(obs.to(self.device))
            self.next_obs[self.ptr : end].copy_(next_obs.to(self.device))
        else:
            first = self.capacity - self.ptr
            self.obs[self.ptr :].copy_(obs[:first].to(self.device))
            self.next_obs[self.ptr :].copy_(next_obs[:first].to(self.device))
            rem = n - first
            self.obs[:rem].copy_(obs[first:].to(self.device))
            self.next_obs[:rem].copy_(next_obs[first:].to(self.device))
        self.ptr = end % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(
        self, batch_size: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniformly sample ``batch_size`` transitions using ``generator``."""
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty TransitionReplayBuffer.")
        idx = torch.randint(
            0, self.size, (batch_size,), generator=generator, device="cpu"
        ).to(self.device)
        return self.obs[idx], self.next_obs[idx]


@dataclass
class AMPConfig:
    """Hyperparameters for PPO, GAIL, and AMP training."""

    mode: DiscriminatorMode = "amp"
    use_discriminator: bool = True
    proportion_env_reward: float = 0.5
    feature_indices: tuple[int, ...] | None = None
    actor_hidden_dims: tuple[int, ...] = (128, 128)
    critic_hidden_dims: tuple[int, ...] = (128, 128)
    disc_hidden_dims: tuple[int, ...] = (256, 256)
    init_std: float = 0.5
    std_range: tuple[float, float] = (0.05, 1.0)
    normalize_obs: bool = True
    rollout_steps: int = 128
    ppo_epochs: int = 6
    num_minibatches: int = 4
    learning_rate: float = 3e-4
    disc_learning_rate: float = 3e-4
    disc_updates_per_iter: int = 4
    disc_batch_size: int = 256
    disc_grad_penalty_weight: float = 5.0
    disc_logit_reg_weight: float = 0.05
    disc_replay_capacity: int = 50_000
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    max_grad_norm: float = 1.0
    seed: int = 0
    device: str = "cpu"


@dataclass(frozen=True)
class TrainStepStats:
    """Metrics recorded after one rollout + PPO/discriminator update iteration."""

    env_steps: int
    mean_env_reward: float
    mean_style_reward: float
    mean_blended_reward: float
    policy_loss: float
    value_loss: float
    entropy: float
    disc_loss: float
    disc_grad_penalty: float


@dataclass(frozen=True)
class EvalStats:
    """Deterministic policy evaluation summary on unmodified task rewards."""

    mean_return: float
    std_return: float
    solved_rate: float
    mean_length: float
    n_episodes: int


class AMPTrainer:
    """Clipped PPO trainer with optional AMP or GAIL transition discriminator."""

    def __init__(
        self,
        vec_env: VecEnvBackend,
        cfg: AMPConfig,
        expert_transitions: (
            ExpertDataset
            | tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]
            | None
        ) = None,
    ) -> None:
        """Initialize policy, optional discriminator, buffers, and optimizers.

        Args:
            vec_env: Vectorized CPU or mjlab GPU environment backend.
            cfg: Training configuration.
            expert_transitions: State-only expert transitions ``(obs, next_obs)`` or
                :class:`ExpertDataset` (actions, if present, are never accessed).
        """
        self.vec_env = vec_env
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        torch.manual_seed(cfg.seed)
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(cfg.seed + 17)

        self.actor_critic = ActorCritic(
            obs_dim=vec_env.obs_dim,
            act_dim=vec_env.act_dim,
            actor_hidden_dims=cfg.actor_hidden_dims,
            critic_hidden_dims=cfg.critic_hidden_dims,
            init_std=cfg.init_std,
            std_range=cfg.std_range,
            normalize_obs=cfg.normalize_obs,
            action_low=vec_env.action_low,
            action_high=vec_env.action_high,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=cfg.learning_rate
        )

        self.discriminator: MotionDiscriminator | None = None
        self.disc_optimizer: torch.optim.Optimizer | None = None
        self.expert_obs: torch.Tensor | None = None
        self.expert_next_obs: torch.Tensor | None = None
        self.replay_buffer: TransitionReplayBuffer | None = None

        if cfg.use_discriminator:
            if expert_transitions is None:
                raise ValueError(
                    "expert_transitions must be provided when cfg.use_discriminator=True."
                )
            if isinstance(expert_transitions, ExpertDataset):
                exp_s = torch.as_tensor(
                    expert_transitions.obs, dtype=torch.float32, device=self.device
                )
                exp_ns = torch.as_tensor(
                    expert_transitions.next_obs, dtype=torch.float32, device=self.device
                )
            else:
                exp_s = torch.as_tensor(
                    expert_transitions[0], dtype=torch.float32, device=self.device
                )
                exp_ns = torch.as_tensor(
                    expert_transitions[1], dtype=torch.float32, device=self.device
                )
            if (
                exp_s.ndim != 2
                or exp_s.shape != exp_ns.shape
                or exp_s.shape[1] != vec_env.obs_dim
            ):
                raise ValueError(
                    f"Expected expert transitions of shape (M, {vec_env.obs_dim}), "
                    f"got {tuple(exp_s.shape)} and {tuple(exp_ns.shape)}."
                )
            self.expert_obs = exp_s
            self.expert_next_obs = exp_ns
            self.discriminator = MotionDiscriminator(
                obs_dim=vec_env.obs_dim,
                mode=cfg.mode,
                feature_indices=cfg.feature_indices,
                hidden_dims=cfg.disc_hidden_dims,
                grad_penalty_weight=cfg.disc_grad_penalty_weight,
                logit_reg_weight=cfg.disc_logit_reg_weight,
            ).to(self.device)
            self.discriminator.update_normalizer(self.expert_obs, self.expert_next_obs)
            self.disc_optimizer = torch.optim.Adam(
                self.discriminator.parameters(), lr=cfg.disc_learning_rate
            )
            self.replay_buffer = TransitionReplayBuffer(
                capacity=cfg.disc_replay_capacity,
                obs_dim=vec_env.obs_dim,
                device=self.device,
            )

        self._current_obs = self.vec_env.reset(seed=cfg.seed)
        self.total_env_steps = 0

    def _update_discriminator(
        self, pol_obs: torch.Tensor, pol_next_obs: torch.Tensor
    ) -> tuple[float, float]:
        assert (
            self.discriminator is not None
            and self.disc_optimizer is not None
            and self.expert_obs is not None
            and self.expert_next_obs is not None
            and self.replay_buffer is not None
        )
        n_pol = int(pol_obs.shape[0])
        if n_pol == 0:
            return 0.0, 0.0

        n_exp = int(self.expert_obs.shape[0])
        batch_size = min(self.cfg.disc_batch_size, max(16, n_pol))
        losses: list[float] = []
        gps: list[float] = []

        for _ in range(self.cfg.disc_updates_per_iter):
            exp_idx = torch.randint(
                0, n_exp, (batch_size,), generator=self._rng, device="cpu"
            ).to(self.device)
            b_exp_obs = self.expert_obs[exp_idx]
            b_exp_next = self.expert_next_obs[exp_idx]

            if self.replay_buffer.size > 0:
                half = batch_size // 2
                rem = batch_size - half
                cur_idx = torch.randint(
                    0, n_pol, (half,), generator=self._rng, device="cpu"
                ).to(self.device)
                rep_obs, rep_next = self.replay_buffer.sample(rem, self._rng)
                b_pol_obs = torch.cat([pol_obs[cur_idx], rep_obs], dim=0)
                b_pol_next = torch.cat([pol_next_obs[cur_idx], rep_next], dim=0)
            else:
                cur_idx = torch.randint(
                    0, n_pol, (batch_size,), generator=self._rng, device="cpu"
                ).to(self.device)
                b_pol_obs = pol_obs[cur_idx]
                b_pol_next = pol_next_obs[cur_idx]

            loss, stats = self.discriminator.compute_loss(
                b_exp_obs, b_exp_next, b_pol_obs, b_pol_next, update_rms=True
            )
            self.disc_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), self.cfg.max_grad_norm
            )
            self.disc_optimizer.step()
            losses.append(stats.total_loss)
            gps.append(stats.grad_penalty)

        self.replay_buffer.add(pol_obs, pol_next_obs)
        return float(np.mean(losses)), float(np.mean(gps))

    def collect_and_update(self, rollout_steps: int | None = None) -> TrainStepStats:
        """Collect one on-policy rollout and update actor-critic (and discriminator).

        Args:
            rollout_steps: Optional override of ``cfg.rollout_steps`` to hit exact
                evaluation step boundaries.

        Returns:
            :class:`TrainStepStats` summary of the iteration.
        """
        horizon = int(rollout_steps or self.cfg.rollout_steps)
        n_envs = self.vec_env.num_envs
        obs_dim = self.vec_env.obs_dim
        act_dim = self.vec_env.act_dim

        obs_buf = torch.zeros(
            (horizon, n_envs, obs_dim), dtype=torch.float32, device=self.device
        )
        next_obs_buf = torch.zeros(
            (horizon, n_envs, obs_dim), dtype=torch.float32, device=self.device
        )
        act_buf = torch.zeros(
            (horizon, n_envs, act_dim), dtype=torch.float32, device=self.device
        )
        logp_buf = torch.zeros(
            (horizon, n_envs), dtype=torch.float32, device=self.device
        )
        val_buf = torch.zeros(
            (horizon, n_envs), dtype=torch.float32, device=self.device
        )
        env_rew_buf = torch.zeros(
            (horizon, n_envs), dtype=torch.float32, device=self.device
        )
        term_buf = torch.zeros((horizon, n_envs), dtype=torch.bool, device=self.device)
        trunc_buf = torch.zeros((horizon, n_envs), dtype=torch.bool, device=self.device)
        valid_trans_buf = torch.zeros(
            (horizon, n_envs), dtype=torch.bool, device=self.device
        )

        with torch.no_grad():
            for t in range(horizon):
                obs_buf[t] = self._current_obs
                raw_act, env_act, logp, val = self.actor_critic.act_and_value(
                    self._current_obs, generator=self._rng
                )
                step_out = self.vec_env.step(env_act)
                act_buf[t] = raw_act
                logp_buf[t] = logp
                val_buf[t] = val
                next_obs_buf[t] = step_out.transition_next_obs
                env_rew_buf[t] = step_out.rewards
                term_buf[t] = step_out.terminated
                trunc_buf[t] = step_out.truncated
                valid_trans_buf[t] = step_out.valid_transition
                self._current_obs = step_out.obs

        self.total_env_steps += horizon * n_envs

        if self.cfg.normalize_obs:
            self.actor_critic.obs_rms.update(obs_buf.reshape(-1, obs_dim))

        if self.cfg.use_discriminator and self.discriminator is not None:
            flat_obs = obs_buf.reshape(-1, obs_dim)
            flat_next = next_obs_buf.reshape(-1, obs_dim)
            flat_valid = valid_trans_buf.reshape(-1)
            style_flat = torch.zeros(
                horizon * n_envs, dtype=torch.float32, device=self.device
            )
            if bool(flat_valid.any()):
                style_flat[flat_valid] = self.discriminator.compute_style_reward(
                    flat_obs[flat_valid], flat_next[flat_valid]
                )
            style_rew_buf = style_flat.reshape(horizon, n_envs)
            blended_rew_buf = self.discriminator.combine_rewards(
                env_rew_buf, style_rew_buf, self.cfg.proportion_env_reward
            )
            disc_loss, disc_gp = self._update_discriminator(
                flat_obs[flat_valid], flat_next[flat_valid]
            )
        else:
            style_rew_buf = torch.zeros_like(env_rew_buf)
            blended_rew_buf = env_rew_buf
            disc_loss, disc_gp = 0.0, 0.0

        with torch.no_grad():
            last_val = self.actor_critic.value(self._current_obs)
            trunc_only = trunc_buf & (~term_buf) & valid_trans_buf
            trunc_bootstrap = torch.zeros_like(val_buf)
            if bool(trunc_only.any()):
                trunc_bootstrap[trunc_only] = self.actor_critic.value(
                    next_obs_buf[trunc_only]
                )

            adv_buf = torch.zeros_like(val_buf)
            gae = torch.zeros(n_envs, dtype=torch.float32, device=self.device)
            done_buf = term_buf | trunc_buf
            for t in reversed(range(horizon)):
                after_val = last_val if t == horizon - 1 else val_buf[t + 1]
                done_t = done_buf[t]
                next_val = torch.where(done_t, trunc_bootstrap[t], after_val)
                delta = blended_rew_buf[t] + self.cfg.gamma * next_val - val_buf[t]
                non_done = (~done_t).float()
                gae = delta + self.cfg.gamma * self.cfg.gae_lambda * non_done * gae
                adv_buf[t] = gae
            ret_buf = adv_buf + val_buf

        total_batch = horizon * n_envs
        b_obs = obs_buf.reshape(total_batch, obs_dim)
        b_act = act_buf.reshape(total_batch, act_dim)
        b_logp = logp_buf.reshape(total_batch)
        b_adv = adv_buf.reshape(total_batch)
        b_ret = ret_buf.reshape(total_batch)
        b_val = val_buf.reshape(total_batch)

        b_adv = (b_adv - b_adv.mean()) / (b_adv.std(unbiased=False) + 1e-8)
        minibatch_size = max(1, total_batch // max(1, self.cfg.num_minibatches))

        pol_losses: list[float] = []
        val_losses: list[float] = []
        entropies: list[float] = []

        for _ in range(self.cfg.ppo_epochs):
            perm = torch.randperm(total_batch, generator=self._rng, device="cpu").to(
                self.device
            )
            for start in range(0, total_batch, minibatch_size):
                mb_idx = perm[start : start + minibatch_size]
                new_logp, entropy, new_val = self.actor_critic.evaluate_actions(
                    b_obs[mb_idx], b_act[mb_idx]
                )
                ratio = torch.exp(new_logp - b_logp[mb_idx])
                mb_adv = b_adv[mb_idx]
                surr1 = ratio * mb_adv
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param
                    )
                    * mb_adv
                )
                pol_loss = -torch.min(surr1, surr2).mean()

                val_clipped = b_val[mb_idx] + torch.clamp(
                    new_val - b_val[mb_idx], -self.cfg.clip_param, self.cfg.clip_param
                )
                v_loss_unclipped = (new_val - b_ret[mb_idx]).square()
                v_loss_clipped = (val_clipped - b_ret[mb_idx]).square()
                val_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                ent_mean = entropy.mean()

                loss = (
                    pol_loss
                    + self.cfg.value_loss_coef * val_loss
                    - self.cfg.entropy_coef * ent_mean
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                pol_losses.append(float(pol_loss.detach().item()))
                val_losses.append(float(val_loss.detach().item()))
                entropies.append(float(ent_mean.detach().item()))

        return TrainStepStats(
            env_steps=self.total_env_steps,
            mean_env_reward=float(env_rew_buf.mean().item()),
            mean_style_reward=float(style_rew_buf.mean().item()),
            mean_blended_reward=float(blended_rew_buf.mean().item()),
            policy_loss=float(np.mean(pol_losses)),
            value_loss=float(np.mean(val_losses)),
            entropy=float(np.mean(entropies)),
            disc_loss=disc_loss,
            disc_grad_penalty=disc_gp,
        )

    def train(
        self,
        total_steps: int,
        eval_interval: int | None = None,
        eval_callback: Callable[[ActorCritic, int], None] | None = None,
    ) -> list[TrainStepStats]:
        """Train until ``total_env_steps`` reaches ``total_steps``.

        Args:
            total_steps: Total environment interactions across all parallel envs.
            eval_interval: Optional step interval at which ``eval_callback`` fires.
            eval_callback: Optional callback ``fn(actor_critic, env_steps)``.

        Returns:
            List of :class:`TrainStepStats` from each rollout update.
        """
        history: list[TrainStepStats] = []
        next_eval = eval_interval if eval_interval and eval_interval > 0 else None
        if eval_callback is not None:
            eval_callback(self.actor_critic, self.total_env_steps)

        n_envs = self.vec_env.num_envs
        while self.total_env_steps < total_steps:
            target = total_steps if next_eval is None else min(total_steps, next_eval)
            rem_steps = max(n_envs, target - self.total_env_steps)
            step_horizon = min(self.cfg.rollout_steps, max(1, rem_steps // n_envs))
            stats = self.collect_and_update(rollout_steps=step_horizon)
            history.append(stats)
            if (
                eval_callback is not None
                and next_eval is not None
                and self.total_env_steps >= next_eval
            ):
                eval_callback(self.actor_critic, self.total_env_steps)
                assert eval_interval is not None
                next_eval += eval_interval

        if eval_callback is not None and (
            next_eval is None
            or (self.total_env_steps < next_eval and eval_interval is not None)
        ):
            # Trigger final evaluation if total_steps was not an exact multiple of eval_interval.
            if next_eval is None or (self.total_env_steps % (eval_interval or 1) != 0):
                eval_callback(self.actor_critic, self.total_env_steps)

        return history


@dataclass
class BCConfig:
    """Hyperparameters for supervised Behaviour Cloning on ``(obs, act)`` pairs."""

    actor_hidden_dims: tuple[int, ...] = (128, 128)
    critic_hidden_dims: tuple[int, ...] = (128, 128)
    normalize_obs: bool = True
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cpu"


class BehaviorCloningTrainer:
    """Supervised MSE trainer matching :class:`ActorCritic` architecture and normalizer."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: BCConfig,
        action_low: np.ndarray | float = -1.0,
        action_high: np.ndarray | float = 1.0,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(cfg.seed + 31)
        self.actor_critic = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            actor_hidden_dims=cfg.actor_hidden_dims,
            critic_hidden_dims=cfg.critic_hidden_dims,
            normalize_obs=cfg.normalize_obs,
            action_low=action_low,
            action_high=action_high,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            list(self.actor_critic.actor_trunk.parameters())
            + list(self.actor_critic.actor_mean.parameters()),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    def fit(
        self,
        obs: np.ndarray | torch.Tensor,
        actions: np.ndarray | torch.Tensor,
        epoch_callback: Callable[[ActorCritic, int, float], None] | None = None,
    ) -> list[float]:
        """Fit actor mean network on expert ``(obs, actions)`` pairs.

        Args:
            obs: Expert observations ``(N, obs_dim)``.
            actions: Expert actions ``(N, act_dim)``.
            epoch_callback: Optional ``fn(actor_critic, epoch, mse_loss)``.

        Returns:
            Per-epoch MSE training losses.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if obs_t.shape[0] != act_t.shape[0]:
            raise ValueError(
                f"obs and actions count mismatch: {obs_t.shape[0]} vs {act_t.shape[0]}."
            )
        if self.cfg.normalize_obs:
            self.actor_critic.obs_rms.update(obs_t)

        act_targets = torch.clamp(
            act_t, self.actor_critic.action_low, self.actor_critic.action_high
        )
        n_samples = int(obs_t.shape[0])
        batch_size = min(self.cfg.batch_size, max(1, n_samples))
        epoch_losses: list[float] = []

        if epoch_callback is not None:
            epoch_callback(self.actor_critic, 0, float("nan"))

        for epoch in range(1, self.cfg.epochs + 1):
            perm = torch.randperm(n_samples, generator=self._rng, device="cpu").to(
                self.device
            )
            batch_mse: list[float] = []
            for start in range(0, n_samples, batch_size):
                idx = perm[start : start + batch_size]
                norm_obs = self.actor_critic._norm(obs_t[idx])
                pred = self.actor_critic.actor_mean(
                    self.actor_critic.actor_trunk(norm_obs)
                )
                loss = F.mse_loss(pred, act_targets[idx])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_mse.append(float(loss.detach().item()))
            mean_mse = float(np.mean(batch_mse))
            epoch_losses.append(mean_mse)
            if epoch_callback is not None:
                epoch_callback(self.actor_critic, epoch, mean_mse)

        return epoch_losses


def train_bc(
    obs: np.ndarray | torch.Tensor,
    actions: np.ndarray | torch.Tensor,
    cfg: BCConfig | None = None,
    *,
    action_low: np.ndarray | float = -1.0,
    action_high: np.ndarray | float = 1.0,
    epoch_callback: Callable[[ActorCritic, int, float], None] | None = None,
) -> tuple[ActorCritic, list[float]]:
    """Train a Behaviour Cloning policy on expert ``(obs, act)`` pairs.

    Args:
        obs: Expert observations ``(N, obs_dim)``.
        actions: Expert actions ``(N, act_dim)``.
        cfg: Optional :class:`BCConfig`.
        action_low: Action space lower bound.
        action_high: Action space upper bound.
        epoch_callback: Optional callback per epoch.

    Returns:
        Tuple ``(trained_actor_critic, epoch_losses)``.
    """
    obs_arr = np.asarray(obs, dtype=np.float32)
    act_arr = np.asarray(actions, dtype=np.float32)
    bc_cfg = cfg or BCConfig()
    trainer = BehaviorCloningTrainer(
        obs_dim=int(obs_arr.shape[1]),
        act_dim=int(act_arr.shape[1]),
        cfg=bc_cfg,
        action_low=action_low,
        action_high=action_high,
    )
    losses = trainer.fit(obs_arr, act_arr, epoch_callback=epoch_callback)
    return trainer.actor_critic, losses


def evaluate_policy_cpu(
    actor: ActorCritic,
    env_id: str,
    n_episodes: int = 20,
    seed: int = 10_000,
) -> EvalStats:
    """Evaluate deterministic policy on unmodified CPU ``make_env(env_id)`` episodes.

    Args:
        actor: Policy network to evaluate.
        env_id: Registered MyoSuite environment ID.
        n_episodes: Number of seeded episodes to roll out.
        seed: Base evaluation seed (episode ``k`` uses ``seed + k``).

    Returns:
        :class:`EvalStats` with mean/std return and final-step solved rate.
    """
    env = make_env(env_id)
    was_training = actor.training
    actor.eval()
    returns: list[float] = []
    lengths: list[int] = []
    solved_flags: list[float] = []
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            ep_ret = 0.0
            ep_len = 0
            last_info: dict = {}
            while True:
                act = actor.predict_np(np.asarray(obs, dtype=np.float32))
                obs, rew, term, trunc, last_info = env.step(act)
                ep_ret += float(rew)
                ep_len += 1
                if term or trunc:
                    break
            returns.append(ep_ret)
            lengths.append(ep_len)
            solved_flags.append(float(bool(last_info.get("solved", False))))
    finally:
        if was_training:
            actor.train()
        env.close()

    return EvalStats(
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        solved_rate=float(np.mean(solved_flags)),
        mean_length=float(np.mean(lengths)),
        n_episodes=n_episodes,
    )


def collect_expert_dataset(
    env_id: str,
    n_transitions: int = 2000,
    seed: int = 100,
    expert_steps: int = 24_576,
    n_envs: int = 8,
    actor_hidden_dims: tuple[int, ...] = (128, 128),
    critic_hidden_dims: tuple[int, ...] = (128, 128),
    expert_policy: Callable[[np.ndarray], np.ndarray] | ActorCritic | None = None,
) -> ExpertDataset:
    """Collect clean demonstration transitions on ``env_id`` with zero external weights.

    When ``expert_policy`` is ``None``, trains a compact in-memory task-reward PPO
    teacher for ``expert_steps`` interactions, evaluates it, and rolls out seeded
    demonstration episodes (preferring solved and high-return episodes) to gather
    ``n_transitions`` transitions.

    Args:
        env_id: Registered MyoSuite environment ID.
        n_transitions: Exact number of transitions to return.
        seed: Deterministic seed for teacher training and rollout collection.
        expert_steps: Environment steps used to train the in-memory PPO teacher.
        n_envs: Parallel CPU environments for teacher training.
        actor_hidden_dims: Teacher actor hidden layer sizes.
        critic_hidden_dims: Teacher critic hidden layer sizes.
        expert_policy: Optional pre-built policy callable or :class:`ActorCritic`.

    Returns:
        :class:`ExpertDataset` with ``n_transitions`` rows.
    """
    if n_transitions <= 0:
        raise ValueError(f"n_transitions must be positive, got {n_transitions}.")
    t0 = time.perf_counter()
    teacher_mean_ret = 0.0
    teacher_solved = 0.0
    actual_train_steps = 0

    if expert_policy is None:
        vec_env = CpuVecEnv(env_id, num_envs=n_envs, device="cpu")
        try:
            teacher_cfg = AMPConfig(
                use_discriminator=False,
                proportion_env_reward=1.0,
                actor_hidden_dims=actor_hidden_dims,
                critic_hidden_dims=critic_hidden_dims,
                rollout_steps=min(128, max(16, expert_steps // max(1, n_envs))),
                seed=seed,
                device="cpu",
            )
            teacher_trainer = AMPTrainer(vec_env, teacher_cfg)
            teacher_trainer.train(total_steps=expert_steps)
            teacher_actor = teacher_trainer.actor_critic
            actual_train_steps = teacher_trainer.total_env_steps
        finally:
            vec_env.close()
        eval_res = evaluate_policy_cpu(
            teacher_actor, env_id, n_episodes=10, seed=seed + 50_000
        )
        teacher_mean_ret = eval_res.mean_return
        teacher_solved = eval_res.solved_rate
        policy_fn: Callable[[np.ndarray], np.ndarray] = teacher_actor.predict_np
    elif isinstance(expert_policy, ActorCritic):
        eval_res = evaluate_policy_cpu(
            expert_policy, env_id, n_episodes=10, seed=seed + 50_000
        )
        teacher_mean_ret = eval_res.mean_return
        teacher_solved = eval_res.solved_rate
        policy_fn = expert_policy.predict_np
    else:
        policy_fn = expert_policy

    env = make_env(env_id)
    rng = np.random.default_rng(seed + 90_000)
    act_low = np.asarray(env.action_space.low, dtype=np.float32)
    act_high = np.asarray(env.action_space.high, dtype=np.float32)

    episodes: list[
        tuple[bool, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    target_pool_steps = int(math.ceil(n_transitions * 1.5))
    collected_steps = 0
    ep_idx = 0
    try:
        while collected_steps < target_pool_steps:
            obs, _ = env.reset(seed=seed + 70_000 + ep_idx)
            ep_idx += 1
            ep_obs: list[np.ndarray] = []
            ep_next: list[np.ndarray] = []
            ep_act: list[np.ndarray] = []
            ep_rew: list[float] = []
            ep_done: list[bool] = []
            last_info: dict = {}
            while True:
                obs_f = np.asarray(obs, dtype=np.float32)
                det_act = np.asarray(policy_fn(obs_f), dtype=np.float32)
                if ep_idx % 4 == 0:
                    jitter = rng.normal(0.0, 0.03, size=det_act.shape).astype(
                        np.float32
                    )
                    act = np.clip(det_act + jitter, act_low, act_high)
                else:
                    act = np.clip(det_act, act_low, act_high)
                next_obs, rew, term, trunc, last_info = env.step(act)
                next_f = np.asarray(next_obs, dtype=np.float32)
                done = bool(term or trunc)
                ep_obs.append(obs_f)
                ep_next.append(next_f)
                ep_act.append(act)
                ep_rew.append(float(rew))
                ep_done.append(done)
                obs = next_f
                if done:
                    break
            solved = bool(last_info.get("solved", False))
            ep_ret = float(sum(ep_rew))
            episodes.append(
                (
                    solved,
                    ep_ret,
                    np.stack(ep_obs, axis=0),
                    np.stack(ep_next, axis=0),
                    np.stack(ep_act, axis=0),
                    np.asarray(ep_rew, dtype=np.float32),
                    np.asarray(ep_done, dtype=bool),
                )
            )
            collected_steps += len(ep_rew)
    finally:
        env.close()

    episodes.sort(key=lambda item: (item[0], item[1]), reverse=True)
    obs_cat = np.concatenate([e[2] for e in episodes], axis=0)[:n_transitions]
    next_cat = np.concatenate([e[3] for e in episodes], axis=0)[:n_transitions]
    act_cat = np.concatenate([e[4] for e in episodes], axis=0)[:n_transitions]
    rew_cat = np.concatenate([e[5] for e in episodes], axis=0)[:n_transitions]
    done_cat = np.concatenate([e[6] for e in episodes], axis=0)[:n_transitions]
    wall_s = time.perf_counter() - t0

    return ExpertDataset(
        obs=obs_cat,
        next_obs=next_cat,
        actions=act_cat,
        rewards=rew_cat,
        dones=done_cat,
        env_id=env_id,
        teacher_mean_return=teacher_mean_ret,
        teacher_solved_rate=teacher_solved,
        teacher_train_steps=actual_train_steps,
        generation_wall_time_s=wall_s,
    )


class _DeterministicExportModule(nn.Module):
    """Self-contained deterministic policy with folded observation normalization."""

    obs_mean: torch.Tensor
    obs_std: torch.Tensor
    action_low: torch.Tensor
    action_high: torch.Tensor

    def __init__(self, actor: ActorCritic) -> None:
        super().__init__()
        self.normalize_obs = actor.normalize_obs_enabled
        self.register_buffer("obs_mean", actor.obs_rms.mean.detach().cpu().clone())
        self.register_buffer(
            "obs_std",
            torch.sqrt(actor.obs_rms.var.detach().cpu() + actor.obs_rms.eps).clone(),
        )
        self.register_buffer("action_low", actor.action_low.detach().cpu().clone())
        self.register_buffer("action_high", actor.action_high.detach().cpu().clone())
        self.trunk = copy.deepcopy(actor.actor_trunk).cpu().eval()
        self.mean_head = copy.deepcopy(actor.actor_mean).cpu().eval()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        if self.normalize_obs:
            x = torch.clamp((x - self.obs_mean) / self.obs_std, -10.0, 10.0)
        mean = self.mean_head(self.trunk(x))
        return torch.clamp(mean, self.action_low, self.action_high)


def export_policy_torchscript(actor: ActorCritic, path: str | Path) -> Path:
    """Export ``actor`` to a standalone TorchScript archive loadable without ``myosuite_amp``.

    Args:
        actor: Trained :class:`ActorCritic` module.
        path: Output file path (e.g., ``tmp_dir / "policy.pt"``).

    Returns:
        Resolved output :class:`Path`.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _DeterministicExportModule(actor).eval()
    dummy = torch.zeros((1, actor.obs_dim), dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy)
        traced.save(str(out))
    return out


def export_policy_onnx(actor: ActorCritic, path: str | Path, opset: int = 17) -> Path:
    """Export ``actor`` to a single self-contained ONNX graph matching MyoSuite conventions.

    Uses ``myosuite.utils.export_onnx._export_policy`` (dynamic batch axis ``obs -> action``,
    folded running observation statistics, ``dynamo=False``).

    Args:
        actor: Trained :class:`ActorCritic` module.
        path: Destination ``.onnx`` path.
        opset: ONNX opset version.

    Returns:
        Output :class:`Path`.

    Raises:
        ImportError: If the ``onnx`` package is not installed in the active environment.
    """
    if importlib.util.find_spec("onnx") is None:
        raise ImportError(
            "ONNX export requires the 'onnx' package (`pip install onnx onnxruntime`). "
            "Use `export_policy_torchscript` when 'onnx' is unavailable."
        )
    from myosuite.utils.export_onnx import _export_policy

    out = Path(path)
    wrapper = _DeterministicExportModule(actor).eval()
    _export_policy(wrapper, actor.obs_dim, out, opset)
    return out
