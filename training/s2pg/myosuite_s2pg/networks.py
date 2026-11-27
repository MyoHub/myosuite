# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
#
# Network design and IndependentJointDist adapted from LocoMuJoCo
# (https://github.com/robfiras/loco-mujoco,
# loco_mujoco/algorithms/experimental/s2pg_ppo_jax.py), Copyright (c) 2024
# Al-Hafez, MIT License (see LICENSE-loco-mujoco.txt), based on:
# Firas Al-Hafez, Guoping Zhao, Jan Peters, and Davide Tateo,
# "Time-Efficient Reinforcement Learning with Stochastic Stateful Policies",
# ICLR 2024. Reimplemented in PyTorch for MyoSuite.
"""PyTorch actor-critic networks and joint distributions for S2PG-PPO."""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.distributions import Independent, Normal

RNNType = Literal["gru", "lstm", "vanilla"]
VALID_RNN_TYPES: tuple[RNNType, ...] = ("gru", "lstm", "vanilla")


def stored_hidden_dim(hidden_state_dim: int, rnn_type: str) -> int:
    """Return the flat latent vector width stored per environment step.

    Args:
        hidden_state_dim: Width of a single recurrent hidden vector.
        rnn_type: Recurrent transition cell type (``"gru"``, ``"lstm"``, or ``"vanilla"``).

    Returns:
        ``2 * hidden_state_dim`` for ``"lstm"`` (concatenated ``[h, c]``) and
        ``hidden_state_dim`` for ``"gru"`` or ``"vanilla"``.

    Raises:
        ValueError: If ``hidden_state_dim <= 0`` or ``rnn_type`` is unsupported.
    """
    if hidden_state_dim <= 0:
        raise ValueError(f"hidden_state_dim must be positive, got {hidden_state_dim}.")
    if rnn_type not in VALID_RNN_TYPES:
        raise ValueError(
            f"Unsupported rnn_type {rnn_type!r}; choose from {VALID_RNN_TYPES}."
        )
    return 2 * hidden_state_dim if rnn_type == "lstm" else hidden_state_dim


def _make_activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(
        f"Unsupported activation {name!r}; choose from tanh, elu, relu, silu."
    )


def _orthogonal_linear(
    in_features: int, out_features: int, gain: float = math.sqrt(2.0)
) -> nn.Linear:
    layer = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


def _build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    activation: str = "tanh",
    gain: float = math.sqrt(2.0),
) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    curr_dim = in_dim
    for width in hidden_dims:
        layers.append(_orthogonal_linear(curr_dim, width, gain=gain))
        layers.append(_make_activation(activation))
        curr_dim = width
    return nn.Sequential(*layers), curr_dim


class RunningMeanStd(nn.Module):
    """Running observation mean and variance with parallel Welford updates."""

    mean: torch.Tensor
    var: torch.Tensor
    count: torch.Tensor

    def __init__(
        self,
        dim: int,
        eps: float = 1e-8,
        init_count: float = 1e-4,
        clip: float | None = 10.0,
    ) -> None:
        """Initialize running statistics buffers.

        Args:
            dim: Feature dimension of observations.
            eps: Numerical floor added inside variance square root.
            init_count: Initial pseudo-count preventing division by zero.
            clip: Optional symmetric bound applied after standardization.
        """
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")
        self.dim = dim
        self.eps = eps
        self.clip = clip
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float32))
        self.register_buffer(
            "count", torch.tensor(float(init_count), dtype=torch.float32)
        )

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update running mean and variance from a batch of observations.

        Args:
            x: Tensor with trailing dimension ``dim``.
        """
        flat = x.detach().reshape(-1, self.dim).to(dtype=torch.float32)
        if flat.shape[0] == 0:
            return
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)
        batch_count = float(flat.shape[0])

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * (self.count * batch_count / total_count)

        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total_count)
        self.count.fill_(total_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize ``x`` using current running statistics."""
        normed = (x - self.mean) * torch.rsqrt(self.var + self.eps)
        if self.clip is not None:
            normed = torch.clamp(normed, -self.clip, self.clip)
        return normed


class IndependentJointDist:
    """Factorized diagonal Gaussian over joint action and next hidden state ``(a_t, z_{t+1})``.

    In Stochastic Stateful Policy Gradient (S2PG), the stochastic policy kernel is
    ``p_theta(a_t, z_{t+1} | o_t, z_t, a_{t-1}) = pi_theta(a_t | o_t, z_t, a_{t-1})
    * q_theta(z_{t+1} | o_t, z_t, a_{t-1})``, so log-densities and differential
    entropies add across the action and latent components.
    """

    def __init__(
        self,
        mean_a: torch.Tensor,
        std_a: torch.Tensor,
        mean_z: torch.Tensor,
        std_z: torch.Tensor,
    ) -> None:
        """Construct the joint distribution from component means and standard deviations.

        Args:
            mean_a: Action mean of shape ``(..., action_dim)``.
            std_a: Action standard deviation broadcastable to ``mean_a``.
            mean_z: Next-hidden-state mean of shape ``(..., latent_dim)``.
            std_z: Next-hidden-state standard deviation broadcastable to ``mean_z``.
        """
        if mean_a.shape[:-1] != mean_z.shape[:-1]:
            raise ValueError(
                f"Batch shapes of mean_a {mean_a.shape} and mean_z {mean_z.shape} differ."
            )
        self.action_dim = mean_a.shape[-1]
        self.latent_dim = mean_z.shape[-1]
        self.joint_dim = self.action_dim + self.latent_dim
        self.mean_a = mean_a
        self.mean_z = mean_z
        self.std_a = std_a.expand_as(mean_a)
        self.std_z = std_z.expand_as(mean_z)
        self.dist_a = Independent(Normal(self.mean_a, self.std_a), 1)
        self.dist_z = Independent(Normal(self.mean_z, self.std_z), 1)

    @property
    def mode(self) -> torch.Tensor:
        """Return concatenated deterministic mode ``concat([mean_a, mean_z])``."""
        return torch.cat([self.mean_a, self.mean_z], dim=-1)

    @property
    def mean(self) -> torch.Tensor:
        """Return concatenated mean ``concat([mean_a, mean_z])``."""
        return self.mode

    def split(self, joint_az: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split concatenated ``joint_az`` into ``(action, next_hidden_state)``.

        Args:
            joint_az: Tensor with trailing dimension ``action_dim + latent_dim``.

        Returns:
            Tuple ``(a, z_next)``.
        """
        if joint_az.shape[-1] != self.joint_dim:
            raise ValueError(
                f"Expected trailing joint dim {self.joint_dim}, got {joint_az.shape[-1]}."
            )
        return (
            joint_az[..., : self.action_dim],
            joint_az[..., self.action_dim :],
        )

    def log_prob(self, joint_az: torch.Tensor) -> torch.Tensor:
        """Evaluate joint log-density ``log p(a_t) + log p(z_{t+1})``."""
        action, z_next = self.split(joint_az)
        return self.dist_a.log_prob(action) + self.dist_z.log_prob(z_next)

    def entropy(self) -> torch.Tensor:
        """Evaluate joint differential entropy ``H(a_t) + H(z_{t+1})``."""
        return self.dist_a.entropy() + self.dist_z.entropy()

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Draw a stochastic sample ``concat([a_t, z_{t+1}])``."""
        action = self.dist_a.sample(sample_shape)
        z_next = self.dist_z.sample(sample_shape)
        return torch.cat([action, z_next], dim=-1)


class ActorCriticS2PG(nn.Module):
    """Stochastic stateful actor-critic ported from LocoMuJoCo to PyTorch.

    Because ``z_{t+1}`` is sampled stochastically during rollout and stored alongside
    ``(o_t, z_t, a_{t-1}, a_t)`` in the rollout buffer, PPO updates evaluate single-step
    transitions ``(o_t, z_t, a_{t-1}) -> (a_t, z_{t+1})`` in flat shuffled minibatches
    without truncated backpropagation through time (BPTT).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_state_dim: int = 64,
        rnn_type: RNNType = "gru",
        ema_coef: float = 1.0,
        pre_rnn_obs_dims: Sequence[int] = (128,),
        pre_rnn_act_dims: Sequence[int] = (32,),
        post_rnn_obs_dims: Sequence[int] = (64,),
        actor_hidden_dims: Sequence[int] = (128,),
        critic_hidden_dims: Sequence[int] = (256, 256),
        vanilla_rnn_dims: Sequence[int] = (128,),
        activation: str = "tanh",
        init_std_a: float = 0.5,
        init_std_z: float = 0.1,
        normalize_obs: bool = True,
        log_std_bounds: tuple[float, float] = (-5.0, 1.0),
    ) -> None:
        """Initialize the S2PG actor-critic network.

        Args:
            obs_dim: Environment observation dimension.
            action_dim: Environment action dimension.
            hidden_state_dim: Recurrent cell width (``stored_hidden_dim`` doubles for LSTM).
            rnn_type: One of ``"gru"``, ``"lstm"``, or ``"vanilla"``.
            ema_coef: Exponential smoothing weight in ``[0, 1]`` for ``next_z_mean``.
                ``1.0`` uses the raw cell transition ``f_theta(o_t, z_t, a_{t-1})``.
            pre_rnn_obs_dims: MLP widths embedding normalized ``obs`` before the RNN cell.
            pre_rnn_act_dims: MLP widths embedding ``prev_action`` before the RNN cell.
            post_rnn_obs_dims: MLP widths re-embedding normalized ``obs`` after the RNN cell.
            actor_hidden_dims: MLP widths after concatenating post-RNN obs and RNN readout.
            critic_hidden_dims: Critic MLP widths over ``[obs_norm, hidden_state, prev_action]``.
            vanilla_rnn_dims: Hidden widths of the MLP transition when ``rnn_type="vanilla"``.
            activation: Nonlinearity name (``"tanh"``, ``"elu"``, ``"relu"``, ``"silu"``).
            init_std_a: Initial Gaussian exploration std for actions.
            init_std_z: Initial Gaussian exploration std for next hidden states.
            normalize_obs: Whether to standardize observations with :class:`RunningMeanStd`.
            log_std_bounds: ``(min_log_std, max_log_std)`` bounds applied before ``exp``.
        """
        super().__init__()
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError(
                f"obs_dim ({obs_dim}) and action_dim ({action_dim}) must be positive."
            )
        if not 0.0 <= ema_coef <= 1.0:
            raise ValueError(f"ema_coef must lie in [0, 1], got {ema_coef}.")
        if init_std_a <= 0.0 or init_std_z <= 0.0:
            raise ValueError("init_std_a and init_std_z must be strictly positive.")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_state_dim = hidden_state_dim
        self.rnn_type: RNNType = rnn_type
        self.latent_dim = stored_hidden_dim(hidden_state_dim, rnn_type)
        self.ema_coef = float(ema_coef)
        self.normalize_obs = normalize_obs
        self.min_log_std, self.max_log_std = log_std_bounds

        self.obs_rms = RunningMeanStd(obs_dim)

        self.obs_pre_rnn, obs_pre_dim = _build_mlp(
            obs_dim, pre_rnn_obs_dims, activation=activation
        )
        self.act_pre_rnn, act_pre_dim = _build_mlp(
            action_dim, pre_rnn_act_dims, activation=activation
        )
        rnn_input_dim = obs_pre_dim + act_pre_dim

        self.gru_cell: nn.GRUCell | None = None
        self.lstm_cell: nn.LSTMCell | None = None
        self.vanilla_cell: nn.Sequential | None = None

        if rnn_type == "gru":
            self.gru_cell = nn.GRUCell(rnn_input_dim, hidden_state_dim)
            nn.init.orthogonal_(self.gru_cell.weight_ih)
            nn.init.orthogonal_(self.gru_cell.weight_hh)
            nn.init.zeros_(self.gru_cell.bias_ih)
            nn.init.zeros_(self.gru_cell.bias_hh)
        elif rnn_type == "lstm":
            self.lstm_cell = nn.LSTMCell(rnn_input_dim, hidden_state_dim)
            nn.init.orthogonal_(self.lstm_cell.weight_ih)
            nn.init.orthogonal_(self.lstm_cell.weight_hh)
            nn.init.zeros_(self.lstm_cell.bias_ih)
            nn.init.zeros_(self.lstm_cell.bias_hh)
        else:
            vanilla_trunk, vanilla_last = _build_mlp(
                rnn_input_dim + self.latent_dim,
                vanilla_rnn_dims,
                activation=activation,
            )
            self.vanilla_cell = nn.Sequential(
                vanilla_trunk,
                _orthogonal_linear(vanilla_last, hidden_state_dim, gain=1.0),
                nn.Tanh(),
            )

        self.obs_post_rnn, obs_post_dim = _build_mlp(
            obs_dim, post_rnn_obs_dims, activation=activation
        )
        self.actor_trunk, actor_last_dim = _build_mlp(
            obs_post_dim + hidden_state_dim, actor_hidden_dims, activation=activation
        )
        self.action_mean_head = _orthogonal_linear(
            actor_last_dim, action_dim, gain=0.01
        )

        self.log_std_a = nn.Parameter(
            torch.full((action_dim,), math.log(init_std_a), dtype=torch.float32)
        )
        self.log_std_z = nn.Parameter(
            torch.full((self.latent_dim,), math.log(init_std_z), dtype=torch.float32)
        )

        critic_in_dim = obs_dim + self.latent_dim + action_dim
        self.critic_trunk, critic_last_dim = _build_mlp(
            critic_in_dim, critic_hidden_dims, activation=activation
        )
        self.critic_head = _orthogonal_linear(critic_last_dim, 1, gain=1.0)

    def initial_state(
        self, batch_size: int, device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zeroed ``(hidden_state, prev_action)`` tensors for ``batch_size`` envs."""
        dev = (
            torch.device(device)
            if device is not None
            else self.action_mean_head.weight.device
        )
        hidden = torch.zeros(
            (batch_size, self.latent_dim), dtype=torch.float32, device=dev
        )
        prev_action = torch.zeros(
            (batch_size, self.action_dim), dtype=torch.float32, device=dev
        )
        return hidden, prev_action

    def normalize_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize ``obs`` if observation standardization is enabled."""
        return self.obs_rms(obs) if self.normalize_obs else obs

    def _transition_cell(
        self, rnn_in: torch.Tensor, hidden_state: torch.Tensor
    ) -> torch.Tensor:
        if rnn_in.dim() == 2:
            return self._transition_cell_2d(rnn_in, hidden_state)
        batch_shape = rnn_in.shape[:-1]
        flat_in = rnn_in.reshape(-1, rnn_in.shape[-1])
        flat_h = hidden_state.reshape(-1, self.latent_dim)
        flat_out = self._transition_cell_2d(flat_in, flat_h)
        return flat_out.reshape(*batch_shape, self.latent_dim)

    def _transition_cell_2d(
        self, rnn_in: torch.Tensor, hidden_state: torch.Tensor
    ) -> torch.Tensor:
        if self.gru_cell is not None:
            return self.gru_cell(rnn_in, hidden_state)
        if self.lstm_cell is not None:
            h_prev = hidden_state[:, : self.hidden_state_dim]
            c_prev = hidden_state[:, self.hidden_state_dim :]
            h_next, c_next = self.lstm_cell(rnn_in, (h_prev, c_prev))
            return torch.cat([h_next, c_next], dim=-1)
        assert self.vanilla_cell is not None
        return self.vanilla_cell(torch.cat([rnn_in, hidden_state], dim=-1))

    def _actor_means_from_norm_obs(
        self,
        obs_norm: torch.Tensor,
        hidden_state: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_pre = self.obs_pre_rnn(obs_norm)
        act_pre = self.act_pre_rnn(prev_action)
        rnn_in = torch.cat([obs_pre, act_pre], dim=-1)
        raw_next_z = self._transition_cell(rnn_in, hidden_state)
        if self.ema_coef < 1.0:
            next_z_mean = (
                1.0 - self.ema_coef
            ) * hidden_state + self.ema_coef * raw_next_z
        else:
            next_z_mean = raw_next_z

        rnn_readout = next_z_mean[..., : self.hidden_state_dim]
        obs_post = self.obs_post_rnn(obs_norm)
        actor_feat = self.actor_trunk(torch.cat([obs_post, rnn_readout], dim=-1))
        action_mean = self.action_mean_head(actor_feat)
        return action_mean, next_z_mean

    def _critic_from_norm_obs(
        self,
        obs_norm: torch.Tensor,
        hidden_state: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> torch.Tensor:
        critic_in = torch.cat([obs_norm, hidden_state, prev_action], dim=-1)
        return self.critic_head(self.critic_trunk(critic_in)).squeeze(-1)

    def _stds(self) -> tuple[torch.Tensor, torch.Tensor]:
        std_a = torch.exp(
            torch.clamp(self.log_std_a, self.min_log_std, self.max_log_std)
        )
        std_z = torch.exp(
            torch.clamp(self.log_std_z, self.min_log_std, self.max_log_std)
        )
        return std_a, std_z

    def actor_means(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute deterministic ``(action_mean, next_z_mean)``."""
        obs_norm = self.normalize_observation(obs)
        return self._actor_means_from_norm_obs(obs_norm, hidden_state, prev_action)

    def deterministic_act(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return deterministic evaluation outputs ``(mean_a, mean_z)``."""
        return self.actor_means(obs, hidden_state, prev_action)

    def get_dist(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> IndependentJointDist:
        """Return the joint policy/latent Gaussian ``IndependentJointDist``."""
        mean_a, mean_z = self.actor_means(obs, hidden_state, prev_action)
        std_a, std_z = self._stds()
        return IndependentJointDist(mean_a, std_a, mean_z, std_z)

    def get_value(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate critic ``V(o_t, z_t, a_{t-1})``."""
        obs_norm = self.normalize_observation(obs)
        return self._critic_from_norm_obs(obs_norm, hidden_state, prev_action)

    def get_dist_and_value(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> tuple[IndependentJointDist, torch.Tensor]:
        """Evaluate both the joint distribution and critic sharing normalized ``obs``."""
        obs_norm = self.normalize_observation(obs)
        mean_a, mean_z = self._actor_means_from_norm_obs(
            obs_norm, hidden_state, prev_action
        )
        std_a, std_z = self._stds()
        value = self._critic_from_norm_obs(obs_norm, hidden_state, prev_action)
        return IndependentJointDist(mean_a, std_a, mean_z, std_z), value

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ``(action_mean, next_z_mean, value)``."""
        obs_norm = self.normalize_observation(obs)
        mean_a, mean_z = self._actor_means_from_norm_obs(
            obs_norm, hidden_state, prev_action
        )
        value = self._critic_from_norm_obs(obs_norm, hidden_state, prev_action)
        return mean_a, mean_z, value


class ActorCriticFeedforward(nn.Module):
    """Matched stateless feedforward actor-critic for apple-to-apple PPO comparisons."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        actor_hidden_dims: Sequence[int] = (128, 128),
        critic_hidden_dims: Sequence[int] = (256, 256),
        activation: str = "tanh",
        init_std_a: float = 0.5,
        normalize_obs: bool = True,
        log_std_bounds: tuple[float, float] = (-5.0, 1.0),
    ) -> None:
        """Initialize the stateless PPO baseline network.

        Args:
            obs_dim: Environment observation dimension.
            action_dim: Environment action dimension.
            actor_hidden_dims: Actor MLP widths.
            critic_hidden_dims: Critic MLP widths.
            activation: Nonlinearity name.
            init_std_a: Initial Gaussian action std.
            normalize_obs: Whether to standardize observations with :class:`RunningMeanStd`.
            log_std_bounds: ``(min_log_std, max_log_std)`` bounds applied before ``exp``.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.normalize_obs = normalize_obs
        self.min_log_std, self.max_log_std = log_std_bounds

        self.obs_rms = RunningMeanStd(obs_dim)
        self.actor_trunk, actor_last_dim = _build_mlp(
            obs_dim, actor_hidden_dims, activation=activation
        )
        self.action_mean_head = _orthogonal_linear(
            actor_last_dim, action_dim, gain=0.01
        )
        self.log_std_a = nn.Parameter(
            torch.full((action_dim,), math.log(init_std_a), dtype=torch.float32)
        )

        self.critic_trunk, critic_last_dim = _build_mlp(
            obs_dim, critic_hidden_dims, activation=activation
        )
        self.critic_head = _orthogonal_linear(critic_last_dim, 1, gain=1.0)

    def normalize_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize ``obs`` if observation standardization is enabled."""
        return self.obs_rms(obs) if self.normalize_obs else obs

    def deterministic_act(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic action mean ``mean_a``."""
        obs_norm = self.normalize_observation(obs)
        return self.action_mean_head(self.actor_trunk(obs_norm))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Evaluate stateless critic ``V(o_t)``."""
        obs_norm = self.normalize_observation(obs)
        return self.critic_head(self.critic_trunk(obs_norm)).squeeze(-1)

    def get_dist_and_value(self, obs: torch.Tensor) -> tuple[Independent, torch.Tensor]:
        """Return diagonal Gaussian action distribution and critic value."""
        obs_norm = self.normalize_observation(obs)
        mean_a = self.action_mean_head(self.actor_trunk(obs_norm))
        std_a = torch.exp(
            torch.clamp(self.log_std_a, self.min_log_std, self.max_log_std)
        ).expand_as(mean_a)
        value = self.critic_head(self.critic_trunk(obs_norm)).squeeze(-1)
        return Independent(Normal(mean_a, std_a), 1), value


class StatefulS2PGExportWrapper(nn.Module):
    """Deterministic stateful inference module ``(obs, hidden_state, prev_action) -> (action, next_hidden_state)``."""

    def __init__(
        self,
        actor_critic: ActorCriticS2PG,
        clip_actions: float | None = None,
    ) -> None:
        """Wrap ``actor_critic`` for TorchScript or ONNX stateful export.

        Args:
            actor_critic: Trained :class:`ActorCriticS2PG` instance (deep-copied to CPU eval mode).
            clip_actions: Optional symmetric bound applied to the exported action output.
        """
        super().__init__()
        self.policy = copy.deepcopy(actor_critic).cpu().eval()
        for param in self.policy.parameters():
            param.requires_grad_(False)
        self.clip_actions = clip_actions
        self.obs_dim = self.policy.obs_dim
        self.action_dim = self.policy.action_dim
        self.latent_dim = self.policy.latent_dim

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ``(action, next_hidden_state)`` with embedded running observation stats."""
        action, next_hidden = self.policy.deterministic_act(
            obs, hidden_state, prev_action
        )
        if self.clip_actions is not None:
            action = torch.clamp(action, -self.clip_actions, self.clip_actions)
        return action, next_hidden


def export_torchscript(
    actor_critic: ActorCriticS2PG,
    output: Path | str | None = None,
    clip_actions: float | None = None,
) -> torch.jit.ScriptModule:
    """Trace a deterministic stateful TorchScript policy.

    Args:
        actor_critic: Trained :class:`ActorCriticS2PG`.
        output: Optional path where the traced TorchScript archive is saved.
        clip_actions: Optional symmetric clip applied to actions.

    Returns:
        Traced :class:`torch.jit.ScriptModule` accepting ``(obs, hidden_state, prev_action)``.
    """
    wrapper = StatefulS2PGExportWrapper(actor_critic, clip_actions=clip_actions)
    example_inputs = (
        torch.zeros((1, wrapper.obs_dim), dtype=torch.float32),
        torch.zeros((1, wrapper.latent_dim), dtype=torch.float32),
        torch.zeros((1, wrapper.action_dim), dtype=torch.float32),
    )
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        traced = torch.jit.trace(wrapper, example_inputs)
        if output is not None:
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            traced.save(str(out_path))
    return traced


def export_onnx_stateful(
    actor_critic: ActorCriticS2PG,
    output: Path | str,
    opset: int = 17,
    clip_actions: float | None = None,
) -> Path:
    """Export a deterministic stateful policy to a single-file ONNX graph.

    Args:
        actor_critic: Trained :class:`ActorCriticS2PG`.
        output: Destination ``.onnx`` path.
        opset: ONNX opset version.
        clip_actions: Optional symmetric clip applied to actions.

    Returns:
        Resolved output :class:`~pathlib.Path`.
    """
    wrapper = StatefulS2PGExportWrapper(actor_critic, clip_actions=clip_actions)
    example_inputs = (
        torch.zeros((1, wrapper.obs_dim), dtype=torch.float32),
        torch.zeros((1, wrapper.latent_dim), dtype=torch.float32),
        torch.zeros((1, wrapper.action_dim), dtype=torch.float32),
    )
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        example_inputs,
        str(out_path),
        opset_version=opset,
        input_names=["obs", "hidden_state", "prev_action"],
        output_names=["action", "next_hidden_state"],
        dynamic_axes={
            "obs": {0: "batch"},
            "hidden_state": {0: "batch"},
            "prev_action": {0: "batch"},
            "action": {0: "batch"},
            "next_hidden_state": {0: "batch"},
        },
        do_constant_folding=True,
        dynamo=False,
    )
    return out_path
