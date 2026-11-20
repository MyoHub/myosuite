# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Adversarial motion-prior and GAIL transition discriminators in PyTorch.

Implements state-transition ``(s_t, s_{t+1})`` adversarial discriminators and
reward blending following Peng et al. (SIGGRAPH 2021, AMP), Ho & Ermon
(NeurIPS 2016, GAIL), and the JAX design of Firas Al-Hafez et al.
(``robfiras/loco-mujoco``, MIT License; see ``LICENSE-loco-mujoco.txt``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

DiscriminatorMode = Literal["amp", "gail"]
ActivationName = Literal["relu", "tanh", "elu"]


def _make_activation(name: ActivationName) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()
    raise ValueError(
        f"Unsupported activation {name!r}; expected 'relu', 'tanh', or 'elu'."
    )


class RunningMeanStd(nn.Module):
    """Buffer-backed parallel Welford running mean and variance normalizer."""

    mean: torch.Tensor
    var: torch.Tensor
    count: torch.Tensor

    def __init__(self, shape: int | tuple[int, ...], eps: float = 1e-5) -> None:
        """Initialize running statistics buffers.

        Args:
            shape: Feature shape (int or tuple of ints).
            eps: Numerical floor added to variance before square root.
        """
        super().__init__()
        norm_shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.shape = norm_shape
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(norm_shape, dtype=torch.float32))
        self.register_buffer("var", torch.ones(norm_shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float32))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update running mean and variance from a batch of vectors.

        Args:
            x: Input tensor whose trailing dimensions match ``self.shape``.
        """
        flat = (
            x.detach()
            .to(dtype=torch.float32, device=self.mean.device)
            .reshape(-1, *self.shape)
        )
        batch_count = float(flat.shape[0])
        if batch_count == 0.0:
            return
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * (self.count * batch_count / total_count)
        self.mean.copy_(new_mean)
        self.var.copy_(torch.clamp(m2 / total_count, min=1e-8))
        self.count.fill_(float(total_count.item()))

    def normalize(self, x: torch.Tensor, clip: float | None = 10.0) -> torch.Tensor:
        """Standardize ``x`` using current running mean and variance.

        Args:
            x: Input tensor of trailing shape ``self.shape``.
            clip: Optional symmetric clamp bound on normalized values.

        Returns:
            Normalized float32 tensor.
        """
        out = (x - self.mean) / torch.sqrt(self.var + self.eps)
        if clip is not None:
            out = torch.clamp(out, -clip, clip)
        return out

    def forward(self, x: torch.Tensor, clip: float | None = 10.0) -> torch.Tensor:
        """Alias of :meth:`normalize` for ``nn.Module`` composition."""
        return self.normalize(x, clip=clip)


@dataclass(frozen=True)
class DiscriminatorLossStats:
    """Scalar diagnostics returned by a discriminator optimization step."""

    total_loss: float
    gan_loss: float
    grad_penalty: float
    weighted_grad_penalty: float
    logit_reg_loss: float
    expert_logit_mean: float
    policy_logit_mean: float


class MotionDiscriminator(nn.Module):
    """State-transition discriminator ``D(s_t, s_{t+1})`` supporting AMP and GAIL."""

    feature_idx: torch.Tensor | None

    def __init__(
        self,
        obs_dim: int,
        mode: DiscriminatorMode = "amp",
        feature_indices: Sequence[int] | None = None,
        hidden_dims: tuple[int, ...] = (256, 256),
        activation: ActivationName = "relu",
        normalize_inputs: bool = True,
        grad_penalty_weight: float = 5.0,
        logit_reg_weight: float = 0.05,
        reward_eps: float = 1e-7,
    ) -> None:
        """Initialize the transition discriminator.

        Args:
            obs_dim: Single-step observation dimension.
            mode: ``"amp"`` (least-squares GAN + bounded quadratic style reward)
                or ``"gail"`` (BCE-with-logits + softplus style reward).
            feature_indices: Optional indices into ``obs_t`` and ``obs_{t+1}``
                selecting motion-prior features (e.g., excluding task goal dims).
                Defaults to ``None`` (full ``(obs_t, obs_{t+1})`` concatenation).
            hidden_dims: MLP trunk layer widths.
            activation: Hidden activation function name.
            normalize_inputs: Normalize concatenated transitions with ``RunningMeanStd``.
            grad_penalty_weight: Weight ``w_gp`` on expert gradient penalty
                ``0.5 * E[||grad_x D(x_expert)||^2]``.
            logit_reg_weight: L2 regularization weight on output linear head weights.
            reward_eps: Numerical epsilon inside GAIL log-reward ``-log(1 - sigmoid(D) + eps)``.
        """
        super().__init__()
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}.")
        if mode not in ("amp", "gail"):
            raise ValueError(f"mode must be 'amp' or 'gail', got {mode!r}.")
        self.obs_dim = int(obs_dim)
        self.mode: DiscriminatorMode = mode
        self.normalize_inputs = bool(normalize_inputs)
        self.grad_penalty_weight = float(grad_penalty_weight)
        self.logit_reg_weight = float(logit_reg_weight)
        self.reward_eps = float(reward_eps)

        if feature_indices is not None:
            idx_list = [int(i) for i in feature_indices]
            if not idx_list:
                raise ValueError("feature_indices must not be empty when provided.")
            if min(idx_list) < 0 or max(idx_list) >= self.obs_dim:
                raise ValueError(
                    f"feature_indices out of bounds for obs_dim={self.obs_dim}: {idx_list}"
                )
            self.register_buffer(
                "feature_idx", torch.tensor(idx_list, dtype=torch.long)
            )
            feat_dim = len(idx_list)
        else:
            self.feature_idx = None
            feat_dim = self.obs_dim

        self.transition_dim = 2 * feat_dim
        self.input_rms = RunningMeanStd(self.transition_dim)

        layers: list[nn.Module] = []
        in_dim = self.transition_dim
        for width in hidden_dims:
            linear = nn.Linear(in_dim, int(width))
            nn.init.orthogonal_(linear.weight, gain=1.0)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(_make_activation(activation))
            in_dim = int(width)
        self.trunk = nn.Sequential(*layers)
        self.logit_head = nn.Linear(in_dim, 1)
        nn.init.orthogonal_(self.logit_head.weight, gain=1.0)
        nn.init.zeros_(self.logit_head.bias)

    def extract_transition(
        self, obs: torch.Tensor, next_obs: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate ``(obs_t, obs_{t+1})`` after optional feature slicing.

        Args:
            obs: Current observation tensor ``(..., obs_dim)``.
            next_obs: Next observation tensor ``(..., obs_dim)``.

        Returns:
            Transition feature tensor ``(..., transition_dim)``.
        """
        if self.feature_idx is not None:
            s_t = torch.index_select(obs, dim=-1, index=self.feature_idx)
            s_next = torch.index_select(next_obs, dim=-1, index=self.feature_idx)
        else:
            s_t = obs
            s_next = next_obs
        return torch.cat([s_t, s_next], dim=-1)

    @torch.no_grad()
    def update_normalizer(self, obs: torch.Tensor, next_obs: torch.Tensor) -> None:
        """Update transition running statistics from ``(obs, next_obs)``."""
        if self.normalize_inputs:
            self.input_rms.update(self.extract_transition(obs, next_obs))

    def forward_features(self, transitions: torch.Tensor) -> torch.Tensor:
        """Evaluate raw scalar logits on already-prepared transition vectors.

        Args:
            transitions: Tensor of shape ``(..., transition_dim)``.

        Returns:
            Scalar logits ``(...,)``.
        """
        return self.logit_head(self.trunk(transitions)).squeeze(-1)

    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        """Evaluate discriminator logits ``D(s_t, s_{t+1})``.

        Args:
            obs: Current observations ``(..., obs_dim)``.
            next_obs: Next observations ``(..., obs_dim)``.

        Returns:
            Discriminator logits ``(...,)``.
        """
        x = self.extract_transition(obs, next_obs)
        if self.normalize_inputs:
            x = self.input_rms.normalize(x)
        return self.forward_features(x)

    def compute_loss(
        self,
        expert_obs: torch.Tensor,
        expert_next_obs: torch.Tensor,
        policy_obs: torch.Tensor,
        policy_next_obs: torch.Tensor,
        *,
        update_rms: bool = True,
    ) -> tuple[torch.Tensor, DiscriminatorLossStats]:
        """Compute AMP or GAIL discriminator loss with logit decay and gradient penalty.

        Args:
            expert_obs: Expert states ``s_t`` of shape ``(B, obs_dim)``.
            expert_next_obs: Expert next states ``s_{t+1}`` of shape ``(B, obs_dim)``.
            policy_obs: Policy states ``s_t`` of shape ``(B, obs_dim)``.
            policy_next_obs: Policy next states ``s_{t+1}`` of shape ``(B, obs_dim)``.
            update_rms: Update running input normalizer on the combined batch.

        Returns:
            Tuple ``(total_loss, stats)``.
        """
        raw_exp = self.extract_transition(expert_obs, expert_next_obs).detach()
        raw_pol = self.extract_transition(policy_obs, policy_next_obs).detach()
        if update_rms and self.normalize_inputs:
            self.input_rms.update(torch.cat([raw_exp, raw_pol], dim=0))

        x_exp = (
            self.input_rms.normalize(raw_exp).detach()
            if self.normalize_inputs
            else raw_exp
        )
        x_pol = (
            self.input_rms.normalize(raw_pol).detach()
            if self.normalize_inputs
            else raw_pol
        )
        x_exp.requires_grad_(True)

        exp_logits = self.forward_features(x_exp)
        pol_logits = self.forward_features(x_pol)

        if self.mode == "amp":
            pos_loss = 0.5 * torch.mean((exp_logits - 1.0).square())
            neg_loss = 0.5 * torch.mean((pol_logits + 1.0).square())
            gan_loss = pos_loss + neg_loss
        else:
            pos_loss = F.binary_cross_entropy_with_logits(
                exp_logits, torch.ones_like(exp_logits)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                pol_logits, torch.zeros_like(pol_logits)
            )
            gan_loss = 0.5 * (pos_loss + neg_loss)

        grad = torch.autograd.grad(
            outputs=exp_logits.sum(),
            inputs=x_exp,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_penalty = 0.5 * torch.mean(torch.sum(grad.square(), dim=-1))
        weighted_gp = self.grad_penalty_weight * grad_penalty
        logit_reg = self.logit_reg_weight * torch.sum(self.logit_head.weight.square())
        total_loss = gan_loss + logit_reg + weighted_gp

        stats = DiscriminatorLossStats(
            total_loss=float(total_loss.detach().item()),
            gan_loss=float(gan_loss.detach().item()),
            grad_penalty=float(grad_penalty.detach().item()),
            weighted_grad_penalty=float(weighted_gp.detach().item()),
            logit_reg_loss=float(logit_reg.detach().item()),
            expert_logit_mean=float(exp_logits.detach().mean().item()),
            policy_logit_mean=float(pol_logits.detach().mean().item()),
        )
        return total_loss, stats

    @torch.no_grad()
    def compute_style_reward(
        self, obs: torch.Tensor, next_obs: torch.Tensor
    ) -> torch.Tensor:
        """Compute adversarial motion-prior style reward for transitions ``(obs, next_obs)``.

        In ``"amp"`` mode returns ``clamp(1 - 0.25 * (D(x) - 1)^2, 0, 1)`` in ``[0, 1]``.
        In ``"gail"`` mode returns ``-log(1 - sigmoid(D(x)) + eps)`` (stabilized via
        ``sigmoid(-D(x))``).

        Args:
            obs: Current observations ``(..., obs_dim)``.
            next_obs: Next observations ``(..., obs_dim)``.

        Returns:
            Non-negative style reward tensor ``(...,)``.
        """
        logits = self.forward(obs, next_obs)
        if self.mode == "amp":
            return torch.clamp(1.0 - 0.25 * (logits - 1.0).square(), min=0.0, max=1.0)
        return -torch.log(torch.sigmoid(-logits) + self.reward_eps)

    @staticmethod
    def combine_rewards(
        env_reward: torch.Tensor,
        style_reward: torch.Tensor,
        proportion_env_reward: float = 0.5,
    ) -> torch.Tensor:
        """Blend environment task reward and adversarial style reward.

        Matches LocoMuJoCo's ``proportion_env_reward`` blending::

            (1.0 - proportion_env_reward) * style_reward + proportion_env_reward * env_reward

        Args:
            env_reward: Unmodified environment reward tensor.
            style_reward: Motion-prior style reward tensor of matching shape.
            proportion_env_reward: Weight in ``[0, 1]`` placed on ``env_reward``.

        Returns:
            Blended reward tensor.
        """
        if not 0.0 <= proportion_env_reward <= 1.0:
            raise ValueError(
                f"proportion_env_reward must lie in [0, 1], got {proportion_env_reward}."
            )
        return (
            1.0 - proportion_env_reward
        ) * style_reward + proportion_env_reward * env_reward
