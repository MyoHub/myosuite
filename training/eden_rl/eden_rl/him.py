# Copyright 2024 InternRobotics / OpenRobotLab contributors (MIT License) and
# 2025-2026 Embodied AI Nexus (Eden Contributors, Apache-2.0).
# Modifications Copyright (c) MyoSuite Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0. See NOTICE.txt in this package.
# Upstream sources:
# - HIMLoco: Long et al. (ICLR 2024), "Hybrid Internal Model: Learning Agile Legged
#   Locomotion with Simulated Robot Response", arXiv:2312.11460,
#   https://github.com/InternRobotics/HIMLoco and https://github.com/OpenRobotLab/HomieRL (MIT).
# - Eden: https://github.com/embodied-ai-nexus/Eden (Apache-2.0),
#   packages/eden/src/eden/learning/wrappers/rsl_rl/him/{estimator,ppo}.py.
"""Hybrid Internal Model (HIM) proprioceptive history estimator & symmetry losses.

Adapted upstream components:
    :func:`sinkhorn`, :class:`HIMEstimator` SwAV + target regression architecture, and
    :func:`him_symmetry_losses` (adapted from HIMLoco / OpenHomie and Eden).

MyoSuite-specific integration:
    Generalizes HIMLoco's fixed 3-D velocity target to arbitrary ``target_dim`` so
    MyoSuite policies can regress **either** 3-D base linear velocity **or** unobserved
    physiological muscle states (e.g. 3CC-r active/fatigued motor-unit fractions,
    sarcopenia peak-force scales, or external object loads), plus single-input
    ONNX-exportable :class:`HIMActor`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn, optim

from eden_rl.simba import resolve_activation


@torch.no_grad()
def sinkhorn(scores: torch.Tensor, eps: float = 0.05, iters: int = 3) -> torch.Tensor:
    """Sinkhorn-Knopp balanced prototype assignment over ``(batch, num_prototypes)``."""
    max_per_row = scores.max(dim=-1, keepdim=True).values
    q = torch.exp((scores - max_per_row) / eps).T
    k, b = q.shape[0], q.shape[1]
    q = q / q.sum().clamp_min(1e-12)
    for _ in range(iters):
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        q = q / float(k)
        q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        q = q / float(b)
    return (q * float(b)).T


class HIMEstimator(nn.Module):
    """Self-supervised proprioceptive history encoder (MSE target regression + SwAV loss).

    Args:
        temporal_steps: Number of stacked history frames ``H``.
        num_one_step_obs: Width of a single actor observation frame ``D``.
        target_dim: Explicit privileged target width ``T`` appended to the critic frame
            (``3`` for base linear velocity, or ``num_muscles`` for 3CC-r fatigue /
            sarcopenia force scales in MyoSuite).
        num_extra_obs: Optional non-temporal features appended to ``obs_history``.
        enc_hidden_dims: Hidden widths of the history encoder MLP.
        tar_hidden_dims: Hidden widths of the one-step SwAV target encoder MLP.
        latent_dim: Width of the L2-normalized implicit dynamics latent.
        activation: Activation name (``"elu"``, ``"silu"``, ...).
        learning_rate: Learning rate for the estimator's internal Adam optimizer.
        max_grad_norm: Gradient norm clip for estimator updates.
        num_prototype: Number of SwAV unit-norm prototypes.
        temperature: Softmax temperature for SwAV swapped cross-entropy.
        target_skip_dim: Leading command entries skipped from ``next_critic_obs`` when
            feeding the SwAV target encoder.
    """

    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        *,
        target_dim: int = 3,
        num_extra_obs: int = 0,
        enc_hidden_dims: Sequence[int] = (256, 256),
        tar_hidden_dims: Sequence[int] = (256, 256),
        latent_dim: int = 32,
        activation: str = "elu",
        learning_rate: float = 1e-3,
        max_grad_norm: float = 10.0,
        num_prototype: int = 64,
        temperature: float = 3.0,
        target_skip_dim: int = 3,
    ) -> None:
        super().__init__()
        self.temporal_steps = int(temporal_steps)
        self.num_one_step_obs = int(num_one_step_obs)
        self.target_dim = int(target_dim)
        self.num_extra_obs = int(num_extra_obs)
        self.num_latent = int(latent_dim)
        self.max_grad_norm = float(max_grad_norm)
        self.temperature = float(temperature)
        self.target_skip_dim = int(target_skip_dim)

        enc_in = self.temporal_steps * self.num_one_step_obs + self.num_extra_obs
        enc_layers: list[nn.Module] = []
        for h in enc_hidden_dims:
            enc_layers.extend(
                [nn.Linear(enc_in, int(h)), resolve_activation(activation)]
            )
            enc_in = int(h)
        enc_layers.append(nn.Linear(enc_in, self.target_dim + self.num_latent))
        self.encoder = nn.Sequential(*enc_layers)

        tar_in = self.num_one_step_obs + self.target_dim - self.target_skip_dim
        if tar_in <= 0:
            raise ValueError(
                f"target_skip_dim={self.target_skip_dim} leaves non-positive target input width {tar_in}."
            )
        tar_layers: list[nn.Module] = []
        for h in tar_hidden_dims:
            tar_layers.extend(
                [nn.Linear(tar_in, int(h)), resolve_activation(activation)]
            )
            tar_in = int(h)
        tar_layers.append(nn.Linear(tar_in, self.num_latent))
        self.target = nn.Sequential(*tar_layers)

        self.proto = nn.Embedding(int(num_prototype), self.num_latent)
        self.learning_rate = float(learning_rate)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def encode(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(pred_target, z_latent)`` with gradients attached."""
        out = self.encoder(obs_history)
        pred_target = out[..., : self.target_dim]
        z = F.normalize(out[..., self.target_dim :], dim=-1, p=2)
        return pred_target, z

    def forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return detached ``(pred_target, z_latent)`` for policy conditioning."""
        pred_target, z = self.encode(obs_history.detach())
        return pred_target.detach(), z.detach()

    def update(
        self,
        obs_history: torch.Tensor,
        next_critic_obs: torch.Tensor,
        lr: float | None = None,
    ) -> tuple[float, float]:
        """Run one estimator step on ``next_critic_obs = [next_obs(D), privileged_target(T)]``."""
        if lr is not None:
            self.learning_rate = float(lr)
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate

        end = self.num_one_step_obs + self.target_dim
        true_target = next_critic_obs[:, self.num_one_step_obs : end].detach()
        swav_in = next_critic_obs[:, self.target_skip_dim : end].detach()

        pred_target, z_s = self.encode(obs_history.detach())
        z_t = F.normalize(self.target(swav_in), dim=-1, p=2)

        with torch.no_grad():
            self.proto.weight.copy_(F.normalize(self.proto.weight.data, dim=-1, p=2))

        score_s = z_s @ self.proto.weight.T
        score_t = z_t @ self.proto.weight.T
        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)
        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).sum(dim=-1).mean()
        estimation_loss = F.mse_loss(pred_target, true_target)
        loss = estimation_loss + swap_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return float(estimation_loss.item()), float(swap_loss.item())


class HIMActor(nn.Module):
    """Deployment-ready policy mapping ``obs_history`` -> ``[obs_t, pred_target, z] -> action``.

    Can be exported directly to a single-input ONNX graph because :class:`HIMEstimator`
    is folded into ``forward(obs_history)``.
    """

    def __init__(
        self,
        estimator: HIMEstimator,
        act_dim: int,
        hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.estimator = estimator
        in_dim = (
            estimator.num_one_step_obs + estimator.target_dim + estimator.num_latent
        )
        layers: list[nn.Module] = []
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, int(h)), resolve_activation(activation)])
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, int(act_dim)))
        self.actor = nn.Sequential(*layers)

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        # Most recent one-step observation sits at the front of the history vector.
        obs_t = obs_history[..., : self.estimator.num_one_step_obs]
        pred_target, z = self.estimator.encode(obs_history)
        actor_in = torch.cat([obs_t, pred_target.detach(), z.detach()], dim=-1)
        return self.actor(actor_in)


def him_symmetry_losses(
    actor_fn: Callable[[torch.Tensor], torch.Tensor],
    critic_fn: Callable[[torch.Tensor], torch.Tensor],
    obs_batch: torch.Tensor,
    critic_obs_batch: torch.Tensor,
    mirror_obs_fn: Callable[[torch.Tensor], torch.Tensor],
    mirror_critic_obs_fn: Callable[[torch.Tensor], torch.Tensor],
    mirror_act_fn: Callable[[torch.Tensor], torch.Tensor],
    symmetry_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute bilateral mirror losses ``(actor_sym_loss, critic_sym_loss)``."""
    mirrored_act = actor_fn(mirror_obs_fn(obs_batch))
    target_act = mirror_act_fn(actor_fn(obs_batch)).detach()
    actor_sym = symmetry_scale * torch.mean(
        torch.sum((mirrored_act - target_act) ** 2, dim=-1)
    )

    mirrored_val = critic_fn(mirror_critic_obs_fn(critic_obs_batch))
    target_val = critic_fn(critic_obs_batch).detach()
    critic_sym = symmetry_scale * torch.mean((mirrored_val - target_val) ** 2)
    return actor_sym, critic_sym
