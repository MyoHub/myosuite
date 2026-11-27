# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Observation normalizer, Actor-Critic networks, and ONNX export for SHAC."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn


class RunningMeanStd(nn.Module):
    """Differentiable-passthrough Welford running observation normalizer."""

    def __init__(self, shape: tuple[int, ...], eps: float = 1e-5) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(eps, dtype=torch.float32))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        flat = x.detach().reshape(-1, x.shape[-1]).to(dtype=torch.float32)
        b_mean = flat.mean(dim=0)
        b_var = flat.var(dim=0, unbiased=False)
        b_count = float(flat.shape[0])
        delta = b_mean - self.mean
        tot = self.count + b_count
        new_mean = self.mean + delta * (b_count / tot)
        m_a = self.var * self.count
        m_b = b_var * b_count
        m2 = m_a + m_b + delta * delta * (self.count * b_count / tot)
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / tot)
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor, *, update: bool = False) -> torch.Tensor:
        if update:
            self.update(x)
        normed = (x - self.mean) / torch.sqrt(self.var + 1e-5)
        return torch.clamp(normed, -10.0, 10.0)


def _make_mlp(
    in_dim: int, hidden_dims: tuple[int, ...], out_dim: int, *, layernorm: bool = True
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.ELU())
        prev = h
    last = nn.Linear(prev, out_dim)
    nn.init.orthogonal_(last.weight, gain=0.01 if out_dim > 1 else 1.0)
    nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Reparameterized Gaussian actor and scalar value critic for SHAC."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        init_std: float = 0.4,
        std_range: tuple[float, float] = (0.05, 1.0),
    ) -> None:
        super().__init__()
        self.obs_norm = RunningMeanStd((obs_dim,))
        self.actor_net = _make_mlp(obs_dim, hidden_dims, act_dim)
        self.critic_net = _make_mlp(obs_dim, hidden_dims, 1)
        self.std_param = nn.Parameter(torch.full((act_dim,), float(init_std)))
        self.std_range = std_range

    def action_std(self) -> torch.Tensor:
        lo, hi = self.std_range
        return (
            self.std_param
            + (torch.clamp(self.std_param, lo, hi) - self.std_param).detach()
        )

    def act_reparam(
        self, obs: torch.Tensor, *, update_norm: bool = False
    ) -> torch.Tensor:
        norm_obs = self.obs_norm(obs, update=update_norm)
        mu = self.actor_net(norm_obs)
        return mu + self.action_std() * torch.randn_like(mu)

    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_net(self.obs_norm(obs, update=False))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_net(self.obs_norm(obs, update=False)).squeeze(-1)


class _FoldedShacActor(nn.Module):
    def __init__(self, model: ActorCritic) -> None:
        super().__init__()
        self.obs_norm = copy.deepcopy(model.obs_norm).cpu().eval()
        self.actor_net = copy.deepcopy(model.actor_net).cpu().eval()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.actor_net(self.obs_norm(obs, update=False)), -1.0, 1.0)


def export_shac_onnx(
    model: ActorCritic, output: str | Path, *, opset: int = 17
) -> Path:
    """Export deterministic SHAC actor with folded normalizer to a single .onnx file."""
    from myosuite.utils.export_onnx import _export_policy

    out_path = Path(output)
    obs_dim = int(model.obs_norm.mean.shape[-1])
    _export_policy(_FoldedShacActor(model), obs_dim, out_path, opset)
    return out_path
