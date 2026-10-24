# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Deterministic MLP actor and critic, state-dict compatible with rsl_rl's ``MLPModel``."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class Normalizer(nn.Module):
    """Running mean / std observation normaliser (rsl_rl ``EmpiricalNormalization`` layout)."""

    def __init__(self, dim: int, eps: float = 1e-2) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("_mean", torch.zeros(1, dim))
        self.register_buffer("_var", torch.ones(1, dim))
        self.register_buffer("_std", torch.ones(1, dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + self.eps)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Merge the batch statistics of ``x`` ``(..., dim)``."""
        x = x.reshape(-1, x.shape[-1])
        n = x.shape[0]
        total = self.count + n
        rate = n / total
        mean_b = x.mean(0, keepdim=True)
        var_b = x.var(0, unbiased=False, keepdim=True)
        delta = mean_b - self._mean
        self._mean += rate * delta
        self._var += rate * (var_b - self._var + delta * (mean_b - self._mean))
        self._std = self._var.sqrt()
        self.count = total


def mlp(dims: list[int], act: type[nn.Module] = nn.ELU) -> nn.Sequential:
    """``Linear``/activation stack without an activation after the last layer."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Normalised-observation MLP producing raw env actions.

    Args:
        obs_dim: Observation width.
        act_dim: Action width.
        hidden: Hidden layer widths.
    """

    def __init__(
        self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (256, 256, 128)
    ) -> None:
        super().__init__()
        self.obs_normalizer = Normalizer(obs_dim)
        self.mlp = mlp([obs_dim, *hidden, act_dim])
        # rsl_rl stores a Gaussian std next to the mean network; kept for checkpoint compatibility.
        self.distribution_std = nn.Parameter(torch.ones(act_dim), requires_grad=False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.obs_normalizer(obs))

    @classmethod
    def from_rsl_rl(cls, path: str | Path, device: str | torch.device = "cpu") -> Actor:
        """Actor of an rsl_rl ``OnPolicyRunner`` checkpoint (scalar std)."""
        sd = torch.load(path, map_location=device, weights_only=False)[
            "actor_state_dict"
        ]
        lin = sorted(
            int(k.split(".")[1])
            for k in sd
            if k.startswith("mlp.") and k.endswith(".weight")
        )
        dims = [sd[f"mlp.{lin[0]}.weight"].shape[1]] + [
            sd[f"mlp.{i}.weight"].shape[0] for i in lin
        ]
        actor = cls(dims[0], dims[-1], tuple(dims[1:-1])).to(device)
        actor.load_state_dict(_from_rsl_rl_keys(sd))
        return actor

    def rsl_rl_state_dict(self) -> dict[str, torch.Tensor]:
        """State dict in rsl_rl ``MLPModel`` key layout (for the existing playback tools)."""
        return {
            ("distribution.std_param" if k == "distribution_std" else k): v
            for k, v in self.state_dict().items()
        }


def _from_rsl_rl_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        ("distribution_std" if k == "distribution.std_param" else k): v
        for k, v in sd.items()
    }


class Critic(nn.Module):
    """Value MLP on normalised features, bounded to ``[0, v_max]``.

    With rewards in ``[0, 1]`` no return exceeds ``1 / (1 - gamma)``; the bound keeps
    the actor's bootstrap term from chasing values the critic extrapolates off-data.
    """

    def __init__(
        self, in_dim: int, v_max: float, hidden: tuple[int, ...] = (512, 512, 256)
    ) -> None:
        super().__init__()
        self.v_max = v_max
        self.normalizer = Normalizer(in_dim)
        self.mlp = mlp([in_dim, *hidden, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v_max * torch.sigmoid(self.mlp(self.normalizer(x)).squeeze(-1))
