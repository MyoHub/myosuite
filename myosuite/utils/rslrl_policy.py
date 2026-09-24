# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Load an mjlab / RSL-RL checkpoint as a deterministic policy.

mjlab tasks of the basic suite are CPU twins (same obs vector, action space and
control timing as the CPU env of the same ``env_id``), so the actor of an
mjlab training run can drive the CPU env directly. This module rebuilds that
actor from ``model_*.pt`` using RSL-RL's own modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from rsl_rl.modules import MLP, EmpiricalNormalization


class RslRlPolicy(torch.nn.Module):
    """Deterministic RSL-RL actor: ``mlp(normalizer(obs))`` -> action mean.

    Args:
        mlp: Actor MLP.
        normalizer: Observation normalizer (``Identity`` when training used none).
        action_dim: Action dimension; a ``2 * action_dim`` MLP output (state-
            dependent std) is reduced to its mean slice.
    """

    def __init__(
        self, mlp: torch.nn.Module, normalizer: torch.nn.Module, action_dim: int
    ) -> None:
        super().__init__()
        self.mlp = mlp
        self.normalizer = normalizer
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.mlp(self.normalizer(obs))
        if out.shape[-1] != self.action_dim:
            out = out.reshape(*out.shape[:-1], 2, self.action_dim)[..., 0, :]
        return out

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Numpy convenience wrapper (single or batched observations)."""
        x = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        return self(x).numpy()


def _agent_params(checkpoint: Path) -> dict[str, Any]:
    """``params/agent.yaml`` written by ``scripts/train_mjlab.py`` (if present)."""
    path = checkpoint.parent / "params" / "agent.yaml"
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.unsafe_load(f) or {}


def load_rslrl_policy(checkpoint: str | Path, action_dim: int) -> RslRlPolicy:
    """Rebuild the deterministic actor of an RSL-RL checkpoint.

    Layer sizes are read from the weights; the activation comes from the run's
    ``params/agent.yaml`` (default ``elu``).

    Args:
        checkpoint: ``model_*.pt`` from an mjlab training run.
        action_dim: Action dimension of the env the policy will drive.

    Returns:
        The policy in eval mode on CPU.
    """
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor_state: dict[str, torch.Tensor] = state["actor_state_dict"]
    actor_cfg = _agent_params(checkpoint).get("actor", {})

    linear = sorted(
        (k for k in actor_state if k.startswith("mlp.") and k.endswith(".weight")),
        key=lambda k: int(k.split(".")[1]),
    )
    shapes = [tuple(actor_state[k].shape) for k in linear]
    obs_dim, out_dim = shapes[0][1], shapes[-1][0]
    hidden = [s[0] for s in shapes[:-1]]

    mlp = MLP(obs_dim, out_dim, hidden, actor_cfg.get("activation", "elu"))
    mlp.load_state_dict(
        {
            k.removeprefix("mlp."): v
            for k, v in actor_state.items()
            if k.startswith("mlp.")
        }
    )
    if any(k.startswith("obs_normalizer.") for k in actor_state):
        normalizer: torch.nn.Module = EmpiricalNormalization(obs_dim)
        normalizer.load_state_dict(
            {
                k.removeprefix("obs_normalizer."): v
                for k, v in actor_state.items()
                if k.startswith("obs_normalizer.")
            }
        )
    else:
        normalizer = torch.nn.Identity()
    return RslRlPolicy(mlp, normalizer, action_dim).eval()
