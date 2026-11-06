"""ActorCritic policy for directional locomotion (BC / DAgger / PPO)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class _SiLULN(nn.Module):
    """Linear → LayerNorm → SiLU residual block."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        nn.init.orthogonal_(self.fc.weight, 2.0**0.5)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.ln(self.fc(x)))


class ActorCritic(nn.Module):
    """6-layer SiLU+LayerNorm actor-critic.

    Default shape matches the directional locomotion task:
      obs_dim=522, act_dim=354, hidden=512, layers=6.

    Args:
        obs_dim: Observation dimension.
        act_dim: Action dimension (number of muscles).
        hidden: Hidden layer width.
        layers: Total number of linear layers including the input projection.
    """

    def __init__(
        self,
        obs_dim: int = 522,
        act_dim: int = 354,
        hidden: int = 512,
        layers: int = 6,
    ) -> None:
        super().__init__()
        trunk: list[nn.Module] = [nn.Linear(obs_dim, hidden), nn.SiLU()]
        nn.init.orthogonal_(trunk[0].weight, 2.0**0.5)
        nn.init.zeros_(trunk[0].bias)
        for _ in range(layers - 1):
            trunk.append(_SiLULN(hidden))
        self.trunk = nn.Sequential(*trunk)
        self.actor_mean = nn.Linear(hidden, act_dim)
        nn.init.orthogonal_(self.actor_mean.weight, 0.01)
        nn.init.zeros_(self.actor_mean.bias)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.critic.weight, 1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (action_mean, value) for a batch of observations."""
        feat = self.trunk(obs)
        return self.actor_mean(feat), self.critic(feat).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action for one observation vector (numpy in, numpy out)."""
        t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
        return self.actor_mean(self.trunk(t)).squeeze(0).numpy()

    @classmethod
    def load(cls, path: str | Path, **kwargs: int) -> ActorCritic:
        """Load from a saved state-dict file.

        Args:
            path: Path to the `.pt` checkpoint.
            **kwargs: Override constructor arguments (obs_dim, act_dim, etc.).

        Returns:
            Loaded ActorCritic in eval mode.
        """
        net = cls(**kwargs)
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        state = (
            ckpt
            if not isinstance(ckpt, dict)
            else (ckpt.get("model_state_dict") or ckpt.get("policy") or ckpt)
        )
        net.load_state_dict(state)
        net.eval()
        return net

    @classmethod
    def load_expanded(
        cls,
        path: str | Path,
        new_obs_dim: int,
        act_dim: int = 354,
        hidden: int = 512,
        layers: int = 6,
    ) -> ActorCritic:
        """Warm-start a larger-``obs_dim`` network from a smaller checkpoint.

        Copies the pretrained input layer's weights into the leading
        ``old_obs_dim`` columns of the new (larger) input layer. New input
        dims beyond that (e.g. appended chase-tag opponent/role features) are
        zeroed rather than left at their random orthogonal init — those raw
        features (meters, m/s) are untrained and, at typical magnitude,
        random weights on them injected large noise into the trunk's first
        layer, corrupting the warm-started policy's behavior even before any
        fine-tuning. Zeroing makes the expanded network exactly reproduce the
        pretrained policy's behavior on the shared 528 dims until PPO
        fine-tuning teaches it to actually use the new features. All other
        layers (unrelated to ``obs_dim``) are copied unchanged. Used to
        fine-tune a chase-tag-aware policy starting from a pretrained
        directional-locomotion checkpoint (e.g. ``bc_directional_v2``)
        instead of training from scratch.

        The pretrained checkpoint's actual input width is read from its own
        ``trunk.0.weight`` shape rather than assumed from this class's
        default ``obs_dim`` constructor value — real directional-locomotion
        checkpoints are 528-dim, not this class's unrelated default of 522.

        Args:
            path: Path to the pretrained (smaller-``obs_dim``) `.pt` checkpoint.
            new_obs_dim: Observation dimension of the new, larger network
                (e.g. 537 for the additive chase-tag obs).
            act_dim: Action dimension (number of muscles).
            hidden: Hidden layer width.
            layers: Total number of linear layers including input projection.

        Returns:
            A new :class:`ActorCritic` with shape ``(new_obs_dim, act_dim)``,
            its leading input columns warm-started from ``path``.
        """
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        old_state = (
            ckpt
            if not isinstance(ckpt, dict)
            else (ckpt.get("model_state_dict") or ckpt.get("policy") or ckpt)
        )
        old_obs_dim = old_state["trunk.0.weight"].shape[1]
        net = cls(obs_dim=new_obs_dim, act_dim=act_dim, hidden=hidden, layers=layers)
        new_state = net.state_dict()
        for key, value in old_state.items():
            if key == "trunk.0.weight":
                new_state[key][:, :old_obs_dim] = value
                new_state[key][:, old_obs_dim:] = 0.0
            else:
                new_state[key] = value
        net.load_state_dict(new_state)
        return net


_HEADING_DIRS = torch.tensor(
    [
        [1.0, 0.0],
        [0.70710678, 0.70710678],
        [0.0, 1.0],
        [-0.70710678, 0.70710678],
        [-1.0, 0.0],
        [-0.70710678, -0.70710678],
        [0.0, -1.0],
        [0.70710678, -0.70710678],
    ],
    dtype=torch.float32,
)


class HeadingTablePolicy(nn.Module):
    """528-d walker: ActorCritic mean plus a left-leg bias per heading octant."""

    def __init__(self, ac: ActorCritic, table: torch.Tensor) -> None:
        super().__init__()
        self.trunk = ac.trunk
        self.actor_mean = ac.actor_mean
        self.register_buffer("dirs", _HEADING_DIRS.clone())
        self.register_buffer("table", torch.as_tensor(table, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Do not pre-clamp the BC mean; the env clips actions.
        base = self.actor_mean(self.trunk(obs))
        heading = obs[:, 520:522]
        idx = torch.argmax(heading @ self.dirs.T, dim=-1)
        out = base.clone()
        out[:, 314:354] = torch.clamp(out[:, 314:354] + self.table[idx], 0.0, 1.0)
        return out

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
        t = t.to(self.table.device)
        return self.forward(t).squeeze(0).cpu().numpy()

    @classmethod
    def load(cls, path: str | Path) -> HeadingTablePolicy:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        if not (isinstance(ckpt, dict) and ckpt.get("kind") == "heading_table"):
            raise ValueError(f"{path} is not a heading-table checkpoint")
        ac = ActorCritic(obs_dim=528, act_dim=354)
        ac.load_state_dict(ckpt["actor"])
        return cls(ac, ckpt["table"]).eval()


def load_directional_policy(path: str | Path) -> nn.Module:
    """Load a 528-d walker: heading-table package or a raw ActorCritic state-dict."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and ckpt.get("kind") == "heading_table":
        return HeadingTablePolicy.load(path)
    return ActorCritic.load(path, obs_dim=528, act_dim=354)
