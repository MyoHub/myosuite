# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""A ChaseTag player that starts as the tutorial 5.4 directional walker.

:class:`FbvsDirectionalAdapter` turns an FBVs player's 546-d observation into
the 528-d directional observation of ``_directional_obs``, with the heading
pointed at the opponent (chaser) or away from it (runner), as
``chasetag_heading_directional_obs``. :class:`DirectionalWalkerActor` is an
RSL-RL actor made of that fixed adapter and the 5.4 network (6-layer
SiLU+LayerNorm, :class:`~myosuite.envs.myo.tasks.mimic.policy.ActorCritic`),
optionally loaded from a 5.4 checkpoint, so PPO fine-tunes one 546 -> 354
model that starts out walking. The adapter sits in the actor's observation
normalizer slot, so RSL-RL's TorchScript/ONNX copies (self-play snapshots,
exports) carry it.
"""

from __future__ import annotations

from typing import Any

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from torch import nn

from myosuite.envs.myo.tasks.mimic.policy import ActorCritic

FBVS_OBS_DIM = 546
DIRECTIONAL_OBS_DIM = 528


class FbvsDirectionalAdapter(nn.Module):
    """FBVs player observation (546) -> 5.4 directional observation (528); no parameters.

    FBVs layout: ``qpos`` 0:89 (root 0:7), ``qvel`` 89:177 (root 89:95), ``act``
    177:531, own pelvis 531:537, opponent ``[rel_pos, rel_vel, dist]`` 537:544,
    role 544:546 (``[1, 0]`` chaser).
    """

    in_dim = FBVS_OBS_DIM

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        w, x, y, z = obs[..., 3], obs[..., 4], obs[..., 5], obs[..., 6]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        c, s = torch.cos(-yaw), torch.sin(-yaw)
        vx, vy, vz = obs[..., 89], obs[..., 90], obs[..., 91]
        wx, wy, wz = obs[..., 92], obs[..., 93], obs[..., 94]
        # Heading toward the opponent as chaser, away from it as runner.
        sign = torch.where(obs[..., 544] > 0.5, 1.0, -1.0)
        theta = torch.atan2(sign * obs[..., 538], sign * obs[..., 537])
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
        tail = torch.stack(
            [
                c * vx - s * vy,  # root_vel_body
                s * vx + c * vy,
                torch.cos(theta),  # heading_cmd
                torch.sin(theta),
                roll,  # orientation (as _directional_obs)
                pitch,
                c * wx - s * wy,
                s * wx + c * wy,
                wz,
                vz,
            ],
            dim=-1,
        )
        return torch.cat(
            [obs[..., 7:89], obs[..., 95:177], obs[..., 177:531], tail], dim=-1
        )


class WalkerNet(nn.Module):
    """The 5.4 actor: SiLU+LayerNorm trunk and mean head (``ActorCritic`` keys)."""

    def __init__(self, act_dim: int = 354, hidden: int = 512, layers: int = 6) -> None:
        super().__init__()
        net = ActorCritic(DIRECTIONAL_OBS_DIM, act_dim, hidden, layers)
        self.trunk, self.actor_mean = net.trunk, net.actor_mean

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_mean(self.trunk(obs))

    def load_walker(self, checkpoint: str) -> None:
        """Copy the actor of a 5.4 ``ActorCritic`` checkpoint (critic and std dropped)."""
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        self.load_state_dict(
            {k: v for k, v in state.items() if k.startswith(("trunk.", "actor_mean."))}
        )


class DirectionalWalkerActor(MLPModel):
    """RSL-RL actor ``walker(adapter(obs))``: one 546 -> 354 ChaseTag player.

    Args:
        obs: Observation dictionary.
        obs_groups: Observation sets to groups.
        obs_set: The set this model reads (``"actor"``).
        output_dim: Muscles (354).
        hidden_dims: Unused (the 5.4 network is fixed: *hidden* x *layers*).
        activation: Unused.
        obs_normalization: Must be ``False`` (the 5.4 network takes raw inputs).
        distribution_cfg: Output distribution (Gaussian noise for PPO).
        walker_checkpoint: 5.4 checkpoint to start from (``""``: random init).
        hidden: Width of the 5.4 network.
        layers: Linear layers of its trunk.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (),
        activation: str = "silu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        walker_checkpoint: str = "",
        hidden: int = 512,
        layers: int = 6,
    ) -> None:
        del hidden_dims, activation
        if obs_normalization:
            raise ValueError("DirectionalWalkerActor reads raw observations.")
        super().__init__(
            obs, obs_groups, obs_set, output_dim, (8,), "elu", False, distribution_cfg
        )
        if self.obs_dim != FBVS_OBS_DIM:
            raise ValueError(f"Expected the {FBVS_OBS_DIM}-d FBVs observation.")
        self.obs_normalizer = FbvsDirectionalAdapter()
        self.mlp = WalkerNet(output_dim, hidden, layers)
        if walker_checkpoint:
            self.mlp.load_walker(walker_checkpoint)

    @staticmethod
    def deterministic_modules(
        actor_state: dict[str, torch.Tensor], actor_cfg: dict[str, Any]
    ) -> tuple[nn.Module, nn.Module]:
        """``(net, adapter)`` of a trained checkpoint, for ``load_rslrl_policy``."""
        out_dim = actor_state["mlp.actor_mean.weight"].shape[0]
        net = WalkerNet(
            out_dim, actor_cfg.get("hidden", 512), actor_cfg.get("layers", 6)
        )
        net.load_state_dict(
            {
                k.removeprefix("mlp."): v
                for k, v in actor_state.items()
                if k.startswith("mlp.")
            }
        )
        return net, FbvsDirectionalAdapter()
