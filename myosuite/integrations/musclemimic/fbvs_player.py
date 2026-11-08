# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Turn a tutorial 5.4 directional walker into a ChaseTag FBVs player (546 -> 354).

:class:`FbvsDirectionalAdapter` maps an FBVs player's 546-d observation to the
528-d directional observation (``_directional_obs``) with the heading pointed
at the opponent (chaser) or away from it (runner), as
:func:`~myosuite.envs.myo.tasks.mimic.chasetag_obs.chasetag_heading_directional_obs`.
:func:`export_fbvs_onnx` wraps a walker checkpoint with it and writes the ONNX
file the MyoChallenge server runs. Needs only torch (no rsl_rl).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from myosuite.envs.myo.tasks.mimic.policy import (
    ActorCritic,
    HeadingTablePolicy,
    load_directional_policy,
)
from myosuite.physics.quat import quat_yaw

FBVS_OBS_DIM = 546
DIRECTIONAL_OBS_DIM = 528
ACT_DIM = 354


class FbvsDirectionalAdapter(nn.Module):
    """FBVs player observation (546) -> directional walker observation (528).

    FBVs layout: ``qpos`` 0:89 (root 0:7), ``qvel`` 89:177 (root 89:95), ``act``
    177:531, own pelvis 531:537, opponent ``[rel_pos, rel_vel, dist]`` 537:544,
    role 544:546 (``[1, 0]`` chaser, ``[0, 1]`` runner). The heading is the
    world-frame direction to the opponent, flipped for the runner.
    """

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        qpos, qvel = obs[:, 0:89], obs[:, 89:177]
        w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
        yaw = quat_yaw(torch, qpos[:, 3:7])
        c, s = torch.cos(-yaw), torch.sin(-yaw)
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0))
        vx, vy, vz = qvel[:, 0], qvel[:, 1], qvel[:, 2]
        wx, wy, wz = qvel[:, 3], qvel[:, 4], qvel[:, 5]
        delta = obs[:, 537:539]
        delta = torch.where(obs[:, 545:546] > 0.5, -delta, delta)  # runner flees
        heading = delta / torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
        tail = torch.stack(
            [
                c * vx - s * vy,  # root velocity in the yaw frame
                s * vx + c * vy,
                heading[:, 0],
                heading[:, 1],
                roll,
                pitch,
                c * wx - s * wy,  # angular velocity in the yaw frame
                s * wx + c * wy,
                wz,
                vz,
            ],
            dim=-1,
        )
        return torch.cat([qpos[:, 7:], qvel[:, 6:], obs[:, 177:531], tail], dim=-1)


class _ActorMean(nn.Module):
    """Deterministic action of an :class:`ActorCritic` (its forward also returns a value)."""

    def __init__(self, ac: ActorCritic) -> None:
        super().__init__()
        self.trunk = ac.trunk
        self.actor_mean = ac.actor_mean

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_mean(self.trunk(obs))


class FbvsPlayer(nn.Module):
    """546-d FBVs observation -> 354 muscle actions in ``[0, 1]``.

    Args:
        walker: Network on the 528-d directional observation, or on the 546-d
            FBVs observation itself when ``adapter`` is ``None``.
        adapter: Observation adapter applied before ``walker``.
    """

    def __init__(self, walker: nn.Module, adapter: nn.Module | None) -> None:
        super().__init__()
        self.walker = walker
        self.adapter = adapter

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs if self.adapter is None else self.adapter(obs)
        return torch.clamp(self.walker(x), 0.0, 1.0)


def load_fbvs_player(checkpoint: str | Path) -> FbvsPlayer:
    """Build an FBVs player from a 5.4 walker or an FBVs-trained checkpoint.

    Args:
        checkpoint: A heading-table package, or an :class:`ActorCritic` state
            dict (bare, or under ``model_state_dict`` / ``policy``) on the 528-d
            directional observation (wrapped with the adapter) or on the 546-d
            FBVs observation (used as is).

    Returns:
        The player in eval mode.

    Raises:
        ValueError: If the checkpoint holds no ``ActorCritic`` weights or its
            observation size is neither 528 nor 546.
    """
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and state.get("kind") == "heading_table":
        return FbvsPlayer(
            HeadingTablePolicy.load(checkpoint), FbvsDirectionalAdapter()
        ).eval()
    weights = state
    if isinstance(state, dict):
        weights = state.get("model_state_dict") or state.get("policy") or state
    if not isinstance(weights, dict) or "trunk.0.weight" not in weights:
        keys = sorted(weights)[:10] if isinstance(weights, dict) else type(weights)
        raise ValueError(
            f"{checkpoint} is not a heading-table package or ActorCritic state dict "
            f"(no 'trunk.0.weight'; keys: {keys})"
        )
    obs_dim = int(weights["trunk.0.weight"].shape[1])
    if obs_dim == DIRECTIONAL_OBS_DIM:
        return FbvsPlayer(
            _ActorMean(load_directional_policy(checkpoint)), FbvsDirectionalAdapter()
        ).eval()
    if obs_dim == FBVS_OBS_DIM:
        walker = _ActorMean(
            ActorCritic.load(checkpoint, obs_dim=FBVS_OBS_DIM, act_dim=ACT_DIM)
        )
        return FbvsPlayer(walker, None).eval()
    raise ValueError(
        f"Expected an obs_dim of {DIRECTIONAL_OBS_DIM} or {FBVS_OBS_DIM}, got {obs_dim}"
    )


def export_fbvs_onnx(
    checkpoint: str | Path, onnx_path: str | Path, opset: int = 17
) -> Path:
    """Export :func:`load_fbvs_player` as ONNX: ``obs [batch, 546] -> action [batch, 354]``.

    Args:
        checkpoint: Walker checkpoint (see :func:`load_fbvs_player`).
        onnx_path: Output ``.onnx`` file.
        opset: ONNX opset version.

    Returns:
        The written path.
    """
    onnx_path = Path(onnx_path)
    torch.onnx.export(
        load_fbvs_player(checkpoint),
        torch.zeros(1, FBVS_OBS_DIM),
        str(onnx_path),
        opset_version=opset,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,
    )
    return onnx_path


__all__ = [
    "ACT_DIM",
    "DIRECTIONAL_OBS_DIM",
    "FBVS_OBS_DIM",
    "FbvsDirectionalAdapter",
    "FbvsPlayer",
    "export_fbvs_onnx",
    "load_fbvs_player",
]
