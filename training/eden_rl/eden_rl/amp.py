# Copyright 2025-2026 Embodied AI Nexus (Eden Contributors, Apache-2.0).
# Modifications Copyright (c) MyoSuite Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0. See NOTICE.txt in this package.
# Upstream sources:
# - AMP: Peng et al. (SIGGRAPH 2021), "AMP: Adversarial Motion Priors for Stylized
#   Physics-Based Character Control", ACM Trans. Graph. 40(4), arXiv:2104.02180, and
#   https://github.com/TeleHuman/humanoid_skateboarding.
# - Eden: https://github.com/embodied-ai-nexus/Eden (Apache-2.0),
#   packages/eden/src/eden/learning/wrappers/rsl_rl/amp_ppo/{amp,ppo}.py.
"""Adversarial Motion Prior (AMP) sampler, normalizer, discriminator, and RSL-RL wrapper.

Adapted upstream components:
    :class:`AmpMotionSampler` window interpolation, :class:`AmpReplayBuffer`,
    :class:`FrameNormalizer`, :class:`AmpDiscriminator` (spectral-norm LSGAN + WGAN-GP),
    and :class:`AmpPPO` (adapted from AMP / HUSKY and Eden).

MyoSuite-specific integration:
    Extends :class:`AmpMotionSampler` to ingest ``myoMimic-v0`` refitted ``.npz``
    trajectory clips (``qpos`` slices + ``fps``/``dt``) and in-memory tensor sequences
    directly without ``eden.utils.assets``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

AMP_PPO_CLASS_NAME = "eden_rl.amp:AmpPPO"
_AMP_CHECKPOINT_KEYS = (
    "amp_discriminator_state_dict",
    "amp_optimizer_state_dict",
    "amp_normalizer_state_dict",
)


def _load_clip_list(
    source: str | Path | Sequence[str | Path | np.ndarray | torch.Tensor],
    *,
    qpos_slice: slice | None = None,
    default_fps: float = 50.0,
) -> tuple[list[np.ndarray], float]:
    """Load motion arrays ``(T_i, D)`` from a bundle, ``myoMimic`` ``.npz``, directory, or sequence."""
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_dir():
            files = sorted(p.glob("*.npz"))
            if not files:
                raise FileNotFoundError(f"No .npz motion clips found in directory {p}.")
            clips: list[np.ndarray] = []
            fps = default_fps
            for f in files:
                sub_clips, fps = _load_clip_list(
                    f, qpos_slice=qpos_slice, default_fps=fps
                )
                clips.extend(sub_clips)
            return clips, fps

        with np.load(p, allow_pickle=False) as data:
            fps = (
                float(data["fps"])
                if "fps" in data
                else (1.0 / float(data["dt"]) if "dt" in data else default_fps)
            )
            if "num_motions" in data:
                raw = [
                    np.asarray(data[f"motion_{i}"], dtype=np.float32)
                    for i in range(int(data["num_motions"]))
                ]
            elif "qpos" in data:
                q = np.asarray(data["qpos"], dtype=np.float32)
                if qpos_slice is not None:
                    q = q[:, qpos_slice]
                raw = [q]
            elif "joint_pos" in data:
                raw = [np.asarray(data["joint_pos"], dtype=np.float32)]
            else:
                raise KeyError(
                    f"Unsupported motion .npz {p}; expected 'num_motions'/'motion_i', 'qpos', or 'joint_pos' "
                    f"(found {sorted(data.files)})."
                )
            return raw, fps

    clips_out: list[np.ndarray] = []
    fps_out = default_fps
    for item in source:
        if isinstance(item, (str, Path)):
            sub, fps_out = _load_clip_list(
                item, qpos_slice=qpos_slice, default_fps=fps_out
            )
            clips_out.extend(sub)
        elif isinstance(item, torch.Tensor):
            arr = item.detach().cpu().numpy().astype(np.float32)
            clips_out.append(arr[:, qpos_slice] if qpos_slice is not None else arr)
        else:
            arr = np.asarray(item, dtype=np.float32)
            clips_out.append(arr[:, qpos_slice] if qpos_slice is not None else arr)
    return clips_out, fps_out


class AmpMotionSampler:
    """Samples continuous-time interpolated ``(batch, num_frames, frame_dim)`` windows."""

    def __init__(
        self,
        motion_source: str | Path | Sequence[str | Path | np.ndarray | torch.Tensor],
        num_frames: int,
        device: torch.device | str = "cpu",
        *,
        qpos_slice: slice | None = None,
        default_fps: float = 50.0,
    ) -> None:
        raw_clips, fps = _load_clip_list(
            motion_source, qpos_slice=qpos_slice, default_fps=default_fps
        )
        if not raw_clips:
            raise ValueError("AmpMotionSampler received zero motion clips.")
        self.device = torch.device(device)
        self.num_frames = int(num_frames)
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.motions = [
            torch.as_tensor(c, dtype=torch.float32, device=self.device)
            for c in raw_clips
        ]
        self.frame_dim = int(self.motions[0].shape[1])
        for i, m in enumerate(self.motions):
            if m.ndim != 2 or m.shape[1] != self.frame_dim:
                raise ValueError(
                    f"Clip {i} has shape {tuple(m.shape)}, expected (*, {self.frame_dim})."
                )
            if m.shape[0] < self.num_frames + 2:
                raise ValueError(
                    f"Clip {i} has {m.shape[0]} frames; need >= {self.num_frames + 2}."
                )

    def sample(self, batch_size: int) -> torch.Tensor:
        """Draw ``(batch_size, num_frames, frame_dim)`` windows (oldest frame first)."""
        weights = torch.tensor(
            [float(m.shape[0]) for m in self.motions], device=self.device
        )
        counts = torch.multinomial(weights, batch_size, replacement=True).bincount(
            minlength=len(self.motions)
        )
        offsets = torch.arange(
            -(self.num_frames - 2), 2, device=self.device, dtype=torch.float32
        )
        windows: list[torch.Tensor] = []
        for motion, count in zip(self.motions, counts.tolist()):
            if count == 0:
                continue
            n_rows = motion.shape[0]
            base = torch.empty(count, device=self.device).uniform_(
                float(self.num_frames - 2), float(n_rows - 2)
            )
            idx = base.unsqueeze(1) + offsets.unsqueeze(0)
            low = idx.floor().long()
            frac = (idx - low.float()).unsqueeze(-1)
            frames = motion[low] * (1.0 - frac) + motion[low + 1] * frac
            windows.append(frames)
        return torch.cat(windows, dim=0)


class AmpReplayBuffer:
    """Ring buffer storing flattened policy motion windows ``(capacity, window_dim)``."""

    def __init__(
        self,
        window_dim: int,
        capacity: int = 100_000,
        device: torch.device | str = "cpu",
    ) -> None:
        self._buffer = torch.zeros(int(capacity), int(window_dim), device=device)
        self._capacity = int(capacity)
        self._next = 0
        self._size = 0

    def insert(self, windows: torch.Tensor) -> None:
        n = int(windows.shape[0])
        if n == 0:
            return
        idx = (
            self._next + torch.arange(n, device=self._buffer.device)
        ) % self._capacity
        self._buffer[idx] = windows.detach().to(self._buffer.device)
        self._next = int((self._next + n) % self._capacity)
        self._size = min(self._size + n, self._capacity)

    def sample(self, batch_size: int) -> torch.Tensor:
        if self._size == 0:
            raise ValueError(
                "AmpReplayBuffer is empty; insert windows before sampling."
            )
        idx = torch.randint(self._size, (batch_size,), device=self._buffer.device)
        return self._buffer[idx]

    def __len__(self) -> int:
        return self._size


class FrameNormalizer(nn.Module):
    """Welford running mean/variance over raw single frames, applied per-frame to windows."""

    def __init__(self, frame_dim: int, clip: float = 10.0, eps: float = 1e-2) -> None:
        super().__init__()
        self.clip = float(clip)
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(int(frame_dim)))
        self.register_buffer("var", torch.ones(int(frame_dim)))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, frames: torch.Tensor) -> None:
        flat = frames.reshape(-1, frames.shape[-1])
        batch_count = int(flat.shape[0])
        if batch_count == 0:
            return
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean.add_(delta * (batch_count / total))
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var.copy_(
            (m_a + m_b + delta.square() * self.count * batch_count / total) / total
        )
        self.count.copy_(total)

    def normalize(self, frames: torch.Tensor) -> torch.Tensor:
        std = (self.var + self.eps**2).sqrt()
        return ((frames - self.mean) / std).clamp(-self.clip, self.clip)


class AmpDiscriminator(nn.Module):
    """Spectral-norm LSGAN motion discriminator with style reward and expert gradient penalty."""

    def __init__(
        self, window_dim: int, hidden_dims: Sequence[int] = (256, 256)
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(window_dim)
        for h in hidden_dims:
            layers.extend([spectral_norm(nn.Linear(in_dim, int(h))), nn.ReLU()])
            in_dim = int(h)
        self.trunk = nn.Sequential(*layers)
        self.head = spectral_norm(nn.Linear(in_dim, 1))

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(windows))

    def style_reward(self, windows: torch.Tensor, coef: float = 5.0) -> torch.Tensor:
        with torch.no_grad():
            d = self.forward(windows).squeeze(-1)
        return float(coef) * torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0)

    def gradient_penalty(
        self, expert_windows: torch.Tensor, scale: float = 5.0
    ) -> torch.Tensor:
        expert = expert_windows.detach().clone().requires_grad_(True)
        d = self.forward(expert)
        (grad,) = torch.autograd.grad(
            outputs=d,
            inputs=expert,
            grad_outputs=torch.ones_like(d),
            create_graph=True,
            retain_graph=True,
        )
        return float(scale) * grad.norm(2, dim=1).square().mean()

    def lsgan_step(
        self,
        optimizer: torch.optim.Optimizer,
        policy_windows: torch.Tensor,
        expert_windows: torch.Tensor,
        *,
        grad_pen_scale: float = 5.0,
    ) -> dict[str, float]:
        """Execute one LSGAN + expert-gradient-penalty discriminator step."""
        policy_d = self.forward(policy_windows)
        expert_d = self.forward(expert_windows)
        expert_loss = torch.nn.functional.mse_loss(expert_d, torch.ones_like(expert_d))
        policy_loss = torch.nn.functional.mse_loss(policy_d, -torch.ones_like(policy_d))
        amp_loss = 0.5 * (expert_loss + policy_loss)
        grad_pen = self.gradient_penalty(expert_windows, grad_pen_scale)

        optimizer.zero_grad()
        (amp_loss + grad_pen).backward()
        optimizer.step()
        return {
            "amp": float(amp_loss.item()),
            "amp_grad_pen": float(grad_pen.item()),
            "amp_expert_d": float(expert_d.mean().item()),
            "amp_policy_d": float(policy_d.mean().item()),
        }


try:
    from rsl_rl.algorithms import PPO as _RslPPO
except ImportError:  # pragma: no cover - optional [rsl] extra
    _RslPPO = None  # type: ignore[assignment,misc]


if _RslPPO is not None:

    class AmpPPO(_RslPPO):  # type: ignore[misc]
        """RSL-RL PPO augmented with :class:`AmpDiscriminator` over an AMP observation group."""

        def __init__(
            self,
            actor: Any,
            critic: Any,
            storage: Any,
            *,
            amp_motion_file: str | Path,
            amp_qpos_slice: slice | None = None,
            amp_obs_group: str = "amp",
            amp_num_frames: int = 5,
            amp_reward_coef: float = 5.0,
            amp_reward_scale: float = 0.02,
            amp_replay_size: int = 100_000,
            amp_discr_hidden_dims: tuple[int, ...] = (256, 256),
            amp_grad_pen_scale: float = 5.0,
            amp_trunk_weight_decay: float = 1e-3,
            amp_head_weight_decay: float = 1e-1,
            amp_warm_start: bool = False,
            **ppo_kwargs: Any,
        ) -> None:
            super().__init__(actor, critic, storage, **ppo_kwargs)
            self.amp_obs_group = amp_obs_group
            self.amp_warm_start = amp_warm_start
            self.amp_num_frames = amp_num_frames
            self.amp_reward_coef = amp_reward_coef
            self.amp_reward_scale = amp_reward_scale
            self.amp_grad_pen_scale = amp_grad_pen_scale

            self.amp_expert = AmpMotionSampler(
                amp_motion_file,
                amp_num_frames,
                self.device,
                qpos_slice=amp_qpos_slice,
            )
            window_dim = self.amp_expert.frame_dim * amp_num_frames
            self.discriminator = AmpDiscriminator(window_dim, amp_discr_hidden_dims).to(
                self.device
            )
            self.amp_normalizer = FrameNormalizer(self.amp_expert.frame_dim).to(
                self.device
            )
            self.amp_replay = AmpReplayBuffer(window_dim, amp_replay_size, self.device)
            self.amp_optimizer = torch.optim.Adam(
                [
                    {
                        "params": self.discriminator.trunk.parameters(),
                        "weight_decay": amp_trunk_weight_decay,
                    },
                    {
                        "params": self.discriminator.head.parameters(),
                        "weight_decay": amp_head_weight_decay,
                    },
                ],
                lr=self.learning_rate,
            )
            self._window_dim = window_dim
            self._style_reward_sum = torch.zeros((), device=self.device)
            self._style_reward_steps = 0

        def _split_amp_obs(self, obs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            amp = obs[self.amp_obs_group]
            if amp.shape[-1] == self._window_dim:
                return amp, torch.ones(
                    amp.shape[0], dtype=torch.bool, device=amp.device
                )
            if amp.shape[-1] != self._window_dim + 1:
                raise ValueError(
                    f"AMP group {self.amp_obs_group!r} expected dim {self._window_dim} or "
                    f"{self._window_dim + 1}, got {amp.shape[-1]}."
                )
            return amp[:, : self._window_dim], amp[:, self._window_dim] > 0.5

        def process_env_step(
            self,
            obs: Any,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            extras: dict[str, torch.Tensor],
        ) -> None:
            raw_windows, gate = self._split_amp_obs(obs)
            frames = raw_windows.view(
                -1, self.amp_num_frames, self.amp_expert.frame_dim
            )
            norm_flat = self.amp_normalizer.normalize(frames).flatten(1)
            style = self.discriminator.style_reward(norm_flat, self.amp_reward_coef)
            style = style * gate.float() * self.amp_reward_scale
            rewards = rewards + style
            self.amp_replay.insert(raw_windows)
            self._style_reward_sum = self._style_reward_sum + style.mean()
            self._style_reward_steps += 1
            super().process_env_step(obs, rewards, dones, extras)

        def update(self) -> dict[str, float]:
            batch_size = (
                self.storage.num_envs
                * self.storage.num_transitions_per_env
                // self.num_mini_batches
            )
            loss_dict: dict[str, float] = super().update()
            for group in self.amp_optimizer.param_groups:
                group["lr"] = self.learning_rate

            num_updates = self.num_learning_epochs * self.num_mini_batches
            totals = {
                "amp": 0.0,
                "amp_grad_pen": 0.0,
                "amp_expert_d": 0.0,
                "amp_policy_d": 0.0,
            }
            for _ in range(num_updates):
                pol_w = self.amp_replay.sample(batch_size).view(
                    -1, self.amp_num_frames, self.amp_expert.frame_dim
                )
                exp_w = self.amp_expert.sample(batch_size)
                self.amp_normalizer.update(pol_w[:, -1])
                self.amp_normalizer.update(exp_w[:, -1])
                pol_in = self.amp_normalizer.normalize(pol_w).flatten(1)
                exp_in = self.amp_normalizer.normalize(exp_w).flatten(1)
                step_metrics = self.discriminator.lsgan_step(
                    self.amp_optimizer,
                    pol_in,
                    exp_in,
                    grad_pen_scale=self.amp_grad_pen_scale,
                )
                for k, v in step_metrics.items():
                    totals[k] += v

            for k, v in totals.items():
                loss_dict[k] = v / num_updates
            loss_dict["amp_style_reward"] = float(self._style_reward_sum) / max(
                self._style_reward_steps, 1
            )
            self._style_reward_sum.zero_()
            self._style_reward_steps = 0
            return loss_dict
