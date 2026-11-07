# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""InstantTrack training changes for myoSuperMimic.

InstantTrack (Huy-Quang Dao, https://github.com/Huy-Quang-Dao/instantTrack, MIT) speeds up
SuperMimic training on the Unitree G1. This module re-implements the parts of it that carry over
to a muscle-driven body; the trainer turns each on with its own flag (``--instanttrack`` sets them
all to InstantTrack's values):

- :class:`FailureResets` (``--fail-reset``): a share of the episode resets starts one 1 s bin before
  the clip bins where recent episodes terminated (per-bin termination rate, EMA).
- :class:`FootContact` (``--contact-reward``, imagination only): a smooth foot-contact indicator is
  matched to the reference contact state, and horizontal foot speed is penalised while the reference
  foot is down. The reference contact uses a per-clip floor (InstantTrack's ``--ref_contact_fix``).
- :class:`MuonAdam` (``--pi-opt muon`` / ``--wm-opt muon``): Muon on the hidden weight matrices of an
  MLP, Adam on its first and last layer and on all vectors.
- :func:`policy_lr_multiplier` (``--pi-lr-mult``): a larger imagined batch with a policy learning-rate
  boost that decays to 1x.

Not carried over: the reference-pose residual policy (muscles have no joint-position targets), the
left-right mirror augmentation of world-model batches (needs a muscle and joint mirror map), and
the CUDA-graph / side-stream systems work (GPU engineering on top of a working configuration).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


# ---------------------------------------------------------------------- failure-focused resets
class FailureResets:
    """Per-bin termination rates and the reset starts they send episodes to.

    Bins are ``bin_frames`` clip frames long; a bin's statistic is an EMA (0.9) of whether the
    episodes that ended in it terminated (1) or were truncated (0), and all bins fade by 0.999 at
    every update so mastered bins drop out. A ``share`` of the resets starts uniformly inside the bin
    *before* a failure bin of the same clip, drawn in proportion to the failure rates.

    Args:
        n_frames: Frames of each clip (one entry per clip).
        bin_frames: Frames per bin.
        share: Share of resets that go to the failure bins (InstantTrack: 0.3).
        device: Torch device.
    """

    def __init__(
        self,
        n_frames: list[int],
        bin_frames: int,
        share: float,
        device: str | torch.device,
    ) -> None:
        self.share = float(share)
        self.bin_frames = max(int(bin_frames), 1)
        nb = [max(math.ceil(n / self.bin_frames), 1) for n in n_frames]
        self.first_bin = torch.tensor(
            [0, *torch.tensor(nb).cumsum(0).tolist()[:-1]], device=device
        )
        self.n_frames = torch.tensor(n_frames, device=device)
        self.bin_clip = torch.repeat_interleave(
            torch.arange(len(nb), device=device), torch.tensor(nb, device=device)
        )
        self.bin_start = (
            torch.arange(len(self.bin_clip), device=device)
            - self.first_bin[self.bin_clip]
        ) * self.bin_frames
        prev = torch.arange(len(self.bin_clip), device=device) - 1
        same = (prev >= 0) & (self.bin_clip[prev.clamp(min=0)] == self.bin_clip)
        self.bin_prev = torch.where(same, prev, prev + 1)
        self.fail = torch.zeros(len(self.bin_clip), device=device)

    def bins(self, clip: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        """Global bin of *frame* of *clip* (frames past the clip end go to its last bin)."""
        n = self.n_frames[clip]
        f = torch.minimum(frame, n - 1).clamp(min=0)
        return self.first_bin[clip] + torch.div(
            f, self.bin_frames, rounding_mode="floor"
        )

    def update(
        self, clip: torch.Tensor, frame: torch.Tensor, terminated: torch.Tensor
    ) -> None:
        """Record episodes that ended at *frame* of *clip*; *terminated* false = truncated."""
        self.fail.mul_(0.999)
        if frame.numel() == 0:
            return
        b = self.bins(clip, frame)
        x = terminated.float()
        sums = torch.zeros_like(self.fail).index_add_(0, b, x)
        cnt = torch.zeros_like(self.fail).index_add_(0, b, torch.ones_like(x))
        hit = cnt > 0
        mean = sums / cnt.clamp(min=1)
        self.fail.copy_(torch.where(hit, 0.9 * self.fail + 0.1 * mean, self.fail))

    def sample(
        self, offsets: torch.Tensor, clip: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Replace a ``share`` of the default reset *offsets* by starts before failure bins."""
        n = offsets.shape[0]
        if n == 0 or self.share <= 0:
            return offsets
        dev = offsets.device
        if clip is None:
            clip = torch.zeros(n, dtype=torch.long, device=dev)
        w = torch.zeros_like(self.fail).index_add_(0, self.bin_prev, self.fail)
        if float(w.sum()) <= 0:
            return offsets
        pick = torch.rand(n, device=dev) < self.share
        # a failure bin of each env's own clip: restrict the distribution per clip
        w_env = w.unsqueeze(0) * (self.bin_clip.unsqueeze(0) == clip.unsqueeze(1))
        has = w_env.sum(-1) > 0
        pick = pick & has
        if not bool(pick.any()):
            return offsets
        b = torch.multinomial(w_env[pick], 1).squeeze(-1)
        c = clip[pick]
        start = self.bin_start[b]
        end = torch.minimum(start + self.bin_frames, self.n_frames[c] - 1)
        f = start + (torch.rand(len(b), device=dev) * (end - start).clamp(min=1)).long()
        out = offsets.clone()
        out[pick] = f.to(offsets.dtype)
        return out

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"fail": self.fail}

    def load_state_dict(self, d: dict[str, torch.Tensor]) -> None:
        self.fail.copy_(d["fail"])


# ---------------------------------------------------------------------- contact reward
class FootContact:
    """InstantTrack's contact term on the tracked foot sites.

    Reference contact: a foot site is down when it is within ``margin`` of its own floor (the
    ``q`` quantile of its height over the clip) and its horizontal speed is below ``vmax``. Simulated
    contact: ``sigmoid((floor + margin - z) / width)``. Reward per foot site:
    ``1 - (c_sim - c_ref)^2 + c_ref * exp(-|v_xy|^2 / 0.3^2)``, averaged over the foot sites, so in
    ``[0, 2]``.

    Args:
        ref_sites: ``(T, n_sites, 3)`` reference site positions at the control rate.
        foot: Indices of the foot sites among the tracked sites.
        ctrl_dt: Control timestep.
    """

    def __init__(
        self,
        ref_sites: torch.Tensor,
        foot: list[int],
        ctrl_dt: float,
        margin: float = 0.015,
        vmax: float = 0.5,
        q: float = 0.05,
        width: float = 0.005,
        slip_scale: float = 0.3,
    ) -> None:
        if not foot:
            raise ValueError(
                "the contact reward needs foot sites among the tracked sites"
            )
        self.foot = torch.as_tensor(foot, device=ref_sites.device, dtype=torch.long)
        self.ctrl_dt = float(ctrl_dt)
        z = ref_sites[:, self.foot, 2]
        self.level = torch.quantile(z, q, dim=0) + margin  # (n_foot,)
        v = torch.zeros_like(ref_sites[:, self.foot, :2])
        v[1:] = (ref_sites[1:, self.foot, :2] - ref_sites[:-1, self.foot, :2]) / ctrl_dt
        v[0] = v[1] if len(v) > 1 else v[0]
        self.ref = ((z < self.level) & (v.norm(dim=-1) < vmax)).float()  # (T, n_foot)
        self.width = width
        self.slip_scale = slip_scale

    def reward(
        self, sites: torch.Tensor, sites_prev: torch.Tensor, frame: torch.Tensor
    ) -> torch.Tensor:
        """Contact reward at the clip *frame* (already held at the last frame)."""
        z = sites[..., self.foot, 2]
        c_sim = torch.sigmoid((self.level - z) / self.width)
        c_ref = self.ref[frame]
        v = (sites[..., self.foot, :2] - sites_prev[..., self.foot, :2]) / self.ctrl_dt
        slip = (v**2).sum(-1)
        r = (1 - (c_sim - c_ref) ** 2) + c_ref * torch.exp(-slip / self.slip_scale**2)
        return r.mean(-1)


FOOT_SITE_KEYS = ("ankle", "toes", "calcn", "foot")


def foot_site_indices(site_names: list[str]) -> list[int]:
    """Tracked sites that belong to the feet (by name)."""
    return [
        i
        for i, n in enumerate(site_names)
        if any(k in n.lower() for k in FOOT_SITE_KEYS)
    ]


# ---------------------------------------------------------------------- optimizer
class MuonAdam:
    """Muon on the hidden weight matrices of ``net``'s ``Linear`` layers, Adam on the rest.

    The first and the last ``Linear`` layer and all biases stay on Adam (as in InstantTrack). Muon's
    step is scaled to Adam's update RMS (``adjust_lr_fn="match_rms_adamw"``), so both share one
    learning rate. Exposes ``param_groups``, ``zero_grad``, ``step``, ``state_dict`` and
    ``load_state_dict`` like a single optimizer.
    """

    def __init__(
        self,
        net: nn.Module,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        fused: bool = False,
    ) -> None:
        lin = [m for m in net.modules() if isinstance(m, nn.Linear)]
        hidden = {id(m.weight) for m in lin[1:-1]}
        muon = [p for p in net.parameters() if id(p) in hidden]
        adam = [p for p in net.parameters() if id(p) not in hidden]
        self.opts: list[torch.optim.Optimizer] = [
            torch.optim.Adam(adam, lr=lr, betas=betas, fused=fused or None)
        ]
        if muon:
            self.opts.append(
                torch.optim.Muon(
                    muon, lr=lr, weight_decay=0.0, adjust_lr_fn="match_rms_adamw"
                )
            )

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [g for o in self.opts for g in o.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for o in self.opts:
            o.step()

    def state_dict(self) -> dict[str, Any]:
        return {f"opt{i}": o.state_dict() for i, o in enumerate(self.opts)}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        for i, o in enumerate(self.opts):
            o.load_state_dict(d[f"opt{i}"])


def make_optimizer(
    kind: str,
    net: nn.Module,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    fused: bool = False,
) -> Any:
    """``"adam"`` (the paper's) or ``"muon"`` (:class:`MuonAdam`) for *net*'s parameters."""
    if kind == "adam":
        return torch.optim.Adam(
            net.parameters(), lr=lr, betas=betas, fused=fused or None
        )
    if kind == "muon":
        return MuonAdam(net, lr, betas, fused)
    raise ValueError(f"unknown optimizer {kind!r} (adam, muon)")


# ---------------------------------------------------------------------- schedules
def policy_lr_multiplier(
    updates: int, mult: float, decay_at: int, decay_len: int
) -> float:
    """``mult`` until update ``decay_at``, then a linear decay to 1 over ``decay_len`` updates."""
    if mult <= 1.0:
        return 1.0
    if updates <= decay_at:
        return mult
    frac = min((updates - decay_at) / max(decay_len, 1), 1.0)
    return mult + (1.0 - mult) * frac
