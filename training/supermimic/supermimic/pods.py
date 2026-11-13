# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""PODS-style action targets and median-relative per-sample action-gradient clipping (opt-in runner options).

PODS (Zamora Mora et al., ICML 2021): instead of one optimizer step on the window gradient, form per-step action
targets ``a_t + eta * dJ/da_t`` and regress the policy onto them. The recipe follows NablaMimic's ``--action-target``
and ``--act-cot-clip`` (github.com/sailab-ai/NablaMimic, Apache-2.0, Copyright 2026 The NablaMimic Authors;
``train.py: action_target_update`` and ``nets.py: clip_rows_to_median``), re-implemented here in PyTorch for
excitation actions: the in-graph clip of each step's action gradient to ``kappa`` x the median over rows that carry
gradient, the zero offset added after that clip, the targets' gradients clipped again against one median over all
live rows, and ``eta`` from the RMS over the rows used for regression. Everything is fixed-shape tensor code (no
boolean indexing, no host reads), so it can be captured in the runner's CUDA graph.
"""

from __future__ import annotations

import torch


def live_median(norm: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    """The (lower) median of ``norm`` over the entries where ``live``, without a host read.

    Args:
        norm: ``(n,)`` row norms.
        live: ``(n,)`` bool mask of the rows to take the median over.

    Returns:
        A 0-dim tensor: the median, ``inf`` when no row is live.
    """
    s = torch.where(live, norm, torch.full_like(norm, float("inf"))).sort().values
    k = ((live.sum() - 1).clamp(min=0) // 2).view(1)
    return s.gather(0, k).squeeze(0)


def clip_scale(norm: torch.Tensor, live: torch.Tensor, kappa: float) -> torch.Tensor:
    """Per-row factors that scale rows above ``kappa`` x the live median norm down to it (1 elsewhere)."""
    thr = kappa * live_median(norm, live)
    return (thr / norm.clamp(min=1e-30)).clamp(max=1.0)


class MedianRowClip(torch.autograd.Function):
    """Identity on ``x`` (B, n). Backward: each row's gradient clipped to ``kappa`` x the median norm of the rows that
    carry gradient (rows of ended or masked trajectories are exactly zero and would pull the median down). The
    gradient of ``probe`` (2,) receives [fraction of live rows clipped, fraction of their summed norm removed]."""

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, probe: torch.Tensor, kappa: float
    ) -> torch.Tensor:
        ctx.kappa = kappa
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        norm = g.float().norm(dim=-1)
        live = norm > 0  # NaN rows are not live: they never set the threshold
        sc = clip_scale(norm, live, ctx.kappa)
        n = live.sum().clamp(min=1)
        frac = (live & (sc < 1)).sum() / n
        kept = torch.where(live, norm * sc, 0.0).sum()
        removed = 1 - kept / torch.where(live, norm, 0.0).sum().clamp(min=1e-30)
        return g * sc.to(g.dtype)[:, None], torch.stack((frac, removed)), None


def pods_targets(
    a: torch.Tensor,
    g: torch.Tensor,
    delta: torch.Tensor | float,
    kappa: float = 0.0,
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Action targets ``clamp(a + eta * g, 0, 1)`` and the regression weights of their rows.

    A row is used for regression (weight 1) if its gradient is finite and nonzero and ``valid`` allows it; non-finite
    gradient entries are zeroed, so they never contaminate other rows. With ``kappa > 0`` the rows' gradients are
    first clipped to ``kappa`` x their median norm over the used rows. ``eta`` makes the RMS of ``eta * g`` over the
    used rows ``delta`` excitation units; with no used row it is 0 and the targets equal ``a``.

    Args:
        a: ``(..., nu)`` executed excitations of the window.
        g: Ascent direction ``dJ/da`` of the same shape.
        delta: RMS target step in excitation units (float or 0-dim tensor).
        kappa: Median-relative clip of the rows' gradient norms (0: off).
        valid: Optional ``(...)`` bool mask of rows that may be used (e.g. finite policy inputs).

    Returns:
        The targets ``(..., nu)``, the weights ``(...)`` and statistics (0-dim tensors): ``eta``, ``used`` (fraction
        of rows used), ``nonfinite`` (rows with a non-finite gradient), ``clip`` (fraction of used rows clipped).
    """
    finite = torch.isfinite(g)
    nonfinite = (~finite).any(-1)
    g = torch.where(finite, g, 0.0)
    norm = g.norm(dim=-1)
    live = (norm > 0) & ~nonfinite
    if valid is not None:
        live = live & valid
    g = torch.where(live[..., None], g, 0.0)
    clip = torch.zeros((), device=g.device)
    if kappa > 0:
        sc = clip_scale(norm.flatten(), live.flatten(), kappa).view_as(norm)
        g = g * sc[..., None]
        clip = (live & (sc < 1)).sum() / live.sum().clamp(min=1)
    n = live.sum()
    ms = g.square().sum() / (n * g.shape[-1]).clamp(min=1)
    eta = torch.where(n > 0, delta / ms.sqrt().clamp(min=1e-30), 0.0)
    target = (a + eta * g).clamp(0.0, 1.0)
    w = live.to(a.dtype)
    stats = {
        "eta": eta,
        "used": w.mean(),
        "nonfinite": nonfinite.sum().float(),
        "clip": clip,
    }
    return target, w, stats
