# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Per-sample clipping of the gradient that reaches an imagined action.

After NablaMimic's ``clip_rows_to_median`` (Lu & Chen 2026): in the backward pass
each row's gradient norm is capped at ``kappa`` times the batch median row norm, so
a few exploding imagined trajectories cannot dominate the actor update.
"""

from __future__ import annotations

import torch


class _MedianRowClip(torch.autograd.Function):
    """Identity forward; rows of the incoming gradient clipped to ``kappa`` x median norm."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        probe: torch.Tensor,
        kappa: float,
    ) -> torch.Tensor:
        ctx.kappa = kappa
        return x.view_as(x)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, g: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        g, frac = clip_rows(g, ctx.kappa)
        # The probe's "gradient" carries the clipped fraction back to the caller.
        return g, frac.reshape(()), None


def clip_rows(g: torch.Tensor, kappa: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale rows of *g* whose norm exceeds ``kappa`` x the median row norm down to it.

    Args:
        g: ``(..., B, n)`` gradient rows; the median is taken over the ``B`` rows of
            each leading index (e.g. per imagined step).
        kappa: Threshold relative to the median row norm (a zero median clips nothing).

    Returns:
        The clipped rows and the fraction of rows that were clipped.
    """
    norm = g.norm(dim=-1)
    thr = kappa * norm.median(dim=-1, keepdim=True).values
    clipped = (norm > thr) & (thr > 0)
    scale = torch.where(clipped, thr / norm.clamp_min(1e-30), torch.ones_like(norm))
    return g * scale[..., None], clipped.float().mean()


def clip_rows_to_median(
    x: torch.Tensor, kappa: float, probe: torch.Tensor | None = None
) -> torch.Tensor:
    """Return *x* unchanged; clip each row of its gradient to ``kappa`` x the median row norm.

    Args:
        x: ``(..., n)`` tensor whose rows (last dimension) are samples' actions.
        kappa: Clip threshold relative to the median row norm of the batch.
        probe: Optional scalar leaf with ``requires_grad``; after ``backward`` its
            ``grad`` is the sum over calls of the fraction of rows that were clipped.

    Returns:
        A tensor equal to *x*.
    """
    if probe is None:
        probe = torch.zeros((), device=x.device, requires_grad=True)
    return _MedianRowClip.apply(x, probe, kappa)
