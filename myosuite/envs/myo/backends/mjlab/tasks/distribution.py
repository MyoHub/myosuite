# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""rsl_rl action distribution with a bounded, still trainable std.

Configs reference it by import path (``distribution_cfg["class_name"]``), which
rsl_rl's ``MLPModel`` resolves with ``rsl_rl.utils.resolve_callable``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from rsl_rl.modules.distribution import GaussianDistribution


def check_init_std(init_std: float, std_range: tuple[float, float]) -> None:
    """Require ``std_range[0] < init_std < std_range[1]``.

    Args:
        init_std: Initial std.
        std_range: ``(min, max)`` bounds of the std.

    Raises:
        ValueError: If ``init_std`` is not strictly inside ``std_range``.
    """
    lo, hi = std_range
    if not lo < init_std < hi:
        raise ValueError(
            f"init_std={init_std} must lie strictly inside std_range=({lo}, {hi})."
        )


class ProjectedGaussianDistribution(GaussianDistribution):
    """``GaussianDistribution`` whose std parameter is projected onto ``std_range``.

    rsl_rl's own ``std_range`` clamps the std inside the forward pass. That clamp
    has no gradient at or beyond a bound, so a std that reaches a bound, starts on
    one, or is loaded from a checkpoint beyond one is never trained again. Here
    :meth:`update` projects the parameter itself into the range (in place, without
    gradient) before every forward pass, and the parent keeps its wide default
    range, so the gradient stays live at the bounds. The checkpoint key
    (``distribution.std_param`` or ``distribution.log_std_param``) is unchanged,
    so checkpoints load strictly in both directions.

    Args:
        output_dim: Dimension of the action space.
        init_std: Initial std, strictly inside ``std_range``.
        std_range: ``(min, max)`` bounds the std parameter is projected onto.
        **kwargs: Further arguments of ``GaussianDistribution`` (``std_type``, ...).
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_range: tuple[float, float] = (1e-6, 1e6),
        **kwargs: Any,
    ) -> None:
        check_init_std(init_std, std_range)
        super().__init__(output_dim, init_std=init_std, **kwargs)
        # Same lower limit as rsl_rl's std_range (avoids a zero std).
        self.projection_range = (max(float(std_range[0]), 1e-6), float(std_range[1]))

    def update(self, mlp_output: torch.Tensor) -> None:
        """Project the std parameter onto ``projection_range``, then build the Gaussian."""
        lo, hi = self.projection_range
        with torch.no_grad():
            if self.std_type == "scalar":
                self.std_param.clamp_(lo, hi)
            else:
                self.log_std_param.clamp_(math.log(lo), math.log(hi))
        super().update(mlp_output)
