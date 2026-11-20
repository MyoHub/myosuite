# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Welford online running mean/variance for observation normalisation.

Provides a pure-function API (update, normalize) so it can be used both
on CPU (numpy) and in JAX jit/scan contexts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _is_jax_array(*values: Any) -> bool:
    for v in values:
        if v is not None and (
            hasattr(v, "__jax_array__") or type(v).__module__.startswith("jax")
        ):
            return True
    return False


@dataclass
class RunningMeanStd:
    """Online Welford estimator state.

    Args:
        mean: Running mean, shape ``(dim,)``.
        var: Running variance, shape ``(dim,)``.
        count: Number of samples seen.
    """

    mean: Any
    var: Any
    count: Any = 0

    @classmethod
    def zeros(cls, dim: int) -> RunningMeanStd:
        """Initialise with zero mean and unit variance.

        Args:
            dim: Observation dimension.

        Returns:
            New :class:`RunningMeanStd` instance.
        """
        return cls(
            mean=np.zeros(dim, dtype=np.float64),
            var=np.ones(dim, dtype=np.float64),
            count=0,
        )


def update(stats: RunningMeanStd, batch: Any) -> RunningMeanStd:
    """Update stats with a new batch of observations (Welford parallel form).

    Args:
        stats: Current running stats.
        batch: Shape ``(B, dim)`` or ``(dim,)`` for a single sample.

    Returns:
        New :class:`RunningMeanStd` with updated state.
    """
    if _is_jax_array(stats.mean, stats.var, batch):
        import jax.numpy as jnp

        x = jnp.asarray(batch, dtype=jnp.asarray(stats.mean).dtype)
        if x.ndim == 1:
            x = x[jnp.newaxis]
        b = x.shape[0]
        if b == 0:
            return RunningMeanStd(mean=stats.mean, var=stats.var, count=stats.count)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
    else:
        x = np.asarray(batch, dtype=np.float64)
        if x.ndim == 1:
            x = x[np.newaxis]
        b = x.shape[0]
        if b == 0:
            return RunningMeanStd(
                mean=stats.mean.copy(), var=stats.var.copy(), count=stats.count
            )
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)

    n = stats.count
    total = n + b
    delta = batch_mean - stats.mean
    new_mean = stats.mean + delta * b / total
    new_var = (stats.var * n + batch_var * b + delta * delta * n * b / total) / total
    return RunningMeanStd(mean=new_mean, var=new_var, count=total)


def normalize(
    stats: RunningMeanStd,
    obs: Any,
    eps: float = 1e-8,
    clip: float = 10.0,
) -> Any:
    """Normalise observations using running stats.

    Args:
        stats: Current running stats.
        obs: Shape ``(..., dim)``.
        eps: Numerical stability offset.
        clip: Symmetric clip bound for normalised output.

    Returns:
        Normalised array, same shape as ``obs``, clipped to ``[-clip, clip]``.
    """
    if _is_jax_array(stats.mean, stats.var, obs):
        import jax.numpy as jnp

        normed = (obs - stats.mean) / jnp.sqrt(stats.var + eps)
        return jnp.clip(normed, -clip, clip).astype(jnp.float32)

    normed = (obs - stats.mean) / np.sqrt(stats.var + eps)
    return np.clip(normed, -clip, clip).astype(np.float32)
