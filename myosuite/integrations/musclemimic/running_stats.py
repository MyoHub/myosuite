# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Shared Welford incremental mean/variance helpers for MuscleMimic inference.

Both variants mirror ``musclemimic.algorithms.common.networks.RunningMeanStd``:
the current batch updates the statistics first, then the same batch is
normalised with the updated statistics.
"""

from __future__ import annotations

import numpy as np


def numpy_running_mean_std_update(
    *,
    obs: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply one RunningMeanStd update step and return normalised obs (numpy).

    Args:
        obs: Current observation(s), shape ``(obs_dim,)`` or ``(B, obs_dim)``.
        mean: Running mean accumulator, shape ``(obs_dim,)``.
        var: Running variance accumulator, shape ``(obs_dim,)``.
        count: Running sample count, scalar.

    Returns:
        Tuple ``(normalized_obs, new_mean, new_var, new_count)`` all float32.
    """
    obs = np.asarray(obs, dtype=np.float32)
    obs_2d = np.atleast_2d(obs)
    batch_mean = np.mean(obs_2d, axis=0).astype(np.float32)
    batch_var = (np.var(obs_2d, axis=0) + 1e-6).astype(np.float32)
    batch_count = np.asarray(obs_2d.shape[0], dtype=np.float32)

    mean = np.asarray(mean, dtype=np.float32)
    var = np.asarray(var, dtype=np.float32)
    count = np.asarray(count, dtype=np.float32)

    updated_count = count + batch_count
    delta = batch_mean - mean
    new_mean = mean + delta * batch_count / updated_count
    m_a = var * count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + np.square(delta) * count * batch_count / updated_count
    new_var = m2 / updated_count
    normalized = (obs_2d - new_mean) / np.sqrt(new_var + 1e-8)

    normalized = np.asarray(normalized, dtype=np.float32)
    if np.asarray(obs).ndim == 1:
        normalized = normalized[0]
    return (
        normalized,
        np.asarray(new_mean, dtype=np.float32),
        np.asarray(new_var, dtype=np.float32),
        np.asarray(updated_count, dtype=np.float32),
    )


def torch_running_mean_std_update(
    obs: torch.Tensor,  # noqa: F821
    run_mean: torch.Tensor,  # noqa: F821
    run_var: torch.Tensor,  # noqa: F821
    run_count: torch.Tensor,  # noqa: F821
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # noqa: F821
    """Apply one RunningMeanStd update step and return normalised obs (torch).

    Args:
        obs: Current observations, shape ``(B, obs_dim)``.
        run_mean: Running mean accumulator, shape ``(obs_dim,)``.
        run_var: Running variance accumulator, shape ``(obs_dim,)``.
        run_count: Running sample count, scalar tensor.

    Returns:
        Tuple ``(normalized_obs, new_mean, new_var, new_count)`` all float32.
    """
    import torch

    batch_count = torch.as_tensor(
        float(obs.shape[0]), dtype=torch.float32, device=obs.device
    )
    if obs.shape[0] == 1:
        batch_mean = obs.squeeze(0)
        batch_var = torch.zeros_like(batch_mean) + 1e-6
    else:
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0, unbiased=False) + 1e-6

    updated_count = run_count + batch_count
    delta = batch_mean - run_mean
    new_mean = run_mean + delta * batch_count / updated_count
    m_a = run_var * run_count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + torch.square(delta) * run_count * batch_count / updated_count
    new_var = m2 / updated_count
    normalized = (obs - new_mean) / torch.sqrt(new_var + 1e-8)
    return normalized, new_mean.detach(), new_var.detach(), updated_count.detach()


__all__ = ["numpy_running_mean_std_update", "torch_running_mean_std_update"]
