# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reusable backend-agnostic helpers for Mimic tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MimicTrackingConfig:
    """Shared tracking constants used across CPU/MJX/mjlab bindings."""

    reward_scale: float = 20.0
    success_threshold: float = 0.04


def resolve_mimic_site_ids(model, site_names: tuple[str, ...]) -> np.ndarray:
    """Return model site ids for an ordered tuple of site names."""
    ids = [model.site(name).id for name in site_names]
    return np.asarray(ids, dtype=np.int32)


def compute_mimic_tracking_error(
    current_site_pos: np.ndarray,
    target_site_pos: np.ndarray,
) -> float:
    """Compute mean Euclidean tracking error over mimic sites."""
    distances = np.linalg.norm(current_site_pos - target_site_pos, axis=1)
    return float(distances.mean())


def compute_mimic_reward(track_err: float, cfg: MimicTrackingConfig) -> float:
    """Compute bounded dense tracking reward."""
    return float(np.exp(-cfg.reward_scale * track_err))


def sample_mimic_target_sites(
    rng: np.random.Generator,
    low: np.ndarray,
    high: np.ndarray,
    n_sites: int,
) -> np.ndarray:
    """Sample mimic target site positions (world frame), i.i.d. per site.

    Matches the sampling contract of
    :meth:`myosuite.envs.myo.backends.mjx.musclemimic_base.MjxMuscleMimicBase.sample_task`:
    shape ``(n_sites, 3)``, each axis drawn uniformly in
    ``[low[i], high[i]]`` for ``i in {0,1,2}``, broadcast across sites.

    Args:
        rng: NumPy random generator (e.g. ``env.np_random`` on Gymnasium CPU).
        low: Lower box corner, shape ``(3,)``.
        high: Upper box corner, shape ``(3,)``.
        n_sites: Number of mimic sites.

    Returns:
        Array of shape ``(n_sites, 3)``, ``float32``.
    """
    lo = np.asarray(low, dtype=np.float64)
    hi = np.asarray(high, dtype=np.float64)
    return rng.uniform(lo, hi, size=(int(n_sites), 3)).astype(np.float32)
