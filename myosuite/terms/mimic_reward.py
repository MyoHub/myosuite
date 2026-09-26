# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""DeepMimic-style reward term functions for MuscleMimic motion imitation.

All functions are pure and backend-agnostic: they use ``accessor.array_module()``
so the same code runs on CPU (numpy), MJX (jax.numpy) and mjlab (torch).

Weights follow the MuscleMimic paper defaults:
  site tracking 0.6, joint pos 0.1, joint vel 0.1,
  root pos 0.1, root vel 0.1, root orientation 0.01.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Reward config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MimicTrackingConfig:
    """Shared tracking constants used across CPU/MJX/mjlab bindings."""

    reward_scale: float = 20.0
    success_threshold: float = 0.04


def compute_mimic_tracking_error(
    current_site_pos: np.ndarray,
    target_site_pos: np.ndarray,
) -> float:
    """Compute mean Euclidean tracking error over mimic sites.

    Args:
        current_site_pos: Current world-frame site positions, shape ``(n_sites, 3)``.
        target_site_pos: Target site positions from the clip, same shape.

    Returns:
        Mean Euclidean distance across sites.
    """
    distances = np.linalg.norm(current_site_pos - target_site_pos, axis=1)
    return float(distances.mean())


def compute_mimic_reward(track_err: float, cfg: MimicTrackingConfig) -> float:
    """Compute bounded dense tracking reward.

    Args:
        track_err: Mean Euclidean tracking error, e.g. from
            :func:`compute_mimic_tracking_error`.
        cfg: Tracking config supplying the exponential decay scale.

    Returns:
        Scalar reward in ``(0, 1]``.
    """
    return float(np.exp(-cfg.reward_scale * track_err))


@dataclass(frozen=True)
class MimicRewardWeights:
    """Per-term weights for the DeepMimic composite reward."""

    site: float = 0.6
    joint_pos: float = 0.1
    joint_vel: float = 0.1
    root_pos: float = 0.1
    root_vel: float = 0.1
    root_orient: float = 0.01


@dataclass(frozen=True)
class MimicRewardScales:
    """Exponential decay scale per term (higher → sharper penalty for error)."""

    site: float = 20.0
    joint_pos: float = 10.0
    joint_vel: float = 1.0
    root_pos: float = 20.0
    root_vel: float = 1.0
    root_orient: float = 10.0


DEFAULT_WEIGHTS = MimicRewardWeights()
DEFAULT_SCALES = MimicRewardScales()


# ---------------------------------------------------------------------------
# Individual reward terms
# ---------------------------------------------------------------------------


def mimic_site_tracking_reward(
    xp: Any,
    current_site_pos: Any,
    ref_site_pos: Any,
    scale: float = DEFAULT_SCALES.site,
) -> Any:
    """Exponential site-position tracking reward.

    Args:
        xp: Array module (numpy / jax.numpy / torch).
        current_site_pos: Current world-frame site positions, shape ``(..., n_sites, 3)``.
        ref_site_pos: Reference positions from clip, same shape.
        scale: Exponential decay scale.

    Returns:
        Scalar reward in (0, 1].
    """
    diff = current_site_pos - ref_site_pos
    mean_dist = xp.sqrt((diff * diff).sum(axis=-1)).mean(axis=-1)
    return xp.exp(-scale * mean_dist)


def mimic_joint_pos_reward(
    xp: Any,
    current_qpos: Any,
    ref_qpos: Any,
    scale: float = DEFAULT_SCALES.joint_pos,
) -> Any:
    """Exponential joint-position tracking reward.

    Args:
        xp: Array module.
        current_qpos: Current joint positions, shape ``(..., nq)``.
        ref_qpos: Reference joint positions, same shape.
        scale: Exponential decay scale.

    Returns:
        Scalar reward in (0, 1].
    """
    diff = current_qpos - ref_qpos
    mean_sq = (diff * diff).mean(axis=-1)
    return xp.exp(-scale * mean_sq)


def mimic_joint_vel_reward(
    xp: Any,
    current_qvel: Any,
    ref_qvel: Any,
    scale: float = DEFAULT_SCALES.joint_vel,
) -> Any:
    """Exponential joint-velocity tracking reward.

    Args:
        xp: Array module.
        current_qvel: Current joint velocities, shape ``(..., nv)``.
        ref_qvel: Reference joint velocities, same shape.
        scale: Exponential decay scale.

    Returns:
        Scalar reward in (0, 1].
    """
    diff = current_qvel - ref_qvel
    mean_sq = (diff * diff).mean(axis=-1)
    return xp.exp(-scale * mean_sq)


def mimic_root_pos_reward(
    xp: Any,
    current_root_pos: Any,
    ref_root_pos: Any,
    scale: float = DEFAULT_SCALES.root_pos,
) -> Any:
    """Exponential root-position tracking reward.

    Args:
        xp: Array module.
        current_root_pos: Current root position, shape ``(..., 3)``.
        ref_root_pos: Reference root position, same shape.
        scale: Exponential decay scale.

    Returns:
        Scalar reward in (0, 1].
    """
    diff = current_root_pos - ref_root_pos
    dist = xp.sqrt((diff * diff).sum(axis=-1))
    return xp.exp(-scale * dist)


def mimic_root_vel_reward(
    xp: Any,
    current_root_vel: Any,
    ref_root_vel: Any,
    scale: float = DEFAULT_SCALES.root_vel,
) -> Any:
    """Exponential root-velocity tracking reward.

    Args:
        xp: Array module.
        current_root_vel: Current root linear velocity, shape ``(..., 3)``.
        ref_root_vel: Reference root linear velocity, same shape.
        scale: Exponential decay scale.

    Returns:
        Scalar reward in (0, 1].
    """
    diff = current_root_vel - ref_root_vel
    mean_sq = (diff * diff).mean(axis=-1)
    return xp.exp(-scale * mean_sq)


def mimic_root_orient_reward(
    xp: Any,
    current_root_quat: Any,
    ref_root_quat: Any,
    scale: float = DEFAULT_SCALES.root_orient,
) -> Any:
    """Exponential root-orientation tracking reward using quaternion geodesic distance.

    Args:
        xp: Array module.
        current_root_quat: Current root quaternion (w, x, y, z), shape ``(..., 4)``.
        ref_root_quat: Reference root quaternion, same shape.
        scale: Exponential decay scale.

    Returns:
        Scalar reward in (0, 1].
    """
    # |q1 · q2| = cos(half-angle) → angle = 2 * arccos(|dot|)
    dot = (current_root_quat * ref_root_quat).sum(axis=-1)
    # Clamp to [-1, 1] to avoid NaN from floating-point drift
    clamped = _clamp(xp, dot, -1.0, 1.0)
    angle = 2.0 * _acos(xp, _abs(xp, clamped))
    return xp.exp(-scale * angle)


def mimic_composite_reward(
    xp: Any,
    current_site_pos: Any,
    ref_site_pos: Any,
    current_qpos: Any | None,
    ref_qpos: Any | None,
    current_qvel: Any | None,
    ref_qvel: Any | None,
    current_root_pos: Any | None,
    ref_root_pos: Any | None,
    current_root_vel: Any | None,
    ref_root_vel: Any | None,
    current_root_quat: Any | None,
    ref_root_quat: Any | None,
    weights: MimicRewardWeights = DEFAULT_WEIGHTS,
    scales: MimicRewardScales = DEFAULT_SCALES,
) -> dict[str, Any]:
    """Weighted DeepMimic composite reward.

    Args:
        xp: Array module (numpy / jax.numpy / torch).
        current_site_pos: Shape ``(..., n_sites, 3)``.
        ref_site_pos: Shape ``(..., n_sites, 3)``.
        current_qpos: Shape ``(..., nq)`` or ``None``.
        ref_qpos: Shape ``(..., nq)`` or ``None``.
        current_qvel: Shape ``(..., nv)`` or ``None``.
        ref_qvel: Shape ``(..., nv)`` or ``None``.
        current_root_pos: Shape ``(..., 3)`` or ``None``.
        ref_root_pos: Shape ``(..., 3)`` or ``None``.
        current_root_vel: Shape ``(..., 3)`` or ``None``.
        ref_root_vel: Shape ``(..., 3)`` or ``None``.
        current_root_quat: Shape ``(..., 4)`` (w, x, y, z) or ``None``.
        ref_root_quat: Shape ``(..., 4)`` or ``None``.
        weights: Per-term weights (must sum ≤ 1).
        scales: Per-term exponential decay scales.

    Returns:
        Dict with individual term rewards, weighted ``dense`` total, and
        ``solved`` / ``done`` booleans.
    """
    r_site = mimic_site_tracking_reward(xp, current_site_pos, ref_site_pos, scales.site)
    dense = weights.site * r_site

    r_jpos = None
    if current_qpos is not None and ref_qpos is not None:
        r_jpos = mimic_joint_pos_reward(xp, current_qpos, ref_qpos, scales.joint_pos)
        dense = dense + weights.joint_pos * r_jpos

    r_jvel = None
    if current_qvel is not None and ref_qvel is not None:
        r_jvel = mimic_joint_vel_reward(xp, current_qvel, ref_qvel, scales.joint_vel)
        dense = dense + weights.joint_vel * r_jvel

    r_rpos = None
    if current_root_pos is not None and ref_root_pos is not None:
        r_rpos = mimic_root_pos_reward(
            xp, current_root_pos, ref_root_pos, scales.root_pos
        )
        dense = dense + weights.root_pos * r_rpos

    r_rvel = None
    if current_root_vel is not None and ref_root_vel is not None:
        r_rvel = mimic_root_vel_reward(
            xp, current_root_vel, ref_root_vel, scales.root_vel
        )
        dense = dense + weights.root_vel * r_rvel

    r_rorient = None
    if current_root_quat is not None and ref_root_quat is not None:
        r_rorient = mimic_root_orient_reward(
            xp, current_root_quat, ref_root_quat, scales.root_orient
        )
        dense = dense + weights.root_orient * r_rorient
    # Compute mean site distance for termination / solved check
    diff = current_site_pos - ref_site_pos
    mean_site_dist = float(xp.sqrt((diff * diff).sum(axis=-1)).mean())

    return {
        "site": r_site,
        "joint_pos": r_jpos,
        "joint_vel": r_jvel,
        "root_pos": r_rpos,
        "root_vel": r_rvel,
        "root_orient": r_rorient,
        "dense": dense,
        "mean_site_dist": mean_site_dist,
        "solved": mean_site_dist < 0.04,
        "done": False,
    }


# ---------------------------------------------------------------------------
# Private helpers — keeps xp usage backend-agnostic without if/else chains
# ---------------------------------------------------------------------------


def _clamp(xp: Any, x: Any, lo: float, hi: float) -> Any:
    try:
        return xp.clip(x, lo, hi)
    except AttributeError:
        return xp.clamp(x, lo, hi)  # torch


def _abs(xp: Any, x: Any) -> Any:
    return xp.abs(x)


def _acos(xp: Any, x: Any) -> Any:
    try:
        return xp.arccos(x)
    except AttributeError:
        return xp.acos(x)  # torch
