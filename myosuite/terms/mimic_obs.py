# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Observation and termination terms for MuscleMimic motion imitation.

``mimic_lookahead_obs`` produces the 5-step lookahead used in the
MuscleMimic paper: k future frames at stride 20, encoding relative site
positions and root kinematics so the policy can anticipate motion.

All functions are backend-agnostic (numpy / jax.numpy / torch).
"""

from __future__ import annotations


import numpy as np


# ---------------------------------------------------------------------------
# Lookahead observation
# ---------------------------------------------------------------------------


def mimic_lookahead_obs(
    current_frame: int,
    clip_site_xpos: np.ndarray,
    clip_root_pos: np.ndarray | None,
    clip_root_vel: np.ndarray | None,
    current_root_pos: np.ndarray,
    k: int = 5,
    stride: int = 20,
) -> np.ndarray:
    """Build the k-step lookahead observation vector from a motion clip.

    Encodes future site positions (relative to current root) and root
    kinematics deltas at strides {stride, 2*stride, …, k*stride}.
    Wraps around at the end of the clip.

    Args:
        current_frame: Current frame index in the clip.
        clip_site_xpos: Full clip site positions, shape ``(T, n_sites, 3)``.
        clip_root_pos: Full clip root positions, shape ``(T, 3)`` or ``None``.
        clip_root_vel: Full clip root velocities, shape ``(T, 3)`` or ``None``.
        current_root_pos: Current environment root position, shape ``(3,)``.
        k: Number of lookahead steps.
        stride: Frame stride between steps.

    Returns:
        1-D float32 array of length:
        ``k * (n_sites*3 + 3*(root_pos included) + 3*(root_vel included) + 1)``.
    """
    T = clip_site_xpos.shape[0]
    parts: list[np.ndarray] = []
    for step in range(1, k + 1):
        future_frame = (current_frame + step * stride) % T
        # Relative site positions
        future_sites = clip_site_xpos[future_frame]  # (n_sites, 3)
        rel_sites = (future_sites - current_root_pos).reshape(-1)
        parts.append(rel_sites.astype(np.float32))
        # Root position delta
        if clip_root_pos is not None:
            delta_rpos = (clip_root_pos[future_frame] - current_root_pos).astype(
                np.float32
            )
            parts.append(delta_rpos)
        # Root velocity
        if clip_root_vel is not None:
            parts.append(clip_root_vel[future_frame].astype(np.float32))
        # Phase [0, 1]
        phase = np.array([future_frame / max(T - 1, 1)], dtype=np.float32)
        parts.append(phase)
    return np.concatenate(parts)


def mimic_lookahead_obs_size(
    n_sites: int,
    has_root_pos: bool = True,
    has_root_vel: bool = True,
    k: int = 5,
) -> int:
    """Return the flat size of :func:`mimic_lookahead_obs` output.

    Args:
        n_sites: Number of tracked sites.
        has_root_pos: Whether clip_root_pos is provided.
        has_root_vel: Whether clip_root_vel is provided.
        k: Number of lookahead steps.

    Returns:
        Integer flat dimension.
    """
    per_step = n_sites * 3 + 1
    if has_root_pos:
        per_step += 3
    if has_root_vel:
        per_step += 3
    return k * per_step


# ---------------------------------------------------------------------------
# Early termination
# ---------------------------------------------------------------------------


def mimic_should_terminate(
    current_site_pos: np.ndarray,
    ref_site_pos: np.ndarray,
    current_root_pos: np.ndarray,
    ref_root_pos: np.ndarray | None,
    site_err_threshold: float = 1.0,
    root_err_threshold: float = 0.3,
) -> bool:
    """Return True when the agent has deviated past an irrecoverable threshold.

    Matches MuscleMimic early-termination logic:
    - Mean Euclidean site tracking error > ``site_err_threshold`` (default 1 m)
    - Root position error > ``root_err_threshold`` (default 0.3 m)

    Args:
        current_site_pos: Shape ``(n_sites, 3)``.
        ref_site_pos: Shape ``(n_sites, 3)``.
        current_root_pos: Shape ``(3,)``.
        ref_root_pos: Shape ``(3,)`` or ``None`` to disable root-error checks.
        site_err_threshold: Mean site distance threshold in metres.
        root_err_threshold: Root position L2 distance threshold in metres.

    Returns:
        ``True`` if episode should terminate.
    """
    diffs = current_site_pos - ref_site_pos
    mean_site_err = float(np.linalg.norm(diffs, axis=-1).mean())
    if ref_root_pos is None:
        return mean_site_err > site_err_threshold
    root_err = float(np.linalg.norm(current_root_pos - ref_root_pos))
    return (mean_site_err > site_err_threshold) or (root_err > root_err_threshold)
