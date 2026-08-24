# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Shared opponent-relative pose observation helper.

Pure-numpy, backend-agnostic, dimension-agnostic (2D or 3D). Used by both the
two-agent FBVs env (:mod:`myosuite.terms.multiplayer.chase_tag_vs_obs`) and
the single-agent-vs-scripted-opponent :class:`ChaseTagEnv` variants. Each
call site resolves its own pelvis/opponent position and velocity arrays (body
name, site id, or mocap body — whichever applies) and hands plain arrays to
this function; it does not know about MuJoCo model/data structures itself.
"""

from __future__ import annotations

import numpy as np


def relative_pose_obs(
    self_pos: np.ndarray,
    self_vel: np.ndarray,
    opponent_pos: np.ndarray,
    opponent_vel: np.ndarray,
) -> np.ndarray:
    """Return ``[rel_pos, rel_vel, dist]`` between an agent and its opponent.

    Args:
        self_pos: This agent's position, shape ``(2,)`` or ``(3,)``.
        self_vel: This agent's linear velocity, same shape as ``self_pos``.
        opponent_pos: Opponent position, same shape as ``self_pos``.
        opponent_vel: Opponent linear velocity, same shape as ``self_pos``.

    Returns:
        Float32 array of shape ``(2 * len(self_pos) + 1,)``:
        ``opponent_pos - self_pos``, ``opponent_vel - self_vel``, and the
        scalar Euclidean distance between the two positions.
    """
    rel_pos = np.asarray(opponent_pos) - np.asarray(self_pos)
    rel_vel = np.asarray(opponent_vel) - np.asarray(self_vel)
    dist = np.array([np.linalg.norm(rel_pos)], dtype=np.float32)
    return np.concatenate([rel_pos, rel_vel, dist]).astype(np.float32)


__all__ = ["relative_pose_obs"]
