# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Joint ranges of the limited scalar joints, located in a ``qpos`` layout."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

# Width of each joint type in ``qpos`` (free, ball, slide, hinge).
_QPOS_WIDTH = np.array([7, 4, 1, 1])
_SCALAR_JOINTS = (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE)


def joint_range_from_model(
    mj_model: mujoco.MjModel, joint_ids: Any = None, qpos_offset: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Ranges of the limited hinge/slide joints and their ``qpos`` indices.

    The layout is ``qpos_offset`` entries followed by the ``qpos`` of
    *joint_ids* in order; with all joints and no offset it is MuJoCo's ``qpos``
    (the index is ``jnt_qposadr``). Ball-joint limits (a cone angle) and free
    joints have no per-coordinate range and are skipped.

    Args:
        mj_model: Compiled model.
        joint_ids: Joint ids forming the layout (default: all joints).
        qpos_offset: Entries in front of the first joint (e.g. 7 for a free root
            that is not part of *joint_ids*).

    Returns:
        ``(qpos_ids, ranges)``: integer indices, shape ``(k,)``, and
        ``[lower, upper]`` ranges, shape ``(k, 2)``.
    """
    ids = np.arange(mj_model.njnt) if joint_ids is None else np.asarray(joint_ids)
    ids = ids.astype(int)
    jtype = np.asarray(mj_model.jnt_type)[ids]
    width = _QPOS_WIDTH[jtype]
    start = qpos_offset + np.cumsum(width) - width
    keep = np.asarray(mj_model.jnt_limited)[ids].astype(bool) & np.isin(
        jtype, _SCALAR_JOINTS
    )
    ranges = np.asarray(mj_model.jnt_range, dtype=np.float64)[ids[keep]]
    return start[keep].astype(np.int64), ranges.reshape(-1, 2)
