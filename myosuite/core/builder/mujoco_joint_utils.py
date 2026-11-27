# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Small MuJoCo joint indexing helpers shared across scripts and envs."""

from __future__ import annotations

import mujoco


def joint_qpos_width(model: mujoco.MjModel, joint_id: int) -> int:
    """Return the qpos span width for a MuJoCo joint.

    Args:
        model: Compiled MuJoCo model.
        joint_id: Joint id in ``model``.

    Returns:
        Number of qpos entries occupied by the joint.
    """
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1
