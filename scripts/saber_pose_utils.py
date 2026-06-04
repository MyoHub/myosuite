# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Script-local helpers for saber posture setup and deterministic baselines."""

from __future__ import annotations

import mujoco
import numpy as np

from myosuite.utils.mujoco_joint_utils import joint_qpos_width


def reset_non_arm_joints_to_upright(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    reset_qpos: np.ndarray,
) -> None:
    """Keep arm posture while restoring non-arm joints to reset pose.

    Args:
        model: Compiled MuJoCo model.
        data: MuJoCo runtime data to modify in-place.
        reset_qpos: Reference reset qpos vector.
    """
    if data.qpos.shape[0] >= 7 and reset_qpos.shape[0] >= 7:
        data.qpos[:7] = reset_qpos[:7]

    for joint_id in range(int(model.njnt)):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if joint_name.endswith("_l") or joint_name.endswith("_r"):
            continue
        qadr = int(model.jnt_qposadr[joint_id])
        span = joint_qpos_width(model, joint_id)
        data.qpos[qadr : qadr + span] = reset_qpos[qadr : qadr + span]

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def build_finger_flexor_mask(model: mujoco.MjModel) -> np.ndarray:
    """Return actuator mask for bilateral finger flexor/intrinsic groups."""
    prefixes = ("FDS", "FDP", "FPL", "LU_RB", "RI", "UI_UB", "OP")
    mask = np.zeros(model.nu, dtype=np.float32)
    for idx in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) or ""
        if any(name.startswith(prefix) for prefix in prefixes):
            mask[idx] = 1.0
    return mask


def build_non_finger_qpos_mask(model: mujoco.MjModel) -> np.ndarray:
    """Return qpos mask of entries to clamp (all except bilateral finger joints)."""
    finger_tokens = (
        "cmc",
        "mp_",
        "mcp",
        "pm",
        "md",
        "ip_",
        "ip_flexion",
        "abduction",
    )
    free_qpos_idx: set[int] = set()
    for joint_id in range(model.njnt):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        ).lower()
        if not (name.endswith("_l") or name.endswith("_r")):
            continue
        if any(token in name for token in finger_tokens):
            free_qpos_idx.add(int(model.jnt_qposadr[joint_id]))

    mask = np.ones(model.nq, dtype=bool)
    for idx in free_qpos_idx:
        mask[idx] = False
    return mask
