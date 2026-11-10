# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The 528-dim directional walking observation, built from an ``MjData`` state.

The observation of ``MuscleMimicFullbodyDirectionalEnv`` / ``myoFullBodyDirectional-v0``
(``qpos_local(82) + qvel_local(82) + act(354) + root_vel_body(2) + heading_cmd(2) +
orientation(6)``) for an arbitrary heading command. The ChaseTag observations
(:mod:`~myosuite.envs.myo.tasks.mimic.chasetag_obs`) reuse it as their prefix.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from myosuite.physics.quat import quat_yaw

DIRECTIONAL_OBS_DIM = 528
"""Width of :func:`directional_obs` on the full body (82 + 82 + 354 + 2 + 2 + 6)."""


def directional_obs(
    model: mujoco.MjModel, data: mujoco.MjData, heading_theta: float
) -> np.ndarray:
    """Build the directional-task observation from an ``MjData`` state.

    Reproduces ``MuscleMimicFullbodyDirectionalEnv._get_obs_dict`` (same feature
    order: qpos_local, qvel_local, act, root_vel_body, heading_cmd, orientation),
    so a checkpoint trained on it is obs-compatible with the CPU env.

    Args:
        model: Compiled single-body full-body model (free root joint first).
        data: Its current state.
        heading_theta: World-frame heading command (rad).

    Returns:
        Float32 array of :data:`DIRECTIONAL_OBS_DIM` entries on the full body.
    """
    qpos_local = data.qpos[7:].astype(np.float32)
    qvel_local = data.qvel[6:].astype(np.float32)
    act = data.act.astype(np.float32) if model.na > 0 else np.zeros(0, np.float32)

    # Body-frame root velocity: world velocity rotated by the inverse pelvis yaw.
    yaw = float(quat_yaw(np, data.qpos[3:7]))
    vx, vy = float(data.qvel[0]), float(data.qvel[1])
    c, s = math.cos(-yaw), math.sin(-yaw)
    root_vel_body = np.array([c * vx - s * vy, s * vx + c * vy], dtype=np.float32)

    heading_cmd = np.array(
        [math.cos(heading_theta), math.sin(heading_theta)], dtype=np.float32
    )

    # Roll / pitch of the root quaternion and body-frame angular velocity.
    w, x, y, z = (
        float(data.qpos[3]),
        float(data.qpos[4]),
        float(data.qpos[5]),
        float(data.qpos[6]),
    )
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    wx_w, wy_w, wz_w = float(data.qvel[3]), float(data.qvel[4]), float(data.qvel[5])
    wx_b = c * wx_w - s * wy_w
    wy_b = s * wx_w + c * wy_w
    vz = float(data.qvel[2])
    orientation = np.array([roll, pitch, wx_b, wy_b, wz_w, vz], dtype=np.float32)

    return np.concatenate(
        [qpos_local, qvel_local, act, root_vel_body, heading_cmd, orientation]
    ).astype(np.float32)


__all__ = ["DIRECTIONAL_OBS_DIM", "directional_obs"]
