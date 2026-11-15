# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Write a TERRA reference walk (synthetic gait on the TERRA actor) for the parity fixture."""

import sys

import mujoco
import numpy as np

from myosuite.core.trajectory_io import MotionClip
from myosuite.integrations.musclemimic import (
    MIMIC_SITES,
    TerrainHeights,
    compose_waypoint_reference,
    build_terrain_fullbody_spec,
)

FRAMES = 120  # 15 control steps plus TERRA's 100-frame lookahead


def main(out: str) -> None:
    model = build_terrain_fullbody_spec().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    t = np.arange(500) / 100
    qpos = np.repeat(model.qpos0[None], len(t), 0)
    qpos[:, 1] -= t
    qpos[:, 7:] += 0.05 * np.sin(2 * np.pi * t)[:, None]
    clip = MotionClip(
        qpos=qpos, qvel=None, site_xpos=None, site_names=None, frequency_hz=100.0
    )
    heights = TerrainHeights(model)
    ref = compose_waypoint_reference(
        model,
        clip,
        model.qpos0[:2],
        np.array([[0.0, -2.0], [1.0, -2.5]]),
        0.01,
        lambda xy: heights(data, xy),
        MIMIC_SITES,
    )
    np.savez(
        out,
        ref_qpos=ref.qpos[:FRAMES].astype(np.float32),
        joint_names=np.array([model.joint(i).name for i in range(model.njnt)]),
        actuator_names=np.array([model.actuator(i).name for i in range(model.nu)]),
    )


if __name__ == "__main__":
    main(sys.argv[1])
