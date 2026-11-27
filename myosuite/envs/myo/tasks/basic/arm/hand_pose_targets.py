# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Targets of the ``myoHandPose*`` tasks, by joint name.

The target vectors list the 23 joints of the released MyoHand (myo_sim 0.2.x).
The revised hand of myo_sim >= 0.3 adds CMC4/CMC5 and internal ``node_*``
joints; matched by name, these stay at neutral and unscored. Random targets
on the revised hand come from measured active ranges
(``hand_pose_targets.json``, with sources).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np

_DATA = Path(__file__).with_name("hand_pose_targets.json")

# Scored joints of the hand pose tasks, in the released hand's qpos order.
HAND_POSE_JOINTS: tuple[str, ...] = (
    "pro_sup_r",
    "deviation_r",
    "flexion_r",
    "cmc_flexion_r",
    "cmc_abduction_r",
    "mp_flexion_r",
    "ip_flexion_r",
    "mcp2_flexion_r",
    "mcp2_abduction_r",
    "pm2_flexion_r",
    "md2_flexion_r",
    "mcp3_flexion_r",
    "mcp3_abduction_r",
    "pm3_flexion_r",
    "md3_flexion_r",
    "mcp4_flexion_r",
    "mcp4_abduction_r",
    "pm4_flexion_r",
    "md4_flexion_r",
    "mcp5_flexion_r",
    "mcp5_abduction_r",
    "pm5_flexion_r",
    "md5_flexion_r",
)


@functools.cache
def _data() -> dict:
    return json.loads(_DATA.read_text())


def revised_hand_marker_joint() -> str:
    """Joint that only the revised MyoHand (myo_sim >= 0.3) has."""
    return _data()["marker_joint"]


def revised_hand_active_range() -> dict[str, tuple[float, float]]:
    """Random-target range on the revised MyoHand, in radians.

    Returns:
        ``{joint_name: (lo, hi)}`` for the joints of :data:`HAND_POSE_JOINTS`,
        in that order.
    """
    ranges = _data()["ranges"]
    return {
        name: tuple(float(x) for x in np.deg2rad(ranges[name]["range"]))
        for name in HAND_POSE_JOINTS
    }
