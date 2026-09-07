# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Glue between GMR-family retargeting (amathislab/gmr_plus) and MyoSuite.

Converts a GMR ``.pkl`` retargeted-motion file (``root_pos``, ``root_rot``,
``dof_pos``, ``fps``) on the ``myofullbody`` skeleton into a MyoSuite
``MotionClip``-compatible ``.npz`` (``qpos``, ``qvel``, ``site_xpos``) that
``myosuite.envs.myo.tasks.mimic.clip_env.MuscleMimicClipEnvV0`` and
``myosuite.core.trajectory_io.load_motion_clip`` can consume directly.

This does not depend on ``general_motion_retargeting`` (gmr_plus) at import
time — only on the ``.pkl`` file it produces — so it has no effect on
MyoSuite's own dependency footprint.
"""

from myosuite.integrations.soma_gmr.retarget_io import (
    GMR_CLIP_SITE_ORDER,
    gmr_pkl_to_motion_clip_npz,
    load_gmr_pkl,
)

__all__ = [
    "GMR_CLIP_SITE_ORDER",
    "gmr_pkl_to_motion_clip_npz",
    "load_gmr_pkl",
]
