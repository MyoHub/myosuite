# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Task config for MuscleMimic bimanual task on mjlab backend."""

from __future__ import annotations

from dataclasses import dataclass

from myosuite.integrations.musclemimic.mimic_common import MimicTrackingConfig

_TRACKING = MimicTrackingConfig()


@dataclass
class MuscleMimicBimanualCfg:
    """Configuration for mjlab MuscleMimic bimanual task."""

    sim_dt: float = 0.002
    ctrl_dt: float = 0.01
    max_episode_steps: int = 1000
    num_envs: int = 1024
    tracking_reward_scale: float = _TRACKING.reward_scale
    tracking_success_threshold: float = _TRACKING.success_threshold
