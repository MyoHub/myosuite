# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Task config for MuscleMimic full-body task on mjlab backend."""

from __future__ import annotations

from dataclasses import dataclass

from myosuite.terms.mimic_reward import MimicTrackingConfig

_TRACKING = MimicTrackingConfig()


@dataclass
class MuscleMimicFullbodyCfg:
    """Configuration for mjlab MuscleMimic full-body task."""

    sim_dt: float = 0.002
    ctrl_dt: float = 0.01
    max_episode_steps: int = 1000
    num_envs: int = 1024
    tracking_reward_scale: float = _TRACKING.reward_scale
    tracking_success_threshold: float = _TRACKING.success_threshold
