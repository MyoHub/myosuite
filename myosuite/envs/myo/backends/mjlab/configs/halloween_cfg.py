# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Task config for MyoHalloween (MyoFullBody pumpkin dodge) on mjlab."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HalloweenCfg:
    """Configuration for ``myoHalloween-v0``."""

    sim_dt: float = 0.002
    ctrl_dt: float = 0.01
    max_episode_steps: int = 500
    num_envs: int = 1024
    throw_speed: float = 1.0
