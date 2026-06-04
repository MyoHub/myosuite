# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Task config for the finger fingertip-reaching task on the mjlab backend."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

_SIMHIVE = pathlib.Path(__file__).parents[5] / "simhive" / "myo_sim"


@dataclass
class FingerReachCfg:
    """Configuration for the finger reach task on mjlab."""

    model_path: pathlib.Path = field(
        default_factory=lambda: _SIMHIVE / "finger" / "myofinger_v0.xml"
    )
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02
    max_episode_steps: int = 100
    tip_site_name: str = "IFtip"
    target_reach_range: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ] = ((0.1, -0.1, 0.1), (0.27, 0.1, 0.3))
    reach_thd: float = 0.05
    reach_reward_weight: float = 1.0
    bonus_weight: float = 4.0
    penalty_weight: float = 50.0
    num_envs: int = 4096
