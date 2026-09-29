# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""TaskConfig: Bolt-like sparse sprint on myo_sim ``myofullbody_simple``."""

from __future__ import annotations

from dataclasses import dataclass, field

from myosuite.core.config import (
    ActuatorGroupSpec,
    BackendConfig,
    GoalSpec,
    ObsSpec,
    RewardSpec,
    TaskConfig,
)
from myosuite.core.registry import register_task
from myosuite.envs.myo.tasks.basic.full_body.athletic_contract import (
    AthleticSpec,
    SPRINT_CTRL_DT,
    SPRINT_FALL_HEIGHT,
    SPRINT_HEADING_DIR,
    SPRINT_LAMBDA_LIMIT,
    SPRINT_LAMBDA_MID_LANE,
    SPRINT_LAMBDA_VEL,
    SPRINT_N_SUBSTEPS,
    SPRINT_SIM_DT,
)

DEFAULT_ATHLETIC = AthleticSpec()


@dataclass
class FullbodySimpleSprintTask(TaskConfig):
    """Sparse sprint: ``r = λ_vel·v + λ_mid·lane − |λ_limit|·pain`` (+ alive).

    Model is myo_sim ``myofullbody_simple``. Registered as ``myoFullbodySimpleSprint-v0``.
    """

    model: str = "myofullbody_simple"
    max_episode_steps: int = 1000
    backend: BackendConfig = field(
        default_factory=lambda: BackendConfig(
            n_substeps=SPRINT_N_SUBSTEPS,
            ctrl_dt=SPRINT_CTRL_DT,
            sim_dt=SPRINT_SIM_DT,
        )
    )
    obs: ObsSpec = field(
        default_factory=lambda: ObsSpec(
            keys=[
                "joint_pos",
                "joint_vel",
                "muscle_act",
                "root_planar_vel",
            ],
            extra={"heading_dir": SPRINT_HEADING_DIR},
        )
    )
    goal: GoalSpec = field(default_factory=lambda: GoalSpec(target_type="joint_angles"))
    reward: RewardSpec = field(
        default_factory=lambda: RewardSpec(
            terms=["sprint"],
            weights={"sprint": 1.0},
            extra={
                "heading_dir": SPRINT_HEADING_DIR,
                "lambda_vel": SPRINT_LAMBDA_VEL,
                "lambda_mid_lane": SPRINT_LAMBDA_MID_LANE,
                "lambda_limit": SPRINT_LAMBDA_LIMIT,
                "fall_height": SPRINT_FALL_HEIGHT,
                "alive_bonus": DEFAULT_ATHLETIC.alive_bonus,
                "mid_lane_weight": DEFAULT_ATHLETIC.mid_lane_weight,
            },
        )
    )
    actuators: list[ActuatorGroupSpec] = field(
        default_factory=lambda: [
            ActuatorGroupSpec(name="muscles", normalize_actions=True)
        ]
    )


register_task(FullbodySimpleSprintTask(), env_id="myoFullbodySimpleSprint-v0")
