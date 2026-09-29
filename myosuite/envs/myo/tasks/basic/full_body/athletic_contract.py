# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Shared AthleticSpec MDP contract (Bolt SprintConfig / msk_envs LaneConfig λs).

Used by CPU ModularTaskEnv and mjlab registrations so sibling athletic tracks
share one set of reward weights and control timing.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bolt / msk_envs SprintConfig / LaneConfig lambdas.
SPRINT_LAMBDA_VEL: float = 1e-2
SPRINT_LAMBDA_MID_LANE: float = 1e-2
SPRINT_LAMBDA_LIMIT: float = -3e-4
SPRINT_LAMBDA_SPRING: float = 0.0
SPRINT_LAMBDA_DAMPER: float = 0.0
SPRINT_LAMBDA_MUSCLE_PASSIVE: float = 0.0

# Control timing used by Bolt sprint (~30 Hz decision rate, sim_dt=1e-4 / 333 substeps).
SPRINT_CTRL_DT: float = 1.0 / 30.0
SPRINT_SIM_DT: float = 1e-4
SPRINT_N_SUBSTEPS: int = int(round(SPRINT_CTRL_DT / SPRINT_SIM_DT))  # 333

SPRINT_FALL_HEIGHT: float = 0.6  # metres; Bolt WarpSprintEnv MIN_ROOT_HEIGHT
SPRINT_ALIVE_BONUS: float = 0.0  # Bolt has no explicit alive term; keep 0
# Default for Z-up myo_sim composites (myofullbody_simple). Bolt sprinter is Y-up +X.
SPRINT_HEADING_DIR: tuple[float, float] = (-1.0, 0.0)
SPRINT_MID_LANE_WEIGHT: float = 2.5  # Bolt MID_LANE_WEIGHT
SPRINT_LANE_WIDTH: float = 0.6
SPRINT_FACING_ANGLE_DEG: float = 30.0


@dataclass(frozen=True)
class AthleticSpec:
    """Frozen MDP knobs shared across backends for athletic / sprint tasks."""

    lambda_vel: float = SPRINT_LAMBDA_VEL
    lambda_mid_lane: float = SPRINT_LAMBDA_MID_LANE
    lambda_limit: float = SPRINT_LAMBDA_LIMIT
    lambda_spring: float = SPRINT_LAMBDA_SPRING
    lambda_damper: float = SPRINT_LAMBDA_DAMPER
    lambda_muscle_passive: float = SPRINT_LAMBDA_MUSCLE_PASSIVE
    ctrl_dt: float = SPRINT_CTRL_DT
    sim_dt: float = SPRINT_SIM_DT
    n_substeps: int = SPRINT_N_SUBSTEPS
    fall_height: float = SPRINT_FALL_HEIGHT
    alive_bonus: float = SPRINT_ALIVE_BONUS
    heading_dir: tuple[float, float] = SPRINT_HEADING_DIR
    mid_lane_weight: float = SPRINT_MID_LANE_WEIGHT
    lane_width: float = SPRINT_LANE_WIDTH
    facing_angle_deg: float = SPRINT_FACING_ANGLE_DEG
