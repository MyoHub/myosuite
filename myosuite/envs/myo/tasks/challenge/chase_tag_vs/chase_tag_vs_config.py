# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Configuration dataclass for the two-agent chase-tag environment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChaseTagVsConfig:
    """All hyperparameters for the 1v1 chase-tag scene/reward.

    Rules (matching MyoChallenge)
    -----------------------------
    Roles are assigned each episode: the **chaser** must tag (come within
    ``tag_radius_m`` of) the **runner** before ``maxTime`` seconds elapse.
    A binary tag at distance ≤ ``tag_radius_m`` ends the episode immediately.
    Roles can alternate each episode (``task_choice="random"``) or be fixed.

    Score
    -----
    * CHASE win  → ``1 - t / maxTime``  (earlier tag = higher score)
    * EVADE win  → ``t / maxTime``      (surviving longer = higher score)
    * Loss       → ``0``

    This matches the MyoChallenge ``chasetag_v0`` scoring function.
    """

    # --- Episode length (MyoChallenge: 20 s) ---
    maxTime: float = 20.0  # seconds
    ctrl_dt: float = 0.01
    sim_dt: float = 0.002

    @property
    def max_episode_steps(self) -> int:
        return round(self.maxTime / self.ctrl_dt)

    # --- Role assignment ---
    # "CHASE" | "EVADE" | "random"
    task_choice: str = "random"

    # --- Agent placement ---
    agent_separation_m: float = 2.0

    # --- Tag detection (MyoChallenge: 0.5 m) ---
    tag_radius_m: float = 0.5
    # One step of tag = full KO threshold → instant episode end via the
    # standard health/KO contract in MultiAgentTaskEnv.
    ko_health_threshold: float = 1.0

    # --- Fall detection ---
    fall_pelvis_z_threshold: float = 0.7

    # --- Reward weights ---
    r_win_bonus: float = 1.0  # one-shot bonus on win (tag or survive)
    r_loss_penalty: float = -1.0  # one-shot penalty on loss
    r_fall_penalty: float = -5.0  # one-shot penalty for falling
    r_survival_per_step: float = 0.01  # small per-step alive reward
    r_act_reg: float = -0.001  # L2 on muscle activations
    # Proximity shaping (chaser closes, runner escapes)
    r_chaser_proximity: float = 0.5
    r_runner_proximity: float = -0.5

    # --- Observation flags ---
    observe_own_joints: bool = True
    observe_own_muscles: bool = True
    observe_opponent_pelvis: bool = True
    observe_role: bool = True
