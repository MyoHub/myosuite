# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Configuration dataclass for the two-agent competitive boxing environment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BoxingVsConfig:
    """All hyperparameters for ``BoxingVsEnv``.

    Physics timing
    --------------
    ctrl_dt and sim_dt must satisfy ``ctrl_dt / sim_dt == n_substeps`` (integer).
    The defaults match the MuscleMimic fullbody CPU compile path.

    Agent placement
    ---------------
    The MuscleMimic fullbody model faces **−Y** by default (matching the P0
    boxing bag placement).  Agent 0 is therefore placed at positive Y and uses
    the default orientation so it faces −Y toward agent 1.  Agent 1 is placed
    at negative Y and rotated 180° around Z so it also faces its opponent.

    Health / damage
    ---------------
    Damage accumulates from proximity-based fist hits.  An agent is knocked out
    when its health reaches ``ko_health_threshold``.  The fall condition is
    independent: any agent whose pelvis drops below ``fall_com_z_threshold``
    metres is considered down regardless of health.

    Reward
    ------
    All weights default to values that make a decisive round-ending event (KO /
    fall) worth ~10× the per-step signal so the curriculum is clear.
    """

    # --- Episode length ---
    max_episode_steps: int = 500  # 500 * 0.01 s = 5 s per round
    ctrl_dt: float = 0.01  # control timestep (s)
    sim_dt: float = 0.002  # MuJoCo simulation timestep (s)

    # --- Agent placement ---
    agent_separation_m: float = 1.6  # centre-to-centre distance along Y axis (m)

    # --- Model ---
    disable_fingers: bool = True  # remove finger joints/muscles for speed

    # --- Hit detection (proximity-based, matching BoxingP0) ---
    hit_threshold_head_m: float = 0.15  # fist-to-head-zone proximity threshold (m)
    hit_threshold_body_m: float = 0.18  # fist-to-body-zone proximity threshold (m)
    w_head: float = 2.0  # head-hit damage multiplier
    w_body: float = 1.0  # body-hit damage multiplier

    # --- Health / KO ---
    ko_health_threshold: float = 100.0  # cumulative damage that triggers KO

    # --- Fall detection ---
    fall_pelvis_z_threshold: float = 0.7  # pelvis Z below this = agent is down (m)

    # --- Reward weights ---
    r_damage_delivered: float = 0.1  # per unit of damage dealt this step
    r_damage_received: float = -0.1  # per unit of damage received this step
    r_ko_bonus: float = 10.0  # one-shot bonus for knocking out opponent
    r_fall_bonus: float = 5.0  # one-shot bonus for flooring opponent
    r_self_fall_penalty: float = -5.0  # one-shot penalty for falling down
    r_survival_per_step: float = 0.01  # small per-step alive reward
    r_act_reg: float = -0.001  # L2 regularisation on muscle activations

    # --- Observation flags ---
    observe_own_joints: bool = True
    observe_own_muscles: bool = True
    observe_opponent_keypoints: bool = True  # fist + head + body + pelvis positions
    observe_opponent_velocities: bool = True  # fist + pelvis linear velocities
    observe_opponent_joint_subset: bool = True  # shoulder + elbow joints (~20 floats)
    observe_health: bool = True  # own + opponent normalised health
