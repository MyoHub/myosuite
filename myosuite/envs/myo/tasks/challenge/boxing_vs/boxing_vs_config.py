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

    # ------------------------------------------------------------------
    # Reward v1 options (capstone 2026).  Every default below reproduces the
    # legacy behaviour exactly; see docs/boxing_vs_reward_v1.md.
    # ------------------------------------------------------------------

    # --- Model: physical contact between the two boxers ---
    # Per-agent 3-bit collision masks: the boxers collide with each other and
    # with the floor, never with themselves; fingers stay floor-only.  Adds a
    # massless collision sphere per glove.
    cross_agent_collisions: bool = False
    glove_collision_radius_m: float = 0.05
    glove_solref: tuple[float, float] = (0.02, 1.0)
    # The visual glove/helmet meshes add ~2.1 kg per hand and ~5.8 kg to the
    # head (inertia is computed from all geoms).  True makes them massless so
    # the bodies match the MuscleMimic model the pretrained policy was trained on.
    massless_visual_gear: bool = False

    # --- Hit detection ---
    # "proximity": legacy fist-to-zone distance, every step.
    # "contact": one event per glove impact (rising edge + cooldown), damage
    # scaled by the approach speed along the contact normal.  Needs
    # ``cross_agent_collisions=True``.
    hit_detection: str = "proximity"
    hit_min_speed_mps: float = 1.5  # slower impacts are touches, not hits
    hit_full_speed_mps: float = 4.5  # damage saturates at this speed
    hit_cooldown_s: float = 0.15  # min time between two events of one glove

    # --- Reward composition ---
    # "legacy": combine_combat_reward (default).  "v1": see docs.
    reward_mode: str = "legacy"
    # "legacy": +r_fall_bonus whenever the opponent falls, r_self_fall_penalty
    # when falling.  "attributed": the opponent's fall only pays if this agent
    # hit it within ``knockdown_window_s``; a double fall gives
    # ``r_double_fall_penalty`` to both.  reward_mode="v1" always uses it.
    fall_reward_mode: str = "legacy"
    knockdown_window_s: float = 1.0
    r_double_fall_penalty: float = -5.0
    # v1 weights
    r_hit: float = 1.0  # x damage dealt (and -r_hit x damage received)
    shaping_alpha: float = 1.0  # multiplies alive/reach/facing; anneal 1 -> 0
    r_alive: float = 0.01  # per step while standing
    r_reach: float = 1.0  # per metre the closest fist gets closer to a target
    r_facing: float = 0.01  # per step x cos(angle between heading and opponent)
    r_effort: float = -0.1  # x mean(clip(action, 0, 1) ** 2)

    # --- Reset / opponent ---
    # "default": model rest pose.  "clip": joint angles and root height/tilt
    # from ``reset_clip_frame`` of ``reset_clip_path`` (placement kept).
    reset_pose: str = "default"
    reset_clip_path: str = (
        "boxing_motions/MyoFullBody/gmr/Transitions_mocap/mazen_c3d/"
        "punchboxing_push_poses.npz"
    )
    reset_clip_frame: int = 0
    reset_noise_separation_m: float = 0.0  # uniform +/- on the start distance
    reset_noise_lateral_m: float = 0.0  # uniform +/- per agent along X
    reset_noise_yaw_deg: float = 0.0  # uniform +/- per agent around Z
    reset_noise_joint_rad: float = 0.0  # uniform +/- on every hinge joint
    # None, "agent_0" or "agent_1": that agent is held kinematically in its
    # reset pose every control step (a standing dummy that cannot fall).
    dummy_opponent: str | None = None


def boxing_vs_level1_config(**overrides) -> BoxingVsConfig:
    """Recommended Level-1 setup: learner = agent_0 vs a held dummy (agent_1).

    Contact-based hits, reward v1, clip start pose, massless visual gear and
    small reset noise.  ``agent_separation_m`` puts the boxers close; tune it
    with ``overrides``.
    """
    params = {
        "cross_agent_collisions": True,
        "massless_visual_gear": True,
        "hit_detection": "contact",
        "reward_mode": "v1",
        "fall_reward_mode": "attributed",
        "reset_pose": "clip",
        "reset_noise_separation_m": 0.05,
        "reset_noise_lateral_m": 0.05,
        "reset_noise_yaw_deg": 10.0,
        "reset_noise_joint_rad": 0.02,
        "dummy_opponent": "agent_1",
        "agent_separation_m": 0.4,
    }
    params.update(overrides)
    return BoxingVsConfig(**params)
