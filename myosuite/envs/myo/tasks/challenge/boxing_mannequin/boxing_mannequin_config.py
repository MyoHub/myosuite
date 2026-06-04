# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Configuration dataclass for the boxer-vs-mannequin task."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BoxingMannequinConfig:
    """Hyperparameters for the boxer-vs-mannequin MuJoCo task.

    Args:
        max_episode_steps: Episode horizon in control steps.
        ctrl_dt: Control interval in seconds.
        sim_dt: MuJoCo integration timestep in seconds.
        agent_separation_m: Initial distance between boxer and mannequin.
        disable_fingers: Whether to remove finger joints from MuscleMimic boxer.
        hit_threshold_head_m: Fist-to-head-zone hit radius.
        hit_threshold_body_m: Fist-to-chest-zone hit radius.
        hit_threshold_waist_m: Fist-to-waist-zone hit radius.
        w_head: Head-hit damage weight.
        w_body: Chest-hit damage weight.
        w_waist: Waist-hit damage weight.
        ko_health_threshold: Accumulated damage threshold for KO.
        fall_pelvis_z_threshold: Boxer pelvis-Z threshold for fall detection.
        r_damage_delivered: Dense reward scale for dealing damage.
        r_damage_received: Dense reward scale for receiving damage.
        r_ko_bonus: Bonus when opponent is KO'd.
        r_fall_bonus: Bonus when opponent falls.
        r_self_fall_penalty: Penalty when learner falls.
        r_survival_per_step: Per-step alive reward.
        r_act_reg: L2 action regularization scale.
        mannequin_guard_y: Mannequin glove guard Y offset in local frame.
        mannequin_guard_z: Mannequin glove guard Z offset in local frame.
        attack_interval_sec_min: Minimum interval between punch starts.
        attack_interval_sec_max: Maximum interval between punch starts.
        attack_duration_sec_min: Minimum duration of one scripted punch.
        attack_duration_sec_max: Maximum duration of one scripted punch.
        attack_target_noise_xy_m: XY noise added to scripted attack targets.
        attack_target_noise_z_m: Z noise added to scripted attack targets.
        target_hit_normal_force_threshold_n: Normal-force threshold in Newtons
            used to mark a pad as hit in observations.
        include_boxing_targets_scene: Whether to include the static 6-target
            boxing machine scene from ``boxing_8targets_scene.xml``.
    """

    max_episode_steps: int = 500
    ctrl_dt: float = 0.01
    sim_dt: float = 0.002
    agent_separation_m: float = 0.6

    disable_fingers: bool = True

    hit_threshold_head_m: float = 0.15
    hit_threshold_body_m: float = 0.18
    hit_threshold_waist_m: float = 0.2
    w_head: float = 2.0
    w_body: float = 1.0
    w_waist: float = 0.7

    ko_health_threshold: float = 100.0
    fall_pelvis_z_threshold: float = 0.7

    r_damage_delivered: float = 0.1
    r_damage_received: float = -0.1
    r_ko_bonus: float = 10.0
    r_fall_bonus: float = 5.0
    r_self_fall_penalty: float = -5.0
    r_survival_per_step: float = 0.01
    r_act_reg: float = -0.001

    mannequin_guard_y: float = 0.25
    mannequin_guard_z: float = 0.25
    attack_interval_sec_min: float = 0.2
    attack_interval_sec_max: float = 0.8
    attack_duration_sec_min: float = 0.12
    attack_duration_sec_max: float = 0.35
    attack_target_noise_xy_m: float = 0.06
    attack_target_noise_z_m: float = 0.04
    target_hit_normal_force_threshold_n: float = 15.0
    include_boxing_targets_scene: bool = False
