# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""TaskConfig for myoChallengeSaberP0-v0."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

from myosuite.core.config import (
    BackendConfig,
    GoalSpec,
    ObsSpec,
    RewardSpec,
    TaskConfig,
)
from myosuite.envs.myo.tasks.challenge.saber.saber_target_pool import (
    SABER_POOL_ADVANCE_DIRECTION,
    SABER_POOL_GRAVEYARD_Z,
    SABER_POOL_IDEAL_HIT_Y_RANGE,
    SABER_POOL_MAX_TARGET_MISSES,
    SABER_POOL_KILL_PLANE_Y,
    SABER_POOL_MISS_ZONE_Y,
    SABER_POOL_OBSTACLE_FRACTION,
    SABER_POOL_RECYCLE_CONTACT_SUBSTRINGS,
    SABER_POOL_SPAWN_INTERVAL_STEPS,
    SABER_POOL_SPAWN_PORTAL_X,
    SABER_POOL_SPAWN_PORTAL_Y,
    SABER_POOL_SPAWN_PORTAL_Z,
    SABER_POOL_SPEED_MPS_RANGE,
    SABER_POOL_SPHERE_RADIUS,
    SABER_TARGET_POOL_SIZE,
)
from myosuite.envs.myo.tasks.challenge.saber_scene_assets import (
    saber_scene_callable,
)


@dataclass
class SaberP0Cfg:
    """Typed configuration for the saber target-pool task.

    Replaces the untyped ``reward.extra`` dict so callers get IDE completion
    and static type checking instead of raw string-key lookups.
    """

    # --- target pool geometry / spawning ---
    pool_size: int = SABER_TARGET_POOL_SIZE
    graveyard_z: float = SABER_POOL_GRAVEYARD_Z
    spawn_interval_steps: int = SABER_POOL_SPAWN_INTERVAL_STEPS
    sphere_radius: float = SABER_POOL_SPHERE_RADIUS
    kill_plane_y: float = SABER_POOL_KILL_PLANE_Y
    miss_zone_y: float = SABER_POOL_MISS_ZONE_Y
    spawn_portal_x: tuple[float, float] = SABER_POOL_SPAWN_PORTAL_X
    spawn_portal_y: tuple[float, float] = SABER_POOL_SPAWN_PORTAL_Y
    spawn_portal_z: tuple[float, float] = SABER_POOL_SPAWN_PORTAL_Z
    advance_direction: tuple[float, float, float] = SABER_POOL_ADVANCE_DIRECTION
    saber_target_speed_mps_range: tuple[float, float] = SABER_POOL_SPEED_MPS_RANGE
    ideal_hit_y_range: tuple[float, float] = SABER_POOL_IDEAL_HIT_Y_RANGE
    obstacle_fraction: float = SABER_POOL_OBSTACLE_FRACTION

    # --- contact recycling ---
    recycle_on_body_or_helmet_contact: bool = True
    recycle_contact_geom_substrings: tuple[str, ...] = field(
        default_factory=lambda: tuple(SABER_POOL_RECYCLE_CONTACT_SUBSTRINGS)
    )

    # --- hit scoring weights ---
    w_hit: float = 2.0
    w_correct_hand: float = 4.0
    w_correct_face: float = 3.0
    slicing_accuracy_epsilon: float = 1.0e-3

    # --- termination ---
    max_target_misses: int | None = SABER_POOL_MAX_TARGET_MISSES
    use_health_status: bool = False

    # --- preferred-hand / hit-area enforcement ---
    enforce_preferred_hand: bool = True
    enforce_preferred_hit_area: bool = True

    # --- posture ---
    upright_body_name: str = "torso"
    upright_posture_threshold: float = 0.30

    # --- keyframe pose debug term ---
    saber_keyframe_pose_error_scale: float = 40.0

    # --- flags used by registration only ---
    enable_saber_target_pool: bool = True
    drive_saber_targets_kinematics: bool = False


@dataclass
class SaberP0Task(TaskConfig):
    """CPU prototype: dual saber vs a 100-cube mocap pool (top-10 obs)."""

    model: str = "musclemimic_myotorso_bimanual_fingers"
    scene: Callable = saber_scene_callable
    max_episode_steps: int = 1000  # 6000
    use_health_status: bool = False
    muscle_fatigue: bool = False
    saber_cfg: SaberP0Cfg = field(default_factory=SaberP0Cfg)
    backend: BackendConfig = field(
        default_factory=lambda: BackendConfig(
            n_substeps=5,
            ctrl_dt=0.01,
            sim_dt=0.002,
            extra={"enforce_step_timing_alignment": True},
        )
    )
    obs: ObsSpec = field(
        default_factory=lambda: ObsSpec(
            keys=[
                "joint_pos",
                "joint_vel",
                "muscle_act",
                "saber_target_pool",
                "time",
            ],
            extra={
                "left_saber_site_name": "left_saber_tip",
                "right_saber_site_name": "right_saber_tip",
                "num_visible": 10,
                "site_prefix": "saber_target_",
                "site_suffix": "_site",
            },
        )
    )
    goal: GoalSpec = field(
        default_factory=lambda: GoalSpec(target_type="site_positions")
    )
    reward: RewardSpec = field(
        default_factory=lambda: RewardSpec(
            terms=[
                "saber_target_pool",
                "upright_posture",
                "saber_keyframe_pose",
                "movement_efficiency",
            ],
            weights={
                "saber_target_pool": 1.0,
                "upright_posture": 2.0,
                "saber_keyframe_pose": 1.0,
                "movement_efficiency": 0.01,
            },
        )
    )

    def __post_init__(self) -> None:
        """Sync use_health_status; populate obs.extra for CPU obs terms."""
        if "health_status" not in self.obs.keys:
            self.obs.keys.append("health_status")
        # saber_cfg is authoritative when set explicitly by the caller.
        # The top-level use_health_status field only wins when saber_cfg was
        # not customised (i.e. still holds the dataclass default of False).
        if not self.saber_cfg.use_health_status:
            self.saber_cfg.use_health_status = bool(self.use_health_status)
        # obs.extra carries pool layout parameters consumed by saber_target_pool_obs.
        self.obs.extra["pool_size"] = self.saber_cfg.pool_size
        self.obs.extra["graveyard_z"] = self.saber_cfg.graveyard_z


def saber_term_params(cfg: SaberP0Cfg) -> dict[str, Any]:
    """Return the kwargs that saber obs/reward/termination terms expect from config.

    Both the CPU ``ModularTaskEnv`` (passed flat via ``**extra`` in
    ``get_reward_dict``) and the mjlab ``register_mjlab_saber`` path (passed as
    per-term ``params=``) use this function so the field mapping lives in one
    place.  Term functions accept ``**kwargs`` and ignore unrecognised keys, so
    it is safe to pass the full dict to every term.
    """
    return {
        # saber_target_pool_reward
        "w_hit": float(cfg.w_hit),
        "w_correct_hand": float(cfg.w_correct_hand),
        "w_correct_face": float(cfg.w_correct_face),
        # upright_posture_reward / upright_posture_failure
        "upright_posture_threshold": float(cfg.upright_posture_threshold),
        # saber_keyframe_pose_reward
        "saber_keyframe_pose_error_scale": float(cfg.saber_keyframe_pose_error_scale),
        # saber_pool_done
        "max_target_misses": cfg.max_target_misses,
        "use_health_status": bool(cfg.use_health_status),
    }
