# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""mjlab registration for the saber task (myoChallengeSaberP0-v0).

This module owns only registration — env config assembly, mjlab task registration,
and the public helpers callers use to wire up variants. All env logic (term
functions, task state, reset events, action classes) lives in saber_mjlab_env.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
        _init_state_from_model,
        _mimic_early_termination,
        _mimic_obs_act,
        _mimic_obs_clip_phase,
        _mimic_obs_clip_ref_qpos,
        _mimic_obs_clip_ref_qvel,
        _mimic_obs_err,
        _mimic_obs_lookahead,
        _mimic_obs_site_pos,
        _mimic_obs_target,
        _mimic_tracking_reward,
        _muscle_tendon_names,
        _normalize_mimic_reward_mode,
        _reward_mode_uses_env_objective,
        _reward_mode_uses_mimic_objective,
        _strip_spec_keyframes,
    )
except ModuleNotFoundError as _e:
    raise ModuleNotFoundError(
        "myosuite mjlab saber registration requires the 'musclemimic' integration: "
        "pip install myosuite[musclemimic]"
    ) from _e
from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import (
    MyoMuscleActivationActionCfg,
)
from myosuite.envs.myo.backends.mjlab.saber_mjlab_env import (
    _SABER_ENTITY_NAME,
    _LEFT_SABER_ENTITY_NAME,
    _RIGHT_SABER_ENTITY_NAME,
    SaberTaskLogicState,
    _get_saber_logic,
    _saber_obs_from_shared,
    _saber_reward_dense_from_shared,
    _saber_termination_from_shared,
    _saber_keyframe_reset_event,
    _saber_mimic_qpos_obs,
    _saber_mimic_qvel_obs,
    _saber_mimic_rsi_event,
    _saber_reset_event,
    _saber_startup_event,
    _saber_target_pool_obs,
    _target_entity_name,
)
from myosuite.envs.myo.tasks.challenge.saber_scene_assets import (
    build_isolated_saber_target_spec,
    build_left_saber_spec,
    build_right_saber_spec,
    build_saber_body_mjmodel_and_spec,
    build_saber_body_spec,
    build_saber_scene_mjmodel_and_spec,
)
from myosuite.envs.myo.tasks.challenge.saber_task_config import (
    SaberP0Task,
    saber_term_params,
)
from myosuite.envs.myo.tasks.challenge.saber_task_spec import (
    SABER_P0_ENV_ID,
    SABER_P0_MIMIC_ENV_ID,
)
from myosuite.terms.base_obs import (
    health_status_obs,
    joint_pos_obs,
    joint_vel_obs,
    muscle_act_obs,
    time_obs,
)
from myosuite.terms.base_reward import (
    movement_efficiency_reward,
    saber_keyframe_pose_reward,
    saber_target_pool_reward,
    upright_posture_reward,
)
from myosuite.terms.base_termination import (
    saber_pool_done,
    upright_posture_failure,
)

if TYPE_CHECKING:
    from myosuite.core.trajectory_io import MotionClip


@dataclass
class SaberP0MjlabCfg:
    """Legacy public config shim retained for mjlab registration/export hooks."""

    sim_dt: float = 0.002
    ctrl_dt: float = 0.01
    max_episode_steps: int = 1000
    num_envs: int = 256


@dataclass(frozen=True)
class SaberMimicMixCfg:
    """Optional mimic augmentation passed to the with-mimic registration helper."""

    clips: tuple[MotionClip, ...]
    reward_mode: str = "augmented"
    mimic_reward_weight: float = 5.0
    env_reward_weight: float = 1.0
    use_deepmimic_reward: bool = True
    use_lookahead: bool = True
    use_early_termination: bool = False
    use_reference_state_initialization: bool = False

    def __post_init__(self) -> None:
        clips = tuple(self.clips)
        if not clips:
            raise ValueError("SaberMimicMixCfg requires at least one motion clip.")
        object.__setattr__(self, "clips", clips)
        object.__setattr__(
            self,
            "reward_mode",
            _normalize_mimic_reward_mode(self.reward_mode),
        )


def _camera_sensor_cfg_from_free_camera(
    *,
    name: str,
    mj_model: Any,
    lookat: tuple[float, float, float] | list[float],
    distance: float,
    azimuth: float,
    elevation: float,
    width: int,
    height: int,
) -> Any:
    import mujoco
    from mjlab.sensor import CameraSensorCfg

    data = mujoco.MjData(mj_model)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(mj_model, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE.value
    cam.fixedcamid = -1
    cam.trackbodyid = -1
    cam.lookat[:] = np.asarray(lookat, dtype=np.float64)
    cam.distance = float(distance)
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)

    headpos = np.zeros(3, dtype=np.float64)
    forward = np.zeros(3, dtype=np.float64)
    up = np.zeros(3, dtype=np.float64)
    right = np.zeros(3, dtype=np.float64)
    mujoco.mjv_cameraFrame(headpos, forward, up, right, data, cam)

    cam_rot = np.column_stack((right, up, -forward)).reshape(9)
    cam_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(cam_quat, cam_rot)

    return CameraSensorCfg(
        name=name,
        pos=tuple(float(x) for x in headpos),
        quat=tuple(float(x) for x in cam_quat),
        fovy=float(mj_model.vis.global_.fovy),
        width=int(width),
        height=int(height),
        data_types=("rgb",),
        use_textures=True,
        use_shadows=False,
    )


def _make_saber_env_cfg(*, play: bool = False) -> Any:
    """Build the default (no mimic) mjlab env config for the saber task."""
    from mjlab.actuator.actuator import TransmissionType
    from mjlab.envs.mdp import terminations as mdp_terminations
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        _XmlWrappedActuatorCfg,
    )

    task_cfg = SaberP0Task()
    scene_mj_model, _ = build_saber_scene_mjmodel_and_spec()
    robot_mj_model, _ = build_saber_body_mjmodel_and_spec()
    keyframe_reset_fn = _saber_keyframe_reset_event(_SABER_ENTITY_NAME, robot_mj_model)
    tendons = _muscle_tendon_names(robot_mj_model)

    def _robot_spec_fn() -> Any:
        spec = build_saber_body_spec()
        _strip_spec_keyframes(spec)
        return spec

    articulation = EntityArticulationInfoCfg(
        actuators=(
            _XmlWrappedActuatorCfg(
                target_names_expr=tendons,
                transmission_type=TransmissionType.TENDON,
            ),
        )
    )
    init_state = _init_state_from_model(robot_mj_model)
    robot_entity_cfg = EntityCfg(
        spec_fn=_robot_spec_fn,
        articulation=articulation,
        init_state=init_state,
    )
    scene_entities: dict[str, Any] = {
        _SABER_ENTITY_NAME: robot_entity_cfg,
        _LEFT_SABER_ENTITY_NAME: EntityCfg(spec_fn=build_left_saber_spec),
        _RIGHT_SABER_ENTITY_NAME: EntityCfg(spec_fn=build_right_saber_spec),
    }
    saber_cfg = task_cfg.saber_cfg
    for index in range(saber_cfg.pool_size):
        scene_entities[_target_entity_name(index)] = EntityCfg(
            spec_fn=(
                lambda target_index=index: build_isolated_saber_target_spec(
                    target_index
                )
            )
        )
    scene_cfg = SceneCfg(
        num_envs=1 if play else SaberP0MjlabCfg().num_envs,
        entities=scene_entities,
    )
    if play:
        scene_cfg.sensors = (
            _camera_sensor_cfg_from_free_camera(
                name="observer_camera",
                mj_model=scene_mj_model,
                lookat=(-0.35, -0.25, 1.1),
                distance=2.5,
                azimuth=165.0,
                elevation=-14.0,
                width=640,
                height=480,
            ),
            _camera_sensor_cfg_from_free_camera(
                name="game_camera",
                mj_model=scene_mj_model,
                lookat=(0.0, -0.25, 1.1),
                distance=0.25,
                azimuth=-90.0,
                elevation=-14.0,
                width=640,
                height=480,
            ),
        )

    obs_terms: dict[str, Any] = {
        "joint_pos": ObservationTermCfg(
            func=lambda env, **kwargs: _saber_obs_from_shared(
                env, joint_pos_obs, **kwargs
            )
        ),
        "joint_vel": ObservationTermCfg(
            func=lambda env, **kwargs: _saber_obs_from_shared(
                env, joint_vel_obs, **kwargs
            )
        ),
        "muscle_act": ObservationTermCfg(
            func=lambda env, **kwargs: _saber_obs_from_shared(
                env, muscle_act_obs, **kwargs
            )
        ),
        "saber_target_pool": ObservationTermCfg(func=_saber_target_pool_obs),
        "time": ObservationTermCfg(
            func=lambda env, **kwargs: _saber_obs_from_shared(env, time_obs, **kwargs)
        ),
        "health_status": ObservationTermCfg(
            func=lambda env, **kwargs: _saber_obs_from_shared(
                env,
                health_status_obs,
                include_task_state=True,
                **kwargs,
            )
        ),
    }
    observations = {
        "policy": ObservationGroupCfg(terms=obs_terms),
        "actor": ObservationGroupCfg(terms=obs_terms),
        "critic": ObservationGroupCfg(terms=obs_terms),
    }
    ctrl_dt = float(task_cfg.backend.ctrl_dt)
    _term_params = saber_term_params(saber_cfg)
    rewards: dict[str, Any] = {
        "saber_target_pool": RewardTermCfg(
            func=lambda env, **kwargs: _saber_reward_dense_from_shared(
                env,
                saber_target_pool_reward,
                **kwargs,
            ),
            weight=float(task_cfg.reward.weights["saber_target_pool"]) / ctrl_dt,
            params=_term_params,
        ),
        "upright_posture": RewardTermCfg(
            func=lambda env, **kwargs: _saber_reward_dense_from_shared(
                env,
                upright_posture_reward,
                **kwargs,
            ),
            weight=float(task_cfg.reward.weights["upright_posture"]) / ctrl_dt,
            params=_term_params,
        ),
        "saber_keyframe_pose": RewardTermCfg(
            func=lambda env, **kwargs: _saber_reward_dense_from_shared(
                env,
                saber_keyframe_pose_reward,
                **kwargs,
            ),
            weight=float(task_cfg.reward.weights["saber_keyframe_pose"]) / ctrl_dt,
            params=_term_params,
        ),
        "movement_efficiency": RewardTermCfg(
            func=lambda env, **kwargs: _saber_reward_dense_from_shared(
                env,
                movement_efficiency_reward,
                **kwargs,
            ),
            weight=float(task_cfg.reward.weights["movement_efficiency"]) / ctrl_dt,
            params=_term_params,
        ),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=mdp_terminations.time_out, time_out=True),
        "nan_detection": TerminationTermCfg(
            func=mdp_terminations.nan_detection,
            time_out=False,
        ),
        "saber_done": TerminationTermCfg(
            func=lambda env, **kwargs: _saber_termination_from_shared(
                env,
                saber_pool_done,
                **kwargs,
            ),
            time_out=False,
            params=_term_params,
        ),
        "upright_posture_failure": TerminationTermCfg(
            func=lambda env, **kwargs: _saber_termination_from_shared(
                env,
                upright_posture_failure,
                **kwargs,
            ),
            time_out=False,
            params=_term_params,
        ),
    }
    events = {
        "saber_startup": EventTermCfg(
            func=_saber_startup_event,
            mode="startup",
            params={"keyframe_reset_fn": keyframe_reset_fn},
        ),
        "saber_step": EventTermCfg(
            func=SaberTaskLogicState,
            mode="step",
            params={"task_cfg": task_cfg, "mj_model": robot_mj_model},
        ),
        "saber_reset": EventTermCfg(
            func=_saber_reset_event,
            mode="reset",
            params={"keyframe_reset_fn": keyframe_reset_fn},
        ),
    }
    return _build_manager_env_cfg(
        task_cfg=task_cfg,
        scene_cfg=scene_cfg,
        observations=observations,
        rewards=rewards,
        terminations=terminations,
        events=events,
        tendons=tendons,
    )


def _make_saber_mimic_env_cfg(
    *,
    play: bool = False,
    mimic_mix: SaberMimicMixCfg,
) -> Any:
    """Build the mimic-augmented mjlab env config for the saber task."""
    from mjlab.actuator.actuator import TransmissionType
    from mjlab.envs.mdp import terminations as mdp_terminations
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        _XmlWrappedActuatorCfg,
    )

    task_cfg = SaberP0Task()
    scene_mj_model, _ = build_saber_scene_mjmodel_and_spec()
    robot_mj_model, _ = build_saber_body_mjmodel_and_spec()
    keyframe_reset_fn = _saber_keyframe_reset_event(_SABER_ENTITY_NAME, robot_mj_model)
    tendons = _muscle_tendon_names(robot_mj_model)
    ctrl_dt = float(task_cfg.backend.ctrl_dt)

    mix_clips = mimic_mix.clips if len(mimic_mix.clips) > 1 else mimic_mix.clips[0]
    clip_state_available = all(
        clip.qpos is not None and clip.qvel is not None for clip in mimic_mix.clips
    )
    use_rsi = clip_state_available and mimic_mix.use_reference_state_initialization

    def _robot_spec_fn() -> Any:
        spec = build_saber_body_spec()
        _strip_spec_keyframes(spec)
        return spec

    articulation = EntityArticulationInfoCfg(
        actuators=(
            _XmlWrappedActuatorCfg(
                target_names_expr=tendons,
                transmission_type=TransmissionType.TENDON,
            ),
        )
    )
    init_state = _init_state_from_model(robot_mj_model)
    robot_entity_cfg = EntityCfg(
        spec_fn=_robot_spec_fn,
        articulation=articulation,
        init_state=init_state,
    )
    scene_entities: dict[str, Any] = {
        _SABER_ENTITY_NAME: robot_entity_cfg,
        _LEFT_SABER_ENTITY_NAME: EntityCfg(spec_fn=build_left_saber_spec),
        _RIGHT_SABER_ENTITY_NAME: EntityCfg(spec_fn=build_right_saber_spec),
    }
    saber_cfg = task_cfg.saber_cfg
    for index in range(saber_cfg.pool_size):
        scene_entities[_target_entity_name(index)] = EntityCfg(
            spec_fn=(
                lambda target_index=index: build_isolated_saber_target_spec(
                    target_index
                )
            )
        )
    scene_cfg = SceneCfg(
        num_envs=1 if play else SaberP0MjlabCfg().num_envs,
        entities=scene_entities,
    )
    if play:
        scene_cfg.sensors = (
            _camera_sensor_cfg_from_free_camera(
                name="observer_camera",
                mj_model=scene_mj_model,
                lookat=(-0.35, -0.25, 1.1),
                distance=2.5,
                azimuth=165.0,
                elevation=-14.0,
                width=640,
                height=480,
            ),
            _camera_sensor_cfg_from_free_camera(
                name="game_camera",
                mj_model=scene_mj_model,
                lookat=(0.0, -0.25, 1.1),
                distance=0.25,
                azimuth=-90.0,
                elevation=-14.0,
                width=640,
                height=480,
            ),
        )

    obs_terms: dict[str, Any] = {
        "joint_pos": ObservationTermCfg(
            func=lambda env, **kwargs: _saber_obs_from_shared(
                env, joint_pos_obs, **kwargs
            )
        ),
        "joint_vel": ObservationTermCfg(
            func=lambda env, **kwargs: _saber_obs_from_shared(
                env, joint_vel_obs, **kwargs
            )
        ),
        "muscle_act": ObservationTermCfg(
            func=lambda env, **kwargs: _saber_obs_from_shared(
                env, muscle_act_obs, **kwargs
            )
        ),
        "saber_target_pool": ObservationTermCfg(func=_saber_target_pool_obs),
        "time": ObservationTermCfg(
            func=lambda env, **kwargs: _saber_obs_from_shared(env, time_obs, **kwargs)
        ),
        "health_status": ObservationTermCfg(
            func=lambda env, **kwargs: _saber_obs_from_shared(
                env,
                health_status_obs,
                include_task_state=True,
                **kwargs,
            )
        ),
        "qpos": ObservationTermCfg(func=_saber_mimic_qpos_obs),
        "qvel": ObservationTermCfg(func=_saber_mimic_qvel_obs),
        "act": ObservationTermCfg(func=_mimic_obs_act(_SABER_ENTITY_NAME)),
        "mimic_site_pos": ObservationTermCfg(
            func=_mimic_obs_site_pos(_SABER_ENTITY_NAME, "saber", mix_clips, ctrl_dt)
        ),
        "mimic_site_target": ObservationTermCfg(
            func=_mimic_obs_target(_SABER_ENTITY_NAME, "saber", mix_clips, ctrl_dt)
        ),
        "mimic_site_err": ObservationTermCfg(
            func=_mimic_obs_err(_SABER_ENTITY_NAME, "saber", mix_clips, ctrl_dt)
        ),
        "clip_phase": ObservationTermCfg(
            func=_mimic_obs_clip_phase(_SABER_ENTITY_NAME, "saber", mix_clips, ctrl_dt)
        ),
    }
    if clip_state_available:
        obs_terms["clip_ref_qpos"] = ObservationTermCfg(
            func=_mimic_obs_clip_ref_qpos(
                _SABER_ENTITY_NAME, "saber", mix_clips, ctrl_dt
            )
        )
        obs_terms["clip_ref_qvel"] = ObservationTermCfg(
            func=_mimic_obs_clip_ref_qvel(
                _SABER_ENTITY_NAME, "saber", mix_clips, ctrl_dt
            )
        )
    if mimic_mix.use_lookahead:
        obs_terms["mimic_lookahead"] = ObservationTermCfg(
            func=_mimic_obs_lookahead(_SABER_ENTITY_NAME, "saber", mix_clips, ctrl_dt)
        )
    observations = {
        "policy": ObservationGroupCfg(terms=obs_terms),
        "actor": ObservationGroupCfg(terms=obs_terms),
        "critic": ObservationGroupCfg(terms=obs_terms),
    }
    reward_mode = mimic_mix.reward_mode
    env_scale = float(mimic_mix.env_reward_weight)
    _term_params = saber_term_params(saber_cfg)
    rewards: dict[str, Any] = {}
    if _reward_mode_uses_env_objective(reward_mode):
        rewards["saber_target_pool"] = RewardTermCfg(
            func=lambda env, **kwargs: _saber_reward_dense_from_shared(
                env,
                saber_target_pool_reward,
                **kwargs,
            ),
            weight=env_scale
            * float(task_cfg.reward.weights["saber_target_pool"])
            / ctrl_dt,
            params=_term_params,
        )
        rewards["upright_posture"] = RewardTermCfg(
            func=lambda env, **kwargs: _saber_reward_dense_from_shared(
                env,
                upright_posture_reward,
                **kwargs,
            ),
            weight=env_scale
            * float(task_cfg.reward.weights["upright_posture"])
            / ctrl_dt,
            params=_term_params,
        )
        rewards["saber_keyframe_pose"] = RewardTermCfg(
            func=lambda env, **kwargs: _saber_reward_dense_from_shared(
                env,
                saber_keyframe_pose_reward,
                **kwargs,
            ),
            weight=env_scale
            * float(task_cfg.reward.weights["saber_keyframe_pose"])
            / ctrl_dt,
            params=_term_params,
        )
        rewards["movement_efficiency"] = RewardTermCfg(
            func=lambda env, **kwargs: _saber_reward_dense_from_shared(
                env,
                movement_efficiency_reward,
                **kwargs,
            ),
            weight=env_scale
            * float(task_cfg.reward.weights["movement_efficiency"])
            / ctrl_dt,
            params=_term_params,
        )
    if _reward_mode_uses_mimic_objective(reward_mode):
        rewards["mimic_tracking"] = RewardTermCfg(
            func=_mimic_tracking_reward(
                _SABER_ENTITY_NAME, "saber", mix_clips, ctrl_dt
            ),
            weight=float(mimic_mix.mimic_reward_weight) / ctrl_dt,
        )
    terminations = {
        "time_out": TerminationTermCfg(func=mdp_terminations.time_out, time_out=True),
        "nan_detection": TerminationTermCfg(
            func=mdp_terminations.nan_detection, time_out=False
        ),
        "saber_done": TerminationTermCfg(
            func=lambda env, **kwargs: _saber_termination_from_shared(
                env,
                saber_pool_done,
                **kwargs,
            ),
            time_out=False,
            params=_term_params,
        ),
        "upright_posture_failure": TerminationTermCfg(
            func=lambda env, **kwargs: _saber_termination_from_shared(
                env,
                upright_posture_failure,
                **kwargs,
            ),
            time_out=False,
            params=_term_params,
        ),
    }
    if (
        _reward_mode_uses_mimic_objective(reward_mode)
        and mimic_mix.use_early_termination
    ):
        terminations["mimic_deviation"] = TerminationTermCfg(
            func=_mimic_early_termination(
                _SABER_ENTITY_NAME, "saber", mix_clips, ctrl_dt
            ),
            time_out=False,
        )
    events: dict[str, Any] = {
        "saber_startup": EventTermCfg(
            func=_saber_startup_event,
            mode="startup",
            params={"keyframe_reset_fn": keyframe_reset_fn},
        ),
        "saber_step": EventTermCfg(
            func=SaberTaskLogicState,
            mode="step",
            params={"task_cfg": task_cfg, "mj_model": robot_mj_model},
        ),
    }
    if use_rsi:
        events["mimic_rsi"] = EventTermCfg(
            func=_saber_mimic_rsi_event(mix_clips, ctrl_dt),
            mode="reset",
        )
        events["saber_reset"] = EventTermCfg(func=_saber_reset_event, mode="reset")
    else:
        events["saber_reset"] = EventTermCfg(
            func=_saber_reset_event,
            mode="reset",
            params={"keyframe_reset_fn": keyframe_reset_fn},
        )
    return _build_manager_env_cfg(
        task_cfg=task_cfg,
        scene_cfg=scene_cfg,
        observations=observations,
        rewards=rewards,
        terminations=terminations,
        events=events,
        tendons=tendons,
    )


def _build_manager_env_cfg(
    *,
    task_cfg: SaberP0Task,
    scene_cfg: Any,
    observations: dict[str, Any],
    rewards: dict[str, Any],
    terminations: dict[str, Any],
    events: dict[str, Any],
    tendons: tuple[str, ...],
) -> Any:
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.viewer import ViewerConfig

    return ManagerBasedRlEnvCfg(
        scene=scene_cfg,
        decimation=max(
            1, int(round(task_cfg.backend.ctrl_dt / task_cfg.backend.sim_dt))
        ),
        episode_length_s=float(task_cfg.max_episode_steps)
        * float(task_cfg.backend.ctrl_dt),
        observations=observations,
        actions={
            "muscles": MyoMuscleActivationActionCfg(
                entity_name=_SABER_ENTITY_NAME,
                actuator_names=tuple(t.removesuffix("_tendon") for t in tendons),
                tendon_names=tendons,
                action_mode="excitation",
                muscle_fatigue=task_cfg.muscle_fatigue,
                ctrl_dt=task_cfg.backend.ctrl_dt,
                pre_physics_fn=lambda env: _get_saber_logic(env).advance_target_pool(),
            )
        },
        rewards=rewards,
        terminations=terminations,
        events=events,
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=float(task_cfg.backend.sim_dt),
                integrator="euler",
                ccd_iterations=500,
            ),
            njmax=1024,
            nconmax=1024,
        ),
        viewer=ViewerConfig(width=640, height=480),
        auto_reset=True,
    )


def register_saber_p0_mjlab_task(
    register_mjlab_task: Any,
    rl_cfg_fn: Any,
    *,
    task_id: str = SABER_P0_ENV_ID,
) -> None:
    """Register the default saber mjlab env (no mimic augmentation)."""
    register_mjlab_task(
        task_id=task_id,
        env_cfg=_make_saber_env_cfg(play=False),
        play_env_cfg=_make_saber_env_cfg(play=True),
        rl_cfg=rl_cfg_fn(save_interval=150),
        runner_cls=None,
    )


def register_saber_p0_mjlab_task_with_mimic_mix(
    register_mjlab_task: Any,
    rl_cfg_fn: Any,
    mimic_mix: SaberMimicMixCfg | None = None,
    *,
    clips: tuple[MotionClip, ...] | None = None,
    task_id: str = SABER_P0_MIMIC_ENV_ID,
    reward_mode: str = "augmented",
    mimic_reward_weight: float = 5.0,
    env_reward_weight: float = 1.0,
    use_deepmimic_reward: bool = True,
    use_lookahead: bool = True,
    use_early_termination: bool = False,
    use_reference_state_initialization: bool = False,
) -> None:
    """Register the saber mjlab env augmented with mimic clip observations and rewards.

    Can be called either with a pre-built ``SaberMimicMixCfg`` (preferred) or
    with the individual keyword arguments for convenience.
    """
    if mimic_mix is None:
        if clips is None:
            raise ValueError("Either mimic_mix or clips must be provided.")
        mimic_mix = SaberMimicMixCfg(
            clips=clips,
            reward_mode=reward_mode,
            mimic_reward_weight=mimic_reward_weight,
            env_reward_weight=env_reward_weight,
            use_deepmimic_reward=use_deepmimic_reward,
            use_lookahead=use_lookahead,
            use_early_termination=use_early_termination,
            use_reference_state_initialization=use_reference_state_initialization,
        )
    register_mjlab_task(
        task_id=task_id,
        env_cfg=_make_saber_mimic_env_cfg(play=False, mimic_mix=mimic_mix),
        play_env_cfg=_make_saber_mimic_env_cfg(play=True, mimic_mix=mimic_mix),
        rl_cfg=rl_cfg_fn(save_interval=150),
        runner_cls=None,
    )
