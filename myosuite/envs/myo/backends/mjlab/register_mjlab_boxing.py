# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""mjlab registration for the BoxingP0 task (myoChallengeBoxingP0Mjlab-v0).

This module owns only registration — env config assembly, mjlab task registration,
and the public helpers callers use to wire up variants. All env logic (term
functions, task state, reset events, action classes) lives in boxing_mjlab_env.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
        _init_state_from_model,
        _muscle_tendon_names,
        _strip_spec_keyframes,
    )
except ModuleNotFoundError as _e:
    raise ModuleNotFoundError(
        "myosuite mjlab boxing registration requires the 'musclemimic' integration: "
        "pip install myosuite[musclemimic]"
    ) from _e

from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import (
    MyoMuscleActivationActionCfg,
)
from myosuite.envs.myo.backends.mjlab.boxing_mjlab_env import (
    _BOXING_ENTITY_NAME,
    BoxingTaskState,
    _boxing_act_reg_reward,
    _boxing_fall_termination,
    _boxing_fist_obs,
    _boxing_health_obs,
    _boxing_kinematics_obs,
    _boxing_mannequin_pre_physics,
    _boxing_pad_damage_reward,
    _boxing_pelvis_obs,
    _boxing_reset_event,
    _boxing_startup_event,
    _boxing_survival_reward,
    _boxing_target_pad_obs,
    _boxing_upright_posture_reward,
)

BOXING_P0_MJLAB_ENV_ID = "myoChallengeBoxingP0Mjlab-v0"


@dataclass
class BoxingP0MjlabCfg:
    """Default training hyper-parameters for BoxingP0 mjlab."""

    sim_dt: float = 0.002
    ctrl_dt: float = 0.01
    max_episode_steps: int = 500
    num_envs: int = 256


def _build_boxing_boxer_spec_and_model() -> tuple[Any, Any]:
    """Return ``(mj_model, spec_fn)`` for the boxer-only component of P0.

    We compile the full boxing P0 scene (boxer + static 6-pad machine) to get
    correct combined ids, then strip keyframes for mjlab.
    """
    from myosuite.envs.myo.tasks.challenge.boxing_mannequin.boxing_mannequin_config import (
        BoxingMannequinConfig,
    )
    from myosuite.envs.myo.tasks.challenge.boxing_mannequin.boxing_mannequin_model import (
        build_boxing_mannequin_model,
        _build_boxing_mannequin_spec,
    )

    config = BoxingMannequinConfig(include_boxing_targets_scene=True)
    mj_model, _, meta = build_boxing_mannequin_model(config)

    def _spec_fn() -> Any:
        spec = _build_boxing_mannequin_spec(config)
        _strip_spec_keyframes(spec)
        return spec

    return mj_model, meta, _spec_fn


def _make_boxing_p0_env_cfg(*, play: bool = False) -> Any:
    """Build the mjlab env config for the BoxingP0 task."""
    from mjlab.actuator.actuator import TransmissionType
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.envs.mdp import terminations as mdp_terminations
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.viewer import ViewerConfig
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        _XmlWrappedActuatorCfg,
    )
    from myosuite.envs.myo.tasks.challenge.boxing_mannequin.boxing_mannequin_task_config import (
        BoxingMannequinTaskConfig,
    )

    task_cfg = BoxingMannequinTaskConfig(include_boxing_targets_scene=True)
    mj_model, meta, spec_fn = _build_boxing_boxer_spec_and_model()

    # Muscle tendons from the boxer-only component of the combined model.
    # We extract tendons from the full combined model (agent_0 prefix = "a0_").
    # Since _muscle_tendon_names works on the standalone model, we build a
    # standalone boxer model for tendon extraction.
    from myosuite.integrations.musclemimic.fullbody_model import (
        build_mimic_fullbody_spec,
        default_mimic_fullbody_config,
    )

    fullbody_cfg = default_mimic_fullbody_config()
    fullbody_cfg.disable_fingers = task_cfg.disable_fingers
    fullbody_cfg.sim_dt = task_cfg.sim_dt
    boxer_spec, _ = build_mimic_fullbody_spec(fullbody_cfg)
    boxer_model = boxer_spec.compile()
    tendons = _muscle_tendon_names(boxer_model)

    articulation = EntityArticulationInfoCfg(
        actuators=(
            _XmlWrappedActuatorCfg(
                target_names_expr=tendons,
                transmission_type=TransmissionType.TENDON,
            ),
        )
    )
    init_state = _init_state_from_model(boxer_model)
    robot_entity_cfg = EntityCfg(
        spec_fn=spec_fn,
        articulation=articulation,
        init_state=init_state,
    )

    hyper = BoxingP0MjlabCfg()
    scene_cfg = SceneCfg(
        num_envs=1 if play else hyper.num_envs,
        entities={_BOXING_ENTITY_NAME: robot_entity_cfg},
    )

    ctrl_dt = float(task_cfg.ctrl_dt)

    obs_terms: dict[str, Any] = {
        "kinematics": ObservationTermCfg(func=_boxing_kinematics_obs),
        "fist": ObservationTermCfg(func=_boxing_fist_obs),
        "pelvis": ObservationTermCfg(func=_boxing_pelvis_obs),
        "target_pads": ObservationTermCfg(func=_boxing_target_pad_obs),
        "health": ObservationTermCfg(func=_boxing_health_obs),
    }
    observations = {
        "policy": ObservationGroupCfg(terms=obs_terms),
        "actor": ObservationGroupCfg(terms=obs_terms),
        "critic": ObservationGroupCfg(terms=obs_terms),
    }

    rewards: dict[str, Any] = {
        "pad_damage": RewardTermCfg(
            func=_boxing_pad_damage_reward,
            weight=0.1 / ctrl_dt,
        ),
        "upright_posture": RewardTermCfg(
            func=_boxing_upright_posture_reward,
            weight=2.0 / ctrl_dt,
        ),
        "act_reg": RewardTermCfg(
            func=_boxing_act_reg_reward,
            weight=-0.001 / ctrl_dt,
        ),
        "survival": RewardTermCfg(
            func=_boxing_survival_reward,
            weight=0.01 / ctrl_dt,
        ),
    }

    terminations: dict[str, Any] = {
        "time_out": TerminationTermCfg(func=mdp_terminations.time_out, time_out=True),
        "nan_detection": TerminationTermCfg(
            func=mdp_terminations.nan_detection, time_out=False
        ),
        "self_fall": TerminationTermCfg(
            func=_boxing_fall_termination,
            time_out=False,
        ),
    }

    events: dict[str, Any] = {
        "boxing_startup": EventTermCfg(
            func=_boxing_startup_event,
            mode="startup",
        ),
        "boxing_step": EventTermCfg(
            func=BoxingTaskState,
            mode="step",
            params={"task_cfg": task_cfg, "meta": meta, "mj_model": mj_model},
        ),
        "boxing_reset": EventTermCfg(
            func=_boxing_reset_event,
            mode="reset",
        ),
    }

    decimation = max(1, int(round(ctrl_dt / float(task_cfg.sim_dt))))

    return ManagerBasedRlEnvCfg(
        scene=scene_cfg,
        decimation=decimation,
        episode_length_s=float(task_cfg.max_episode_steps) * ctrl_dt,
        observations=observations,
        actions={
            "muscles": MyoMuscleActivationActionCfg(
                entity_name=_BOXING_ENTITY_NAME,
                actuator_names=tuple(t.removesuffix("_tendon") for t in tendons),
                tendon_names=tendons,
                action_mode="excitation",
                ctrl_dt=ctrl_dt,
                pre_physics_fn=_boxing_mannequin_pre_physics,
            )
        },
        rewards=rewards,
        terminations=terminations,
        events=events,
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=float(task_cfg.sim_dt),
                integrator="euler",
                ccd_iterations=500,
            ),
            njmax=1024,
            nconmax=1024,
        ),
        viewer=ViewerConfig(width=640, height=480),
        auto_reset=True,
    )


def register_boxing_p0_mjlab_task(
    register_mjlab_task: Any,
    rl_cfg_fn: Any,
) -> None:
    """Register the BoxingP0 mjlab env with mjlab's task registry."""
    register_mjlab_task(
        task_id=BOXING_P0_MJLAB_ENV_ID,
        env_cfg=_make_boxing_p0_env_cfg(play=False),
        play_env_cfg=_make_boxing_p0_env_cfg(play=True),
        rl_cfg=rl_cfg_fn(save_interval=150),
        runner_cls=None,
    )
