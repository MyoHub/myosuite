# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""mjlab twin of ``myoMimic-v0`` (``myosuite.envs.myo.tasks.mimic.tracking.env.MimicEnv``).

Preset, overrides, model, control step and muscle-command features are read from the
CPU registration (``EnvConfig.task_kwargs``), so both halves are one MDP: the body on a
plane with the foot contact sensors, the CPU physics options, excitations clipped to
[0, 1], the ``policy`` and ``goal`` observation groups, MimicReward, the failure rule
and the clip end as a time-out.
"""

from __future__ import annotations

import dataclasses
import functools

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from myosuite.core.muscle_conditions import apply_sarcopenia_to_spec
from myosuite.envs.myo.backends.mjlab.tasks import cpu_reference as ref
from myosuite.envs.myo.backends.mjlab.tasks import mdp as base_mdp
from myosuite.envs.myo.tasks.mimic.tracking.config import MimicTaskCfg, resolve_task
from myosuite.envs.myo.tasks.mimic.tracking.models import (
    DEFAULT_MODEL,
    GROUND_BODY,
    TOUCH_SENSOR,
    MimicModelCfg,
    compiled_body,
    compiled_scene,
    model_cfg,
)

from . import mdp
from .commands import MotionCommandCfg

ENTITY = "robot"
CMD = {"command_name": "motion"}
_NJMAX, _NCONMAX = 768, 96  # MuscleMimic's per-env MuJoCo-Warp budget for this body


def _body_spec(
    model: MimicModelCfg, self_collisions: bool, sarcopenia: dict | None
) -> mujoco.MjSpec:
    spec = model.build_spec(self_collisions)
    if sarcopenia is not None:
        apply_sarcopenia_to_spec(spec, force_scale=sarcopenia.get("force_scale", 0.5))
    return spec


def _entity_cfg(
    model: MimicModelCfg, task: MimicTaskCfg, cpu: ref.CpuTaskSpec
) -> EntityCfg:
    m = compiled_body(model.name)
    root_q = m.qpos0[:7]
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(map(float, root_q[:3])),
            rot=tuple(map(float, root_q[3:7])),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        spec_fn=functools.partial(
            _body_spec,
            model,
            task.self_collisions,
            cpu.wrapper_kwargs("SarcopeniaWrapper"),
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                XmlActuatorCfg(
                    target_names_expr=model.muscle_tendon_expr,
                    transmission_type=TransmissionType.TENDON,
                    command_field="effort",
                ),
            ),
            soft_joint_pos_limit_factor=1.0,
        ),
    )


def make_mimic_env_cfg(env_id: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    """mjlab twin of the CPU registration of *env_id*.

    Args:
        env_id: CPU env id (also the mjlab task id).
        play: One env playing every clip from its start, no time limit, auto-reset.

    Returns:
        The env config.
    """
    cpu = ref.cpu_task_spec(env_id)
    kw = cpu.kwargs
    task = resolve_task(kw.get("preset", "supermimic"), kw.get("overrides"))
    model = model_cfg(kw.get("model", DEFAULT_MODEL))
    if play:
        task = dataclasses.replace(
            task,
            episode_length_s=None,
            reset=dataclasses.replace(task.reset, mode="start", auto_reset=True),
        )
    cpu_model = compiled_scene(model.name, task.self_collisions)
    timestep = float(cpu_model.opt.timestep)

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                n: ObservationTermCfg(func=mdp.track_obs, params={**CMD, "name": n})
                for n in task.obs.policy
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "goal": ObservationGroupCfg(
            terms={"goal": ObservationTermCfg(func=mdp.goal, params=dict(CMD))},
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }
    rc = task.reward
    rewards = {  # mjlab multiplies the weights by the step dt
        f"mimic_{t.name}": RewardTermCfg(
            func=mdp.track_reward,
            weight=t.weight * rc.scale / rc.ref_dt,
            params={**CMD, "name": t.name, "kernel": t.kernel},
        )
        for t in rc.terms
    }
    terminations = {
        base_mdp.SYNC_TERM: TerminationTermCfg(func=base_mdp.sync_forward),
        "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
        "motion_end": TerminationTermCfg(
            func=mdp.motion_end, params=dict(CMD), time_out=True
        ),
        **{
            f"{name}_dev": TerminationTermCfg(
                func=mdp.track_failure, params={**CMD, "name": name, "threshold": thr}
            )
            for name, thr in task.failure.thresholds().items()
        },
    }
    if task.episode_length_s is None:
        terminations.pop("time_out")
    # excitations clipped to [0, 1], as MimicEnv.step
    action = ref.action_cfg(cpu, ENTITY, action_range=(0.0, 1.0), muscle_sigmoid=False)
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1 if play else 1024,
            entities={ENTITY: _entity_cfg(model, task, cpu)},
            sensors=(
                ContactSensorCfg(
                    name=TOUCH_SENSOR,
                    primary=ContactMatch(
                        mode="body", pattern=model.foot_bodies, entity=ENTITY
                    ),
                    secondary=ContactMatch(mode="body", pattern=GROUND_BODY),
                    fields=("force",),
                    reduce="netforce",
                ),
            ),
        ),
        observations=observations,
        actions={"muscles": action},
        commands={
            "motion": MotionCommandCfg(task=task, model=model.name, entity_name=ENTITY)
        },
        events={},
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name=ENTITY,
            body_name=model.tracked_bodies[0],
            distance=3.0,
            fovy=55.0,
            elevation=-5.0,
            azimuth=120.0,
        ),
        sim=SimulationCfg(
            nconmax=_NCONMAX,
            njmax=_NJMAX,
            mujoco=ref.mujoco_cfg_from_model(cpu_model, timestep=timestep),
        ),
        decimation=cpu.frame_skip,
        episode_length_s=1e9
        if task.episode_length_s is None
        else task.episode_length_s,
        auto_reset=task.reset.auto_reset,
    )
