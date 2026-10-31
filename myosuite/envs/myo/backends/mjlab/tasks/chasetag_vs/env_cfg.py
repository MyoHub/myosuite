# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""mjlab twin of ``myoChallengeChaseTagFBVsTrain-v0`` (CPU ``ChaseTagVsTrainEnv``).

The CPU registration (its FBVs task config) is the source of truth for the
model, control step, horizon, rules and reward weights. Each player is one
entity built from the single-agent full-body spec of the FBVs builder; the
world floor and the arena fence are added to the scene as
``build_combined_spec`` / ``add_arena`` add them on the CPU, and each player
starts in its FBVs standing pose (``agent_0`` / ``agent_1`` placement).
"""

from __future__ import annotations

import dataclasses
import functools

import mujoco
import numpy as np
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import terminations as mdp_terminations
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg

from myosuite.envs.myo.backends.mjlab.tasks import cpu_reference as ref
from myosuite.envs.myo.backends.mjlab.tasks import mdp
from myosuite.envs.myo.backends.mjlab.tasks.chasetag_vs import mdp as ct
from myosuite.envs.myo.tasks.challenge.chase_tag_vs.arena import add_arena
from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_config import (
    ChaseTagVsConfig,
)
from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_model import (
    _FULLBODY_FLOOR_RGBA,
    _build_fullbody_agent_spec,
)
from myosuite.envs.myo.tasks.challenge.chase_tag_vs.train_env import (
    CHASETAG_VS_OBS_KEYS,
    LEARNER,
    OPPONENT,
    train_task_config,
)

_CPU_AGENTS = (LEARNER, OPPONENT)  # CPU agent of each player (mdp.PLAYERS)
_CPU_PREFIX = {LEARNER: "a0_", OPPONENT: "a1_"}
# Two full bodies; CPU rollouts peak at ~20 contacts and ~200 constraint rows.
_NCONMAX, _NJMAX = 512, 2048


@functools.cache
def _cpu_reference(
    sim_dt: float, separation_m: float
) -> tuple[mujoco.MjModel, np.ndarray]:
    """The CPU two-player model and its FBVs standing ``qpos`` (cached per layout)."""
    from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_task_config import (  # noqa: PLC0415
        ChaseTagVsFullbodySteeredTaskConfig,
    )

    config = ChaseTagVsFullbodySteeredTaskConfig(
        sim_dt=sim_dt, agent_separation_m=separation_m
    )
    model, _, meta = config.build_model()
    return model, np.asarray(meta.standing_qpos)


def _player_spec(sim_dt: float) -> mujoco.MjSpec:
    """One FBVs full-body player: no keyframes, no own floor (hidden on the CPU)."""
    spec, _ = _build_fullbody_agent_spec(ChaseTagVsConfig(sim_dt=sim_dt))
    for key in list(spec.keys):
        spec.delete(key)
    spec.delete(spec.geom("floor"))
    return spec


def _add_floor_and_arena(spec: mujoco.MjSpec) -> None:
    """World floor and fence of the CPU scene (``build_combined_spec``, ``add_arena``)."""
    floor = spec.worldbody.add_geom(name="floor")
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [8.0, 8.0, 0.1]
    floor.rgba = _FULLBODY_FLOOR_RGBA
    floor.contype = floor.conaffinity = 1
    add_arena(spec)


def _player_layout(
    model: mujoco.MjModel, agent: str
) -> tuple[list[int], list[str], list[str]]:
    """``qpos`` addresses, joint names and actuator tendons of *agent* (CPU model)."""
    prefix = _CPU_PREFIX[agent]
    joints = [j for j in range(model.njnt) if model.joint(j).name.startswith(prefix)]
    adr = []
    for j in joints:
        width = 7 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
        adr += range(int(model.jnt_qposadr[j]), int(model.jnt_qposadr[j]) + width)
    names = [model.joint(j).name[len(prefix) :] for j in joints]
    tendons = [
        model.tendon(int(model.actuator_trnid[a, 0])).name[len(prefix) :]
        for a in range(model.nu)
        if model.actuator(a).name.startswith(prefix)
    ]
    return adr, names, tendons


def _player_entity(
    sim_dt: float, qpos: np.ndarray, names: list[str], tendons: list[str]
) -> EntityCfg:
    """Player entity starting at its CPU standing pose *qpos* (root first)."""
    init_state = EntityCfg.InitialStateCfg(
        pos=tuple(float(v) for v in qpos[:3]),
        rot=tuple(float(v) for v in qpos[3:7]),
        joint_pos={
            f"^{n}$": float(v) for n, v in zip(names[1:], qpos[7:], strict=True)
        },
        joint_vel={".*": 0.0},
    )
    actuator = XmlActuatorCfg(
        target_names_expr=tuple(f"^{t}$" for t in tendons),
        transmission_type=TransmissionType.TENDON,
    )
    return EntityCfg(
        spec_fn=functools.partial(_player_spec, sim_dt),
        articulation=EntityArticulationInfoCfg(actuators=(actuator,)),
        init_state=init_state,
    )


def make_chasetag_vs_env_cfg(env_id: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    """``ManagerBasedRlEnvCfg`` of the train env, from its CPU registration.

    Args:
        env_id: ``myoChallengeChaseTagFBVsTrain-v0``.
        play: Build the evaluation config (identical; the eval script fills the
            self-play pool with the evaluated policy, a mirror match).

    Returns:
        The env config. The opponent plays frozen snapshots of the learner
        (``ChaseTagVsActionCfg.opponent``; a ``.pt`` ``opponent_policy`` in the
        CPU registration makes it that fixed checkpoint).
    """
    del play
    task = ref.cpu_task_spec(env_id)
    if task.muscle_conditions:
        raise ValueError(f"{env_id} has no muscle-condition variants.")
    config = train_task_config(
        task.kwargs.get("task_config"), task.kwargs.get("frame_skip")
    )
    model, standing = _cpu_reference(config.sim_dt, config.agent_separation_m)
    players = [_player_layout(model, agent) for agent in _CPU_AGENTS]

    def group(player: int) -> ObservationGroupCfg:
        funcs = dict(
            qpos=ct.obs_qpos,
            qvel=ct.obs_qvel,
            act=ct.obs_act,
            pelvis=ct.obs_pelvis,
            opponent_relative=ct.obs_opponent_relative,
            role=ct.obs_role,
        )
        return ObservationGroupCfg(
            terms={
                key: ObservationTermCfg(func=funcs[key], params={"player": player})
                for key in CHASETAG_VS_OBS_KEYS
            },
        )

    opponent = task.kwargs.get("opponent_policy")
    checkpoint = str(opponent) if str(opponent).endswith(".pt") else None
    actions = {
        "muscles": ct.ChaseTagVsActionCfg(
            opponent="policy" if checkpoint else "self_play",
            opponent_checkpoint=checkpoint,
        )
    }
    fall = config.fall_pelvis_z_threshold
    radius = config.tag_radius_m
    reward_fields = [f.name for f in dataclasses.fields(ChaseTagVsConfig)]
    chase_tag = {
        k: getattr(config, k)
        for k in reward_fields
        if k.startswith("r_") or k in ("tag_radius_m", "fall_pelvis_z_threshold")
    }
    steering = {
        "clip_repo": config.gait_clip_repo,
        "clip_file": config.gait_clip_filename,
        "target_speed": config.gait_target_speed,
        "w_jpos": config.w_gait_jpos,
        "w_jvel": config.w_gait_jvel,
        "w_heading": config.w_steering,
    }
    # FBVs reward of agent_0: the dense chase-tag terms plus gait style and steering.
    rewards = {
        "chase_tag": RewardTermCfg(
            func=ct.ChaseTagReward, weight=1.0, params=chase_tag
        ),
        "gait_steering": RewardTermCfg(
            func=ct.GaitSteeringReward, weight=1.0, params=steering
        ),
    }
    # FBVs: the episode ends when either player falls, or at the tag.
    terminations = {
        "time_out": TerminationTermCfg(func=mdp_terminations.time_out, time_out=True),
        "learner_fell": TerminationTermCfg(
            func=ct.fallen, params={"player": 0, "threshold": fall}
        ),
        "opponent_fell": TerminationTermCfg(
            func=ct.fallen, params={"player": 1, "threshold": fall}
        ),
        "tagged": TerminationTermCfg(func=ct.tagged, params={"radius": radius}),
    }
    events = {
        ct.MATCH_TERM: EventTermCfg(func=ct.Match, mode="reset"),
        **{
            f"reset_{name}": EventTermCfg(
                func=mdp.reset_to_cpu_state,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(name),
                    "qpos": tuple(float(q) for q in standing[adr]),
                    "qvel": (0.0,) * (len(adr) - 1),
                },
            )
            for name, (adr, _, _) in zip(ct.PLAYERS, players, strict=True)
        },
    }
    won = {"radius": radius, "max_time": config.maxTime}
    metrics = {
        "success": MetricsTermCfg(func=ct.learner_won, params=won, reduce="last"),
        "score": MetricsTermCfg(func=ct.score, params=won, reduce="last"),
        "tagged": MetricsTermCfg(
            func=ct.tagged, params={"radius": radius}, reduce="last"
        ),
        "learner_fell": MetricsTermCfg(
            func=ct.fallen, params={"player": 0, "threshold": fall}, reduce="last"
        ),
        "opponent_fell": MetricsTermCfg(
            func=ct.fallen, params={"player": 1, "threshold": fall}, reduce="last"
        ),
    }
    entities = {
        name: _player_entity(config.sim_dt, standing[adr], names, tendons)
        for name, (adr, names, tendons) in zip(ct.PLAYERS, players, strict=True)
    }
    step_dt = config.sim_dt * config.n_substeps
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=1, entities=entities, spec_fn=_add_floor_and_arena, env_spacing=0.0
        ),
        observations={"policy": group(0), "opponent": group(1)},
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        metrics=metrics,
        events=events,
        # CPU <option> (Euler integrator, solver iterations, ...) of the model.
        sim=SimulationCfg(
            mujoco=ref.mujoco_cfg_from_model(model), njmax=_NJMAX, nconmax=_NCONMAX
        ),
        decimation=config.n_substeps,
        episode_length_s=ref.episode_length_s(task.max_episode_steps, step_dt),
        scale_rewards_by_dt=False,
    )
