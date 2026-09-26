# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Directional leg locomotion (mjlab twin of the CPU ``myoLegDirectional*-v0`` envs).

The CPU envs are ``TaskConfig`` driven (``ModularTaskEnv``); model, timing, heading and
reward parameters are read from that config. Model, action pipeline and reset (keyframe 0)
are the CPU ones, so policies transfer between the backends.
"""

from __future__ import annotations

import gymnasium as gym
from mjlab.envs import ManagerBasedRlEnvCfg
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
from myosuite.envs.myo.backends.mjlab.tasks.leg import directional_mdp as dmdp
from myosuite.envs.myo.backends.mjlab.tasks.leg.stand_env_cfg import _NCONMAX, _NJMAX

ENTITY = "robot"
COMMAND = "heading"


def make_leg_directional_env_cfg(
    env_id: str, play: bool = False
) -> ManagerBasedRlEnvCfg:
    """mjlab twin of the CPU ``TaskConfig`` registration of *env_id*.

    Args:
        env_id: CPU env id (also the mjlab task id).
        play: Play/eval variant (identical: the CPU env has no noise or DR).

    Returns:
        The env config.
    """
    del play
    import myosuite  # noqa: F401, PLC0415  (registers the CPU envs)

    config = gym.spec(env_id).kwargs["task_config"]
    reward_extra = config.reward.extra
    task = ref.CpuTaskSpec(
        env_id,
        {"model_path": config.model, "frame_skip": config.backend.n_substeps},
        config.max_episode_steps,
    )
    info = ref.compiled_info(task)
    robot = SceneEntityCfg(ENTITY)
    speed = float(reward_extra["target_speed"])
    reward_params = {"command_name": COMMAND, "target_speed": speed, "asset_cfg": robot}

    commands = {
        COMMAND: mdp.HeadingCommandCfg(
            heading_dir=tuple(config.obs.extra["heading_dir"]),
            randomize=bool(reward_extra.get("randomize_heading", False)),
        )
    }
    obs_funcs = {
        "joint_pos": (mdp.qpos, {"asset_cfg": robot}),
        "joint_vel": (dmdp.joint_vel, {"asset_cfg": robot}),
        "muscle_act": (mdp.act, {"asset_cfg": robot}),
        "root_planar_vel": (dmdp.root_planar_vel, {"asset_cfg": robot}),
        "heading_cmd": (dmdp.heading_cmd, {"command_name": COMMAND}),
    }
    terms = {
        key: ObservationTermCfg(func=obs_funcs[key][0], params=obs_funcs[key][1])
        for key in config.obs.keys
    }
    observations = {
        "actor": ObservationGroupCfg(terms),
        "critic": ObservationGroupCfg(dict(terms)),
    }
    reward_funcs = {
        "heading": (dmdp.heading_term, reward_params),
        "act_reg": (dmdp.act_reg_term, {"asset_cfg": robot}),
    }
    rewards = {
        term: RewardTermCfg(
            func=reward_funcs[term][0],
            weight=float(config.reward.weight_for(term)),
            params=reward_funcs[term][1],
        )
        for term in config.reward.terms
    }
    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fallen": TerminationTermCfg(func=dmdp.fallen, params=reward_params),
    }
    metrics = {
        "success": MetricsTermCfg(
            func=dmdp.heading_solved, params=reward_params, reduce="last"
        )
    }
    # ModularTaskEnv.reset: keyframe 0 (qpos and qvel).
    events = {
        "reset_scene_to_default": EventTermCfg(
            func=mdp.reset_to_cpu_state,
            mode="reset",
            params={
                "asset_cfg": robot,
                "qpos": info.key_qpos[0],
                "qvel": info.key_qvel[0],
            },
        ),
    }

    step_dt = info.opt_timestep * task.frame_skip
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities={ENTITY: ref.robot_entity_cfg(task, init_qpos=info.key_qpos[0])},
            num_envs=1,
        ),
        observations=observations,
        actions={
            "muscles": ref.action_cfg(
                task, ENTITY, action_range=(0.0, 1.0), muscle_sigmoid=False
            )
        },
        commands=commands,
        rewards=rewards,
        terminations=terminations,
        metrics=metrics,
        events=events,
        sim=SimulationCfg(mujoco=info.mujoco_cfg, njmax=_NJMAX, nconmax=_NCONMAX),
        decimation=task.frame_skip,
        episode_length_s=ref.episode_length_s(task.max_episode_steps, step_dt),
        scale_rewards_by_dt=False,
    )
