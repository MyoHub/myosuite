# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Leg walking task (mjlab twin of CPU ``LegWalkEnvV0``).

Every task parameter is read from the CPU registration of the same ``env_id`` (see
``cpu_reference``); the model, action pipeline, observation, reward, termination and reset
follow the CPU env.
"""

from __future__ import annotations

import functools

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
from myosuite.envs.myo.backends.mjlab.tasks.leg import walk_mdp
from myosuite.envs.myo.backends.mjlab.tasks.leg.terrain import add_terrain
from myosuite.envs.myo.backends.mjlab.tasks.leg.stand_env_cfg import (
    _NCONMAX,
    _NJMAX,
    hide_terrain,
)
from myosuite.envs.myo.tasks.basic.leg.walk import LegWalkEnvV0

ENTITY = "robot"
# LegTerrainEnvV0 ends the episode when the centre of mass is < 0.61 m above the feet.
_KNEE_MARGIN = 0.61


def _qpos_index(info: ref.CompiledModelInfo, name: str) -> int:
    return int(info.jnt_qposadr[info.joint_names.index(name)])


def walk_params(
    task: ref.CpuTaskSpec, info: ref.CompiledModelInfo
) -> walk_mdp.WalkParams:
    """Constants of the CPU ``LegWalkEnvV0`` registration of *task*."""
    kw = task.kwargs
    target_rot = kw.get("target_rot")
    if target_rot is None:  # CPU: the root orientation of keyframe 0
        target_rot = info.key_qpos[0][3:7]
    return walk_mdp.WalkParams(
        hip_flex_indices=(
            _qpos_index(info, "hip_flexion_l"),
            _qpos_index(info, "hip_flexion_r"),
        ),
        hip_angle_indices=tuple(
            _qpos_index(info, n)
            for n in (
                "hip_adduction_l",
                "hip_adduction_r",
                "hip_rotation_l",
                "hip_rotation_r",
            )
        ),
        target_rot=tuple(float(x) for x in target_rot),
        target_vel=(
            float(kw.get("target_x_vel", 0.0)),
            float(kw.get("target_y_vel", 1.2)),
        ),
        min_height=float(kw.get("min_height", 0.8)),
        max_rot=float(kw.get("max_rot", 0.8)),
        hip_period=int(kw.get("hip_period", 100)),
        knee_margin=_KNEE_MARGIN if kw.get("terrain") else None,
    )


def make_leg_walk_env_cfg(env_id: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    """mjlab twin of the CPU ``LegWalkEnvV0`` / ``LegTerrainEnvV0`` registration of *env_id*.

    Args:
        env_id: CPU env id (also the mjlab task id).
        play: Play/eval variant (identical: the CPU env has no noise or DR).

    Returns:
        The env config.

    Raises:
        NotImplementedError: For ``reset_type`` other than ``"init"``.
    """
    del play
    task = ref.cpu_task_spec(env_id)
    kw = task.kwargs
    if kw.get("reset_type", "init") != "init":
        raise NotImplementedError("Only reset_type='init' is ported.")
    info = ref.compiled_info(task)
    robot = SceneEntityCfg(ENTITY)
    if kw.get("terrain"):  # height field baked in (see terrain.py)
        model_edit = functools.partial(
            add_terrain, terrain=kw["terrain"], variant=kw.get("variant")
        )
    else:
        model_edit = hide_terrain
    walk = walk_params(task, info)
    hip_period = walk.hip_period

    obs_funcs = {
        "qpos_without_xy": (walk_mdp.qpos_without_xy, {"asset_cfg": robot}),
        "qvel": (mdp.qvel, {"asset_cfg": robot}),
        "com_vel": (walk_mdp.com_vel_obs, {}),
        "torso_angle": (walk_mdp.torso_angle, {"asset_cfg": robot}),
        "feet_heights": (walk_mdp.feet_heights, {"asset_cfg": robot}),
        "height": (walk_mdp.height, {}),
        "feet_rel_positions": (walk_mdp.feet_rel_positions, {"asset_cfg": robot}),
        "phase_var": (walk_mdp.phase_var, {"hip_period": hip_period}),
        "muscle_length": (walk_mdp.muscle_length, {}),
        "muscle_velocity": (walk_mdp.muscle_velocity, {}),
        "muscle_force": (walk_mdp.muscle_force, {}),
        "act": (mdp.act, {"asset_cfg": robot}),
    }
    obs_keys = list(kw.get("obs_keys", LegWalkEnvV0.DEFAULT_OBS_KEYS))
    if info.na > 0 and "act" not in obs_keys:
        obs_keys.append("act")
    # LegWalkEnvV0 clips its observation to the +-10 observation space.
    terms = {
        key: ObservationTermCfg(
            func=obs_funcs[key][0], params=obs_funcs[key][1], clip=(-10.0, 10.0)
        )
        for key in obs_keys
        if key in obs_funcs
    }
    observations = {
        "actor": ObservationGroupCfg(terms),
        "critic": ObservationGroupCfg(dict(terms)),
    }

    weights = kw.get("weighted_reward_keys", LegWalkEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS)
    rewards = {
        key: RewardTermCfg(
            func=walk_mdp.walk_term,
            weight=float(weight),
            params={"key": key, "walk": walk, "asset_cfg": robot},
        )
        for key, weight in weights.items()
    }
    terminations = {
        mdp.SYNC_TERM: TerminationTermCfg(func=mdp.sync_kinematics),
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fallen": TerminationTermCfg(
            func=walk_mdp.walk_done, params={"walk": walk, "asset_cfg": robot}
        ),
    }
    # Standard success metric (Episode_Metrics/success): the CPU "solved" flag.
    metrics = {
        "success": MetricsTermCfg(
            func=walk_mdp.walk_solved,
            params={
                "target_x_vel": walk.target_vel[0],
                "target_y_vel": walk.target_vel[1],
                "walk": walk,
                "asset_cfg": robot,
            },
            reduce="last",
        )
    }
    # CPU reset_type "init": keyframe 2 (qpos and qvel).
    events = {
        "reset_scene_to_default": EventTermCfg(
            func=mdp.reset_to_cpu_state,
            mode="reset",
            params={
                "asset_cfg": robot,
                "qpos": info.key_qpos[2],
                "qvel": info.key_qvel[2],
            },
        ),
    }

    step_dt = info.opt_timestep * task.frame_skip
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities={
                ENTITY: ref.robot_entity_cfg(
                    task, init_qpos=info.key_qpos[2], spec_edits=(model_edit,)
                )
            },
            num_envs=1,
        ),
        observations=observations,
        actions={
            "muscles": ref.action_cfg(
                task, ENTITY, action_range=(0.0, 1.0), muscle_sigmoid=False
            )
        },
        rewards=rewards,
        terminations=terminations,
        metrics=metrics,
        events=events,
        sim=SimulationCfg(mujoco=info.mujoco_cfg, njmax=_NJMAX, nconmax=_NCONMAX),
        decimation=task.frame_skip,
        episode_length_s=ref.episode_length_s(task.max_episode_steps, step_dt),
        scale_rewards_by_dt=False,
    )
