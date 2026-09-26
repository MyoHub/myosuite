# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Cartesian reach task configuration (mjlab twin of CPU ``ReachEnvV0``).

Model-specific configurations live in ``config/``. Every task parameter is read
from the CPU registration of the same ``env_id`` (see ``cpu_reference``).
"""

from __future__ import annotations

import numpy as np
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
from myosuite.envs.myo.backends.mjlab.tasks.reach import mdp
from myosuite.envs.myo.tasks.basic.arm.reach import ReachEnvV0

ENTITY = "robot"
COMMAND = "reach"
# ReachEnvV0 keeps penalties off until data.time > 2 * ctrl_dt.
_PENALTY_DELAY_STEPS = 2


def make_reach_env_cfg(env_id: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    """mjlab twin of the CPU ``ReachEnvV0`` registration of *env_id*.

    Args:
        env_id: CPU env id (also the mjlab task id).
        play: Play/eval variant (identical: the CPU env has no noise or DR).

    Returns:
        The env config.
    """
    del play
    task = ref.cpu_task_spec(env_id)
    kw = task.kwargs
    info = ref.compiled_info(task)
    ranges = kw["target_reach_range"]
    tip_sites = tuple(ranges)
    tips = SceneEntityCfg(ENTITY, site_names=tip_sites, preserve_order=True)
    robot = SceneEntityCfg(ENTITY)

    low = np.concatenate([np.asarray(r[0], dtype=float) for r in ranges.values()])
    high = np.concatenate([np.asarray(r[1], dtype=float) for r in ranges.values()])
    commands = {
        COMMAND: mdp.ReachTargetCommandCfg(
            entity_name=ENTITY,
            low=tuple(low.tolist()),
            high=tuple(high.tolist()),
            tip_sites=tip_sites,
        )
    }

    obs_funcs = {
        "qpos": (mdp.qpos, {"asset_cfg": robot}),
        "qvel": (mdp.qvel, {"asset_cfg": robot}),
        "act": (mdp.act, {"asset_cfg": robot}),
        "tip_pos": (mdp.tip_pos, {"asset_cfg": tips}),
        "reach_err": (mdp.reach_err, {"command_name": COMMAND, "asset_cfg": tips}),
    }
    obs_keys = list(kw.get("obs_keys", ReachEnvV0.DEFAULT_OBS_KEYS))
    if info.na > 0 and "act" not in obs_keys:
        obs_keys.append("act")
    # ReachEnvV0 does not clip observations to its observation space.
    terms = {
        key: ObservationTermCfg(func=obs_funcs[key][0], params=obs_funcs[key][1])
        for key in obs_keys
        if key in obs_funcs
    }
    observations = {
        "actor": ObservationGroupCfg(terms),
        "critic": ObservationGroupCfg(dict(terms)),
    }

    reach_params = {
        "command_name": COMMAND,
        "far_th": float(kw.get("far_th", 0.35)),
        "penalty_start_step": ref.first_step_after(
            _PENALTY_DELAY_STEPS * (info.opt_timestep * task.frame_skip),
            info.opt_timestep,
            task.frame_skip,
        ),
        "asset_cfg": tips,
    }
    weights = kw.get("weighted_reward_keys", ReachEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS)
    rewards = {
        key: RewardTermCfg(
            func=mdp.reach_term,
            weight=float(weight),
            params={"key": key, **reach_params},
        )
        for key, weight in weights.items()
    }

    terminations = {
        # Tip sites are derived quantities: refresh them before scoring (CPU
        # runs mj_kinematics right after stepping).
        mdp.SYNC_TERM: TerminationTermCfg(func=mdp.sync_kinematics),
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "reach_failed": TerminationTermCfg(func=mdp.reach_failed, params=reach_params),
    }

    # Standard success metric (logged as Episode_Metrics/success): the CPU
    # "solved" flag on the final step of the episode.
    metrics = {
        "success": MetricsTermCfg(
            func=mdp.reach_term,
            params={"key": "solved", **reach_params},
            reduce="last",
        )
    }

    events = {
        "reset_scene_to_default": EventTermCfg(
            func=mdp.reset_scene_to_default, mode="reset"
        ),
    }

    step_dt = info.opt_timestep * task.frame_skip
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(entities={ENTITY: ref.robot_entity_cfg(task)}, num_envs=1),
        observations=observations,
        actions={"muscles": ref.action_cfg(task, ENTITY)},
        commands=commands,
        rewards=rewards,
        terminations=terminations,
        metrics=metrics,
        events=events,
        sim=SimulationCfg(mujoco=info.mujoco_cfg),
        decimation=task.frame_skip,
        episode_length_s=ref.episode_length_s(task.max_episode_steps, step_dt),
        scale_rewards_by_dt=False,
    )
