# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Leg stand-and-reach task (mjlab twin of CPU ``LegReachEnvV0``).

Every task parameter is read from the CPU registration of the same ``env_id``
(see ``cpu_reference``).
"""

from __future__ import annotations

import mujoco
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
from myosuite.envs.myo.tasks.basic.leg.reach import LegReachEnvV0

ENTITY = "robot"
COMMAND = "reach"
# LegReachEnvV0 keeps penalties off until data.time > 2 * ctrl_dt.
_PENALTY_DELAY_STEPS = 2
# MuJoCo Warp pre-allocates constraint/contact buffers; the default is too small
# for the contact-rich legs (nefc overflow gives NaN observations). Maxima only.
_NJMAX, _NCONMAX = 512, 256


def hide_terrain(spec: mujoco.MjSpec) -> None:
    """Replay ``LegReachEnvV0.__init__``: move the hfield terrain 10 m below the legs.

    Args:
        spec: Entity spec of the leg model (a flat-ground model has no terrain).
    """
    terrain = spec.geom("terrain")
    if terrain is not None:
        terrain.pos = np.array([0.0, 0.0, -10.0])
        terrain.rgba[3] = 0.0


def make_leg_stand_env_cfg(env_id: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    """mjlab twin of the CPU ``LegReachEnvV0`` registration of *env_id*.

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
        COMMAND: mdp.RelativeReachTargetCommandCfg(
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
    obs_keys = list(kw.get("obs_keys", LegReachEnvV0.DEFAULT_OBS_KEYS))
    if info.na > 0 and "act" not in obs_keys:
        obs_keys.append("act")
    # LegReachEnvV0 clips its observation to the +-10 observation space.
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
    weights = kw.get("weighted_reward_keys", LegReachEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS)
    rewards = {
        key: RewardTermCfg(
            func=mdp.leg_reach_term,
            weight=float(weight),
            params={"key": key, **reach_params},
        )
        for key, weight in weights.items()
    }

    terminations = {
        mdp.SYNC_TERM: TerminationTermCfg(func=mdp.sync_kinematics),
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "reach_failed": TerminationTermCfg(
            func=mdp.leg_reach_failed, params=reach_params
        ),
    }

    # Standard success metric (logged as Episode_Metrics/success): the CPU
    # "solved" flag on the final step of the episode.
    metrics = {
        "success": MetricsTermCfg(
            func=mdp.leg_reach_term,
            params={"key": "solved", **reach_params},
            reduce="last",
        )
    }

    # CPU reset: keyframe 0 (qpos and qvel), with the first qpos entry of every
    # joint perturbed by U(joint_random_range) and clipped to the joint range.
    noise = tuple(kw.get("joint_random_range", (0.0, 0.0)))
    events = {
        "reset_scene_to_default": EventTermCfg(
            func=mdp.reset_to_cpu_state,
            mode="reset",
            params={
                "asset_cfg": robot,
                "qpos": info.key_qpos[0],
                "qvel": info.key_qvel[0],
                "joint_noise": noise,
            },
        ),
    }

    step_dt = info.opt_timestep * task.frame_skip
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities={
                ENTITY: ref.robot_entity_cfg(
                    task, init_qpos=info.key_qpos[0], spec_edits=(hide_terrain,)
                )
            },
            num_envs=1,
        ),
        observations=observations,
        actions={"muscles": ref.action_cfg(task, ENTITY)},
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
