# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Joint-pose task configuration (mjlab twin of CPU ``PoseEnvV0`` / ``TorsoEnvV0``).

Model-specific configurations live in ``config/``. Every task parameter is read
from the CPU registration of the same ``env_id`` (see ``cpu_reference``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
from myosuite.envs.myo.backends.mjlab.tasks.pose import mdp
from myosuite.envs.myo.tasks.basic.arm.pose import PoseEnvV0
from myosuite.envs.myo.tasks.basic.torso.pose import TorsoEnvV0

ENTITY = "robot"
COMMAND = "pose"

_REWARD_FUNCS = {
    "pose": mdp.pose_dist,
    "sparse": mdp.pose_dist,
    "bonus": mdp.pose_bonus,
    "penalty": mdp.pose_penalty,
    "solved": mdp.pose_solved,
    "done": mdp.pose_done,
}


@dataclass(frozen=True)
class _PoseVariant:
    """How a CPU pose env class specializes the shared pose task.

    Attributes:
        default_obs_keys: Class ``DEFAULT_OBS_KEYS``.
        default_weights: Class ``DEFAULT_RWD_KEYS_AND_WEIGHTS``.
        default_pose_thd: Constructor default of ``pose_thd``.
        far_thd: Distance above which the penalty applies and the episode ends.
        obs_clip: Observation clip (``PoseEnvV0`` clips to its ``+-10`` space).
        target_always_mean: Target is the range mean (``TorsoEnvV0``).
    """

    default_obs_keys: tuple[str, ...]
    default_weights: dict[str, float]
    default_pose_thd: float
    far_thd: float
    obs_clip: tuple[float, float] | None
    target_always_mean: bool


_POSE_ENV = _PoseVariant(
    default_obs_keys=tuple(PoseEnvV0.DEFAULT_OBS_KEYS),
    default_weights=dict(PoseEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS),
    default_pose_thd=0.35,
    far_thd=2 * math.pi,
    obs_clip=(-10.0, 10.0),
    target_always_mean=False,
)
_TORSO_ENV = _PoseVariant(
    default_obs_keys=tuple(TorsoEnvV0.DEFAULT_OBS_KEYS),
    default_weights=dict(TorsoEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS),
    default_pose_thd=0.25,
    far_thd=math.pi,
    obs_clip=None,
    target_always_mean=True,
)


def _target_bounds(
    task: ref.CpuTaskSpec, variant: _PoseVariant
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """CPU target: ``target_jnt_range`` (positional, dict order) or fixed value."""
    kw = task.kwargs
    fixed = variant.target_always_mean or kw.get("target_type", "generate") == "fixed"
    if kw.get("target_jnt_range") is not None:
        rng = np.array(list(kw["target_jnt_range"].values()), dtype=float)
        low, high = rng[:, 0], rng[:, 1]
        if fixed:
            low = high = rng.mean(axis=1)
    else:
        low = high = np.asarray(kw["target_jnt_value"], dtype=float)
    return tuple(low.tolist()), tuple(high.tolist())


def _make_env_cfg(env_id: str, variant: _PoseVariant) -> ManagerBasedRlEnvCfg:
    task = ref.cpu_task_spec(env_id)
    kw = task.kwargs
    info = ref.compiled_info(task)
    robot = SceneEntityCfg(ENTITY)

    low, high = _target_bounds(task, variant)
    commands = {
        COMMAND: mdp.JointPoseCommandCfg(entity_name=ENTITY, low=low, high=high)
    }

    obs_funcs = {
        "qpos": (mdp.qpos, {"asset_cfg": robot}),
        "qvel": (mdp.qvel, {"asset_cfg": robot}),
        "act": (mdp.act, {"asset_cfg": robot}),
        "pose_err": (mdp.pose_err, {"command_name": COMMAND, "asset_cfg": robot}),
    }
    obs_keys = list(kw.get("obs_keys", variant.default_obs_keys))
    if info.na > 0 and "act" not in obs_keys:
        obs_keys.append("act")
    terms = {
        key: ObservationTermCfg(
            func=obs_funcs[key][0], params=obs_funcs[key][1], clip=variant.obs_clip
        )
        for key in obs_keys
        if key in obs_funcs
    }
    observations = {
        "actor": ObservationGroupCfg(terms),
        "critic": ObservationGroupCfg(dict(terms)),
    }

    pose_params = {
        "command_name": COMMAND,
        "pose_thd": float(kw.get("pose_thd", variant.default_pose_thd)),
        "far_thd": variant.far_thd,
        "asset_cfg": robot,
    }
    rewards = {}
    for key, weight in kw.get("weighted_reward_keys", variant.default_weights).items():
        if key == "act_reg":
            rewards[key] = RewardTermCfg(
                func=mdp.act_norm, weight=float(weight), params={"asset_cfg": robot}
            )
        else:
            rewards[key] = RewardTermCfg(
                func=_REWARD_FUNCS[key], weight=float(weight), params=pose_params
            )

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "pose_diverged": TerminationTermCfg(
            func=mdp.pose_diverged,
            params={
                "command_name": COMMAND,
                "far_thd": variant.far_thd,
                "asset_cfg": robot,
            },
        ),
    }

    # Standard success metric (logged as Episode_Metrics/success): the CPU
    # "solved" flag on the final step of the episode.
    metrics = {
        "success": MetricsTermCfg(
            func=mdp.pose_solved, params=pose_params, reduce="last"
        )
    }

    events = {
        "reset_scene_to_default": EventTermCfg(
            func=mdp.reset_scene_to_default, mode="reset"
        ),
    }
    if kw.get("reset_type", "init") == "random":
        events["reset_joints_random"] = EventTermCfg(
            func=mdp.reset_joints_uniform_in_range,
            mode="reset",
            params={"asset_cfg": robot},
        )
    if kw.get("weight_bodyname") is not None:
        events["carry_weight"] = EventTermCfg(
            func=mdp.randomize_carry_weight,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    ENTITY, body_names=(kw["weight_bodyname"],)
                ),
                "mass_range": tuple(kw["weight_range"]),
            },
        )

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


def make_pose_env_cfg(env_id: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    """mjlab twin of the CPU ``PoseEnvV0`` registration of *env_id*.

    Args:
        env_id: CPU env id (also the mjlab task id).
        play: Play/eval variant (identical: the CPU env has no noise or DR to
            switch off).

    Returns:
        The env config.
    """
    del play
    return _make_env_cfg(env_id, _POSE_ENV)


def make_torso_pose_env_cfg(env_id: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    """mjlab twin of the CPU ``TorsoEnvV0`` registration of *env_id*.

    Args:
        env_id: CPU env id (also the mjlab task id).
        play: Play/eval variant (identical to the training variant).

    Returns:
        The env config.
    """
    del play
    return _make_env_cfg(env_id, _TORSO_ENV)
