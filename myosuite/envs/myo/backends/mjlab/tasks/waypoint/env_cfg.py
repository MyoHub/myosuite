# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""mjlab twin of the CPU waypoint envs (``myosuite.envs.waypoint.WaypointEnv``).

Model (recipe or path plus ``edit_fn``), timing, reset pose, route distribution and
reward weights are read from the CPU registration, so policies transfer.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

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
from myosuite.envs.myo.backends.mjlab.tasks import mdp
from myosuite.envs.myo.backends.mjlab.tasks.waypoint import mdp as wmdp
from myosuite.envs.waypoint import (
    RootLayout,
    WaypointTaskCfg,
    heading_yaw,
    initial_state,
)

ENTITY = "robot"
_NJMAX, _NCONMAX = 512, 256


@dataclasses.dataclass(frozen=True)
class _ResetInfo:
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    start: tuple[float, float]
    start_yaw: float
    layout: RootLayout


@functools.cache
def _reset_info(
    model_key: tuple[Any, ...],
    site: str,
    qpos: tuple[float, ...] | None,
    qvel: tuple[float, ...] | None,
) -> _ResetInfo:
    model_path, model_recipe, edit_fn = model_key
    kwargs = {
        "model_path": model_path,
        "model_recipe": model_recipe,
        "edit_fn": edit_fn,
    }
    model = ref.build_cpu_spec(ref.CpuTaskSpec("", kwargs, 0)).compile()
    pos, vel = initial_state(model, qpos, qvel)
    layout = RootLayout.from_model(model)
    data = mujoco.MjData(model)
    data.qpos[:] = pos
    mujoco.mj_kinematics(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    return _ResetInfo(
        tuple(map(float, pos)),
        tuple(map(float, vel)),
        tuple(map(float, data.site_xpos[site_id, :2])),
        float(heading_yaw(np, pos, layout.has_free_root)),
        layout,
    )


def make_waypoint_env_cfg(env_id: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    """mjlab twin of the CPU waypoint registration of *env_id*.

    Args:
        env_id: CPU env id (also the mjlab task id).
        play: Play/eval variant (identical: the CPU env has no extra noise or DR).

    Returns:
        The env config.
    """
    del play
    cpu = ref.cpu_task_spec(env_id)
    if cpu.kwargs.get("model") is not None:
        raise ValueError(
            "mjlab requires model_path or model_recipe, not a compiled model"
        )
    if cpu.kwargs.get("model_path") is not None:
        cpu = dataclasses.replace(cpu, kwargs={**cpu.kwargs, "model_recipe": None})
    task_cfg: WaypointTaskCfg = cpu.kwargs.get("task") or WaypointTaskCfg()
    # Raw controls clipped to the control range, as WaypointEnv.step.
    task = dataclasses.replace(
        cpu,
        kwargs={
            **cpu.kwargs,
            "frame_skip": int(cpu.kwargs.get("frame_skip", 5)),
            "normalize_act": False,
        },
    )
    info = ref.compiled_info(task)
    init_qpos, init_qvel = (
        cpu.kwargs.get("initial_qpos"),
        cpu.kwargs.get("initial_qvel"),
    )
    reset = _reset_info(
        ref._model_key(task),
        task_cfg.site_name,
        None if init_qpos is None else tuple(map(float, init_qpos)),
        None if init_qvel is None else tuple(map(float, init_qvel)),
    )
    robot = SceneEntityCfg(ENTITY)

    obs_funcs = {
        "qpos": wmdp.qpos,
        "qvel": wmdp.qvel,
        "act": wmdp.act,
        "waypoint_targets": wmdp.waypoint_targets,
    }
    terms = {
        key: ObservationTermCfg(func=obs_funcs[key], params={"asset_cfg": robot})
        for key in task_cfg.obs_keys
    }
    observations = {
        "actor": ObservationGroupCfg(terms),
        "critic": ObservationGroupCfg(dict(terms)),
    }
    terminations = {
        # The route advances on the site position: refresh it first (CPU runs mj_forward).
        mdp.SYNC_TERM: TerminationTermCfg(func=mdp.sync_forward),
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "waypoints_done": TerminationTermCfg(func=wmdp.advance_route),
        "failed": TerminationTermCfg(func=wmdp.failed),
    }
    weights = task_cfg.reward
    rewards = {
        key: RewardTermCfg(
            func=wmdp.reward_component, weight=weight, params={"key": key}
        )
        for key, weight in (
            ("progress", weights.progress),
            ("arrived", weights.arrival),
            ("failed", -weights.failure),
        )
    }
    noise = task_cfg.reset_noise * reset.layout.noise_mask
    events = {
        "reset_pose": EventTermCfg(
            func=wmdp.reset_pose,
            mode="reset",
            params={
                "asset_cfg": robot,
                "qpos": reset.qpos,
                "qvel": reset.qvel,
                "noise": tuple(map(float, noise)),
                "low": tuple(map(float, reset.layout.low)),
                "high": tuple(map(float, reset.layout.high)),
            },
        ),
    }
    step_dt = info.opt_timestep * task.frame_skip
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities={ENTITY: ref.robot_entity_cfg(task, init_qpos=reset.qpos)},
            num_envs=1,
        ),
        observations=observations,
        actions={"muscles": ref.action_cfg(task, ENTITY)},
        commands={
            wmdp.COMMAND: wmdp.WaypointCommandCfg(
                task=task_cfg,
                entity_name=ENTITY,
                start=reset.start,
                start_yaw=reset.start_yaw,
            )
        },
        rewards=rewards,
        terminations=terminations,
        metrics={"success": MetricsTermCfg(func=wmdp.solved, reduce="last")},
        events=events,
        sim=SimulationCfg(mujoco=info.mujoco_cfg, njmax=_NJMAX, nconmax=_NCONMAX),
        decimation=task.frame_skip,
        episode_length_s=ref.episode_length_s(task.max_episode_steps, step_dt),
        scale_rewards_by_dt=False,
    )
