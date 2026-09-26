# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MDP terms of the leg walking task (mjlab twin of CPU ``LegWalkEnvV0``).

Observation terms mirror ``LegWalkEnvV0._get_obs_dict`` (post-step quantities are read
through :func:`cpu_post_step_field`, like the CPU env), the reward terms call the shared
:func:`walk_env_reward`, and the termination is the CPU ``_get_done``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEntityAccessor
from myosuite.envs.myo.backends.mjlab.tasks.mdp import cpu_post_step_field
from myosuite.terms.base_reward import locomotion_solved, walk_env_reward

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(frozen=True)
class WalkParams:
    """Constants of ``LegWalkEnvV0`` needed by the reward and the done condition.

    Attributes:
        hip_flex_indices: ``qpos`` indices of ``hip_flexion_{l,r}``.
        hip_angle_indices: ``qpos`` indices of ``hip_adduction_{l,r}``, ``hip_rotation_{l,r}``.
        target_rot: Target root quaternion (CPU: ``key_qpos[0][3:7]``).
        target_vel: ``(target_x_vel, target_y_vel)`` of the centre of mass.
        min_height: Centre-of-mass height below which the episode ends.
        max_rot: Rotation limit of the done condition.
        hip_period: Steps of the reference hip cycle.
        knee_margin: Terrain env: the episode also ends when the centre of mass is less than
            this above the mean foot height (CPU ``_get_knee_condition``); ``None``: off.
    """

    hip_flex_indices: tuple[int, int]
    hip_angle_indices: tuple[int, int, int, int]
    target_rot: tuple[float, ...]
    target_vel: tuple[float, float]
    min_height: float
    max_rot: float
    hip_period: int
    knee_margin: float | None = None


def _body_ids(env: ManagerBasedRlEnv, entity: str, *names: str) -> list[int]:
    model = env.sim.mj_model
    return [int(model.body(f"{entity}/{n}").id) for n in names]


def _mass(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.as_tensor(
        env.sim.mj_model.body_mass, dtype=torch.float32, device=env.device
    )


def _com(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Mass-weighted centre of mass ``(N, 3)`` of all bodies."""
    mass = _mass(env)
    return (mass[None, :, None] * env.sim.data.xipos).sum(1) / mass.sum()


def com_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
    """CPU ``_get_com_velocity``: mass-weighted ``-cvel[3:5]`` (pre-kinematics values)."""
    mass = _mass(env)
    cvel = -cpu_post_step_field(env, "cvel")
    return ((mass[None, :, None] * cvel).sum(1) / mass.sum())[:, 3:5]


def qpos_without_xy(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return MjlabEntityAccessor(env, asset_cfg.name).joint_pos()[:, 2:]


def com_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    return com_velocity(env)


def torso_angle(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    (torso,) = _body_ids(env, asset_cfg.name, "torso")
    return env.sim.data.xquat[:, torso]


def feet_heights(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    left, right = _body_ids(env, asset_cfg.name, "talus_l", "talus_r")
    return env.sim.data.xpos[:, [left, right], 2]


def height(env: ManagerBasedRlEnv) -> torch.Tensor:
    return _com(env)[:, 2:3]


def feet_rel_positions(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    left, right, pelvis = _body_ids(env, asset_cfg.name, "talus_l", "talus_r", "pelvis")
    xpos = env.sim.data.xpos
    return torch.cat(
        [xpos[:, left] - xpos[:, pelvis], xpos[:, right] - xpos[:, pelvis]], 1
    )


def phase_var(env: ManagerBasedRlEnv, hip_period: int) -> torch.Tensor:
    steps = env.episode_length_buf.float()
    return ((steps / hip_period) % 1.0).unsqueeze(-1)


def muscle_length(env: ManagerBasedRlEnv) -> torch.Tensor:
    return cpu_post_step_field(env, "actuator_length")


def muscle_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
    return cpu_post_step_field(env, "actuator_velocity").clip(-100.0, 100.0)


def muscle_force(env: ManagerBasedRlEnv) -> torch.Tensor:
    return (cpu_post_step_field(env, "actuator_force") / 1000.0).clip(-100.0, 100.0)


def walk_components(
    env: ManagerBasedRlEnv, walk: WalkParams, asset_cfg: SceneEntityCfg
) -> dict[str, Any]:
    """All CPU ``LegWalkEnvV0`` reward entries (shared ``walk_env_reward``) for the state.

    Args:
        env: The environment.
        walk: Task constants.
        asset_cfg: Robot entity.

    Returns:
        The reward dict of :func:`walk_env_reward` plus ``vel_error``.
    """
    accessor = MjlabEntityAccessor(env, asset_cfg.name)
    com_vel = com_velocity(env)
    task_state = {
        "qpos": accessor.joint_pos(),
        "height": height(env)[:, 0],
        "com_vel": com_vel,
        "phase_var": phase_var(env, walk.hip_period),
    }
    comps = walk_env_reward(
        accessor,
        task_state,
        hip_flex_indices=walk.hip_flex_indices,
        hip_angle_indices=walk.hip_angle_indices,
        target_rot=torch.tensor(walk.target_rot, device=env.device),
        target_vel=walk.target_vel,
        min_height=walk.min_height,
        max_rot=walk.max_rot,
    )
    if walk.knee_margin is not None:
        left, right = _body_ids(env, asset_cfg.name, "talus_l", "talus_r")
        feet = env.sim.data.xpos[:, [left, right], 2].mean(1)
        knee = (task_state["height"] - feet) < walk.knee_margin
        comps["done"] = torch.logical_or(comps["done"], knee)
    target = torch.tensor(walk.target_vel, device=env.device)
    comps["vel_error"] = torch.linalg.norm(target - com_vel, dim=1)
    return comps


def walk_term(
    env: ManagerBasedRlEnv, key: str, walk: WalkParams, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """One entry (``key``) of the CPU reward dict, as a float tensor."""
    return walk_components(env, walk, asset_cfg)[key].float()


def walk_done(
    env: ManagerBasedRlEnv, walk: WalkParams, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """CPU ``_get_done``: centre of mass too low or root rotated too far."""
    return walk_components(env, walk, asset_cfg)["done"].bool()


def walk_solved(
    env: ManagerBasedRlEnv,
    target_x_vel: float,
    target_y_vel: float,
    walk: WalkParams,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """CPU ``solved``: upright and within the tolerance of the target velocity.

    ``target_x_vel`` / ``target_y_vel`` duplicate ``walk.target_vel`` so tools (the eval
    videos) can read the commanded velocity from the metric's parameters.
    """
    comps = walk_components(env, walk, asset_cfg)
    return locomotion_solved(torch, comps["vel_error"], comps["done"].bool()).float()
