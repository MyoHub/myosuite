# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MDP terms of the leg walking task (mjlab twin of CPU ``LegWalkEnvV0``).

Observation terms mirror ``LegWalkEnvV0._get_obs_dict``, the reward terms call the shared
:func:`walk_env_reward`, and the termination is the CPU ``_get_done``. Rewards and the
termination read derived quantities after the task's ``sync_forward`` term, like the CPU
env after ``mj_forward``.

Body ids, body masses and target vectors are resolved on the device once per term, so
the step makes no host-device copies. The ``fallen`` termination
(:class:`WalkDone`) evaluates the CPU reward dict once per step; the reward terms and
the success metric of that step read it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEntityAccessor
from myosuite.terms.base_reward import locomotion_solved, walk_env_reward

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.manager_base import ManagerTermBaseCfg

# Termination key of :class:`WalkDone`, which the reward and metric terms read.
FALLEN_TERM = "fallen"


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


class WalkModel:
    """Body masses and ids of the walk entity, on the env device.

    Args:
        env: The environment.
        entity: Scene entity key of the legs.
    """

    def __init__(self, env: ManagerBasedRlEnv, entity: str) -> None:
        model = env.sim.mj_model
        # The CPU centre of mass weighs every body of the model.
        self.mass = torch.as_tensor(
            model.body_mass, dtype=torch.float32, device=env.device
        )
        self.total_mass = self.mass.sum()
        ids = {
            name: int(model.body(f"{entity}/{name}").id)
            for name in ("torso", "pelvis", "talus_l", "talus_r")
        }
        self.torso, self.pelvis = ids["torso"], ids["pelvis"]
        self.left, self.right = ids["talus_l"], ids["talus_r"]
        self.feet = torch.tensor([self.left, self.right], device=env.device)


def _com(env: ManagerBasedRlEnv, model: WalkModel) -> torch.Tensor:
    """Mass-weighted centre of mass ``(N, 3)`` of all bodies."""
    return (model.mass[None, :, None] * env.sim.data.xipos).sum(1) / model.total_mass


def com_velocity(env: ManagerBasedRlEnv, model: WalkModel) -> torch.Tensor:
    """CPU ``_get_com_velocity``: mass-weighted ``-cvel[3:5]``."""
    cvel = -env.sim.data.cvel  # no entity.data API for cvel / actuator_* (CLAUDE.md)
    return ((model.mass[None, :, None] * cvel).sum(1) / model.total_mass)[:, 3:5]


class _WalkTerm(ManagerTermBase):
    """Term holding the :class:`WalkModel` of its ``asset_cfg`` entity."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        self.model = WalkModel(env, cfg.params["asset_cfg"].name)


def qpos_without_xy(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return MjlabEntityAccessor(env, asset_cfg.name).joint_pos()[:, 2:]


class ComVelocity(_WalkTerm):
    """Observation ``com_vel``."""

    def __call__(
        self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        return com_velocity(env, self.model)


class TorsoAngle(_WalkTerm):
    """Observation ``torso_angle``: torso orientation quaternion."""

    def __call__(
        self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        return env.sim.data.xquat[:, self.model.torso]


class FeetHeights(_WalkTerm):
    """Observation ``feet_heights``: z of both tali."""

    def __call__(
        self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        return env.sim.data.xpos[:, self.model.feet, 2]


class Height(_WalkTerm):
    """Observation ``height``: centre-of-mass height."""

    def __call__(
        self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        return _com(env, self.model)[:, 2:3]


class FeetRelPositions(_WalkTerm):
    """Observation ``feet_rel_positions``: tali relative to the pelvis."""

    def __call__(
        self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        m, xpos = self.model, env.sim.data.xpos
        return torch.cat(
            [xpos[:, m.left] - xpos[:, m.pelvis], xpos[:, m.right] - xpos[:, m.pelvis]],
            1,
        )


def phase_var(env: ManagerBasedRlEnv, hip_period: int) -> torch.Tensor:
    steps = env.episode_length_buf.float()
    return ((steps / hip_period) % 1.0).unsqueeze(-1)


def muscle_length(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.sim.data.actuator_length


def muscle_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.sim.data.actuator_velocity.clip(-100.0, 100.0)


def muscle_force(env: ManagerBasedRlEnv) -> torch.Tensor:
    return (env.sim.data.actuator_force / 1000.0).clip(-100.0, 100.0)


def walk_components(
    env: ManagerBasedRlEnv,
    walk: WalkParams,
    asset_cfg: SceneEntityCfg,
    model: WalkModel,
    target_rot: torch.Tensor,
    target_vel: torch.Tensor,
) -> dict[str, Any]:
    """All CPU ``LegWalkEnvV0`` reward entries (shared ``walk_env_reward``) for the state.

    Args:
        env: The environment.
        walk: Task constants.
        asset_cfg: Robot entity.
        model: Body masses and ids of the entity.
        target_rot: ``walk.target_rot`` on the env device.
        target_vel: ``walk.target_vel`` on the env device.

    Returns:
        The reward dict of :func:`walk_env_reward` plus ``vel_error``.
    """
    accessor = MjlabEntityAccessor(env, asset_cfg.name)
    com_vel = com_velocity(env, model)
    task_state = {
        "qpos": accessor.joint_pos(),
        "height": _com(env, model)[:, 2],
        "com_vel": com_vel,
        "phase_var": phase_var(env, walk.hip_period),
    }
    comps = walk_env_reward(
        accessor,
        task_state,
        hip_flex_indices=walk.hip_flex_indices,
        hip_angle_indices=walk.hip_angle_indices,
        target_rot=target_rot,
        target_vel=walk.target_vel,
        min_height=walk.min_height,
        max_rot=walk.max_rot,
    )
    if walk.knee_margin is not None:
        feet = env.sim.data.xpos[:, model.feet, 2].mean(1)
        knee = (task_state["height"] - feet) < walk.knee_margin
        comps["done"] = torch.logical_or(comps["done"], knee)
    comps["vel_error"] = torch.linalg.norm(target_vel - com_vel, dim=1)
    return comps


class WalkDone(_WalkTerm):
    """Termination ``fallen`` (CPU ``_get_done``): centre of mass too low or root rotated too far.

    It evaluates the whole CPU reward dict of the step (after ``sync_forward``) and keeps
    it for the reward terms and the success metric, which mjlab evaluates next in the same
    step, on the same state.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self._walk: WalkParams = cfg.params["walk"]
        self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._target_rot = torch.tensor(self._walk.target_rot, device=env.device)
        self._target_vel = torch.tensor(self._walk.target_vel, device=env.device)
        self._components: dict[str, Any] = {}
        self._step = -1  # common_step_counter of _components

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Forget the components: they describe the state before the reset."""
        del env_ids
        self._step = -1

    def components(self, env: ManagerBasedRlEnv) -> dict[str, Any]:
        """The reward dict of the current state, evaluated once per env step."""
        if self._step != env.common_step_counter:
            self._components = walk_components(
                env,
                self._walk,
                self._asset_cfg,
                self.model,
                self._target_rot,
                self._target_vel,
            )
            self._step = env.common_step_counter
        return self._components

    def __call__(
        self, env: ManagerBasedRlEnv, walk: WalkParams, asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        self._step = -1  # first walk term of the step: evaluate the new state
        return self.components(env)["done"].bool()


def _step_components(env: ManagerBasedRlEnv) -> dict[str, Any]:
    """The step's reward dict, from the ``fallen`` termination term."""
    done: WalkDone = env.termination_manager.get_term_cfg(FALLEN_TERM).func
    return done.components(env)


def walk_term(
    env: ManagerBasedRlEnv, key: str, walk: WalkParams, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """One entry (``key``) of the CPU reward dict, as a float tensor.

    ``walk`` / ``asset_cfg`` are those of the ``fallen`` term, which evaluates the dict.
    """
    return _step_components(env)[key].float()


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
    comps = _step_components(env)
    return locomotion_solved(torch, comps["vel_error"], comps["done"].bool()).float()
