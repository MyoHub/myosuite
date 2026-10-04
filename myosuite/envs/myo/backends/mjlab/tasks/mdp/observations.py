# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Observation terms shared by the MyoSuite mjlab tasks (CPU obs layout)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_apply_inverse,
    quat_from_angle_axis,
    quat_mul,
)

from myosuite.core.sensorimotor import FixedLagBuffer
from myosuite.envs.myo.backends.mjlab.mjlab_env_base import (
    MjlabEntityAccessor,
    normalize_mjlab_env_ids,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import FreeJointChain

_ROBOT = SceneEntityCfg("robot")


@dataclass(kw_only=True)
class DelayedObservationCfg(ObservationTermCfg):
    """Observation term ``term_func`` received ``delay_steps`` control steps late.

    Use with ``func=DelayedObservation``; ``params`` are ``term_func``'s.

    Attributes:
        term_func: The undelayed observation function.
        delay_steps: Delay in control steps.
    """

    term_func: Callable[..., torch.Tensor]
    delay_steps: int


class DelayedObservation(ManagerTermBase):
    """Fixed sensorimotor delay of an observation term (CPU ``SensorimotorCfg``).

    Step ``t`` returns the frame of step ``max(0, t - k)`` since the env's last
    reset, which fills the history with the reset frame. mjlab's own
    ``delay_*_lag`` fields are not used: their ``DelayBuffer`` draws from the
    global torch RNG every step even for a fixed lag (shifting every later
    random draw), and its partial-reset handling differs between mjlab
    versions. One frame is stored per control step (``common_step_counter``);
    a recompute in the same step (``reset(env_ids=...)``) only fills the rows
    reset since the last frame.
    """

    def __init__(self, cfg: DelayedObservationCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        self._func = cfg.term_func
        self._lag = int(cfg.delay_steps)
        self._buffer: FixedLagBuffer | None = None
        self._out: torch.Tensor | None = None
        self._step = -1
        self._pending = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._has_pending = False

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._pending[normalize_mjlab_env_ids(self._env, env_ids)] = True
        self._has_pending = True

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
        obs = self._func(env, **params)
        step = int(env.common_step_counter)
        if self._buffer is None or self._out is None:
            self._buffer = FixedLagBuffer(self._lag, obs)
            self._out, self._step = obs.clone(), step
            self._has_pending = False
            return self._out
        if self._has_pending:  # rows reset since the last frame: history = reset frame
            rows = self._pending.clone()
            self._buffer.refill(obs, rows)
            self._out[rows] = obs[rows]
            self._pending[:] = False
            self._has_pending = False
        if step != self._step:
            self._out, self._step = self._buffer.push(obs), step
        return self._out


def qpos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """CPU ``obs["qpos"]``: the entity's ``qpos`` in MuJoCo layout."""
    return MjlabEntityAccessor(env, asset_cfg.name).joint_pos()


def qvel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """CPU ``obs["qvel"]``: ``qvel * ctrl_dt``."""
    return MjlabEntityAccessor(env, asset_cfg.name).joint_vel() * env.step_dt


def _hinge_quats(angles: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Quaternions of the rotations about x, y, z by the columns of *angles*."""
    axes = torch.eye(3, device=angles.device, dtype=angles.dtype)
    return tuple(
        quat_from_angle_axis(angles[:, i], axes[i].expand(angles.shape[0], 3))
        for i in range(3)
    )


def _quat0(chain: FreeJointChain, like: torch.Tensor) -> torch.Tensor:
    quat0 = torch.tensor(chain.quat0, device=like.device, dtype=like.dtype)
    return quat0.expand(like.shape[0], 4)


def chains_to_qpos(q: torch.Tensor, chains: tuple[FreeJointChain, ...]) -> torch.Tensor:
    """Entity ``qpos`` with every 6-DoF chain as a 7-value freejoint block.

    A chain is 3 slides ``s`` then hinges ``(a, b, c)`` about x, y, z on a body of rest
    pose ``(pos0, quat0)``: the freejoint position is ``pos0 + R(quat0) s`` and the
    orientation ``quat0 * qx(a) * qy(b) * qz(c)``.
    """
    pieces, cursor = [], 0
    for chain in chains:
        start = chain.chain_start
        pieces.append(q[:, cursor:start])
        qx, qy, qz = _hinge_quats(q[:, start + 3 : start + 6])
        pos0 = torch.tensor(chain.pos0, device=q.device, dtype=q.dtype)
        quat0 = _quat0(chain, q)
        pieces.append(pos0 + quat_apply(quat0, q[:, start : start + 3]))
        pieces.append(quat_mul(quat_mul(quat_mul(quat0, qx), qy), qz))
        cursor = start + 6
    pieces.append(q[:, cursor:])
    return torch.cat(pieces, dim=-1)


def chains_to_qvel(
    q: torch.Tensor, v: torch.Tensor, chains: tuple[FreeJointChain, ...]
) -> torch.Tensor:
    """Entity ``qvel`` with every 6-DoF chain as a freejoint block (also 6 values).

    Freejoint velocity: world linear velocity ``R(quat0) s_dot`` and the body-frame
    angular velocity ``qz^-1 (qy^-1 (a_dot x) + b_dot y) + c_dot z``.
    """
    pieces, cursor = [], 0
    eye = torch.eye(3, device=v.device, dtype=v.dtype)
    for chain in chains:
        start = chain.chain_start
        pieces.append(v[:, cursor:start])
        _, qy, qz = _hinge_quats(q[:, start + 3 : start + 6])
        rate = v[:, start + 3 : start + 6]
        omega = quat_apply_inverse(qy, rate[:, :1] * eye[0]) + rate[:, 1:2] * eye[1]
        omega = quat_apply_inverse(qz, omega) + rate[:, 2:3] * eye[2]
        pieces.append(quat_apply(_quat0(chain, v), v[:, start : start + 3]))
        pieces.append(omega)
        cursor = start + 6
    pieces.append(v[:, cursor:])
    return torch.cat(pieces, dim=-1)


def qpos_chains(
    env: ManagerBasedRlEnv,
    chains: tuple[FreeJointChain, ...],
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """CPU ``obs["qpos"]`` of a model whose freejoints became 6-DoF chains."""
    return chains_to_qpos(MjlabEntityAccessor(env, asset_cfg.name).joint_pos(), chains)


def qvel_chains(
    env: ManagerBasedRlEnv,
    chains: tuple[FreeJointChain, ...],
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """CPU ``obs["qvel"]`` (``qvel * ctrl_dt``) for the same models as ``qpos_chains``."""
    acc = MjlabEntityAccessor(env, asset_cfg.name)
    return chains_to_qvel(acc.joint_pos(), acc.joint_vel(), chains) * env.step_dt


def act(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """CPU ``obs["act"]``: muscle activation state."""
    return MjlabEntityAccessor(env, asset_cfg.name).muscle_act()
