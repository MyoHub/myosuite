# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Observation terms shared by the MyoSuite mjlab tasks (CPU obs layout)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_apply_inverse,
    quat_from_angle_axis,
    quat_mul,
)

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEntityAccessor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import FreeJointChain

_ROBOT = SceneEntityCfg("robot")


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
