# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Opt-in effort reward terms for mjlab twins (wrap :mod:`myosuite.terms.effort`).

Each returns the effort measure itself (non-negative, shape ``(N,)``); give the
``RewardTermCfg`` a negative weight to penalise it, e.g.
``RewardTermCfg(func=mdp.metabolic_energy_rate, weight=-1e-3)``. They read
actuator length/velocity/force and ``qfrc_actuator``, so register
``mdp.sync_forward`` as the first termination (cross-backend contract).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEntityAccessor
from myosuite.terms import effort

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def muscle_mechanical_power(
    env: ManagerBasedRlEnv, mode: str = "abs", asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
    """Muscle mechanical power (W), see :func:`myosuite.terms.effort.muscle_mechanical_power`."""
    out = effort.muscle_mechanical_power(
        MjlabEntityAccessor(env, asset_cfg.name), {}, mode=mode
    )
    return out["muscle_power_abs" if mode == "abs" else "muscle_power_positive"]


def metabolic_energy_rate(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT, **kwargs: Any
) -> torch.Tensor:
    """Umberger metabolic rate (W), see :func:`myosuite.terms.effort.metabolic_energy_rate`."""
    accessor = MjlabEntityAccessor(env, asset_cfg.name)
    return effort.metabolic_energy_rate(accessor, {}, **kwargs)["metabolic_rate"]


def consumed_endurance_step(
    env: ManagerBasedRlEnv,
    shoulder_dof_ids: Any,
    max_shoulder_torque: float = effort.CE_MAX_SHOULDER_TORQUE_MALE,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Percent of shoulder endurance spent this step, see :func:`myosuite.terms.effort.consumed_endurance`."""
    out = effort.consumed_endurance(
        MjlabEntityAccessor(env, asset_cfg.name),
        {},
        shoulder_dof_ids=shoulder_dof_ids,
        max_shoulder_torque=max_shoulder_torque,
    )
    return out["ce_step"]


def fatigue_mf(
    env: ManagerBasedRlEnv,
    action_name: str = "muscles",
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Mean fatigued fraction MF of the twin's 3CC-r state, see :func:`myosuite.terms.effort.fatigue_effort`."""
    state = env.action_manager.get_term(action_name).fatigue_state
    if state is None:
        return torch.zeros(env.num_envs, device=env.device)
    out = effort.fatigue_effort(
        MjlabEntityAccessor(env, asset_cfg.name), {"fatigue": state}
    )
    return out["fatigue_mf"]


def joint_limit_discomfort(
    env: ManagerBasedRlEnv, margin: float = 0.1, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
    """Smooth joint-limit discomfort, see :func:`myosuite.terms.effort.joint_limit_discomfort`."""
    out = effort.joint_limit_discomfort(
        MjlabEntityAccessor(env, asset_cfg.name), {}, margin=margin
    )
    return out["joint_limit_discomfort"]
