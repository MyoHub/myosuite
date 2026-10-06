# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reusable challenge env utilities for native Gymnasium rewrites."""

from __future__ import annotations

import mujoco
import numpy as np

from myosuite.envs.muscle_stages import CtrlStageHost
from myosuite.terms.base_action import (
    sigmoid_muscle_activation,
)


def solved_step_count(path: dict) -> float:
    """Number of steps of a rollout path on which the task was solved."""
    return float(np.sum(np.asarray(path["env_infos"]["rwd_dict"]["solved"]) * 1.0))


def mean_effort(paths: list, key: str = "act_reg", sign: float = -1.0) -> float:
    """Mean over paths of the per-path mean of ``rwd_dict[key]``, times ``sign``."""
    return float(
        sign * np.mean([np.mean(p["env_infos"]["rwd_dict"][key]) for p in paths])
    )


def joint_limit_forces(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Generalized forces of the joint-limit constraints alone.

    Args:
        model: MuJoCo model.
        data: MuJoCo data holding the constraint forces of the current state.

    Returns:
        ``J^T f`` over the joint-limit rows of the constraint Jacobian, shape ``(nv,)``.
    """
    is_limit = data.efc_type == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
    limit_force = np.where(is_limit, data.efc_force, 0.0)
    qfrc = np.zeros(model.nv)
    mujoco.mj_mulJacTVec(model, data, qfrc, limit_force)
    return qfrc


class MuscleActionMixin(CtrlStageHost):
    """Shared action mapping of the challenge muscle envs.

    Noise, fatigue and reafferentation are wrapper-installed stages that run
    after the map (:mod:`myosuite.envs.muscle_stages`).
    """

    model: any
    data: any
    normalize_act: bool
    np_random: np.random.Generator
    action_space: any
    _muscle_act_ind: np.ndarray

    def apply_action(self, action: np.ndarray) -> None:
        """Project and write control action into MuJoCo ctrl buffer."""
        ctrl = np.clip(action, self.action_space.low, self.action_space.high).astype(
            np.float64
        )
        if self.model.na > 0 and self.normalize_act:
            ctrl[self._muscle_act_ind] = sigmoid_muscle_activation(
                ctrl[self._muscle_act_ind], np
            )
        elif self.normalize_act and self.model.nu > 0:
            cr = self.model.actuator_ctrlrange
            ctrl = np.mean(cr, axis=-1) + ctrl * (cr[:, 1] - cr[:, 0]) / 2.0
        ctrl = self._run_ctrl_stages(ctrl)
        self.data.ctrl[:] = ctrl
