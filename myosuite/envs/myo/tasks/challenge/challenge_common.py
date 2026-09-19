# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reusable challenge env utilities for native Gymnasium rewrites."""

from __future__ import annotations

import numpy as np

from myosuite.core.muscle_conditions import apply_sarcopenia_to_model
from myosuite.physics.fatigue import CumulativeFatigue
from myosuite.terms.base_action import sigmoid_muscle_activation


class MuscleActionMixin:
    """Shared muscle-condition setup and action processing."""

    model: any
    data: any
    frame_skip: int
    normalize_act: bool
    muscle_condition: str
    action_space: any
    _muscle_act_ind: np.ndarray

    def init_muscle_condition(self) -> None:
        """Apply configured muscle-condition behavior."""
        if self.muscle_condition == "sarcopenia":
            apply_sarcopenia_to_model(self.model, force_scale=0.5)
        elif self.muscle_condition == "fatigue":
            self.muscle_fatigue = CumulativeFatigue(
                self.model, self.frame_skip, seed=None
            )
        elif self.muscle_condition == "reafferentation":
            self.epl_pos = self.model.actuator("EPL").id
            self.eip_pos = self.model.actuator("EIP").id

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
        if self.muscle_condition == "fatigue":
            ctrl[self._muscle_act_ind], _, _ = self.muscle_fatigue.compute_act(
                ctrl[self._muscle_act_ind]
            )
        elif self.muscle_condition == "reafferentation":
            ctrl[self.epl_pos] = ctrl[self.eip_pos].copy()
            ctrl[self.eip_pos] = 0.0
        self.data.ctrl[:] = ctrl
