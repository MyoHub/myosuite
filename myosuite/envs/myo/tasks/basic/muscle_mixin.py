# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Muscle-condition setup, per-episode reset and action mapping of the basic hand/arm envs."""

from __future__ import annotations

from typing import Any

import numpy as np

from myosuite.core.muscle_conditions import apply_sarcopenia_to_model
from myosuite.physics.fatigue import CumulativeFatigue
from myosuite.terms.base_action import (
    MotorNoiseCfg,
    sample_motor_noise,
    sigmoid_muscle_activation,
)


class MuscleConditionMixin:
    """Shared muscle-condition behaviour of the pose, key-turn, obj-hold, pen and SAR-reorient envs.

    The host env provides ``model``, ``data``, ``frame_skip``, ``normalize_act``,
    ``muscle_condition``, ``motor_noise``, ``fatigue_reset_vec``,
    ``fatigue_reset_random``, ``np_random``, ``_muscle_act_ind`` and the
    actuator-name suffix ``_name_sfx``.
    """

    model: Any
    data: Any
    frame_skip: int
    normalize_act: bool
    muscle_condition: str
    motor_noise: MotorNoiseCfg = MotorNoiseCfg()  # off unless the host env sets it
    fatigue_reset_vec: Any
    fatigue_reset_random: bool
    np_random: np.random.Generator
    _muscle_act_ind: np.ndarray
    _name_sfx: str

    def _init_muscle_condition(self) -> None:
        """Apply the muscle condition to the compiled model."""
        if self.muscle_condition == "sarcopenia":
            apply_sarcopenia_to_model(self.model, force_scale=0.5)
        elif self.muscle_condition == "fatigue":
            self.muscle_fatigue = CumulativeFatigue(
                self.model, self.frame_skip, seed=None
            )
        elif self.muscle_condition == "reafferentation":
            sfx = self._name_sfx
            self.EPLpos = self.model.actuator(f"EPL{sfx}").id
            self.EIPpos = self.model.actuator(f"EIP{sfx}").id

    def _reset_muscle_condition(self) -> None:
        """Reset the fatigue state for a new episode (drawn from the env RNG)."""
        if self.muscle_condition == "fatigue":
            self.muscle_fatigue.reset(
                fatigue_reset_vec=self.fatigue_reset_vec,
                fatigue_reset_random=self.fatigue_reset_random,
                np_random=self.np_random,
            )

    def _apply_action(self, action: np.ndarray) -> None:
        """Map an action to MuJoCo ctrl and write it to ``data.ctrl``.

        Args:
            action: Action vector in the action space (already clipped).

        Note:
            ctrl stays float32 through the sigmoid so the result is truncated back
            to float32 precision, as in the original ``BaseV0.step()`` (parity).
        """
        ctrl = action.copy()

        if self.model.na > 0 and self.normalize_act:
            # Muscle actuators: [-1, 1] -> [0, 1].
            ctrl[self._muscle_act_ind] = sigmoid_muscle_activation(
                ctrl[self._muscle_act_ind], np
            )
        elif self.normalize_act:
            # Motor actuators: [-1, 1] -> ctrl range (float64 promotes the result).
            ctrl_range = self.model.actuator_ctrlrange
            ctrl = (
                np.mean(ctrl_range, axis=-1)
                + ctrl * (ctrl_range[:, 1] - ctrl_range[:, 0]) / 2.0
            )

        # Motor noise on muscle excitations (before fatigue); no RNG draw when off.
        ctrl[self._muscle_act_ind] = sample_motor_noise(
            ctrl[self._muscle_act_ind],
            self.motor_noise,
            self.np_random.standard_normal,
            np,
        )

        if self.muscle_condition == "fatigue":
            ctrl[self._muscle_act_ind], _, _ = self.muscle_fatigue.compute_act(
                ctrl[self._muscle_act_ind]
            )
        elif self.muscle_condition == "reafferentation":
            ctrl[self.EPLpos] = ctrl[self.EIPpos].copy()
            ctrl[self.EIPpos] = 0.0

        self.data.ctrl[:] = ctrl
