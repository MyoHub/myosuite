# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Action mapping of the basic hand/arm envs; muscle stages come from wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np

from myosuite.envs.muscle_stages import CtrlStageHost
from myosuite.terms.base_action import sigmoid_muscle_activation


class MuscleActionMixin(CtrlStageHost):
    """Shared action mapping of the pose, key-turn, obj-hold, pen and SAR-reorient envs.

    Noise, fatigue and reafferentation are not part of the env: wrappers install
    them as stages (:mod:`myosuite.envs.muscle_stages`) that run after the map.
    The host env provides ``model``, ``data``, ``normalize_act``, ``np_random``,
    ``_muscle_act_ind`` and the actuator-name suffix ``_name_sfx``.
    """

    model: Any
    data: Any
    normalize_act: bool
    np_random: np.random.Generator
    _muscle_act_ind: np.ndarray
    _name_sfx: str

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

        self.data.ctrl[:] = self._run_ctrl_stages(ctrl)
