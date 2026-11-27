# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""SAR reorient episodes start palm up, as in the legacy env."""

from __future__ import annotations

import numpy as np
import pytest

import myosuite  # noqa: F401
from myosuite import make_env
from myosuite.core.terms.base_action import (
    inverse_buchanan_excitation,
    sigmoid_muscle_activation,
)


pytestmark = pytest.mark.tier1


@pytest.mark.parametrize("geometry", ["8", "100", "ID", "OOD"])
def test_object_rests_in_the_relaxed_hand(geometry: str) -> None:
    env = make_env(f"myoHandReorient{geometry}-v0")
    resting_tone = float(
        inverse_buchanan_excitation(sigmoid_muscle_activation(0.0, np), np)
    )
    action = np.full(env.action_space.shape, resting_tone, dtype=np.float32)
    for seed in range(2):
        env.reset(seed=seed)
        # A full episode (max_episode_steps=50) without dropping the object.
        for step in range(50):
            _, _, terminated, _, _ = env.step(action)
            assert not terminated, f"seed {seed}: object dropped at step {step}"
