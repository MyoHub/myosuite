# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Every muscle of the registered envs has a functional fatigue group.

The name rules and parameter rows are tested in myo_sim (``myo_sim.physics.fatigue``).
"""

from __future__ import annotations

import importlib.util

import mujoco
import pytest

from myosuite import make_env
from myosuite.core.muscle_conditions import _muscle_group

pytestmark = pytest.mark.tier1

_MUSCLEMIMIC = importlib.util.find_spec("musclemimic_models") is not None


_MODELS = [
    "myoElbowPose1D6MRandom-v0",
    "myoHandPoseRandom-v0",
    "myoArmReachRandom-v0",
    "myoLegWalk-v0",
    "myoChallengeOslRunFixed-v0",
    "myoTorsoPoseFixed-v0",
    "myoChallengeTableTennisP1-v0",
    "myoChallengeSoccerP1-v0",
    pytest.param(
        "myoMimicFullbody-v0",
        marks=pytest.mark.skipif(not _MUSCLEMIMIC, reason="needs musclemimic_models"),
    ),
    "myoChallengeChaseTagFBVs-v0",
]


@pytest.mark.parametrize("env_id", _MODELS)
def test_every_muscle_has_a_group(env_id: str) -> None:
    """No muscle of these models falls back to Default by name."""
    model = make_env(env_id).unwrapped.model
    muscles = [
        model.actuator(i).name
        for i in range(model.nu)
        if model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE
    ]
    assert muscles
    assert [m for m in muscles if _muscle_group(m) == "Default"] == []
