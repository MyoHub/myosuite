# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Functional muscle groups of the fatigue model: name rules, rows and model coverage."""

from __future__ import annotations

import importlib.util

import mujoco
import pytest

from myosuite import make_env
from myosuite.core.muscle_conditions import MUSCLE_FATIGUE_PARAMS, _muscle_group
from myosuite.core.muscle_groups import FUNCTIONAL_MUSCLE_GROUPS, MUSCLE_FMG

pytestmark = pytest.mark.tier1

_MUSCLEMIMIC = importlib.util.find_spec("musclemimic_models") is not None


@pytest.mark.parametrize(
    ("name", "group"),
    [
        ("BIClong", "Elbow-Flexor"),
        ("robot/BIClong", "Elbow-Flexor"),  # mjlab scene prefix
        ("ECRL_r", "Wrist-Extensor"),  # hand-pose models
        ("BIClong_left", "Elbow-Flexor"),  # MuscleMimic left arm
        ("a1_BIClong_left", "Elbow-Flexor"),  # two-agent scene
        ("robot/a0_soleus_r", "Ankle-Plantarflexor"),
        ("R:ECRL", "Wrist-Extensor"),  # side prefix
        ("soleus_l", "Ankle-Plantarflexor"),  # leg muscles keep their side
        ("rect_abd_r", "Trunk"),
        ("QL_post_I.2-L3_l", "Trunk"),
        ("Ps_L5_VB_r", "Trunk"),  # spinal psoas fascicle
        ("psoas_r", "Hip-Flexor"),  # leg psoas
        ("unknown_muscle", "Default"),
    ],
)
def test_muscle_group_name_rules(name: str, group: str) -> None:
    assert _muscle_group(name) == group


def test_groups_are_declared_and_resolve_to_a_row() -> None:
    """Every assigned group is declared and has a row of its own or of its parent joint.

    ``Hip-*`` and ``Thigh-Hamstring`` have no row (documented Default fallback).
    """
    assert set(MUSCLE_FMG.values()) <= FUNCTIONAL_MUSCLE_GROUPS
    rowless = {g for g in set(MUSCLE_FMG.values()) if g not in MUSCLE_FATIGUE_PARAMS}
    unresolved = {g for g in rowless if g.split("-")[0] not in MUSCLE_FATIGUE_PARAMS}
    assert {g.split("-")[0] for g in unresolved} == {"Hip", "Thigh"}


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
    pytest.param(
        "myoChallengeChaseTagFBVs-v0",
        marks=pytest.mark.skipif(not _MUSCLEMIMIC, reason="needs musclemimic_models"),
    ),
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
