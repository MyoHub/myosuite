# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""SAR reorient action mapping and muscle conditions match the other basic envs."""

from __future__ import annotations

import numpy as np
import pytest

import myosuite
from myosuite.envs.myo.tasks.basic.arm.reorient_sar import ReorientSAREnvV0
from myosuite.envs.wrappers import FatigueWrapper
from myosuite.terms.base_action import sigmoid_muscle_activation
from myosuite.utils import gym

myosuite.register_all_envs()

pytestmark = pytest.mark.tier1

_GEOMETRIES = ("8", "100", "ID", "OOD")
_CONDITIONS = {
    "myoSarc": "SarcopeniaWrapper",
    "myoFati": "FatigueWrapper",
    "myoReaf": "ReafferentationWrapper",
}


def _wrapper_names(env) -> list[str]:
    names = []
    while hasattr(env, "env"):
        names.append(type(env).__name__)
        env = env.env
    return names


def _make(env_id: str) -> ReorientSAREnvV0:
    env = gym.make(env_id).unwrapped
    env.reset(seed=0)
    return env


@pytest.mark.parametrize("value", [-1.0, -0.5, 0.0, 1.0])
def test_normalized_action_is_mapped_through_muscle_sigmoid(value: float) -> None:
    env = _make("myoHandReorient8-v0")
    action = np.full(env.action_space.shape, value, dtype=np.float32)
    env.step(action)
    expected = action.copy()
    expected[:] = sigmoid_muscle_activation(action, np)
    np.testing.assert_array_equal(env.data.ctrl, expected)


@pytest.mark.parametrize(
    ("prefix", "geometry"),
    [(p, g) for p in _CONDITIONS for g in _GEOMETRIES],
)
def test_condition_variants_carry_their_condition(prefix: str, geometry: str) -> None:
    env = gym.make(f"{prefix}HandReorient{geometry}-v0")
    names = _wrapper_names(env)
    assert _CONDITIONS[prefix] in names
    assert not {*_CONDITIONS.values()} - {_CONDITIONS[prefix]} & {*names}


def test_sarcopenia_halves_peak_force() -> None:
    healthy = _make("myoHandReorient8-v0")
    sarc = _make("myoSarcHandReorient8-v0")
    assert np.all(healthy.model.actuator_gainprm[:, 2] > 0)
    np.testing.assert_allclose(
        sarc.model.actuator_gainprm[:, 2], 0.5 * healthy.model.actuator_gainprm[:, 2]
    )
    sarc.reset(seed=1)
    np.testing.assert_allclose(
        sarc.model.actuator_gainprm[:, 2], 0.5 * healthy.model.actuator_gainprm[:, 2]
    )


def test_fatigue_limits_excitation_and_resets() -> None:
    env = gym.make("myoFatiHandReorient8-v0")
    env.reset(seed=0)
    action = np.ones(env.action_space.shape, dtype=np.float32)
    for _ in range(20):
        env.step(action)
    fatigue = env.muscle_fatigue
    ctrl = env.unwrapped.data.ctrl
    assert np.all(fatigue.MF > 0)
    np.testing.assert_allclose(ctrl, fatigue.MA, rtol=1e-6)
    assert np.all(ctrl < sigmoid_muscle_activation(1.0, np))
    env.reset(seed=1)
    np.testing.assert_array_equal(fatigue.MF, 0.0)
    np.testing.assert_array_equal(fatigue.MA, 0.0)


def test_fatigue_variant_samples_the_healthy_geometry() -> None:
    healthy = gym.make("myoHandReorientOOD-v0").unwrapped
    fatigued = gym.make("myoHandReorientOOD-v0")
    fatigued = FatigueWrapper(fatigued, fatigue_reset_random=True).unwrapped
    gid = healthy.obj_gid
    for seed in range(3):
        healthy.reset(seed=seed)
        fatigued.reset(seed=seed)
        assert healthy.model.geom_type[gid] == fatigued.model.geom_type[gid]
        np.testing.assert_array_equal(
            healthy.model.geom_size[gid], fatigued.model.geom_size[gid]
        )


def test_reafferentation_reroutes_eip_to_epl() -> None:
    env = _make("myoReafHandReorient8-v0")
    eip, epl = env.model.actuator("EIP_r").id, env.model.actuator("EPL_r").id
    action = -np.ones(env.action_space.shape, dtype=np.float32)
    action[eip] = 1.0
    env.step(action)
    assert env.data.ctrl[eip] == 0.0
    assert env.data.ctrl[epl] == pytest.approx(sigmoid_muscle_activation(1.0, np))


def test_removed_muscle_condition_kwarg_raises() -> None:
    with pytest.raises(TypeError, match="muscle_condition"):
        gym.make("myoHandReorient8-v0", muscle_condition="sarcopenia")
