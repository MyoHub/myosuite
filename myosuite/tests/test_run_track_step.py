# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for the OslRun (``RunTrackEnv``) step hot path.

The step reads the OSL gains in place instead of deep-copying the state
machine, gathers its observations with precomputed indices and computes the
joint-limit forces of the pain term once. Each fast path is compared bit for
bit with the per-name / copying computation it replaces, in the same state.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

import mujoco
import numpy as np
import pytest

import myosuite  # noqa: F401  (registers the envs)
from myosuite.envs.myo.assets.leg.myoosl_control import MyoOSLController
from myosuite.utils import gym
from myosuite import make_env

pytestmark = pytest.mark.tier1

ENV_ID = "myoChallengeOslRunFixed-v0"
N_STEPS = 60


@pytest.fixture(scope="module")
def env() -> Iterator[gym.Env]:
    env = make_env(ENV_ID)
    yield env
    env.close()


def _rollout(env: gym.Env, seed: int) -> Iterator[Any]:
    """Seeded random-action steps (resetting on termination); yields the env."""
    space = env.action_space
    actions = np.random.default_rng(seed).uniform(
        space.low, space.high, size=(N_STEPS, *space.shape)
    )
    env.unwrapped.reset(seed=seed)
    for action in actions.astype(space.dtype):
        *_, terminated, truncated, _ = env.unwrapped.step(action)
        yield env.unwrapped
        if terminated or truncated:
            env.unwrapped.reset()


def _assert_bits_equal(actual: Any, expected: Any, what: str) -> None:
    a, e = np.asarray(actual), np.asarray(expected)
    assert a.dtype == e.dtype and a.shape == e.shape, what
    assert a.tobytes() == e.tobytes(), what


def _reference_pain(u: Any) -> float:
    """The pain term with one masked ``mj_mulJacTVec`` per joint (the old path)."""
    pain = 0.0
    for joint in u.PAIN_JNT:
        efc_force = u.data.efc_force.copy()
        efc_force[u.data.efc_type != mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT] = 0.0
        qfrc = np.zeros(u.model.nv)
        mujoco.mj_mulJacTVec(u.model, u.data, qfrc, efc_force)
        frc = qfrc[u.model.joint(joint).dofadr].squeeze()
        pain += np.clip(np.abs(frc), -1000, 1000) / 1000
    return pain / len(u.PAIN_JNT)


def test_osl_step_does_not_deepcopy(
    env: gym.Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-step OSL torque reads the state machine gains without copying."""
    calls: list[str] = []
    deepcopy = copy.deepcopy

    def counting_deepcopy(obj: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(type(obj).__name__)
        return deepcopy(obj, *args, **kwargs)

    monkeypatch.setattr(copy, "deepcopy", counting_deepcopy)
    for _ in _rollout(env, seed=0):
        pass
    assert not calls, f"{len(calls)} deepcopy calls in {N_STEPS} steps: {calls[:4]}"


def test_osl_getters_still_return_copies() -> None:
    """Public getters keep their copy semantics: editing a result changes nothing."""
    ctrl = MyoOSLController(body_mass=80.0)
    ctrl.start()
    live = ctrl.STATE_MACHINE.current_state.state_variables
    before = dict(live)
    state = ctrl.STATE_MACHINE.get_current_state
    state.state_variables["knee_stiffness"] = -1.0
    gains = state.get_variables()
    gains["knee_damping"] = -1.0
    assert ctrl.STATE_MACHINE.current_state.state_variables == before


def test_osl_torque_matches_copying_getters() -> None:
    """get_osl_torque equals the impedance law on the deep-copied gains, in every phase."""
    rng = np.random.default_rng(0)
    ctrl = MyoOSLController(body_mass=80.0)
    ctrl.start()
    visited = set()
    for _ in range(400):
        sens = {
            "knee_angle": rng.uniform(-0.2, 1.5),
            "knee_vel": rng.uniform(-3.0, 3.0),
            "ankle_angle": rng.uniform(-0.5, 0.5),
            "ankle_vel": rng.uniform(-3.0, 3.0),
            "load": rng.uniform(0.0, 1000.0),
        }
        ctrl.update(sens)
        visited.add(ctrl.STATE_MACHINE.current_state.name)
        gains = ctrl.STATE_MACHINE.get_current_state.get_variables()
        torque = ctrl.get_osl_torque()
        for jnt in ("knee", "ankle"):
            peak = ctrl.HARDWARE[jnt]["peak_torque"]
            expected = np.clip(
                gains[f"{jnt}_stiffness"]
                * (gains[f"{jnt}_target_angle"] - sens[f"{jnt}_angle"])
                - gains[f"{jnt}_damping"] * sens[f"{jnt}_vel"],
                -peak,
                peak,
            )
            _assert_bits_equal(torque[jnt], expected, f"{jnt} torque")
    assert visited == {"e_stance", "l_stance", "e_swing", "l_swing"}


def test_osl_torque_needs_a_running_state_machine() -> None:
    ctrl = MyoOSLController(body_mass=80.0)
    with pytest.raises(RuntimeError, match="not running"):
        ctrl.get_osl_torque()


def test_obs_pain_and_osl_inputs_match_per_name_reads(env: gym.Env) -> None:
    """Index-gathered obs, OSL sensor inputs and pain equal the per-name reads."""
    nonzero_pain = 0
    for u in _rollout(env, seed=1):
        d = u.data
        bio_jnt, bio_act = u.BIOLOGICAL_JNT, u.BIOLOGICAL_ACT
        expected = {
            "internal_qpos": np.array([d.joint(j).qpos[0] for j in bio_jnt]),
            "internal_qvel": np.array([d.joint(j).qvel[0] for j in bio_jnt]) * u.dt,
            "grf": np.array([d.sensor(n).data[0] for n in u.grf_sensor_names]),
            "muscle_length": np.array([d.actuator(a).length[0] for a in bio_act]),
            "muscle_velocity": np.clip(
                np.array([d.actuator(a).velocity[0] for a in bio_act]), -100, 100
            ),
            "muscle_force": np.clip(
                np.array([d.actuator(a).force[0] for a in bio_act]) / 1000, -100, 100
            ),
        }
        obs_dict = u._get_obs_dict(u._accessor)
        for key, value in expected.items():
            _assert_bits_equal(obs_dict[key], value, key)
        knee, ankle = d.joint("osl_knee_angle_r"), d.joint("osl_ankle_angle_r")
        osl_sens = {
            "knee_angle": knee.qpos[0],
            "knee_vel": knee.qvel[0],
            "ankle_angle": ankle.qpos[0],
            "ankle_vel": ankle.qvel[0],
            "load": -1.0 * d.sensor("r_osl_load").data[1],
        }
        for key, value in u._get_osl_sens().items():
            _assert_bits_equal(value, osl_sens[key], key)
        pain = u._get_pain(obs_dict)
        _assert_bits_equal(pain, _reference_pain(u), "pain")
        nonzero_pain += pain != 0.0
    assert nonzero_pain > N_STEPS // 2, "the rollout never loads a joint limit"


def test_pain_multiplies_the_constraint_jacobian_once(
    env: gym.Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The joint-limit forces of all pain joints come from one J^T f product."""
    u = env.unwrapped
    u.reset(seed=2)
    u.step(np.zeros(env.action_space.shape, dtype=env.action_space.dtype))
    calls: list[int] = []
    mul_jac_t_vec = mujoco.mj_mulJacTVec

    def counting(*args: Any) -> None:
        calls.append(1)
        mul_jac_t_vec(*args)

    monkeypatch.setattr(mujoco, "mj_mulJacTVec", counting)
    u._get_pain({})
    assert len(calls) == 1
