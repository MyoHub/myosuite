# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Muscle strength scaling and calibration against MuJoCo's own forces and known subjects."""

from __future__ import annotations

import copy

import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.core.muscle_strength import (
    MaxTorqueMeasurement,
    calibrate_muscle_strength,
    max_torque_terms,
    scale_muscle_strength,
    scale_muscle_strength_spec,
    strength_factor_from_body_mass,
)
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

pytestmark = pytest.mark.tier1

_FLEXORS, _EXTENSORS = ("BIClong", "BICshort", "BRA"), ("TRIlong", "TRIlat", "TRImed")

# A muscle with an automatic peak force (force="-1") and a motor.
_XML = """
<mujoco>
  <worldbody>
    <site name="origin" pos="0 0 .1"/>
    <body><joint name="hinge" axis="0 1 0" range="-1 1" stiffness="4"/>
      <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02" mass="1"/>
      <site name="insertion" pos=".2 0 .05"/></body>
  </worldbody>
  <tendon><spatial name="path"><site site="origin"/><site site="insertion"/></spatial></tendon>
  <actuator>
    <muscle name="m" tendon="path" force="-1" scale="200"/>
    <motor name="motor" joint="hinge" gear="3"/>
  </actuator>
</mujoco>
"""


def _forces(model: mujoco.MjModel, qpos: np.ndarray, level: float) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.act[:] = level
    data.ctrl[:] = level
    mujoco.mj_forward(model, data)
    return data.actuator_force.copy()


@pytest.fixture(scope="module")
def elbow() -> mujoco.MjModel:
    env = make_env("myoElbowPose1D6MRandom-v0")
    yield env.unwrapped.model
    env.close()


def test_scaling_multiplies_active_and_passive_force(elbow: mujoco.MjModel) -> None:
    factors = np.random.default_rng(0).uniform(0.5, 2.0, elbow.nu)
    scaled = copy.deepcopy(elbow)
    scale_muscle_strength(scaled, factors)
    q = np.array([0.3])  # short flexors, stretched extensors: passive forces present
    for level in (0.0, 1.0):
        np.testing.assert_allclose(
            _forces(scaled, q, level), factors * _forces(elbow, q, level), rtol=1e-12
        )
    assert np.abs(_forces(elbow, q, 0.0)).max() > 1.0


def test_automatic_peak_force_and_spec_path() -> None:
    model = mujoco.MjModel.from_xml_string(_XML)
    scaled = copy.deepcopy(model)
    scale_muscle_strength(scaled, [2.0, 5.0])
    q = np.array([0.4])
    np.testing.assert_allclose(
        _forces(scaled, q, 1.0), [2.0, 1.0] * _forces(model, q, 1.0), rtol=1e-12
    )
    spec = mujoco.MjSpec.from_string(_XML)
    from_spec = scale_muscle_strength_spec(spec, [2.0, 5.0]).compile()
    np.testing.assert_allclose(
        _forces(from_spec, q, 1.0), _forces(scaled, q, 1.0), rtol=1e-12
    )
    with pytest.raises(ValueError):
        scale_muscle_strength(scaled, [0.0, 1.0])


def test_trial_torque_matches_mujoco(elbow: mujoco.MjModel) -> None:
    trial = MaxTorqueMeasurement(
        "r_elbow_flex", np.array([1.57]), torque=50.0, direction=-1.0
    )
    active, passive, joints = max_torque_terms(elbow, trial)
    data = mujoco.MjData(elbow)
    data.qpos[:] = trial.qpos
    data.act[:] = active > 0  # extensors at full activation
    mujoco.mj_forward(elbow, data)
    mujoco_torque = -(data.qfrc_actuator + data.qfrc_passive)[0]
    np.testing.assert_allclose(
        active[active > 0].sum() + passive.sum() + joints, mujoco_torque, rtol=1e-10
    )
    assert {elbow.actuator(i).name for i in np.flatnonzero(active > 0)} == set(
        _EXTENSORS
    )


def test_trial_torque_includes_joint_springs() -> None:
    model = mujoco.MjModel.from_xml_string(_XML)
    trial = MaxTorqueMeasurement("hinge", np.array([0.4]), torque=5.0)
    active, passive, joints = max_torque_terms(model, trial)
    assert joints == pytest.approx(-4 * 0.4)
    data = mujoco.MjData(model)
    data.qpos[:], data.act[:] = trial.qpos, (active > 0)[0]
    mujoco.mj_forward(model, data)
    mujoco_torque = (
        data.actuator_force[0] * data.actuator_moment[0] + data.qfrc_passive[0]
    )
    np.testing.assert_allclose(
        max(active[0], 0) + passive[0] + joints, mujoco_torque, rtol=1e-10
    )


def test_coupled_trial_torque_is_the_virtual_work_of_the_muscles() -> None:
    env = make_env("myoArmReachRandom-v0")
    model = env.unwrapped.model
    couplings = JointCouplings.from_model(model)
    qpos = model.qpos0.copy()
    adr = model.joint("shoulder_elv_r").qposadr[0]
    qpos[adr] = 0.8
    active, _, _ = max_torque_terms(
        model, MaxTorqueMeasurement("shoulder_elv_r", qpos, torque=40.0)
    )
    # Active force times each muscle's length change along the coupled motion
    # (dependent joints re-projected), by central differences of actuator_length.
    h = 1e-6
    length = []
    for sign in (-1, 1):
        q = qpos.copy()
        q[adr] += sign * h
        project_joint_couplings(couplings, q)
        data = mujoco.MjData(model)
        data.qpos[:] = q
        mujoco.mj_fwdPosition(model, data)
        length.append(data.actuator_length.copy())
    centre = qpos.copy()
    project_joint_couplings(couplings, centre)
    active_force = _forces(model, centre, 1.0) - _forces(model, centre, 0.0)
    virtual = active_force * (length[1] - length[0]) / (2 * h)
    np.testing.assert_allclose(active, virtual, rtol=1e-4, atol=1e-3)
    assert np.abs(active).max() > 10.0
    env.close()


def test_calibration_recovers_a_known_subject(elbow: mujoco.MjModel) -> None:
    truth = np.array(
        [1.3 if elbow.actuator(i).name in _FLEXORS else 0.8 for i in range(elbow.nu)]
    )
    subject = copy.deepcopy(elbow)
    scale_muscle_strength(subject, truth)
    trials = []
    for angle in (0.5, 1.2, 1.9):
        for direction in (1.0, -1.0):
            probe = MaxTorqueMeasurement(
                "r_elbow_flex", np.array([angle]), 1.0, direction
            )
            active, passive, joints = max_torque_terms(subject, probe)
            torque = active[active > 0].sum() + passive.sum() + joints
            trials.append(
                MaxTorqueMeasurement(
                    "r_elbow_flex", np.array([angle]), torque, direction
                )
            )
    cal = calibrate_muscle_strength(elbow, trials)
    np.testing.assert_allclose(cal.predicted, cal.measured, rtol=5e-3)
    assert (
        np.abs(cal.nominal / cal.measured - 1).max() > 0.15
    )  # the nominal model was off
    for group, value in ((_FLEXORS, 1.3), (_EXTENSORS, 0.8)):
        ids = [elbow.actuator(n).id for n in group]
        assert abs(cal.factors[ids].mean() - value) < 0.05


def test_muscles_no_trial_involves_keep_the_prior() -> None:
    env = make_env("myoArmReachRandom-v0")
    model = env.unwrapped.model
    qpos = model.qpos0.copy()
    qpos[model.joint("elbow_flexion_r").qposadr[0]] = 1.57
    prior = strength_factor_from_body_mass(60.0, 75.0)
    cal = calibrate_muscle_strength(
        model, [MaxTorqueMeasurement("elbow_flexion_r", qpos, 45.0)], prior=prior
    )
    assert prior == pytest.approx(0.8)
    involved = {model.actuator(i).name for i in np.flatnonzero(cal.involved)}
    assert {"BIClong", "BRA"} <= involved and "DELT1" not in involved
    np.testing.assert_allclose(cal.factors[~cal.involved], prior)
    np.testing.assert_allclose(cal.predicted, [45.0], rtol=0.05)
    env.close()


def test_rejects_invalid_trials(elbow: mujoco.MjModel) -> None:
    with pytest.raises(ValueError):
        calibrate_muscle_strength(elbow, [])
    with pytest.raises(ValueError):
        calibrate_muscle_strength(
            elbow, [MaxTorqueMeasurement("r_elbow_flex", np.array([1.0]), 0.0)]
        )
    with pytest.raises(ValueError):
        max_torque_terms(
            elbow, MaxTorqueMeasurement("no_such_joint", np.array([1.0]), 10.0)
        )
    with pytest.raises(ValueError):
        strength_factor_from_body_mass(-1.0, 75.0)
