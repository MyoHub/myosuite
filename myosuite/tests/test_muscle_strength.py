# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Facade contracts and MyoSuite integration checks for muscle strength calibration."""

from __future__ import annotations

import copy
import importlib
import subprocess
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.core import muscle_strength
from myosuite.core.muscle_strength import (
    MaxTorqueMeasurement,
    StrengthCalibration,
    calibrate_muscle_strength,
    max_torque_terms,
    scale_muscle_strength,
    scale_muscle_strength_spec,
    strength_factor_from_body_mass,
)
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

pytestmark = pytest.mark.tier1

_FLEXORS, _EXTENSORS = ("BIClong", "BICshort", "BRA"), ("TRIlong", "TRIlat", "TRImed")

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


def _require_library() -> None:
    try:
        mod = importlib.import_module("myo_sim.physics.muscle_strength")
    except ModuleNotFoundError:
        pytest.skip("myo_sim.physics.muscle_strength (#186/#195) not installed")
    if not hasattr(mod, "calibrate_muscle_strength"):
        pytest.skip("myo_sim.physics.muscle_strength lacks MVC calibration (#195)")


def _forces(model: mujoco.MjModel, qpos: np.ndarray, level: float) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.act[:] = level
    data.ctrl[:] = level
    mujoco.mj_forward(model, data)
    return data.actuator_force.copy()


@pytest.fixture(scope="module")
def elbow() -> mujoco.MjModel:
    _require_library()
    env = make_env("myoElbowPose1D6MRandom-v0")
    yield env.unwrapped.model
    env.close()


def test_facade_is_lazy_and_reports_actionable_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = f"""
import importlib.abc
import sys
class MissingStrength(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'myo_sim.physics.muscle_strength':
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, MissingStrength())
import mujoco
import numpy as np
from myosuite.core.muscle_conditions import _peak_force, apply_sarcopenia_to_model
from myosuite.core.muscle_strength import (
    MaxTorqueMeasurement,
    calibrate_muscle_strength,
    max_torque_terms,
    scale_muscle_strength,
    scale_muscle_strength_spec,
    strength_factor_from_body_mass,
)
assert 'myo_sim.physics.muscle_strength' not in sys.modules
spec = mujoco.MjSpec.from_string({_XML!r})
model = spec.copy().compile()
gain = _peak_force(model)
apply_sarcopenia_to_model(model, 0.5)
np.testing.assert_allclose(model.actuator_gainprm[0, 2], gain[0] * 0.5)
trial = MaxTorqueMeasurement('hinge', np.array([0.2]), 5.0)
calls = (
    (lambda: scale_muscle_strength(model, 1.0), 'MyoHub/myo_sim#186'),
    (lambda: scale_muscle_strength_spec(spec, 1.0), 'MyoHub/myo_sim#186'),
    (lambda: strength_factor_from_body_mass(70.0, 75.0), 'MyoHub/myo_sim#195'),
    (lambda: max_torque_terms(model, trial), 'MyoHub/myo_sim#195'),
    (lambda: calibrate_muscle_strength(model, [trial]), 'MyoHub/myo_sim#195'),
)
for call, expected in calls:
    try:
        call()
    except ImportError as exc:
        assert expected in str(exc) and 'Install' in str(exc)
    else:
        raise AssertionError('missing API did not raise ImportError')
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr

    def broken(name: str) -> object:
        raise ModuleNotFoundError("unrelated dependency defect", name="broken_dep")

    monkeypatch.setattr(muscle_strength.importlib, "import_module", broken)
    model = mujoco.MjModel.from_xml_string(_XML)
    with pytest.raises(ModuleNotFoundError) as exc_info:
        scale_muscle_strength(model, 1.0)
    assert exc_info.value.name == "broken_dep"

    monkeypatch.setattr(
        muscle_strength.importlib,
        "import_module",
        lambda name: SimpleNamespace(scale_muscle_strength=lambda m, f: None),
    )
    with pytest.raises(ImportError, match="MyoHub/myo_sim#195"):
        calibrate_muscle_strength(
            model, [MaxTorqueMeasurement("hinge", np.zeros(1), 5.0)]
        )


def test_scaling_multiplies_active_and_passive_force(elbow: mujoco.MjModel) -> None:
    factors = np.random.default_rng(0).uniform(0.5, 2.0, elbow.nu)
    scaled = copy.deepcopy(elbow)
    scale_muscle_strength(scaled, factors)
    q = np.array([0.3])
    for level in (0.0, 1.0):
        np.testing.assert_allclose(
            _forces(scaled, q, level), factors * _forces(elbow, q, level), rtol=1e-12
        )
    assert np.abs(_forces(elbow, q, 0.0)).max() > 1.0


def test_automatic_peak_force_and_spec_path() -> None:
    _require_library()
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
    data.act[:] = active > 0
    mujoco.mj_forward(elbow, data)
    mujoco_torque = -(data.qfrc_actuator + data.qfrc_passive)[0]
    np.testing.assert_allclose(
        active[active > 0].sum() + passive.sum() + joints, mujoco_torque, rtol=1e-10
    )
    assert {elbow.actuator(i).name for i in np.flatnonzero(active > 0)} == set(
        _EXTENSORS
    )


def test_coupled_trial_torque_is_the_virtual_work_of_the_muscles() -> None:
    _require_library()
    env = make_env("myoArmReachRandom-v0")
    try:
        model = env.unwrapped.model
        couplings = JointCouplings.from_model(model)
        qpos = model.qpos0.copy()
        adr = model.joint("shoulder_elv_r").qposadr[0]
        qpos[adr] = 0.8
        active, _, _ = max_torque_terms(
            model, MaxTorqueMeasurement("shoulder_elv_r", qpos, torque=40.0)
        )
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
    finally:
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
    assert isinstance(cal, StrengthCalibration)
    np.testing.assert_allclose(cal.predicted, cal.measured, rtol=5e-3)
    assert np.abs(cal.nominal / cal.measured - 1).max() > 0.15
    for group, value in ((_FLEXORS, 1.3), (_EXTENSORS, 0.8)):
        ids = [elbow.actuator(n).id for n in group]
        assert abs(cal.factors[ids].mean() - value) < 0.05


def test_muscles_no_trial_involves_keep_the_prior() -> None:
    _require_library()
    env = make_env("myoArmReachRandom-v0")
    try:
        model = env.unwrapped.model
        qpos = model.qpos0.copy()
        qpos[model.joint("elbow_flexion_r").qposadr[0]] = 1.57
        prior = strength_factor_from_body_mass(60.0, 75.0)
        cal = calibrate_muscle_strength(
            model, [MaxTorqueMeasurement("elbow_flexion_r", qpos, 45.0)], prior=prior
        )
        assert prior == pytest.approx(0.8)
        involved = {model.actuator(i).name for i in np.flatnonzero(cal.involved)}
        assert {"BIClong_r", "BRA_r"} <= involved or {"BIClong", "BRA"} <= involved
        assert not ({"DELT1_r", "DELT1"} & involved)
        np.testing.assert_allclose(cal.factors[~cal.involved], prior)
        np.testing.assert_allclose(cal.predicted, [45.0], rtol=0.05)
    finally:
        env.close()
