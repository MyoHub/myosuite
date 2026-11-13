# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Model transforms retain their native contracts and lazy dependency boundary."""

from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from myosuite.biomechanics.calibration import scale_body_mass
from myosuite.core.muscle_strength import (
    scale_muscle_strength,
    scale_muscle_strength_spec,
)

pytestmark = pytest.mark.tier1

_XML = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="torso"><geom size=".1" mass="20"/><body name="arm">
    <joint name="elbow" range="0 3"/><geom size=".1" mass="3"/>
  </body></body></worldbody><actuator>
  <muscle name="m" joint="elbow" lengthrange="0 3" force="-1"/>
  </actuator></mujoco>"""


def _force(model: mujoco.MjModel, level: float) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[:] = 2.8
    data.act[:] = level
    mujoco.mj_forward(model, data)
    return data.actuator_force.copy()


def test_missing_development_apis_are_lazy_and_leave_legacy_sarcopenia_available() -> (
    None
):
    code = f"""
import importlib.abc
import sys
class MissingApis(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('myo_sim.physics.muscle_strength', 'myo_sim.physics.body_mass'):
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, MissingApis())
import mujoco
import numpy as np
from myosuite.core.muscle_conditions import _peak_force, apply_sarcopenia_to_model
from myosuite.core.muscle_strength import scale_muscle_strength, scale_muscle_strength_spec
from myosuite.biomechanics.calibration import scale_body_mass, max_joint_torque
assert 'myo_sim.physics.muscle_strength' not in sys.modules
assert 'myo_sim.physics.body_mass' not in sys.modules
spec = mujoco.MjSpec.from_string({_XML!r})
model = spec.copy().compile()
gain, bias = _peak_force(model), model.actuator_biasprm.copy()
apply_sarcopenia_to_model(model, 0.5)
np.testing.assert_allclose(model.actuator_gainprm[:, 2], gain * 0.5)
np.testing.assert_array_equal(model.actuator_biasprm, bias)
assert np.isfinite(max_joint_torque(model, np.array([2.8]), 'elbow')[0])
calls = (
    (lambda: scale_muscle_strength(model, 1.0), 'myo_sim.physics.muscle_strength'),
    (lambda: scale_muscle_strength_spec(spec, 1.0), 'myo_sim.physics.muscle_strength'),
    (lambda: scale_body_mass(model, represented_mass=3, bodies=['arm']), 'myo_sim.physics.body_mass'),
)
for call, feature in calls:
    try:
        call()
    except ImportError as exc:
        assert feature in str(exc) and 'Install' in str(exc)
    else:
        raise AssertionError('missing model API was not reported')
class BrokenDependency(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('myo_sim.physics.muscle_strength', 'myo_sim.physics.body_mass'):
            raise ModuleNotFoundError('unrelated dependency defect', name='broken_dependency')
sys.meta_path[0] = BrokenDependency()
for call, feature in calls:
    try:
        call()
    except ModuleNotFoundError as exc:
        assert exc.name == 'broken_dependency'
    else:
        raise AssertionError('unrelated import defect was hidden')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("spec_path", [False, True])
def test_strength_facade_matches_native_force_and_ownership(spec_path: bool) -> None:
    spec = mujoco.MjSpec.from_string(_XML)
    original = spec.copy().compile()
    if importlib.util.find_spec("myo_sim.physics.muscle_strength") is None:
        with pytest.raises(ImportError, match="myo_sim.physics.muscle_strength"):
            scale_muscle_strength_spec(
                spec, 1.7
            ) if spec_path else scale_muscle_strength(original, 1.7)
        return
    if spec_path:
        assert scale_muscle_strength_spec(spec, 1.7) is spec
        scaled = spec.compile()
    else:
        scaled = copy.deepcopy(original)
        assert scale_muscle_strength(scaled, 1.7) is None
    for level in (0.0, 1.0):
        np.testing.assert_allclose(
            _force(scaled, level), 1.7 * _force(original, level), rtol=1e-12
        )


def test_mass_facade_keeps_both_modes_and_preserves_native_force() -> None:
    original = mujoco.MjModel.from_xml_string(_XML)
    if importlib.util.find_spec("myo_sim.physics.body_mass") is None:
        with pytest.raises(ImportError, match="myo_sim.physics.body_mass"):
            scale_body_mass(original, represented_mass=6, bodies=["arm"])
        return
    represented = scale_body_mass(original, represented_mass=6, bodies=["arm"])
    ratio = scale_body_mass(
        original, participant_mass=90, reference_mass=60, bodies=["arm"]
    )
    assert represented is not original and ratio is not original
    assert represented.body("arm").mass[0] == pytest.approx(6)
    assert ratio.body("arm").mass[0] == pytest.approx(4.5)
    assert original.body("arm").mass[0] == pytest.approx(3)
    assert represented.body("torso").mass[0] == pytest.approx(20)
    for level in (0.0, 1.0):
        np.testing.assert_allclose(
            _force(represented, level), _force(original, level), rtol=1e-12
        )
