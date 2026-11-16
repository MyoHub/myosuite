# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Hand-handle grip: the grip posture and the grip contacts in static optimization."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite.biomechanics.emg_validation import EMGTrial
from myosuite.biomechanics.grip import GripHandle, solve_handle_grip
from myosuite.biomechanics.static_optimization import (
    GripContactConfig,
    StaticOptimizationConfig,
    StaticOptimizer,
)
from scripts.validation.validate_emg import trial_optimization_config

pytestmark = pytest.mark.tier1

# A palm on a wrist hinge and one finger on a knuckle hinge (flexion is +y),
# in the frame of a fixed forearm. FDS crosses both joints, FCR the wrist, EXT
# extends the knuckle. Flexed 90 deg, the finger wraps a handle under the palm.
_HAND_XML = """
<mujoco><compiler angle="radian"/><option gravity="0 0 0"/><worldbody>
  <body name="palm"><joint name="wrist" axis="0 1 0"/>
    <geom type="capsule" fromto="0 0 0 .08 0 0" size=".01" mass=".3"/>
    <body name="finger" pos=".08 0 0">
      <joint name="knuckle_flexion" axis="0 1 0" range="0 1.7"/>
      <geom type="capsule" fromto="0 0 0 .04 0 0" size=".008" mass=".05"/>
    </body></body></worldbody>
  <tendon><fixed name="fds"><joint joint="wrist" coef=".02"/>
    <joint joint="knuckle_flexion" coef=".01"/></fixed></tendon>
  <actuator>
    <motor name="FDS" tendon="fds" gear="300" ctrlrange="0 1"/>
    <motor name="FCR" joint="wrist" gear="6" ctrlrange="0 1"/>
    <motor name="EXT" joint="knuckle_flexion" gear="-3" ctrlrange="0 1"/>
  </actuator></mujoco>"""
_HANDLE = GripHandle(radius=0.015, half_length=0.03)
# The handle touches the palm's underside and the flexed finger's inner side.
_GRIP = GripContactConfig(
    handle_body=_HANDLE.name,
    hand_body="palm",
    handle_pos=(0.057, 0.0, -0.025),
    handle_quat=(np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0),
    loaded_joint="wrist",
    optimal_force=1e4,  # gravity is off: no body weight to scale it
)
_QPOS = np.array([0.0, 0.5 * np.pi])
# Isometric wrist flexion: the dynamometer resists 3 N m.
_LOAD = np.array([-3.0, 0.0])


def _models() -> tuple[mujoco.MjModel, mujoco.MjModel]:
    """The hand without and with the handle."""
    spec = mujoco.MjSpec.from_string(_HAND_XML)
    _HANDLE.add_to(spec)
    return mujoco.MjModel.from_xml_string(_HAND_XML), spec.compile()


def test_default_path_ignores_the_handle() -> None:
    """Without ``grip`` a model with the handle solves bit for bit as one without."""
    plain, handled = _models()
    zero = np.zeros(2)
    for cfg in (None, StaticOptimizationConfig(supported_dofs=(0,))):
        a = StaticOptimizer(plain, cfg).solve(_QPOS, zero, zero, _LOAD)
        b = StaticOptimizer(handled, cfg).solve(_QPOS, zero, zero, _LOAD)
        for name in ("activation", "reserve", "constraint_force", "joint_torque"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        assert not b.contact_force.any()


def test_grip_contacts_take_the_finger_flexor_torque() -> None:
    """With the grip, FDS helps the wrist and the handle, not EXT, holds its finger torque."""
    _, model = _models()
    zero = np.zeros(2)
    free = StaticOptimizer(model).solve(_QPOS, zero, zero, _LOAD)
    # Minimum effort without the handle: FDS = EXT = 1/6, FCR = 1/3.
    np.testing.assert_allclose(free.activation, [1 / 6, 1 / 3, 1 / 6], atol=1e-3)
    assert not free.contact_force.any()

    cfg = StaticOptimizationConfig(tolerance=1e-6, grip=_GRIP)
    held = StaticOptimizer(model, cfg).solve(_QPOS, zero, zero, _LOAD)
    assert held.status == "solved"
    np.testing.assert_allclose(held.activation, [0.25, 0.25, 0.0], atol=2e-3)
    # The grip force appears: the handle pushes on the palm and the finger.
    palm, finger = model.body("palm").id, model.body("finger").id
    assert np.linalg.norm(held.contact_force[palm]) > 10.0
    assert np.linalg.norm(held.contact_force[finger]) > 10.0
    # It transmits the dynamometer's 3 N m and takes FDS's 0.75 N m at the knuckle.
    np.testing.assert_allclose(held.contact_qfrc, [-3.0, -0.75], atol=2e-3)
    np.testing.assert_allclose(held.reserve, 0.0, atol=2e-3)


def test_grip_posture_wraps_the_handle() -> None:
    """The solver closes the finger on the handle under the palm, without overlap."""
    _, model = _models()
    grip = solve_handle_grip(
        model,
        np.zeros(2),
        hand_body="palm",
        joints=["knuckle_flexion"],
        contact_bodies=["palm", "finger"],
        axis=np.array([0.0, 1.0, 0.0]),
        handle_body=_HANDLE.name,
        starts=8,
    )
    assert max(abs(gap) for _, gap in grip.gaps) < 5e-4
    assert grip.max_penetration < 5e-4
    knuckle = dict(grip.joint_qpos)["knuckle_flexion"]
    assert 0.5 < knuckle < 1.7
    qpos = grip.apply(model, np.zeros((4, 2)))
    np.testing.assert_allclose(qpos[:, 1], knuckle)


def test_trials_opt_in_to_the_grip() -> None:
    """Only a trial's ``meta["grip"]`` adds the grip; other trials keep the defaults."""
    kw = {
        "trial_id": "t",
        "time": np.arange(5) / 100.0,
        "signals": np.zeros((5, 1)),
        "channel_names": ("FDS",),
    }
    plain = trial_optimization_config(EMGTrial(**kw))
    assert plain == StaticOptimizationConfig()
    meta = {
        "grip": {
            "handle_body": "grip_handle",
            "hand_body": "palm",
            "handle_pos": [0.0, 0.0, 0.0],
            "handle_quat": [1.0, 0.0, 0.0, 0.0],
            "loaded_joint": "wrist",
            "contact_distance": 0.005,
            "optimal_force": None,
        }
    }
    gripped = trial_optimization_config(EMGTrial(**kw, meta=meta))
    assert gripped.grip == GripContactConfig(
        "grip_handle", "palm", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), "wrist"
    )
