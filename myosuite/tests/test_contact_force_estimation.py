# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Ground reactions estimated by static optimization, against statics and friction limits."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings
from myosuite.physics.static_optimization import (
    StaticOptimizationConfig,
    StaticOptimizer,
)

pytestmark = pytest.mark.tier1

_CONTACTS = StaticOptimizationConfig(contact_forces=True)

# A 2 kg box on the floor, and a 1 kg box free above it.
_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 .1"/>
    <body name="box" pos="0 0 {z}">
      <freejoint/>
      <geom type="box" size=".1 .1 .1" mass="2"/>
    </body>
    <body name="top" pos="0 0 {top}">
      <freejoint/>
      <geom type="box" size=".05 .05 .05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _boxes(z: float = 0.1) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_XML.format(z=z, top=z + 0.15))


def _solve(model: mujoco.MjModel, qacc_box: np.ndarray, config=_CONTACTS):
    qacc = np.zeros(model.nv)
    qacc[:3] = qacc_box
    qacc[6:9] = qacc_box  # the top box moves with the lower one
    return StaticOptimizer(model, config).solve(model.qpos0, np.zeros(model.nv), qacc)


def test_resting_box_is_carried_by_the_floor() -> None:
    model = _boxes()
    frame = _solve(model, np.zeros(3))
    box, top = model.body("box").id, model.body("top").id
    np.testing.assert_allclose(frame.contact_force[box], [0, 0, 2 * 9.81], atol=0.05)
    np.testing.assert_allclose(
        frame.contact_force[0], -frame.contact_force[box], atol=1e-9
    )
    assert np.abs(frame.reserve[:6]).max() < 0.05
    # The top box touches only the lower box, not a world-fixed body: its residual holds it.
    np.testing.assert_allclose(frame.contact_force[top], 0.0)
    np.testing.assert_allclose(frame.reserve[6:9], [0, 0, 9.81], atol=1e-6)
    # Without contact estimation the residual carries everything.
    off = _solve(model, np.zeros(3), StaticOptimizationConfig())
    np.testing.assert_allclose(off.contact_force, 0.0)
    np.testing.assert_allclose(off.reserve[:3], [0, 0, 2 * 9.81], atol=1e-6)


@pytest.mark.parametrize(("gap", "carried"), [(0.005, True), (0.02, False)])
def test_contact_reach(gap: float, carried: bool) -> None:
    frame = _solve(_boxes(0.1 + gap), np.zeros(3))
    support = frame.contact_force[1, 2]
    if carried:  # within contact_distance (1 cm): mostly the contact
        assert 0.99 * 2 * 9.81 < support < 2 * 9.81 + 1e-6
    else:
        assert support == 0.0
        np.testing.assert_allclose(frame.reserve[2], 2 * 9.81, atol=1e-6)


def test_contact_capacity_fades_with_the_gap() -> None:
    # Residuals as cheap as contacts, so the split follows each contact's capacity.
    cheap = StaticOptimizationConfig(
        contact_forces=True, residual_optimal_force=3 * 9.81
    )
    share = [
        _solve(_boxes(0.1 + gap), np.zeros(3), cheap).contact_force[1, 2]
        for gap in (0.0, 0.004, 0.008)
    ]
    assert share[0] > share[1] > share[2] > 0.0


def test_friction_and_pulling_limits() -> None:
    model = _boxes()
    # Half a g sideways: friction (mu = 1) accelerates the box.
    side = _solve(model, np.array([0.5 * 9.81, 0, 0]))
    np.testing.assert_allclose(
        side.contact_force[1], [2 * 0.5 * 9.81, 0, 2 * 9.81], atol=0.05
    )
    assert np.abs(side.reserve[:3]).max() < 0.05
    # Pulled down at 2 g: contacts cannot pull, the residual does it.
    down = _solve(model, np.array([0, 0, -2 * 9.81]))
    np.testing.assert_allclose(down.contact_force[1], 0.0, atol=1e-6)
    np.testing.assert_allclose(down.reserve[2], -2 * 9.81, atol=1e-5)


def test_standing_leg_is_carried_by_both_feet() -> None:
    env = make_env("myoLegWalk-v0")
    model = env.unwrapped.model
    qpos = model.key_qpos[0].copy()
    project_joint_couplings(JointCouplings.from_model(model), qpos)
    zero = np.zeros(model.nv)
    frame = StaticOptimizer(model, _CONTACTS).solve(qpos, zero, zero)
    weight = model.body_mass[1:].sum() * 9.81
    assert frame.status == "solved"
    np.testing.assert_allclose(frame.contact_force[1:, 2].sum(), weight, rtol=5e-3)
    for side in ("r", "l"):
        foot = sum(
            frame.contact_force[model.body(f"{b}_{side}").id, 2]
            for b in ("calcn", "toes")
        )
        assert 0.4 * weight < foot < 0.6 * weight
    assert np.abs(frame.reserve[:6]).max() < 0.01 * weight
    assert frame.activation.max() < 0.3
    # MuJoCo's own actuator force closes the balance with reserves and constraint forces.
    data = mujoco.MjData(model)
    data.qpos[:], data.act[:] = qpos, frame.activation
    mujoco.mj_fwdPosition(model, data)
    mujoco.mj_fwdVelocity(model, data)
    mujoco.mj_fwdActuation(model, data)
    np.testing.assert_allclose(
        data.qfrc_actuator + frame.reserve + frame.constraint_force,
        frame.joint_torque,
        atol=1e-3,
    )
    env.close()
