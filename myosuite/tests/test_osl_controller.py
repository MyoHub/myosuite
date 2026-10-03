# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The OSL impedance controller's state machine flag and hardware parameters."""

from __future__ import annotations

import pytest

from myosuite.envs.myo.assets.leg.myoosl_control import MyoOSLController

pytestmark = pytest.mark.tier1


def _sensors(knee_angle: float) -> dict[str, float]:
    return {
        "knee_angle": knee_angle,
        "knee_vel": 0.0,
        "load": 0.0,
        "ankle_angle": 0.0,
        "ankle_vel": 0.0,
    }


def test_state_machine_is_running_follows_start_and_stop() -> None:
    """is_running reports the running flag (it used to recurse forever)."""
    controller = MyoOSLController(body_mass=70.0)
    machine = controller.STATE_MACHINE
    assert machine.is_running is False
    controller.start()
    assert machine.is_running is True
    machine.stop()
    assert machine.is_running is False


def test_set_motor_param_stores_the_value_used_by_the_torque() -> None:
    """set_motor_param stores the value (it stored the parameter's name)."""
    controller = MyoOSLController(body_mass=70.0)
    controller.start()
    controller.set_motor_param("knee", "peak_torque", 5.0)
    assert controller.HARDWARE["knee"]["peak_torque"] == 5.0

    # A large knee angle error saturates the commanded torque at the new peak.
    controller.update(_sensors(knee_angle=-3.0))
    assert abs(controller.get_osl_torque()["knee"]) == pytest.approx(5.0)
