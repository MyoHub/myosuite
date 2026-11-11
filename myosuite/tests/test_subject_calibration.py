# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Subject calibration, fatigued MVC torque and external wrench regression tests."""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest

from myosuite.biomechanics.calibration import (
    SegmentLengthMeasurement,
    check_segment_lengths,
    endurance_time,
    fit_fatigue_parameters,
    fit_strength_from_mvcs,
    max_joint_torque,
    posture_at,
    scale_body_mass,
    scale_strength,
    strength_from_mvc,
)
from myosuite.biomechanics.external_loads import (
    force_at_body,
    force_at_site,
    held_mass_qfrc,
)

pytestmark = pytest.mark.tier1


def _arm(*, auto: bool = False, coupled: bool = False) -> mujoco.MjModel:
    child = (
        '<body><joint name="dependent" axis="0 1 0" range="0 6"/>'
        '<geom size=".05" mass="1"/></body>'
        if coupled
        else ""
    )
    equality = (
        '<equality><joint joint1="dependent" joint2="elbow" polycoef="0 2 0 0 0"/>'
        "</equality>"
        if coupled
        else ""
    )
    force = -1 if auto else 100
    return mujoco.MjModel.from_xml_string(
        f"""<mujoco><compiler angle="radian"/><worldbody>
        <body name="torso"><geom size=".1" mass="20"/>
        <body name="arm"><joint name="elbow" axis="0 1 0" range="0 3"
        stiffness="2" springref="0"/>
        <geom type="capsule" fromto="0 0 0 1 0 0" size=".04" mass="3"/>
        <site name="elbow_landmark"/><site name="hand" pos="1 0 0"/>{child}</body></body></worldbody>
        {equality}<actuator>
        <muscle name="flexor" joint="elbow" gear="-1" lengthrange="-3 0" force="{force}"/>
        <muscle name="extensor" joint="elbow" lengthrange="0 3" force="80"/>
        </actuator></mujoco>"""
    )


def test_selected_mvc_retains_unscaled_passive_and_joint_torque() -> None:
    model = _arm()
    q = posture_at(model, {"elbow": 2.7})
    original, _ = max_joint_torque(model, q, "elbow")
    target = original * 0.5
    cal = strength_from_mvc(model, q, "elbow", target)
    assert cal.scale != pytest.approx(0.5)
    assert cal.agonists == ("flexor",)
    scaled = scale_strength(model, cal.scale, cal.agonists)
    assert scaled.actuator_gainprm[1, 2] == model.actuator_gainprm[1, 2]
    achieved, _ = max_joint_torque(scaled, q, "elbow")
    assert achieved == pytest.approx(target, abs=1e-8)
    json.dumps(cal.record(), allow_nan=False)
    ext = strength_from_mvc(model, q, "elbow", -60.0)
    assert max_joint_torque(
        scale_strength(model, ext.scale, ext.agonists), q, "elbow", -1
    )[0] == pytest.approx(-60.0)


def test_multi_posture_strength_fit_exposes_mismatch() -> None:
    model = _arm()
    postures = [posture_at(model, {"elbow": q}) for q in (0.3, 1.5, 2.7)]
    actual = scale_strength(model, 1.4, ["flexor"])
    measured = [max_joint_torque(actual, q, "elbow")[0] for q in postures]
    fit = fit_strength_from_mvcs(model, postures, "elbow", measured)
    assert fit.scale == pytest.approx(1.4)
    assert fit.rmse < 1e-8
    measured[0] += 10
    assert fit_strength_from_mvcs(model, postures, "elbow", measured).rmse > 1
    json.dumps(fit.record(), allow_nan=False)


def test_fatigued_mvc_caps_active_not_passive_forces() -> None:
    model = _arm()
    q = posture_at(model, {"elbow": 2.7})
    fresh = max_joint_torque(model, q, "elbow")[0]
    passive = max_joint_torque(model, q, "elbow", max_activation=np.zeros(2))[0]
    half = max_joint_torque(model, q, "elbow", max_activation=np.array([0.5, 1]))[0]
    assert half == pytest.approx(passive + 0.5 * (fresh - passive))
    assert half != pytest.approx(0.5 * fresh)
    with pytest.raises(ValueError, match="max_activation"):
        max_joint_torque(model, q, "elbow", max_activation=np.array([1.1, 0]))


def test_partial_mass_scaling_refreshes_constants_and_preserves_auto_strength() -> None:
    model = _arm(auto=True)
    original_mass = model.body_mass.copy()
    original_subtree = model.body_subtreemass.copy()
    q = posture_at(model, {"elbow": 1.5})
    torque = max_joint_torque(model, q, "elbow")[0]
    scaled = scale_body_mass(
        model, participant_mass=90, reference_mass=60, bodies=["arm"]
    )
    assert scaled.body_mass[model.body("arm").id] == pytest.approx(4.5)
    assert scaled.body_mass[model.body("torso").id] == 20
    assert scaled.body_subtreemass[model.body("torso").id] == pytest.approx(24.5)
    assert max_joint_torque(scaled, q, "elbow")[0] == pytest.approx(torque)
    assert scaled.actuator_gainprm[0, 2] > 0
    np.testing.assert_array_equal(model.body_mass, original_mass)
    np.testing.assert_array_equal(model.body_subtreemass, original_subtree)
    represented = scale_body_mass(model, represented_mass=6, bodies=["arm"])
    assert represented.body_mass[model.body("arm").id] == pytest.approx(6)
    assert max_joint_torque(scale_strength(model, 2, ["flexor"]), q, "elbow")[
        0
    ] == pytest.approx(2 * (torque + 3) - 3)
    with pytest.raises(ValueError, match="reference_mass"):
        scale_body_mass(model, participant_mass=90)


def test_joint_coupled_torque_uses_virtual_work() -> None:
    model = _arm(coupled=True)
    q = posture_at(model, {"elbow": 1.0})
    assert q[1] == pytest.approx(2)
    model.jnt_stiffness[model.joint("dependent").id] = 3
    # Child spring torque -3*2 is transferred with dq_child/dq_elbow=2.
    baseline = max_joint_torque(model, q, "elbow")[0]
    model.jnt_stiffness[model.joint("dependent").id] = 0
    assert max_joint_torque(model, q, "elbow")[0] - baseline == pytest.approx(12)
    with pytest.raises(ValueError, match="independent"):
        max_joint_torque(model, q, "dependent")


def test_wrench_mapping_matches_lever_arm_and_rotation() -> None:
    model = _arm()
    for theta in (0.0, 0.7, 1.5):
        q = posture_at(model, {"elbow": theta})
        expected = 2 * 9.81 * np.cos(theta)
        assert held_mass_qfrc(model, q, 2, site="hand")[0] == pytest.approx(expected)
        from_site = force_at_site(model, q, "hand", np.array([0, 0, -10.0]))
        from_body = force_at_body(
            model,
            q,
            "arm",
            np.array([0, 0, -10.0]),
            point=np.array([np.cos(theta), 0, -np.sin(theta)]),
        )
        np.testing.assert_allclose(from_site, from_body, atol=1e-10)
        assert force_at_site(model, q, "hand", np.zeros(3), torque=np.array([0, 5, 0]))[
            0
        ] == pytest.approx(5)
        assert force_at_site(model, q, "hand", np.array([0, 0, -10]), frame="site")[
            0
        ] == pytest.approx(10)
        np.testing.assert_allclose(
            force_at_body(
                model,
                q,
                "arm",
                np.array([0, 0, -10]),
                point=np.array([1, 0, 0]),
                frame="body",
            ),
            force_at_site(model, q, "hand", np.array([0, 0, -10]), frame="site"),
        )
    with pytest.raises(ValueError, match="choose body"):
        held_mass_qfrc(model, np.zeros(1), 2, body="arm", site="hand")


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_calibration_inputs_fail_early(bad: float) -> None:
    with pytest.raises(ValueError):
        scale_strength(_arm(), bad)
    with pytest.raises(ValueError):
        scale_body_mass(_arm(), represented_mass=bad)
    with pytest.raises(ValueError):
        endurance_time([0.3], 0.01, 0.002, 5, dt=bad)


def test_fatigue_fit_rejects_insufficient_recovery_measurements() -> None:
    with pytest.raises(ValueError, match="three distinct"):
        fit_fatigue_parameters([0.3, 0.5], [200, 70], fit_recovery=True)


def test_geometry_verification_reports_segment_mismatch() -> None:
    checks = check_segment_lengths(
        _arm(),
        np.array([0.7]),
        [
            SegmentLengthMeasurement("elbow_landmark", "hand", 1.01),
            SegmentLengthMeasurement("elbow_landmark", "hand", 0.9),
        ],
    )
    assert checks[0].model_length == pytest.approx(1)
    assert checks[0].within_tolerance and not checks[1].within_tolerance
    assert checks[1].error == pytest.approx(0.1)
    json.dumps([check.record() for check in checks], allow_nan=False)
