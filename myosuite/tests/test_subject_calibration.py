# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Participant model preparation and steady MVC regression tests."""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest

from myosuite.biomechanics.calibration import (
    SegmentLengthMeasurement,
    check_segment_lengths,
    scale_body_mass,
    fit_strength_from_mvcs,
    max_joint_torque,
    posture_at,
    scale_strength,
    strength_from_mvc,
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


def test_early_activation_probes_are_steady_and_do_not_modify_the_model() -> None:
    model = _arm()
    q = posture_at(model, {"elbow": 1.0})
    nominal = max_joint_torque(model, q, "elbow")[0]
    model.actuator_actearly[:] = 1
    assert max_joint_torque(model, q, "elbow")[0] == pytest.approx(nominal)
    assert np.all(model.actuator_actearly == 1)
    cal = strength_from_mvc(model, q, "elbow", 40.0)
    scaled = scale_strength(model, cal.scale, cal.agonists)
    data = mujoco.MjData(scaled)
    data.qpos[:] = q
    flexor = scaled.actuator("flexor").id
    data.act[scaled.actuator_actadr[flexor]] = 1
    data.ctrl[flexor] = 1
    mujoco.mj_forward(scaled, data)
    assert (data.qfrc_actuator + data.qfrc_passive)[0] == pytest.approx(40.0)
    data.act[scaled.actuator_actadr[flexor]] = 0.5
    data.ctrl[flexor] = 0.5
    mujoco.mj_forward(scaled, data)
    cap = np.array([0.5, 1.0])
    assert max_joint_torque(scaled, q, "elbow", max_activation=cap)[0] == pytest.approx(
        (data.qfrc_actuator + data.qfrc_passive)[0]
    )


@pytest.mark.parametrize("clamp", ["actuator", "joint", "multi_state"])
def test_mvc_apis_reject_nonaffine_or_multistate_actuators(clamp: str) -> None:
    model = _arm()
    if clamp == "actuator":
        model.actuator_forcelimited[0] = True
        model.actuator_forcerange[0] = [-50.0, 0.0]
    elif clamp == "joint":
        model.jnt_actfrclimited[0] = True
        model.jnt_actfrcrange[0] = [0.0, 20.0]
    else:
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><joint name="elbow"/>'
            '<geom size=".1" mass="1"/></body></worldbody>'
            '<actuator><general joint="elbow" dyntype="user" actdim="2"/>'
            "</actuator></mujoco>"
        )
    q = posture_at(model, {"elbow": 1.0})
    with pytest.raises(ValueError, match="Force clamps and multi-state"):
        max_joint_torque(model, q, "elbow")
    with pytest.raises(ValueError, match="Force clamps and multi-state"):
        strength_from_mvc(model, q, "elbow", 40.0)
    with pytest.raises(ValueError, match="Force clamps and multi-state"):
        fit_strength_from_mvcs(model, [q, q], "elbow", [40.0, 40.0])


@pytest.mark.parametrize("route_to_actuation", [False, True])
def test_gravity_corrected_mvc_rejects_compensation_in_either_routing(
    route_to_actuation: bool,
) -> None:
    model = _arm()
    model.body_gravcomp[model.body("arm").id] = 1
    model.jnt_actgravcomp[:] = route_to_actuation
    q = posture_at(model, {"elbow": 1.0})
    with pytest.raises(ValueError, match="gravity compensation"):
        max_joint_torque(model, q, "elbow")
    with pytest.raises(ValueError, match="gravity compensation"):
        strength_from_mvc(model, q, "elbow", 40.0)
    with pytest.raises(ValueError, match="gravity compensation"):
        fit_strength_from_mvcs(model, [q, q], "elbow", [40.0, 40.0])


def _stateless_muscle() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><worldbody><body>'
        '<joint name="elbow" range="0 3"/><geom size=".1" mass="1"/>'
        '</body></worldbody><actuator><general name="flexor" joint="elbow" '
        'gear="-1" dyntype="none" gaintype="muscle" biastype="muscle" '
        'gainprm=".75 1.05 100 200 .5 1.6 1.5 1.3 1.2" '
        'biasprm=".75 1.05 100 200 .5 1.6 1.5 1.3 1.2" '
        'lengthrange="-3 0" ctrllimited="true" ctrlrange="0 1"/>'
        "</actuator></mujoco>"
    )


@pytest.mark.parametrize("lower,upper", [(0.0, 0.5), (0.2, 1.0)])
def test_mvc_rejects_restricted_stateless_controls(lower: float, upper: float) -> None:
    model = _stateless_muscle()
    model.actuator_ctrlrange[0] = [lower, upper]
    q = posture_at(model, {"elbow": 1.0})
    data = mujoco.MjData(model)
    data.qpos[:] = q
    data.ctrl[:] = 0.5
    mujoco.mj_forward(model, data)
    assert data.qfrc_actuator[0] == pytest.approx(49.0)
    with pytest.raises(ValueError, match="stateless muscle control clamps"):
        max_joint_torque(model, q, "elbow", max_activation=np.array([0.5]))
    with pytest.raises(ValueError, match="stateless muscle control clamps"):
        strength_from_mvc(model, q, "elbow", 40.0)
    with pytest.raises(ValueError, match="stateless muscle control clamps"):
        fit_strength_from_mvcs(model, [q, q], "elbow", [40.0, 40.0])


@pytest.mark.parametrize(
    "lower,upper,disable_clamp",
    [(0.0, 1.0, False), (-1.0, 2.0, False), (0.0, 0.5, True)],
)
def test_supported_stateless_controls_match_direct_actuation(
    lower: float, upper: float, disable_clamp: bool
) -> None:
    model = _stateless_muscle()
    model.actuator_ctrlrange[0] = [lower, upper]
    if disable_clamp:
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)
    q = posture_at(model, {"elbow": 1.0})
    for activation in (0.5, 1.0):
        data = mujoco.MjData(model)
        data.qpos[:] = q
        data.ctrl[:] = activation
        mujoco.mj_forward(model, data)
        prediction = max_joint_torque(
            model, q, "elbow", max_activation=np.array([activation])
        )[0]
        assert prediction == pytest.approx(data.qfrc_actuator[0])


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_strength_scale_fails_early(bad: float) -> None:
    with pytest.raises(ValueError):
        scale_strength(_arm(), bad)


def test_selected_scaling_preserves_other_parameters_and_nonmuscle_bias() -> None:
    model = _arm(auto=True)
    original_gain, original_bias = (
        model.actuator_gainprm.copy(),
        model.actuator_biasprm.copy(),
    )
    scaled = scale_strength(model, 2.0, ["extensor"])
    np.testing.assert_array_equal(scaled.actuator_gainprm[0], original_gain[0])
    np.testing.assert_array_equal(scaled.actuator_biasprm[0], original_bias[0])
    assert scaled.actuator_gainprm[1, 2] == pytest.approx(160)
    np.testing.assert_array_equal(model.actuator_gainprm, original_gain)
    np.testing.assert_array_equal(model.actuator_biasprm, original_bias)
    model.actuator_biastype[0] = mujoco.mjtBias.mjBIAS_NONE
    scaled = scale_strength(model, 2.0, ["flexor"])
    np.testing.assert_array_equal(scaled.actuator_biasprm[0], original_bias[0])


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


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_represented_mass_fails_early(bad: float) -> None:
    with pytest.raises(ValueError):
        scale_body_mass(_arm(), represented_mass=bad)
