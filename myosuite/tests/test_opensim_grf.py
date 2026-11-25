# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for ``myosuite.biomechanics.opensim_grf``."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite.biomechanics.opensim_grf import (
    normalize_opensim_motion,
    opensim_grf_to_qfrc,
)
from myosuite.biomechanics.opensim_motion import GAIT2392_SIGNS, OpenSimMotion

pytestmark = pytest.mark.tier1

_FLOATING_FOOT_XML = """
<mujoco model="floating_foot">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="pelvis" pos="0 0 0.9">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="10.0" diaginertia="0.2 0.2 0.2"/>
      <body name="calcn_r" pos="0.05 -0.1 -0.85">
        <joint name="ankle_angle_r" type="hinge" axis="0 0 1" range="-0.8 0.8"/>
        <inertial pos="0.08 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_normalize_opensim_motion_and_grf_wrench_equivalence() -> None:
    mot_raj = OpenSimMotion(
        time=np.array([0.0, 0.01, 0.02, 0.03]),
        labels=(
            "/jointset/ground_pelvis/pelvis_tx/value",
            "/jointset/ground_pelvis/pelvis_tx/speed",
            "/jointset/walker_knee_r/knee_angle_r/value",
        ),
        values=np.array(
            [
                [0.1, 1.2, 0.3],
                [0.11, 1.2, 0.35],
                [0.12, 1.2, 0.4],
                [0.13, 1.2, 0.45],
            ]
        ),
        in_degrees=False,
    )
    clean_raj, signs_raj = normalize_opensim_motion(mot_raj)
    assert clean_raj.labels == ("pelvis_tx", "knee_angle_r")
    assert signs_raj == {}

    mot_gait = OpenSimMotion(
        time=mot_raj.time,
        labels=("pelvis_tx", "knee_angle_r"),
        values=mot_raj.values[:, [0, 2]],
        in_degrees=True,
    )
    _, signs_gait = normalize_opensim_motion(mot_gait)
    assert signs_gait == GAIT2392_SIGNS

    model = mujoco.MjModel.from_xml_string(_FLOATING_FOOT_XML)
    time = np.linspace(0.0, 0.03, 4)
    qpos = np.tile(model.qpos0, (4, 1))
    qpos[:, 0] = np.linspace(0.0, 0.06, 4)
    qpos[:, 7] = np.linspace(-0.1, 0.2, 4)

    grf = OpenSimMotion(
        time=time,
        labels=(
            "ground_force_r_vx",
            "ground_force_r_vy",
            "ground_force_r_vz",
            "ground_force_r_px",
            "ground_force_r_py",
            "ground_force_r_pz",
            "ground_torque_r_x",
            "ground_torque_r_y",
            "ground_torque_r_z",
        ),
        values=np.array(
            [
                [-40.0, 680.0, 15.0, 0.18, 0.0, -0.05, 0.0, 3.5, 0.0],
                [-20.0, 720.0, 10.0, 0.22, 0.0, -0.04, 0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 331.7, -99.0, 42.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, np.nan, np.inf, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        in_degrees=False,
    )
    loads_xml = """
    <OpenSimDocument Version="40000">
      <ExternalLoads>
        <objects>
          <ExternalForce name="RightGRF">
            <applied_to_body>calcn_r</applied_to_body>
            <force_expressed_in_body>ground</force_expressed_in_body>
            <point_expressed_in_body>ground</point_expressed_in_body>
            <force_identifier>ground_force_r_v</force_identifier>
            <point_identifier>ground_force_r_p</point_identifier>
            <torque_identifier>ground_torque_r_</torque_identifier>
          </ExternalForce>
        </objects>
      </ExternalLoads>
    </OpenSimDocument>
    """
    g_quat = (float(np.sqrt(0.5)), float(np.sqrt(0.5)), 0.0, 0.0)
    qfrc = opensim_grf_to_qfrc(model, qpos, time, grf, loads_xml, ground_quat=g_quat)
    assert np.allclose(qfrc[2:], 0.0, atol=1e-12)

    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(g_quat))
    r_mat = rot.reshape(3, 3)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[0]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    expected = np.zeros(model.nv)
    mujoco.mj_applyFT(
        model,
        data,
        r_mat @ grf.values[0, 0:3],
        r_mat @ grf.values[0, 6:9],
        r_mat @ grf.values[0, 3:6],
        model.body("calcn_r").id,
        expected,
    )
    assert np.allclose(qfrc[0], expected, atol=1e-10)


def test_opensim_grf_rejects_non_ground_frame_and_unmatched_loads() -> None:
    model = mujoco.MjModel.from_xml_string(_FLOATING_FOOT_XML)
    time = np.array([0.0, 0.01])
    qpos = np.tile(model.qpos0, (2, 1))
    grf = OpenSimMotion(
        time=time,
        labels=("f_vx", "f_vy", "f_vz"),
        values=np.ones((2, 3)),
        in_degrees=False,
    )
    bad_frame_xml = """
    <ExternalLoads><objects><ExternalForce name="bad">
      <applied_to_body>calcn_r</applied_to_body>
      <force_expressed_in_body>calcn_r</force_expressed_in_body>
      <point_expressed_in_body>ground</point_expressed_in_body>
      <force_identifier>f_v</force_identifier>
    </ExternalForce></objects></ExternalLoads>
    """
    with pytest.raises(ValueError, match="non-ground frame"):
        opensim_grf_to_qfrc(model, qpos, time, grf, bad_frame_xml)

    unmatched_xml = """
    <ExternalLoads><objects><ExternalForce name="missing">
      <applied_to_body>unknown_body</applied_to_body>
      <force_identifier>f_v</force_identifier>
    </ExternalForce></objects></ExternalLoads>
    """
    with pytest.raises(ValueError, match="no ExternalForce entries matched"):
        opensim_grf_to_qfrc(model, qpos, time, grf, unmatched_xml)


def test_opensim_grf_preserves_pure_couple_torque_when_force_is_zero() -> None:
    """Finite free moments in zero-force frames are applied, while non-finite inactive torques are masked."""
    model = mujoco.MjModel.from_xml_string(_FLOATING_FOOT_XML)
    time = np.array([0.0, 0.01])
    qpos = np.tile(model.qpos0, (2, 1))
    values = np.array(
        [
            # Frame 0: zero force, finite pure couple (0, 3.5, -1.2), NaN COP
            [0.0, 0.0, 0.0, np.nan, np.nan, np.nan, 0.0, 3.5, -1.2],
            # Frame 1: zero force, NaN torque and COP -> masked to zero
            [0.0, 0.0, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ],
        dtype=np.float64,
    )
    grf = OpenSimMotion(
        time=time,
        labels=(
            "f_vx",
            "f_vy",
            "f_vz",
            "f_px",
            "f_py",
            "f_pz",
            "t_x",
            "t_y",
            "t_z",
        ),
        values=values,
        in_degrees=False,
    )
    with_point_xml = """
    <ExternalLoads><objects><ExternalForce name="couple_with_cop">
      <applied_to_body>calcn_r</applied_to_body>
      <force_expressed_in_body>ground</force_expressed_in_body>
      <point_expressed_in_body>ground</point_expressed_in_body>
      <force_identifier>f_v</force_identifier>
      <point_identifier>f_p</point_identifier>
      <torque_identifier>t_</torque_identifier>
    </ExternalForce></objects></ExternalLoads>
    """
    without_point_xml = """
    <ExternalLoads><objects><ExternalForce name="couple_no_cop">
      <applied_to_body>calcn_r</applied_to_body>
      <force_expressed_in_body>ground</force_expressed_in_body>
      <force_identifier>f_v</force_identifier>
      <torque_identifier>t_</torque_identifier>
    </ExternalForce></objects></ExternalLoads>
    """
    g_quat = (1.0, 0.0, 0.0, 0.0)
    qfrc_cop = opensim_grf_to_qfrc(
        model, qpos, time, grf, with_point_xml, ground_quat=g_quat
    )
    qfrc_no_cop = opensim_grf_to_qfrc(
        model, qpos, time, grf, without_point_xml, ground_quat=g_quat
    )

    data = mujoco.MjData(model)
    data.qpos[:] = qpos[0]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    expected0 = np.zeros(model.nv)
    mujoco.mj_applyFT(
        model,
        data,
        np.zeros(3),
        np.array([0.0, 3.5, -1.2]),
        np.zeros(3),
        model.body("calcn_r").id,
        expected0,
    )
    assert np.linalg.norm(expected0) > 0.0
    np.testing.assert_allclose(qfrc_cop[0], expected0, atol=1e-12)
    np.testing.assert_allclose(qfrc_no_cop[0], expected0, atol=1e-12)
    np.testing.assert_allclose(qfrc_cop[1], 0.0, atol=1e-12)
    np.testing.assert_allclose(qfrc_no_cop[1], 0.0, atol=1e-12)
