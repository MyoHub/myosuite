# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for ``myosuite.biomechanics.mocap_io``."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite.biomechanics.inverse_dynamics import inverse_dynamics
from myosuite.biomechanics.mocap_io import (
    fill_marker_gaps,
    force_plates_to_qfrc,
    read_c3d_trial,
    read_markers,
    write_trc,
)
from myosuite.core.retargeting import retarget

pytestmark = pytest.mark.tier1

_FOOT_XML = """
<mujoco model="foot_plate">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="pelvis" pos="0 0 0.9">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="60.0" diaginertia="1.0 1.0 1.0"/>
      <body name="calcn_r" pos="0.0 -0.1 -0.88">
        <joint name="ankle_r" type="hinge" axis="0 1 0" range="-0.8 0.8"/>
        <inertial pos="0.05 0 0" mass="10.0" diaginertia="0.05 0.05 0.05"/>
        <site name="r_heel" pos="0 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_trc_write_read_roundtrip_and_gap_filling(tmp_path: Path) -> None:
    rate = 100.0
    t = np.arange(30) / rate
    true_pts = np.stack(
        [
            np.column_stack((0.4 * np.cos(2 * t), 0.4 * np.sin(2 * t), 1.1 + 0.1 * t)),
            np.column_stack((0.1 * t, np.full_like(t, -0.1), np.zeros_like(t))),
        ],
        axis=1,
    )
    gapped = true_pts.copy()
    gapped[5:8, 0] = np.nan  # 3-frame gap (0.03 s <= 0.05 s): filled
    gapped[15:24, 1] = np.nan  # 9-frame gap (0.09 s > 0.05 s): preserved as NaN

    trc_path = write_trc(
        tmp_path / "trial.trc", ("r_heel", "m2"), gapped, rate, units="mm"
    )
    loaded = read_markers(trc_path)
    assert loaded.names == ("r_heel", "m2") and loaded.rate == rate
    assert loaded.missing_mask.shape == (30, 2)
    assert loaded.missing_mask[5:8, 0].all()
    assert loaded.missing_mask[15:24, 1].all()
    assert not loaded.missing_mask[0, 0]
    assert np.isnan(loaded.positions[5:8, 0]).all()
    assert np.isnan(loaded.positions[15:24, 1]).all()
    finite = np.isfinite(gapped)
    np.testing.assert_allclose(loaded.positions[finite], gapped[finite], atol=1e-6)

    filled = fill_marker_gaps(loaded.positions, rate, max_gap_s=0.05, method="cubic")
    assert np.isfinite(filled[5:8, 0]).all()
    np.testing.assert_allclose(filled[5:8, 0], true_pts[5:8, 0], atol=1e-4)
    assert np.isnan(filled[15:24, 1]).all()

    model = mujoco.MjModel.from_xml_string(_FOOT_XML)
    clip = retarget(model, trc_path, ground_sites=())
    assert clip.qpos.shape == (30, model.nq)


def test_c3d_trial_mock_and_force_plate_qfrc_balances_standing_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = mujoco.MjModel.from_xml_string(_FOOT_XML)
    total_mass = float(model.body_mass[1:].sum())
    weight_n = total_mass * 9.81

    # 4 video frames at 100 Hz, 8 analog samples at 200 Hz; frames 0..1 loaded, 2..3 unloaded.
    xyz_mm = np.zeros((3, 1, 4))
    xyz_mm[1, 0, :] = -100.0
    xyz_mm[2, 0, :] = 20.0
    analogs = np.zeros((1, 2, 8))
    analogs[0, 0, :] = np.linspace(0.05, 0.45, 8)  # EMG1

    force_a = np.zeros((3, 8))
    force_a[2, :4] = -weight_n  # Foot pushes down on plate -> upward reaction +weight_n
    cop_a = np.zeros((3, 8))
    cop_a[:, :4] = [[50.0], [-100.0], [0.0]]  # mm -> (0.05, -0.10, 0.0) m
    cop_a[:, 4:] = np.nan  # Unloaded flight phase produces NaN CoP
    tz_a = np.zeros((3, 8))
    tz_a[2, :4] = -4000.0  # N*mm -> +4.0 N*m upward reaction free torque

    def fake_c3d(path: str, extract_forceplat_data: bool = False) -> dict:
        out: dict = {
            "parameters": {
                "POINT": {
                    "LABELS": {"value": ["r_heel"]},
                    "UNITS": {"value": ["mm"]},
                    "RATE": {"value": [100.0]},
                },
                "ANALOG": {
                    "LABELS": {"value": ["EMG_TA", "EMG_SOL"]},
                    "UNITS": {"value": ["mV", "mV"]},
                    "RATE": {"value": [200.0]},
                },
            },
            "data": {
                "points": np.concatenate([xyz_mm, np.ones((1, 1, 4))]),
                "meta_points": {"residuals": np.zeros((1, 1, 4))},
                "analogs": analogs,
            },
        }
        if extract_forceplat_data:
            out["data"]["platform"] = [
                {
                    "unit_force": "N",
                    "unit_moment": "Nmm",
                    "unit_position": "mm",
                    "corners": np.array(
                        [[250, -250, -250, 250], [250, 250, -250, -250], [0, 0, 0, 0]],
                        float,
                    ),
                    "force": force_a,
                    "center_of_pressure": cop_a,
                    "Tz": tz_a,
                }
            ]
        return out

    monkeypatch.setitem(sys.modules, "ezc3d", types.SimpleNamespace(c3d=fake_c3d))
    trial = read_c3d_trial(tmp_path / "walk.c3d")
    assert trial.analog_names == ("EMG_TA", "EMG_SOL")
    assert trial.analogs.shape == (8, 2) and trial.analog_rate == 200.0
    assert len(trial.force_plates) == 1
    plate = trial.force_plates[0]
    np.testing.assert_allclose(plate.force[:2, 2], weight_n)
    np.testing.assert_allclose(plate.force[2:], 0.0)
    assert np.isfinite(plate.cop).all()
    np.testing.assert_allclose(plate.free_torque[:2, 2], 4.0)

    qpos = np.tile(model.qpos0, (4, 1))
    qfrc = force_plates_to_qfrc(
        model, qpos, trial.force_plates, contact_bodies=("calcn_r",)
    )
    assert np.isfinite(qfrc).all()
    np.testing.assert_allclose(qfrc[2:], 0.0, atol=1e-12)

    # Vertical root generalized force balances static body weight exactly.
    id_static = inverse_dynamics(
        model, qpos[:2], np.zeros((2, model.nv)), np.zeros((2, model.nv))
    )
    np.testing.assert_allclose(qfrc[:2, 2], id_static[:, 2], atol=1e-10)
    # Ankle hinge axis +Y receives lever-arm moment -(cop_x - ankle_x) * Fz = -0.05 * weight_n.
    np.testing.assert_allclose(qfrc[:2, 6], -0.05 * weight_n, atol=1e-10)

    from myosuite.biomechanics import external_loads, mocap_io
    from myosuite.integrations import capture

    assert mocap_io.read_markers is capture.read_markers
    assert mocap_io.read_c3d_trial is capture.read_c3d_trial
    assert mocap_io.write_trc is capture.write_trc
    assert mocap_io.fill_marker_gaps is capture.fill_marker_gaps
    assert mocap_io.force_plates_to_qfrc is external_loads.force_plates_to_qfrc


def test_real_ezc3d_binary_roundtrip(tmp_path: Path) -> None:
    ezc3d = pytest.importorskip("ezc3d")
    c = ezc3d.c3d()
    c["parameters"]["POINT"]["UNITS"]["value"] = ["mm"]
    c["parameters"]["POINT"]["RATE"]["value"] = [100.0]
    c["parameters"]["POINT"]["LABELS"]["value"] = ["r_heel"]
    pts = np.zeros((4, 1, 5))
    pts[2, 0, :] = 20.0
    pts[3, 0, :] = 1.0
    c["data"]["points"] = pts
    c["parameters"]["ANALOG"]["RATE"]["value"] = [200.0]
    c["parameters"]["ANALOG"]["LABELS"]["value"] = [
        "Fx1",
        "Fy1",
        "Fz1",
        "Mx1",
        "My1",
        "Mz1",
        "EMG1",
    ]
    c["parameters"]["ANALOG"]["UNITS"]["value"] = [
        "N",
        "N",
        "N",
        "Nmm",
        "Nmm",
        "Nmm",
        "V",
    ]
    analogs = np.zeros((1, 7, 10))
    analogs[0, 2, :] = -550.0
    analogs[0, 6, :] = 0.25
    c["data"]["analogs"] = analogs
    c["parameters"]["FORCE_PLATFORM"]["USED"]["value"] = [1]
    c["parameters"]["FORCE_PLATFORM"]["TYPE"]["value"] = [2]
    c["parameters"]["FORCE_PLATFORM"]["ZERO"]["value"] = [0, 0]
    c["parameters"]["FORCE_PLATFORM"]["CORNERS"]["value"] = np.array(
        [[[250, -250, -250, 250], [250, 250, -250, -250], [0, 0, 0, 0]]], dtype=float
    )
    c["parameters"]["FORCE_PLATFORM"]["ORIGIN"]["value"] = np.array(
        [[0.0], [0.0], [-20.0]]
    )
    c["parameters"]["FORCE_PLATFORM"]["CHANNEL"]["value"] = np.arange(1, 7).reshape(
        6, 1
    )

    c3d_file = tmp_path / "real.c3d"
    c.write(str(c3d_file))
    trial = read_c3d_trial(c3d_file)
    assert trial.markers.positions.shape == (5, 1, 3)
    np.testing.assert_allclose(trial.markers.positions[:, 0, 2], 0.02, atol=1e-6)
    assert trial.analog_names[-1] == "EMG1"
    np.testing.assert_allclose(trial.analogs[:, 6], 0.25, atol=1e-6)
    np.testing.assert_allclose(trial.force_plates[0].force[:, 2], 550.0, atol=1e-5)
