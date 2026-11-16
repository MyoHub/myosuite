# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""CLI and dataset-adapter tests for ``scripts/validation/``."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import scipy.io as sio

from scripts.validation.emg_sources import CAMARGO_CHANNELS, Camargo2021Source
from scripts.validation.emg_upper_limb import (
    EMG2POSE_JOINT_NAMES,
    Maurice2020Source,
    GrabMyoSource,
    UltraMoCap2025Source,
    _axis_rot,
    map_cyberglove_to_hand_qpos,
    map_emg2pose_to_hand_qpos,
    map_upper_limb_to_qpos,
    map_upper_limb_torques_to_qfrc,
    read_wfdb_record,
    thoracohumeral_to_upper_limb_coords,
    xsens_urdf_to_upper_limb_coords,
)
from scripts.validation.validate_emg import main as cli_main

pytestmark = pytest.mark.tier1

_TWO_MUSCLE_ARM_XML = """
<mujoco model="two_muscle_hinge">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="upper" pos="0 0 0.5">
      <site name="flex_orig" pos="0.04 0 0.12"/>
      <site name="ext_orig" pos="-0.04 0 0.12"/>
      <site name="lig_orig" pos="0.02 0 0.04"/>
      <body name="forearm" pos="0 0 0">
        <inertial pos="0 0 -0.15" mass="1.2" diaginertia="0.012 0.012 0.002"/>
        <joint name="elbow_flexion" type="hinge" axis="0 -1 0" range="-1.5 1.5" damping="0.8"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.30" size="0.025"/>
        <site name="flex_ins" pos="0.035 0 -0.05"/>
        <site name="ext_ins" pos="-0.035 0 -0.05"/>
        <site name="lig_ins" pos="0.02 0 -0.04"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial name="bic_tendon"><site site="flex_orig"/><site site="flex_ins"/></spatial>
    <spatial name="tri_tendon"><site site="ext_orig"/><site site="ext_ins"/></spatial>
    <spatial name="capsule_ligament" stiffness="25.0" damping="0.5" springlength="0.075 0.095">
      <site site="lig_orig"/><site site="lig_ins"/>
    </spatial>
  </tendon>
  <actuator>
    <muscle name="biceps_r" tendon="bic_tendon" force="120" lengthrange="0.09 0.20"/>
    <muscle name="triceps_r" tendon="tri_tendon" force="120" lengthrange="0.09 0.20"/>
  </actuator>
</mujoco>
"""

_MINI_SCAPULOTHORACIC_XML = """
<mujoco model="mini_scapulothoracic">
  <compiler angle="radian"/>
  <worldbody>
    <body name="chest_r" pos="0 0 1.0">
      <geom name="scapulothoracic_ellipsoid_r" type="ellipsoid" size="0.07 0.07 0.12"/>
      <body name="clavicle_r" pos="0.01 0.08 0.10">
        <joint name="sternoclavicular_r2_r" type="hinge" axis="0 0 1"/>
        <joint name="sternoclavicular_r3_r" type="hinge" axis="1 0 0"/>
        <joint name="sternoclavicular_r1_r" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 0" mass="0.15" diaginertia="1e-4 1e-4 1e-4"/>
        <body name="scapula_r" pos="-0.02 0.12 0.01">
          <joint name="acromioclavicular_r" type="ball"/>
          <inertial pos="0 0 0" mass="0.35" diaginertia="2e-4 2e-4 2e-4"/>
          <body name="humerus_r" pos="0 0.02 -0.02">
            <joint name="elv_angle_r" type="hinge" axis="0 0 1"/>
            <joint name="shoulder_elv_r" type="hinge" axis="1 0 0"/>
            <joint name="shoulder1_r2_r" type="hinge" axis="0 0 1"/>
            <joint name="shoulder_rot_r" type="hinge" axis="0 0 1"/>
            <inertial pos="0 0 -0.15" mass="1.8" diaginertia="0.01 0.01 0.002"/>
            <body name="ulna_r" pos="0 0 -0.28">
              <joint name="elbow_flexion_r" type="hinge" axis="0 1 0" range="0 2.3"/>
              <inertial pos="0 0 -0.12" mass="1.0" diaginertia="0.005 0.005 0.001"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_MINI_HAND_XML = """
<mujoco model="mini_hand">
  <compiler angle="radian"/>
  <worldbody>
    <body name="palm" pos="0 0 0">
      <body name="thumb" pos="0.02 0 0">
        <joint name="cmc_abduction_r" type="hinge" axis="1 0 0" range="-0.5 0.8"/>
        <joint name="cmc_flexion_r" type="hinge" axis="0 1 0" range="-0.5 0.9"/>
        <inertial pos="0 0 0" mass="0.02" diaginertia="1e-5 1e-5 1e-5"/>
      </body>
      <body name="index" pos="0.04 0 0">
        <joint name="mcp2_flexion_r" type="hinge" axis="0 1 0" range="-0.2 1.5"/>
        <joint name="pm2_flexion_r" type="hinge" axis="0 1 0" range="0.0 1.6"/>
        <joint name="md2_flexion_r" type="hinge" axis="0 1 0" range="0.0 1.4"/>
        <inertial pos="0 0 0" mass="0.02" diaginertia="1e-5 1e-5 1e-5"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_init_manifest_and_cli_ab_comparison(tmp_path: Path) -> None:
    tmpl = tmp_path / "starter_manifest.json"
    assert cli_main(["--init-manifest", str(tmpl)]) == 0
    assert json.loads(tmpl.read_text(encoding="utf-8"))["name"] == "custom_lab_emg"

    xml_a = tmp_path / "arm_a.xml"
    xml_b = tmp_path / "arm_b.xml"
    xml_a.write_text(_TWO_MUSCLE_ARM_XML, encoding="utf-8")
    xml_b.write_text(
        _TWO_MUSCLE_ARM_XML.replace('stiffness="25.0"', 'stiffness="45.0"'),
        encoding="utf-8",
    )

    t_emg = np.linspace(0.0, 1.0, 201)
    emg_csv = tmp_path / "trial01_emg.csv"
    rows = ["time,biceps,triceps"] + [
        f"{t:.4f},{0.2 + 0.5 * np.sin(np.pi * t):.5f},{0.05:.5f}" for t in t_emg
    ]
    emg_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    t_kin = np.linspace(0.0, 1.0, 101)
    angles_deg = np.degrees(0.3 + 0.4 * np.sin(np.pi * t_kin))
    mot_lines = [
        "Coordinates",
        "version=1",
        f"nRows={len(t_kin)}",
        "nColumns=2",
        "inDegrees=yes",
        "endheader",
        "time\telbow_flexion",
        *[f"{t:.4f}\t{a:.5f}" for t, a in zip(t_kin, angles_deg, strict=True)],
    ]
    kin_mot = tmp_path / "trial01_ik.mot"
    kin_mot.write_text("\n".join(mot_lines) + "\n", encoding="utf-8")

    manifest = {
        "name": "tmp_custom_arm",
        "signal_kind": "mvc_normalized",
        "kinematics_rate_hz": 100.0,
        "channel_map": {
            "name": "arm_2ch",
            "version": 1,
            "bindings": [
                {"channel": "biceps", "actuators": ["biceps_r"]},
                {"channel": "triceps", "actuators": ["triceps_r"]},
            ],
        },
        "trials": [
            {
                "trial_id": "subj01_flex_01",
                "emg_path": "trial01_emg.csv",
                "kinematics_path": "trial01_ik.mot",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    out_dir = tmp_path / "report"
    npz_dir = tmp_path / "exported_npz"
    mm_dir = tmp_path / "mm_evalkit_out"
    rc = cli_main(
        [
            "--dataset",
            "custom",
            "--manifest",
            str(manifest_path),
            "--model",
            str(xml_a),
            "--model-b",
            str(xml_b),
            "--rollout-audit",
            "--export-npz",
            str(npz_dir),
            "--export-mm-evalkit",
            str(mm_dir),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    summary = json.loads(
        (out_dir / "emg_validation_summary.json").read_text(encoding="utf-8")
    )
    assert (
        summary["model_a"]["model_fingerprint"]
        != summary["model_b"]["model_fingerprint"]
    )
    assert "comparison" in summary
    assert summary["comparison"]["shared_units"] == 1
    assert (out_dir / "emg_validation_summary.md").exists()
    assert (out_dir / "emg_validation_trials.jsonl").exists()
    assert (mm_dir / "policies.yaml").exists()
    assert len(list(mm_dir.glob("*_episodes.npz"))) == 2
    assert len(list(npz_dir.glob("*.npz"))) == 1

    try:
        from mm_evalkit.policies import load_policy_specs, validate_policy_files
        from mm_evalkit.workflows.bimanual.rollout import open_rollout
    except ImportError:
        pass
    else:
        specs = load_policy_specs(mm_dir / "policies.yaml")
        validate_policy_files(specs)
        assert len(specs) == 2
        for spec in specs:
            with open_rollout(spec.trajectory_path) as (_, ro_summary):
                assert ro_summary.n_episodes == 1

    from myosuite.integrations.mm_evalkit.plugin import _wheel_npz_main, custom_emg_main

    wheel_out = tmp_path / "wheel_report"
    assert (
        _wheel_npz_main(
            [
                "--npz-dir",
                str(npz_dir),
                "--model",
                str(xml_a),
                "--model-b",
                str(xml_b),
                "--out-dir",
                str(wheel_out),
            ]
        )
        == 0
    )
    assert (wheel_out / "emg_validation_summary.json").exists()

    plugin_out = tmp_path / "plugin_report"
    assert (
        custom_emg_main(
            [
                "--npz-dir",
                str(npz_dir),
                "--model",
                str(xml_a),
                "--out-dir",
                str(plugin_out),
            ]
        )
        == 0
    )
    assert (plugin_out / "emg_validation_summary.json").exists()

    import importlib

    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    repo_root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads(
        (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert "eval" in pyproject["project"]["optional-dependencies"]
    ds_eps = pyproject["project"]["entry-points"]["mm_evalkit.datasets"]
    expected_plugins = {
        "rajagopal2016",
        "lencioni2019",
        "dimitrov2023",
        "camargo2021",
        "quesada2024",
        "emg2pose",
        "ninapro-db5",
        "grabmyo",
        "ultramocap2025",
        "maurice2020",
        "custom-emg",
    }
    for ep_key in expected_plugins:
        assert ep_key in ds_eps
        mod_name, attr_name = ds_eps[ep_key].split(":")
        assert callable(getattr(importlib.import_module(mod_name), attr_name))

    try:
        from mm_evalkit.registry import registered_datasets, run_dataset
    except ImportError:
        pass
    else:
        names = {d.name for d in registered_datasets()}
        if expected_plugins <= names:
            reg_out = tmp_path / "registry_report"
            assert (
                run_dataset(
                    "custom-emg",
                    [
                        "--npz-dir",
                        str(npz_dir),
                        "--model",
                        str(xml_a),
                        "--out-dir",
                        str(reg_out),
                    ],
                )
                == 0
            )
            assert (reg_out / "emg_validation_summary.json").exists()


def test_upper_limb_and_hand_coordinate_mappers() -> None:
    m_st = mujoco.MjModel.from_xml_string(_MINI_SCAPULOTHORACIC_XML)
    coords = {
        "elv_angle": np.linspace(0.5, 1.2, 6),
        "shoulder_elv": np.linspace(0.2, 1.4, 6),
        "shoulder_rot": np.linspace(-0.3, 0.1, 6),
        "elbow_flexion": np.linspace(0.4, 1.5, 6),
    }
    q_st, ctrl_adr, has_st = map_upper_limb_to_qpos(m_st, coords)
    assert q_st.shape == (6, m_st.nq)
    assert has_st is True
    assert len(ctrl_adr) >= 4
    ac_adr = int(m_st.jnt_qposadr[m_st.joint("acromioclavicular_r").id])
    assert np.allclose(
        np.linalg.norm(q_st[:, ac_adr : ac_adr + 4], axis=1), 1.0, atol=1e-7
    )
    torques = {
        "elv_angle": np.full(6, 2.0),
        "shoulder_elv": np.linspace(5.0, 15.0, 6),
        "shoulder_rot": np.full(6, -1.0),
        "elbow_flexion": np.linspace(3.0, 8.0, 6),
    }
    ext_qfrc = map_upper_limb_torques_to_qfrc(m_st, q_st, 0.02, coords, torques)
    assert ext_qfrc.shape == (6, m_st.nv)
    assert np.isfinite(ext_qfrc).all()

    m_hand = mujoco.MjModel.from_xml_string(_MINI_HAND_XML)
    pose_20 = np.full((8, len(EMG2POSE_JOINT_NAMES)), 0.35, dtype=np.float64)
    q_e2p, e2p_adr = map_emg2pose_to_hand_qpos(m_hand, pose_20)
    assert q_e2p.shape == (8, m_hand.nq)
    assert len(e2p_adr) == 5

    glove = np.tile(np.linspace(80.0, 140.0, 22), (8, 1))
    rest_med = np.full(22, 90.0)
    span = np.full(22, 60.0)
    q_cg, cg_adr = map_cyberglove_to_hand_qpos(m_hand, glove, rest_med, span)
    assert q_cg.shape == (8, m_hand.nq)
    assert len(cg_adr) == 5


def test_camargo2021_source_synthetic(tmp_path: Path) -> None:
    leg_xml = """
    <mujoco model="mini_leg">
      <compiler angle="radian"/>
      <worldbody>
        <body name="pelvis" pos="0 0 1.0">
          <freejoint name="root"/>
          <inertial pos="0 0 0" mass="10.0" diaginertia="0.1 0.1 0.1"/>
          <body name="femur_r" pos="0 -0.08 0">
            <joint name="hip_flexion_r" type="hinge" axis="0 0 1" range="-0.6 1.5"/>
            <inertial pos="0 0 -0.2" mass="7.0" diaginertia="0.08 0.08 0.01"/>
            <body name="tibia_r" pos="0 0 -0.4">
              <joint name="knee_angle_r" type="hinge" axis="0 0 -1" range="0 2.1"/>
              <inertial pos="0 0 -0.2" mass="3.5" diaginertia="0.04 0.04 0.005"/>
              <body name="talus_r" pos="0 0 -0.4">
                <joint name="ankle_angle_r" type="hinge" axis="0 0 1" range="-0.8 0.8"/>
                <inertial pos="0.05 0 -0.03" mass="1.2" diaginertia="0.005 0.005 0.002"/>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
      <actuator>
        <general name="gasmed_r" joint="ankle_angle_r" gear="-40"/>
        <general name="tibant_r" joint="ankle_angle_r" gear="25"/>
        <general name="soleus_r" joint="ankle_angle_r" gear="-50"/>
        <general name="vasmed_r" joint="knee_angle_r" gear="-60"/>
        <general name="vaslat_r" joint="knee_angle_r" gear="-60"/>
        <general name="recfem_r" joint="knee_angle_r" gear="-45"/>
        <general name="bflh_r" joint="knee_angle_r" gear="35"/>
        <general name="bfsh_r" joint="knee_angle_r" gear="20"/>
        <general name="semiten_r" joint="knee_angle_r" gear="30"/>
        <general name="grac_r" joint="hip_flexion_r" gear="15"/>
        <general name="glmed1_r" joint="hip_flexion_r" gear="-20"/>
        <general name="glmed2_r" joint="hip_flexion_r" gear="-20"/>
        <general name="glmed3_r" joint="hip_flexion_r" gear="-20"/>
      </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(leg_xml)
    mode_dir = tmp_path / "AB06" / "10_09_18" / "levelground"
    for sub in ("emg", "ik", "id", "gcRight", "conditions"):
        (mode_dir / sub).mkdir(parents=True)

    t_ik = np.linspace(0.0, 2.0, 401)
    t_emg = np.linspace(0.0, 2.0, 2001)
    rng = np.random.default_rng(7)
    emg_cols = ("Header", *CAMARGO_CHANNELS, "rightexternaloblique")
    emg_data = np.column_stack([t_emg, rng.normal(0.0, 0.05, size=(len(t_emg), 11))])
    sio.savemat(
        mode_dir / "emg" / "levelground_normal_01.mat",
        {"columns": np.array(emg_cols, dtype=object), "data": emg_data},
    )

    ik_cols = (
        "Header",
        "pelvis_tx",
        "pelvis_ty",
        "pelvis_tz",
        "pelvis_tilt",
        "pelvis_list",
        "pelvis_rotation",
        "hip_flexion_r",
        "knee_angle_r",
        "ankle_angle_r",
    )
    ik_data = np.column_stack(
        [
            t_ik,
            0.8 * t_ik,
            np.full_like(t_ik, 0.95),
            np.zeros_like(t_ik),
            np.zeros_like(t_ik),
            np.zeros_like(t_ik),
            np.zeros_like(t_ik),
            20.0 * np.sin(2.0 * np.pi * t_ik),
            -35.0 * (0.5 - 0.5 * np.cos(2.0 * np.pi * t_ik)),
            10.0 * np.sin(2.0 * np.pi * t_ik),
        ]
    )
    sio.savemat(
        mode_dir / "ik" / "levelground_normal_01.mat",
        {"columns": np.array(ik_cols, dtype=object), "data": ik_data},
    )

    id_cols = (
        "Header",
        "hip_flexion_r_moment",
        "knee_angle_r_moment",
        "ankle_angle_r_moment",
    )
    ank_m = np.where(
        (t_ik >= 0.5) & (t_ik <= 1.5), -90.0 * np.sin(np.pi * (t_ik - 0.5)), 0.0
    )
    id_data = np.column_stack(
        [
            t_ik,
            25.0 * np.cos(2.0 * np.pi * t_ik),
            40.0 * np.sin(2.0 * np.pi * t_ik),
            ank_m,
        ]
    )
    sio.savemat(
        mode_dir / "id" / "levelground_normal_01.mat",
        {"columns": np.array(id_cols, dtype=object), "data": id_data},
    )

    hs_pct = np.mod((t_ik - 0.45) * 100.0, 100.0)
    sio.savemat(
        mode_dir / "gcRight" / "levelground_normal_01.mat",
        {
            "columns": np.array(("Header", "HeelStrike", "ToeOff"), dtype=object),
            "data": np.column_stack([t_ik, hs_pct, hs_pct]),
        },
    )
    sio.savemat(
        mode_dir / "conditions" / "levelground_normal_01.mat",
        {"labels": np.array(["walk"] * len(t_ik), dtype=object)},
    )

    src = Camargo2021Source(tmp_path)
    assert list(src.trial_ids()) == ["AB06/levelground/levelground_normal_01"]
    trial = src.get_trial("AB06/levelground/levelground_normal_01", model=model)
    assert trial.channel_names == CAMARGO_CHANNELS
    assert trial.qpos is not None and trial.external_qfrc is not None
    assert trial.meta["force_plates_matched"] is True
    assert 0.9 <= float(trial.time[-1]) <= 1.1


def test_grabmyo_wfdb_and_source_synthetic(tmp_path: Path) -> None:
    part_dir = tmp_path / "Session1" / "session1_participant1"
    part_dir.mkdir(parents=True)
    fs = 2048
    n_samp = 2048
    ch_names = ("F1", "F2", "U1", "W1", "W2")
    rng = np.random.default_rng(11)

    for gid, amp in ((11, 0.15), (12, 0.18), (13, 0.12), (17, 0.015)):
        stem = f"session1_participant1_gesture{gid}_trial1"
        phys = rng.normal(0.0, amp, size=(n_samp, len(ch_names)))
        gain = 10000.0
        base = 200
        raw_i16 = np.clip(np.rint(phys * gain + base), -32768, 32767).astype("<i2")
        (part_dir / f"{stem}.dat").write_bytes(raw_i16.tobytes())
        hea_lines = [
            f"{stem} {len(ch_names)} {fs} {n_samp}",
            *[f"{stem}.dat 16 {gain}({base})/mV 16 0 0 0 0 {nm}" for nm in ch_names],
        ]
        (part_dir / f"{stem}.hea").write_text(
            "\n".join(hea_lines) + "\n", encoding="utf-8"
        )

    fs_out, names_out, sig_out = read_wfdb_record(
        part_dir / "session1_participant1_gesture11_trial1.hea"
    )
    assert fs_out == 2048.0
    assert names_out == ch_names
    assert sig_out.shape == (n_samp, 5)

    hand_xml = """
    <mujoco model="mini_wrist_hand">
      <compiler angle="radian"/>
      <worldbody>
        <body name="forearm" pos="0 0 0">
          <joint name="pro_sup_r" type="hinge" axis="0 0 1" range="-1.5 1.5"/>
          <joint name="flexion_r" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
          <inertial pos="0 0 0" mass="0.5" diaginertia="1e-3 1e-3 1e-3"/>
        </body>
      </worldbody>
      <actuator>
        <general name="fcr_r" joint="flexion_r" gear="15"/>
        <general name="ecrl_r" joint="flexion_r" gear="-15"/>
        <general name="pt_r" joint="pro_sup_r" gear="12"/>
        <general name="sup_r" joint="pro_sup_r" gear="-12"/>
      </actuator>
    </mujoco>
    """
    m_wh = mujoco.MjModel.from_xml_string(hand_xml)
    src = GrabMyoSource(tmp_path, rest_slice_s=0.2, gesture_slice_s=0.4)
    assert list(src.trial_ids()) == ["Session1/participant1/trial1_wrist_forearm"]
    trial = src.get_trial("Session1/participant1/trial1_wrist_forearm", model=m_wh)
    assert trial.channel_names == ("F1", "F2", "W1", "W2")
    assert trial.qpos is not None
    assert len(trial.meta["controlled_qpos_adr"]) == 2

    xml_wh = tmp_path / "mini_wrist_hand.xml"
    xml_wh.write_text(hand_xml, encoding="utf-8")
    try:
        from mm_evalkit.registry import registered_datasets, run_dataset
    except ImportError:
        pass
    else:
        if "grabmyo" in {d.name for d in registered_datasets()}:
            grab_out = tmp_path / "grabmyo_plugin_report"
            assert (
                run_dataset(
                    "grabmyo",
                    [
                        "--data-root",
                        str(tmp_path),
                        "--model",
                        str(xml_wh),
                        "--out-dir",
                        str(grab_out),
                    ],
                )
                == 0
            )
            assert (grab_out / "emg_validation_summary.json").exists()


def test_thoracohumeral_decomposition_round_trip() -> None:
    m_hand = mujoco.MjModel.from_xml_string(_MINI_HAND_XML)
    e_x, e_y, _ = np.eye(3)
    truth = np.array([[0.4, 0.9, -0.3], [1.2, 1.6, 0.5], [-0.8, 0.3, 0.2]])
    r_ht = np.stack(
        [
            _axis_rot(e_y, p)
            @ _axis_rot(-e_x, e)
            @ _axis_rot(e_y, -p)
            @ _axis_rot(e_y, r)
            for p, e, r in truth
        ]
    )
    coords = thoracohumeral_to_upper_limb_coords(m_hand, r_ht)
    got = np.column_stack(
        [coords["elv_angle"], coords["shoulder_elv"], coords["shoulder_rot"]]
    )
    assert np.allclose(got, truth, atol=1e-6)

    zeros = np.zeros(2)
    tpose = {
        f"angle_{j}": zeros.copy()
        for j in (
            "jT9T8_x",
            "jT9T8_y",
            "jT9T8_z",
            "jRightC7Shoulder_x",
            "jRightShoulder_x",
            "jRightShoulder_y",
            "jRightShoulder_z",
            "jRightElbow_y",
            "jRightElbow_z",
            "jL5S1_x",
            "jL5S1_y",
            "jL4L3_x",
            "jL4L3_y",
            "jL1T12_x",
            "jL1T12_y",
        )
    }
    tpose["angle_jL5S1_y"] = np.full(2, 0.2)
    xs = xsens_urdf_to_upper_limb_coords(m_hand, tpose)
    assert np.allclose(xs["shoulder_elv"], 0.5 * np.pi, atol=1e-6)
    assert np.allclose(xs["elv_angle"], 0.0, atol=1e-6)
    assert np.allclose(xs["flex_extension"], -0.2)
    tpose["angle_jRightShoulder_x"] = np.full(2, 0.5 * np.pi)
    hanging = xsens_urdf_to_upper_limb_coords(m_hand, tpose)
    assert np.allclose(hanging["shoulder_elv"], 0.0, atol=1e-5)


def test_ultramocap2025_source_synthetic(tmp_path: Path) -> None:
    task_dir = tmp_path / "P01" / "Elbow Flexion"
    (task_dir / "EMG").mkdir(parents=True)
    (task_dir / "Processed IK").mkdir()
    n_frames = 300
    rng = np.random.default_rng(3)
    emg_rows = ["Frame,Sub Frame,IM EMG4,IM EMG5,IM EMG6"] + [
        f"{f},{s},{rng.normal(0, 1e-4):.3e},{rng.normal(0, 1e-4):.3e},{rng.normal(0, 1e-4):.3e}"
        for f in range(1, n_frames + 1)
        for s in range(20)
    ]
    (task_dir / "EMG" / "P01_ElbowFlexion_Normal.csv").write_text(
        "\n".join(emg_rows) + "\n", encoding="utf-8"
    )
    t = np.arange(n_frames) / 100.0
    ik_cols = {
        "time": t,
        "SC_y": np.zeros(n_frames),
        "SC_x": np.zeros(n_frames),
        "SC_z": np.zeros(n_frames),
        "arm_flex_r": 20.0 * np.sin(np.pi * t),
        "arm_add_r": np.full(n_frames, -10.0),
        "arm_rot_r": np.zeros(n_frames),
        "elbow_flex_r": 60.0 + 40.0 * np.sin(2.0 * np.pi * t),
        "pro_sup_r": np.full(n_frames, 90.0),
    }
    ik_rows = [",".join(ik_cols)] + [
        ",".join(f"{v[i]:.5f}" for v in ik_cols.values()) for i in range(n_frames)
    ]
    (task_dir / "Processed IK" / "P01_ElbowFlexion_Normal.csv").write_text(
        "\n".join(ik_rows) + "\n", encoding="utf-8"
    )

    model = mujoco.MjModel.from_xml_string(_MINI_SCAPULOTHORACIC_XML)
    src = UltraMoCap2025Source(tmp_path, window_duration_s=2.0)
    assert list(src.trial_ids()) == ["P01/ElbowFlexion/Normal"]
    trial = src.get_trial("P01/ElbowFlexion/Normal", model=model)
    assert trial.channel_names == ("biceps_brachii", "triceps_long", "deltoid_middle")
    assert trial.signals.shape[1] == 3
    assert 1.9 <= float(trial.time[-1]) <= 2.0
    assert trial.qpos is not None and trial.kinematics_time is not None
    assert trial.qpos.shape == (len(trial.kinematics_time), model.nq)
    elbow = int(model.jnt_qposadr[model.joint("elbow_flexion_r").id])
    t_win = trial.kinematics_time + trial.meta["window_start_s"]
    expected = np.radians(60.0 + 40.0 * np.sin(2.0 * np.pi * t_win))
    assert np.allclose(trial.qpos[:, elbow], np.clip(expected, 0.0, 2.3), atol=1e-3)
    cmap = src.default_channel_map(model)
    assert [b.channel for b in cmap.bindings] == list(trial.channel_names)


def test_maurice2020_source_synthetic(tmp_path: Path) -> None:
    rel = Path("Participant_1") / "NE1"
    stem = "Participant_1_Group_A_Session_NE1_Block_1"
    root = tmp_path / "AndyData-lab-onePersonWithExoskeleton"
    for sub in ("emg/emg_csv", "jointangles/jointangles_csv", "info/info_csv"):
        (root / sub / rel).mkdir(parents=True)

    t_emg = 100.0 + np.arange(6000) / 1000.0
    rng = np.random.default_rng(5)
    emg_lines = ["timestamp,anterior_deltoid,erector_spinae_longissimus", "[s],[V],[V]"]
    emg_lines += [
        f"{t:.4f},{rng.normal(0, 0.05):.5f},{rng.normal(0, 0.01):.5f}" for t in t_emg
    ]
    (root / "emg/emg_csv" / rel / f"{stem}.emg.csv").write_text(
        "\n".join(emg_lines) + "\n", encoding="utf-8"
    )

    t_ja = 100.0 + np.arange(360) / 60.0
    joints = (
        "jL5S1_x",
        "jL5S1_y",
        "jL4L3_x",
        "jL4L3_y",
        "jL1T12_x",
        "jL1T12_y",
        "jT9T8_x",
        "jT9T8_y",
        "jT9T8_z",
        "jRightC7Shoulder_x",
        "jRightShoulder_x",
        "jRightShoulder_y",
        "jRightShoulder_z",
        "jRightElbow_y",
        "jRightElbow_z",
    )
    ja = {f"angle_{j}": np.zeros_like(t_ja) for j in joints}
    ja["angle_jRightShoulder_x"] = np.linspace(1.4, 0.2, len(t_ja))
    ja["angle_jRightElbow_z"] = np.full_like(t_ja, 1.0)
    ja_lines = [",".join(["timestamp", *ja]), ",".join(["[s]"] + ["[rad]"] * len(ja))]
    ja_lines += [
        ",".join([f"{t:.4f}"] + [f"{v[i]:.5f}" for v in ja.values()])
        for i, t in enumerate(t_ja)
    ]
    ja_lines.append(ja_lines[3])  # stale trailing row from an earlier block
    (root / "jointangles/jointangles_csv" / rel / f"{stem}.jointangles.csv").write_text(
        "\n".join(ja_lines) + "\n", encoding="utf-8"
    )

    # Two trials: new start (7), in start (1), target (2), moving (3), hold (4).
    info = [(100.5, 7, 1), (101.0, 1, 1), (101.5, 2, 1), (101.8, 3, 1), (102.2, 4, 1)]
    info += [(103.0, 7, 2), (103.5, 1, 2), (104.0, 2, 2), (104.3, 3, 2), (104.7, 4, 2)]
    # Trial 3 runs past the joint-angle recording and is skipped.
    info += [(105.0, 7, 3), (105.5, 1, 3), (106.0, 2, 3), (106.3, 3, 3), (106.7, 4, 3)]
    info_lines = ["timestamp,state,trial", "[s],[N/A],[N/A]"] + [
        f"{t},{s}.0,{k}.0" for t, s, k in info
    ]
    (root / "info/info_csv" / rel / f"{stem}.info.csv").write_text(
        "\n".join(info_lines) + "\n", encoding="utf-8"
    )

    model = mujoco.MjModel.from_xml_string(_MINI_SCAPULOTHORACIC_XML)
    src = Maurice2020Source(tmp_path)
    assert list(src.trial_ids()) == ["P01/NE1/block1/trial01", "P01/NE1/block1/trial02"]
    trial = src.get_trial("P01/NE1/block1/trial01", model=model)
    assert trial.channel_names == ("anterior_deltoid", "erector_spinae_longissimus")
    assert trial.meta["window_s"] == [100.5, 103.0]
    assert src._load_block((1, "NE1", 1))["ja"][-1, 0] > 105.9
    assert np.allclose(np.diff(trial.time), 1e-3)
    assert trial.qpos is not None and np.isfinite(trial.qpos).all()
    elbow = int(model.jnt_qposadr[model.joint("elbow_flexion_r").id])
    assert np.allclose(trial.qpos[:, elbow], 1.0, atol=1e-6)
    cmap = src.default_channel_map(model)
    assert [b.channel for b in cmap.bindings] == ["anterior_deltoid"]
