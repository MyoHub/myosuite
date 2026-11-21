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

    from myosuite.integrations.emg import Camargo2021Source as IntegratedCamargo
    from myosuite.integrations.mm_evalkit import CheckoutCapabilityError
    from myosuite.integrations.mm_evalkit.plugin import (
        _wheel_npz_main,
        custom_emg_main,
    )
    from myosuite.integrations.mm_evalkit.workflow import load_target_model

    assert Camargo2021Source is IntegratedCamargo

    empty_wt = tmp_path / "not_a_myosim_checkout"
    empty_wt.mkdir()
    with pytest.raises(CheckoutCapabilityError, match="myo_sim source checkout"):
        load_target_model(f"{empty_wt}:myoleg")

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
    for ep_key in (
        "rajagopal2016",
        "lencioni2019",
        "dimitrov2023",
        "camargo2021",
        "custom-emg",
    ):
        assert ep_key in ds_eps
        mod_name, attr_name = ds_eps[ep_key].split(":")
        assert callable(getattr(importlib.import_module(mod_name), attr_name))

    try:
        from mm_evalkit.registry import registered_datasets, run_dataset
    except ImportError:
        pass
    else:
        names = {d.name for d in registered_datasets()}
        if "custom-emg" in names:
            assert {
                "rajagopal2016",
                "lencioni2019",
                "dimitrov2023",
                "camargo2021",
                "custom-emg",
            } <= names
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

    from myosuite.integrations.mm_evalkit.plugin import camargo2021_main

    leg_xml_path = tmp_path / "mini_leg.xml"
    leg_xml_path.write_text(leg_xml, encoding="utf-8")
    plugin_camargo_out = tmp_path / "camargo_plugin_out"
    assert (
        camargo2021_main(
            [
                "--data-root",
                str(tmp_path),
                "--model",
                str(leg_xml_path),
                "--out-dir",
                str(plugin_camargo_out),
            ]
        )
        == 0
    )
    assert (plugin_camargo_out / "emg_validation_summary.json").exists()
