# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the lightweight MOTOR job and output adapter."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from myosuite.biomechanics.inverse_dynamics import clip_kinematics, inverse_dynamics
from myosuite.integrations.citations import MOTOR_CITATION, get_integration_citation
from myosuite.integrations.motor import (
    MOTOR_TO_MUJOCO_FRAME,
    MUJOCO_TO_MOTOR_FRAME,
    MotorJobConfig,
    MotorPersonInput,
    motor_activations_to_model,
    motor_to_motion_clip,
    motor_to_qpos,
    motor_touches_to_qfrc,
    read_motor_output,
    read_motor_outputs,
    unmangle_motor_joint_name,
    write_motor_job,
)

pytestmark = pytest.mark.tier1

_SYNTHETIC_BIPED_XML = """
<mujoco model="motor_adapter_fixture">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <asset>
    <mesh name="pelvis_mesh" vertex="0 0 0  0.1 0 0  0 0.1 0  0 0 0.1"/>
    <mesh name="femur_mesh"  vertex="0 0 0  0.08 0 0  0 0.08 0  0 0 -0.2"/>
    <mesh name="tibia_mesh"  vertex="0 0 0  0.06 0 0  0 0.06 0  0 0 -0.2"/>
    <mesh name="calcn_mesh"  vertex="0 0 0  0.12 0 0  0 0.06 0  0 0 -0.04"/>
  </asset>
  <worldbody>
    <body name="Full Body" pos="0 0 0">
      <freejoint name="root"/>
      <body name="pelvis" pos="0.02 -0.01 0.03" euler="0.05 -0.03 0.02">
        <inertial pos="0 0 0" mass="12.0" diaginertia="0.12 0.10 0.08"/>
        <geom type="mesh" mesh="pelvis_mesh" contype="0" conaffinity="0"/>
        <site name="quad_origin" pos="0.04 0 -0.02"/>
        <body name="femur_r" pos="0 -0.09 -0.08">
          <joint name="hip_flexion_r" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
          <joint name="hip_adduction_r" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
          <inertial pos="0 0 -0.2" mass="8.0" diaginertia="0.09 0.09 0.02"/>
          <geom type="mesh" mesh="femur_mesh" contype="0" conaffinity="0"/>
          <body name="tibia_r" pos="0 0 -0.4">
            <joint name="knee_angle_r" type="hinge" axis="0 1 0" range="-2.2 0.1"/>
            <joint name="knee_coupled_r" type="hinge" axis="0 0 1" range="-1.0 1.0"/>
            <inertial pos="0 0 -0.18" mass="4.5" diaginertia="0.05 0.05 0.01"/>
            <geom type="mesh" mesh="tibia_mesh" contype="0" conaffinity="0"/>
            <site name="quad_insert" pos="0.04 0 -0.05"/>
            <body name="calcn_r" pos="0 0 -0.4">
              <joint name="ankle_angle_r" type="hinge" axis="0 1 0" range="-0.9 0.9"/>
              <inertial pos="0.05 0 -0.02" mass="1.5" diaginertia="0.01 0.01 0.005"/>
              <geom type="mesh" mesh="calcn_mesh" contype="0" conaffinity="0"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint joint1="knee_coupled_r" joint2="knee_angle_r" polycoef="0 0.35 -0.1 0 0"/>
  </equality>
  <tendon>
    <spatial name="rectus-femoris_tendon">
      <site site="quad_origin"/>
      <site site="quad_insert"/>
    </spatial>
    <spatial name="unmapped_tendon">
      <site site="quad_origin"/>
      <site site="quad_insert"/>
    </spatial>
  </tendon>
  <actuator>
    <muscle name="rectus-femoris_r" tendon="rectus-femoris_tendon" force="600"/>
    <muscle name="unmapped_muscle_r" tendon="unmapped_tendon" force="200"/>
  </actuator>
</mujoco>
"""


def _write_synthetic_motor_bundle(
    folder: Path, model: mujoco.MjModel
) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a ground-truth trajectory and write exact MOTOR out.json + out.npz."""
    num_frames = 12
    rate_hz = 60.0
    dt = 1.0 / rate_hz
    times = np.linspace(0.0, 1.0, num_frames)

    qpos_true = np.tile(model.qpos0, (num_frames, 1))
    for idx, phase in enumerate(times):
        qpos_true[idx, 0:3] = [
            0.05 * phase,
            -0.03 * phase,
            0.92 - 0.08 * np.sin(np.pi * phase),
        ]
        rot = Rotation.from_euler(
            "xyz", [0.04 * phase, 0.12 * np.sin(np.pi * phase), -0.08 * phase]
        )
        qpos_true[idx, 3:7] = np.roll(rot.as_quat(), 1)
        qpos_true[idx, model.jnt_qposadr[model.joint("hip_flexion_r").id]] = (
            -0.5 * np.sin(np.pi * phase)
        )
        qpos_true[idx, model.jnt_qposadr[model.joint("hip_adduction_r").id]] = (
            0.15 * phase
        )
        knee = -0.9 * np.sin(np.pi * phase)
        qpos_true[idx, model.jnt_qposadr[model.joint("knee_angle_r").id]] = knee
        qpos_true[idx, model.jnt_qposadr[model.joint("knee_coupled_r").id]] = (
            0.35 * knee - 0.1 * knee**2
        )
        qpos_true[idx, model.jnt_qposadr[model.joint("ankle_angle_r").id]] = (
            0.4 * np.sin(np.pi * phase)
        )

    bone_names = ["calcn_r", "femur_r", "pelvis", "tibia_r"]
    bones_yup = np.zeros((num_frames, len(bone_names), 7), dtype=np.float64)
    cop_mj = np.zeros((num_frames, 3), dtype=np.float64)
    grf_mj = np.zeros((num_frames, 3), dtype=np.float64)
    ext_qfrc = np.zeros((num_frames, model.nv), dtype=np.float64)

    data = mujoco.MjData(model)
    calcn_bid = model.body("calcn_r").id
    for frame in range(num_frames):
        data.qpos[:] = qpos_true[frame]
        mujoco.mj_fwdPosition(model, data)
        for bi, bname in enumerate(bone_names):
            bid = model.body(bname).id
            pos_yup = MUJOCO_TO_MOTOR_FRAME @ data.xpos[bid]
            rot_yup = MUJOCO_TO_MOTOR_FRAME @ data.xmat[bid].reshape(3, 3)
            quat_xyzw = Rotation.from_matrix(rot_yup).as_quat()
            bones_yup[frame, bi, :3] = pos_yup
            bones_yup[frame, bi, 3:7] = quat_xyzw
        if frame < 10:
            cop_mj[frame] = data.xpos[calcn_bid] + np.array([0.03, -0.01, -0.02])
            grf_mj[frame] = np.array(
                [12.0, -18.0, 260.0 + 30.0 * np.sin(np.pi * times[frame])]
            )
            mujoco.mj_applyFT(
                model,
                data,
                grf_mj[frame],
                np.zeros(3),
                cop_mj[frame],
                calcn_bid,
                ext_qfrc[frame],
            )

    kin = clip_kinematics(
        model, qpos_true, dt=dt, lowpass_hz=None, project_couplings=False
    )
    tau_true = inverse_dynamics(
        model, kin.qpos, kin.qvel, kin.qacc, external_qfrc=ext_qfrc
    )

    mangled_coords = [
        "fixture_worldbody_Full_Body_pelvis_femur_r__j0_hip_flexion_r",
        "fixture_worldbody_Full_Body_pelvis_femur_r__j0_femur_r_hip_adduction_r",
        "fixture_worldbody_Full_Body_pelvis_femur_r_tibia_r__j0_knee_angle_r",
        "fixture_worldbody_Full_Body_pelvis_femur_r_tibia_r_calcn_r_ankle_angle_r",
        "fixture_worldbody_Full_Body_pelvis_thoracic_FE",
    ]
    coord_matrix = np.column_stack(
        [
            qpos_true[:, model.jnt_qposadr[model.joint("hip_flexion_r").id]],
            qpos_true[:, model.jnt_qposadr[model.joint("hip_adduction_r").id]],
            qpos_true[:, model.jnt_qposadr[model.joint("knee_angle_r").id]],
            qpos_true[:, model.jnt_qposadr[model.joint("ankle_angle_r").id]],
            np.linspace(0.0, 0.2, num_frames),
        ]
    )

    weight_n = float(model.body_subtreemass[0] * 9.81)
    frames_json = []
    for frame in range(num_frames):
        touches = {}
        if frame < 10:
            f_yup = (MUJOCO_TO_MOTOR_FRAME @ grf_mj[frame]).tolist()
            c_yup = (MUJOCO_TO_MOTOR_FRAME @ cop_mj[frame]).tolist()
            touches["RightFoot"] = [*f_yup, *c_yup]
        frames_json.append(
            {
                "angles": [],
                "targets": [],
                "torques": [0.0, 0.0, 0.0, 0.0],
                "own torques": [
                    float(
                        tau_true[
                            frame, model.jnt_dofadr[model.joint("hip_flexion_r").id]
                        ]
                    ),
                    float(
                        tau_true[
                            frame, model.jnt_dofadr[model.joint("hip_adduction_r").id]
                        ]
                    ),
                    float(
                        tau_true[
                            frame, model.jnt_dofadr[model.joint("knee_angle_r").id]
                        ]
                    ),
                    float(
                        tau_true[
                            frame, model.jnt_dofadr[model.joint("ankle_angle_r").id]
                        ]
                    ),
                ],
                "ground": (MUJOCO_TO_MOTOR_FRAME @ grf_mj[frame]).tolist(),
                "lowest": 0.0 if frame < 10 else 0.04,
                "touches": touches,
                "feet": {
                    "RightFoot": [0.0 if frame < 10 else 0.04, float(grf_mj[frame, 2])]
                },
                "residual": [1.0, 2.0, -0.5, 0.1, -0.2, 0.05],
                "reserve": 0.25,
                "activations": [0.35 + 0.02 * frame, 0.8],
                "forces": [210.0 + 5.0 * frame, 95.0],
            }
        )

    out_json = folder / "out.json"
    out_npz = folder / "out.npz"
    out_json.write_text(
        json.dumps(
            {
                "engine": "synthetic test engine",
                "card": None,
                "rate": rate_hz,
                "people": {
                    "subj01": {
                        "model": {},
                        "muscles": ["rectus_femoris_r", "motor_only_muscle"],
                        "joints": mangled_coords[:4],
                        "bones": bone_names,
                        "rides": {"RightFoot": "calcn_r"},
                        "com height": 0.9,
                        "kg": float(model.body_subtreemass[0]),
                        "weight N": weight_n,
                        "adjusted": {
                            "mass %": -0.5,
                            "torso centre of mass cm": [1.2, -0.4, 0.0],
                        },
                        "pushes": {},
                        "frames": frames_json,
                        "seconds": {"fit": 1.2, "total": 2.5},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        out_npz,
        **{
            "subj01/coordinates": coord_matrix,
            "subj01/coordinate names": np.array(mangled_coords),
            "subj01/bones": bones_yup,
        },
    )
    return out_json, qpos_true, tau_true, grf_mj


def test_motor_citation_registered() -> None:
    citation = get_integration_citation("motor")
    assert citation is MOTOR_CITATION
    assert "Babon-Innovations-b-v/motor" in citation.project_url


def test_frame_matrices_are_proper_orthogonal_inverses() -> None:
    np.testing.assert_allclose(
        MOTOR_TO_MUJOCO_FRAME @ MUJOCO_TO_MOTOR_FRAME, np.eye(3), atol=1e-15
    )
    assert np.isclose(np.linalg.det(MOTOR_TO_MUJOCO_FRAME), 1.0)
    # MuJoCo +Z up maps to MOTOR +Y up; MuJoCo -Y forward maps to MOTOR +Z forward.
    np.testing.assert_allclose(MUJOCO_TO_MOTOR_FRAME @ [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(
        MUJOCO_TO_MOTOR_FRAME @ [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]
    )


def test_unmangle_motor_joint_name_handles_wrappers_and_underscores() -> None:
    pairs = [
        ("lumbar5", "flex_extension"),
        ("lumbar5", "lat_bending"),
        ("tibia_r", "knee_angle_r"),
        ("femur-r", "hip-flexion-r"),
    ]
    assert (
        unmangle_motor_joint_name(
            "Model_worldbody_Full_Body_sacrum_lumbar5__j0_lumbar5__j1_lat_bending",
            pairs,
        )
        == "lat_bending"
    )
    assert (
        unmangle_motor_joint_name(
            "Model_worldbody_Full_Body_pelvis_femur_r_tibia_r__j0_knee_angle_r",
            pairs,
        )
        == "knee_angle_r"
    )
    assert (
        unmangle_motor_joint_name(
            "Model_worldbody_Full_Body_pelvis_femur_r__j0_hip_flexion_r",
            pairs,
        )
        == "hip-flexion-r"
    )


def test_motor_roundtrip_kinematics_wrenches_torques_and_activations(
    tmp_path: Path,
) -> None:
    model = mujoco.MjModel.from_xml_string(_SYNTHETIC_BIPED_XML)
    out_json, qpos_true, tau_true, grf_mj = _write_synthetic_motor_bundle(
        tmp_path, model
    )

    all_trials = read_motor_outputs(out_json)
    assert list(all_trials) == ["subj01"]
    trial = read_motor_output(out_json)
    assert trial.person == "subj01"
    assert trial.num_frames == 12
    assert np.isclose(trial.dt, 1.0 / 60.0)
    assert np.isclose(trial.rra_mass_delta_pct, -0.5)
    np.testing.assert_allclose(trial.rra_torso_com_delta_m, [0.012, -0.004, 0.0])

    # Z-up ground force & active contact mask
    np.testing.assert_allclose(trial.ground_force, grf_mj, atol=1e-12)
    right_foot = trial.touches["RightFoot"]
    assert right_foot.in_contact[:10].all() and not right_foot.in_contact[10:].any()
    assert np.isnan(right_foot.cop[10:]).all()

    # Exact 6-DoF root (with non-zero pelvis offset/rotation) + hinges + polynomial coupling
    qpos_rec, unmapped = motor_to_qpos(model, trial)
    assert unmapped == ("fixture_worldbody_Full_Body_pelvis_thoracic_FE",)
    np.testing.assert_allclose(qpos_rec[:, :3], qpos_true[:, :3], atol=1e-12)
    dots = np.abs(np.sum(qpos_rec[:, 3:7] * qpos_true[:, 3:7], axis=1))
    np.testing.assert_allclose(dots, 1.0, atol=1e-12)
    np.testing.assert_allclose(qpos_rec[:, 7:], qpos_true[:, 7:], atol=1e-12)

    # MotionClip conversion
    clip = motor_to_motion_clip(model, trial)
    assert clip.num_frames == 12
    assert clip.frequency_hz == 60.0
    assert clip.metadata["unmapped_coordinates"] == [
        "fixture_worldbody_Full_Body_pelvis_thoracic_FE"
    ]

    # Contact wrench mapping reproduces inverse-dynamics joint torques to machine precision
    qfrc_ext = motor_touches_to_qfrc(model, trial, qpos_rec)
    np.testing.assert_allclose(qfrc_ext[10:], 0.0, atol=1e-15)
    kin = clip_kinematics(
        model, qpos_rec, dt=trial.dt, lowpass_hz=None, project_couplings=False
    )
    tau_rec = inverse_dynamics(
        model, kin.qpos, kin.qvel, kin.qacc, external_qfrc=qfrc_ext
    )
    knee_dof = int(model.jnt_dofadr[model.joint("knee_angle_r").id])
    np.testing.assert_allclose(
        tau_rec[:, knee_dof], trial.own_torques[:, 2], atol=1e-10
    )

    # Muscle activation and negative-tension MuJoCo force alignment
    act, frc, matched = motor_activations_to_model(model, trial)
    assert list(matched) == [True, False]
    np.testing.assert_allclose(act[:, 0], trial.activations[:, 0], atol=1e-12)
    np.testing.assert_allclose(frc[:, 0], -trial.muscle_tension[:, 0], atol=1e-12)
    assert np.isnan(act[:, 1]).all() and np.isnan(frc[:, 1]).all()

    # Hicks et al. (2015) residual check summary
    summary = trial.check_residuals(force_limit_bw=0.05, moment_limit_bw_ht=0.05)
    assert summary.passed_all_frames
    assert summary.fraction_within_limits == 1.0
    tight = trial.check_residuals(force_limit_bw=0.001, moment_limit_bw_ht=0.05)
    assert not tight.passed_all_frames
    assert tight.fraction_within_limits == 0.0


def test_write_motor_job_writes_valid_bundle_and_validates_inputs(
    tmp_path: Path,
) -> None:
    rig_bind = np.tile(np.eye(4), (2, 1, 1))
    world = np.tile(np.eye(4), (5, 1, 1, 1))
    ground = np.array([[[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0]]])
    points = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 1.7, 0.0], [0.0, 0.5, 0.1]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    weights = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [0.2, 0.8]])

    person = MotorPersonInput(
        name="p01",
        mass_kg=70.0,
        height_m=1.70,
        rig_names=("Root", "Hips"),
        rig_parents=(-1, 0),
        rig_bind=rig_bind,
        segments=("Hips",),
        world=world,
        ground=ground,
        points=points,
        faces=faces,
        weights=weights,
    )
    job_path = write_motor_job(
        tmp_path / "job_bundle",
        [person],
        MotorJobConfig(rate_hz=30.0, whole_trial="muscles"),
    )
    job = json.loads(job_path.read_text(encoding="utf-8"))
    assert job["rate"] == 30.0
    assert job["motor"] == "muscles"
    assert len(job["people"]) == 1
    target_npz = Path(job["people"][0]["targets"])
    assert target_npz.is_file()
    with np.load(target_npz, allow_pickle=False) as data:
        assert data["world"].shape == (5, 1, 4, 4)
        assert data["weights"].shape == (4, 2)

    with pytest.raises(ValueError, match="at least one"):
        write_motor_job(tmp_path / "empty", [])


def test_motor_module_imports_no_external_solver_packages() -> None:
    motor_py = Path(__file__).resolve().parents[1] / "integrations" / "motor.py"
    tree = ast.parse(motor_py.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported_roots.add(node.module.split(".")[0])
    forbidden = {"motor", "newton", "warp", "mujoco_warp", "pxr", "torch", "jax"}
    assert not (imported_roots & forbidden)
