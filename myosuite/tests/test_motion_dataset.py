# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for ``myosuite.biomechanics.motion_dataset`` and ``myosuite.integrations.motion_datasets`` (#833)."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite.biomechanics.motion_dataset import (
    DatasetLicense,
    JointSpaceClipSource,
    KinematicTrial,
    ManifestMotionDataset,
    MotionDataset,
    SubjectMetadata,
    ensure_redistributable,
    retarget_trial,
)
from myosuite.biomechanics.retarget import (
    Landmark,
    LandmarkMap,
    OrientationTarget,
    kinematic_fingerprint,
    read_refit_meta,
    save_refit,
)
from myosuite.biomechanics.retarget_smplh import (
    SMPLHMap,
    SMPLHOrientation,
    SMPLHPoint,
    smplh_source,
)
from myosuite.biomechanics.smplh import SMPLH_JOINTS, SMPLH_VERTICES
from myosuite.integrations.motion_datasets import (
    AMASS_RESEARCH_LICENSE,
    AMASSMotionDataset,
    MarkerMotionDataset,
    OpenSimMotDataset,
    RetargetedNPZDataset,
    load_manifest_dataset,
)

pytestmark = pytest.mark.tier1


_TOY_BODY_XML = """
<mujoco model="toy_leg">
  <compiler angle="radian"/>
  <worldbody>
    <body name="pelvis" pos="0 0 0.9">
      <freejoint name="root"/>
      <geom name="pelvis_g" type="sphere" size="0.05"/>
      <site name="pelvis_s" pos="0 0 0"/>
      <site name="rasi_s" pos="0.08 -0.10 0"/>
      <site name="lasi_s" pos="0.08 0.10 0"/>
      <body name="femur_r" pos="0 -0.1 -0.1">
        <joint name="hip_flexion_r" type="hinge" axis="0 1 0" limited="true" range="-1.5 1.5"/>
        <geom name="femur_g" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.03"/>
        <body name="tibia_r" pos="0 0 -0.4" euler="0 0.15 0">
          <joint name="knee_angle_r" type="hinge" axis="0 1 0" limited="true" range="0 2.2"/>
          <geom name="tibia_g" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.025"/>
          <site name="knee_r_s" pos="0 0 0"/>
          <site name="ankle_r_s" pos="0 0 -0.4"/>
          <body name="patella_r" pos="0.04 0 0">
            <joint name="knee_coupled_r" type="hinge" axis="0 1 0" limited="true" range="0 2.2"/>
            <geom name="patella_g" type="sphere" size="0.015"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint name="knee_eq" joint1="knee_coupled_r" joint2="knee_angle_r" polycoef="0 0.5 0 0 0"/>
  </equality>
</mujoco>
"""


@pytest.fixture()
def toy_model_and_map() -> tuple[mujoco.MjModel, LandmarkMap]:
    model = mujoco.MjModel.from_xml_string(_TOY_BODY_XML)
    lmap = LandmarkMap(
        name="toy_map",
        version="1",
        landmarks=(
            Landmark(source="PELVIS", body="pelvis", site="pelvis_s", weight=1.0),
            Landmark(source="R_ANKLE", body="tibia_r", site="ankle_r_s", weight=1.0),
        ),
        orientations=(OrientationTarget(source="pelvis", body="pelvis", weight=0.1),),
    )
    return model, lmap


def _write_mot(path: Path, n_frames: int = 8, fps: float = 100.0) -> None:
    times = np.arange(n_frames) / fps
    hip = 15.0 * np.sin(2.0 * np.pi * times)
    # gait2392 convention: negative knee_angle_r (deg) maps to positive flexion via GAIT2392_SIGNS
    knee = -25.0 * np.abs(np.sin(2.0 * np.pi * times))
    lines = [
        "Coordinates",
        "version=1",
        f"nRows={n_frames}",
        "nColumns=9",
        "inDegrees=yes",
        "endheader",
        "time\tpelvis_tilt\tpelvis_list\tpelvis_rotation\tpelvis_tx\tpelvis_ty\tpelvis_tz\thip_flexion_r\tknee_angle_r",
    ]
    for t, h, k in zip(times, hip, knee, strict=True):
        lines.append(f"{t:.4f}\t0.0\t0.0\t0.0\t0.0\t0.9\t0.0\t{h:.6f}\t{k:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_trc(
    path: Path, markers_by_name: dict[str, np.ndarray], fps: float = 50.0
) -> None:
    names = list(markers_by_name)
    n_frames = len(next(iter(markers_by_name.values())))
    n_markers = len(names)
    header_names = "\t\t\t".join(names) + "\t\t"
    sub_cols = "\t".join(f"X{k + 1}\tY{k + 1}\tZ{k + 1}" for k in range(n_markers))
    lines = [
        "PathFileType\t4\t(X/Y/Z)\ttest.trc",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{fps:g}\t{fps:g}\t{n_frames}\t{n_markers}\tm\t{fps:g}\t1\t{n_frames}",
        f"Frame#\tTime\t{header_names}",
        f"\t\t{sub_cols}",
    ]
    for i in range(n_frames):
        t = i / fps
        coords = "\t".join(
            f"{markers_by_name[nm][i, c]:.6f}" for nm in names for c in range(3)
        )
        lines.append(f"{i + 1}\t{t:.4f}\t{coords}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_license_enforcement_blocks_non_redistributable_export(
    tmp_path: Path, toy_model_and_map: tuple[mujoco.MjModel, LandmarkMap]
) -> None:
    model, lmap = toy_model_and_map
    cc_by = DatasetLicense(
        name="CC-BY-4.0",
        spdx_or_url="CC-BY-4.0",
        citation="Test et al. 2026",
        redistributable_derivatives=True,
        requires_registration=False,
    )
    ensure_redistributable(cc_by)
    with pytest.raises(PermissionError, match="redistributable_derivatives=False"):
        ensure_redistributable(AMASS_RESEARCH_LICENSE)

    qpos = np.tile(model.qpos0, (4, 1))
    restricted_out = tmp_path / "restricted.npz"
    # Local caching (default require_redistributable=False) succeeds:
    save_refit(
        restricted_out,
        model,
        qpos,
        50.0,
        {
            "model": "toy_leg",
            "model_version": "0.1",
            "license": AMASS_RESEARCH_LICENSE.as_dict(),
        },
    )
    assert restricted_out.is_file()

    # Redistributable export refuses AMASS / non-redistributable derivatives:
    with pytest.raises(PermissionError, match="redistributable_derivatives=False"):
        save_refit(
            tmp_path / "blocked_upload.npz",
            model,
            qpos,
            50.0,
            {
                "model": "toy_leg",
                "model_version": "0.1",
                "license": AMASS_RESEARCH_LICENSE.as_dict(),
            },
            require_redistributable=True,
        )

    # Redistributable export succeeds when license permits derivatives:
    allowed_out = tmp_path / "allowed_upload.npz"
    save_refit(
        allowed_out,
        model,
        qpos,
        50.0,
        {
            "model": "toy_leg",
            "model_version": "0.1",
            "model_fingerprint": kinematic_fingerprint(
                model, ("pelvis", "tibia_r"), ((), ())
            ),
            "license": cc_by.as_dict(),
        },
        require_redistributable=True,
    )
    meta = read_refit_meta(allowed_out)
    assert meta is not None and meta["license"]["redistributable_derivatives"] is True


def test_joint_space_fast_path_opensim_and_memory_trials(
    tmp_path: Path, toy_model_and_map: tuple[mujoco.MjModel, LandmarkMap]
) -> None:
    model, lmap = toy_model_and_map
    mot_file = tmp_path / "S01" / "walk_01.mot"
    mot_file.parent.mkdir(parents=True)
    _write_mot(mot_file, n_frames=10, fps=100.0)

    lic = DatasetLicense(
        name="CC-BY-4.0",
        spdx_or_url="CC-BY-4.0",
        redistributable_derivatives=True,
    )
    ds = OpenSimMotDataset(
        tmp_path,
        license_meta=lic,
        splits={"S01/walk_01": "val"},
        subjects={
            "S01": SubjectMetadata(
                subject_id="S01", height_m=1.74, mass_kg=68.0, sex="female"
            )
        },
    )
    assert isinstance(ds, MotionDataset)
    assert ds.trial_ids() == ("S01/walk_01",)
    assert ds.trial_ids(split="val") == ("S01/walk_01",)
    assert ds.trial_ids(split="train") == ()

    fp = kinematic_fingerprint(model, ("pelvis", "tibia_r"), ((), ()))
    qpos, freq, meta = retarget_trial(
        ds,
        "S01/walk_01",
        model,
        "toy_leg",
        "0.2.5",
        lmap,
        model_fingerprint=fp,
    )
    assert qpos.shape == (10, model.nq)
    assert freq == pytest.approx(100.0)
    # Fast-path skipped Gauss-Newton iterations while still projecting couplings!
    assert meta["iterations_max"] == 0
    assert meta["metrics"]["coupling_residual"] < 1e-12
    assert meta["subject"]["height_m"] == pytest.approx(1.74)
    assert meta["license"]["name"] == "CC-BY-4.0"
    # Verify exact knee coupling: knee_coupled_r == 0.5 * knee_angle_r
    ka = model.joint("knee_angle_r").qposadr[0]
    kc = model.joint("knee_coupled_r").qposadr[0]
    np.testing.assert_allclose(qpos[:, kc], 0.5 * qpos[:, ka], atol=1e-12)

    # Also test in-memory qpos KinematicTrial + RetargetedNPZDataset round-trip
    npz_path = tmp_path / "retargeted" / "clip0.npz"
    save_refit(npz_path, model, qpos, freq, meta, require_redistributable=True)
    npz_ds = RetargetedNPZDataset(tmp_path / "retargeted", license_meta=lic)
    assert isinstance(npz_ds, MotionDataset)
    qpos2, freq2, meta2 = retarget_trial(
        npz_ds, "clip0", model, "toy_leg", "0.2.5", lmap
    )
    np.testing.assert_allclose(qpos2, qpos, atol=1e-12)
    assert freq2 == pytest.approx(100.0)
    assert meta2["joint_space_kind"] == "npz"

    # In-memory KinematicTrial with deliberate unprojected coupling + HOI object dict
    raw_unprojected = qpos.copy()
    raw_unprojected[:, kc] = 0.0
    mem_trial = KinematicTrial(
        trial_id="mem_hoi_01",
        format="joint-space",
        fps=60.0,
        arrays={"qpos": raw_unprojected},
        objects={"cylinder": {"pose_wxyz_xyz": [1, 0, 0, 0, 0.3, 0.0, 0.8]}},
        license=lic,
    )
    src = JointSpaceClipSource()
    targets = src.targets(mem_trial, model, lmap)
    assert targets.qpos is not None
    np.testing.assert_allclose(
        targets.qpos[:, kc], 0.5 * targets.qpos[:, ka], atol=1e-12
    )


def test_marker_dataset_with_static_calibration_and_manifest_loader(
    tmp_path: Path, toy_model_and_map: tuple[mujoco.MjModel, LandmarkMap]
) -> None:
    model, _ = toy_model_and_map
    marker_lmap = LandmarkMap(
        name="toy_markers",
        version="1",
        landmarks=(
            Landmark(source="PELVIS", body="pelvis", site="pelvis_s", weight=1.0),
            Landmark(source="RASI", body="pelvis", site="rasi_s", weight=1.0),
            Landmark(source="LASI", body="pelvis", site="lasi_s", weight=1.0),
            Landmark(source="R_KNEE", body="tibia_r", site="knee_r_s", weight=1.0),
            Landmark(source="R_ANKLE", body="tibia_r", site="ankle_r_s", weight=1.0),
        ),
    )
    n_frames = 6
    data = mujoco.MjData(model)
    tracks_z = {
        k: np.empty((n_frames, 3))
        for k in ("PELVIS", "RASI", "LASI", "R_KNEE", "R_ANKLE")
    }
    site_of = {
        "PELVIS": "pelvis_s",
        "RASI": "rasi_s",
        "LASI": "lasi_s",
        "R_KNEE": "knee_r_s",
        "R_ANKLE": "ankle_r_s",
    }
    for i in range(n_frames):
        q = model.qpos0.copy()
        q[model.joint("hip_flexion_r").qposadr[0]] = -0.15 - 0.05 * i
        q[model.joint("knee_angle_r").qposadr[0]] = 0.2 + 0.08 * i
        q[model.joint("knee_coupled_r").qposadr[0]] = 0.1 + 0.04 * i
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        for key, sname in site_of.items():
            tracks_z[key][i] = data.site_xpos[model.site(sname).id]

    to_yup = lambda p: np.stack([p[:, 0], p[:, 2], -p[:, 1]], axis=-1)  # noqa: E731
    trc_dir = tmp_path / "markers"
    trc_dir.mkdir()
    static_trc = trc_dir / "S02_static.trc"
    dyn_trc = trc_dir / "S02_flexion.trc"
    _write_trc(static_trc, {k: to_yup(v[:3]) for k, v in tracks_z.items()}, fps=50.0)
    _write_trc(dyn_trc, {k: to_yup(v) for k, v in tracks_z.items()}, fps=50.0)

    lic = DatasetLicense(name="CC-BY-4.0", redistributable_derivatives=True)
    mds = MarkerMotionDataset(
        trc_dir,
        license_meta=lic,
        axes="Y-up",
        static_trials={"S02_flexion": static_trc},
    )
    assert isinstance(mds, MotionDataset)
    assert mds.trial_ids() == ("S02_flexion",)
    from myosuite.biomechanics.retarget import RefitConfig

    qpos, freq, meta = retarget_trial(
        mds,
        "S02_flexion",
        model,
        "toy_leg",
        "0.2.5",
        marker_lmap,
        config=RefitConfig(smooth_weight=0.0, max_iters=60),
    )
    assert freq == pytest.approx(50.0)
    assert meta["metrics"]["landmark_rmse_mm"] < 5.0
    assert "PELVIS" in meta["static_marker_medians_m"]

    # Declarative JSON manifest loading both marker and OpenSim trials without custom Python classes
    mot_path = tmp_path / "S02_walk.mot"
    _write_mot(mot_path, n_frames=6, fps=50.0)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(
        json.dumps(
            {
                "name": "lab_manifest_demo",
                "license": {
                    "name": "CC-BY-4.0",
                    "spdx_or_url": "CC-BY-4.0",
                    "redistributable_derivatives": True,
                },
                "subjects": {"S02": {"height_m": 1.80, "mass_kg": 76.0, "sex": "male"}},
                "trials": [
                    {
                        "trial_id": "marker_trial",
                        "format": "markers",
                        "paths": {
                            "dynamic": "markers/S02_flexion.trc",
                            "static": "markers/S02_static.trc",
                        },
                        "axes": "Y-up",
                        "subject_id": "S02",
                        "split": "train",
                    },
                    {
                        "trial_id": "mot_trial",
                        "format": "opensim-mot",
                        "path": "S02_walk.mot",
                        "subject_id": "S02",
                        "split": "test",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_ds = load_manifest_dataset(manifest_file)
    assert isinstance(manifest_ds, ManifestMotionDataset)
    assert isinstance(manifest_ds, MotionDataset)
    assert manifest_ds.trial_ids(split="train") == ("marker_trial",)
    assert manifest_ds.trial_ids(split="test") == ("mot_trial",)
    q_m, _, meta_m = retarget_trial(
        manifest_ds, "marker_trial", model, "toy_leg", "0.2.5", marker_lmap
    )
    q_o, _, meta_o = retarget_trial(
        manifest_ds, "mot_trial", model, "toy_leg", "0.2.5", marker_lmap
    )
    assert len(q_m) == 6 and meta_m["dataset"] == "lab_manifest_demo"
    assert len(q_o) == 6 and meta_o["iterations_max"] == 0


def test_amass_dataset_and_in_memory_smplh_trial(
    tmp_path: Path, toy_model_and_map: tuple[mujoco.MjModel, LandmarkMap]
) -> None:
    model, _ = toy_model_and_map
    # Build a synthetic minimal SMPL+H neutral model.npz so CI needs no external body model files
    n_verts = max(SMPLH_VERTICES.values()) + 1
    n_joints = len(SMPLH_JOINTS)
    v_template = np.zeros((n_verts, 3), dtype=np.float64)
    for idx in SMPLH_VERTICES.values():
        v_template[idx] = [0.0, -0.85, 0.05]
    shapedirs = np.zeros((n_verts, 3, 16), dtype=np.float64)
    posedirs = np.zeros((n_verts, 3, 9 * (n_joints - 1)), dtype=np.float64)
    j_reg = np.zeros((n_joints, n_verts), dtype=np.float64)
    v_template[0] = [0.0, 0.0, 0.0]
    v_template[1] = [-0.1, -0.8, 0.0]
    j_reg[0, 0] = 1.0  # Pelvis
    j_reg[SMPLH_JOINTS.index("R_Ankle"), 1] = 1.0
    weights = np.zeros((n_verts, n_joints), dtype=np.float64)
    weights[:, 0] = 1.0
    parents = np.zeros((2, n_joints), dtype=np.int64)
    parents[0, 0] = -1
    parents[1] = np.arange(n_joints)

    bm_dir = tmp_path / "body_models" / "neutral"
    bm_dir.mkdir(parents=True)
    np.savez(
        bm_dir / "model.npz",
        v_template=v_template,
        shapedirs=shapedirs,
        posedirs=posedirs,
        J_regressor=j_reg,
        weights=weights,
        kintree_table=parents,
    )

    smap = SMPLHMap(
        name="toy_smplh",
        version="1",
        points=(
            SMPLHPoint(
                name="pelvis", smplh=("Pelvis",), body="pelvis", model=("pelvis",)
            ),
            SMPLHPoint(
                name="r_ankle",
                smplh=("R_Ankle",),
                body="tibia_r",
                model=("tibia_r",),
                parent="pelvis",
            ),
        ),
        orientations=(SMPLHOrientation(joint="Pelvis", body="pelvis", weight=0.1),),
        hinges=(),
        posture=(),
        feet=((), ()),
        contact_vertices=(("L_Heel",), ("R_Heel",)),
    )
    src, lmap = smplh_source(model, tmp_path / "body_models", smap=smap)

    # 1. Disk-backed AMASSMotionDataset
    amass_dir = tmp_path / "KIT" / "subj01"
    amass_dir.mkdir(parents=True)
    clip_path = amass_dir / "walk01_poses.npz"
    np.savez(
        clip_path,
        poses=np.zeros((5, 3 * n_joints)),
        betas=np.zeros(16),
        trans=np.zeros((5, 3)),
        gender=np.array("neutral"),
        mocap_framerate=np.float64(60.0),
    )
    ads = AMASSMotionDataset(tmp_path / "KIT", name="kit", smplh_source=src)
    assert isinstance(ads, MotionDataset)
    assert ads.license.redistributable_derivatives is False
    assert ads.license.requires_registration is True
    assert ads.trial_ids() == ("subj01/walk01_poses",)
    q_disk, fps_disk, meta_disk = retarget_trial(
        ads, "subj01/walk01_poses", model, "toy_leg", "0.2.5", lmap
    )
    assert q_disk.shape == (5, model.nq) and fps_disk == pytest.approx(60.0)
    assert meta_disk["license"]["redistributable_derivatives"] is False

    # 2. In-memory KinematicTrial (SMPL-X / GVHMR normalized layout without temp files)
    mem_amass = KinematicTrial(
        trial_id="gvhmr_mem_01",
        format="amass-smplh",
        fps=30.0,
        arrays={
            "poses": np.zeros((5, 3 * n_joints)),
            "betas": np.zeros(16),
            "trans": np.zeros((5, 3)),
            "gender": "neutral",
        },
        subject=SubjectMetadata(subject_id="cam01", sex="neutral"),
        license=AMASS_RESEARCH_LICENSE,
    )
    targets_mem = src.targets(mem_amass, model, lmap)
    assert targets_mem.pos.shape == (5, 2, 3)
    assert targets_mem.frequency == pytest.approx(30.0)
