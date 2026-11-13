# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for InstantHMR MHR70 retargeting (:mod:`myosuite.biomechanics.retarget_instanthmr`)."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.biomechanics import retarget as rt
from myosuite.biomechanics import retarget_instanthmr as ri
from myosuite.biomechanics import retarget_smplh as rs
from myosuite.envs.myo.tasks.mimic.tracking.models import (
    ModelInfo,
    compiled_body,
    kinematic_fingerprint,
    model_version,
)
from myosuite.envs.myo.tasks.mimic.tracking.motion import load_motion_lib
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

pytestmark = pytest.mark.tier2

_JIDX = {name: i for i, name in enumerate(ri.MHR70_JOINTS)}


@pytest.fixture(scope="module")
def fullbody() -> mujoco.MjModel:
    return compiled_body("myofullbody")


@pytest.fixture(scope="module")
def myolegs() -> mujoco.MjModel:
    env = make_env("myoLegWalk-v0")
    try:
        yield env.unwrapped.model.__copy__()
    finally:
        env.close()


def _mhr70_rig(
    m: mujoco.MjModel, imap: ri.InstantHMRMap
) -> tuple[np.ndarray, np.ndarray]:
    """Attach each of the 70 MHR keypoints rigidly to *m* in its calibration pose."""
    cal = ri.calibrate(m, imap)
    d = mujoco.MjData(m)
    d.qpos[:] = cal.qpos
    mujoco.mj_kinematics(m, d)
    fwd, left, up = cal.canon_to_model.T
    has_arms = "humerus_l" in {m.body(i).name for i in range(m.nbody)}

    body_ids = np.zeros(len(ri.MHR70_JOINTS), dtype=np.int64)
    world_pts = np.zeros((len(ri.MHR70_JOINTS), 3))

    def bind(name: str, body: str, world_pos: np.ndarray) -> None:
        idx = _JIDX[name]
        body_ids[idx] = m.body(body).id
        world_pts[idx] = world_pos

    pelvis_p = d.xpos[m.body("pelvis").id]
    for idx in range(len(ri.MHR70_JOINTS)):
        body_ids[idx] = m.body("pelvis").id
        world_pts[idx] = pelvis_p

    for side, sfx, sign in (("left", "_l", 1.0), ("right", "_r", -1.0)):
        bind(f"{side}_hip", f"femur{sfx}", d.xpos[m.body(f"femur{sfx}").id])
        bind(f"{side}_knee", f"tibia{sfx}", d.xpos[m.body(f"tibia{sfx}").id])
        bind(f"{side}_ankle", f"talus{sfx}", d.xpos[m.body(f"talus{sfx}").id])
        bind(f"{side}_heel", f"calcn{sfx}", d.xpos[m.body(f"calcn{sfx}").id])
        toes_p = d.xpos[m.body(f"toes{sfx}").id]
        bind(f"{side}_big_toe_tip", f"toes{sfx}", toes_p - sign * 0.025 * left)
        bind(f"{side}_small_toe_tip", f"toes{sfx}", toes_p + sign * 0.025 * left)

        if has_arms:
            hum_p = d.xpos[m.body(f"humerus{sfx}").id]
            ulna_p = d.xpos[m.body(f"ulna{sfx}").id]
            lun_p = d.xpos[m.body(f"lunate{sfx}").id]
            bind(f"{side}_shoulder", f"humerus{sfx}", hum_p)
            bind(f"{side}_acromion", f"humerus{sfx}", hum_p + 0.03 * up)
            bind(f"{side}_elbow", f"ulna{sfx}", ulna_p)
            bind(f"{side}_olecranon", f"humerus{sfx}", ulna_p - 0.035 * fwd)
            bind(f"{side}_cubital_fossa", f"humerus{sfx}", ulna_p + 0.035 * fwd)
            bind(f"{side}_wrist", f"lunate{sfx}", lun_p)
            bind(
                f"{side}_middle_third_joint",
                f"lunate{sfx}",
                d.xpos[m.body(f"3proxph{sfx}").id],
            )
            bind(
                f"{side}_index_third_joint",
                f"lunate{sfx}",
                d.xpos[m.body(f"2proxph{sfx}").id],
            )
            bind(
                f"{side}_pinky_third_joint",
                f"lunate{sfx}",
                d.xpos[m.body(f"5proxph{sfx}").id],
            )
        else:
            bind(
                f"{side}_shoulder", "pelvis", pelvis_p + 0.52 * up + sign * 0.18 * left
            )

    if has_arms:
        head_p = d.xpos[m.body("head").id]
        bind("neck", "torso", d.xpos[m.body("neck").id])
        bind("left_ear", "head", head_p + 0.07 * left)
        bind("right_ear", "head", head_p - 0.07 * left)
        bind("nose", "head", head_p + 0.09 * fwd)
        bind("left_eye", "head", head_p + 0.08 * fwd + 0.03 * left)
        bind("right_eye", "head", head_p + 0.08 * fwd - 0.03 * left)
    else:
        bind("neck", "pelvis", pelvis_p + 0.56 * up)

    local_off = np.empty_like(world_pts)
    for idx in range(len(ri.MHR70_JOINTS)):
        b = body_ids[idx]
        local_off[idx] = d.xmat[b].reshape(3, 3).T @ (world_pts[idx] - d.xpos[b])
    return body_ids, local_off


def _synthesize_mhr70_cam(
    m: mujoco.MjModel,
    imap: ri.InstantHMRMap,
    qpos: np.ndarray,
    *,
    subject_scale: float = 1.0,
    cam_to_world: np.ndarray | None = None,
    cam_offset_m: tuple[float, float, float] = (0.1, 0.35, 3.0),
) -> np.ndarray:
    cal = ri.calibrate(m, imap)
    R_cw = (
        cam_to_world
        if cam_to_world is not None
        else cal.canon_to_model @ ri.CAM_TO_WORLD
    )
    body_ids, local_off = _mhr70_rig(m, imap)
    d = mujoco.MjData(m)
    jw = np.empty((len(qpos), len(ri.MHR70_JOINTS), 3))
    for t, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        R_b = d.xmat[body_ids].reshape(-1, 3, 3)
        jw[t] = d.xpos[body_ids] + np.einsum("kij,kj->ki", R_b, local_off)
    jw_scaled = jw * subject_scale
    return jw_scaled @ R_cw + np.asarray(cam_offset_m, np.float64)


def test_mhr70_schema_and_gap_interpolation(tmp_path: Path) -> None:
    assert len(ri.MHR70_JOINTS) == 70
    assert np.isclose(np.linalg.det(ri.CAM_TO_WORLD), 1.0)
    assert np.allclose(ri.CAM_TO_WORLD.T @ ri.CAM_TO_WORLD, np.eye(3))

    t = np.linspace(0.0, 1.0, 11)
    local = np.zeros((11, 70, 3), dtype=np.float32)
    local[:, :, 1] = np.arange(70, dtype=np.float32)[None, :] * 0.01
    cam_trans = np.stack([2.0 * t, np.zeros_like(t), 3.0 + t], axis=-1).astype(
        np.float32
    )
    valid = np.ones(11, dtype=bool)
    valid[4:7] = False
    local_corrupt = local.copy()
    local_corrupt[4:7] = np.nan
    local_corrupt[8, 0, 0] = np.inf

    clip_path = tmp_path / "local_clip.npz"
    np.savez(
        clip_path,
        joints_3d_local=local_corrupt,
        cam_trans=cam_trans,
        frequency=30.0,
        valid=valid,
    )
    loaded = ri.load_instanthmr(clip_path)
    assert loaded["fps"] == 30.0
    assert loaded["interpolated_frames"] == 4
    expected = local + cam_trans[:, None, :]
    assert np.allclose(loaded["joints_3d_cam"], expected, atol=1e-6)

    bad_rot = tmp_path / "bad_rot.npz"
    np.savez(
        bad_rot,
        joints_3d_cam=expected,
        fps=30.0,
        cam_to_world=-np.eye(3),
    )
    with pytest.raises(ValueError, match="proper rotation"):
        ri.load_instanthmr(bad_rot)

    too_few = tmp_path / "too_few.npz"
    np.savez(
        too_few,
        joints_3d_cam=expected,
        fps=30.0,
        valid=np.array([True] + [False] * 10),
    )
    with pytest.raises(ValueError, match="at least 2 valid frames"):
        ri.load_instanthmr(too_few)


def test_triad_orientations_and_knee_flexion(fullbody: mujoco.MjModel) -> None:
    cal = ri.calibrate(fullbody, ri.INSTANTHMR_MYOFULLBODY)
    qpos = np.tile(cal.qpos, (4, 1))
    couplings = JointCouplings.from_model(fullbody)
    qpos[1, fullbody.jnt_qposadr[fullbody.joint("knee_angle_l").id]] = 0.8
    project_joint_couplings(couplings, qpos)

    j_cam = _synthesize_mhr70_cam(fullbody, ri.INSTANTHMR_MYOFULLBODY, qpos)
    jw = j_cam @ ri.CAM_TO_WORLD.T
    R = ri.segment_orientations(jw, ri.INSTANTHMR_MYOFULLBODY)
    assert R.shape == (4, len(ri.INSTANTHMR_MYOFULLBODY.orientations), 3, 3)
    assert np.allclose(np.swapaxes(R, -1, -2) @ R, np.eye(3), atol=1e-6)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6)

    jw_hyper = jw.copy()
    jw_hyper[2, _JIDX["left_knee"], 0] -= 0.04
    flex = ri.knee_flexion(jw_hyper)
    assert flex["l"][1] > 0.7
    assert flex["l"][2] < -0.05


def test_round_trip_myofullbody_and_axial_twists(
    fullbody: mujoco.MjModel, tmp_path: Path
) -> None:
    assert ri.landmark_bodies(ri.INSTANTHMR_MYOFULLBODY) == rs.landmark_bodies(
        rs.SMPLH_MYOFULLBODY
    )
    assert rt.kinematic_fingerprint(
        fullbody,
        ri.landmark_bodies(ri.INSTANTHMR_MYOFULLBODY),
        ri.INSTANTHMR_MYOFULLBODY.feet,
    ) == kinematic_fingerprint("myofullbody")

    cal = ri.calibrate(fullbody, ri.INSTANTHMR_MYOFULLBODY)
    T = 25
    phase = np.linspace(0.0, 2.0 * np.pi, T)
    q_gt = np.tile(cal.qpos, (T, 1))
    q_gt[:, 0] = np.linspace(0.0, 0.5, T)

    def set_j(name: str, vals: np.ndarray) -> None:
        j = fullbody.joint(name).id
        lo, hi = fullbody.jnt_range[j]
        q_gt[:, fullbody.jnt_qposadr[j]] = np.clip(vals, lo, hi)

    # Exercise both straight-knee (frames 0..5) and flexed-knee axial hip rotation,
    # straight-elbow humeral rotation, forearm pronation/supination, and subtalar roll.
    knee_wave = np.where(phase < 1.2, 0.0, 0.7 * np.sin(phase - 1.2) ** 2)
    set_j("knee_angle_l", knee_wave)
    set_j("hip_rotation_l", 0.35 * np.sin(phase))
    set_j("hip_flexion_l", 0.30 * np.sin(phase))
    set_j("shoulder_elv_l", 1.2 + 0.25 * np.cos(phase))
    set_j("elbow_flexion_l", np.where(phase < 1.5, 0.0, 0.9 * np.sin(phase - 1.5) ** 2))
    set_j("shoulder_rot_l", 0.45 * np.sin(phase))
    set_j("pro_sup_l", 0.65 * np.cos(phase))
    set_j("subtalar_angle_l", 0.20 * np.cos(phase))
    project_joint_couplings(JointCouplings.from_model(fullbody), q_gt)

    low, _ = rt._foot_kinematics(fullbody, cal.lmap, q_gt)
    q_gt[:, 2] -= low.min(axis=1)

    clip_path = tmp_path / "axial.npz"
    np.savez(
        clip_path,
        joints_3d_cam=_synthesize_mhr70_cam(fullbody, ri.INSTANTHMR_MYOFULLBODY, q_gt),
        fps=60.0,
    )
    src, lmap = ri.instanthmr_source(
        fullbody, ri.INSTANTHMR_MYOFULLBODY, target_sigma=0.0, ground_sigma=0.0
    )
    ik = rt.LandmarkIK(fullbody, lmap, rt.RefitConfig(smooth_weight=0.0))
    targets = src.targets(clip_path, fullbody, lmap)
    q_fit, _ = ik.fit(targets.pos, targets.quat, targets.seed, targets.posture)
    metrics = rt.validate_refit(ik, targets, q_fit)

    assert metrics["coupling_residual"] == 0.0
    assert metrics["limit_violations"] == 0
    assert metrics["landmark_rmse_mm"] < 5.0

    for jname in ("shoulder_rot_l", "pro_sup_l", "elbow_flexion_l"):
        adr = fullbody.jnt_qposadr[fullbody.joint(jname).id]
        err_deg = np.degrees(np.max(np.abs(q_fit[:, adr] - q_gt[:, adr])))
        assert err_deg < 2.5, f"{jname} max error {err_deg:.2f} deg"

    for jname in ("hip_rotation_l", "subtalar_angle_l"):
        adr = fullbody.jnt_qposadr[fullbody.joint(jname).id]
        corr = float(np.corrcoef(q_fit[:, adr], q_gt[:, adr])[0, 1])
        err_deg = np.degrees(np.max(np.abs(q_fit[:, adr] - q_gt[:, adr])))
        assert corr > 0.94, f"{jname} correlation {corr:.3f}"
        assert err_deg < 7.0, f"{jname} max error {err_deg:.2f} deg"


def test_round_trip_myolegs(myolegs: mujoco.MjModel, tmp_path: Path) -> None:
    cal = ri.calibrate(myolegs, ri.INSTANTHMR_MYOLEGS)
    T = 20
    phase = np.linspace(0.0, np.pi, T)
    q_gt = np.tile(cal.qpos, (T, 1))
    q_gt[:, 1] = -np.linspace(0.0, 0.4, T)
    q_gt[:, myolegs.jnt_qposadr[myolegs.joint("hip_flexion_r").id]] = 0.4 * np.sin(
        phase
    )
    q_gt[:, myolegs.jnt_qposadr[myolegs.joint("knee_angle_r").id]] = 0.6 * np.sin(phase)
    q_gt[:, myolegs.jnt_qposadr[myolegs.joint("hip_rotation_l").id]] = 0.25 * np.cos(
        phase
    )
    project_joint_couplings(JointCouplings.from_model(myolegs), q_gt)

    clip_path = tmp_path / "legs.npz"
    np.savez(
        clip_path,
        joints_3d_cam=_synthesize_mhr70_cam(myolegs, ri.INSTANTHMR_MYOLEGS, q_gt),
        fps=50.0,
    )
    src, lmap = ri.instanthmr_source(myolegs, ri.INSTANTHMR_MYOLEGS, target_sigma=0.0)
    ik = rt.LandmarkIK(myolegs, lmap, rt.RefitConfig(smooth_weight=0.0))
    q_fit, _, meta = rt.refit_clip(
        clip_path, myolegs, "myolegs", model_version("myofullbody"), lmap, src, ik=ik
    )
    assert meta["metrics"]["coupling_residual"] == 0.0
    assert meta["metrics"]["limit_violations"] == 0
    assert meta["metrics"]["landmark_rmse_mm"] < 1.0
    assert q_fit.shape == (T, myolegs.nq)


def test_metric_scaling_tilted_camera_and_flight_phase(
    fullbody: mujoco.MjModel, tmp_path: Path
) -> None:
    cal = ri.calibrate(fullbody, ri.INSTANTHMR_MYOFULLBODY)
    T = 41
    t = np.linspace(0.0, 1.0, T)
    q_gt = np.tile(cal.qpos, (T, 1))
    q_gt[:, 0] = 0.8 * t

    low0 = rt._foot_kinematics(fullbody, cal.lmap, q_gt[:1])[0].min()
    q_gt[:, 2] -= low0
    # Stance on [0..12] and [28..40], parabolic 0.25 m jump apex at frame 20.
    jump = np.zeros(T)
    air = slice(13, 28)
    tau = np.linspace(-1.0, 1.0, 15)
    jump[air] = 0.25 * (1.0 - tau**2)
    q_gt[:, 2] += jump

    # Pitch the camera downward by 20 deg around world y.
    pitch = np.radians(20.0)
    R_pitch = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ]
    )
    R_cw = R_pitch @ ri.CAM_TO_WORLD
    j_cam = _synthesize_mhr70_cam(
        fullbody,
        ri.INSTANTHMR_MYOFULLBODY,
        q_gt,
        subject_scale=0.8,
        cam_to_world=R_cw,
    )
    clip_path = tmp_path / "jump_tilted.npz"
    np.savez(clip_path, joints_3d_cam=j_cam, fps=60.0, cam_to_world=R_cw)

    src, lmap = ri.instanthmr_source(
        fullbody, ri.INSTANTHMR_MYOFULLBODY, target_sigma=0.0, ground_sigma=0.01
    )
    q_fit, _, meta = rt.refit_clip(
        clip_path, fullbody, "myofullbody", model_version("myofullbody"), lmap, src
    )

    assert np.isclose(meta["root_scale"], 1.25, atol=1e-3)
    # Scaled subject walked 0.8 * 0.8 m in camera space -> rescaled back to 0.8 m on model.
    assert np.isclose(q_fit[-1, 0] - q_fit[0, 0], 0.8, atol=0.02)

    low_fit, _ = rt._foot_kinematics(fullbody, lmap, q_fit)
    stance_sole = low_fit[:10].min(axis=1)
    apex_sole = float(low_fit[20].min())
    assert np.max(np.abs(stance_sole)) < 0.01
    assert apex_sole > 0.20, f"aerial apex should stay airborne, got {apex_sole:.3f} m"
    assert meta["metrics"]["penetration_mm"] == 0.0


def test_dropped_frames_avoid_velocity_spikes(
    fullbody: mujoco.MjModel, tmp_path: Path
) -> None:
    cal = ri.calibrate(fullbody, ri.INSTANTHMR_MYOFULLBODY)
    T = 51
    t = np.linspace(0.0, 1.0, T)
    q_gt = np.tile(cal.qpos, (T, 1))
    info = ModelInfo(fullbody)
    q_gt[:, 0] = 0.6 * t
    q_gt[:, fullbody.jnt_qposadr[fullbody.joint("hip_flexion_l").id]] = 0.35 * np.sin(
        2 * np.pi * t
    )
    q_gt[:, fullbody.jnt_qposadr[fullbody.joint("hip_flexion_r").id]] = -0.35 * np.sin(
        2 * np.pi * t
    )
    project_joint_couplings(info.couplings, q_gt)
    low0 = rt._foot_kinematics(fullbody, cal.lmap, q_gt)[0].min()
    q_gt[:, 2] -= low0

    j_cam = _synthesize_mhr70_cam(fullbody, ri.INSTANTHMR_MYOFULLBODY, q_gt)
    valid = np.ones(T, dtype=bool)
    valid[20:24] = False
    j_cam[20:24] = np.nan

    clip_path = tmp_path / "dropout.npz"
    np.savez(clip_path, joints_3d_cam=j_cam, fps=60.0, valid=valid)

    src, lmap = ri.instanthmr_source(fullbody, ri.INSTANTHMR_MYOFULLBODY)
    _, _, meta = rt.refit_clip(
        clip_path, fullbody, "myofullbody", model_version("myofullbody"), lmap, src
    )
    assert meta["interpolated_frames"] == 4
    assert meta["metrics"]["velocity_spikes"] == 0
    assert meta["metrics"]["limit_violations"] == 0


def test_cli_and_myomimic_motion_lib(fullbody: mujoco.MjModel, tmp_path: Path) -> None:
    from scripts.mimic import retarget_instanthmr as cli

    cal = ri.calibrate(fullbody, ri.INSTANTHMR_MYOFULLBODY)
    T = 31
    t = np.linspace(0.0, 0.5, T)
    q_gt = np.tile(cal.qpos, (T, 1))
    q_gt[:, 0] = 0.4 * t
    low0 = rt._foot_kinematics(fullbody, cal.lmap, q_gt)[0].min()
    q_gt[:, 2] -= low0

    in_dir = tmp_path / "raw_clips"
    in_dir.mkdir()
    np.savez(
        in_dir / "walk_01.npz",
        joints_3d_cam=_synthesize_mhr70_cam(fullbody, ri.INSTANTHMR_MYOFULLBODY, q_gt),
        fps=60.0,
    )
    out_root = tmp_path / "mimic_out"
    report_dir = tmp_path / "report"

    rc = cli.main(
        [
            str(in_dir),
            "--model",
            "myofullbody",
            "--out-root",
            str(out_root),
            "--report-dir",
            str(report_dir),
            "--workers",
            "1",
        ]
    )
    assert rc == 0

    refit_npz = out_root / "raw_clips" / "walk_01.npz"
    assert refit_npz.is_file()
    meta = rt.read_refit_meta(refit_npz)
    assert meta is not None
    assert meta["source"] == "InstantHMR/raw_clips/walk_01.npz"
    assert meta["source_format"] == "instanthmr-mhr70"
    assert meta["model_fingerprint"] == kinematic_fingerprint("myofullbody")
    summary = json.loads((report_dir / "retarget_summary.json").read_text())
    assert summary["clips"] == 1 and summary["errors"] == []

    # Second run skips unchanged clips cleanly.
    assert (
        cli.main(
            [
                str(in_dir),
                "--model",
                "myofullbody",
                "--out-root",
                str(out_root),
                "--report-dir",
                str(report_dir),
                "--workers",
                "1",
            ]
        )
        == 0
    )

    lib = load_motion_lib(str(refit_npz), "", "myofullbody", True, 0.02, 0.5)
    assert lib.num_frames >= 2


def test_video_extraction_tracks_primary_person_and_dropouts(
    fullbody: mujoco.MjModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types
    from dataclasses import dataclass

    from scripts.mimic import retarget_instanthmr as cli

    cal = ri.calibrate(fullbody, ri.INSTANTHMR_MYOFULLBODY)
    T = 12
    q_gt = np.tile(cal.qpos, (T, 1))
    q_gt[:, 0] = np.linspace(0.0, 0.25, T)
    low0 = rt._foot_kinematics(fullbody, cal.lmap, q_gt)[0].min()
    q_gt[:, 2] -= low0
    j_cam = _synthesize_mhr70_cam(fullbody, ri.INSTANTHMR_MYOFULLBODY, q_gt)

    @dataclass
    class _Person:
        bbox: np.ndarray
        joints_3d_cam: np.ndarray

    @dataclass
    class _FrameRes:
        persons: list[_Person]

    class _FakeCap:
        def __init__(self, _path: str) -> None:
            self._idx = 0

        def isOpened(self) -> bool:
            return True

        def get(self, _prop: int) -> float:
            return 30.0

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self._idx >= T:
                return False, None
            self._idx += 1
            return True, np.zeros((64, 64, 3), dtype=np.uint8)

        def release(self) -> None:
            pass

    class _FakePipeline:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._idx = 0

        def __call__(
            self, _rgb: np.ndarray, timestamp: float | None = None
        ) -> _FrameRes:
            assert timestamp is not None
            k = self._idx
            self._idx += 1
            if k in (4, 5):
                return _FrameRes([])
            primary = _Person(np.array([100.0, 50.0, 300.0, 450.0]), j_cam[k])
            bystander = _Person(
                np.array([10.0, 10.0, 60.0, 90.0]), j_cam[k] + np.array([5.0, 0.0, 0.0])
            )
            return _FrameRes([bystander, primary])

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.CAP_PROP_FPS = 5  # type: ignore[attr-defined]
    fake_cv2.COLOR_BGR2RGB = 4  # type: ignore[attr-defined]
    fake_cv2.VideoCapture = _FakeCap  # type: ignore[attr-defined]
    fake_cv2.cvtColor = lambda img, _code: img  # type: ignore[attr-defined]

    fake_hmr = types.ModuleType("instanthmr")
    fake_hmr.PosePipeline = _FakePipeline  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "instanthmr", fake_hmr)

    video = tmp_path / "demo.mp4"
    video.write_bytes(b"fake")
    out_root = tmp_path / "video_out"
    rc = cli.main(
        [
            str(video),
            "--model",
            "myofullbody",
            "--out-root",
            str(out_root),
            "--dataset",
            "cam",
            "--workers",
            "1",
        ]
    )
    assert rc == 0
    refit_npz = out_root / "cam" / "demo.npz"
    meta = rt.read_refit_meta(refit_npz)
    assert meta is not None
    assert meta["interpolated_frames"] == 2
    assert meta["metrics"]["landmark_rmse_mm"] < 5.0
