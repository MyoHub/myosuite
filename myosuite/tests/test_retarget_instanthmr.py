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


def _closed_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return float(abs(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum()) / 6.0)


def _write_test_muscle_glb(path: Path, ref_model: mujoco.MjModel) -> Path:
    import struct

    from myosuite.viz.skin_glb import _TO_GLTF

    d = mujoco.MjData(ref_model)
    d.qpos[:] = ref_model.qpos0
    mujoco.mj_forward(ref_model, d)

    def bone_geom(body_name: str) -> int:
        bid = ref_model.body(body_name).id
        return next(
            g
            for g in range(ref_model.ngeom)
            if ref_model.geom_bodyid[g] == bid
            and ref_model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        )

    bone_bodies = ("femur_r", "tibia_r", "humerus_r", "ulna_r")
    g_ids = [bone_geom(b) for b in bone_bodies]
    joint_pos_mj = np.stack([d.geom_xpos[g] for g in g_ids], axis=0)
    joint_pos_gltf = joint_pos_mj @ _TO_GLTF.T

    blob = bytearray()
    views: list[dict[str, object]] = []
    accessors: list[dict[str, object]] = []

    def add(arr: np.ndarray, kind: str, comp: int) -> int:
        arr = np.ascontiguousarray(arr)
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": arr.nbytes})
        blob.extend(arr.tobytes())
        acc: dict[str, object] = {
            "bufferView": len(views) - 1,
            "componentType": comp,
            "count": int(len(arr)),
            "type": kind,
        }
        if kind == "VEC3" and comp == 5126:
            acc["min"] = arr.min(0).tolist()
            acc["max"] = arr.max(0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    def prism_between(
        p0: np.ndarray, p1: np.ndarray, j0: int, j1: int
    ) -> dict[str, object]:
        verts_mj = np.array(
            [
                p0 + [0.01, 0.0, 0.0],
                p0 + [-0.005, 0.009, 0.0],
                p0 + [-0.005, -0.009, 0.0],
                p1 + [0.01, 0.0, 0.0],
                p1 + [-0.005, 0.009, 0.0],
                p1 + [-0.005, -0.009, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [
                [0, 2, 1],
                [3, 4, 5],
                [0, 1, 4],
                [0, 4, 3],
                [1, 2, 5],
                [1, 5, 4],
                [2, 0, 3],
                [2, 3, 5],
            ],
            dtype=np.uint32,
        )
        jnts = np.array(
            [[j0, j1, 0, 0]] * 6,
            dtype=np.uint16,
        )
        wts = np.array(
            [
                [0.85, 0.15, 0.0, 0.0],
                [0.85, 0.15, 0.0, 0.0],
                [0.85, 0.15, 0.0, 0.0],
                [0.15, 0.85, 0.0, 0.0],
                [0.15, 0.85, 0.0, 0.0],
                [0.15, 0.85, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        return {
            "attributes": {
                "POSITION": add(
                    (verts_mj @ _TO_GLTF.T).astype(np.float32), "VEC3", 5126
                ),
                "JOINTS_0": add(jnts, "VEC4", 5123),
                "WEIGHTS_0": add(wts, "VEC4", 5126),
            },
            "indices": add(faces.ravel(), "SCALAR", 5125),
        }

    prim_leg = prism_between(joint_pos_mj[0], joint_pos_mj[1], 0, 1)
    prim_arm = prism_between(joint_pos_mj[2], joint_pos_mj[3], 2, 3)

    nodes: list[dict[str, object]] = [
        {"name": f"bone_{b}", "translation": joint_pos_gltf[i].tolist()}
        for i, b in enumerate(bone_bodies)
    ]
    nodes.append({"name": "Armature", "children": [0, 1, 2, 3]})
    nodes.append({"name": "recfem_r", "mesh": 0, "skin": 0})
    nodes.append({"name": "BIClong_r", "mesh": 1, "skin": 0})

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [4, 5, 6]}],
        "nodes": nodes,
        "meshes": [
            {"name": "recfem_r", "primitives": [prim_leg]},
            {"name": "BIClong_r", "primitives": [prim_arm]},
        ],
        "skins": [{"name": "Rig", "joints": [0, 1, 2, 3], "skeleton": 4}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }
    txt = json.dumps(gltf).encode("utf-8")
    txt += b" " * (-len(txt) % 4)
    blob += b"\0" * (-len(blob) % 4)
    raw = (
        struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(txt) + 8 + len(blob))
        + struct.pack("<I4s", len(txt), b"JSON")
        + txt
        + struct.pack("<I4s", len(blob), b"BIN\0")
        + bytes(blob)
    )
    path.write_bytes(raw)
    return path


def test_subject_scaling_invariants_fullbody_and_myolegs(
    fullbody: mujoco.MjModel,
) -> None:
    from myosuite.envs.myo.tasks.mimic.tracking.models import model_cfg
    from myosuite.viz.muscle_tubes import muscle_actuators

    spec_fb = model_cfg("myofullbody").build_spec(True)
    gt_scales = {
        "pelvis": 1.08,
        "sacrum": 1.12,
        "clavicle_l": 1.05,
        "clavicle_r": 1.05,
        "femur_l": 1.18,
        "femur_r": 1.18,
        "tibia_l": 1.14,
        "tibia_r": 1.14,
        "talus_l": 1.10,
        "talus_r": 1.10,
        "humerus_l": 1.15,
        "humerus_r": 1.15,
        "ulna_l": 1.11,
        "ulna_r": 1.11,
        "lunate_l": 1.06,
        "lunate_r": 1.06,
    }
    gt_model = ri.scale_spec_from_instanthmr(
        spec_fb, gt_scales, body_mass=82.5
    ).compile()

    cal_gt = ri.calibrate(gt_model, ri.INSTANTHMR_MYOFULLBODY)
    q_gt = np.tile(cal_gt.qpos, (10, 1))
    j_cam = _synthesize_mhr70_cam(gt_model, ri.INSTANTHMR_MYOFULLBODY, q_gt)
    clip_dict = {"joints_3d_cam": j_cam, "fps": 30.0}

    est = ri.estimate_instanthmr_scales(clip_dict, fullbody, ri.INSTANTHMR_MYOFULLBODY)
    for key, val in gt_scales.items():
        assert est[key] == pytest.approx(val, abs=1e-4), f"{key}: {est[key]} != {val}"

    m_scaled, _, _ = ri.scale_model_from_instanthmr(
        spec_fb, clip_dict, ri.INSTANTHMR_MYOFULLBODY, body_mass=82.5
    )
    assert m_scaled.body_mass[1:].sum() == pytest.approx(82.5, rel=1e-6)
    scaphoid_id = m_scaled.body("scaphoid_r").id
    assert np.all(m_scaled.body_inertia[scaphoid_id] > 0)

    # Muscle peak force stays constant and normalized resting tendon length is preserved.
    muscles = muscle_actuators(fullbody)
    assert np.allclose(
        m_scaled.actuator_gainprm[muscles, 2],
        fullbody.actuator_gainprm[muscles, 2],
        rtol=1e-5,
    )
    q0 = fullbody.qpos0.copy()
    project_joint_couplings(JointCouplings.from_model(fullbody), q0)
    d0 = mujoco.MjData(fullbody)
    d0.qpos[:] = q0
    mujoco.mj_forward(fullbody, d0)

    qs = m_scaled.qpos0.copy()
    project_joint_couplings(JointCouplings.from_model(m_scaled), qs)
    ds = mujoco.MjData(m_scaled)
    ds.qpos[:] = qs
    mujoco.mj_forward(m_scaled, ds)

    tids = fullbody.actuator_trnid[muscles, 0]
    lr0 = fullbody.actuator_lengthrange[muscles]
    lrs = m_scaled.actuator_lengthrange[muscles]
    norm0 = (d0.ten_length[tids] - lr0[:, 0]) / (lr0[:, 1] - lr0[:, 0])
    norms = (ds.ten_length[tids] - lrs[:, 0]) / (lrs[:, 1] - lrs[:, 0])
    assert np.allclose(norm0, norms, atol=1e-6)

    # Sole floor height at assembled qpos0 matches nominal model.
    assert ri._sole_lowest_z(m_scaled, qs) == pytest.approx(
        ri._sole_lowest_z(fullbody, q0), abs=1e-6
    )

    # Polynomial knee coupling residual remains zero across flexion angles.
    couplings = JointCouplings.from_model(m_scaled)
    q_flex = qs.copy()
    q_flex[m_scaled.jnt_qposadr[m_scaled.joint("knee_angle_r").id]] = 1.2
    project_joint_couplings(couplings, q_flex)
    q_check = q_flex.copy()
    project_joint_couplings(couplings, q_check)
    assert np.max(np.abs(q_check - q_flex)) < 1e-12

    # Also verify scaling on myoLegWalk-v0 MjSpec (keyframes + 80 leg muscles).
    env = make_env("myoLegWalk-v0")
    try:
        spec_leg = env.unwrapped._mj_spec.copy()
        m_leg0 = env.unwrapped.model.__copy__()
    finally:
        env.close()
    cal_leg = ri.calibrate(m_leg0, ri.INSTANTHMR_MYOLEGS)
    j_leg = _synthesize_mhr70_cam(
        m_leg0,
        ri.INSTANTHMR_MYOLEGS,
        np.tile(cal_leg.qpos, (8, 1)),
        subject_scale=1.12,
    )
    m_leg_s, _, leg_scales = ri.scale_model_from_instanthmr(
        spec_leg, {"joints_3d_cam": j_leg, "fps": 30.0}, body_mass=76.0
    )
    assert leg_scales["femur_r"] == pytest.approx(1.12, abs=1e-4)
    assert leg_scales["tibia_l"] == pytest.approx(1.12, abs=1e-4)
    assert m_leg_s.body_mass[1:].sum() == pytest.approx(76.0, rel=1e-6)


def test_solve_instanthmr_ik_metric_scaled_roundtrip() -> None:
    from myosuite.core.trajectory_io import expand_motion_clip_to_model
    from myosuite.envs.myo.tasks.mimic.tracking.models import model_cfg

    spec = model_cfg("myofullbody").build_spec(True)
    subject_scale = 1.15
    gt_model = ri.scale_spec_from_instanthmr(
        spec,
        {
            k: subject_scale
            for k in (
                "pelvis",
                "sacrum",
                "clavicle_l",
                "clavicle_r",
                "femur_l",
                "femur_r",
                "tibia_l",
                "tibia_r",
                "talus_l",
                "talus_r",
                "humerus_l",
                "humerus_r",
                "ulna_l",
                "ulna_r",
                "lunate_l",
                "lunate_r",
            )
        },
    ).compile()
    cal = ri.calibrate(gt_model, ri.INSTANTHMR_MYOFULLBODY)

    T = 25
    t = np.linspace(0.0, 0.4, T)
    q_gt = np.tile(cal.qpos, (T, 1))
    q_gt[:, 0] = 0.55 * (t / t[-1])
    for jn, amp in (
        ("hip_flexion_r", 0.3),
        ("knee_angle_r", 0.45),
        ("elbow_flexion_l", 0.5),
    ):
        q_gt[:, gt_model.jnt_qposadr[gt_model.joint(jn).id]] = amp * np.sin(
            np.pi * t / t[-1]
        )
    project_joint_couplings(JointCouplings.from_model(gt_model), q_gt)
    low0 = rt._foot_kinematics(gt_model, cal.lmap, q_gt)[0].min()
    q_gt[:, 2] -= low0

    j_cam = _synthesize_mhr70_cam(gt_model, ri.INSTANTHMR_MYOFULLBODY, q_gt)
    res = ri.solve_instanthmr_ik(
        {"joints_3d_cam": j_cam, "fps": 60.0},
        spec=spec,
        scale=True,
        body_mass=80.0,
        target_sigma=0.0,
    )
    assert res.scales is not None
    # Knee condyle slide coupling shifts tibia origin ~1.4 mm over 0.45 rad flexion.
    assert res.scales["femur_r"] == pytest.approx(subject_scale, abs=3e-3)
    assert res.metrics["root_scale"] == pytest.approx(1.0)
    assert res.metrics["landmark_rmse_mm"] < 5.0
    assert res.metrics["coupling_residual"] < 1e-9
    # Metric 0.55 m forward progression preserved 1:1 on the scaled subject.
    dx_fit = float(res.qpos[-1, 0] - res.qpos[0, 0])
    assert dx_fit == pytest.approx(0.55, abs=0.015)
    assert res.qvel.shape == (T, res.model.nv)
    assert res.qacc.shape == (T, res.model.nv)
    assert res.landmark_error.shape == (T, len(cal.lmap.landmarks))
    expanded = expand_motion_clip_to_model(res.clip, res.model)
    assert np.allclose(expanded.qpos, res.qpos)


def test_reconstruct_muscle_meshes_volumetric_static_opt_and_glb(
    tmp_path: Path,
) -> None:
    import struct

    env = make_env("myoLegWalk-v0")
    try:
        spec_leg = env.unwrapped._mj_spec.copy()
        m_leg = env.unwrapped.model.__copy__()
    finally:
        env.close()

    cal = ri.calibrate(m_leg, ri.INSTANTHMR_MYOLEGS)
    T = 12
    t = np.linspace(0.0, 0.35, T)
    q_gt = np.tile(cal.qpos, (T, 1))
    q_gt[:, m_leg.jnt_qposadr[m_leg.joint("knee_angle_r").id]] = 0.8 * np.sin(
        np.pi * t / t[-1]
    )
    q_gt[:, m_leg.jnt_qposadr[m_leg.joint("hip_flexion_r").id]] = 0.4 * np.sin(
        np.pi * t / t[-1]
    )
    project_joint_couplings(JointCouplings.from_model(m_leg), q_gt)
    low0 = rt._foot_kinematics(m_leg, cal.lmap, q_gt)[0].min()
    q_gt[:, 2] -= low0

    j_cam = _synthesize_mhr70_cam(m_leg, ri.INSTANTHMR_MYOLEGS, q_gt, subject_scale=1.1)
    recon = ri.reconstruct_from_instanthmr(
        {"joints_3d_cam": j_cam, "fps": 30.0},
        spec=spec_leg,
        scale=True,
        activation_mode="static_optimization",
        mesh_mode="volumetric",
        target_sigma=0.0,
    )
    mus = recon.muscles
    assert mus.num_muscles == 80
    assert isinstance(mus.vertices, np.ndarray)
    assert mus.vertices.shape[:2] == (T, 80)
    assert mus.faces.ndim == 2 and mus.faces.shape[1] == 3
    assert np.isfinite(mus.activation).all()
    assert float(mus.activation.min()) >= 0.0
    assert float(mus.activation.max()) <= 1.0
    assert float(mus.activation.max()) > 0.01

    # Constant-volume muscle belly preservation across flexion.
    rf_idx = mus.muscle_names.index("recfem_r")
    v_f0 = _closed_mesh_volume(mus.muscle_vertices(rf_idx)[0], mus.muscle_faces(rf_idx))
    v_mid = _closed_mesh_volume(
        mus.muscle_vertices(rf_idx)[T // 2], mus.muscle_faces(rf_idx)
    )
    assert v_f0 > 0.0
    assert v_mid == pytest.approx(v_f0, rel=0.03)

    glb_path = mus.write_glb(
        tmp_path / "leg_muscles.glb",
        frame=T // 2,
        model=recon.model,
        qpos=recon.ik.qpos,
        include_bones=True,
    )
    buf = glb_path.read_bytes()
    magic, ver, total_len = struct.unpack_from("<4sII", buf)
    assert magic == b"glTF" and ver == 2 and total_len == len(buf)
    jlen, jtype = struct.unpack_from("<I4s", buf, 12)
    assert jtype == b"JSON"
    gltf = json.loads(buf[20 : 20 + jlen])
    node_names = {n.get("name") for n in gltf["nodes"]}
    assert {"Muscles", "Bones", "recfem_r"} <= node_names
    assert len(gltf["materials"]) >= 81

    npz_path = mus.save_npz(tmp_path / "leg_muscles.npz")
    with np.load(npz_path, allow_pickle=False) as d:
        assert d["vertices"].shape == mus.vertices.shape
        assert d["activation"].shape == (T, 80)
        assert list(d["muscle_names"])[rf_idx] == "recfem_r"


def test_reconstruct_skinned_glb_muscle_mesh_with_scaling(
    fullbody: mujoco.MjModel, myolegs: mujoco.MjModel, tmp_path: Path
) -> None:
    from myosuite.envs.myo.tasks.mimic.tracking.models import model_cfg
    from myosuite.viz import blender_render as br
    from myosuite.viz.muscle_meshes import reconstruct_muscle_meshes

    glb_path = _write_test_muscle_glb(tmp_path / "test_atlas.glb", fullbody)

    # 1. Scaled myofullbody (1.2x): both leg & arm muscles bind, scale by 1.2x, and deform.
    spec_fb = model_cfg("myofullbody").build_spec(True)
    m_scaled = ri.scale_spec_from_instanthmr(
        spec_fb,
        {
            "pelvis": 1.2,
            "sacrum": 1.2,
            "femur_r": 1.2,
            "tibia_r": 1.2,
            "humerus_r": 1.2,
            "ulna_r": 1.2,
        },
    ).compile()
    q = np.tile(m_scaled.qpos0, (5, 1))
    q[:, m_scaled.jnt_qposadr[m_scaled.joint("elbow_flexion_r").id]] = np.linspace(
        0.0, 1.0, 5
    )
    project_joint_couplings(JointCouplings.from_model(m_scaled), q)

    recon_fb = reconstruct_muscle_meshes(
        m_scaled,
        q,
        fps=30.0,
        activation_mode="shortening",
        mesh_mode=glb_path,
        reference_model=fullbody,
    )
    assert recon_fb.muscle_names == ("recfem_r", "BIClong_r")
    v_bic = recon_fb.muscle_vertices("BIClong_r")
    assert v_bic.shape == (5, 6, 3)
    assert not np.allclose(v_bic[0], v_bic[-1])

    recon_unscaled = reconstruct_muscle_meshes(
        fullbody,
        fullbody.qpos0[None, :],
        fps=30.0,
        activation_mode="none",
        mesh_mode=glb_path,
        reference_model=fullbody,
    )
    span_unscaled = float(
        np.ptp(recon_unscaled.muscle_vertices("recfem_r")[0], axis=0).max()
    )
    span_scaled = float(np.ptp(recon_fb.muscle_vertices("recfem_r")[0], axis=0).max())
    assert span_scaled == pytest.approx(1.2 * span_unscaled, rel=0.02)

    # Verify blender_render._muscle_rigs uses unscaled reference when model is scaled.
    cfg = br.RenderConfig(
        env="myoMimicFullbody-v0", output=tmp_path / "br_out", muscle_mesh="atlas"
    )
    bone_geoms = {
        m_scaled.mesh(m_scaled.geom_dataid[g]).name: g
        for g in range(m_scaled.ngeom)
        if m_scaled.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        and m_scaled.geom_dataid[g] >= 0
    }
    rigs = br._muscle_rigs(cfg, m_scaled, bone_geoms)
    assert len(rigs) >= 1

    # 2. Partial skeleton (myolegs): leg muscle binds, arm muscle is cleanly omitted.
    recon_leg = reconstruct_muscle_meshes(
        myolegs,
        np.tile(myolegs.qpos0, (4, 1)),
        fps=30.0,
        activation_mode="none",
        mesh_mode=glb_path,
        reference_model=fullbody,
    )
    assert recon_leg.muscle_names == ("recfem_r",)


def test_cli_scale_and_muscle_export(myolegs: mujoco.MjModel, tmp_path: Path) -> None:
    from scripts.mimic import retarget_instanthmr as cli

    cal = ri.calibrate(myolegs, ri.INSTANTHMR_MYOLEGS)
    T = 10
    q_gt = np.tile(cal.qpos, (T, 1))
    q_gt[:, 0] = np.linspace(0.0, 0.2, T)
    low0 = rt._foot_kinematics(myolegs, cal.lmap, q_gt)[0].min()
    q_gt[:, 2] -= low0

    clip_in = tmp_path / "subj_walk.npz"
    np.savez(
        clip_in,
        joints_3d_cam=_synthesize_mhr70_cam(
            myolegs, ri.INSTANTHMR_MYOLEGS, q_gt, subject_scale=1.08
        ),
        fps=30.0,
    )
    out_root = tmp_path / "scaled_out"
    rc = cli.main(
        [
            str(clip_in),
            "--model",
            "myolegs",
            "--out-root",
            str(out_root),
            "--dataset",
            "subj01",
            "--scale",
            "--body-mass",
            "78.0",
            "--reconstruct-muscles",
            "volumetric",
            "--activation",
            "shortening",
            "--export-glb",
        ]
    )
    assert rc == 0
    ds_dir = out_root / "subj01"
    assert (ds_dir / "subject_scales.json").is_file()
    assert (ds_dir / "subj_walk.npz").is_file()
    assert (ds_dir / "subj_walk_muscles.npz").is_file()
    assert (ds_dir / "subj_walk_muscles.glb").is_file()

    meta = rt.read_refit_meta(ds_dir / "subj_walk.npz")
    assert meta is not None
    assert meta["scaled"] is True
    assert meta["subject_body_mass_kg"] == 78.0
    assert meta["muscle_mesh_mode"] == "volumetric"
    assert meta["muscle_glb"] == "subj_walk_muscles.glb"
