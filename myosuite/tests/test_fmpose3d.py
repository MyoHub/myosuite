# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""retarget_fmpose3d: synthetic FMPose3D poses of MyoLeg and MyoFullBody round-trip."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from myosuite.core.body_scaling import _assembled_qpos
from myosuite.core.trajectory_io import MotionClip
from myosuite.integrations.fmpose3d import (
    _LEVEL_CAMERA,
    _TRUNK_BACK,
    _TRUNK_HEIGHT,
    H36M_JOINTS,
    _joints_in,
    _with_sites,
    retarget_fmpose3d,
)
from myosuite.tests.support.optional_deps import require_musclemimic_models

pytestmark = pytest.mark.tier1

FPS = 30.0
Motion = dict[str, Callable[[np.ndarray], np.ndarray]]
# Camera-to-world rotation of a level camera in FMPose3D's camera frame
# (x right, y down, z forward), kept apart from the module's own constant.
LEVEL = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
# FMPose3D's demo camera-to-world rotation (wxyz), from
# fmpose3d/inference_api/fmpose3d.py (_DEFAULT_CAM_ROTATION).
FMPOSE3D_DEMO_ROTATION = np.array(
    [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
)
# Where FMPose3D puts Human3.6M's root and thorax, as (up the trunk, forward) in
# metres from the hip and shoulder midpoints. Measured on its demo runner: the
# root 9-22 mm up, the thorax (a neck-base point) 43-53 mm up and 24-28 mm forward.
H36M_ROOT = (0.015, 0.0)
H36M_THORAX = (0.045, 0.025)
HALF_SHOULDERS = 0.17  # for MyoLeg, which has no shoulder sites

# A stride-like swing of both legs, with the knees flexed so that hip rotation shows.
LEG_MOTION: Motion = {
    "hip_flexion_r": lambda t: 0.1 + 0.5 * np.sin(2 * np.pi * t),
    "knee_angle_r": lambda t: 0.05 + 0.5 * (1 - np.cos(2 * np.pi * t)),
    "hip_adduction_r": lambda t: 0.1 * np.sin(2 * np.pi * t),
    "hip_rotation_r": lambda t: 0.2 + 0 * t,
    "hip_flexion_l": lambda t: 0.1 - 0.4 * np.sin(2 * np.pi * t),
    "knee_angle_l": lambda t: 0.05 + 0.3 * (1 - np.cos(2 * np.pi * t + 1)),
}
# Arms raised in front with bent elbows. The elevation stays above 0.4 rad and
# the elbow flexion above 0.8 rad, where the plane of elevation and the humeral
# rotation are well defined, and the lumbar spine stays neutral (it is held).
ARM_MOTION: Motion = {
    "elv_angle_r": lambda t: 0.6 + 0 * t,
    "shoulder_elv_r": lambda t: 0.7 + 0.3 * np.sin(2 * np.pi * t),
    "shoulder_rot_r": lambda t: 0.3 + 0 * t,
    "elbow_flex_r": lambda t: 0.8 + 0.5 * (1 - np.cos(2 * np.pi * t)) / 2,
    "elv_angle_l": lambda t: 0.4 + 0 * t,
    "shoulder_elv_l": lambda t: 0.7 - 0.3 * np.sin(2 * np.pi * t),
    "elbow_flex_l": lambda t: 1.0 + 0 * t,
}


def _myoleg() -> mujoco.MjSpec:
    import myosuite  # noqa: F401, PLC0415  (registers the envs)
    from myosuite.utils import gym  # noqa: PLC0415
    from myosuite.utils.asset_path_resolver import resolve_model_xml_path  # noqa: PLC0415

    path = resolve_model_xml_path(gym.spec("myoLegWalk-v0").kwargs["model_path"])
    return mujoco.MjSpec.from_file(str(path))


def _fullbody() -> mujoco.MjSpec:
    require_musclemimic_models()
    from ml_collections import config_dict  # noqa: PLC0415

    from myosuite.integrations.musclemimic.fullbody_model import (  # noqa: PLC0415
        build_mimic_fullbody_spec,
        default_mimic_fullbody_config,
    )

    return build_mimic_fullbody_spec(
        config_dict.create(**dict(default_mimic_fullbody_config()))
    )[0]


def _coupled(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """*qpos* with every active joint coupling evaluated (chains included)."""
    q = qpos.copy()
    couplings = [
        e
        for e in range(model.neq)
        if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and model.eq_active0[e]
    ]
    for _ in range(len(couplings)):
        for e in couplings:
            first, second = model.eq_obj1id[e], model.eq_obj2id[e]
            offset = 0.0
            if second >= 0:
                address = model.jnt_qposadr[second]
                offset = q[address] - model.qpos0[address]
            q[model.jnt_qposadr[first]] = model.qpos0[
                model.jnt_qposadr[first]
            ] + np.polynomial.polynomial.polyval(offset, model.eq_data[e, :5])
    return q


def _motion(model: mujoco.MjModel, motion: Motion, frames: int = 19) -> np.ndarray:
    """Truth ``qpos`` per frame: the assembled seed with *motion* applied.

    19 frames put t = 1/3 and 2/3 on the grid, where the right knee is at
    0.8 rad and its coupled knee rotation leaves its own range.
    """
    t = np.linspace(0.0, 1.0, frames)
    qpos = np.repeat(_assembled_qpos(model)[None], frames, axis=0)
    for name, angle in motion.items():
        qpos[:, model.jnt_qposadr[model.joint(name).id]] = angle(t)
    return np.array([_coupled(model, q) for q in qpos])


def _fake_fmpose3d(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    camera: np.ndarray,
    widen: float = 0.055,
    stretch: float = 1.1,
) -> np.ndarray:
    """FMPose3D-like poses of *model*: root-relative, in *camera*'s frame.

    The root and thorax sit where FMPose3D puts them (``H36M_ROOT``,
    ``H36M_THORAX``), the hips (with their legs) sit *widen* further out, as
    Human3.6M's do, and every bone is *stretch* times as long. *camera* is the
    camera-to-world rotation.
    """
    data = mujoco.MjData(model)
    poses = np.full((len(qpos), len(H36M_JOINTS), 3), np.nan)
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        for index, joint in enumerate(H36M_JOINTS):
            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "h36m_" + joint)
            if site >= 0:
                poses[frame, index] = data.site_xpos[site]
    at = H36M_JOINTS.index
    left = poses[:, at("l_hip")] - poses[:, at("r_hip")]
    if np.isnan(poses[:, at("r_shoulder")]).all():  # MyoLeg: around its trunk point
        side = HALF_SHOULDERS * left / np.linalg.norm(left, axis=1, keepdims=True)
        poses[:, at("r_shoulder")] = poses[:, at("thorax")] - side
        poses[:, at("l_shoulder")] = poses[:, at("thorax")] + side
    hips = 0.5 * (poses[:, at("r_hip")] + poses[:, at("l_hip")])
    shoulders = 0.5 * (poses[:, at("r_shoulder")] + poses[:, at("l_shoulder")])
    up = shoulders - hips
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    forward = np.cross(left, up)
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    poses[:, at("hip")] = hips + H36M_ROOT[0] * up + H36M_ROOT[1] * forward
    poses[:, at("thorax")] = shoulders + H36M_THORAX[0] * up + H36M_THORAX[1] * forward
    poses -= poses[:, :1]
    lateral = (
        poses[:, H36M_JOINTS.index("l_hip")] - poses[:, H36M_JOINTS.index("r_hip")]
    )
    lateral /= np.linalg.norm(lateral, axis=1, keepdims=True)
    for side, sign in (("r", -1.0), ("l", 1.0)):
        for part in ("hip", "knee", "ankle"):
            poses[:, H36M_JOINTS.index(f"{side}_{part}")] += sign * widen * lateral
    return stretch * poses @ camera


def _turned_pitched_camera() -> np.ndarray:
    """Camera-to-world rotation of a camera turned 120 degrees and pitched 15 down."""
    yaw, pitch = np.deg2rad(120.0), np.deg2rad(-15.0)
    turn = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    tilt = np.array(
        [
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ]
    )
    return turn @ tilt @ LEVEL


def _root_tilt(model: mujoco.MjModel, qpos: np.ndarray, body: str) -> np.ndarray:
    """Angle (degrees) of *body*'s z-axis from the world z-axis, per frame."""
    data = mujoco.MjData(model)
    tilt = []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        tilt.append(data.xmat[model.body(body).id][8])
    return np.rad2deg(np.arccos(np.clip(tilt, -1.0, 1.0)))


def _assert_every_joint(
    model: mujoco.MjModel, clip: MotionClip, truth: np.ndarray, tol_deg: float
) -> None:
    """Every hinge joint comes back within *tol_deg*; held joints keep the seed."""
    for joint in range(model.njnt):
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE:
            address = model.jnt_qposadr[joint]
            np.testing.assert_allclose(
                clip.qpos[:, address],
                truth[:, address],
                atol=np.deg2rad(tol_deg),
                err_msg=model.joint(joint).name,
            )
    seed = _assembled_qpos(model)
    for name in clip.metadata["fmpose3d"]["held"]:
        address = model.jnt_qposadr[model.joint(name).id]
        np.testing.assert_allclose(
            clip.qpos[:, address], seed[address], rtol=0, atol=1e-9, err_msg=name
        )


def test_camera_frame_matches_fmpose3d() -> None:
    """FMPose3D's own camera-to-world rotation agrees with the level camera used here."""
    demo = np.zeros(9)
    mujoco.mju_quat2Mat(demo, FMPOSE3D_DEMO_ROTATION)
    demo = demo.reshape(3, 3)
    image_up, right = np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    assert (demo @ image_up)[2] > np.cos(np.deg2rad(15.0))  # pitched about 11 degrees
    assert abs((demo @ right)[2]) < 0.05
    np.testing.assert_allclose(_LEVEL_CAMERA @ image_up, [0.0, 0.0, 1.0])
    np.testing.assert_allclose((_LEVEL_CAMERA @ right)[2], 0.0)


def test_myoleg_angles_survive_the_camera_wide_hips_and_long_bones() -> None:
    spec = _myoleg()
    truth_model = _with_sites(spec, _joints_in(spec), None, None)
    truth = _motion(truth_model, LEG_MOTION)
    poses = _fake_fmpose3d(truth_model, truth, _turned_pitched_camera())
    model, clip = retarget_fmpose3d(spec, poses, fps=FPS)
    _assert_every_joint(model, clip, truth, tol_deg=0.5)
    # The 15 degree camera pitch is gone: the trunk leans as the model's does.
    assert _root_tilt(model, clip.qpos, "root").max() < 0.5
    np.testing.assert_allclose(clip.metadata["sole_height"], 0.0, atol=1e-9)
    meta = clip.metadata["fmpose3d"]
    assert meta["frame"] == "camera" and meta["valid_frames"].all()
    # No point moves the foot joints; the patella follows the free knee.
    assert set(meta["held"]) == {
        f"{joint}_{side}"
        for joint in ("ankle_angle", "subtalar_angle", "mtp_angle")
        for side in "rl"
    }
    assert set(meta["joints"]) == {
        "hip",
        "r_hip",
        "r_knee",
        "r_ankle",
        "l_hip",
        "l_knee",
        "l_ankle",
        "thorax",
    }
    assert model.nq == spec.compile().nq  # the sites leave the qpos layout alone


def test_myofullbody_arms_and_legs_round_trip() -> None:
    spec = _fullbody()
    truth_model = _with_sites(spec, _joints_in(spec), None, None)
    truth = _motion(truth_model, {**LEG_MOTION, **ARM_MOTION}, frames=12)
    poses = _fake_fmpose3d(truth_model, truth, _turned_pitched_camera())
    model, clip = retarget_fmpose3d(spec, poses, fps=FPS)
    _assert_every_joint(model, clip, truth, tol_deg=0.5)
    assert len(clip.metadata["fmpose3d"]["joints"]) == 14
    assert _root_tilt(model, clip.qpos, "Full Body").max() < 0.5
    held = set(clip.metadata["fmpose3d"]["held"])
    assert {"flex_extension", "L1_L2_FE", "pro_sup_r", "pro_sup_l"} <= held
    assert {"deviation_r", "flexion_l", "ankle_angle_l", "mtp_angle_r"} <= held
    assert not held & {*LEG_MOTION, *ARM_MOTION, "lat_bending", "axial_rotation"}


def test_myoleg_trunk_point_is_myofullbody_shoulder_midpoint() -> None:
    model = _fullbody().compile()
    data = mujoco.MjData(model)
    data.qpos[:] = _assembled_qpos(model)
    mujoco.mj_kinematics(model, data)
    x = {b: data.body(b).xpos for b in ("femur_r", "femur_l", "humerus_r", "humerus_l")}
    up = 0.5 * (x["humerus_r"] + x["humerus_l"]) - 0.5 * (x["femur_r"] + x["femur_l"])
    forward = np.cross(x["femur_l"] - x["femur_r"], [0.0, 0.0, 1.0])
    assert up[2] == pytest.approx(_TRUNK_HEIGHT, abs=1e-3)
    assert -up @ forward / np.linalg.norm(forward) == pytest.approx(
        _TRUNK_BACK, abs=1e-3
    )


def test_results_arrays_and_npz_files_agree(tmp_path: Path) -> None:
    spec = _myoleg()
    truth_model = _with_sites(spec, _joints_in(spec), None, None)
    poses = _fake_fmpose3d(truth_model, _motion(truth_model, LEG_MOTION, 6), LEVEL)
    valid = np.array([True, True, False, True, True, True])
    masked = np.where(valid[:, None, None], poses, np.nan)
    _, from_array = retarget_fmpose3d(spec, masked, fps=FPS)
    result = SimpleNamespace(poses_3d=poses, valid_frames_mask=valid)  # a Pose3DResult
    _, from_result = retarget_fmpose3d(spec, result, fps=FPS)
    np.savez(tmp_path / "walk.npz", poses_3d=poses, valid_frames_mask=valid, fps=FPS)
    _, from_file = retarget_fmpose3d(spec, tmp_path / "walk.npz")
    for clip in (from_result, from_file):
        np.testing.assert_allclose(clip.qpos, from_array.qpos)
    assert from_file.frequency_hz == FPS and not from_file.metadata["solved"][2]
    np.testing.assert_allclose(
        from_file.qpos[2], from_file.qpos[1]
    )  # gap holds the pose
    # A level camera needs no vertical estimate: its own frame keeps the body upright.
    model, level = retarget_fmpose3d(spec, result, fps=FPS, upright=False)
    assert _root_tilt(model, level.qpos, "root").max() < 0.5


def test_root_and_thorax_joints_are_not_used() -> None:
    """FMPose3D synthesises them; the hip and shoulder midpoints stand in."""
    spec = _myoleg()
    truth_model = _with_sites(spec, _joints_in(spec), None, None)
    poses = _fake_fmpose3d(truth_model, _motion(truth_model, LEG_MOTION, 4), LEVEL)
    _, clip = retarget_fmpose3d(spec, poses, fps=FPS)
    blanked = poses.copy()
    blanked[:, [H36M_JOINTS.index("hip"), H36M_JOINTS.index("thorax")]] = np.nan
    _, same = retarget_fmpose3d(spec, blanked, fps=FPS)
    np.testing.assert_array_equal(same.qpos, clip.qpos)


def test_world_frame_poses_from_the_demo(tmp_path: Path) -> None:
    """The demo saves one ``pose3d`` per frame in FMPose3D's z-up world frame."""
    spec = _myoleg()
    truth_model = _with_sites(spec, _joints_in(spec), None, None)
    truth = _motion(truth_model, LEG_MOTION, 3)[1:2]
    np.savez(
        tmp_path / "0000_3D.npz",
        pose3d=_fake_fmpose3d(truth_model, truth, np.eye(3))[0],
    )
    model, clip = retarget_fmpose3d(spec, tmp_path / "0000_3D.npz", upright=False)
    assert clip.metadata["fmpose3d"]["frame"] == "world" and len(clip.qpos) == 1
    _assert_every_joint(model, clip, truth, tol_deg=0.5)


def test_subject_scales_are_applied_first() -> None:
    spec = _myoleg()
    nominal = _with_sites(spec, _joints_in(spec), None, None)
    truth = _motion(nominal, LEG_MOTION)
    poses = _fake_fmpose3d(nominal, truth, LEVEL)
    model, clip = retarget_fmpose3d(
        spec, poses, fps=FPS, scales={"femur_r": 1.1, "femur_l": 1.1}
    )

    def thigh(m: mujoco.MjModel) -> float:
        data = mujoco.MjData(m)
        data.qpos[:] = _assembled_qpos(m)
        mujoco.mj_kinematics(m, data)
        return float(
            np.linalg.norm(data.site("h36m_r_knee").xpos - data.site("h36m_r_hip").xpos)
        )

    assert thigh(model) == pytest.approx(1.1 * thigh(nominal), rel=1e-9)
    _assert_every_joint(model, clip, truth, tol_deg=0.5)


def test_bad_inputs_are_refused(tmp_path: Path) -> None:
    spec = _myoleg()
    with pytest.raises(ValueError, match="26-joint animal"):
        retarget_fmpose3d(spec, np.zeros((3, 26, 3)), fps=FPS)
    pelvis_only = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><body name="pelvis"><freejoint/><geom size=".1"/>'
        "</body></worldbody></mujoco>"
    )
    with pytest.raises(ValueError, match="MyoLeg"):
        retarget_fmpose3d(pelvis_only, np.zeros((17, 3)))
    truth_model = _with_sites(spec, _joints_in(spec), None, None)
    poses = _fake_fmpose3d(truth_model, _motion(truth_model, LEG_MOTION, 2), LEVEL)
    no_shoulders = poses.copy()
    no_shoulders[
        :, [H36M_JOINTS.index("r_shoulder"), H36M_JOINTS.index("l_shoulder")]
    ] = np.nan
    with pytest.raises(ValueError, match="upright=False"):
        retarget_fmpose3d(spec, no_shoulders, fps=FPS)
    with pytest.raises(ValueError, match="valid_frames_mask"):
        retarget_fmpose3d(
            spec, SimpleNamespace(poses_3d=poses, valid_frames_mask=np.ones(3, bool))
        )
    with pytest.raises(ValueError, match='"camera" or "world"'):
        retarget_fmpose3d(spec, poses, fps=FPS, frame="lab")
    np.savez(tmp_path / "other.npz", keypoints=np.zeros((2, 17, 2)))
    with pytest.raises(ValueError, match="poses_3d"):
        retarget_fmpose3d(spec, tmp_path / "other.npz")
