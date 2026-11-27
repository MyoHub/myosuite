# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for HOI clip transforms and mink-based contact-aware MuJoCo IK."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from myosuite.biomechanics.hand_retarget import preset_for_model
from myosuite.biomechanics.hoi_retarget import (
    HOIClip,
    HoiPhysicsIKConfig,
    mirror_hoi_clip_lr,
    resample_hoi_clip,
    retarget_hoi_mujoco,
    rotate_hoi_clip,
    translate_hoi_clip,
    trim_hoi_clip,
)
from myosuite.core.model_builder import ModelBuilder
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings
from myosuite.physics.quat import get_quat_ops

pytestmark = pytest.mark.tier1


def _sample_hoi_clip(n_frames: int = 6, with_object: bool = True) -> HOIClip:
    t = np.linspace(0.0, 1.0, n_frames)
    wrist_pos = np.stack([0.2 * t, -0.1 + 0.3 * t, 0.4 + 0.05 * t], axis=-1)
    wrist_rot = Rotation.from_rotvec(
        np.outer(t, np.array([0.3, -0.2, 0.5]))
    ).as_matrix()
    local_kp = np.linspace(-0.08, 0.08, 21 * 3).reshape(1, 21, 3)
    keypoints = wrist_pos[:, None, :] + np.einsum("tab,tkb->tka", wrist_rot, local_kp)
    if not with_object:
        return HOIClip(
            fps=30.0,
            keypoints=keypoints,
            wrist_pos=wrist_pos,
            wrist_rot=wrist_rot,
            side="right",
        )
    obj_pos = wrist_pos + np.array([0.05, -0.02, 0.03])
    obj_xyzw = Rotation.from_rotvec(
        np.outer(t, np.array([0.0, 0.0, np.pi / 2.0]))
    ).as_quat()
    obj_wxyz = obj_xyzw[:, [3, 0, 1, 2]]
    return HOIClip(
        fps=30.0,
        keypoints=keypoints,
        wrist_pos=wrist_pos,
        wrist_rot=wrist_rot,
        side="right",
        object_pos=obj_pos,
        object_quat=obj_wxyz,
    )


def test_translate_hoi_clip() -> None:
    clip = _sample_hoi_clip(with_object=True)
    offset = np.array([0.15, -0.25, 0.35])
    moved = translate_hoi_clip(clip, offset)

    np.testing.assert_allclose(moved.wrist_pos, clip.wrist_pos + offset, atol=1e-12)
    np.testing.assert_allclose(
        moved.keypoints, clip.keypoints + offset[None, None, :], atol=1e-12
    )
    assert moved.object_pos is not None
    assert clip.object_pos is not None
    np.testing.assert_allclose(moved.object_pos, clip.object_pos + offset, atol=1e-12)
    np.testing.assert_allclose(moved.wrist_rot, clip.wrist_rot, atol=1e-12)

    no_obj = translate_hoi_clip(_sample_hoi_clip(with_object=False), offset)
    assert no_obj.object_pos is None
    assert no_obj.object_quat is None


def test_rotate_hoi_clip_preserves_rigid_geometry_and_quats() -> None:
    clip = _sample_hoi_clip(with_object=True)
    assert rotate_hoi_clip(clip, angle_deg=0.0) is clip
    with pytest.raises(ValueError, match="non-zero rotation axis"):
        rotate_hoi_clip(clip, axis=(0.0, 0.0, 0.0), angle_deg=45.0)

    axis = np.array([1.0, 2.0, -1.5])
    angle_deg = 72.0
    rot_clip = rotate_hoi_clip(clip, axis=axis, angle_deg=angle_deg)

    r_expected = Rotation.from_rotvec(
        axis / np.linalg.norm(axis) * np.radians(angle_deg)
    ).as_matrix()
    np.testing.assert_allclose(
        rot_clip.wrist_pos, clip.wrist_pos @ r_expected.T, atol=1e-12
    )
    np.testing.assert_allclose(
        rot_clip.keypoints, clip.keypoints @ r_expected.T, atol=1e-12
    )
    np.testing.assert_allclose(
        np.linalg.det(rot_clip.wrist_rot), np.ones(len(clip.keypoints)), atol=1e-12
    )

    rel_before = np.einsum(
        "tba,tkb->tka", clip.wrist_rot, clip.keypoints - clip.wrist_pos[:, None, :]
    )
    rel_after = np.einsum(
        "tba,tkb->tka",
        rot_clip.wrist_rot,
        rot_clip.keypoints - rot_clip.wrist_pos[:, None, :],
    )
    np.testing.assert_allclose(rel_after, rel_before, atol=1e-12)

    qm = get_quat_ops(np)
    assert rot_clip.object_quat is not None
    assert clip.object_quat is not None
    np.testing.assert_allclose(
        qm.quat2mat(rot_clip.object_quat),
        np.einsum("ab,tbc->tac", r_expected, qm.quat2mat(clip.object_quat)),
        atol=1e-12,
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_mirror_hoi_clip_lr_so3_and_roundtrip(axis: int) -> None:
    clip = _sample_hoi_clip(with_object=True)
    mir = mirror_hoi_clip_lr(clip, axis=axis)
    assert mir.side == "left"
    np.testing.assert_allclose(
        np.linalg.det(mir.wrist_rot), np.ones(len(clip.keypoints)), atol=1e-12
    )
    assert mir.wrist_pos[1, axis] == pytest.approx(-clip.wrist_pos[1, axis])
    for other in (0, 1, 2):
        if other != axis:
            assert mir.wrist_pos[1, other] == pytest.approx(clip.wrist_pos[1, other])

    qm = get_quat_ops(np)
    assert mir.object_quat is not None
    assert clip.object_quat is not None
    m_world = np.ones(3)
    m_world[axis] = -1.0
    m_diag = np.diag(m_world)
    np.testing.assert_allclose(
        qm.quat2mat(mir.object_quat),
        np.einsum("ab,tbc,cd->tad", m_diag, qm.quat2mat(clip.object_quat), m_diag),
        atol=1e-12,
    )

    roundtrip = mirror_hoi_clip_lr(mir, axis=axis)
    assert roundtrip.side == "right"
    np.testing.assert_allclose(roundtrip.keypoints, clip.keypoints, atol=1e-12)
    np.testing.assert_allclose(roundtrip.wrist_pos, clip.wrist_pos, atol=1e-12)
    np.testing.assert_allclose(roundtrip.wrist_rot, clip.wrist_rot, atol=1e-12)


def test_mirror_hoi_clip_invalid_axis() -> None:
    clip = _sample_hoi_clip(with_object=False)
    with pytest.raises(ValueError, match="axis must be 0, 1, or 2"):
        mirror_hoi_clip_lr(clip, axis=3)


def test_trim_hoi_clip_slices_and_copies() -> None:
    clip = _sample_hoi_clip(n_frames=8, with_object=True)
    trimmed = trim_hoi_clip(clip, start=2, end=6)
    assert len(trimmed.keypoints) == 4
    np.testing.assert_allclose(trimmed.wrist_pos, clip.wrist_pos[2:6], atol=1e-12)
    assert trimmed.object_pos is not None
    assert clip.object_pos is not None
    np.testing.assert_allclose(trimmed.object_pos, clip.object_pos[2:6], atol=1e-12)

    trimmed.wrist_pos[0] = 999.0
    assert clip.wrist_pos[2, 0] != 999.0


def test_resample_hoi_clip_slerp_and_speed_scaling() -> None:
    clip = _sample_hoi_clip(n_frames=5, with_object=True)
    assert clip.object_quat is not None
    # Flip sign on alternating quaternions to verify antipodal hemisphere safety.
    clip.object_quat[1::2] *= -1.0

    with pytest.raises(ValueError, match="speed_scale must be positive"):
        resample_hoi_clip(clip, speed_scale=0.0)

    upsampled = resample_hoi_clip(clip, speed_scale=1.0, out_fps=60.0)
    assert upsampled.fps == pytest.approx(60.0)
    assert len(upsampled.keypoints) == 9
    # Midpoint of 0 -> 90 deg Z rotation must be 45 deg Z rotation.
    assert upsampled.object_quat is not None
    qm = get_quat_ops(np)
    mid_mat = qm.quat2mat(upsampled.object_quat[4])
    expected_mid = Rotation.from_rotvec([0.0, 0.0, np.pi / 4.0]).as_matrix()
    np.testing.assert_allclose(mid_mat, expected_mid, atol=1e-12)

    fast = resample_hoi_clip(clip, speed_scale=2.0, out_fps=30.0)
    assert len(fast.keypoints) == 3
    np.testing.assert_allclose(fast.wrist_pos[-1], clip.wrist_pos[-1], atol=1e-12)

    single = trim_hoi_clip(clip, 0, 1)
    single_out = resample_hoi_clip(single, out_fps=120.0)
    assert len(single_out.keypoints) == 1
    assert single_out.fps == pytest.approx(120.0)


def test_retarget_hoi_mujoco_tracks_tips_and_resolves_object_contact() -> None:
    model, spec = ModelBuilder().attach_fragment("hand").build()
    preset = preset_for_model(model)
    couplings = JointCouplings.from_model(model)

    q_goal = model.qpos0.copy()
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mcp2_flexion_r")
    q_goal[model.jnt_qposadr[jid]] = 0.55
    project_joint_couplings(couplings, q_goal)
    data = mujoco.MjData(model)
    data.qpos[:] = q_goal
    mujoco.mj_kinematics(model, data)
    tips_goal = np.stack(
        [
            data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)]
            for s in preset.tip_sites
        ],
        axis=0,
    )
    tip_seq = np.repeat(tips_goal[None, :, :], 4, axis=0)

    res = retarget_hoi_mujoco(
        spec,
        tip_seq,
        cfg=HoiPhysicsIKConfig(settle_steps=0, smooth_window=1),
    )
    assert res["qpos"].shape == (4, model.nq)
    assert res["tip_error_mean_m"] < 0.005

    # Add a free sphere colliding with the index fingertip target and verify settle_steps pushes out.
    hoi_spec = spec.copy()
    obj_body = hoi_spec.worldbody.add_body(name="grasped_ball", pos=tips_goal[1])
    obj_body.add_freejoint(name="grasped_ball_free")
    obj_body.add_geom(
        name="grasped_ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.028, 0.0, 0.0],
        mass=0.1,
        contype=1,
        conaffinity=1,
    )
    hoi_model = hoi_spec.compile()
    obj_poses = np.tile(np.concatenate([tips_goal[1], [1.0, 0.0, 0.0, 0.0]]), (3, 1))

    no_settle = retarget_hoi_mujoco(
        hoi_spec,
        tip_seq[:3],
        object_body="grasped_ball",
        object_poses=obj_poses,
        cfg=HoiPhysicsIKConfig(settle_steps=0, smooth_window=1),
    )
    settled = retarget_hoi_mujoco(
        hoi_spec,
        tip_seq[:3],
        object_body="grasped_ball",
        object_poses=obj_poses,
        cfg=HoiPhysicsIKConfig(settle_steps=8, smooth_window=3, servo_kp=15.0),
    )
    assert settled["qpos"].shape == (3, hoi_model.nq)
    # Without settle, solve_ik puts the index tip right at the sphere center (dist ~ 0);
    # with contact settle, collision pushes the finger outward away from the sphere center.
    d_check = mujoco.MjData(hoi_model)
    if_sid = mujoco.mj_name2id(hoi_model, mujoco.mjtObj.mjOBJ_SITE, "IFtip_r")
    d_check.qpos[:] = no_settle["qpos"][-1]
    mujoco.mj_kinematics(hoi_model, d_check)
    dist_unset = float(np.linalg.norm(d_check.site_xpos[if_sid] - tips_goal[1]))

    d_check.qpos[:] = settled["qpos"][-1]
    mujoco.mj_kinematics(hoi_model, d_check)
    dist_settled = float(np.linalg.norm(d_check.site_xpos[if_sid] - tips_goal[1]))
    assert dist_settled > dist_unset + 0.005
