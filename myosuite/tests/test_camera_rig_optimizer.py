# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for pre-capture multi-camera rig placement optimization."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite.biomechanics.camera_rig_optimizer import (
    RoomBounds,
    evaluate_camera_rig,
    look_at_camera,
    optimize_camera_rig,
)
from myosuite.biomechanics.retarget import Landmark

pytestmark = pytest.mark.tier1


def _bilateral_arm_torso_model() -> (
    tuple[mujoco.MjModel, tuple[Landmark, ...], np.ndarray]
):
    xml = """
    <mujoco model="torso_arms">
      <compiler angle="radian"/>
      <worldbody>
        <body name="torso" pos="0 0 1.25">
          <geom name="chest" type="box" size="0.16 0.22 0.32"/>
          <site name="s_sternum" pos="0.17 0 0.15" size="0.01"/>
          <site name="s_c7" pos="-0.17 0 0.22" size="0.01"/>
          <body name="arm_l" pos="0 0.28 0.20">
            <joint name="elbow_l" type="hinge" axis="0 1 0" range="-2.0 0.5"/>
            <geom name="limb_l" type="capsule" fromto="0 0 0 0 0 -0.35" size="0.04"/>
            <site name="s_wrist_l" pos="0 0.05 -0.35" size="0.01"/>
            <site name="s_elbow_l" pos="0.05 0.05 -0.15" size="0.01"/>
          </body>
          <body name="arm_r" pos="0 -0.28 0.20">
            <joint name="elbow_r" type="hinge" axis="0 1 0" range="-2.0 0.5"/>
            <geom name="limb_r" type="capsule" fromto="0 0 0 0 0 -0.35" size="0.04"/>
            <site name="s_wrist_r" pos="0 -0.05 -0.35" size="0.01"/>
            <site name="s_elbow_r" pos="0.05 -0.05 -0.15" size="0.01"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    lms = (
        Landmark("sternum", "torso", "s_sternum"),
        Landmark("c7", "torso", "s_c7"),
        Landmark("elbow_l", "arm_l", "s_elbow_l"),
        Landmark("wrist_l", "arm_l", "s_wrist_l"),
        Landmark("elbow_r", "arm_r", "s_elbow_r"),
        Landmark("wrist_r", "arm_r", "s_wrist_r"),
    )
    angles = np.linspace(-1.2, -0.2, 6)
    poses = np.column_stack([angles, angles[::-1]])
    return model, lms, poses


def test_optimize_camera_rig_overcomes_torso_occlusion_and_improves_stereo() -> None:
    model, lms, poses = _bilateral_arm_torso_model()
    k_mat = np.array([[900.0, 0.0, 960.0], [0.0, 900.0, 540.0], [0.0, 0.0, 1.0]])
    target = np.array([0.0, 0.0, 1.20])

    # Naive 3-camera cluster on the right wall (y = -2.8): torso occludes left arm.
    naive_cams = [
        look_at_camera(f"naive_{i}", [-0.8 + 0.8 * i, -2.8, 1.5], target, k_mat)
        for i in range(3)
    ]
    _, naive_worst, _, naive_stereo, naive_score = evaluate_camera_rig(
        model,
        poses,
        lms,
        naive_cams,
        image_sizes=[(1920, 1080)] * 3,
        check_occlusion=True,
        min_views=2,
    )
    assert naive_worst == pytest.approx(0.0, abs=1e-9)
    assert naive_stereo <= 0.80

    room = RoomBounds(
        x_bounds=(-2.8, 2.8),
        y_bounds=(-2.8, 2.8),
        height_bounds=(1.1, 2.3),
        min_subject_distance_m=1.2,
        min_camera_separation_m=0.8,
    )
    opt = optimize_camera_rig(
        model,
        poses,
        lms,
        n_cameras=4,
        intrinsic=k_mat,
        room=room,
        min_views=2,
        check_occlusion=True,
        n_candidates=24,
        refine_steps=10,
        seed=7,
    )
    opt_repeat = optimize_camera_rig(
        model,
        poses,
        lms,
        n_cameras=4,
        intrinsic=k_mat,
        room=room,
        min_views=2,
        check_occlusion=True,
        n_candidates=24,
        refine_steps=10,
        seed=7,
    )
    np.testing.assert_allclose(opt.positions, opt_repeat.positions, atol=1e-12)
    assert opt.stereo_fraction >= 0.95
    assert opt.worst_dof_sigma > 20.0
    assert opt.score > naive_score + 10.0
    # Cameras span both sides of the sagittal plane to see around the torso.
    assert opt.positions[:, 1].min() < -0.5
    assert opt.positions[:, 1].max() > 0.5


def test_look_at_camera_principal_point_and_free_mount_mode() -> None:
    k_mat = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    pos = np.array([2.2, -1.5, 1.8])
    tgt = np.array([0.1, 0.2, 1.1])
    cam = look_at_camera("c0", pos, tgt, k_mat)
    np.testing.assert_allclose(cam.center, pos, atol=1e-12)
    np.testing.assert_allclose(cam.project(tgt), [640.0, 360.0], atol=1e-9)

    model, lms, poses = _bilateral_arm_torso_model()
    room = RoomBounds(
        x_bounds=(-2.5, 2.5),
        y_bounds=(-2.5, 2.5),
        height_bounds=(1.2, 2.2),
        min_subject_distance_m=1.1,
        min_camera_separation_m=0.6,
        mount_mode="free",
    )
    opt = optimize_camera_rig(
        model,
        poses[0],
        lms,
        n_cameras=3,
        intrinsic=k_mat,
        room=room,
        image_size=(1280, 720),
        n_candidates=18,
        refine_steps=6,
        seed=3,
    )
    assert opt.positions.shape == (3, 3)
    assert np.all(opt.positions[:, 0] >= -2.5) and np.all(opt.positions[:, 0] <= 2.5)
    assert np.all(opt.positions[:, 2] >= 1.2) and np.all(opt.positions[:, 2] <= 2.2)
    assert opt.worst_dof_sigma > 10.0
