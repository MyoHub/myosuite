# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for walk-frame gravity alignment in video retargeters."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from myosuite import make_env
from myosuite.biomechanics import gvhmr
from myosuite.biomechanics import retarget_instanthmr as ri

pytestmark = pytest.mark.tier2

_JIDX = {name: i for i, name in enumerate(ri.MHR70_JOINTS)}


@pytest.fixture(scope="module")
def myolegs() -> mujoco.MjModel:
    env = make_env("myoLegWalk-v0")
    try:
        yield env.unwrapped.model.__copy__()
    finally:
        env.close()


def _synthetic_walk_mhr70_cam(
    model: mujoco.MjModel,
    *,
    pitch_deg: float = 18.0,
    roll_deg: float = 6.0,
    trunk_lean_deg: float = 10.0,
    n_frames: int = 36,
    fps: float = 30.0,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    imap = ri.INSTANTHMR_MYOLEGS
    cal = ri.calibrate(model, imap)
    d = mujoco.MjData(model)
    d.qpos[:] = cal.qpos
    mujoco.mj_kinematics(model, d)
    fwd, left, up = cal.canon_to_model.T

    base = np.zeros((len(ri.MHR70_JOINTS), 3), dtype=np.float64)
    pelvis_0 = d.xpos[model.body("pelvis").id]
    base[:] = pelvis_0
    lean_vec = (
        np.sin(np.radians(trunk_lean_deg)) * fwd
        + np.cos(np.radians(trunk_lean_deg)) * up
    )
    for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
        base[_JIDX[name]] = pelvis_0 + 0.72 * lean_vec
    base[_JIDX["left_shoulder"]] = pelvis_0 + 0.52 * lean_vec + 0.18 * left
    base[_JIDX["right_shoulder"]] = pelvis_0 + 0.52 * lean_vec - 0.18 * left

    for side, sfx, sign in (("left", "_l", 1.0), ("right", "_r", -1.0)):
        base[_JIDX[f"{side}_hip"]] = d.xpos[model.body(f"femur{sfx}").id]
        base[_JIDX[f"{side}_knee"]] = d.xpos[model.body(f"tibia{sfx}").id]
        base[_JIDX[f"{side}_ankle"]] = d.xpos[model.body(f"talus{sfx}").id]
        base[_JIDX[f"{side}_heel"]] = d.xpos[model.body(f"calcn{sfx}").id]
        toes_p = d.xpos[model.body(f"toes{sfx}").id]
        base[_JIDX[f"{side}_big_toe_tip"]] = toes_p - sign * 0.025 * left
        base[_JIDX[f"{side}_small_toe_tip"]] = toes_p + sign * 0.025 * left

    sole_names = (
        ("left_ankle", "left_heel", "left_big_toe_tip", "left_small_toe_tip"),
        ("right_ankle", "right_heel", "right_big_toe_tip", "right_small_toe_tip"),
    )
    t = np.arange(n_frames, dtype=np.float64) / fps
    phase = 2.0 * np.pi * 1.0 * t
    world = np.repeat(base[None, :, :], n_frames, axis=0)
    # Continuous walking progression along fwd with alternating stance/swing phases.
    prog = 1.2 * t
    world += prog[:, None, None] * fwd[None, None, :]

    for leg_i, s_group in enumerate(sole_names):
        ph = phase + (0.0 if leg_i == 0 else np.pi)
        swing = np.maximum(np.sin(ph), 0.0)
        # During stance (sin <= 0), keep foot planted in world space by cancelling body travel;
        # keep swing symmetric along fwd so median body vector preserves trunk lean.
        foot_rel_x = -0.18 * np.cos(ph)
        foot_lift = 0.06 * swing
        for j_name in s_group:
            idx = _JIDX[j_name]
            world[:, idx] += (
                foot_rel_x[:, None] * fwd[None, :] + foot_lift[:, None] * up[None, :]
            )
        knee_name = "left_knee" if leg_i == 0 else "right_knee"
        world[:, _JIDX[knee_name]] += (
            0.5 * foot_rel_x[:, None] * fwd[None, :]
            + 0.03 * swing[:, None] * up[None, :]
        )

    # Transform model world -> canonical -> pitched/rolled OpenCV camera.
    r_tilt = Rotation.from_euler("xy", [pitch_deg, roll_deg], degrees=True).as_matrix()
    r_true_cam_to_model = cal.canon_to_model @ (ri.CAM_TO_WORLD @ r_tilt.T)
    joints_cam = (world @ r_true_cam_to_model) + np.array([0.0, 0.2, 3.5])
    clip = {
        "joints_3d_cam": joints_cam,
        "fps": np.float64(fps),
        "valid": np.ones(n_frames, dtype=bool),
    }
    return clip, up


def test_pitched_camera_walk_frame_levels_instanthmr_retarget(
    myolegs: mujoco.MjModel,
) -> None:
    clip, model_up = _synthetic_walk_mhr70_cam(
        myolegs, pitch_deg=18.0, roll_deg=6.0, trunk_lean_deg=10.0
    )
    align_walk = ri.estimate_instanthmr_walk_frame(
        clip["joints_3d_cam"], vertical_method="walk"
    )
    align_frame = ri.estimate_instanthmr_walk_frame(
        clip["joints_3d_cam"], vertical_method="frame"
    )
    cal = ri.calibrate(myolegs, ri.INSTANTHMR_MYOLEGS)
    up_walk = (cal.canon_to_model @ align_walk.rotation).T @ model_up
    up_frame = (cal.canon_to_model @ align_frame.rotation).T @ model_up
    # Verify walk-frame vertical cancels trunk lean much better than single-frame body axis.
    _, world_up = _synthetic_walk_mhr70_cam(
        myolegs, pitch_deg=18.0, roll_deg=6.0, trunk_lean_deg=0.0
    )
    align_upright = ri.estimate_instanthmr_walk_frame(
        _synthetic_walk_mhr70_cam(
            myolegs, pitch_deg=18.0, roll_deg=6.0, trunk_lean_deg=0.0
        )[0]["joints_3d_cam"],
        vertical_method="walk",
    )
    up_true_cam = (cal.canon_to_model @ align_upright.rotation).T @ world_up
    ang_walk_deg = float(
        np.degrees(np.arccos(np.clip(np.dot(up_walk, up_true_cam), -1.0, 1.0)))
    )
    ang_frame_deg = float(
        np.degrees(np.arccos(np.clip(np.dot(up_frame, up_true_cam), -1.0, 1.0)))
    )
    assert ang_walk_deg < 2.0
    assert ang_frame_deg > 6.0

    src_fixed, lmap = ri.instanthmr_source(
        myolegs, ri.INSTANTHMR_MYOLEGS, cam_to_world_mode="fixed"
    )
    src_walk, _ = ri.instanthmr_source(
        myolegs, ri.INSTANTHMR_MYOLEGS, cam_to_world_mode="walk"
    )
    tgt_fixed = src_fixed.targets(clip, myolegs, lmap)
    tgt_walk = src_walk.targets(clip, myolegs, lmap)

    # Root target z drift across the 1.2s walk: fixed camera climbs an 18 deg slope,
    # whereas walk-frame alignment keeps the pelvis trajectory level.
    z_drift_fixed = abs(float(tgt_fixed.pos[-1, 0, 2] - tgt_fixed.pos[0, 0, 2]))
    z_drift_walk = abs(float(tgt_walk.pos[-1, 0, 2] - tgt_walk.pos[0, 0, 2]))
    assert z_drift_fixed > 0.30
    assert z_drift_walk < 0.03

    ik_walk = ri.solve_instanthmr_ik(clip, model=myolegs, cam_to_world_mode="walk")
    assert ik_walk.metrics["cam_to_world_mode"] == "walk"
    assert abs(float(ik_walk.qpos[-1, 2] - ik_walk.qpos[0, 2])) < 0.05


def test_gvhmr_level_walk_plane_removes_camera_pitch_drift() -> None:
    t = np.linspace(0.0, 2.0, 40, dtype=np.float32)
    slope_rad = np.radians(12.0)
    # In GVHMR +Y-up coordinates: walking forward along +Z with a 12 deg pitch upward in +Y.
    transl_y_up = np.stack(
        [
            np.zeros_like(t),
            (1.5 * t * np.sin(slope_rad)).astype(np.float32),
            (-1.5 * t * np.cos(slope_rad)).astype(np.float32),
        ],
        axis=-1,
    )
    raw = {
        "global": {
            "global_orient": np.zeros((len(t), 3), dtype=np.float32),
            "body_pose": np.zeros((len(t), 63), dtype=np.float32),
            "betas": np.zeros(10, dtype=np.float32),
            "transl": transl_y_up,
        },
        "pelvis_j0": np.array([0.0, -0.22, 0.0], dtype=np.float32),
    }
    sloped = gvhmr.to_amass_clip(raw, motion_fps=20.0, level_walk_plane=False)
    leveled = gvhmr.to_amass_clip(raw, motion_fps=20.0, level_walk_plane=True)

    assert abs(float(sloped["trans"][-1, 2] - sloped["trans"][0, 2])) > 0.50
    assert abs(float(leveled["trans"][-1, 2] - leveled["trans"][0, 2])) < 1e-4
    np.testing.assert_allclose(leveled["trans"][0], sloped["trans"][0], atol=1e-6)
