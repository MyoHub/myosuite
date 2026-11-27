# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for Biovision Hierarchy (``.bvh`` / LAFAN1) parsing and retargeting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from myosuite.biomechanics import retarget as rt
from myosuite.biomechanics import retarget_bvh as rb
from myosuite.biomechanics import retarget_smplh as rs
from myosuite.envs.myo.tasks.mimic.tracking.models import (
    compiled_body,
    kinematic_fingerprint,
    model_version,
)
from myosuite.envs.myo.tasks.mimic.tracking.motion import load_motion_lib

pytestmark = pytest.mark.tier1


@pytest.fixture(scope="module")
def fullbody() -> mujoco.MjModel:
    return compiled_body("myofullbody")


def _lafan1_skeleton(
    *, toe_base_alias: bool = False, include_end_sites: bool = True
) -> list[dict[str, Any]]:
    l_toe = "LeftToeBase" if toe_base_alias else "LeftToe"
    r_toe = "RightToeBase" if toe_base_alias else "RightToe"

    def end(off: tuple[float, float, float]) -> list[dict[str, Any]]:
        return [{"end_site": True, "offset": off}] if include_end_sites else []

    return [
        {
            "name": "Hips",
            "root": True,
            "offset": (0.0, 0.0, 0.0),
            "children": [
                {
                    "name": "LeftUpLeg",
                    "offset": (9.5, -6.5, 0.0),
                    "children": [
                        {
                            "name": "LeftLeg",
                            "offset": (0.0, -41.0, 0.0),
                            "children": [
                                {
                                    "name": "LeftFoot",
                                    "offset": (0.0, -40.5, 0.0),
                                    "children": [
                                        {
                                            "name": l_toe,
                                            "offset": (0.0, -6.0, 13.0),
                                            "children": end((0.0, -2.0, 6.0)),
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                },
                {
                    "name": "RightUpLeg",
                    "offset": (-9.5, -6.5, 0.0),
                    "children": [
                        {
                            "name": "RightLeg",
                            "offset": (0.0, -41.0, 0.0),
                            "children": [
                                {
                                    "name": "RightFoot",
                                    "offset": (0.0, -40.5, 0.0),
                                    "children": [
                                        {
                                            "name": r_toe,
                                            "offset": (0.0, -6.0, 13.0),
                                            "children": end((0.0, -2.0, 6.0)),
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                },
                {
                    "name": "Spine",
                    "offset": (0.0, 10.0, 0.0),
                    "children": [
                        {
                            "name": "Spine1",
                            "offset": (0.0, 13.0, 0.0),
                            "children": [
                                {
                                    "name": "Spine2",
                                    "offset": (0.0, 13.0, 0.0),
                                    "children": [
                                        {
                                            "name": "Neck",
                                            "offset": (0.0, 14.0, 0.0),
                                            "children": [
                                                {
                                                    "name": "Head",
                                                    "offset": (0.0, 10.0, 0.0),
                                                    "children": end((0.0, 16.0, 0.0)),
                                                }
                                            ],
                                        },
                                        {
                                            "name": "LeftShoulder",
                                            "offset": (4.0, 10.0, 0.0),
                                            "children": [
                                                {
                                                    "name": "LeftArm",
                                                    "offset": (14.0, 0.0, 0.0),
                                                    "children": [
                                                        {
                                                            "name": "LeftForeArm",
                                                            "offset": (28.0, 0.0, 0.0),
                                                            "children": [
                                                                {
                                                                    "name": "LeftHand",
                                                                    "offset": (
                                                                        25.0,
                                                                        0.0,
                                                                        0.0,
                                                                    ),
                                                                    "children": end(
                                                                        (10.0, 0.0, 0.0)
                                                                    ),
                                                                }
                                                            ],
                                                        }
                                                    ],
                                                }
                                            ],
                                        },
                                        {
                                            "name": "RightShoulder",
                                            "offset": (-4.0, 10.0, 0.0),
                                            "children": [
                                                {
                                                    "name": "RightArm",
                                                    "offset": (-14.0, 0.0, 0.0),
                                                    "children": [
                                                        {
                                                            "name": "RightForeArm",
                                                            "offset": (-28.0, 0.0, 0.0),
                                                            "children": [
                                                                {
                                                                    "name": "RightHand",
                                                                    "offset": (
                                                                        -25.0,
                                                                        0.0,
                                                                        0.0,
                                                                    ),
                                                                    "children": end(
                                                                        (
                                                                            -10.0,
                                                                            0.0,
                                                                            0.0,
                                                                        )
                                                                    ),
                                                                }
                                                            ],
                                                        }
                                                    ],
                                                }
                                            ],
                                        },
                                    ],
                                }
                            ],
                        }
                    ],
                },
            ],
        }
    ]


def _write_lafan1_bvh(
    path: Path,
    *,
    num_frames: int,
    fps: float = 30.0,
    rot_order: tuple[str, str, str] = ("Zrotation", "Yrotation", "Xrotation"),
    root_pos_cm: np.ndarray | None = None,
    joint_euler_deg: dict[str, np.ndarray] | None = None,
    toe_base_alias: bool = False,
    include_end_sites: bool = True,
) -> Path:
    """Write a synthetic LAFAN1-hierarchy ``.bvh`` file with per-joint channel values."""
    tree = _lafan1_skeleton(
        toe_base_alias=toe_base_alias, include_end_sites=include_end_sites
    )
    lines = ["HIERARCHY"]
    animated_joints: list[tuple[str, bool]] = []

    def emit(node: dict[str, Any], depth: int) -> None:
        pad = "  " * depth
        ox, oy, oz = node["offset"]
        if node.get("end_site"):
            lines.append(f"{pad}End Site")
            lines.append(f"{pad}{{")
            lines.append(f"{pad}  OFFSET {ox:.6f} {oy:.6f} {oz:.6f}")
            lines.append(f"{pad}}}")
            return
        kw = "ROOT" if node.get("root") else "JOINT"
        lines.append(f"{pad}{kw} {node['name']}")
        lines.append(f"{pad}{{")
        lines.append(f"{pad}  OFFSET {ox:.6f} {oy:.6f} {oz:.6f}")
        if node.get("root"):
            ch_str = " ".join(("Xposition", "Yposition", "Zposition", *rot_order))
            lines.append(f"{pad}  CHANNELS 6 {ch_str}")
            animated_joints.append((node["name"], True))
        else:
            ch_str = " ".join(rot_order)
            lines.append(f"{pad}  CHANNELS 3 {ch_str}")
            animated_joints.append((node["name"], False))
        for child in node.get("children", ()):
            emit(child, depth + 1)
        lines.append(f"{pad}}}")

    emit(tree[0], 0)
    lines.append("MOTION")
    lines.append(f"Frames: {num_frames}")
    lines.append(f"Frame Time: {1.0 / fps:.10f}")

    total_ch = 3 + 3 * len(animated_joints)
    data = np.zeros((num_frames, total_ch), dtype=np.float64)
    if root_pos_cm is None:
        data[:, 1] = 96.0
    else:
        data[:, :3] = root_pos_cm

    euler_map = joint_euler_deg or {}
    col = 0
    for name, is_root in animated_joints:
        rot_col = col + 3 if is_root else col
        if name in euler_map:
            vals = np.asarray(euler_map[name], dtype=np.float64)
            data[:, rot_col : rot_col + 3] = np.broadcast_to(vals, (num_frames, 3))
        col = rot_col + 3

    for row in data:
        lines.append(" ".join(f"{v:.8f}" for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _make_walking_bvh(
    path: Path,
    *,
    num_frames: int = 36,
    fps: float = 30.0,
    rot_order: tuple[str, str, str] = ("Zrotation", "Yrotation", "Xrotation"),
    toe_base_alias: bool = False,
    include_end_sites: bool = True,
) -> Path:
    t = np.linspace(0.0, 1.0, num_frames)
    phase = 2.0 * np.pi * t
    root_pos = np.zeros((num_frames, 3))
    root_pos[:, 1] = 96.0 + 1.2 * np.cos(2.0 * phase)
    root_pos[:, 2] = 90.0 * t

    def euler_for_order(rx: np.ndarray, ry: np.ndarray, rz: np.ndarray) -> np.ndarray:
        by_ch = {"Xrotation": rx, "Yrotation": ry, "Zrotation": rz}
        return np.stack([by_ch[ch] for ch in rot_order], axis=-1)

    z = np.zeros(num_frames)
    l_hip_x = -20.0 * np.sin(phase)
    r_hip_x = 20.0 * np.sin(phase)
    l_knee_x = 42.0 * np.maximum(0.0, np.sin(phase - 0.4)) ** 2
    r_knee_x = 42.0 * np.maximum(0.0, -np.sin(phase - 0.4)) ** 2
    l_ankle_x = -8.0 * np.sin(phase)
    r_ankle_x = 8.0 * np.sin(phase)
    l_arm_z = np.full(num_frames, -72.0)
    r_arm_z = np.full(num_frames, 72.0)
    l_arm_y = 15.0 * np.sin(phase)
    r_arm_y = 15.0 * np.sin(phase)
    l_elb_y = -28.0 - 10.0 * np.sin(phase)
    r_elb_y = 28.0 - 10.0 * np.sin(phase)

    joint_euler = {
        "LeftUpLeg": euler_for_order(l_hip_x, z, z),
        "RightUpLeg": euler_for_order(r_hip_x, z, z),
        "LeftLeg": euler_for_order(l_knee_x, z, z),
        "RightLeg": euler_for_order(r_knee_x, z, z),
        "LeftFoot": euler_for_order(l_ankle_x, z, z),
        "RightFoot": euler_for_order(r_ankle_x, z, z),
        "LeftArm": euler_for_order(z, l_arm_y, l_arm_z),
        "RightArm": euler_for_order(z, r_arm_y, r_arm_z),
        "LeftForeArm": euler_for_order(z, l_elb_y, z),
        "RightForeArm": euler_for_order(z, r_elb_y, z),
    }
    return _write_lafan1_bvh(
        path,
        num_frames=num_frames,
        fps=fps,
        rot_order=rot_order,
        root_pos_cm=root_pos,
        joint_euler_deg=joint_euler,
        toe_base_alias=toe_base_alias,
        include_end_sites=include_end_sites,
    )


def _make_jump_bvh(path: Path, *, num_frames: int = 41, fps: float = 60.0) -> Path:
    root_pos = np.zeros((num_frames, 3))
    root_pos[:, 1] = 96.0
    jump_cm = np.zeros(num_frames)
    tau = np.linspace(-1.0, 1.0, 15)
    jump_cm[13:28] = 26.0 * (1.0 - tau**2)
    root_pos[:, 1] += jump_cm

    z = np.zeros(num_frames)
    tuck = np.zeros(num_frames)
    tuck[13:28] = 20.0 * (1.0 - tau**2)
    joint_euler = {
        "LeftUpLeg": np.stack([z, z, -0.5 * tuck], axis=-1),
        "RightUpLeg": np.stack([z, z, -0.5 * tuck], axis=-1),
        "LeftLeg": np.stack([z, z, tuck], axis=-1),
        "RightLeg": np.stack([z, z, tuck], axis=-1),
        "LeftArm": np.stack([np.full(num_frames, -70.0), z, z], axis=-1),
        "RightArm": np.stack([np.full(num_frames, 70.0), z, z], axis=-1),
        "LeftForeArm": np.stack([z, np.full(num_frames, -25.0), z], axis=-1),
        "RightForeArm": np.stack([z, np.full(num_frames, 25.0), z], axis=-1),
    }
    return _write_lafan1_bvh(
        path,
        num_frames=num_frames,
        fps=fps,
        rot_order=("Zrotation", "Yrotation", "Xrotation"),
        root_pos_cm=root_pos,
        joint_euler_deg=joint_euler,
    )


def test_collect_clip_files_suffix_and_defaults(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    bvh_a = tmp_path / "a.bvh"
    bvh_b = tmp_path / "sub" / "b.bvh"
    npz_c = tmp_path / "c.npz"
    bvh_a.write_text("x", encoding="utf-8")
    bvh_b.write_text("x", encoding="utf-8")
    npz_c.write_text("x", encoding="utf-8")

    npz_files, _, _ = rt.collect_clip_files(tmp_path)
    assert npz_files == [npz_c]

    bvh_files, base, dataset = rt.collect_clip_files(tmp_path, suffix="bvh")
    assert bvh_files == [bvh_a, bvh_b]
    assert base == tmp_path and dataset == tmp_path.name

    single, s_base, _ = rt.collect_clip_files(bvh_a, suffix=".bvh")
    assert single == [bvh_a] and s_base == tmp_path


def test_parse_bvh_fk_accuracy_and_rotation_orders(tmp_path: Path) -> None:
    root_cm = np.array(
        [
            [0.0, 96.0, 0.0],
            [10.0, 100.0, 25.0],
            [-5.0, 96.0, 50.0],
            [0.0, 96.0, 0.0],
        ]
    )
    # Frame 0: T-pose.
    # Frame 1: LeftUpLeg Xrotation = -90 deg -> thigh swings forward (+Z in BVH).
    # Frame 2: LeftLeg Xrotation = +90 deg -> shank swings backward (-Z in BVH).
    # Frame 3: LeftArm Zrotation = -90 deg, LeftForeArm Yrotation = -90 deg ->
    #   upper arm hangs straight down (-Y), forearm points forward (+Z)!
    joint_zyx = {
        "LeftUpLeg": np.array([[0, 0, 0], [0, 0, -90], [0, 0, 0], [0, 0, 0]], float),
        "LeftLeg": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 90], [0, 0, 0]], float),
        "LeftArm": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [-90, 0, 0]], float),
        "LeftForeArm": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, -90, 0]], float),
    }
    bvh_path = _write_lafan1_bvh(
        tmp_path / "fk_zyx.bvh",
        num_frames=4,
        fps=60.0,
        rot_order=("Zrotation", "Yrotation", "Xrotation"),
        root_pos_cm=root_cm,
        joint_euler_deg=joint_zyx,
    )
    clip = rb.parse_bvh(bvh_path, scale=0.01)
    assert clip.fps == pytest.approx(60.0, rel=1e-7)
    assert clip.joints[0].parent == -1 and clip.joints[0].name == "Hips"
    assert sum(j.is_end_site for j in clip.joints) == 5
    assert "LeftToe_End" in clip.joint_names and "RightToe_End" in clip.joint_names

    jidx = {n: i for i, n in enumerate(clip.joint_names)}
    # Frame 0: zero rotations -> positions equal root translation + rest_positions.
    np.testing.assert_allclose(
        clip.positions[0],
        clip.rest_positions + np.array([0.0, 0.96, 0.0]),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        clip.rotations[0],
        np.broadcast_to(np.eye(3), clip.rotations[0].shape),
        atol=1e-12,
    )
    # Translation scaling (cm -> m).
    np.testing.assert_allclose(
        clip.positions[1, jidx["Hips"]], [0.10, 1.00, 0.25], atol=1e-9
    )
    clip_raw = rb.parse_bvh(bvh_path, scale=1.0)
    np.testing.assert_allclose(clip_raw.positions, clip.positions * 100.0, atol=1e-9)

    # Frame 1: LeftUpLeg -90 deg about X -> knee at hip + (0, 0, +0.41).
    hip_f1 = clip.positions[1, jidx["LeftUpLeg"]]
    knee_f1 = clip.positions[1, jidx["LeftLeg"]]
    np.testing.assert_allclose(knee_f1 - hip_f1, [0.0, 0.0, 0.41], atol=1e-9)

    # Frame 2: LeftLeg +90 deg about X -> ankle at knee + (0, 0, -0.405).
    knee_f2 = clip.positions[2, jidx["LeftLeg"]]
    ankle_f2 = clip.positions[2, jidx["LeftFoot"]]
    np.testing.assert_allclose(ankle_f2 - knee_f2, [0.0, 0.0, -0.405], atol=1e-9)
    flex = rb.knee_flexion(clip)
    assert np.degrees(flex["l"][2]) == pytest.approx(90.0, abs=1e-6)

    # Frame 3: LeftArm Z=-90 -> elbow at shoulder + (0, -0.28, 0);
    # LeftForeArm local Y=-90 -> in world frame, wrist at elbow + (0, 0, +0.25).
    sh_f3 = clip.positions[3, jidx["LeftArm"]]
    el_f3 = clip.positions[3, jidx["LeftForeArm"]]
    wr_f3 = clip.positions[3, jidx["LeftHand"]]
    np.testing.assert_allclose(el_f3 - sh_f3, [0.0, -0.28, 0.0], atol=1e-9)
    np.testing.assert_allclose(wr_f3 - el_f3, [0.0, 0.0, 0.25], atol=1e-9)

    # Verify intrinsic ZYX vs ZXY non-commutative Euler composition.
    angles = np.array([[30.0, 45.0, 60.0]])
    p_zyx = _write_lafan1_bvh(
        tmp_path / "ord_zyx.bvh",
        num_frames=1,
        rot_order=("Zrotation", "Yrotation", "Xrotation"),
        joint_euler_deg={"Hips": angles},
    )
    p_zxy = _write_lafan1_bvh(
        tmp_path / "ord_zxy.bvh",
        num_frames=1,
        rot_order=("Zrotation", "Xrotation", "Yrotation"),
        joint_euler_deg={"Hips": angles},
    )
    c_zyx = rb.parse_bvh(p_zyx)
    c_zxy = rb.parse_bvh(p_zxy)
    Rz = rb._axis_rotation(2, np.radians([30.0]))[0]
    Ry45 = rb._axis_rotation(1, np.radians([45.0]))[0]
    Rx60 = rb._axis_rotation(0, np.radians([60.0]))[0]
    Rx45 = rb._axis_rotation(0, np.radians([45.0]))[0]
    Ry60 = rb._axis_rotation(1, np.radians([60.0]))[0]
    np.testing.assert_allclose(c_zyx.rotations[0, 0], Rz @ Ry45 @ Rx60, atol=1e-12)
    np.testing.assert_allclose(c_zxy.rotations[0, 0], Rz @ Rx45 @ Ry60, atol=1e-12)
    assert not np.allclose(c_zyx.rotations[0, 0], c_zxy.rotations[0, 0])


def test_parse_bvh_rejects_malformed_files(tmp_path: Path) -> None:
    valid = _write_lafan1_bvh(tmp_path / "ok.bvh", num_frames=2)
    text = valid.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="scale must be finite and positive"):
        rb.parse_bvh(valid, scale=0.0)

    cases = [
        ("no_motion.bvh", text.split("MOTION")[0], "missing MOTION"),
        (
            "bad_hdr.bvh",
            text.replace("HIERARCHY", "GARBAGE", 1),
            "expected 'HIERARCHY'",
        ),
        ("bad_chan.bvh", text.replace("Xrotation", "Wrotation", 1), "invalid channel"),
        (
            "short_data.bvh",
            text.replace("Frames: 2", "Frames: 3"),
            "expected .* motion values",
        ),
        (
            "zero_frames.bvh",
            text.replace("Frames: 2", "Frames: 0"),
            "Frames must be >= 1",
        ),
        (
            "bad_ft.bvh",
            text.replace("Frame Time: ", "Frame Time: -"),
            "Frame Time must be positive",
        ),
        ("nan_data.bvh", text.rsplit(" ", 1)[0] + " nan\n", "non-finite"),
        (
            "end_chan.bvh",
            text.replace(
                "End Site\n          {\n",
                "End Site\n          {\n            CHANNELS 1 Xrotation\n",
                1,
            ),
            "End Site .* cannot declare CHANNELS",
        ),
    ]
    for fname, bad_txt, pattern in cases:
        p = tmp_path / fname
        p.write_text(bad_txt, encoding="utf-8")
        with pytest.raises(ValueError, match=pattern):
            rb.parse_bvh(p)


def test_retarget_bvh_myofullbody_stand_walk_aliases_and_jump(
    fullbody: mujoco.MjModel, tmp_path: Path
) -> None:
    bmap = rb.LAFAN1_MYOFULLBODY
    assert rb.landmark_bodies(bmap) == rs.landmark_bodies(rs.SMPLH_MYOFULLBODY)
    assert rt.kinematic_fingerprint(
        fullbody, rb.landmark_bodies(bmap), bmap.feet
    ) == kinematic_fingerprint("myofullbody")

    # 1. T-pose standing clip.
    stand_bvh = _write_lafan1_bvh(tmp_path / "stand.bvh", num_frames=12, fps=30.0)
    src, lmap = rb.bvh_source(fullbody, bmap)
    ik = rt.LandmarkIK(fullbody, lmap)
    ver = model_version("myofullbody")
    q_stand, freq_stand, meta_stand = rt.refit_clip(
        stand_bvh, fullbody, "myofullbody", ver, lmap, src, ik=ik
    )
    assert freq_stand == pytest.approx(30.0)
    assert q_stand.shape == (12, fullbody.nq)
    assert np.isfinite(q_stand).all()
    assert meta_stand["metrics"]["coupling_residual"] < 1e-8
    assert meta_stand["metrics"]["limit_violations"] == 0
    assert meta_stand["metrics"]["flags"] == []
    low_stand, _ = rt._foot_kinematics(fullbody, lmap, q_stand)
    assert np.max(np.abs(low_stand.min(axis=1))) < 0.005

    # 2. Walking clips across ZYX and ZXY Euler orders, LeftToeBase alias, and no End Sites.
    walk_zyx = _make_walking_bvh(
        tmp_path / "walk_zyx.bvh",
        rot_order=("Zrotation", "Yrotation", "Xrotation"),
    )
    walk_zxy_alias = _make_walking_bvh(
        tmp_path / "walk_zxy_alias.bvh",
        rot_order=("Zrotation", "Xrotation", "Yrotation"),
        toe_base_alias=True,
        include_end_sites=False,
    )
    for clip_path in (walk_zyx, walk_zxy_alias):
        q_walk, _, meta_walk = rt.refit_clip(
            clip_path, fullbody, "myofullbody", ver, lmap, src, ik=ik
        )
        assert q_walk.shape == (36, fullbody.nq)
        assert np.isfinite(q_walk).all()
        assert meta_walk["metrics"]["coupling_residual"] < 1e-8
        assert meta_walk["metrics"]["limit_violations"] == 0
        assert meta_walk["metrics"]["penetration_mm"] < 5.0
        assert meta_walk["metrics"]["flags"] == []
        low_walk, _ = rt._foot_kinematics(fullbody, lmap, q_walk)
        assert float(np.min(low_walk)) >= -1e-4
        assert float(np.percentile(low_walk.min(axis=1), 10)) < 0.02

    # 3. Jump clip with flight phase: stance grounded at z=0, apex airborne.
    jump_bvh = _make_jump_bvh(tmp_path / "jump.bvh")
    src_jump, _ = rb.bvh_source(fullbody, bmap, ground_sigma=0.01)
    q_jump, _, meta_jump = rt.refit_clip(
        jump_bvh, fullbody, "myofullbody", ver, lmap, src_jump, ik=ik
    )
    assert q_jump.shape == (41, fullbody.nq)
    assert np.isfinite(q_jump).all()
    assert meta_jump["metrics"]["coupling_residual"] < 1e-8
    assert meta_jump["metrics"]["limit_violations"] == 0
    assert meta_jump["metrics"]["flags"] == []
    low_jump, _ = rt._foot_kinematics(fullbody, lmap, q_jump)
    stance_sole = low_jump[:10].min(axis=1)
    apex_sole = float(low_jump[20].min())
    assert np.max(np.abs(stance_sole)) < 0.01
    assert apex_sole > 0.18


def test_retarget_bvh_cli_end_to_end(fullbody: mujoco.MjModel, tmp_path: Path) -> None:
    from scripts.mimic import retarget_bvh as cli

    in_dir = tmp_path / "lafan1_sub"
    (in_dir / "actor1").mkdir(parents=True)
    _make_walking_bvh(in_dir / "actor1" / "walk1_subject1.bvh", num_frames=24, fps=30.0)
    _make_jump_bvh(in_dir / "jump1_subject1.bvh", num_frames=41, fps=60.0)

    curation_path = tmp_path / "curation.json"
    curation_path.write_text(
        json.dumps(
            {
                "BVH/lafan1_sub/actor1/walk1_subject1.bvh": {
                    "quality_flags": ["curated_ok"]
                }
            }
        ),
        encoding="utf-8",
    )
    out_root = tmp_path / "out_cache"
    report_dir = tmp_path / "reports"

    argv = [
        str(in_dir),
        "--scale",
        "0.01",
        "--model",
        "myofullbody",
        "--out-root",
        str(out_root),
        "--report-dir",
        str(report_dir),
        "--quality-flags",
        str(curation_path),
        "--workers",
        "1",
    ]
    assert cli.main(argv) == 0

    walk_npz = out_root / "lafan1_sub" / "actor1" / "walk1_subject1.npz"
    jump_npz = out_root / "lafan1_sub" / "jump1_subject1.npz"
    assert walk_npz.is_file() and jump_npz.is_file()

    ver = model_version("myofullbody")
    fp = kinematic_fingerprint("myofullbody")
    for npz_path in (walk_npz, jump_npz):
        rt.check_refit_meta(npz_path, "myofullbody", ver, fp)
        with np.load(npz_path, allow_pickle=False) as d:
            assert d["qpos"].shape[1] == fullbody.nq
            assert np.isfinite(d["qpos"]).all()

    meta_walk = rt.read_refit_meta(walk_npz)
    assert meta_walk is not None
    assert meta_walk["source_format"] == "bvh"
    assert meta_walk["bvh_map"] == "lafan1-bvh->myofullbody"
    assert "curated_ok" in meta_walk["quality_flags"]

    lib = load_motion_lib(
        str(out_root / "lafan1_sub"), "", "myofullbody", True, 0.02, 1.0
    )
    assert lib.num_frames > 0

    summary_json = report_dir / "retarget_summary.json"
    summary_md = report_dir / "retarget_summary.md"
    clips_jsonl = report_dir / "retarget_clips.jsonl"
    assert summary_json.is_file() and summary_md.is_file() and clips_jsonl.is_file()
    summary_1 = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary_1["clips"] == 2
    assert summary_1["skipped"] == 0
    assert summary_1["errors"] == []

    # Direct worker cache check + second CLI run cache hit.
    cached_res = cli._work(
        (
            str(in_dir / "actor1" / "walk1_subject1.bvh"),
            str(walk_npz),
            "BVH/lafan1_sub/actor1/walk1_subject1.bvh",
            {"quality_flags": ["curated_ok"]},
        )
    )
    assert cached_res["skipped"] is True

    assert cli.main(argv) == 0
    summary_2 = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary_2["clips"] == 2
    assert summary_2["skipped"] == 2

    # Fingerprint stamping CLI path.
    assert (
        cli.main(
            [
                "--stamp-fingerprint",
                str(out_root / "lafan1_sub"),
                "--model",
                "myofullbody",
            ]
        )
        == 0
    )
