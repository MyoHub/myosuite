# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for chase-tag heading, clip yaw, and two-agent shadow maps."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite.core.trajectory_yaw import (
    clip_natural_yaw,
    pingpong_frame,
    rotate_xy,
    write_rotated_clip_npz,
    yaw_quat,
)
from myosuite.integrations.musclemimic.two_agent_scene import (
    build_agent_shadow_map,
    copy_agent_to_shadow,
    write_clip_pose,
)
from myosuite.physics import quat_math
from myosuite.terms.multiplayer.chase_tag_vs_obs import agent_is_chaser
from myosuite.terms.multiplayer.heading import (
    ARENA_LIM_M,
    COMPASS_LABELS,
    aim_heading_xy,
    body_yaw,
    clamp_arena_xy,
    compass_index,
    heading_yaw,
    role_heading_xy,
    step_compass,
    wall_repulsion,
)

pytestmark = pytest.mark.tier1


def test_pingpong_frame_is_continuous_at_the_ends() -> None:
    n = 5
    seq = [pingpong_frame(t, n) for t in range(12)]
    assert seq == [0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3]
    assert pingpong_frame(0, 1) == 0


def test_agent_is_chaser_matches_task_roles() -> None:
    assert agent_is_chaser("agent_0", "CHASE")
    assert not agent_is_chaser("agent_1", "CHASE")
    assert not agent_is_chaser("agent_0", "EVADE")
    assert agent_is_chaser("agent_1", "EVADE")


def test_role_heading_points_toward_or_away() -> None:
    me = np.array([0.0, 0.0])
    opp = np.array([2.0, 0.0])
    toward = role_heading_xy(me, opp, pursue=True)
    away = role_heading_xy(me, opp, pursue=False)
    assert body_yaw(toward) == pytest.approx(heading_yaw(toward) + np.pi / 2)
    assert toward[0] == pytest.approx(1.0)
    assert away[0] == pytest.approx(-1.0)
    assert compass_index(heading_yaw(toward)) == COMPASS_LABELS.index("E")
    assert compass_index(heading_yaw(away)) == COMPASS_LABELS.index("W")


def test_step_compass_turns_at_most_one_bin() -> None:
    assert step_compass(0, 0) == 0
    assert step_compass(0, 1) == 1
    assert step_compass(0, 2) == 1
    assert step_compass(0, 7) == 7
    assert step_compass(0, 4) in (1, 7)


def test_wall_repulsion_pushes_inward() -> None:
    v = wall_repulsion(np.array([4.8, 0.0]), arena_lim=5.0, margin=1.5)
    assert v[0] < 0
    assert v[1] == pytest.approx(0.0)
    aimed = aim_heading_xy(
        np.array([4.8, 0.0]),
        np.array([6.0, 0.0]),
        pursue=True,
        arena_lim=5.0,
        margin=1.5,
    )
    assert aimed[0] < 0.0


def test_aim_heading_does_not_walk_through_the_fence() -> None:
    assert ARENA_LIM_M == 6.0
    flee = aim_heading_xy(
        np.array([0.0, -5.8]),
        np.array([0.0, -3.0]),
        pursue=False,
    )
    assert flee[1] >= 0.0
    chase = aim_heading_xy(
        np.array([5.7, 0.0]),
        np.array([8.0, 0.0]),
        pursue=True,
    )
    assert chase[0] <= 0.0
    assert clamp_arena_xy(np.array([7.0, -7.0]))[0] < ARENA_LIM_M


def test_write_rotated_clip_npz_turns_root_xy(tmp_path: Path) -> None:
    t = 4
    qpos = np.zeros((t, 7), dtype=np.float64)
    qpos[:, 3] = 1.0
    qpos[:, 0] = np.linspace(0.0, 1.0, t)
    qvel = np.zeros((t, 6), dtype=np.float64)
    qvel[:, 0] = 0.5
    src = tmp_path / "walk.npz"
    np.savez(src, qpos=qpos, qvel=qvel)
    dst = write_rotated_clip_npz(src, tmp_path / "walk_N.npz", np.pi / 2)
    rot = np.load(dst)
    assert rot["qpos"][-1, 1] == pytest.approx(1.0, abs=1e-6)
    assert rot["qpos"][-1, 0] == pytest.approx(0.0, abs=1e-6)
    assert rot["qvel"][0, 1] == pytest.approx(0.5, abs=1e-6)
    assert clip_natural_yaw(qpos, qvel) == pytest.approx(0.0, abs=1e-6)


def test_rotate_xy_about_origin() -> None:
    xy = np.array([[1.0, 0.0], [2.0, 0.0]])
    out = rotate_xy(xy, np.pi / 2, np.array([0.0, 0.0]))
    assert out[1, 0] == pytest.approx(0.0, abs=1e-6)
    assert out[1, 1] == pytest.approx(2.0, abs=1e-6)


def _tiny_pair_models() -> tuple[mujoco.MjModel, mujoco.MjModel]:
    shadow = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="pelvis" pos="0 0 1">
              <freejoint name="root"/>
              <geom type="sphere" size="0.05"/>
              <body name="shin" pos="0 0 -0.2">
                <joint name="knee" type="hinge" axis="0 1 0"/>
                <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.2"/>
              </body>
            </body>
          </worldbody>
          <actuator>
            <motor name="knee" joint="knee"/>
          </actuator>
        </mujoco>
        """
    )
    fbvs = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="a0_pelvis" pos="0 1 1">
              <freejoint name="a0_root"/>
              <geom type="sphere" size="0.05"/>
              <body name="a0_shin" pos="0 0 -0.2">
                <joint name="a0_knee" type="hinge" axis="0 1 0"/>
                <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.2"/>
              </body>
            </body>
            <body name="a1_pelvis" pos="0 -1 1">
              <freejoint name="a1_root"/>
              <geom type="sphere" size="0.05"/>
              <body name="a1_shin" pos="0 0 -0.2">
                <joint name="a1_knee" type="hinge" axis="0 1 0"/>
                <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.2"/>
              </body>
            </body>
          </worldbody>
          <actuator>
            <motor name="a0_knee" joint="a0_knee"/>
            <motor name="a1_knee" joint="a1_knee"/>
          </actuator>
        </mujoco>
        """
    )
    return fbvs, shadow


def test_agent_shadow_map_roundtrip() -> None:
    fbvs, shadow = _tiny_pair_models()
    fd, sd = mujoco.MjData(fbvs), mujoco.MjData(shadow)
    amap = build_agent_shadow_map(fbvs, shadow, "a0_", [0])
    fd.qpos[amap.qpos_pairs[0][0] : amap.qpos_pairs[0][0] + 2] = [0.3, -0.4]
    fd.qpos[amap.qpos_pairs[1][0]] = 0.25
    copy_agent_to_shadow(fd, sd, amap)
    assert sd.qpos[0] == pytest.approx(0.3)
    assert sd.qpos[1] == pytest.approx(-0.4)
    assert sd.qpos[7] == pytest.approx(0.25)


def test_write_clip_pose_keeps_xy_and_optional_yaw() -> None:
    fbvs, shadow = _tiny_pair_models()
    fd = mujoco.MjData(fbvs)
    amap = build_agent_shadow_map(fbvs, shadow, "a0_", [0])
    fa, sa, _ = amap.qpos_pairs[0]
    fd.qpos[fa : fa + 2] = [1.5, -2.0]
    clip_qpos = np.zeros(shadow.nq)
    clip_qpos[3] = 1.0
    clip_qpos[0:2] = [9.0, 9.0]
    clip_qvel = np.zeros(shadow.nv)
    clip_qvel[0] = 0.4
    write_clip_pose(fd, amap, clip_qpos, clip_qvel, keep_xy=True, keep_yaw=False)
    assert fd.qpos[fa : fa + 2] == pytest.approx([1.5, -2.0])
    qz = yaw_quat(np.pi / 2)
    fd.qpos[fa + 3 : fa + 7] = qz
    write_clip_pose(fd, amap, clip_qpos, clip_qvel, keep_xy=True, keep_yaw=True)
    assert quat_math.quat2yaw(fd.qpos[fa + 3 : fa + 7]) == pytest.approx(
        np.pi / 2, abs=1e-5
    )
    assert fd.qvel[amap.qvel_pairs[0][0] + 1] == pytest.approx(0.4, abs=1e-5)


def test_write_clip_pose_face_and_travel_can_differ() -> None:
    fbvs, shadow = _tiny_pair_models()
    fd = mujoco.MjData(fbvs)
    amap = build_agent_shadow_map(fbvs, shadow, "a0_", [0])
    fa, _, _ = amap.qpos_pairs[0]
    fd.qpos[fa : fa + 2] = [0.0, 1.0]
    clip_qpos = np.zeros(shadow.nq)
    clip_qpos[3] = 1.0
    clip_qvel = np.zeros(shadow.nv)
    clip_qvel[0] = 0.5
    write_clip_pose(
        fd,
        amap,
        clip_qpos,
        clip_qvel,
        keep_xy=True,
        face_yaw=-np.pi / 2,
        travel_yaw=np.pi / 2,
    )
    assert quat_math.quat2yaw(fd.qpos[fa + 3 : fa + 7]) == pytest.approx(
        -np.pi / 2, abs=1e-5
    )
    assert fd.qvel[amap.qvel_pairs[0][0]] == pytest.approx(0.0, abs=1e-5)
    assert fd.qvel[amap.qvel_pairs[0][0] + 1] == pytest.approx(0.5, abs=1e-5)


def test_write_clip_pose_alpha_blends_hinge() -> None:
    fbvs, shadow = _tiny_pair_models()
    fd = mujoco.MjData(fbvs)
    amap = build_agent_shadow_map(fbvs, shadow, "a0_", [0])
    knee = amap.qpos_pairs[1][0]
    fd.qpos[knee] = 0.0
    clip_qpos = np.zeros(shadow.nq)
    clip_qpos[3] = 1.0
    clip_qpos[7] = 0.4
    clip_qvel = np.zeros(shadow.nv)
    write_clip_pose(fd, amap, clip_qpos, clip_qvel, alpha=0.0, act=None)
    assert fd.qpos[knee] == pytest.approx(0.0)
    write_clip_pose(fd, amap, clip_qpos, clip_qvel, alpha=0.5, act=None)
    assert fd.qpos[knee] == pytest.approx(0.2)
    write_clip_pose(fd, amap, clip_qpos, clip_qvel, alpha=1.0, act=None)
    assert fd.qpos[knee] == pytest.approx(0.4)
