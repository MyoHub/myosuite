# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Offscreen rollout recording helpers and ``MotionClip.to_mjdata``."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite.core.trajectory_io import MotionClip
from myosuite.viz.rollout_video import offscreen_size, record_rollout, tracking_camera

_XML = """
<mujoco>
  <visual><global offwidth="320" offheight="200"/></visual>
  <worldbody>
    <body name="floor_marker"><geom size=".05"/></body>
    <body name="root" pos="0 0 1">
      <freejoint/>
      <geom size=".1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_XML)


@pytest.mark.tier1
def test_offscreen_size_clamps_to_the_framebuffer(model: mujoco.MjModel) -> None:
    assert offscreen_size(model, 640, 480) == (320, 200)
    assert offscreen_size(model, 100, 50) == (100, 50)


@pytest.mark.tier1
def test_tracking_camera_follows_the_free_root_by_default(
    model: mujoco.MjModel,
) -> None:
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "root")
    assert tracking_camera(model).trackbodyid == root
    assert tracking_camera(model, "floor_marker").trackbodyid == 1
    with pytest.raises(ValueError):
        tracking_camera(model, "missing")


@pytest.mark.tier1
def test_clip_to_mjdata_poses_and_forwards(model: mujoco.MjModel) -> None:
    qpos = np.zeros((3, model.nq))
    qpos[:, 3] = 1.0
    qpos[2, :3] = [0.5, -0.2, 0.7]
    qvel = np.zeros((3, model.nv))
    qvel[2, 0] = 1.5
    clip = MotionClip(qpos=qpos, qvel=qvel, site_xpos=None, site_names=None)
    data = clip.to_mjdata(model, frame=2)
    np.testing.assert_allclose(data.qpos, qpos[2])
    np.testing.assert_allclose(data.qvel, qvel[2])
    # mj_forward ran: the root body's world position follows qpos.
    np.testing.assert_allclose(data.xpos[2], [0.5, -0.2, 0.7])


@pytest.mark.tier2
def test_record_rollout_renders_each_step_and_stops_early(
    model: mujoco.MjModel,
) -> None:
    try:
        mujoco.Renderer(model, height=8, width=8).close()
    except Exception as exc:  # noqa: BLE001 -- no OpenGL context on this machine
        pytest.skip(f"offscreen rendering unavailable: {exc}")
    data = mujoco.MjData(model)
    calls: list[int] = []

    def step(i: int) -> bool:
        calls.append(i)
        mujoco.mj_step(model, data)
        return i < 4

    frames = record_rollout(
        model,
        data,
        step,
        n_steps=10,
        width=640,
        height=480,
        camera=tracking_camera(model),
    )
    assert calls == [0, 1, 2, 3, 4]
    assert frames.shape == (5, 200, 320, 3)
    assert frames.base is None  # owns its data, not a view of the 10-step buffer
    assert frames.dtype == np.uint8
    assert frames.any()
