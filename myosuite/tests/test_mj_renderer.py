# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Offscreen rendering with MJRenderer and examine_policy (needs an OpenGL context)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

import myosuite.utils.video_io as video_io
from myosuite.utils.policy_utils import examine_policy
from myosuite.viz.mj_renderer import MJRenderer
from myosuite import make_env

pytestmark = pytest.mark.tier2

_XML = """
<mujoco>
  <asset>
    <material name="mirror" rgba=".5 .5 .5 1" reflectance="0.7"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 .1" material="mirror"/>
    <body pos="0 0 .5"><freejoint/><geom type="sphere" size=".2" rgba="0 1 0 1"/></body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(autouse=True)
def _require_gl() -> None:
    try:
        renderer = mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"), 8, 8)
    except Exception as exc:  # noqa: BLE001 - any GL/backend failure means "no context"
        pytest.skip(f"no OpenGL context for offscreen rendering: {exc}")
    renderer.close()


@pytest.fixture()
def scene() -> tuple[mujoco.MjModel, mujoco.MjData, MJRenderer]:
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = MJRenderer(model, data)
    yield model, data, renderer
    renderer.close()


def _assert_same_image(a: np.ndarray, b: np.ndarray) -> None:
    """Two renders of one scene agree up to GPU rounding noise (at most 2 of 255 levels)."""
    assert a.shape == b.shape
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    assert diff.max() <= 2


def test_offscreen_frames_follow_requested_size(scene) -> None:
    """Every call honours width/height (the first call's size used to stick)."""
    _, _, renderer = scene
    big = renderer.render_offscreen(width=320, height=240)
    small = renderer.render_offscreen(width=160, height=120)
    again = renderer.render_offscreen(width=320, height=240)
    assert big.shape == (240, 320, 3)
    assert small.shape == (120, 160, 3)
    _assert_same_image(again, big)


def test_rgb_render_leaves_model_reflectance_untouched(scene) -> None:
    """Rendering must not edit the shared physics model (it zeroed mat_reflectance)."""
    model, _, renderer = scene
    before = model.mat_reflectance.copy()
    frame = renderer.render_offscreen(width=160, height=120)
    np.testing.assert_array_equal(model.mat_reflectance, before)
    assert before.max() > 0.0
    # reflection is off in the scene, so zeroing reflectance changes no pixel
    model.mat_reflectance[:] = 0.0
    _assert_same_image(renderer.render_offscreen(width=160, height=120), frame)


def test_examine_policy_offscreen_video_is_not_black(monkeypatch, tmp_path) -> None:
    videos: dict[str, np.ndarray] = {}
    monkeypatch.setattr(
        video_io,
        "write_video",
        lambda name, frames, **kw: videos.update({name: frames}),
    )
    env = make_env("myoElbowPose1D6MRandom-v0").unwrapped
    env.seed(0)

    class _Policy:
        def get_action(self, obs):
            act = env.action_space.sample()
            return act, {"evaluation": act}

    examine_policy(
        env,
        _Policy(),
        horizon=4,
        render="offscreen",
        frame_size=(160, 120),
        output_dir=str(tmp_path),
    )
    (frames,) = videos.values()
    assert frames.shape == (4, 120, 160, 3)
    assert frames.max() > 0
    env.close()
