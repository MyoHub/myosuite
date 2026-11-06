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
from myosuite.envs.gymnasium_env import MyoGymnasiumEnv

pytestmark = pytest.mark.tier2

_XML = """
<mujoco>
  <asset>
    <material name="mirror" rgba=".5 .5 .5 1" reflectance="0.7"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <camera name="fixed" pos="0 -3 2" xyaxes="1 0 0 0 .5 1"/>
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


@pytest.mark.parametrize("size", [(160, 120), (80, 60)])
def test_mesh_upload_refreshes_every_cached_offscreen_size(
    size: tuple[int, int],
) -> None:
    """Mesh edits refresh old sizes and do not return the latest closed context."""
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><asset><mesh name="cube" vertex="'
        "-.4 -.3 -.2 -.4 -.3 .2 -.4 .3 -.2 -.4 .3 .2 "
        '.4 -.3 -.2 .4 -.3 .2 .4 .3 -.2 .4 .3 .2"/></asset>'
        '<worldbody><light pos="0 0 3"/><geom type="mesh" mesh="cube" '
        'rgba="0 1 0 1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = MyoGymnasiumEnv()
    env.model, env.data = model, data
    env._mj_renderer_compat = MJRenderer(model, data)
    renderer = env._mj_renderer_compat
    fresh = MJRenderer(model, data)
    try:
        before = {
            dimensions: renderer.render_offscreen(
                width=dimensions[0], height=dimensions[1]
            )
            for dimensions in [(160, 120), (80, 60)]
        }
        model.mesh_vert[:] *= 0.5
        env._upload_mesh_to_renderers(0)
        width, height = size
        expected = fresh.render_offscreen(width=width, height=height)
        assert np.abs(expected.astype(int) - before[size].astype(int)).max() > 10
        _assert_same_image(
            renderer.render_offscreen(width=width, height=height), expected
        )
    finally:
        fresh.close()
        env.close()


_HFIELD_XML = (
    '<mujoco><asset><hfield name="terrain" nrow="8" ncol="8" size="1 1 .5 .1"/>'
    '</asset><worldbody><light pos="0 0 3"/><camera name="top" pos="0 -2 2" '
    'xyaxes="1 0 0 0 .7 .7"/><geom type="hfield" hfield="terrain" rgba="0 1 0 1"/>'
    "</worldbody></mujoco>"
)


@pytest.mark.parametrize("path", ["legacy", "gymnasium"])
def test_hfield_upload_refreshes_cached_renderers(path: str) -> None:
    """Terrain resampled at reset (issue 307) reaches renderers created earlier."""
    from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

    model = mujoco.MjModel.from_xml_string(_HFIELD_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = MyoGymnasiumEnv()
    env.model, env.data = model, data
    if path == "legacy":
        env._mj_renderer_compat = MJRenderer(model, data)
        fresh = MJRenderer(model, data)

        def frame(renderer):
            return renderer.render_offscreen(width=160, height=120, camera_id=0)

        cached = env._mj_renderer_compat
    else:
        env._mujoco_renderer = MujocoRenderer(
            model, data, width=160, height=120, camera_id=0
        )
        fresh = MujocoRenderer(model, data, width=160, height=120, camera_id=0)

        def frame(renderer):
            return renderer.render("rgb_array").copy()

        cached = env._mujoco_renderer
    try:
        before = frame(cached)
        model.hfield_data[:] = np.linspace(0.0, 1.0, model.hfield_data.size)
        _assert_same_image(frame(cached), before)  # stale without an upload
        env._upload_hfield_to_renderers(0)
        expected = frame(fresh)
        assert np.abs(expected.astype(int) - before.astype(int)).max() > 10
        _assert_same_image(frame(cached), expected)
    finally:
        fresh.close()
        env.close()


@pytest.mark.parametrize(
    "env_id",
    [
        "myoLegRoughTerrainWalk-v0",
        "myoChallengeChaseTagP2-v0",
        "myoChallengeOslRunRandom-v0",
    ],
)
def test_terrain_reset_uploads_heightfield(env_id: str, monkeypatch) -> None:
    """Every env that resamples terrain at reset uploads the terrain heightfield."""
    env = make_env(env_id)
    uploaded: list[int] = []
    monkeypatch.setattr(env.unwrapped, "_upload_hfield_to_renderers", uploaded.append)
    try:
        env.reset(seed=0)
        model = env.unwrapped.model
        assert uploaded
        assert set(uploaded) == {int(model.geom_dataid[model.geom("terrain").id])}
    finally:
        env.close()


@pytest.mark.parametrize("camera_id", [-1, None, 100])
def test_offscreen_free_camera_follows_updated_settings(scene, camera_id) -> None:
    """Free-camera settings apply to every output mode, including cached renderers."""
    _, _, renderer = scene
    renderer.set_free_camera_settings(
        azimuth=0, elevation=-25, distance=1, lookat=[0, 0, 0.5]
    )
    before = renderer.render_offscreen(
        width=160, height=120, camera_id=camera_id, depth=True, segmentation=True
    )
    renderer.set_free_camera_settings(
        azimuth=125, elevation=-60, distance=3, lookat=[0.7, 0.4, 0.2]
    )
    after = renderer.render_offscreen(
        width=160, height=120, camera_id=camera_id, depth=True, segmentation=True
    )
    for old, new in zip(before, after, strict=True):
        assert not np.array_equal(old, new)


@pytest.mark.parametrize("camera_id", [0, "fixed"])
def test_offscreen_fixed_camera_ignores_free_camera_settings(scene, camera_id) -> None:
    """A named or indexed model camera retains its own view."""
    _, _, renderer = scene
    before = renderer.render_offscreen(width=160, height=120, camera_id=camera_id)
    renderer.set_free_camera_settings(
        azimuth=125, elevation=-60, distance=3, lookat=[0.7, 0.4, 0.2]
    )
    after = renderer.render_offscreen(width=160, height=120, camera_id=camera_id)
    _assert_same_image(before, after)


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
