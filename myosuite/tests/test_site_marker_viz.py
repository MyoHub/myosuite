# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for site_marker_viz and the viz_fn extension to run_passive_viewer_loop.

All tests are pure-Python mocks — no MuJoCo GUI, no real model.
"""

from __future__ import annotations

import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_clip(n_frames: int = 20, n_sites: int = 4, nq: int = 10) -> object:
    """Return a minimal MotionClip-like object."""
    return types.SimpleNamespace(
        site_xpos=np.random.default_rng(0)
        .random((n_frames, n_sites, 3))
        .astype(np.float32),
        qpos=np.zeros((n_frames, nq), dtype=np.float32),
        qvel=np.zeros((n_frames, nq), dtype=np.float32),
        frequency_hz=30.0,
        source_path="test",
    )


def _make_user_scn(maxgeom: int = 50) -> object:
    """Return a minimal MjvScene-like object."""
    geoms = [types.SimpleNamespace() for _ in range(maxgeom)]
    return types.SimpleNamespace(ngeom=0, maxgeom=maxgeom, geoms=geoms)


def _make_model() -> object:
    """Return a minimal MjModel-like object."""
    return types.SimpleNamespace()


# ---------------------------------------------------------------------------
# SiteMarkerViz tests
# ---------------------------------------------------------------------------


class TestSiteMarkerVizInit(unittest.TestCase):
    def test_raises_when_site_xpos_none(self) -> None:
        from myosuite.viz.site_marker_viz import SiteMarkerViz

        clip = _make_clip()
        clip = types.SimpleNamespace(**{**vars(clip), "site_xpos": None})
        with self.assertRaises(ValueError, msg="site_xpos"):
            SiteMarkerViz(clip=clip, site_names=(), model=_make_model())

    def test_constructs_successfully(self) -> None:
        from myosuite.viz.site_marker_viz import SiteMarkerViz

        clip = _make_clip(n_sites=3)
        viz = SiteMarkerViz(clip=clip, site_names=("a", "b", "c"), model=_make_model())
        assert viz is not None


class TestSiteMarkerVizDraw(unittest.TestCase):
    def setUp(self) -> None:
        from myosuite.viz.site_marker_viz import SiteMarkerViz

        self.clip = _make_clip(n_frames=10, n_sites=4)
        self.model = _make_model()
        self.viz = SiteMarkerViz(
            clip=self.clip,
            site_names=("s0", "s1", "s2", "s3"),
            model=self.model,
        )

    def test_draw_sets_ngeom(self) -> None:
        user_scn = _make_user_scn(maxgeom=50)
        with patch("mujoco.mjv_initGeom") as mock_init:
            mock_init.return_value = None
            self.viz.draw(0, user_scn)
        assert user_scn.ngeom == 4  # n_sites

    def test_draw_resets_ngeom_each_call(self) -> None:
        user_scn = _make_user_scn(maxgeom=50)
        user_scn.ngeom = 99  # stale value from previous frame
        with patch("mujoco.mjv_initGeom"):
            self.viz.draw(0, user_scn)
        assert user_scn.ngeom == 4

    def test_draw_clamps_frame_idx_low(self) -> None:
        user_scn = _make_user_scn(maxgeom=50)
        with patch("mujoco.mjv_initGeom") as mock_init:
            mock_init.return_value = None
            self.viz.draw(-5, user_scn)  # should clamp to 0
        assert user_scn.ngeom == 4

    def test_draw_clamps_frame_idx_high(self) -> None:
        user_scn = _make_user_scn(maxgeom=50)
        with patch("mujoco.mjv_initGeom") as mock_init:
            mock_init.return_value = None
            self.viz.draw(9999, user_scn)  # should clamp to n_frames-1
        assert user_scn.ngeom == 4

    def test_draw_respects_maxgeom(self) -> None:
        user_scn = _make_user_scn(maxgeom=2)  # fewer slots than sites
        with patch("mujoco.mjv_initGeom") as mock_init:
            mock_init.return_value = None
            self.viz.draw(0, user_scn)
        assert user_scn.ngeom == 2  # capped at maxgeom

    def test_draw_passes_correct_pos(self) -> None:
        user_scn = _make_user_scn(maxgeom=50)
        captured_positions: list[np.ndarray] = []

        def _capture(*args: object) -> None:
            # mjv_initGeom(geom, type, size, pos, mat, rgba)
            pos = args[3]
            captured_positions.append(np.array(pos, dtype=np.float64))

        with patch("mujoco.mjv_initGeom", side_effect=_capture):
            self.viz.draw(3, user_scn)

        assert len(captured_positions) == 4
        for site_i, pos in enumerate(captured_positions):
            expected = self.clip.site_xpos[3, site_i].astype(np.float64)
            np.testing.assert_allclose(pos, expected, rtol=1e-5)

    def test_draw_uses_configured_rgba(self) -> None:
        from myosuite.viz.site_marker_viz import SiteMarkerViz

        custom_rgba = (0.0, 1.0, 0.0, 1.0)
        viz = SiteMarkerViz(
            clip=self.clip,
            site_names=("s0", "s1", "s2", "s3"),
            model=self.model,
            rgba=custom_rgba,
        )
        user_scn = _make_user_scn(maxgeom=50)
        captured_rgba: list[np.ndarray] = []

        def _capture(*args: object) -> None:
            rgba = args[5]
            captured_rgba.append(np.array(rgba, dtype=np.float64))

        with patch("mujoco.mjv_initGeom", side_effect=_capture):
            viz.draw(0, user_scn)

        assert len(captured_rgba) == 4
        for rgba in captured_rgba:
            np.testing.assert_allclose(rgba, np.array(custom_rgba), rtol=1e-6)

    def test_draw_uses_sphere_geom_type(self) -> None:
        import mujoco

        user_scn = _make_user_scn(maxgeom=50)
        captured_types: list[int] = []

        def _capture(*args: object) -> None:
            captured_types.append(int(args[1]))

        with patch("mujoco.mjv_initGeom", side_effect=_capture):
            self.viz.draw(0, user_scn)

        assert all(t == int(mujoco.mjtGeom.mjGEOM_SPHERE) for t in captured_types)


# ---------------------------------------------------------------------------
# SiteMarkerVizFn tests
# ---------------------------------------------------------------------------


class TestSiteMarkerVizFn(unittest.TestCase):
    def test_as_viz_fn_returns_callable(self) -> None:
        from myosuite.viz.site_marker_viz import SiteMarkerViz

        clip = _make_clip(n_sites=2)
        viz = SiteMarkerViz(clip=clip, site_names=("a", "b"), model=_make_model())
        fn = viz.as_viz_fn()
        assert callable(fn)

    def test_viz_fn_delegates_to_draw(self) -> None:
        from myosuite.viz.site_marker_viz import SiteMarkerViz

        clip = _make_clip(n_sites=2)
        viz = SiteMarkerViz(clip=clip, site_names=("a", "b"), model=_make_model())
        fn = viz.as_viz_fn()
        user_scn = _make_user_scn()
        with patch("mujoco.mjv_initGeom"):
            fn(5, _make_model(), types.SimpleNamespace(), user_scn)
        assert user_scn.ngeom == 2


# ---------------------------------------------------------------------------
# make_site_marker_viz_fn factory tests
# ---------------------------------------------------------------------------


class TestMakeSiteMarkerVizFn(unittest.TestCase):
    def test_returns_callable(self) -> None:
        from myosuite.viz.site_marker_viz import make_site_marker_viz_fn

        clip = _make_clip(n_sites=3)
        fn = make_site_marker_viz_fn(
            clip=clip,
            site_names=("a", "b", "c"),
            model=_make_model(),
        )
        assert callable(fn)

    def test_custom_radius_propagates(self) -> None:
        from myosuite.viz.site_marker_viz import (
            SiteMarkerVizFn,
            make_site_marker_viz_fn,
        )

        clip = _make_clip(n_sites=2)
        fn = make_site_marker_viz_fn(
            clip=clip,
            site_names=("a", "b"),
            model=_make_model(),
            sphere_radius=0.07,
        )
        assert isinstance(fn, SiteMarkerVizFn)
        np.testing.assert_allclose(fn._viz.sphere_radius, 0.07)


# ---------------------------------------------------------------------------
# run_passive_viewer_loop viz_fn integration test
# ---------------------------------------------------------------------------


class TestRunPassiveViewerLoopVizFn(unittest.TestCase):
    """Verify viz_fn is called inside viewer.lock() each step."""

    def test_viz_fn_called_each_step(self) -> None:
        from myosuite.core.mujoco_playback import run_passive_viewer_loop

        model = MagicMock()
        model.opt.timestep = 0.0
        data = MagicMock()

        viz_calls: list[tuple[int, object]] = []

        def _viz_fn(
            step_idx: int,
            _m: object,
            _d: object,
            user_scn: object,
        ) -> None:
            viz_calls.append((step_idx, user_scn))

        # Mock the passive viewer context manager
        mock_user_scn = MagicMock()
        mock_viewer = MagicMock()
        mock_viewer.is_running.side_effect = [True, True, True, False]
        mock_viewer.user_scn = mock_user_scn

        # viewer.lock() returns a context manager
        mock_lock_cm = MagicMock()
        mock_lock_cm.__enter__ = MagicMock(return_value=None)
        mock_lock_cm.__exit__ = MagicMock(return_value=False)
        mock_viewer.lock.return_value = mock_lock_cm

        mock_viewer.__enter__ = MagicMock(return_value=mock_viewer)
        mock_viewer.__exit__ = MagicMock(return_value=False)

        with patch("mujoco.viewer.launch_passive", return_value=mock_viewer):
            run_passive_viewer_loop(
                model=model,
                data=data,
                n_steps=3,
                step_fn=lambda i, m, d: None,
                viz_fn=_viz_fn,
            )

        assert len(viz_calls) == 3
        assert [step for step, _ in viz_calls] == [0, 1, 2]
        # Each call received the user_scn from viewer
        assert all(scn is mock_user_scn for _, scn in viz_calls)

    def test_viz_fn_none_no_lock_called(self) -> None:
        """When viz_fn is None, viewer.lock() must not be called."""
        from myosuite.core.mujoco_playback import run_passive_viewer_loop

        model = MagicMock()
        model.opt.timestep = 0.0
        data = MagicMock()

        mock_viewer = MagicMock()
        mock_viewer.is_running.side_effect = [True, False]
        mock_viewer.__enter__ = MagicMock(return_value=mock_viewer)
        mock_viewer.__exit__ = MagicMock(return_value=False)

        with patch("mujoco.viewer.launch_passive", return_value=mock_viewer):
            run_passive_viewer_loop(
                model=model,
                data=data,
                n_steps=1,
                step_fn=lambda i, m, d: None,
                viz_fn=None,
            )

        mock_viewer.lock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
