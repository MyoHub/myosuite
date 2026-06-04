# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

# Rendering simulation using mujoco.

import time
from typing import Any

import mujoco
import numpy as np
from mujoco import viewer

from myosuite.viz.renderer import Renderer

# Default window dimensions.
DEFAULT_WINDOW_WIDTH = 640
DEFAULT_WINDOW_HEIGHT = 480

# Default window title.
DEFAULT_WINDOW_TITLE = "MyoSuite Viewer"


def _tune_mjv_scene_for_rgb(model: mujoco.MjModel, scene: Any) -> None:
    """Align offscreen ``mjvScene`` flags with MuscleMimic-friendly RGB.

    Disables shadow, reflection, and skybox rendering flags so dense musculoskeletal
    geometry does not black out the frame. Zeros all material reflectances on *model* for stable GL.

    Args:
        model: MuJoCo model whose ``mat_reflectance`` is cleared when materials exist.
        scene: Scene struct from ``mujoco.Renderer.scene`` after ``update_scene``.
    """
    if int(model.nmat) > 0:
        model.mat_reflectance[:] = 0.0
    scene.flags[int(mujoco.mjtRndFlag.mjRND_SHADOW)] = 0
    scene.flags[int(mujoco.mjtRndFlag.mjRND_REFLECTION)] = 0
    scene.flags[int(mujoco.mjtRndFlag.mjRND_SKYBOX)] = 0


class MJRenderer(Renderer):
    """Renders Mujoco Physics objects."""

    def __init__(self, mj_model, mj_data):
        super().__init__(mj_model, mj_data)
        self._window = None
        self._renderer = None
        self._paused = False
        self._user_exit = False

    # viewer callback
    def key_callback(self, keycode):
        if chr(keycode) == " ":
            self._paused = not self._paused

        # Escape
        if keycode == 256:
            self._user_exit = True

    def setup_renderer(self, model, height, width):
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._scene_option = mujoco.MjvOption()
        self._update_renderer_settings(self._scene_option)

    def render_to_window(self):
        """Renders the Physics object to a window.

        The window continuously renders the Physics in a separate thread.

        This function is a no-op if the window was already created.
        """
        if not self._window and not self._user_exit:
            self._window = viewer.launch_passive(
                self._mj_model, self._mj_data, key_callback=self.key_callback
            )
            self._update_camera_properties(self._window.cam)
            self._update_viewer_settings(self._window.opt)

        # self._window.cam.azimuth+=.1 # trick to rotate camera for 360 videos
        self.refresh_window()

    def refresh_window(self):
        """Refreshes the rendered window if one is present."""
        if self._window is None:
            return
        self._window.sync()

        # Keep checking to unpause if paused
        while self._paused and not self._user_exit:
            # print("paused")
            time.sleep(0.2)

        if self._user_exit:
            self.close()

    def render_offscreen(
        self,
        width: int = DEFAULT_WINDOW_WIDTH,
        height: int = DEFAULT_WINDOW_HEIGHT,
        rgb: bool = True,
        depth: bool = False,
        segmentation: bool = False,
        camera_id: int | str = -1,
        device_id=-1,
    ) -> np.ndarray:
        """Renders the camera view as a numpy array of pixels.

        Args:
            width: The viewport width (pixels).
            height: The viewport height (pixels).
            mode: The rendering mode.
            camera_id: The ID of the camera to render from. By default uses the free camera.

        Returns:
            A numpy array of the pixels.
        """
        if camera_id is None:
            camera_id = -1
        if self._renderer is None:
            self.setup_renderer(self._mj_model, width=width, height=height)

        rgb_arr = None
        dpt_arr = None
        seg_arr = None
        if rgb:
            self._renderer.update_scene(
                self._mj_data, camera=camera_id, scene_option=self._scene_option
            )
            _tune_mjv_scene_for_rgb(self._mj_model, self._renderer.scene)
            rgb_arr = self._renderer.render()
        if depth:
            self._renderer.enable_depth_rendering()
            self._renderer.update_scene(
                self._mj_data, camera=camera_id, scene_option=self._scene_option
            )
            dpt_arr = self._renderer.render()
            self._renderer.disable_depth_rendering()
        if segmentation:
            self._renderer.enable_segmentation_rendering()
            self._renderer.update_scene(
                self._mj_data, camera=camera_id, scene_option=self._scene_option
            )
            seg_arr = self._renderer.render()
            self._renderer.disable_segmentation_rendering()

        if depth and segmentation:
            return rgb_arr, dpt_arr, seg_arr
        elif depth:
            return rgb_arr, dpt_arr
        elif segmentation:
            return rgb_arr, seg_arr
        else:
            return rgb_arr

    def _update_viewer_settings(self, viewer):
        """Updates the given camera object with the current camera settings."""
        for key, value in self._viewer_settings.items():
            if key == "render_tendon":
                viewer.flags[7] = value

            if key == "render_actuator":
                viewer.flags[4] = value

    def _update_renderer_settings(self, renderer):
        """Updates the given renderer object with the current camera settings."""
        for key, value in self._viewer_settings.items():
            if key == "render_tendon":
                renderer.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = value
            if key == "render_actuator":
                renderer.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = value
                renderer.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = value

    def close(self) -> None:
        """Cleans up any resources being used by the renderer."""
        if self._closed:
            return
        if self._window is not None:
            self._window.close()
            self._window = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()
