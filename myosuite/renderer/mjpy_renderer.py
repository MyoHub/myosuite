""" =================================================
Copyright (C) 2018 Vikash Kumar, Copyright (C) 2019 The ROBEL Authors
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

"""Rendering simulation using mujoco_py."""

import mujoco_py
import numpy as np

from myosuite.renderer.renderer import Renderer, RenderMode


class MjPyRenderer(Renderer):
    """Class for rendering mujoco_py simulations."""

    def __init__(self, sim):
        assert isinstance(sim, mujoco_py.MjSim), \
            'MjPyRenderer takes a mujoco_py MjSim object.'
        super().__init__(sim)
        self._onscreen_renderer = None
        self._offscreen_renderer = None


    def render_to_window(self):
        """Renders the simulation to a window."""
        if not self._onscreen_renderer:
            self._onscreen_renderer = mujoco_py.MjViewer(self._sim)
            self._update_camera_properties(self._onscreen_renderer.cam)
            self._update_viewer_settings(self._onscreen_renderer.vopt)

        self.refresh_window()
        # self._onscreen_renderer.cam.azimuth+=.1 # trick to rotate camera for 360 videos


    def refresh_window(self):
        """Refreshes the rendered window if one is present."""
        if self._onscreen_renderer is None:
            return
        self._onscreen_renderer.render()


    def render_offscreen(self,
                         width: int,
                         height: int,
                        #  mode: RenderMode = RenderMode.RGB,
                         depth: bool = False,
                         segmentation: bool = False,
                         camera_id: int = -1,
                         device_id=-1) -> np.ndarray:
        """Renders the camera view as a numpy array of pixels.

        Args:
            width: The viewport width (pixels).
            height: The viewport height (pixels).
            mode: The rendering mode.
            camera_id: The ID of the camera to render from. By default, uses
                the free camera.

        Returns:
            A numpy array of the pixels.
        """
        assert width > 0 and height > 0
        render_out = self._sim.render(width=width, height=height, mode='offscreen', camera_name=camera_id, depth=depth, segmentation=segmentation, device_id=device_id)
        # flip image upside down
        if type(render_out) == tuple:
            rgb = render_out[0][::-1, :, :]
            return (rgb, render_out[1])
        else:
            render_out = render_out[::-1, :, :]
            return render_out

        if not self._offscreen_renderer:
            self._offscreen_renderer = mujoco_py.MjRenderContextOffscreen(self._sim, device_id=device_id)

        # Update the camera configuration for the free-camera.
        if camera_id == -1:
            self._update_camera_properties(self._offscreen_renderer.cam)
        # elif type(camera_id) == str:
            # camera_id = self._sim.model.camera_name2id(camera_id)

        # return self._offscreen_renderer.read_pixels(width, height)
        return self._offscreen_renderer.read_pixels(width, height, depth=depth, segmentation=segmentation)

        self._offscreen_renderer.render(width, height, camera_id)
        if mode == RenderMode.RGB:
            data = self._offscreen_renderer.read_pixels(
                width, height, depth=False)
            # Original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == RenderMode.DEPTH:
            data = self._offscreen_renderer.read_pixels(
                width, height, depth=True)[1]
            # Original image is upside-down, so flip it
            return data[::-1, :]
        else:
            raise NotImplementedError(mode)

    def _update_viewer_settings(self, viewer):
        """Updates the given camera object with the current camera settings."""
        for key, value in self._viewer_settings.items():
            if key == 'render_tendon':
                viewer.flags[7] = value
            if key == 'render_actuator':
                viewer.flags[3] = value # mujoco
