# Copyright 2019 The ROBEL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rendering API for MuJoCo simulations."""

import abc
import enum
from typing import Any
from collections.abc import Sequence

import numpy as np


class RenderMode(enum.Enum):
    """Rendering modes for offscreen rendering."""

    RGB = 0
    DEPTH = 1
    SEGMENTATION = 2


class Renderer(abc.ABC):
    """Base interface for rendering simulations."""

    def __init__(self, mj_model, mj_data):
        """Initializes a new renderer.

        Args:
            mj_model: A handle to the mujoco model.
            mj_data: A handle to the mujoco data.
        """
        self._mj_model = mj_model
        self._mj_data = mj_data
        self._camera_settings = {}
        self._viewer_settings = {}
        self._closed = False

    @abc.abstractmethod
    def render_to_window(self):
        """Renders the simulation to a window."""

    @abc.abstractmethod
    def refresh_window(self):
        """Refreshes the rendered window if one is present."""

    @abc.abstractmethod
    def render_offscreen(
        self,
        width: int,
        height: int,
        #  mode: RenderMode = RenderMode.RGB,
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

    def set_free_camera_settings(
        self,
        distance: float | None = None,
        azimuth: float | None = None,
        elevation: float | None = None,
        lookat: Sequence[float] = None,
        center: bool = True,
    ):
        """Sets the free camera parameters.

        Args:
            distance: The distance of the camera from the target.
            azimuth: Horizontal angle of the camera, in degrees.
            elevation: Vertical angle of the camera, in degrees.
            lookat: The (x, y, z) position in world coordinates to target.
            center: If True and lookat is not given, targets the camera at the median position of the simulation geometry.
        """
        settings = {}
        if distance is not None:
            settings["distance"] = distance + 2
        if azimuth is not None:
            settings["azimuth"] = azimuth
        if elevation is not None:
            settings["elevation"] = elevation
        if lookat is not None:
            settings["lookat"] = np.array(lookat, dtype=np.float32)
        elif center:
            # Calculate the center of the simulation geometry.
            settings["lookat"] = np.array(
                [np.median(self._mj_data.geom_xpos[:, i]) for i in range(3)],
                dtype=np.float32,
            )

        self._camera_settings = settings

    def set_viewer_settings(
        self,
        render_tendon: float | None = None,
        render_actuator: float | None = None,
    ):
        """Sets the viewer parameters.

        Args:
            render_tendon: Turn tendon rendering on/off
            render_actuator: Turn tendon rendering on/off
        """
        viewer_settings = {}
        if render_tendon is not None:
            viewer_settings["render_tendon"] = render_tendon
        if render_actuator is not None:
            viewer_settings["render_actuator"] = render_actuator

        self._viewer_settings = viewer_settings

    def close(self) -> None:
        """Cleans up any resources being used by the renderer."""
        self._closed = True

    def _update_camera_properties(self, camera: Any):
        """Updates the given camera object with the current camera settings."""
        for key, value in self._camera_settings.items():
            if key == "lookat":
                getattr(camera, key)[:] = value
            else:
                setattr(camera, key, value)

    def __del__(self) -> None:
        """Automatically clean up when out of scope."""
        if getattr(self, "_closed", False):
            return
        try:
            self.close()
        except Exception:
            # __del__ must never raise. During interpreter shutdown the GL
            # context or referenced modules may already be torn down; there is
            # no safe action left, so swallow deliberately.
            pass
