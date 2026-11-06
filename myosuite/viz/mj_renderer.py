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


def _tune_mjv_scene_for_rgb(scene: Any) -> None:
    """Align offscreen ``mjvScene`` flags with MuscleMimic-friendly RGB.

    Disables shadow, reflection, and skybox rendering flags so dense musculoskeletal
    geometry does not black out the frame. Only the scene changes: with reflection
    off, material reflectance has no effect, so the (physics) model is left as is.

    Args:
        scene: Scene struct from ``mujoco.Renderer.scene`` after ``update_scene``.
    """
    scene.flags[int(mujoco.mjtRndFlag.mjRND_SHADOW)] = 0
    scene.flags[int(mujoco.mjtRndFlag.mjRND_REFLECTION)] = 0
    scene.flags[int(mujoco.mjtRndFlag.mjRND_SKYBOX)] = 0


def _sport_court(scene: Any, model: Any, data: Any) -> None:
    """Add chase-tag identity graphics to the scene, never the physics model."""
    bodies = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"a{i}_pelvis")
        for i in range(2)
    ]
    floors = [
        g for g in scene.geoms[: scene.ngeom] if g.type == mujoco.mjtGeom.mjGEOM_PLANE
    ]
    if min(bodies) < 0 or not floors:
        return
    z = max(float(g.pos[2]) for g in floors) + 0.008
    strokes = []
    for body, color in zip(bodies, ((1.0, 0.17, 0.008, 1), (0.005, 0.55, 0.95, 1))):
        x, y = data.xpos[body, :2]
        for begin, end, radius, width in (
            (0.08, 0.85, 0.46, 0.007),
            (0.28, 0.62, 0.60, 0.0025),
        ):
            angles = np.linspace(begin, end, 49) * 2 * np.pi
            points = np.column_stack(
                (
                    x + radius * np.cos(angles),
                    y + radius * np.sin(angles),
                    np.full(49, z),
                )
            )
            strokes.extend(
                (a, b, width, color) for a, b in zip(points[:-1], points[1:])
            )
    for y in (-1.4, 1.4):
        strokes.append(([-5.8, y, z], [5.8, y, z], 0.01, (0.28, 0.34, 0.4, 1)))
    if scene.ngeom + len(strokes) > scene.maxgeom:
        return
    for start, end, width, color in strokes:
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).ravel(),
            np.asarray(color),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            width,
            np.asarray(start),
            np.asarray(end),
        )
        geom.emission = 0.6
        scene.ngeom += 1


def _sport_mjv_scene(scene: Any, model: Any, data: Any) -> None:
    """Style an RGB scene without editing its shared model or simulation data."""
    for geom in scene.geoms[: scene.ngeom]:
        if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid >= 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom.objid) or ""
            if any(part in name for part in ("fence", "corner")):
                geom.rgba[:3] = [0.035, 0.05, 0.065]
        if geom.type in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD):
            geom.rgba[:3] = [0.055, 0.12, 0.19]
            if hasattr(geom, "texid"):
                geom.texid = -1
            geom.matid = -1
            geom.reflectance = 0.0
            geom.specular = 0.1
    forward = np.asarray(scene.camera[0].forward)
    right = np.cross(forward, scene.camera[0].up)
    down = np.array([0.0, 0.0, -1.0])
    lights = [
        (0.5 * forward + down, [0.65, 0.65, 0.65]),
        (0.5 * right + down, [0.22, 0.07, 0.015]),
        (-0.5 * right + down, [0.015, 0.14, 0.24]),
    ]
    scene.nlight = len(lights)
    for index, (direction, color) in enumerate(lights):
        light = scene.lights[index]
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        light.texid = -1
        light.intensity = 1.0
        light.headlight = False
        light.castshadow = index == 0
        light.pos[:] = [0.0, 0.0, 5.0]
        light.dir[:] = direction / np.linalg.norm(direction)
        light.ambient[:] = 0.5 if index == 0 else 0.0
        light.diffuse[:] = color
        light.specular[:] = 0.08
        light.attenuation[:] = [1.0, 0.0, 0.0]
        light.cutoff = 180.0
        light.exponent = 0.0
    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    _sport_court(scene, model, data)


class MJRenderer(Renderer):
    """Renders Mujoco Physics objects."""

    def __init__(self, mj_model, mj_data):
        super().__init__(mj_model, mj_data)
        self._window = None
        self._renderer = None  # renderer of the latest offscreen call
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}  # by (w, h)
        self._scene_option = None
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
        """Select the offscreen renderer for ``width`` x ``height``, creating it once.

        A ``mujoco.Renderer`` has a fixed image size, so one is kept per size
        (e.g. video frames and visual observations at different resolutions).
        """
        size = (int(width), int(height))
        if size not in self._renderers:
            self._renderers[size] = mujoco.Renderer(
                model, height=size[1], width=size[0]
            )
        self._renderer = self._renderers[size]
        if self._scene_option is None:
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
        *,
        scene: str = "default",
    ) -> np.ndarray:
        """Renders the camera view as a numpy array of pixels.

        Args:
            width: The viewport width (pixels).
            height: The viewport height (pixels).
            mode: The rendering mode.
            camera_id: The ID of the camera to render from. By default uses the free camera.
            scene: ``"default"`` or ``"sport"`` RGB styling. Depth and segmentation
                retain the model's geometry and camera in either case.

        Returns:
            A numpy array of the pixels.
        """
        if scene not in ("default", "sport"):
            raise ValueError(f"Unknown offscreen scene: {scene}")
        if camera_id is None:
            camera_id = -1
        if isinstance(camera_id, int) and camera_id >= 0:
            ncam = int(self._mj_model.ncam)
            if ncam == 0 or camera_id >= ncam:
                camera_id = -1
        camera = camera_id
        if camera_id == -1 and self._camera_settings:
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self._mj_model, camera)
            self._update_camera_properties(camera)
        self.setup_renderer(self._mj_model, width=width, height=height)

        rgb_arr = None
        dpt_arr = None
        seg_arr = None
        if rgb:
            self._renderer.update_scene(
                self._mj_data, camera=camera, scene_option=self._scene_option
            )
            _tune_mjv_scene_for_rgb(self._renderer.scene)
            if scene == "sport":
                _sport_mjv_scene(self._renderer.scene, self._mj_model, self._mj_data)
            rgb_arr = self._renderer.render()
        if depth:
            self._renderer.enable_depth_rendering()
            self._renderer.update_scene(
                self._mj_data, camera=camera, scene_option=self._scene_option
            )
            dpt_arr = self._renderer.render()
            self._renderer.disable_depth_rendering()
        if segmentation:
            self._renderer.enable_segmentation_rendering()
            self._renderer.update_scene(
                self._mj_data, camera=camera, scene_option=self._scene_option
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

    def invalidate_offscreen(self) -> None:
        """Release cached offscreen contexts so runtime asset edits are reloaded."""
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
        self._renderer = None

    def close(self) -> None:
        """Cleans up any resources being used by the renderer."""
        if self._closed:
            return
        if self._window is not None:
            self._window.close()
            self._window = None
        self.invalidate_offscreen()
        super().close()
