# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Record an offscreen MuJoCo rollout as video frames.

Pair with :func:`myosuite.utils.video_io.write_video` to save the frames::

    frames = record_rollout(model, data, step, n_steps=300, camera=tracking_camera(model))
    write_video("rollout.mp4", frames, fps=round(1 / (model.opt.timestep * frame_skip)))
"""

from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np


def offscreen_size(model: mujoco.MjModel, width: int, height: int) -> tuple[int, int]:
    """Clamp a requested frame size to the model's offscreen framebuffer.

    Args:
        model: Compiled model (``model.vis.global_.offwidth/offheight`` bound the size).
        width: Requested width in pixels.
        height: Requested height in pixels.

    Returns:
        ``(width, height)`` that ``mujoco.Renderer`` accepts for *model*.
    """
    max_w = int(model.vis.global_.offwidth)
    max_h = int(model.vis.global_.offheight)
    w = min(width, max_w) if max_w > 0 else width
    h = min(height, max_h) if max_h > 0 else height
    return max(1, int(w)), max(1, int(h))


def tracking_camera(
    model: mujoco.MjModel,
    body: str | int | None = None,
    *,
    distance: float = 3.5,
    azimuth: float = 135.0,
    elevation: float = -20.0,
) -> mujoco.MjvCamera:
    """Camera that follows a body, by default the one carrying the free root joint.

    Args:
        model: Compiled model.
        body: Body name or id to follow (``None``: the free root body, else the world).
        distance: Distance from the body in metres.
        azimuth: Azimuth in degrees.
        elevation: Elevation in degrees (negative looks down).

    Returns:
        A ``mjCAMERA_TRACKING`` camera for :func:`record_rollout`.

    Raises:
        ValueError: If ``body`` is a name the model does not have.
    """
    if body is None:
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        body_id = int(model.jnt_bodyid[free[0]]) if free.size else 0
    elif isinstance(body, str):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if body_id < 0:
            raise ValueError(f"Unknown body {body!r}")
    else:
        body_id = int(body)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = body_id
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def record_rollout(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step: Callable[[int], bool | None],
    n_steps: int,
    *,
    width: int = 640,
    height: int = 480,
    camera: mujoco.MjvCamera | int | str | None = None,
) -> np.ndarray:
    """Call ``step(i)`` *n_steps* times and render *data* after each call.

    Args:
        model: Compiled model.
        data: State that ``step`` advances (rendered after every call).
        step: Advances *data* by one control step; returning ``False`` stops early.
        n_steps: Maximum number of steps.
        width: Requested frame width (clamped by :func:`offscreen_size`).
        height: Requested frame height (clamped by :func:`offscreen_size`).
        camera: ``MjvCamera``, model camera id/name, or ``None`` for the free camera.

    Returns:
        ``uint8`` frames of shape ``(T, H, W, 3)``, one per completed step.
    """
    w, h = offscreen_size(model, width, height)
    frames = np.empty((int(n_steps), h, w, 3), dtype=np.uint8)
    count = 0
    renderer = mujoco.Renderer(model, height=h, width=w)
    try:
        for i in range(int(n_steps)):
            keep_going = step(i)
            if camera is None:
                renderer.update_scene(data)
            else:
                renderer.update_scene(data, camera=camera)
            renderer.render(out=frames[count])
            count += 1
            if keep_going is False:
                break
    finally:
        renderer.close()
    # Copy a short rollout so the unused tail of the buffer can be freed.
    return frames if count == len(frames) else frames[:count].copy()
