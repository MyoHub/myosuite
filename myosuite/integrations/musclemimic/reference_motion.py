# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reference motions for TERRA: a walking clip laid along a waypoint route.

TERRA tracks a full-body reference; it does not plan. :func:`compose_waypoint_reference`
is a simple planner: it loops one gait cycle of a straight walking clip, steers it
along a smoothed path through the waypoints and lifts the root by the terrain height
under it. Feet are not re-targeted onto steps, so it suits small steps and gaps only.
"""

from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from myosuite.core.trajectory_io import (
    MotionClip,
    expand_motion_clip_to_model,
    motion_clip_from_states,
)
from myosuite.physics.quat_math import quat2yaw


def _yaw_quat(yaw: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], -1)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(np.broadcast_shapes(a.shape, b.shape))
    for i in np.ndindex(out.shape[:-1]):
        mujoco.mju_mulQuat(
            out[i], a[i if a.ndim > 1 else ()], b[i if b.ndim > 1 else ()]
        )
    return out


def gait_cycle(clip_qpos: np.ndarray, skip: int, min_len: int) -> tuple[int, int]:
    """Frames ``[a, b)`` of one gait cycle: the closest joint-pose return after *min_len*."""
    joints = clip_qpos[:, 7:]
    a = skip
    tail = joints[a + min_len :]
    if len(tail) == 0:
        raise ValueError("Clip too short for a gait cycle")
    b = a + min_len + int(np.argmin(np.linalg.norm(tail - joints[a], axis=1)))
    return a, b


def _box_filter(x: np.ndarray, width: int, reduce: Callable = np.mean) -> np.ndarray:
    padded = np.concatenate([np.repeat(x[:1], width), x, np.repeat(x[-1:], width)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * width + 1)
    return reduce(windows, axis=-1)


def _smooth_path(points: np.ndarray, spacing: float, radius: float) -> np.ndarray:
    """Dense polyline through *points* with corners rounded over ``radius`` metres."""
    seg = np.diff(points, axis=0)
    length = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0], np.cumsum(length)])
    grid = np.arange(0, s[-1] + spacing, spacing)
    dense = np.column_stack([np.interp(grid, s, points[:, k]) for k in range(2)])
    width = max(1, int(round(radius / spacing)))
    return np.column_stack([_box_filter(dense[:, k], width) for k in range(2)])


def compose_waypoint_reference(
    model: mujoco.MjModel,
    clip: MotionClip,
    start_xy: np.ndarray,
    waypoints: np.ndarray,
    dt: float,
    terrain_height: Callable[[np.ndarray], np.ndarray],
    sites: tuple[str, ...],
    *,
    skip: int = 30,
    min_cycle: int = 80,
    corner_radius: float = 0.3,
    overshoot: float = 0.5,
    bridge: float = 0.3,
) -> MotionClip:
    """Walk *clip* along the route ``start -> waypoints`` at the clip's own speed.

    Args:
        model: The actor model (TERRA actor plus scene).
        clip: A straight walking clip of this actor (resampled to *dt* if needed).
        start_xy: Root start ``(x, y)``.
        waypoints: ``(W, 2)`` route.
        dt: Control step (reference frame spacing).
        terrain_height: ``f(xy (n, 2)) -> heights (n,)`` of the scene.
        sites: Mimic sites to record (TERRA order).
        skip: Clip frames to drop at the start (stance / gait onset).
        min_cycle: Shortest gait cycle in frames.
        corner_radius: Path smoothing radius at turns, metres.
        overshoot: Metres walked past the last waypoint.
        bridge: Gaps narrower than twice this (metres) keep the root at the ground
            height around them; height changes ramp over the same distance.

    Returns:
        The reference, starting at the route start facing the first waypoint.
    """
    if not np.isfinite(dt) or dt <= 0 or skip < 0 or min_cycle <= 0:
        raise ValueError("dt and min_cycle must be positive; skip must be non-negative")
    route = np.vstack(
        [
            np.asarray(start_xy, float).reshape(1, 2),
            np.asarray(waypoints, float).reshape(-1, 2),
        ]
    )
    if (
        len(route) < 2
        or not np.isfinite(route).all()
        or np.any(np.linalg.norm(np.diff(route, axis=0), axis=1) <= 1e-8)
    ):
        raise ValueError("Route needs finite, distinct consecutive waypoints")
    clip = expand_motion_clip_to_model(clip, model)
    qpos = np.asarray(clip.qpos, dtype=float)
    if clip.frequency_hz and abs(clip.frequency_hz * dt - 1) > 1e-6:
        t_src = np.arange(len(qpos)) / clip.frequency_hz
        t_dst = np.arange(0, t_src[-1], dt)
        quaternions = qpos[:, 3:7].copy()
        qpos = np.column_stack(
            [np.interp(t_dst, t_src, qpos[:, k]) for k in range(qpos.shape[1])]
        )
        qpos[:, 3:7] = np.roll(
            Slerp(t_src, Rotation.from_quat(np.roll(quaternions, -1, axis=1)))(
                t_dst
            ).as_quat(),
            1,
            axis=1,
        )
    a, b = gait_cycle(qpos, skip, min_cycle)
    cycle = qpos[a:b]
    step_xy = qpos[b, :2] - qpos[a, :2]
    speed = np.linalg.norm(step_xy) / ((b - a) * dt)
    if not np.isfinite(speed) or speed <= 1e-8:
        raise ValueError("Walking clip must make forward progress")
    heading = np.arctan2(step_xy[1], step_xy[0])
    # Root offsets in the clip's walking frame (lateral sway, height) and yaw about it.
    c, s = np.cos(-heading), np.sin(-heading)
    rel = cycle[:, :2] - qpos[a, :2]
    along = c * rel[:, 0] - s * rel[:, 1]
    lateral = s * rel[:, 0] + c * rel[:, 1]
    progress = np.arange(b - a) * (np.linalg.norm(step_xy) / (b - a))
    sway = along - progress
    clip_yaw = np.array([quat2yaw(q) for q in cycle[:, 3:7]])
    yaw_rel = np.unwrap(clip_yaw) - heading
    tilt = _quat_mul(_yaw_quat(-clip_yaw), cycle[:, 3:7])

    end = route[-1] + overshoot * (route[-1] - route[-2]) / np.linalg.norm(
        route[-1] - route[-2]
    )
    path = _smooth_path(np.vstack([route, end]), spacing=0.02, radius=corner_radius)
    arc = np.concatenate(
        [[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    )
    width = max(1, int(round(bridge / 0.02)))
    ground_path = _box_filter(_box_filter(terrain_height(path), width, np.max), width)
    tangent = np.gradient(path, axis=0)
    path_yaw = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))

    n = int(arc[-1] / (speed * dt))
    t = np.arange(n)
    phase = t % (b - a)
    s_t = speed * dt * t + sway[phase]
    xy = np.column_stack([np.interp(s_t, arc, path[:, k]) for k in range(2)])
    yaw_path = np.interp(s_t, arc, path_yaw)
    normal = np.column_stack([-np.sin(yaw_path), np.cos(yaw_path)])
    xy = xy + lateral[phase, None] * normal
    out = np.repeat(cycle[:1], n, axis=0)
    out[:, 7:] = cycle[phase, 7:]
    out[:, :2] = xy
    out[:, 2] = cycle[phase, 2] + np.interp(s_t, arc, ground_path)
    out[:, 3:7] = _quat_mul(_yaw_quat(yaw_path + yaw_rel[phase]), tilt[phase])
    return motion_clip_from_states(model, out, dt, sites)
