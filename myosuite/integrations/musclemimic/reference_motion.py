# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Waypoint references using the shared MuscleMimic motion container.

The procedural planner fits ankle and toe targets to terrain and accepts explicit
jump segments. The clip composer retains a simple option for gentle terrain.
Neither planner discovers obstacles or guarantees a feasible arbitrary route.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

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


@dataclass(frozen=True)
class GaitParameters:
    speed: float = 0.25
    cycle_s: float = 1.2
    swing_fraction: float = 0.4
    clearance: float = 0.12
    pelvis_height: float = 0.96
    jump_flight_s: float = 0.45
    jump_crouch: float = 0.14
    jump_tuck: float = 0.16
    second_jump_tuck: float = 0.10

    def __post_init__(self) -> None:
        """Reject invalid timing and nonfinite planning parameters."""
        values = (
            self.speed,
            self.cycle_s,
            self.swing_fraction,
            self.clearance,
            self.pelvis_height,
            self.jump_flight_s,
            self.jump_crouch,
            self.jump_tuck,
            self.second_jump_tuck,
        )
        if (
            not np.isfinite(values).all()
            or min(self.speed, self.cycle_s, self.pelvis_height, self.jump_flight_s)
            <= 0
            or not 0 < self.swing_fraction < 1
            or min(
                self.clearance, self.jump_crouch, self.jump_tuck, self.second_jump_tuck
            )
            < 0
        ):
            raise ValueError(
                "Gait timing and clearance parameters must be finite and valid"
            )


def _coupling_projection(model: mujoco.MjModel, q: np.ndarray) -> None:
    """Apply the model's own polynomial joint constraints to an offline pose."""
    ids = np.flatnonzero(model.eq_type == mujoco.mjtEq.mjEQ_JOINT)
    first, second = model.eq_obj1id[ids], model.eq_obj2id[ids]
    c = model.eq_data[ids, :5]
    for _ in range(2):
        source = np.where(second >= 0, q[model.jnt_qposadr[np.maximum(second, 0)]], 0.0)
        q[model.jnt_qposadr[first]] = c[:, 0] + source * (
            c[:, 1] + source * (c[:, 2] + source * (c[:, 3] + source * c[:, 4]))
        )


def _yaw_matrix(yaw: float) -> np.ndarray:
    """Use MuJoCo quaternion conversion for a world-Z heading."""
    quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


def plan_waypoint_reference(
    model: mujoco.MjModel,
    waypoints: np.ndarray,
    height_at: Callable[[tuple[float, float]], float],
    initial_qpos: np.ndarray,
    *,
    sites: tuple[str, ...],
    jump_segments: tuple[int, ...] = (),
    parameters: GaitParameters = GaitParameters(),
) -> tuple[MotionClip, dict]:
    """Generate a 100 Hz terrain-aware walking/jumping reference offline.

    Args:
        model: Calibrated full-body model.
        waypoints: XY route including the starting point.
        height_at: Terrain height at a world XY point.
        initial_qpos: Starting model pose.
        sites: Ordered sites for shared reference kinematics.
        jump_segments: Route-segment indices with explicit jumps (at most two).
        parameters: Gait and jump timing/clearance settings.

    Returns:
        Existing MotionClip and planner diagnostics.
    """
    m, d = model, mujoco.MjData(model)
    base = np.asarray(initial_qpos, dtype=float).copy()
    d.qpos[:] = base
    mujoco.mj_forward(m, d)
    ankle_ids = [m.site(f"ankle_{s}").id for s in ("r", "l")]
    toe_ids = [m.site(f"toe_{s}").id for s in ("r", "l")]
    offsets = [d.site_xpos[s].copy() - base[:3] for s in ankle_ids]
    toe_vectors = [
        d.site_xpos[t].copy() - d.site_xpos[a] for a, t in zip(ankle_ids, toe_ids)
    ]
    spawn_surface = height_at(tuple(base[:2]))
    ankle_z = [float(d.site_xpos[s, 2] - spawn_surface) for s in ankle_ids]
    route = np.asarray(waypoints, dtype=float)
    if (
        route.ndim != 2
        or route.shape[1] != 2
        or len(route) < 2
        or not np.isfinite(route).all()
        or np.any(np.linalg.norm(np.diff(route, axis=0), axis=1) <= 1e-8)
    ):
        raise ValueError("Route needs finite, distinct consecutive XY waypoints")
    if any(i < 0 or i >= len(route) - 1 for i in jump_segments):
        raise ValueError("Jump indices must name route segments")
    if len(jump_segments) > 2:
        raise ValueError("This reference recipe supports at most two jumps")
    times = [0.0, 0.8]
    positions = [route[0], route[0]]
    jumps = []
    for i, (a, b) in enumerate(zip(route[:-1], route[1:])):
        if i in jump_segments:
            start = times[-1]
            duration = parameters.jump_flight_s
            launch, landing = a + 0.18 * (b - a), b + 0.18 * (a - b)
            times.extend(
                [
                    start + 1.7,
                    start + 2.0,
                    start + 2.0 + duration,
                    start + 2.7 + duration,
                    start + 3.2 + duration,
                ]
            )
            positions.extend([a, launch, landing, landing, b])
            jumps.append((start + 2.0, start + 2.0 + duration))
        else:
            length = np.linalg.norm(b - a)
            times.append(times[-1] + length / parameters.speed)
            positions.append(b)
            if i + 1 < len(route) - 1:
                before, after = b - a, route[i + 2] - b
                if np.dot(before, after) < 0.8 * np.linalg.norm(
                    before
                ) * np.linalg.norm(after):
                    times.append(times[-1] + 0.8)
                    positions.append(b)
    times.append(times[-1] + 1.0)
    positions.append(route[-1])
    t = np.arange(int(np.ceil(times[-1] * 100)) + 1) * 0.01
    root = np.stack(
        [np.interp(t, times, np.asarray(positions)[:, j]) for j in range(2)], axis=1
    )
    root = gaussian_filter1d(root, 6, axis=0, mode="nearest")
    velocity = np.gradient(root, 0.01, axis=0)
    heading = np.arctan2(velocity[:, 0], -velocity[:, 1])
    previous = 0.0
    for i in range(len(heading)):
        if np.linalg.norm(velocity[i]) < 0.01:
            heading[i] = previous
        else:
            previous = heading[i]
    heading = gaussian_filter1d(np.unwrap(heading), 20)
    surface = np.asarray([height_at(tuple(xy)) for xy in root])
    surface = gaussian_filter1d(surface, 15)
    root_z = surface + parameters.pelvis_height
    root_z[:80] = np.linspace(base[2], root_z[80], 80)
    flight = np.zeros(len(t), dtype=bool)
    for begin, end in jumps:
        initial = int(round((begin - 0.5) * 100))
        launch_height = height_at(tuple(root[initial]))
        pre = (t >= begin - 0.5) & (t < begin)
        root_z[pre] = (
            launch_height
            + parameters.pelvis_height
            - parameters.jump_crouch * np.sin(np.pi * (t[pre] - begin + 0.5) / 0.5)
        )
        mask = (t >= begin) & (t <= end)
        flight |= mask
        tau = t[mask] - begin
        root_z[mask] = (
            launch_height
            + parameters.pelvis_height
            + 0.5 * 9.81 * tau * (end - begin - tau)
        )
        post = (t > end) & (t <= end + 0.7)
        landing_height = height_at(
            tuple(root[min(len(t) - 1, int(round((end + 0.3) * 100)))])
        )
        root_z[post] = (
            landing_height
            + parameters.pelvis_height
            - 0.04 * np.sin(np.pi * (t[post] - end) / 0.7) ** 2
        )
    q = np.repeat(base[None, :], len(t), axis=0)
    q[:, :2] = root
    q[:, 2] = root_z
    q[:, 3:7] = np.stack(
        [np.cos(heading / 2), np.zeros(len(t)), np.zeros(len(t)), np.sin(heading / 2)],
        axis=1,
    )
    leg_addresses = []
    for side in ("r", "l"):
        names = [
            f"{n}_{side}"
            for n in (
                "hip_flexion",
                "hip_adduction",
                "hip_rotation",
                "knee_angle",
                "ankle_angle",
                "subtalar_angle",
            )
        ]
        ids = [m.joint(n).id for n in names]
        leg_addresses.append(
            (np.asarray([m.jnt_qposadr[j] for j in ids]), m.jnt_range[ids].T)
        )

    def foot_at(time: float, side: int) -> tuple[np.ndarray, float]:
        idx = int(np.clip(round(time * 100), 0, len(t) - 1))
        r = _yaw_matrix(heading[idx])
        p = np.r_[root[idx], 0.0] + r @ np.r_[offsets[side][:2], 0.0]
        p[2] = height_at(tuple(p[:2])) + ankle_z[side]
        return p, heading[idx]

    targets = np.zeros((len(t), 2, 3))
    foot_yaw = np.zeros((len(t), 2))
    for i, time in enumerate(t):
        for side in range(2):
            phase_time = max(0.0, time - 0.8) + side * parameters.cycle_s / 2
            cycle = int(phase_time // parameters.cycle_s)
            phase = phase_time / parameters.cycle_s - cycle
            last = 0.8 + cycle * parameters.cycle_s - side * parameters.cycle_s / 2
            last = max(0.0, last)
            future = last + parameters.cycle_s
            a, ya = foot_at(last, side)
            b, yb = foot_at(future, side)
            if time < 0.8 or phase < 1 - parameters.swing_fraction:
                p, yaw = a, ya
            else:
                f = (phase - 1 + parameters.swing_fraction) / parameters.swing_fraction
                smooth = f * f * (3 - 2 * f)
                p = a + smooth * (b - a)
                p[2] += parameters.clearance * np.sin(np.pi * f)
                yaw = ya + smooth * (yb - ya)
            for jump_index, (begin, end) in enumerate(jumps):
                if begin - 2.0 <= time < begin:
                    p, yaw = foot_at(begin - 1.0, side)
                elif begin <= time <= end:
                    a, ya = foot_at(begin - 1.0, side)
                    b, yb = foot_at(end, side)
                    f = (time - begin) / (end - begin)
                    smooth = f * f * (3 - 2 * f)
                    p, yaw = a + smooth * (b - a), ya + smooth * (yb - ya)
                    tuck = (
                        parameters.jump_tuck
                        if jump_index == 0
                        else parameters.second_jump_tuck
                    )
                    p[2] = (
                        root_z[i]
                        - parameters.pelvis_height
                        + ankle_z[side]
                        + tuck * np.sin(np.pi * (time - begin) / (end - begin))
                    )
                elif end < time <= end + 0.7:
                    p, yaw = foot_at(end, side)
            targets[i, side], foot_yaw[i, side] = p, yaw
    knots = np.unique(np.r_[np.arange(0, len(t), 5), len(t) - 1])
    residuals = []
    previous_legs = [base[a].copy() for a, _ in leg_addresses]
    for i in knots:
        d.qpos[:] = q[i]
        for side, (address, bounds) in enumerate(leg_addresses):
            toe_target = (
                targets[i, side] + _yaw_matrix(foot_yaw[i, side]) @ toe_vectors[side]
            )

            def error(x: np.ndarray) -> np.ndarray:
                d.qpos[address] = x
                _coupling_projection(m, d.qpos)
                mujoco.mj_kinematics(m, d)
                return np.r_[
                    (d.site_xpos[ankle_ids[side]] - targets[i, side]) * 10,
                    (d.site_xpos[toe_ids[side]] - toe_target) * 10,
                    0.005 * (x - previous_legs[side]),
                ]

            initial = np.clip(previous_legs[side], bounds[0] + 1e-7, bounds[1] - 1e-7)
            result = least_squares(
                error, initial, bounds=bounds, max_nfev=35, ftol=1e-5, xtol=1e-5
            )
            q[i, address] = result.x
            previous_legs[side] = result.x
            residuals.append(float(np.linalg.norm(error(result.x)[:6]) / 10))
        _coupling_projection(m, q[i])
    for j in range(7, m.nq):
        q[:, j] = np.interp(t, t[knots], q[knots, j])
    for pose in q:
        _coupling_projection(m, pose)
    v = np.zeros((len(q), m.nv))
    for i in range(len(q) - 1):
        mujoco.mj_differentiatePos(m, v[i], 0.01, q[i], q[i + 1])
    v[-1] = v[-2]
    clip = motion_clip_from_states(m, q, 0.01, sites, qvel=v)
    return clip, {
        "duration_s": float(t[-1]),
        "jump_windows_s": jumps,
        "max_foot_ik_error_m": max(residuals),
        "mean_foot_ik_error_m": float(np.mean(residuals)),
        "foot_targets": targets,
        "root_surface": surface,
        "flight_reference": flight,
    }
