# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Ordered-waypoint terms shared by the CPU waypoint env and its mjlab twin.

Every function takes the array module ``xp`` (numpy or torch) and works on one env
(no leading axis) or a batch (leading axis ``N``): positions ``(..., 2)``, yaw
``(...)``, waypoints ``(..., W, 2)`` and the next-waypoint index ``(...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WaypointRouteCfg:
    """Random route drawn at every reset: a chain of segments from the start pose.

    Attributes:
        num_waypoints: Waypoints per route (the start is not one).
        segment_length: Range of each segment length in metres.
        turn_angle: Range of the heading change before each segment, in radians.
        heading_offset: Angle from the start yaw of the tracked frame to the
            first segment (the actor's walking direction in that frame).
    """

    num_waypoints: int = 4
    segment_length: tuple[float, float] = (1.0, 2.0)
    turn_angle: tuple[float, float] = (-0.8, 0.8)
    heading_offset: float = 0.0


@dataclass(frozen=True)
class WaypointRewardCfg:
    """Weights of :func:`waypoint_reward`.

    Attributes:
        progress: Per metre of distance closed toward the active waypoint.
        arrival: Bonus for each waypoint reached.
        failure: Penalty on the step the episode fails (fall or divergence).
    """

    progress: float = 1.0
    arrival: float = 1.0
    failure: float = 10.0


def quat_yaw(xp: Any, quat: Any) -> Any:
    """Yaw (rotation about world ``z``) of ``wxyz`` quaternions ``(..., 4)``."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return xp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def select_waypoints(xp: Any, waypoints: Any, index: Any) -> Any:
    """Waypoints ``(..., K, 2)`` at integer ``index`` ``(..., K)``, by one-hot sum
    (no gather, so it is the same code for numpy and torch)."""
    slots = xp.cumsum(xp.ones_like(waypoints[..., 0]), -1) - 1
    onehot = slots[..., None, :] == index[..., None]
    return xp.sum(onehot[..., None] * waypoints[..., None, :, :], -2)


def _lookahead_index(xp: Any, next_index: Any, count: int, num: int) -> Any:
    index = xp.stack([next_index] * count, -1)
    offsets = xp.cumsum(xp.ones_like(index), -1) - 1
    return xp.clip(index + offsets, 0, num - 1)


def waypoint_targets_obs(
    xp: Any, position: Any, yaw: Any, waypoints: Any, next_index: Any, lookahead: int
) -> Any:
    """Offsets to the next *lookahead* waypoints in the heading frame, ``(..., 2K)``.

    The heading frame is the world frame rotated by *yaw*. Past the last waypoint
    the last one repeats, so the observation size does not depend on the route.
    """
    index = _lookahead_index(xp, next_index, lookahead, waypoints.shape[-2])
    delta = select_waypoints(xp, waypoints, index) - position[..., None, :]
    c, s = xp.cos(yaw)[..., None], xp.sin(yaw)[..., None]
    local = xp.stack(
        [c * delta[..., 0] + s * delta[..., 1], c * delta[..., 1] - s * delta[..., 0]],
        -1,
    )
    return local.reshape(tuple(local.shape[:-2]) + (2 * lookahead,))


def _distance(xp: Any, a: Any, b: Any) -> Any:
    d = a - b
    return xp.sqrt(xp.sum(d * d, -1))


def distance_to_next(xp: Any, position: Any, waypoints: Any, next_index: Any) -> Any:
    """Horizontal distance to the waypoint at ``next_index`` (the last one when done)."""
    index = xp.clip(next_index, 0, waypoints.shape[-2] - 1)
    return _distance(
        xp, select_waypoints(xp, waypoints, index[..., None])[..., 0, :], position
    )


def waypoint_progress(
    xp: Any,
    position: Any,
    waypoints: Any,
    next_index: Any,
    prev_distance: Any,
    arrival_radius: float,
    failed: Any,
) -> dict[str, Any]:
    """Advance the route by one control step.

    At most one waypoint is reached per step, and only the next one. A failed
    step reaches none.

    Args:
        xp: Array module.
        position: Tracked horizontal position ``(..., 2)``.
        waypoints: Route ``(..., W, 2)``.
        next_index: Index of the active waypoint ``(...)`` (``W`` when finished).
        prev_distance: Distance to the active waypoint after the previous step.
        arrival_radius: Inclusive acceptance radius in metres.
        failed: Whether this step failed ``(...)``.

    Returns:
        ``distance`` to the active waypoint, ``progress`` (``prev_distance -
        distance``), ``arrived``, the updated ``next_index``, ``prev_distance``
        for the next step (to the new active waypoint, so reaching one does not
        jump the reward) and ``solved``.
    """
    num = waypoints.shape[-2]
    finished = next_index >= num
    distance = distance_to_next(xp, position, waypoints, next_index)
    arrived = (distance <= arrival_radius) & ~failed & ~finished
    progress = xp.where(finished, 0 * distance, prev_distance - distance)
    new_index = next_index + arrived
    return {
        "distance": distance,
        "progress": progress,
        "arrived": arrived,
        "next_index": new_index,
        "prev_distance": distance_to_next(xp, position, waypoints, new_index),
        "solved": new_index >= num,
    }


def waypoint_reward(
    accessor: Any, task_state: dict[str, Any], cfg: WaypointRewardCfg, **kwargs: Any
) -> dict[str, Any]:
    """Progress toward the active waypoint, a bonus per arrival and a failure penalty.

    The return is bounded by the route length, so ending an episode early never
    pays (unlike a per-step distance penalty).

    Args:
        accessor: Environment state accessor (for the array module).
        task_state: Output of :func:`waypoint_progress` plus ``failed``.
        cfg: Reward weights.
        **kwargs: Unused.

    Returns:
        ``dense``, ``solved``, ``done`` and the components.
    """
    del kwargs
    failed, solved = task_state["failed"], task_state["solved"]
    dense = (
        cfg.progress * task_state["progress"]
        + cfg.arrival * task_state["arrived"]
        - cfg.failure * failed
    )
    return {
        "progress": task_state["progress"],
        "arrived": task_state["arrived"],
        "dense": dense,
        "solved": solved,
        "done": solved | failed,
    }


def sample_route(
    xp: Any,
    start: Any,
    start_yaw: Any,
    u_length: Any,
    u_turn: Any,
    cfg: WaypointRouteCfg,
) -> Any:
    """Route ``(..., W, 2)`` from uniform ``[0, 1)`` draws ``(..., W)`` of the backend RNG.

    Args:
        xp: Array module.
        start: Start position ``(..., 2)``.
        start_yaw: Start yaw ``(...)`` of the tracked frame.
        u_length: Draws for the segment lengths.
        u_turn: Draws for the heading changes.
        cfg: Route distribution.

    Returns:
        The waypoints.
    """
    lo, hi = cfg.segment_length
    length = lo + (hi - lo) * u_length
    t_lo, t_hi = cfg.turn_angle
    heading = (
        start_yaw[..., None]
        + cfg.heading_offset
        + xp.cumsum(t_lo + (t_hi - t_lo) * u_turn, -1)
    )
    step = xp.stack([length * xp.cos(heading), length * xp.sin(heading)], -1)
    return start[..., None, :] + xp.cumsum(step, -2)


def perturb_qpos(xp: Any, qpos: Any, u: Any, noise: Any, low: Any, high: Any) -> Any:
    """Reset pose plus ``noise * U(-1, 1)`` per coordinate, clipped to ``[low, high]``.

    Args:
        xp: Array module.
        qpos: Reset pose ``(..., nq)``.
        u: Uniform ``[0, 1)`` draws of the backend RNG, same shape.
        noise: Per-coordinate amplitude ``(nq,)`` (zero for the free root).
        low: Lower bounds ``(nq,)`` (``-inf`` where unlimited).
        high: Upper bounds ``(nq,)`` (``inf`` where unlimited).

    Returns:
        The perturbed pose.
    """
    return xp.clip(qpos + noise * (2 * u - 1), low, high)
