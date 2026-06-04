# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""State machine for the 100-cube saber target pool (mocap kinematics).

Also contains ``beat_saber_mocap_translation_delta`` — the per-step mocap
translation helper previously in the deleted ``saber_target_motion`` module —
and ``pool_reset_torch``, a backend-agnostic (duck-typed) variant of
``pool_reset`` for the mjlab Torch batch path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

# Shared defaults used by both scene setup and runtime config.
#
# Tuning guide:
# - ``SABER_POOL_SPAWN_INTERVAL_STEPS``:
#     Lower -> denser stream (more targets per second), Higher -> sparser stream.
# - ``SABER_POOL_SPEED_MPS_RANGE``:
#     Per-target speed sampled uniformly in ``[min,max]`` m/s.
#     Higher values -> faster incoming notes (less reaction time).
# - ``SABER_POOL_MIN_SPAWN_ALONG_ADVANCE_SEPARATION``:
#     Target spacing in metres along the travel axis (Beat Saber cadence feel).
#     This is converted to a minimum spawn step gap using speed and ctrl_dt.
# - ``SABER_POOL_ALONG_ADVANCE_LANE_COUNT``:
#     Number of discrete lanes on the non-travel axis (visual spread).
# - ``SABER_POOL_SPAWN_PORTAL_X/Y/Z``:
#     Spawn volume bounds; for approach along +Y/-Y, widen Z for vertical variety.
SABER_TARGET_POOL_SIZE = 100
SABER_POOL_GRAVEYARD_Z = -10.0
SABER_POOL_SPAWN_INTERVAL_STEPS = 10
SABER_POOL_SPHERE_RADIUS = 0.1
SABER_POOL_KILL_PLANE_Y = 0.85  # 1.10
SABER_POOL_MISS_ZONE_Y = 0.85
SABER_POOL_IDEAL_HIT_Y_RANGE = (0.05, 0.55)
SABER_POOL_MAX_TARGET_MISSES = 3
SABER_POOL_SPAWN_PORTAL_X = (-0.25, 0.25)  # (-0.55, 0.55)
SABER_POOL_SPAWN_PORTAL_Y = (-1.5, -1.5)  # (-2.25, -1.75)
SABER_POOL_SPAWN_PORTAL_Z = (1.15, 1.15)  # (0.65, 1.45)  # (0.45, 1.65)
SABER_POOL_ADVANCE_DIRECTION = (0.0, 1.0, 0.0)
SABER_POOL_SPEED_MPS_RANGE = (0.75, 0.75)
SABER_POOL_MIN_SPAWN_ALONG_ADVANCE_SEPARATION = 1.0  # 0.25
SABER_POOL_ALONG_ADVANCE_LANE_COUNT = 9
SABER_POOL_OBSTACLE_FRACTION = 0  # 0.5  #0.05
SABER_POOL_HIT_AREA_NONE = -1
SABER_POOL_OBSTACLE_CONTACT_IGNORE_SUBSTRINGS = (
    "floor",
    "ground",
    "saber",
    "target",
)
SABER_POOL_RECYCLE_CONTACT_SUBSTRINGS = (
    "helmet",
    "head",
    "cervical",
    "thorax",
    "torso",
)
SABER_POOL_INITIAL_HEALTH_STATUS = 0.5
SABER_POOL_HEALTH_HIT_DELTA = 0.01
SABER_POOL_HEALTH_MISS_DELTA = -0.15
SABER_POOL_HEALTH_OBSTACLE_DELTA = -0.15
SABER_POOL_HEALTH_BAD_CUT_DELTA = -0.10


def graveyard_offset_xy(index: int, pool_size: int) -> tuple[float, float]:
    """Spread graveyard positions slightly so mocap bodies do not coincide."""
    # Grid on a small patch in XZ at y=0 plane projection — index stable layout
    n = int(np.ceil(np.sqrt(float(pool_size))))
    row = index // n
    col = index % n
    span = 0.14
    ox = -0.63 + col * span
    oy = -0.63 + row * span
    return float(ox), float(oy)


@dataclass
class SaberTargetPoolConfig:
    """Hyperparameters for spawn, motion, and scoring boundaries."""

    pool_size: int = SABER_TARGET_POOL_SIZE
    graveyard_z: float = SABER_POOL_GRAVEYARD_Z
    spawn_interval_steps: int = SABER_POOL_SPAWN_INTERVAL_STEPS
    sphere_radius: float = SABER_POOL_SPHERE_RADIUS
    hit_threshold: float | None = None  # default: sphere_radius
    kill_plane_y: float = SABER_POOL_KILL_PLANE_Y
    miss_zone_y: float = SABER_POOL_MISS_ZONE_Y
    ideal_hit_y_range: tuple[float, float] = SABER_POOL_IDEAL_HIT_Y_RANGE
    max_target_misses: int | None = SABER_POOL_MAX_TARGET_MISSES
    spawn_portal_x: tuple[float, float] = SABER_POOL_SPAWN_PORTAL_X
    spawn_portal_y: tuple[float, float] = SABER_POOL_SPAWN_PORTAL_Y
    spawn_portal_z: tuple[float, float] = SABER_POOL_SPAWN_PORTAL_Z
    advance_direction: tuple[float, float, float] = SABER_POOL_ADVANCE_DIRECTION
    speed_mps_range: tuple[float, float] = SABER_POOL_SPEED_MPS_RANGE
    min_spawn_along_advance_separation: float = (
        SABER_POOL_MIN_SPAWN_ALONG_ADVANCE_SEPARATION
    )
    along_advance_lane_count: int = SABER_POOL_ALONG_ADVANCE_LANE_COUNT
    body_name_prefix: str = "saber_target_"
    geom_name_prefix: str = "saber_target_geom_"
    site_suffix: str = "_site"
    recycle_on_body_or_helmet_contact: bool = False
    recycle_contact_geom_substrings: tuple[str, ...] = (
        SABER_POOL_RECYCLE_CONTACT_SUBSTRINGS
    )
    enforce_preferred_hand: bool = True
    enforce_preferred_hit_area: bool = True
    obstacle_fraction: float = SABER_POOL_OBSTACLE_FRACTION
    slicing_accuracy_epsilon: float = 1.0e-3

    def resolved_hit_threshold(self) -> float:
        """Distance threshold for site-based hits (metres)."""
        if self.hit_threshold is not None:
            return float(self.hit_threshold)
        return float(self.sphere_radius)


@dataclass
class SaberTargetPoolRuntime:
    """Mutable per-episode pool state (updated by the env each step)."""

    active: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    vel: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))
    preferred_hand: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int8)
    )
    preferred_hit_area: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int8)
    )
    is_obstacle: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    inactive_ids: list[int] = field(default_factory=list)
    step_index: int = 0
    total_spawned: int = 0
    hits: int = 0
    misses: int = 0
    hit_this_step: bool = False
    miss_zone_this_step: bool = False
    d_min_closest: float = 1e6
    offset_at_hit: float = 0.0
    hit_y_at_hit: float = 0.0
    hit_y_deviation_at_hit: float = 0.0
    face_surface_distance_at_hit: float = 0.0
    slicing_accuracy_this_step: float = 0.0
    timing_accuracy_this_step: float = 0.0
    correct_hand_this_step: bool = False
    correct_face_this_step: bool = False
    wrong_hand_this_step: bool = False
    wrong_face_this_step: bool = False
    obstacle_contact_this_step: bool = False
    correct_hand_hits: int = 0
    correct_face_hits: int = 0
    wrong_hand_hits: int = 0
    wrong_face_hits: int = 0
    obstacle_contacts: int = 0
    d_min_obstacle: float = 1e6
    spawn_lane_cursor: int = 0
    spawn_hand_cursor: int = 0
    spawn_hit_area_cursor: int = 0
    last_spawn_step: int = -10_000
    health_status: float = SABER_POOL_INITIAL_HEALTH_STATUS
    health_depleted_this_step: bool = False


@dataclass(frozen=True)
class SaberTargetSpawnPlan:
    """Backend-agnostic plan for spawning one saber target."""

    slot_index: int
    spawn_pos: np.ndarray
    velocity: np.ndarray
    is_obstacle: bool
    preferred_hand: int
    preferred_hit_area: int
    next_spawn_lane_cursor: int
    next_spawn_hand_cursor: int
    next_spawn_hit_area_cursor: int


def pool_reset(runtime: SaberTargetPoolRuntime, pool_size: int) -> None:
    """Reset counters and mark all targets inactive (caller moves mocap)."""
    runtime.active = np.zeros(pool_size, dtype=bool)
    runtime.vel = np.zeros((pool_size, 3), dtype=np.float64)
    runtime.preferred_hand = np.zeros(pool_size, dtype=np.int8)
    runtime.preferred_hit_area = np.full(
        pool_size, SABER_POOL_HIT_AREA_NONE, dtype=np.int8
    )
    runtime.is_obstacle = np.zeros(pool_size, dtype=bool)
    runtime.inactive_ids = list(range(pool_size))
    runtime.step_index = 0
    runtime.total_spawned = 0
    runtime.hits = 0
    runtime.misses = 0
    runtime.hit_this_step = False
    runtime.miss_zone_this_step = False
    runtime.d_min_closest = 1e6
    runtime.offset_at_hit = 0.0
    runtime.hit_y_at_hit = 0.0
    runtime.hit_y_deviation_at_hit = 0.0
    runtime.face_surface_distance_at_hit = 0.0
    runtime.slicing_accuracy_this_step = 0.0
    runtime.timing_accuracy_this_step = 0.0
    runtime.correct_hand_this_step = False
    runtime.correct_face_this_step = False
    runtime.wrong_hand_this_step = False
    runtime.wrong_face_this_step = False
    runtime.obstacle_contact_this_step = False
    runtime.correct_hand_hits = 0
    runtime.correct_face_hits = 0
    runtime.wrong_hand_hits = 0
    runtime.wrong_face_hits = 0
    runtime.obstacle_contacts = 0
    runtime.d_min_obstacle = 1e6
    runtime.spawn_lane_cursor = 0
    runtime.spawn_hand_cursor = 0
    runtime.spawn_hit_area_cursor = 0
    runtime.last_spawn_step = -10_000
    runtime.health_status = SABER_POOL_INITIAL_HEALTH_STATUS
    runtime.health_depleted_this_step = False


def apply_health_status_update(
    runtime: SaberTargetPoolRuntime,
    *,
    hit: bool = False,
    miss: bool = False,
    obstacle_contact: bool = False,
    wrong_hand: bool = False,
    wrong_face: bool = False,
) -> float:
    """Apply one-step Beat Saber-like health dynamics to ``runtime``.

    Health is stored as a normalized scalar in ``[0, 1]``:

    - hit: ``+1%``
    - miss: ``-15%``
    - obstacle/bomb contact: ``-15%``
    - bad cut (wrong hand or wrong face): ``-10%``
    """

    prev = float(runtime.health_status)
    updated = prev
    if hit:
        updated += SABER_POOL_HEALTH_HIT_DELTA
    if miss:
        updated += SABER_POOL_HEALTH_MISS_DELTA
    if obstacle_contact:
        updated += SABER_POOL_HEALTH_OBSTACLE_DELTA
    if wrong_hand or wrong_face:
        updated += SABER_POOL_HEALTH_BAD_CUT_DELTA
    runtime.health_status = min(1.0, max(0.0, updated))
    runtime.health_depleted_this_step = prev > 0.0 and runtime.health_status <= 0.0
    return float(runtime.health_status)


def _dominant_advance_axis(cfg: SaberTargetPoolConfig) -> int:
    """Return dominant axis index (0:X, 1:Y, 2:Z) of advance direction."""
    d = np.asarray(cfg.advance_direction, dtype=np.float64).reshape(3)
    return int(np.argmax(np.abs(d)))


def _spawn_axis_lanes(cfg: SaberTargetPoolConfig, axis: int) -> np.ndarray:
    """Return evenly spaced lanes along the selected spawn axis."""
    bounds = (
        cfg.spawn_portal_x
        if axis == 0
        else cfg.spawn_portal_y
        if axis == 1
        else cfg.spawn_portal_z
    )
    a0, a1 = float(bounds[0]), float(bounds[1])
    n = max(1, int(cfg.along_advance_lane_count))
    return np.linspace(a0, a1, num=n, dtype=np.float64)


def _select_spawn_axis_value(
    cfg: SaberTargetPoolConfig,
    runtime: SaberTargetPoolRuntime,
    active_axis_vals: list[float],
    axis: int,
) -> float:
    """Select spawn lane value on dominant advance axis."""
    lanes = _spawn_axis_lanes(cfg, axis)
    n_lanes = int(lanes.shape[0])
    start = runtime.spawn_lane_cursor % n_lanes
    min_sep_required = float(cfg.min_spawn_along_advance_separation)

    if not active_axis_vals:
        chosen = start
    else:
        # Pass 1: first RR lane that satisfies separation.
        chosen = -1
        best = start
        best_sep = -1.0
        for step in range(n_lanes):
            idx = (start + step) % n_lanes
            val = float(lanes[idx])
            sep = min(abs(val - av) for av in active_axis_vals)
            if sep > best_sep:
                best = idx
                best_sep = sep
            if sep >= min_sep_required:
                chosen = idx
                break
        # Pass 2 fallback: best-separated lane if no lane can satisfy minimum.
        if chosen < 0:
            chosen = best

    runtime.spawn_lane_cursor = (chosen + 1) % n_lanes
    return float(lanes[chosen])


def _distance_to_interval(value: float, bounds: tuple[float, float]) -> float:
    """Return the absolute deviation from a closed interval, else zero."""
    low = min(float(bounds[0]), float(bounds[1]))
    high = max(float(bounds[0]), float(bounds[1]))
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def _required_spawn_step_gap(cfg: SaberTargetPoolConfig, ctrl_dt: float) -> int:
    """Convert desired along-axis spacing (m) into minimum spawn step gap."""
    speed_hi = max(float(cfg.speed_mps_range[0]), float(cfg.speed_mps_range[1]))
    speed = max(1e-6, speed_hi)
    dt = max(1e-6, float(ctrl_dt))
    return max(
        1, int(math.ceil(float(cfg.min_spawn_along_advance_separation) / (speed * dt)))
    )


def plan_target_spawn(
    cfg: SaberTargetPoolConfig,
    *,
    available_slot_ids: list[int] | np.ndarray,
    active_axis_vals: list[float],
    np_random: np.random.Generator,
    spawn_lane_cursor: int,
    spawn_hand_cursor: int,
    spawn_hit_area_cursor: int,
) -> SaberTargetSpawnPlan | None:
    """Return the shared spawn decision for one target.

    This helper centralizes the CPU and mjlab spawn logic so both backends use
    the same lane selection, obstacle sampling, hand/face assignment, and
    velocity generation.
    """

    slot_ids = [int(slot_id) for slot_id in np.asarray(available_slot_ids).reshape(-1)]
    if not slot_ids:
        return None

    pick = int(np_random.integers(len(slot_ids)))
    slot_index = slot_ids[pick]
    axis = _dominant_advance_axis(cfg)
    lane_axis = 2 if axis != 2 else 1

    runtime = SaberTargetPoolRuntime()
    runtime.spawn_lane_cursor = int(spawn_lane_cursor)
    lane_val = _select_spawn_axis_value(cfg, runtime, active_axis_vals, lane_axis)

    spawn_x = float(np_random.choice(list(cfg.spawn_portal_x)))
    spawn_y = float(np_random.uniform(*cfg.spawn_portal_y))
    spawn_z = float(np_random.uniform(*cfg.spawn_portal_z))
    if axis == 1:
        spawn_y = (
            float(cfg.spawn_portal_y[0])
            if float(cfg.advance_direction[1]) >= 0.0
            else float(cfg.spawn_portal_y[1])
        )
    elif axis == 0:
        spawn_x = (
            float(cfg.spawn_portal_x[0])
            if float(cfg.advance_direction[0]) >= 0.0
            else float(cfg.spawn_portal_x[1])
        )
    else:
        spawn_z = (
            float(cfg.spawn_portal_z[0])
            if float(cfg.advance_direction[2]) >= 0.0
            else float(cfg.spawn_portal_z[1])
        )

    if lane_axis == 0:
        spawn_x = lane_val
    elif lane_axis == 1:
        spawn_y = lane_val
    else:
        spawn_z = lane_val

    obstacle_fraction = min(1.0, max(0.0, float(cfg.obstacle_fraction)))
    is_obstacle = bool(np_random.random() < obstacle_fraction)
    next_spawn_hand_cursor = int(spawn_hand_cursor)
    next_spawn_hit_area_cursor = int(spawn_hit_area_cursor)
    preferred_hand = -1
    preferred_hit_area = SABER_POOL_HIT_AREA_NONE
    if not is_obstacle:
        preferred_hand = next_spawn_hand_cursor % 2
        if spawn_x == float(cfg.spawn_portal_x[0]):
            preferred_hand = 1
        if spawn_x == float(cfg.spawn_portal_x[1]):
            preferred_hand = 0
        next_spawn_hand_cursor += 1

        preferred_hit_area = next_spawn_hit_area_cursor % 5
        next_spawn_hit_area_cursor += 1
        if preferred_hand == 0 and preferred_hit_area == 4:
            preferred_hit_area = 3
        elif preferred_hand == 1 and preferred_hit_area == 3:
            preferred_hit_area = 4

    direction = np.asarray(cfg.advance_direction, dtype=np.float64).reshape(3)
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    s0 = float(cfg.speed_mps_range[0])
    s1 = float(cfg.speed_mps_range[1])
    lo, hi = (s0, s1) if s0 <= s1 else (s1, s0)
    spawn_speed = float(np_random.uniform(lo, hi))

    return SaberTargetSpawnPlan(
        slot_index=slot_index,
        spawn_pos=np.array([spawn_x, spawn_y, spawn_z], dtype=np.float64),
        velocity=(direction * spawn_speed).astype(np.float64),
        is_obstacle=is_obstacle,
        preferred_hand=preferred_hand,
        preferred_hit_area=preferred_hit_area,
        next_spawn_lane_cursor=int(runtime.spawn_lane_cursor),
        next_spawn_hand_cursor=next_spawn_hand_cursor,
        next_spawn_hit_area_cursor=next_spawn_hit_area_cursor,
    )


def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point *p* to line segment *a*->*b*."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = min(1.0, max(0.0, t))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def _closest_point_on_segment(
    p: np.ndarray, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Closest point from *p* to segment *a*->*b*."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return a.copy()
    t = float(np.dot(p - a, ab) / denom)
    t = min(1.0, max(0.0, t))
    return a + t * ab


def _impact_face_index(target: np.ndarray, impact_point: np.ndarray) -> int:
    """Map impact vector to face index (0 front, 1 top, 2 bottom, 3 left, 4 right)."""
    delta = impact_point - target
    ax = abs(float(delta[0]))
    ay = abs(float(delta[1]))
    az = abs(float(delta[2]))
    if ay >= ax and ay >= az:
        return 0
    if az >= ax and az >= ay:
        return 1 if float(delta[2]) >= 0.0 else 2
    return 3 if float(delta[0]) >= 0.0 else 4


def face_surface_distance(
    face: int,
    target: np.ndarray,
    impact_point: np.ndarray,
    radius: float,
) -> float:
    """Return in-face 2D distance from the face center to the impact point."""
    rel = impact_point - target
    half_extent = max(float(radius), 1e-9)
    if face == 0:
        coords = np.array(
            [
                np.clip(float(rel[0]), -half_extent, half_extent),
                np.clip(float(rel[2]), -half_extent, half_extent),
            ],
            dtype=np.float64,
        )
    elif face in (1, 2):
        coords = np.array(
            [
                np.clip(float(rel[0]), -half_extent, half_extent),
                np.clip(float(rel[1]), -half_extent, half_extent),
            ],
            dtype=np.float64,
        )
    else:
        coords = np.array(
            [
                np.clip(float(rel[1]), -half_extent, half_extent),
                np.clip(float(rel[2]), -half_extent, half_extent),
            ],
            dtype=np.float64,
        )
    return float(np.linalg.norm(coords))


def slicing_accuracy(surface_distance: float, radius: float, epsilon: float) -> float:
    """Return exp-decayed face-center accuracy in ``[0, 1]``."""
    half_extent = max(float(radius), 1e-9)
    eps = min(max(float(epsilon), 1e-9), 1.0 - 1e-9)
    distance = max(float(surface_distance), 0.0)
    return float(np.exp(-distance * math.log(1.0 / eps) / half_extent))


def timing_accuracy(
    hit_y: float,
    miss_zone_y: float,
    radius: float,
    *,
    forward_positive_y: bool,
) -> float:
    """Return a late-hit penalty in ``[-1, 0]`` relative to the miss zone."""
    half_extent = max(float(radius), 1e-9)
    overshoot = (
        max(0.0, float(hit_y) - float(miss_zone_y))
        if forward_positive_y
        else max(0.0, float(miss_zone_y) - float(hit_y))
    )
    return -min(overshoot / half_extent, 1.0)


def place_graveyard_positions(
    data: mujoco.MjData,
    mocap_ids: np.ndarray,
    pool_size: int,
    graveyard_z: float,
) -> None:
    """Teleport every pooled mocap body to its graveyard slot."""
    for i in range(pool_size):
        mid = int(mocap_ids[i])
        ox, oy = graveyard_offset_xy(i, pool_size)
        data.mocap_pos[mid] = [ox, oy, graveyard_z]


def pool_pre_physics(
    data: mujoco.MjData,
    cfg: SaberTargetPoolConfig,
    runtime: SaberTargetPoolRuntime,
    mocap_ids: np.ndarray,
    np_random: np.random.Generator,
    ctrl_dt: float,
) -> None:
    """Advance active targets and optionally spawn before ``mj_step``."""
    n = int(runtime.active.shape[0])
    for i in range(n):
        if not runtime.active[i]:
            continue
        mid = int(mocap_ids[i])
        data.mocap_pos[mid] += runtime.vel[i] * float(ctrl_dt)

    runtime.step_index += 1
    if runtime.step_index % cfg.spawn_interval_steps != 0:
        return
    if not runtime.inactive_ids:
        return
    min_step_gap = _required_spawn_step_gap(cfg, ctrl_dt)
    if (runtime.step_index - int(runtime.last_spawn_step)) < min_step_gap:
        return
    axis = _dominant_advance_axis(cfg)
    lane_axis = 2 if axis != 2 else 1
    active_axis_vals: list[float] = []
    for i in range(n):
        if not runtime.active[i]:
            continue
        amid = int(mocap_ids[i])
        active_axis_vals.append(float(data.mocap_pos[amid][lane_axis]))
    spawn_plan = plan_target_spawn(
        cfg,
        available_slot_ids=runtime.inactive_ids,
        active_axis_vals=active_axis_vals,
        np_random=np_random,
        spawn_lane_cursor=runtime.spawn_lane_cursor,
        spawn_hand_cursor=runtime.spawn_hand_cursor,
        spawn_hit_area_cursor=runtime.spawn_hit_area_cursor,
    )
    if spawn_plan is None:
        return
    sid = int(spawn_plan.slot_index)
    runtime.inactive_ids.remove(sid)
    runtime.spawn_lane_cursor = int(spawn_plan.next_spawn_lane_cursor)
    runtime.spawn_hand_cursor = int(spawn_plan.next_spawn_hand_cursor)
    runtime.spawn_hit_area_cursor = int(spawn_plan.next_spawn_hit_area_cursor)
    mid = int(mocap_ids[sid])
    data.mocap_pos[mid] = spawn_plan.spawn_pos
    runtime.is_obstacle[sid] = bool(spawn_plan.is_obstacle)
    runtime.preferred_hand[sid] = np.int8(spawn_plan.preferred_hand)
    runtime.preferred_hit_area[sid] = np.int8(spawn_plan.preferred_hit_area)
    runtime.vel[sid] = spawn_plan.velocity.copy()
    runtime.active[sid] = True
    runtime.total_spawned += 1
    runtime.last_spawn_step = int(runtime.step_index)


def pool_post_physics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cfg: SaberTargetPoolConfig,
    runtime: SaberTargetPoolRuntime,
    mocap_ids: np.ndarray,
    site_ids: np.ndarray,
    geom_ids: np.ndarray,
    recycle_contact_geom_ids: set[int] | None = None,
) -> None:
    """Hit / miss / miss-zone processing after kinematics; recycle mocap."""
    hits_thresh = cfg.resolved_hit_threshold()
    left_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_saber_tip")
    right_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_saber_tip")
    left_base_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "left_saber_blade_base"
    )
    right_base_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "right_saber_blade_base"
    )
    if left_tip_id < 0 or right_tip_id < 0:
        return

    left_tip = data.site_xpos[left_tip_id].copy()
    right_tip = data.site_xpos[right_tip_id].copy()
    # Backward-compatible fallback: if blade-base sites are absent, keep tip-only
    # distance while still running contact and recycle logic.
    left_base = (
        data.site_xpos[left_base_id].copy() if left_base_id >= 0 else left_tip.copy()
    )
    right_base = (
        data.site_xpos[right_base_id].copy() if right_base_id >= 0 else right_tip.copy()
    )
    left_saber_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "left_saber_geom"
    )
    right_saber_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "right_saber_geom"
    )

    runtime.hit_this_step = False
    runtime.miss_zone_this_step = False
    runtime.d_min_closest = 1e6
    runtime.offset_at_hit = 0.0
    runtime.hit_y_at_hit = 0.0
    runtime.hit_y_deviation_at_hit = 0.0
    runtime.face_surface_distance_at_hit = 0.0
    runtime.slicing_accuracy_this_step = 0.0
    runtime.timing_accuracy_this_step = 0.0
    runtime.correct_hand_this_step = False
    runtime.correct_face_this_step = False
    runtime.wrong_hand_this_step = False
    runtime.wrong_face_this_step = False
    runtime.obstacle_contact_this_step = False
    runtime.d_min_obstacle = 1e6
    runtime.health_depleted_this_step = False

    n = int(runtime.active.shape[0])
    forward_positive_y = float(cfg.advance_direction[1]) >= 0.0
    to_recycle: list[int] = []

    contact_hit_targets: set[int] = set()
    saber_contact_hit_targets: set[int] = set()
    obstacle_contact_targets: set[int] = set()
    target_geom_to_index = {int(gid): idx for idx, gid in enumerate(geom_ids.tolist())}
    saber_geom_ids = {int(left_saber_geom_id), int(right_saber_geom_id)}
    saber_geom_ids.discard(-1)
    for k in range(int(data.ncon)):
        c = data.contact[k]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        i1 = target_geom_to_index.get(g1)
        i2 = target_geom_to_index.get(g2)
        if i1 is not None and g2 in saber_geom_ids:
            if bool(runtime.is_obstacle[i1]):
                obstacle_contact_targets.add(i1)
            else:
                saber_contact_hit_targets.add(i1)
        if i2 is not None and g1 in saber_geom_ids:
            if bool(runtime.is_obstacle[i2]):
                obstacle_contact_targets.add(i2)
            else:
                saber_contact_hit_targets.add(i2)

    def _counts_as_obstacle_body_contact(gid: int) -> bool:
        if gid in target_geom_to_index or gid in saber_geom_ids:
            return False
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        lname = gname.lower()
        return not any(
            key in lname for key in SABER_POOL_OBSTACLE_CONTACT_IGNORE_SUBSTRINGS
        )

    for k in range(int(data.ncon)):
        c = data.contact[k]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        i1 = target_geom_to_index.get(g1)
        i2 = target_geom_to_index.get(g2)
        if (
            i1 is not None
            and bool(runtime.is_obstacle[i1])
            and _counts_as_obstacle_body_contact(g2)
        ):
            obstacle_contact_targets.add(i1)
        if (
            i2 is not None
            and bool(runtime.is_obstacle[i2])
            and _counts_as_obstacle_body_contact(g1)
        ):
            obstacle_contact_targets.add(i2)

    if cfg.recycle_on_body_or_helmet_contact and recycle_contact_geom_ids:
        for k in range(int(data.ncon)):
            c = data.contact[k]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            i1 = target_geom_to_index.get(g1)
            i2 = target_geom_to_index.get(g2)
            if i1 is not None and g2 in recycle_contact_geom_ids:
                if bool(runtime.is_obstacle[i1]):
                    obstacle_contact_targets.add(i1)
                else:
                    contact_hit_targets.add(i1)
            if i2 is not None and g1 in recycle_contact_geom_ids:
                if bool(runtime.is_obstacle[i2]):
                    obstacle_contact_targets.add(i2)
                else:
                    contact_hit_targets.add(i2)

    processed: set[int] = set()

    def _classify_hit(
        i: int, target_pos: np.ndarray, use_left: bool
    ) -> tuple[bool, bool, np.ndarray, int]:
        required_left = int(runtime.preferred_hand[i]) == 0
        correct_hand = (not cfg.enforce_preferred_hand) or (required_left == use_left)
        if correct_hand:
            runtime.correct_hand_this_step = True
            runtime.correct_hand_hits += 1
        else:
            runtime.wrong_hand_this_step = True
            runtime.wrong_hand_hits += 1

        base = left_base if use_left else right_base
        tip = left_tip if use_left else right_tip
        closest = _closest_point_on_segment(target_pos, base, tip)
        face = _impact_face_index(target_pos, closest)
        correct_face = (not cfg.enforce_preferred_hit_area) or (
            face == int(runtime.preferred_hit_area[i])
        )
        if correct_face:
            runtime.correct_face_this_step = True
            runtime.correct_face_hits += 1
        else:
            runtime.wrong_face_this_step = True
            runtime.wrong_face_hits += 1

        return correct_hand, correct_face, closest, face

    # Robust path: apply saber-contact hits directly from contact mapping.
    for i in obstacle_contact_targets:
        if i < 0 or i >= n or (not runtime.active[i]):
            continue
        runtime.obstacle_contact_this_step = True
        runtime.obstacle_contacts += 1
        apply_health_status_update(runtime, obstacle_contact=True)
        to_recycle.append(i)
        processed.add(i)

    for i in saber_contact_hit_targets:
        if i < 0 or i >= n or (not runtime.active[i]):
            continue
        sid_site = int(site_ids[i])
        tp = data.site_xpos[sid_site]
        d_l = _point_to_segment_distance(tp, left_base, left_tip)
        d_r = _point_to_segment_distance(tp, right_base, right_tip)
        use_left = d_l <= d_r
        d_min = min(d_l, d_r)
        runtime.d_min_closest = min(runtime.d_min_closest, d_min)
        runtime.hit_this_step = True
        runtime.hits += 1
        correct_hand, correct_face, closest, face = _classify_hit(i, tp, use_left)
        radius = float(cfg.sphere_radius)
        runtime.face_surface_distance_at_hit = (
            face_surface_distance(face, tp, closest, radius) if correct_face else 0.0
        )
        runtime.slicing_accuracy_this_step = (
            slicing_accuracy(
                runtime.face_surface_distance_at_hit,
                radius,
                epsilon=cfg.slicing_accuracy_epsilon,
            )
            if correct_face
            else 0.0
        )
        runtime.hit_y_at_hit = float(tp[1])
        runtime.timing_accuracy_this_step = timing_accuracy(
            runtime.hit_y_at_hit,
            cfg.miss_zone_y,
            radius,
            forward_positive_y=forward_positive_y,
        )
        apply_health_status_update(
            runtime,
            hit=True,
            wrong_hand=not correct_hand,
            wrong_face=not correct_face,
        )
        runtime.offset_at_hit = d_min
        runtime.hit_y_deviation_at_hit = _distance_to_interval(
            runtime.hit_y_at_hit, cfg.ideal_hit_y_range
        )
        to_recycle.append(i)
        processed.add(i)

    for i in range(n):
        if not runtime.active[i]:
            continue
        if i in processed:
            continue
        if i in contact_hit_targets:
            runtime.misses += 1
            apply_health_status_update(runtime, miss=True)
            to_recycle.append(i)
            continue
        sid_site = int(site_ids[i])
        tp = data.site_xpos[sid_site]
        d_l = _point_to_segment_distance(tp, left_base, left_tip)
        d_r = _point_to_segment_distance(tp, right_base, right_tip)
        use_left = d_l <= d_r
        d_min = min(d_l, d_r)
        if bool(runtime.is_obstacle[i]):
            runtime.d_min_obstacle = min(runtime.d_min_obstacle, d_min)
        else:
            runtime.d_min_closest = min(runtime.d_min_closest, d_min)

        mid = int(mocap_ids[i])
        y_t = float(data.mocap_pos[mid][1])
        in_miss_zone = (
            y_t > cfg.miss_zone_y if forward_positive_y else y_t < cfg.miss_zone_y
        )
        if in_miss_zone:
            if not bool(runtime.is_obstacle[i]):
                runtime.miss_zone_this_step = True

        if d_min < hits_thresh:
            if bool(runtime.is_obstacle[i]):
                runtime.obstacle_contact_this_step = True
                runtime.obstacle_contacts += 1
                apply_health_status_update(runtime, obstacle_contact=True)
                to_recycle.append(i)
                continue
            runtime.hit_this_step = True
            runtime.hits += 1
            correct_hand, correct_face, closest, face = _classify_hit(i, tp, use_left)
            radius = float(cfg.sphere_radius)
            runtime.face_surface_distance_at_hit = (
                face_surface_distance(face, tp, closest, radius)
                if correct_face
                else 0.0
            )
            runtime.slicing_accuracy_this_step = (
                slicing_accuracy(
                    runtime.face_surface_distance_at_hit,
                    radius,
                    epsilon=cfg.slicing_accuracy_epsilon,
                )
                if correct_face
                else 0.0
            )
            runtime.hit_y_at_hit = float(tp[1])
            runtime.timing_accuracy_this_step = timing_accuracy(
                runtime.hit_y_at_hit,
                cfg.miss_zone_y,
                radius,
                forward_positive_y=forward_positive_y,
            )
            apply_health_status_update(
                runtime,
                hit=True,
                wrong_hand=not correct_hand,
                wrong_face=not correct_face,
            )
            runtime.offset_at_hit = d_min
            runtime.hit_y_deviation_at_hit = _distance_to_interval(
                runtime.hit_y_at_hit, cfg.ideal_hit_y_range
            )
            to_recycle.append(i)
            continue

        crossed_kill_plane = (
            y_t > cfg.kill_plane_y if forward_positive_y else y_t < cfg.kill_plane_y
        )
        if crossed_kill_plane:
            if not bool(runtime.is_obstacle[i]):
                runtime.misses += 1
                apply_health_status_update(runtime, miss=True)
            to_recycle.append(i)

    for i in set(to_recycle):
        runtime.active[i] = False
        runtime.vel[i] = 0.0
        runtime.is_obstacle[i] = False
        runtime.preferred_hand[i] = np.int8(0)
        runtime.preferred_hit_area[i] = np.int8(SABER_POOL_HIT_AREA_NONE)
        runtime.inactive_ids.append(i)
        mid = int(mocap_ids[i])
        ox, oy = graveyard_offset_xy(i, cfg.pool_size)
        data.mocap_pos[mid] = [ox, oy, cfg.graveyard_z]


def build_pool_indices(
    model: mujoco.MjModel,
    cfg: SaberTargetPoolConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve mocap/site/geom ids for ``saber_target_{i}`` bodies."""
    n = cfg.pool_size
    mocap_ids = np.zeros(n, dtype=np.int32)
    site_ids = np.zeros(n, dtype=np.int32)
    geom_ids = np.zeros(n, dtype=np.int32)
    marker_geom_ids = np.zeros((n, 5), dtype=np.int32)
    prefix = cfg.body_name_prefix
    for i in range(n):
        bname = f"{prefix}{i}"
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid < 0:
            raise ValueError(f"Missing pool body {bname!r}")
        mid = int(model.body_mocapid[bid])
        if mid < 0:
            raise ValueError(f"Body {bname!r} is not mocap")
        mocap_ids[i] = mid
        sname = f"{prefix}{i}{cfg.site_suffix}"
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname)
        if sid < 0:
            raise ValueError(f"Missing pool site {sname!r}")
        site_ids[i] = sid
        gname = f"{cfg.geom_name_prefix}{i}"
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid < 0:
            raise ValueError(f"Missing pool geom {gname!r}")
        geom_ids[i] = gid
        for face_idx, face_name in enumerate(
            ("front", "top", "bottom", "right", "left")
        ):
            mname = f"{gname}_marker_{face_name}"
            mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, mname)
            if mid < 0:
                raise ValueError(f"Missing pool geom {mname!r}")
            marker_geom_ids[i, face_idx] = mid
    return mocap_ids, site_ids, geom_ids, marker_geom_ids


def resolve_recycle_contact_geom_ids(
    model: mujoco.MjModel,
    cfg: SaberTargetPoolConfig,
    target_geom_ids: np.ndarray,
) -> set[int]:
    """Return non-target geom IDs that should recycle targets on contact."""
    target_ids = {int(g) for g in target_geom_ids.tolist()}
    selected: set[int] = set()
    keys = tuple(k.lower() for k in cfg.recycle_contact_geom_substrings)
    for gid in range(int(model.ngeom)):
        if gid in target_ids:
            continue
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        lname = gname.lower()
        if any(key in lname for key in keys):
            selected.add(int(gid))
    return selected


# ---------------------------------------------------------------------------
# Mocap kinematics helper (formerly saber_target_motion.py)
# ---------------------------------------------------------------------------


def beat_saber_mocap_translation_delta(
    direction_world: np.ndarray,
    speed_mps: float,
    dt_sec: float,
) -> np.ndarray:
    """Per-step world-frame translation for mocap strike blocks (Beat Saber style).

    Args:
        direction_world: Unit (or scaled) direction in world coordinates; the
            prototype uses ``[0, -1, 0]`` (decreasing *y*).
        speed_mps: Speed magnitude in metres per second.
        dt_sec: Control timestep in seconds.

    Returns:
        Translation vector ``(3,)`` to add to each mocap position.
    """
    d = np.asarray(direction_world, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(d))
    dhat = d / max(n, 1e-9)
    return (dhat * float(speed_mps) * float(dt_sec)).astype(np.float64)


# ---------------------------------------------------------------------------
# Torch batch reset — mirrors pool_reset() for the mjlab path
# ---------------------------------------------------------------------------


def pool_reset_torch(state: Any, env_ids: Any) -> None:
    """Backend-agnostic (duck-typed) reset of pool state tensors.

    ``state`` must expose the same attribute names as :class:`SaberTaskLogicState`
    (``pool_active``, ``pool_vel``, …).  ``env_ids`` is a Torch index tensor.

    This is the torch-batch counterpart of :func:`pool_reset` and keeps the
    reset semantics defined in one place — both the CPU path (via
    ``pool_reset``) and the mjlab path (via this function) follow the same
    initial conditions.  No data conversion is performed; all operations are
    in-place assignments on the existing tensors, so performance is identical
    to the previous inline implementation in ``SaberTaskLogicState.reset``.
    """
    state.pool_active[env_ids] = False
    state.pool_vel[env_ids] = 0.0
    state.pool_preferred_hand[env_ids] = 0
    state.pool_preferred_hit_area[env_ids] = SABER_POOL_HIT_AREA_NONE
    state.pool_is_obstacle[env_ids] = False
    state.pool_step_index[env_ids] = 0
    state.pool_total_spawned[env_ids] = 0
    state.pool_hits[env_ids] = 0
    state.pool_misses[env_ids] = 0
    state.pool_correct_hand_hits[env_ids] = 0
    state.pool_correct_face_hits[env_ids] = 0
    state.pool_wrong_hand_hits[env_ids] = 0
    state.pool_wrong_face_hits[env_ids] = 0
    state.pool_obstacle_contacts[env_ids] = 0
    state.pool_spawn_lane_cursor[env_ids] = 0
    state.pool_spawn_hand_cursor[env_ids] = 0
    state.pool_spawn_hit_area_cursor[env_ids] = 0
    state.pool_last_spawn_step[env_ids] = -10_000
    state.health_status[env_ids] = SABER_POOL_INITIAL_HEALTH_STATUS
