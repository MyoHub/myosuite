# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Judged athletics events on the public CPU athletic env (any MuJoCo plant, e.g. ``myofullbody``).

An event wraps the lane-running :class:`~myosuite.envs.modular_env.ModularTaskEnv` of its base task and adds the
event's geometry and judge:

* track events (100 m, 110 m hurdles, 1500 m): finish time over a distance, hurdles as static colliders;
* jumps (long jump, high jump): :class:`~myosuite.envs.myo.tasks.basic.full_body.jumps.JumpJudge`, fed with the
  plant's own collision capsules / ellipsoids (:func:`collider_segments`).

Single events are built with :func:`make_event`; :mod:`myosuite.envs.myo.athletic.combined` chains them into
combined events (pentathlon, decathlon).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import gymnasium as gym
import mujoco
import numpy as np

from myosuite.envs.myo.athletic.specs import (
    DEFAULT_TENDONS,
    PlantSpec,
    TendonKind,
    ensure_tendons_applied,
)
from myosuite.envs.myo.tasks.basic.full_body.jumps import (
    HIGH_JUMP,
    LONG_JUMP,
    JumpJudge,
    JumpSpec,
)

EventKind = Literal["track", "jump", "throw", "vault"]

# Feature vector appended to the base observation by every event (same width for all events, so one policy can
# run all of them): [distance to the finish / take-off line / bar (m), next hurdle distance (m), next hurdle
# height (m), bar height (m), event clock (s)].
EVENT_FEATURES = 5


@dataclass(frozen=True)
class HurdleLayout:
    """Hurdles along the lane (World Athletics TR22: men's 110 m = 10 x 1.067 m, 13.72 m, 9.14 m apart)."""

    count: int = 10
    first_m: float = 13.72
    spacing_m: float = 9.14
    height_m: float = 1.067
    width_m: float = 1.18  # TR22: 1.18-1.20 m
    bar_depth_m: float = 0.02  # top bar thickness in the running direction
    bar_height_m: float = 0.07  # top bar 70 mm

    def positions(self) -> np.ndarray:
        return self.first_m + self.spacing_m * np.arange(self.count)


@dataclass(frozen=True)
class EventSpec:
    """One athletics event: geometry, judge, time limit, and its World Athletics points table.

    Points follow the World Athletics combined-events tables: track ``A * (B - T)^C`` (T in s), field
    ``A * (M - B)^C`` (M in ``unit``), floored, 0 when the performance is outside ``B``.
    """

    key: str
    title: str
    kind: EventKind
    base_task: str = "sprint"  # lane-running task the event wraps
    distance_m: float = 0.0  # track events
    hurdles: HurdleLayout | None = None
    jump: JumpSpec | None = None
    time_limit_s: float = 30.0
    points: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (A, B, C)
    unit: Literal["s", "m", "cm"] = "s"
    lambda_finish: float = (
        10.0  # track: reward on finishing, scaled by the mean speed / 10 m/s
    )
    # capabilities the plant needs beyond running (throws need hands and an implement, vault a pole)
    requires: frozenset[str] = field(default_factory=frozenset)

    def score(self, performance: float | None) -> int:
        """World Athletics points for a performance (seconds, or metres for field events)."""
        if performance is None or not math.isfinite(performance) or performance <= 0:
            return 0
        a, b, c = self.points
        if self.kind == "track":
            gap = b - performance
        else:
            value = performance * 100.0 if self.unit == "cm" else performance
            gap = value - b
        return int(math.floor(a * gap**c)) if gap > 0 else 0


# World Athletics men's combined-events table coefficients (decathlon).
EVENTS: dict[str, EventSpec] = {
    "100m": EventSpec(
        key="100m",
        title="100 m",
        kind="track",
        distance_m=100.0,
        time_limit_s=30.0,
        points=(25.4347, 18.0, 1.81),
    ),
    "400m": EventSpec(
        key="400m",
        title="400 m",
        kind="track",
        distance_m=400.0,
        time_limit_s=120.0,
        points=(1.53775, 82.0, 1.81),
    ),
    "1500m": EventSpec(
        key="1500m",
        title="1500 m",
        kind="track",
        distance_m=1500.0,
        time_limit_s=600.0,
        points=(0.03768, 480.0, 1.85),
    ),
    "110m_hurdles": EventSpec(
        key="110m_hurdles",
        title="110 m hurdles",
        kind="track",
        distance_m=110.0,
        hurdles=HurdleLayout(),
        time_limit_s=35.0,
        points=(5.74352, 28.5, 1.92),
    ),
    "long_jump": EventSpec(
        key="long_jump",
        title="Long jump",
        kind="jump",
        jump=LONG_JUMP,
        time_limit_s=10.0,
        points=(0.14354, 220.0, 1.40),
        unit="cm",
    ),
    "high_jump": EventSpec(
        key="high_jump",
        title="High jump",
        kind="jump",
        jump=HIGH_JUMP,
        time_limit_s=10.0,
        points=(0.8465, 75.0, 1.42),
        unit="cm",
    ),
    "shot_put": EventSpec(
        key="shot_put",
        title="Shot put",
        kind="throw",
        points=(51.39, 1.5, 1.05),
        unit="m",
        requires=frozenset({"hands", "throws"}),
    ),
    "discus": EventSpec(
        key="discus",
        title="Discus throw",
        kind="throw",
        points=(12.91, 4.0, 1.10),
        unit="m",
        requires=frozenset({"hands", "throws"}),
    ),
    "javelin": EventSpec(
        key="javelin",
        title="Javelin throw",
        kind="throw",
        points=(10.14, 7.0, 1.08),
        unit="m",
        requires=frozenset({"hands", "throws"}),
    ),
    "pole_vault": EventSpec(
        key="pole_vault",
        title="Pole vault",
        kind="vault",
        points=(0.2797, 100.0, 1.35),
        unit="cm",
        requires=frozenset({"hands", "pole_vault"}),
    ),
}

RUNNABLE_KINDS = frozenset({"track", "jump"})


# --------------------------------------------------------------------------------------------------- geometry
def _task_frame(heading: tuple[float, float]) -> np.ndarray:
    """Rows: task forward, side, up in world coordinates (z-up plants)."""
    hx, hy = heading
    return np.array([[hx, hy, 0.0], [-hy, hx, 0.0], [0.0, 0.0, 1.0]])


def body_colliders(model: mujoco.MjModel, exclude_prefix: str = "event_") -> np.ndarray:
    """Ids of the plant's collision geoms the judge sees: capsules, spheres and ellipsoids on non-world bodies."""
    keep = (
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    )
    ids = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0 or model.geom_type[g] not in keep:
            continue
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if not name.startswith(exclude_prefix):
            ids.append(g)
    return np.asarray(ids, int)


def collider_radius_axis(
    model: mujoco.MjModel, ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per collider: radius, half segment length and the local axis index of the segment.

    Ellipsoids become the capsule along their longest semi-axis whose radius is the larger of the other two
    (it contains the ellipsoid, so bar and ground contacts err on the side of touching).
    """
    radius = np.zeros(len(ids))
    half = np.zeros(len(ids))
    axis = np.full(len(ids), 2, int)
    for i, g in enumerate(ids):
        s = model.geom_size[g]
        t = model.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            radius[i], half[i] = s[0], s[1]
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            radius[i] = s[0]
        else:
            k = int(np.argmax(s))
            r = float(np.max(np.delete(s, k)))
            radius[i], half[i], axis[i] = r, max(float(s[k]) - r, 0.0), k
    return radius, half, axis


def collider_segments(
    data: mujoco.MjData,
    ids: np.ndarray,
    half: np.ndarray,
    axis: np.ndarray,
    frame: np.ndarray,
    origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Collider segment ends in task axes (forward, side, up) relative to ``origin`` (world)."""
    centre = data.geom_xpos[ids]
    mats = data.geom_xmat[ids].reshape(-1, 3, 3)
    direction = mats[np.arange(len(ids)), :, axis]  # column = local axis in world
    a = centre - direction * half[:, None]
    b = centre + direction * half[:, None]
    return (a - origin) @ frame.T, (b - origin) @ frame.T


def foot_labels(model: mujoco.MjModel, ids: np.ndarray) -> np.ndarray:
    """0 = not a foot, 1 = right foot, 2 = left foot (geoms on ``calcn_*`` or its descendants)."""
    roots = {}
    for side, label in (("r", 1), ("l", 2)):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"calcn_{side}")
        if bid >= 0:
            roots[bid] = label
    out = np.zeros(len(ids), int)
    for i, g in enumerate(ids):
        b = int(model.geom_bodyid[g])
        while b > 0:
            if b in roots:
                out[i] = roots[b]
                break
            b = int(model.body_parentid[b])
    return out


def hurdle_scene(
    layout: HurdleLayout, heading: tuple[float, float], start_fwd: float = 0.0
):
    """MjSpec edit adding the hurdles' top bars as static box colliders on the world body."""

    def _edit(spec: mujoco.MjSpec) -> mujoco.MjSpec:
        hx, hy = heading
        yaw = math.atan2(hy, hx)
        quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        for k, f in enumerate(layout.positions()):
            f = start_fwd + float(f)
            g = spec.worldbody.add_geom()
            g.name = f"event_hurdle_{k}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [
                layout.bar_depth_m / 2,
                layout.width_m / 2,
                layout.bar_height_m / 2,
            ]
            g.pos = [f * hx, f * hy, layout.height_m - layout.bar_height_m / 2]
            g.quat = quat
            g.rgba = [0.95, 0.95, 0.95, 1.0]
        return spec

    return _edit


# ------------------------------------------------------------------------------------------------------ env
class AthleticEventEnv(gym.Env):
    """One judged event on the CPU athletic env of its base task.

    Observation: base observation followed by :data:`EVENT_FEATURES` event features. Reward: the base task's
    lane reward plus the judge's event rewards. ``info["event"]`` holds the attempt record; when the event ends,
    ``info["event"]["performance"]`` (s for track, m for jumps) and ``info["event"]["points"]`` are set.
    """

    def __init__(
        self,
        event: EventSpec | str,
        model: str | None = None,
        tendons: TendonKind = DEFAULT_TENDONS,
        **kwargs: Any,
    ):
        from myosuite.envs.myo.athletic.catalog import TASKS, athletic_bundle
        from myosuite.envs.myo.athletic.factory import (
            AthleticModularTaskEnv,
            task_config_from_bundle,
        )

        event = EVENTS[event] if isinstance(event, str) else event
        if event.kind not in RUNNABLE_KINDS:
            raise NotImplementedError(
                f"{event.title}: {event.kind} events need {sorted(event.requires)} on the plant; not available yet"
            )
        bundle = athletic_bundle(event.base_task, model, tendons)
        ensure_tendons_applied(bundle)
        task = replace(TASKS[event.base_task], episode_length_s=event.time_limit_s)
        bundle = replace(bundle, task=task)
        cfg = task_config_from_bundle(bundle)
        if event.hurdles is not None:
            cfg.scene = hurdle_scene(event.hurdles, bundle.heading_dir)
        self.base = AthleticModularTaskEnv(cfg, **kwargs)
        self.metadata = self.base.metadata
        self.render_mode = self.base.render_mode
        self.action_space = self.base.action_space
        self.event = event
        self.bundle = bundle
        self.frame = _task_frame(bundle.heading_dir)
        m = self.base.model
        self._dt = float(self.base._ctrl_dt)
        self._root = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, bundle.plant.body_map.root
        )
        self._ids = body_colliders(m)
        self._radius, self._half, self._axis = collider_radius_axis(m, self._ids)
        self._judge: JumpJudge | None = None
        if event.jump is not None:
            # MuJoCo's soft contact rests a few mm inside the ground; 5 mm separates support from flight.
            jspec = replace(event.jump, contact_eps=max(event.jump.contact_eps, 0.005))
            names = [
                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g)
                for g in self._ids
            ]
            self._judge = JumpJudge(jspec, 1, foot_labels(m, self._ids), names)
        base = self.base.observation_space
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (base.shape[0] + EVENT_FEATURES,), np.float32
        )
        # task axes are measured from the world origin (the plant starts there; hurdles and lines are placed
        # from it), ground at z = 0
        self._origin = np.zeros(3)
        self._t = 0.0
        self._prev_com: np.ndarray | None = None
        self._record: dict[str, Any] = {}
        self._prev_fwd = 0.0

    # ----------------------------------------------------------------------------------------------- helpers
    def _pelvis_task(self) -> np.ndarray:
        d = self.base.data
        return self.frame @ (d.xpos[self._root] - self._origin)

    def _features(self) -> np.ndarray:
        fwd = self._pelvis_task()[0]
        e = self.event
        goal = (
            e.distance_m if e.kind == "track" else (e.jump.line_fwd if e.jump else 0.0)
        )
        nxt_d, nxt_h = 0.0, 0.0
        if e.hurdles is not None:
            ahead = e.hurdles.positions() - fwd
            ahead = ahead[ahead > -0.5]
            if ahead.size:
                nxt_d, nxt_h = float(ahead[0]), e.hurdles.height_m
        bar = (
            e.jump.bar_height
            if (e.jump is not None and e.jump.event == "high")
            else 0.0
        )
        return np.array([goal - fwd, nxt_d, nxt_h, bar, self._t], np.float32)

    def _finish(self, performance: float | None, outcome: str, **extra: Any) -> None:
        self._record.update(
            outcome=outcome,
            performance=performance,
            points=self.event.score(performance),
            **extra,
        )

    # -------------------------------------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs, info = self.base.reset(seed=seed, options=options)
        self._t = 0.0
        self._prev_com = None
        self._prev_fwd = float(self._pelvis_task()[0])
        self._record = {
            "event": self.event.key,
            "outcome": "running",
            "start_fwd": self._prev_fwd,
        }
        if self._judge is not None:
            self._judge.reset(0)
        info["event"] = dict(self._record)
        return np.concatenate([obs, self._features()]).astype(np.float32), info

    def render(self):
        return self.base.render()

    def close(self) -> None:
        self.base.close()

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.base.step(action)
        self._t += self._dt
        done_event = False
        e = self.event
        if e.kind == "track":
            fwd = float(self._pelvis_task()[0])
            if fwd >= e.distance_m > self._prev_fwd:
                frac = (e.distance_m - self._prev_fwd) / max(fwd - self._prev_fwd, 1e-9)
                t_fin = self._t - self._dt * (1.0 - frac)
                self._finish(t_fin, "finished")
                reward += e.lambda_finish * (e.distance_m / t_fin) / 10.0
                done_event = True
            self._prev_fwd = fwd
        else:
            d = self.base.data
            a, b = collider_segments(
                d, self._ids, self._half, self._axis, self.frame, self._origin
            )
            pelvis = self._pelvis_task()
            com = self.frame @ (d.subtree_com[self._root] - self._origin)
            if self._prev_com is None:
                self._prev_com = com
            com_vel = (com - self._prev_com) / self._dt
            self._prev_com = com
            ev = self._judge.update(
                self._dt,
                a[None],
                b[None],
                self._radius,
                com[None],
                com_vel[None],
                np.array([pelvis[2]]),
                self.bundle.plant.fall_height,
            )
            reward += float(ev["bonus"][0])
            if ev["suspend_lane_rules"][0]:
                # flight and landing: the judge ends the attempt (a fall in the pit is legal and keeps the mark);
                # the base env's lane, facing and fall endings do not apply
                terminated = False
            if ev["done"][0]:
                done_event = True
                r = self._judge.finalize(0)
                perf = (
                    r.get("mark_m")
                    if e.jump.event == "long"
                    else (e.jump.bar_height if r.get("cleared") else None)
                )
                self._finish(
                    perf if r.get("valid", r.get("cleared")) else None,
                    r["outcome"],
                    judge=r,
                )
        if not done_event and (
            terminated or truncated or self._t >= e.time_limit_s - 1e-9
        ):
            # the base env ended the attempt (fall, lane rules) or the event's time ran out: no mark
            cause = "fallen_or_out" if terminated else "time_limit"
            if self._judge is not None:
                r = self._judge.finalize(0, cause)
                self._finish(None, r["outcome"], judge=r)
            else:
                self._finish(None, cause)
            done_event = True
            truncated = not terminated
        elif done_event:
            terminated = True  # finished, or the judge gave its verdict
        info["event"] = dict(self._record)
        obs = np.concatenate([obs, self._features()]).astype(np.float32)
        return obs, float(reward), bool(terminated), bool(truncated), info


def event_env_id(
    plant: PlantSpec, key: str, tendons: TendonKind = DEFAULT_TENDONS
) -> str:
    """Gymnasium id of an event or combined event on a plant, same scheme as the athletic tasks:
    ``myo<Plant><Event>[Elastic]-v0`` (``myoFullbody100m-v0``, ``myoFullbodySimpleTFPentathlonElastic-v0``)."""
    title = {"tf_pentathlon": "TFPentathlon"}.get(
        key, "".join(p[:1].upper() + p[1:] for p in key.split("_"))
    )
    suffix = "Elastic" if tendons == "elastic" else ""
    return f"myo{plant.env_id_prefix}{title}{suffix}-v0"


def make_event(
    event: str,
    model: str | None = None,
    tendons: TendonKind = DEFAULT_TENDONS,
    **kwargs: Any,
) -> AthleticEventEnv:
    """A single judged event, e.g. ``make_event("110m_hurdles")`` on ``myofullbody`` with rigid tendons."""
    return AthleticEventEnv(event, model, tendons, **kwargs)
