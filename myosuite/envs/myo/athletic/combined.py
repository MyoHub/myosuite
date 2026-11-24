# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Combined events: one athlete (one plant, one policy) performs a list of judged events.

A :class:`CombinedEventSpec` is an ordered list of :data:`~myosuite.envs.myo.athletic.events.EVENTS` keys,
scored with the World Athletics combined-events points tables. :class:`CombinedEventEnv` runs the events on the
same plant with one observation / action interface: the base observation, the event features and a one-hot of
the current event.

* ``sequence`` (competition): one episode runs every event in order; the total points are in
  ``info["combined"]`` at the end.
* ``random`` (training): each reset draws one event, so a single policy learns all of them.

``TF_PENTATHLON`` (100 m, long jump, high jump, 110 m hurdles, 1500 m) runs on the running plants. ``DECATHLON``
is declared with its throws and pole vault; they need hands and implements (``requires`` on the events) and
build once a plant provides those capabilities (e.g. ``myofullbody``, which has hands, plus an implement scene).

    env = make_combined("tf_pentathlon", model="myofullbody")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
import numpy as np

from myosuite.envs.myo.athletic.events import (
    EVENTS,
    RUNNABLE_KINDS,
    AthleticEventEnv,
    EventSpec,
    event_env_id,
)
from myosuite.envs.myo.athletic.specs import (
    DEFAULT_TENDONS,
    AthleticBundle,
    TendonKind,
    ensure_tendons_applied,
)


@dataclass(frozen=True)
class CombinedEventSpec:
    """An ordered list of events performed by one athlete."""

    key: str
    title: str
    events: tuple[str, ...]

    def specs(self) -> tuple[EventSpec, ...]:
        return tuple(EVENTS[k] for k in self.events)

    def missing(self, capabilities: frozenset[str]) -> list[str]:
        """Events this plant cannot perform yet (not runnable, or capabilities missing)."""
        return [
            e.key
            for e in self.specs()
            if e.kind not in RUNNABLE_KINDS or not e.requires.issubset(capabilities)
        ]

    def score(self, performances: dict[str, float | None]) -> dict[str, Any]:
        """Points per event and the total for a dict of performances (s or m)."""
        pts = {e.key: e.score(performances.get(e.key)) for e in self.specs()}
        return {"points": pts, "total": int(sum(pts.values()))}


COMBINED_EVENTS: dict[str, CombinedEventSpec] = {
    "tf_pentathlon": CombinedEventSpec(
        key="tf_pentathlon",
        title="TF Pentathlon",
        events=("100m", "long_jump", "high_jump", "110m_hurdles", "1500m"),
    ),
    # World Athletics men's decathlon, in competition order (day 1, day 2)
    "decathlon": CombinedEventSpec(
        key="decathlon",
        title="Decathlon",
        events=(
            "100m",
            "long_jump",
            "shot_put",
            "high_jump",
            "400m",
            "110m_hurdles",
            "discus",
            "pole_vault",
            "javelin",
            "1500m",
        ),
    ),
}


class CombinedEventEnv(gym.Env):
    """All events of a :class:`CombinedEventSpec` on one plant, behind one observation / action interface."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        combined: CombinedEventSpec | str,
        model: str | None = None,
        mode: Literal["sequence", "random"] = "sequence",
        tendons: TendonKind = DEFAULT_TENDONS,
        **kwargs: Any,
    ):
        from myosuite.envs.myo.athletic.catalog import DEFAULT_MODEL, PLANTS, TASKS

        combined = COMBINED_EVENTS[combined] if isinstance(combined, str) else combined
        plant = PLANTS[model or DEFAULT_MODEL]
        missing = combined.missing(plant.capabilities)
        if missing:
            raise NotImplementedError(
                f"{combined.title} on {plant.key}: events {missing} need capabilities the plant does not "
                "provide yet (hands, implements, pole)"
            )
        ensure_tendons_applied(
            AthleticBundle(task=TASKS["sprint"], plant=plant, tendons=tendons)
        )
        if mode not in ("sequence", "random"):
            raise ValueError(f"mode must be 'sequence' or 'random', got {mode!r}")
        self.combined = combined
        self.mode = mode
        self.events = [
            AthleticEventEnv(e, plant.key, tendons, **kwargs) for e in combined.specs()
        ]
        n = len(self.events)
        base = self.events[0].observation_space.shape[0]
        if any(e.observation_space.shape[0] != base for e in self.events):
            raise ValueError(
                "events of one combined event must share the observation layout"
            )
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (base + n,), np.float32
        )
        self.action_space = self.events[0].action_space
        self._k = 0
        self._results: dict[str, dict[str, Any]] = {}

    @property
    def current(self) -> AthleticEventEnv:
        return self.events[self._k]

    def _obs(self, obs: np.ndarray) -> np.ndarray:
        onehot = np.zeros(len(self.events), np.float32)
        onehot[self._k] = 1.0
        return np.concatenate([obs, onehot]).astype(np.float32)

    def _summary(self) -> dict[str, Any]:
        perf = {k: r.get("performance") for k, r in self._results.items()}
        out = self.combined.score(perf)
        out.update(performances=perf, results=dict(self._results))
        return out

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options and "event" in options:
            self._k = self.combined.events.index(options["event"])
        elif self.mode == "random":
            self._k = int(self.np_random.integers(len(self.events)))
        else:
            self._k = 0
        self._results = {}
        obs, info = self.current.reset(seed=seed)
        info["event_index"] = self._k
        return self._obs(obs), info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.current.step(action)
        info["event_index"] = self._k
        if not (terminated or truncated):
            return self._obs(obs), reward, False, False, info
        self._results[self.current.event.key] = info["event"]
        if self.mode == "sequence" and self._k + 1 < len(self.events):
            # next event: a fresh start on its own runway; the episode goes on
            self._k += 1
            obs, _ = self.current.reset()
            info["event_index"] = self._k
            return self._obs(obs), reward, False, False, info
        info["combined"] = self._summary()
        return self._obs(obs), reward, terminated, truncated, info


def make_combined(
    combined: str = "tf_pentathlon",
    model: str | None = None,
    tendons: TendonKind = DEFAULT_TENDONS,
    **kwargs: Any,
) -> CombinedEventEnv:
    """A combined event on one plant, e.g. ``make_combined("tf_pentathlon", model="myofullbody")``."""
    return CombinedEventEnv(combined, model, tendons=tendons, **kwargs)


def register_events_and_combined(
    plants: tuple[str, ...] = ("myofullbody",),
    tendons: tuple[TendonKind, ...] = ("rigid", "elastic"),
) -> list[str]:
    """Register the runnable single events and combined events as CPU Gymnasium ids.

    Ids follow the athletic scheme (:func:`~myosuite.envs.myo.athletic.events.event_env_id`):
    ``myoFullbody100m-v0``, ``myoFullbodyLongJump-v0``, ``myoFullbodyTFPentathlon-v0``; elastic tendons add
    ``Elastic``. A tendon variant is registered only once its series element is applied, so no id builds a
    different physics than it names. Episode limits come from the events (``time_limit_s``).
    """
    from myosuite.core.registry import register_env
    from myosuite.envs.myo.athletic.catalog import PLANTS, TASKS

    ids: list[str] = []
    ctrl_dt = 1.0 / 30.0
    for pk in plants:
        plant = PLANTS[pk]
        for kind in tendons:
            try:
                ensure_tendons_applied(
                    AthleticBundle(task=TASKS["sprint"], plant=plant, tendons=kind)
                )
            except NotImplementedError:
                continue
            for ev in EVENTS.values():
                if ev.kind not in RUNNABLE_KINDS or not ev.requires.issubset(
                    plant.capabilities
                ):
                    continue
                env_id = event_env_id(plant, ev.key, kind)
                register_env(
                    env_id=env_id,
                    entry_point="myosuite.envs.myo.athletic.events:AthleticEventEnv",
                    max_episode_steps=int(ev.time_limit_s / ctrl_dt) + 10,
                    kwargs={"event": ev.key, "model": pk, "tendons": kind},
                )
                ids.append(env_id)
            for ce in COMBINED_EVENTS.values():
                if ce.missing(plant.capabilities):
                    continue
                env_id = event_env_id(plant, ce.key, kind)
                total = sum(e.time_limit_s for e in ce.specs())
                register_env(
                    env_id=env_id,
                    entry_point="myosuite.envs.myo.athletic.combined:CombinedEventEnv",
                    max_episode_steps=int(total / ctrl_dt) + 10 * len(ce.events),
                    kwargs={"combined": ce.key, "model": pk, "tendons": kind},
                )
                ids.append(env_id)
    return ids
