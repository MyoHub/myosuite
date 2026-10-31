"""Physical showcase scheduling and scoring, independent of the muscle policy."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import mujoco
import numpy as np

from myosuite.envs.myo.tasks.halloween.config import AvoidanceConfig
from myosuite.envs.myo.tasks.halloween.halloween_model import (
    PUMPKIN_BODY,
    PUMPKIN_COLLISION_GEOM,
    pumpkin_dofadr,
    pumpkin_qposadr,
)


class AvoidanceTask:
    """Own a seeded sequence of frontal throws and per-throw physical metrics.

    Args:
        model: Calibrated TERRA scene model.
        data: Live scene state owned by the environment.
        config: Throw curriculum and recovery intervals, in control steps.
    """

    def __init__(
        self, model: mujoco.MjModel, data: mujoco.MjData, config: AvoidanceConfig
    ) -> None:
        if (
            not config.throws
            or config.recovery_steps < 1
            or config.between_wait_steps < 1
        ):
            raise ValueError(
                "Avoidance needs throws and positive recovery/wait intervals"
            )
        low, high = config.first_wait_steps
        if low < 1 or high <= low:
            raise ValueError("first_wait_steps must be a positive [low, high) interval")
        for threat in config.throws:
            if (
                not np.isfinite(list(asdict(threat).values())).all()
                or min(threat.speed, threat.flight_s) <= 0
            ):
                raise ValueError(
                    "Throw values must be finite, with positive flight_s and speed"
                )
        self.model, self.data, self.config = model, data, config
        self.body = model.body(PUMPKIN_BODY).id
        self.geom = model.geom(PUMPKIN_COLLISION_GEOM).id
        self.qa, self.va = pumpkin_qposadr(model), pumpkin_dofadr(model)
        paired = set(map(int, model.pair_geom1)) | set(map(int, model.pair_geom2))
        self.body_geoms = {
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] not in (0, self.body)
            and (i in paired or model.geom_contype[i] or model.geom_conaffinity[i])
        }

    def reset(self, rng: np.random.Generator) -> None:
        """Initialize all episode state from the environment's seeded generator."""
        self.policy_seed = int(rng.integers(2**31))
        self.steps = self.round_start = self.throw_index = self.launch_id = 0
        self.wait = int(rng.integers(*self.config.first_wait_steps))
        self.home = self.data.qpos[:2].copy()
        self.launched_at: int | None = None
        self.events: list[dict[str, Any]] = []
        self.complete = False
        self.model.body_gravcomp[self.body] = 1
        self._clear_metrics()

    def _clear_metrics(self) -> None:
        self.contacts, self.clearance, self.max_offset = 0, 10.0, 0.0
        self.min_height = float(self.data.qpos[2])

    def count_contacts(self) -> int:
        """Retain physical contact rows at every 500 Hz physics substep."""
        hits = 0
        for contact in self.data.contact:
            if (
                contact.geom1 == self.geom and int(contact.geom2) in self.body_geoms
            ) or (contact.geom2 == self.geom and int(contact.geom1) in self.body_geoms):
                hits += 1
                self.clearance = min(self.clearance, float(contact.dist))
        self.contacts += hits
        return hits

    def _launch(self) -> None:
        threat = self.config.throws[self.throw_index]
        self.model.body_gravcomp[self.body] = 0
        self.data.qpos[self.qa : self.qa + 7] = [
            self.home[0] + threat.target_x,
            self.home[1] + 0.14 - threat.speed * threat.flight_s,
            threat.target_z,
            1,
            0,
            0,
            0,
        ]
        self.data.qvel[self.va : self.va + 6] = [
            0,
            threat.speed,
            0.5 * 9.81 * threat.flight_s,
            0,
            2,
            0,
        ]
        mujoco.mj_forward(self.model, self.data)
        self.launch_id += 1
        self.launched_at = self.steps

    def _result(self) -> dict[str, Any]:
        head = self.data.body("head").xpos
        direction = head - self.data.qpos[:3]
        posture = float(direction[2] / np.linalg.norm(direction))
        offset = float(np.linalg.norm(self.data.qpos[:2] - self.home))
        return dict(
            throw=self.throw_index + 1,
            body_contacts=self.contacts,
            min_surface_distance_m=self.clearance,
            min_pelvis_height_m=self.min_height,
            max_home_distance_m=self.max_offset,
            final_home_distance_m=offset,
            recovered=bool(
                offset < 0.10
                and self.data.qpos[2] > 0.85
                and head[2] > 1.45
                and posture > 0.9
            ),
        )

    def after_step(self) -> None:
        """Update metrics, then publish any launch in the next observation.

        Launching at the observation boundary lets an external controller react
        on the same physical tick as the original explicit tutorial.
        """
        self.steps += 1
        self.max_offset = max(
            self.max_offset, float(np.linalg.norm(self.data.qpos[:2] - self.home))
        )
        self.min_height = min(self.min_height, float(self.data.qpos[2]))
        if (
            self.launched_at is not None
            and np.linalg.norm(
                self.data.qpos[self.qa : self.qa + 3] - self.data.qpos[:3]
            )
            < 1
        ):
            for geom in self.body_geoms:
                distance = mujoco.mj_geomDistance(
                    self.model, self.data, self.geom, geom, 10.0, np.zeros(6)
                )
                self.clearance = min(self.clearance, float(distance))
        if self.steps - self.round_start == self.wait + self.config.recovery_steps:
            self.events.append(self._result())
            self.throw_index += 1
            self.complete = self.throw_index == len(self.config.throws)
            if self.complete:
                return
            self.round_start, self.wait = self.steps, self.config.between_wait_steps
            self.launched_at = None
            self._clear_metrics()
        if self.steps - self.round_start == self.wait:
            self._launch()

    def info(self) -> dict[str, Any]:
        """Return a detached snapshot for a controller or student logger."""
        return dict(
            profile="terra",
            policy_seed=self.policy_seed,
            home=self.home.copy(),
            launch_id=self.launch_id,
            steps_since_launch=None
            if self.launched_at is None
            else self.steps - self.launched_at,
            events=[event.copy() for event in self.events],
            current_event=None if self.complete else self._result(),
            complete=self.complete,
        )
