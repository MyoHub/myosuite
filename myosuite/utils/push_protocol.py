# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Typed physical conditions for paired CPU push experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import gymnasium as gym
import mujoco
import numpy as np


@dataclass(frozen=True)
class PushProtocol:
    """Declared push conditions for a Z-up, flat-observation CPU env.

    Supply SI force/duration or body-weight/natural-time alternatives, never both.
    Natural time is sqrt(leg_length/gravity). Phase is a nominal stride-clock
    fraction from reset, not inferred foot-strike phase. Starts use nearest
    control-step rounding; durations round up. Requested and actual values are
    recorded. Frame/point transforms use control-step zero-order hold.
    """

    body: str
    leg_length_m: float
    steps: int = 400
    force_n: float | None = 40.0
    force_bw: float | None = None
    duration_s: float | None = 0.16
    duration_tau: float | None = None
    start_s: float | None = 1.0
    phase: float | None = None
    stride_period_s: float | None = None
    cycles_before_push: int = 1
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    frame: Literal["world", "body"] = "world"
    point_body_m: tuple[float, float, float] | None = None
    forward_world: tuple[float, float, float] = (0.0, 1.0, 0.0)
    foot_bodies: tuple[str, ...] = ()
    ground_bodies: tuple[str, ...] = ("world",)
    contact_threshold_bw: float = 0.01

    def resolve(self, env: gym.Env) -> dict[str, Any]:
        """Resolve normalizations and timing against the actual fixed model."""
        u, cfg = env.unwrapped, asdict(self)
        model = u.model
        if self.steps < 1 or not isinstance(self.steps, int):
            raise ValueError("steps must be a positive integer")
        if not np.isfinite(self.leg_length_m) or self.leg_length_m <= 0:
            raise ValueError("leg_length_m must be finite and positive")
        g = float(np.linalg.norm(model.opt.gravity))
        if g <= 0 or not np.allclose(model.opt.gravity, [0, 0, -g]):
            raise ValueError("The protocol requires finite, nonzero Z-down gravity")
        body = int(model.body(self.body).id)
        root = int(model.body_rootid[body])
        mass = float(np.sum(model.body_mass[model.body_rootid == root]))
        dt = float(
            getattr(u, "_ctrl_dt", model.opt.timestep * getattr(u, "frame_skip", 1))
        )
        if (
            body == 0
            or not np.any(model.body_rootid[model.jnt_bodyid] == root)
            or mass <= 0
            or not np.isfinite([mass, dt]).all()
            or dt <= 0
        ):
            raise ValueError(
                "A dynamic body, finite mass and positive control dt are required"
            )
        force = self.force_n if self.force_bw is None else self.force_bw * mass * g
        duration = (
            self.duration_s
            if self.duration_tau is None
            else self.duration_tau * np.sqrt(self.leg_length_m / g)
        )
        if (self.force_n is None) == (self.force_bw is None) or (
            self.duration_s is None
        ) == (self.duration_tau is None):
            raise ValueError("Specify exactly one SI or normalized force and duration")
        start = self.start_s
        if self.phase is not None:
            if (
                start is not None
                or not 0 <= self.phase < 1
                or self.stride_period_s is None
                or self.stride_period_s <= 0
                or not isinstance(self.cycles_before_push, int)
                or self.cycles_before_push < 0
            ):
                raise ValueError(
                    "Phase needs start_s=None, a positive stride period and nonnegative cycles"
                )
            start = (self.cycles_before_push + self.phase) * self.stride_period_s
        if (
            start is None
            or not np.isfinite([force, duration, start]).all()
            or force < 0
            or duration <= 0
            or start < 0
        ):
            raise ValueError(
                "Force/start must be finite nonnegative; duration finite positive"
            )
        direction, forward = np.asarray(self.direction), np.asarray(self.forward_world)
        if (
            direction.shape != (3,)
            or forward.shape != (3,)
            or not np.isfinite([direction, forward]).all()
            or np.linalg.norm(direction) == 0
            or np.linalg.norm(forward[:2]) == 0
            or forward[2] != 0
        ):
            raise ValueError(
                "Finite nonzero direction and horizontal forward 3-vectors required"
            )
        start_step, duration_steps = (
            int(np.rint(start / dt)),
            max(1, int(np.ceil(duration / dt - 1e-12))),
        )
        if start_step + duration_steps >= self.steps:
            raise ValueError("The horizon must extend beyond the push")
        if self.frame not in ("world", "body") or (
            self.point_body_m is not None
            and (
                np.asarray(self.point_body_m).shape != (3,)
                or not np.isfinite(self.point_body_m).all()
            )
        ):
            raise ValueError("Declare a world/body frame and finite body-local point")
        feet = [int(model.body(name).id) for name in self.foot_bodies]
        ground = [int(model.body(name).id) for name in self.ground_bodies]
        if (
            set(feet) & set(ground)
            or not np.isfinite(self.contact_threshold_bw)
            or self.contact_threshold_bw < 0
        ):
            raise ValueError(
                "Foot/ground body sets must differ; contact threshold must be nonnegative"
            )
        muscle_states = [
            int(model.actuator_actadr[i])
            for i in range(model.nu)
            if model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE
        ]
        return {
            "requested": cfg,
            "body_id": body,
            "root_id": root,
            "mass_kg": mass,
            "gravity_mps2": g,
            "leg_length_m": self.leg_length_m,
            "ctrl_dt_s": dt,
            "force_world_or_body_n": (
                direction / np.linalg.norm(direction) * force
            ).tolist(),
            "force_bw": float(force / (mass * g)),
            "requested_start_s": float(start),
            "requested_duration_s": float(duration),
            "start_step": start_step,
            "end_step": start_step + duration_steps,
            "actual_start_s": start_step * dt,
            "actual_duration_s": duration_steps * dt,
            "impulse_ns": force * duration_steps * dt,
            "impulse_normalized": force
            * duration_steps
            * dt
            / (mass * np.sqrt(g * self.leg_length_m)),
            "forward_world": (forward / np.linalg.norm(forward)).tolist(),
            "foot_body_ids": feet,
            "ground_body_ids": ground,
            "contact_threshold_n": self.contact_threshold_bw * mass * g,
            "muscle_state_indices": muscle_states,
            "wrench_sampling": "control_step_zero_order_hold",
            "phase_origin": "reset_clock_nominal_stride",
        }
