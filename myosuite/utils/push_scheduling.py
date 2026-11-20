# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Baseline-measured contact phase and explicit initial velocity experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np


@dataclass(frozen=True)
class ContactPhase:
    """Phase between two measured baseline contacts of a declared heel group.

    Zero is heel strike; one is the next same-heel strike. The baseline determines
    the scheduled control step before replaying either arm; the pushed arm never
    gets a different event-triggered start. Initial standing contact is excluded.
    Minimum contact/swing durations suppress single-sample contact chatter.
    """

    heel_bodies: tuple[str, ...]
    phase: float
    stride_index: int = 2
    min_contact_s: float = 0.01
    min_swing_s: float = 0.03

    def __post_init__(self) -> None:
        if not self.heel_bodies or not 0 <= self.phase < 1:
            raise ValueError("Declare heel bodies and phase in [0, 1)")
        if not isinstance(self.stride_index, int) or self.stride_index < 0:
            raise ValueError("stride_index must be a nonnegative integer")
        if (
            not np.isfinite([self.min_contact_s, self.min_swing_s]).all()
            or min(self.min_contact_s, self.min_swing_s) <= 0
        ):
            raise ValueError("Contact and swing dwell times must be finite positive")

    def schedule(self, baseline_states: dict, resolved: dict) -> dict[str, Any]:
        """Resolve actual baseline gait phase, or explicitly record unavailable."""
        names = resolved["requested"]["foot_bodies"]
        if not set(self.heel_bodies) <= set(names):
            raise ValueError("Heel bodies must be included in protocol foot_bodies")
        dt = resolved["ctrl_dt_s"]
        mask = np.any(
            baseline_states["foot_contact_mask"][
                :, [names.index(n) for n in self.heel_bodies]
            ],
            axis=1,
        )
        edges = np.diff(np.r_[False, mask, False].astype(int))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        contact = max(1, int(np.ceil(self.min_contact_s / dt)))
        swing = max(1, int(np.ceil(self.min_swing_s / dt)))
        strikes = [
            int(start)
            for i, (start, end) in enumerate(zip(starts, ends))
            if end - start >= contact
            and start > 0
            and start - (ends[i - 1] if i else 0) >= swing
        ]
        result = dict(resolved)
        result.update(
            contact_phase=asdict(self),
            heel_strike_steps=strikes,
            phase_origin="baseline_heel_strike",
            phase_available=False,
        )
        if len(strikes) > self.stride_index + 1:
            first, second = strikes[self.stride_index : self.stride_index + 2]
            start = int(np.rint(first + self.phase * (second - first)))
            duration = resolved["end_step"] - resolved["start_step"]
            if start + duration < resolved["requested"]["steps"]:
                result.update(
                    start_step=start,
                    end_step=start + duration,
                    actual_start_s=start * dt,
                    measured_stride_s=(second - first) * dt,
                    bracket_heel_strike_steps=[first, second],
                    phase_available=True,
                )
                return result
        result.update(
            start_step=resolved["requested"]["steps"],
            end_step=resolved["requested"]["steps"],
            actual_start_s=None,
            measured_stride_s=None,
            phase_unavailable_reason="insufficient_baseline_heel_strikes_or_horizon",
        )
        return result


def jitter_free_root(
    env: gym.Env, obs: np.ndarray, resolved: dict, *, seed: int, std_natural: float
) -> tuple[np.ndarray, dict]:
    """Apply declared seeded world-XY velocity jitter before capturing either arm.

    Standard deviation is in sqrt(g*leg_length). Requires a free root and the CPU
    env's non-stepping forward() refresh. Observation wrappers need their own
    explicit preparation adapter and are rejected here.
    """
    if not np.isfinite(std_natural) or std_natural < 0:
        raise ValueError("Velocity jitter must be finite nonnegative")
    result = dict(resolved)
    delta = np.zeros(2)
    if std_natural:
        layer = env
        while isinstance(layer, gym.Wrapper):
            if isinstance(layer, gym.ObservationWrapper):
                raise ValueError(
                    "Velocity jitter does not support observation wrappers"
                )
            layer = layer.env
        raw = env.unwrapped
        joint = np.flatnonzero(
            (raw.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
            & (raw.model.jnt_bodyid == resolved["root_id"])
        )
        if len(joint) != 1 or not callable(getattr(raw, "forward", None)):
            raise ValueError(
                "Velocity jitter requires a free root and forward() refresh"
            )
        delta = np.random.default_rng(seed).normal(
            0,
            std_natural * np.sqrt(resolved["gravity_mps2"] * resolved["leg_length_m"]),
            size=2,
        )
        adr = int(raw.model.jnt_dofadr[joint[0]])
        raw.data.qvel[adr : adr + 2] += delta
        clock = raw.data.time
        obs = np.asarray(raw.forward()[0], dtype=np.float32)
        if raw.data.time != clock:
            raise ValueError("forward() must not step the physics clock")
    result["initial_velocity_jitter"] = {
        "std_natural": std_natural,
        "world_xy_delta_mps": delta.tolist(),
        "seed": seed,
    }
    return obs, result
