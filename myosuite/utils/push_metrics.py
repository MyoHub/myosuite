# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Declared recovery criteria and physical diagnostics of paired push traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from myosuite.biomechanics.effort import squared_activation
from myosuite.utils.push_recovery import PairedPushTrial, PushTrace


@dataclass(frozen=True)
class RecoveryCriteria:
    """Thresholds in leg lengths, natural speed, radians and seconds.

    Recovery compares pushed speed/heading with its paired baseline throughout a
    continuous hold interval after the push. Falls use COM height above the Z=0
    floor and root tilt relative to reset. Termination is recorded separately.
    """

    min_height_L: float = 0.5
    max_tilt_rad: float = 1.0
    speed_error_natural: float = 0.15
    heading_error_rad: float = 0.2
    hold_s: float = 0.25
    lateral_error_L: float | None = None
    target_speed_mps: float | None = None
    target_speed_tolerance_mps: float = 0.35
    require_baseline_tracking: bool = False

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            if (
                key != "target_speed_mps"
                and value is not None
                and (not np.isfinite(value) or value < 0)
            ):
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.target_speed_mps is not None and not np.isfinite(self.target_speed_mps):
            raise ValueError("target_speed_mps must be finite")
        if self.require_baseline_tracking and self.target_speed_mps is None:
            raise ValueError("Baseline tracking eligibility requires a target speed")
        if self.hold_s <= 0 or not 0 < self.max_tilt_rad <= np.pi:
            raise ValueError("hold_s must be positive and tilt in (0, pi]")


def _signals(trace: PushTrace, cfg: dict, criteria: RecoveryCriteria) -> dict:
    states, dt = trace.states, cfg["ctrl_dt_s"]
    length, g = cfg["leg_length_m"], cfg["gravity_mps2"]
    forward = np.asarray(cfg["forward_world"])
    side = np.cross([0, 0, 1], forward)
    relative = states["root_rotation"] @ states["root_rotation"][0].T
    orientation = relative @ forward
    speed = np.gradient(states["com_m"], dt, axis=0) @ forward
    heading = np.arctan2(orientation @ side, orientation @ forward)
    tilt = np.arccos(np.clip(relative[:, 2, 2], -1, 1))
    height = states["com_m"][:, 2] / length
    return {
        "speed_mps": speed,
        "speed_natural": speed / np.sqrt(g * length),
        "heading_rad": heading,
        "lateral_L": (states["com_m"] - states["com_m"][0]) @ side / length,
        "height_L": height,
        "tilt_rad": tilt,
        "fall": (height < criteria.min_height_L) | (tilt > criteria.max_tilt_rad),
    }


def _diagnostics(trace: PushTrace, signal: dict, cfg: dict) -> dict[str, Any]:
    muscle = cfg["muscle_state_indices"]
    result = {
        "transitions": len(trace.actions),
        "terminated": trace.terminated,
        "truncated": trace.truncated,
        "reached_horizon": len(trace.actions) == cfg["requested"]["steps"],
        "fall": bool(np.any(signal["fall"])),
        "mean_forward_speed_mps": float(np.mean(signal["speed_mps"])),
        "final_lateral_drift_L": float(signal["lateral_L"][-1]),
        "max_abs_heading_rad": float(np.max(np.abs(signal["heading_rad"]))),
        "min_com_height_L": float(np.min(signal["height_L"])),
        "mean_squared_activation": float(
            squared_activation(trace.states["act"][:, muscle])
        )
        if muscle
        else None,
        "foot_contact_fraction": None,
        "mean_vertical_grf_bw": None,
    }
    if "foot_contact_count" in trace.states:
        result.update(
            foot_contact_fraction=float(
                np.mean(trace.states["foot_contact_count"] > 0)
            ),
            mean_vertical_grf_bw=float(np.mean(trace.states["vertical_grf_n"]))
            / (cfg["mass_kg"] * cfg["gravity_mps2"]),
        )
    return result


def summarize_trial(
    trial: PairedPushTrial, criteria: RecoveryCriteria = RecoveryCriteria()
) -> dict[str, Any]:
    """Report actual dose, eligibility, censoring and paired physical outcomes.

    Scheduled impulse is a protocol target; delivered impulse integrates recorded
    world force over executed control transitions. An incomplete baseline is never
    a recovery success. Missing muscle/contact measurements remain unavailable.
    """
    cfg = trial.protocol
    dt, start, end = cfg["ctrl_dt_s"], cfg["start_step"], cfg["end_step"]
    base = _signals(trial.baseline, cfg, criteria)
    push = _signals(trial.pushed, cfg, criteria)
    n = min(len(base["height_L"]), len(push["height_L"]))
    speed_error = np.abs(push["speed_natural"][:n] - base["speed_natural"][:n])
    angle = push["heading_rad"][:n] - base["heading_rad"][:n]
    heading_error = np.abs(np.arctan2(np.sin(angle), np.cos(angle)))
    drift_error = np.abs(push["lateral_L"][:n] - base["lateral_L"][:n])
    baseline = _diagnostics(trial.baseline, base, cfg)
    pushed = _diagnostics(trial.pushed, push, cfg)
    for signals, diagnostics in ((base, baseline), (push, pushed)):
        error = (
            None
            if criteria.target_speed_mps is None
            else float(
                np.mean(
                    np.abs(
                        signals["speed_mps"][min(end, len(signals["speed_mps"]) - 1) :]
                        - criteria.target_speed_mps
                    )
                )
            )
        )
        diagnostics["post_push_mean_target_speed_error_mps"] = error
        diagnostics["tracks_target_speed"] = (
            None if error is None else error <= criteria.target_speed_tolerance_mps
        )
    eligible = (
        baseline["reached_horizon"]
        and not baseline["fall"]
        and not trial.baseline.terminated
        and cfg.get("phase_available", True)
        and (not criteria.require_baseline_tracking or baseline["tracks_target_speed"])
    )
    reason = None if eligible else "baseline_fall_or_incomplete"
    if not cfg.get("phase_available", True):
        reason = "contact_phase_unavailable"
    elif criteria.require_baseline_tracking and not baseline["tracks_target_speed"]:
        reason = "baseline_does_not_track_command"
    good = (
        (speed_error <= criteria.speed_error_natural)
        & (heading_error <= criteria.heading_error_rad)
        & ~push["fall"][:n]
    )
    if criteria.lateral_error_L is not None:
        good &= drift_error <= criteria.lateral_error_L
    hold = max(1, int(np.ceil(criteria.hold_s / dt))) + 1
    recovery = None
    if eligible and not pushed["fall"] and not trial.pushed.terminated:
        for i in range(end, n - hold + 1):
            if np.all(good[i : i + hold]):
                recovery = (i - end) * dt
                break
    wrench = trial.pushed.applied_wrench
    vector_impulse = np.sum(wrench[:, :3], axis=0) * dt
    scalar_impulse = float(np.sum(np.linalg.norm(wrench[:, :3], axis=-1)) * dt)
    scheduled = cfg["impulse_ns"] if cfg.get("phase_available", True) else 0.0
    exposure = max(0, min(len(wrench), end) - start)
    norm = cfg["mass_kg"] * np.sqrt(cfg["gravity_mps2"] * cfg["leg_length_m"])
    return {
        "seed": trial.seed,
        "criteria": asdict(criteria),
        "baseline": baseline,
        "pushed": pushed,
        "eligible": bool(eligible),
        "ineligible_reason": reason,
        "exposed_transitions": exposure,
        "complete_dose": exposure > 0 and exposure == end - start,
        "scheduled_impulse_ns": scheduled,
        "delivered_scalar_impulse_ns": scalar_impulse,
        "delivered_world_impulse_ns": vector_impulse.tolist(),
        "delivered_angular_impulse_nms": (np.sum(wrench[:, 3:], axis=0) * dt).tolist(),
        "delivered_impulse_normalized": scalar_impulse / norm,
        "delivered_dose_fraction": scalar_impulse / scheduled
        if scheduled > 0
        else None,
        "recovered": bool(eligible and recovery is not None and exposure > 0),
        "recovery_time_s": recovery if exposure > 0 else None,
        "recovery_censored": bool(eligible and exposure > 0 and recovery is None),
        "max_speed_error_natural": float(np.max(speed_error)),
        "max_heading_error_rad": float(np.max(heading_error)),
        "final_lateral_error_L": float(drift_error[-1]),
        "paired_mean_speed_delta_mps": pushed["mean_forward_speed_mps"]
        - baseline["mean_forward_speed_mps"],
        "paired_effort_delta": None
        if baseline["mean_squared_activation"] is None
        else pushed["mean_squared_activation"] - baseline["mean_squared_activation"],
        "recovery_definition": "return_to_paired_baseline",
        "post_push_foot_placement": _placements(trial, cfg),
    }


def _placements(trial: PairedPushTrial, cfg: dict) -> list[dict] | None:
    if "foot_positions_m" not in trial.baseline.states:
        return None
    result = []
    for i, name in enumerate(cfg["requested"]["foot_bodies"]):
        events = []
        for trace in (trial.baseline, trial.pushed):
            rises = (
                np.flatnonzero(
                    np.diff(trace.states["foot_contact_mask"][:, i].astype(int)) == 1
                )
                + 1
            )
            rises = rises[rises >= cfg["end_step"]]
            events.append(
                None
                if not len(rises)
                else {
                    "time_s": float(trace.states["time_s"][rises[0]]),
                    "position_m": trace.states["foot_positions_m"][
                        rises[0], i
                    ].tolist(),
                }
            )
        delta = (
            None
            if any(e is None for e in events)
            else (
                np.asarray(events[1]["position_m"]) - events[0]["position_m"]
            ).tolist()
        )
        result.append(
            {
                "body": name,
                "baseline": events[0],
                "pushed": events[1],
                "world_position_delta_m": delta,
                "matching": "first_threshold_contact_entry_after_push_each_arm",
            }
        )
    return result
