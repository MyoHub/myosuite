# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Long jump and high jump: event geometry, an online judge, and the benchmark metrics.

Both events start like the sprint (run pose at the origin, lane rules on the runway) and are written in task axes
(forward, side, up), so any plant adapter can feed them. Rules follow World Athletics Competition Rules (C2.1,
TR26-TR30, 2026 edition) where a simulation can apply them:

* **Long jump.** A take-off line ``line_fwd`` m down the runway. A supporting (take-off) foot breaking the
  vertical plane of the line, at any time before it leaves the ground, is a foul (TR30.1.1; the foot's forward
  surface, toe included, even off the ground; hands or the swing foot breaking the plane do not count). Another
  body part touching the ground beyond the line before the jump ends the attempt as ``fallen``. The jump is the flight that leaves the ground
  behind the line and lands beyond it; the mark is measured from the line to the nearest point of the ground
  footprint of any body part within ``mark_window_s`` of landing (a timed, flat-ground surrogate of the
  "nearest break", measured perpendicular to the line, TR29 Distance Measurement; ``mark_official_m`` rounds it
  down to the centimetre). Not applied: somersaults (TR30.1.3), landing-area side borders (TR30.1.4-5: the ground
  is one plane) and the pit-exit rule (TR30.2).
* **High jump.** A bar along the side axis at ``line_fwd`` m, between standards
  ``2 * bar_half_length`` apart. ``bar_height`` is the official height, from the ground to the top of the bar
  (TR26 Height Measurement); the bar's centre is ``bar_radius`` lower. The attempt fails if a body collider touches the bar (it would fall), the body
  passes under it, the athlete runs out past the standards, touches the ground beyond the bar plane before
  clearing, or takes off from both feet (TR27.1-27.2). Any bar overlap fails (a conservative static-bar convention:
  real bars fall only when dislodged). Bar contact is tested on the swept motion between control steps, as 3-D
  segment distances. It is cleared when every collider is past the bar plane without touching it.

The judge only needs per-step collider geometry: each body collider as a segment (sphere: zero length) with a
radius, which foot it belongs to (or none), and the whole-body centre of mass. Ground contact is "lowest point
of the collider within ``contact_eps`` of the ground".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# phases
APPROACH, AIRBORNE_CANDIDATE, LANDED, DONE = 0, 1, 2, 3
JUDGE_VERSION = "3"  # bump when a rule or metric definition changes


@dataclass(frozen=True)
class JumpSpec:
    event: Literal["long", "high"]
    line_fwd: float  # long: take-off line; high: bar plane (task forward, m)
    bar_height: float = 0.0  # high: official height, ground to the top of the bar (m, TR26); centre = - bar_radius
    bar_radius: float = 0.015  # 30 mm crossbar (TR26 Crossbar)
    bar_half_length: float = (
        2.0  # 4.00 m crossbar, uprights 4.00-4.04 m apart (TR26, TR27 Apparatus)
    )
    # ground contact: collider lowest point below this height (m). Calibrated against the plugin's ground reaction
    # force (eval_jumps.py contact_vs_grf, sprinter on both events): Bolt's penalty contact only pushes when
    # penetrating, so 0 matches GRF > 1 N on 99.8 % of steps (0 misses); 1 cm flagged 6 % of airborne steps.
    contact_eps: float = 0.0
    # a "jump" whose flight is this short and this flat is a running stride over the line (reported, still legal)
    stride_flight_s: float = 0.2
    stride_angle_deg: float = 10.0
    mark_window_s: float = (
        1.0  # long: landing contacts that count towards the mark (timed surrogate)
    )
    sweep_substeps: int = (
        8  # high: bar contact tested at this many points between control samples
    )
    crossed_margin: float = (
        0.05  # high: every collider this far past the bar plane = cleared
    )
    # reward weights (training; the benchmark reads the metrics, not the return)
    lambda_mark: float = 2.0  # long: once per valid jump, per metre of mark
    # high: per metre of new COM peak inside the take-off zone (paid once per gain, so the total is bounded by the
    # best rise: hopping in place cannot farm it)
    lambda_rise: float = 2.0
    rise_zone: float = (
        3.0  # high: rise reward paid from this distance before the bar to the bar
    )
    lambda_clear: float = 5.0  # high: once when the bar is cleared
    approach_v_cap: float | None = (
        None  # cap on the paid approach speed (m/s); None = sprint reward
    )
    extra: dict = field(default_factory=dict)


LONG_JUMP = JumpSpec("long", line_fwd=20.0)
HIGH_JUMP = JumpSpec("high", line_fwd=15.0, bar_height=1.0, approach_v_cap=6.0)
# benchmark ladder (bar centre heights, m) for "best height cleared"
HIGH_JUMP_LADDER = (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)


def high_jump_at(h: float) -> JumpSpec:
    from dataclasses import replace

    return replace(HIGH_JUMP, bar_height=float(h))


def bar_scene_geoms(spec: JumpSpec) -> list[dict]:
    """The bar as a static capsule in the scene-JSON format (Bolt frame: x forward, y up, z side; capsule axis =
    local z, i.e. along the side axis with an identity quaternion). Bounds are the correct ones (rbound r + h,
    AABB along the local axis), not Bolt c1db1fe's (finding 9), so the bar ends are not culled."""
    if spec.event != "high":
        return []
    r, h = spec.bar_radius, spec.bar_half_length
    return [
        {
            "name": f"hj_bar_{spec.bar_height:.2f}",
            "body": "ground",
            "type": 3,
            "size": [r, h, r],
            "pos": [spec.line_fwd, spec.bar_height - r, 0.0],
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            "friction": [0.8, 0.8, 0.0],
            "stiffness": 29240.177734375,
            "dissipation": 1.0,
            "transition_velocity": 0.1,
            "priority": 9,
            "aabb": [[0.0, 0.0, 0.0], [2 * r, 2 * r, 2 * (h + r)]],
            "rbound": r + h,
        }
    ]


def _seg_seg_dist(p1, q1, p2, q2):
    """Closest distance between 3-D segments p1-q1 (..., 3) and p2-q2 (3,) (Ericson, Real-Time Collision Detection
    5.1.9); the second segment has non-zero length."""
    d1, d2 = q1 - p1, q2 - p2
    r = p1 - p2
    a = np.sum(d1 * d1, -1)
    e = float(d2 @ d2)
    f = r @ d2
    c = np.sum(d1 * r, -1)
    b = d1 @ d2
    pt = a <= 1e-12
    a_s = np.where(pt, 1.0, a)
    den = a * e - b * b
    s = np.where(
        den > 1e-12,
        np.clip((b * f - c * e) / np.where(den > 1e-12, den, 1.0), 0, 1),
        0.0,
    )
    t = (b * s + f) / e
    s = np.where(
        t < 0, np.clip(-c / a_s, 0, 1), np.where(t > 1, np.clip((b - c) / a_s, 0, 1), s)
    )
    t = np.clip(t, 0, 1)
    s = np.where(pt, 0.0, s)
    t = np.where(pt, np.clip(f / e, 0, 1), t)
    return np.linalg.norm(p1 + s[..., None] * d1 - (p2 + t[..., None] * d2), axis=-1)


def _footprint_fwd(a, b, radius, eps):
    """Forward interval of the part of each collider whose surface is below ``eps`` (ground contact footprint along
    the segment; a sphere's is its centre). Returns (touch, fwd_min, fwd_max), shape (..., C)."""
    za, zb = a[..., 2] - radius, b[..., 2] - radius
    ina, inb = za < eps, zb < eps
    dz = zb - za
    ts = np.clip((eps - za) / np.where(np.abs(dz) > 1e-12, dz, 1e-12), 0, 1)
    ps = a + ts[..., None] * (b - a)
    e1 = np.where(ina[..., None], a, ps)
    e2 = np.where(inb[..., None], b, ps)
    return (
        ina | inb,
        np.minimum(e1[..., 0], e2[..., 0]),
        np.maximum(e1[..., 0], e2[..., 0]),
    )


class JumpJudge:
    """Vectorised over N worlds. Call :meth:`reset` per world, then :meth:`update` once per control step with
    collider segments in task axes. ``update`` returns per-world events; :attr:`results` holds the attempt
    record of each world's current episode (filled as the attempt progresses).

    States: APPROACH (support or running strides) -> AIRBORNE_CANDIDATE (a flight that left the ground legally:
    long jump behind the line, high jump inside the take-off zone) -> LANDED (long jump mark window) -> DONE.
    Lane and posture rules are suspended from AIRBORNE_CANDIDATE on (``suspend_lane_rules``)."""

    def __init__(
        self, spec: JumpSpec, n: int, foot: np.ndarray, names: list[str] | None = None
    ):
        self.spec = spec
        self.names = names
        self.n = n
        self.foot = np.asarray(foot)  # (C,) 0 = not a foot, 1 = right, 2 = left
        self.results: list[dict] = [{} for _ in range(n)]
        self.phase = np.zeros(n, int)
        self.was_contact = np.ones(n, bool)
        self.t = np.zeros(n)
        self.last = [
            dict() for _ in range(n)
        ]  # last ground-contact frame before the current flight
        self.flight_start = [dict() for _ in range(n)]
        self.com0 = np.full(n, np.nan)
        self.prev_a = None  # (N, C, 3) segment ends at the previous step (swept bar test, plane crossings)
        self.prev_b = None
        self.crossed_above = np.zeros(n, bool)

    def reset(self, w: int) -> None:
        self.phase[w] = APPROACH
        self.was_contact[w] = True
        self.t[w] = 0.0
        self.last[w] = {}
        self.flight_start[w] = {}
        self.com0[w] = np.nan
        if self.prev_a is not None:
            self.prev_a[w] = np.nan
            self.prev_b[w] = np.nan
        self.crossed_above[w] = False
        self.results[w] = {
            "event": self.spec.event,
            "judge_version": JUDGE_VERSION,
            "line_fwd": self.spec.line_fwd,
            "bar_height": self.spec.bar_height if self.spec.event == "high" else None,
            "outcome": "running",
            "valid": False,
            "com_peak_up": -np.inf,
            "flight_peak_up": -np.inf,
            "min_bar_gap": np.inf,
            "max_fwd_before_takeoff": -np.inf,
        }

    # -------------------------------------------------------------------------------------------------- update
    def update(
        self,
        dt: float,
        seg_a: np.ndarray,  # (N, C, 3) collider segment ends, task axes (fwd, side, up)
        seg_b: np.ndarray,  # (N, C, 3)
        radius: np.ndarray,  # (C,)
        com: np.ndarray,  # (N, 3) whole-body COM, task axes
        com_vel: np.ndarray,  # (N, 3)
        pelvis_up: np.ndarray,  # (N,) pelvis height above ground
        fall_height: float,
    ) -> dict[str, np.ndarray]:
        s = self.spec
        n = self.n
        self.t += dt
        touch, foot_min, foot_max = _footprint_fwd(
            seg_a, seg_b, radius[None, :], s.contact_eps
        )
        any_touch = touch.any(axis=1)
        front = (
            np.maximum(seg_a[..., 0], seg_b[..., 0]) + radius[None, :]
        )  # forward surface (N, C)
        if self.prev_a is None:
            self.prev_a = np.full(seg_a.shape, np.nan)
            self.prev_b = np.full(seg_b.shape, np.nan)
        prev_a, prev_b = self.prev_a.copy(), self.prev_b.copy()
        self.prev_a, self.prev_b = seg_a.copy(), seg_b.copy()
        bonus = np.zeros(n)
        done = np.zeros(n, bool)
        susp_out = np.zeros(n, bool)
        for w in range(n):
            r = self.results[w]
            if self.phase[w] == DONE:
                done[w] = True
                continue
            if np.isnan(self.com0[w]):
                self.com0[w] = com[w, 2]
            r["com_peak_up"] = max(r["com_peak_up"], float(com[w, 2]))
            tw = touch[w]
            # supporting feet: every collider of a foot side with any collider on the ground (toe included)
            sides = {int(f) for f in self.foot[tw] if f > 0}
            sup = np.isin(self.foot, list(sides)) if sides else np.zeros_like(tw)
            if any_touch[w]:
                self.last[w] = {
                    "t": float(self.t[w]),
                    "toe_fwd": float(front[w][sup].max())
                    if sup.any()
                    else float(foot_max[w][tw].max()),
                    "feet": sorted(sides),
                    "nonfoot": bool((self.foot[tw] == 0).any()),
                    "com_vel": com_vel[w].tolist(),
                    "com_up": float(com[w, 2]),
                    "com_fwd": float(com[w, 0]),
                }
            elif (
                self.was_contact[w] and self.last[w]
            ):  # flight starts now; the take-off frame is the last contact
                self.flight_start[w] = {
                    **self.last[w],
                    "dt": float(dt),
                    "t_off": float(self.t[w]),
                    "com_vel_off": com_vel[w].tolist(),
                }
                r["flight_peak_up"] = -np.inf
            if not any_touch[w]:
                r["flight_peak_up"] = max(r["flight_peak_up"], float(com[w, 2]))
            if s.event == "long":
                ended, b, susp = self._long(
                    w,
                    tw,
                    sup,
                    front[w],
                    foot_min[w],
                    foot_max[w],
                    any_touch[w],
                    pelvis_up[w],
                    fall_height,
                )
            else:
                ended, b, susp = self._high(
                    w,
                    seg_a[w],
                    seg_b[w],
                    prev_a[w],
                    prev_b[w],
                    radius,
                    tw,
                    foot_max[w],
                    any_touch[w],
                    com[w],
                    pelvis_up[w],
                    fall_height,
                )
            bonus[w] += b
            susp_out[w] = susp
            if ended:
                self.phase[w] = DONE
                done[w] = True
            self.was_contact[w] = any_touch[w]
        return {"done": done, "bonus": bonus, "suspend_lane_rules": susp_out}

    def _takeoff_record(self, w: int) -> dict:
        fs = self.flight_start[w]
        if not fs:
            return {}
        vx, vs, vz = fs["com_vel_off"]
        # the foot left the ground between the last contact sample and the first airborne one (dt later): add back
        # half a step of gravity (midpoint estimate; error <= g dt / 2 = 0.16 m/s at 30 Hz if the COM is ballistic
        # over that step and the release lies inside it). Positions (toe, COM height) are the last contact sample.
        vz = vz + 9.80665 * fs.get("dt", 0.0) / 2.0
        vh = math.hypot(vx, vs)
        return {
            "takeoff_t": fs["t_off"],
            "takeoff_t_bracket": [fs.get("t", fs["t_off"]), fs["t_off"]],
            "takeoff_toe_fwd": fs["toe_fwd"],
            "takeoff_feet": fs["feet"],
            "single_leg_takeoff": len(fs["feet"]) == 1 and not fs["nonfoot"],
            "takeoff_vx": vx,
            "takeoff_vside": vs,
            "takeoff_vz": vz,
            "takeoff_speed": math.sqrt(vh * vh + vz * vz),
            "takeoff_angle_deg": math.degrees(math.atan2(vz, max(vh, 1e-6))),
            "takeoff_com_up": fs["com_up"],
        }

    # long jump ----------------------------------------------------------------------------------------------
    def _long(
        self, w, touch, sup, front, fmin, fmax, any_touch, pelvis_up, fall_height
    ):
        s, r = self.spec, self.results[w]
        L = s.line_fwd
        if self.phase[w] == AIRBORNE_CANDIDATE:
            if not any_touch:
                return False, 0.0, True
            if (
                touch & (fmax > L)
            ).any():  # first contact beyond the line after a legal take-off: the landing
                self.phase[w] = LANDED
                r.update(self._takeoff_record(w))
                r["landing_t"] = float(self.t[w])
                # airborne control steps x dt (resolution one control step, 1/30 s)
                r["flight_time_s"] = r["landing_t"] - r["takeoff_t"]
                r["mark_fwd"] = float(fmin[touch].min())
                r["board_gap_m"] = L - r["takeoff_toe_fwd"]
                r["com_rise_m"] = r["flight_peak_up"] - r["takeoff_com_up"]
                return False, 0.0, True
            self.phase[w] = APPROACH  # a running stride that came down behind the line
        if self.phase[w] == APPROACH:
            if any_touch:
                over = sup & (front > L)
                if (
                    over.any()
                ):  # TR30.1.1: the take-off foot breaks the plane of the line
                    r.update(
                        outcome="foul",
                        valid=False,
                        mark_m=0.0,
                        foul_fwd=float(front[over].max()),
                    )
                    return True, 0.0, False
                if (
                    touch & (self.foot == 0) & (fmax > L)
                ).any():  # fell over the line before jumping
                    r.update(outcome="fallen", valid=False, mark_m=0.0)
                    return True, 0.0, False
                r["max_fwd_before_takeoff"] = max(
                    r["max_fwd_before_takeoff"],
                    float(front[sup].max()) if sup.any() else float(fmax[touch].max()),
                )
            elif self.flight_start[
                w
            ]:  # left the ground with nothing over the line: a legal flight
                self.phase[w] = AIRBORNE_CANDIDATE
                return False, 0.0, True
            if pelvis_up < fall_height:
                r.update(outcome="fallen", valid=False, mark_m=0.0)
                return True, 0.0, False
            return False, 0.0, False
        # LANDED: the nearest footprint point within the mark window
        if any_touch:
            r["mark_fwd"] = min(r["mark_fwd"], float(fmin[touch].min()))
        if self.t[w] - r["landing_t"] >= s.mark_window_s - 1e-9:
            mark = max(0.0, r["mark_fwd"] - L)
            stride = (
                r["flight_time_s"] < s.stride_flight_s
                and r["takeoff_angle_deg"] < s.stride_angle_deg
            )
            r.update(
                outcome="jump",
                valid=True,
                mark_m=mark,
                mark_official_m=math.floor(mark * 100 + 1e-9) / 100,
                effective_m=mark + r["board_gap_m"],
                stride_like=stride,
            )
            return True, s.lambda_mark * mark, True
        return False, 0.0, True

    # high jump ----------------------------------------------------------------------------------------------
    def _bar_gap(self, a, b, radius):
        s = self.spec
        hc = s.bar_height - s.bar_radius  # bar centre
        p2 = np.array([s.line_fwd, -s.bar_half_length, hc])
        q2 = np.array([s.line_fwd, s.bar_half_length, hc])
        return _seg_seg_dist(a, b, p2, q2) - radius - s.bar_radius

    def _high(
        self,
        w,
        a,
        b,
        pa,
        pb,
        radius,
        touch,
        fmax,
        any_touch,
        com,
        pelvis_up,
        fall_height,
    ):
        s, r = self.spec, self.results[w]
        L, H = s.line_fwd, s.bar_height - s.bar_radius  # H: bar centre
        # bar contact on the swept motion since the previous control sample (3-D segment distance)
        gap = self._bar_gap(a, b, radius)
        if np.isfinite(pa).all():
            for k in range(1, s.sweep_substeps):
                al = k / s.sweep_substeps
                gap = np.minimum(
                    gap, self._bar_gap(pa + al * (a - pa), pb + al * (b - pb), radius)
                )
        g = float(gap.min())
        rear = float(np.minimum(a[:, 0], b[:, 0]).min() - radius.max())
        frontmost = float(np.maximum(a[:, 0], b[:, 0]).max())
        crossing = frontmost >= L - 0.3 and rear <= L + s.crossed_margin
        if crossing:
            r["min_bar_gap"] = min(r["min_bar_gap"], g)
        # a flight from the take-off zone is the attempt: lane and posture rules are suspended from take-off on
        if (
            self.phase[w] == APPROACH
            and not any_touch
            and self.flight_start[w]
            and self.flight_start[w].get("com_fwd", -np.inf) >= L - s.rise_zone
        ):
            self.phase[w] = AIRBORNE_CANDIDATE
        elif (
            self.phase[w] == AIRBORNE_CANDIDATE
            and any_touch
            and not (touch & (fmax > L)).any()
        ):
            self.phase[w] = (
                APPROACH  # came down in front of the bar: a new take-off is needed
            )
        bonus = 0.0
        susp = crossing or self.phase[w] == AIRBORNE_CANDIDATE
        # rise reward in the take-off zone (training shaping): each new COM peak above the start height
        if L - s.rise_zone <= com[0] <= L:
            peak = r.setdefault("zone_peak_up", float(self.com0[w]))
            if com[2] > peak:
                bonus += s.lambda_rise * float(com[2] - peak)
                r["zone_peak_up"] = float(com[2])

        def fail(outcome, **kw):
            r.update(outcome=outcome, valid=False, cleared=False, **kw)
            r.update(self._takeoff_record(w))
            return True, bonus, False

        if g < 0.0:
            k = int(np.argmin(gap))
            return fail("knocked", knocked_by=self.names[k] if self.names else k)
        if com[0] >= L and r.get("com_cross_up") is None:
            r["com_cross_up"] = float(
                com[2]
            )  # a flop's COM may pass below the bar: recorded, not judged
        # every body part must pass over the bar between the standards: classify each collider centre where it
        # crosses the bar plane (interpolated). Colliders do not cover the whole body (the neck between trunk and
        # head), so "no contact" alone would let a runner duck under a high bar.
        pc, cc = 0.5 * (pa + pb), 0.5 * (a + b)
        cross = (pc[:, 0] < L) & (cc[:, 0] >= L)
        if cross.any():
            f = (L - pc[cross, 0]) / (cc[cross, 0] - pc[cross, 0] + 1e-12)
            up = pc[cross, 2] + f * (cc[cross, 2] - pc[cross, 2])
            side = pc[cross, 1] + f * (cc[cross, 1] - pc[cross, 1])
            if (np.abs(side) > s.bar_half_length).any():
                return fail("run_out")
            if (up < H).any():
                # under: the whole body is below the bar (ducked); through: part of it passes below while the rest
                # is at or above bar height (e.g. the lead leg under a hip-high bar), so the bar would be knocked
                top = float((np.maximum(a[:, 2], b[:, 2]) + radius).max())
                outcome = (
                    "under" if (top < H and not self.crossed_above[w]) else "through"
                )
                kw = (
                    {
                        "under_by": self.names[
                            int(np.flatnonzero(cross)[int(np.argmin(up))])
                        ]
                    }
                    if self.names
                    else {}
                )
                return fail(outcome, **kw)
            self.crossed_above[w] = True
        if any_touch and (touch & (fmax > L)).any():
            return fail(
                "ground_beyond"
            )  # touched the ground past the bar plane before clearing (TR27.2)
        if rear > L + s.crossed_margin:
            if (
                self.phase[w] != AIRBORNE_CANDIDATE
            ):  # past the plane without a judged flight (not reachable over a bar)
                return fail("no_takeoff")
            r.update(self._takeoff_record(w))
            two_feet = not r.get("single_leg_takeoff", False)
            r.update(
                outcome="two_foot_takeoff" if two_feet else "cleared",
                valid=not two_feet,
                cleared=not two_feet,
                takeoff_dist_m=L - r.get("takeoff_toe_fwd", np.nan),
                com_peak_minus_bar=r["flight_peak_up"] - s.bar_height,
                com_rise_m=r["flight_peak_up"] - r.get("takeoff_com_up", np.nan),
            )
            return True, (0.0 if two_feet else s.lambda_clear) + bonus, True
        if not susp and pelvis_up < fall_height:
            return fail("fallen")
        return False, bonus, susp

    def finalize(self, w: int, cause: str | None = None) -> dict:
        """Attempt record at the end of the episode. An attempt still running ends as ``cause`` (the env's own
        ending, e.g. toes_out on the runway) or, at the time limit, ``no_jump``."""
        r = dict(self.results[w])
        if r.get("outcome") == "running":
            r.update(outcome=cause or "no_jump", valid=False)
            if self.spec.event == "long":
                r["mark_m"] = 0.0
            else:
                r["cleared"] = False
        for k, v in list(r.items()):
            if isinstance(v, float) and not math.isfinite(v):
                r[k] = None
        return r


def summarize(records: list[dict], event: str) -> dict:
    """Benchmark summary over attempts (one per world)."""
    n = len(records)
    out = {"n": n, "outcomes": {}}
    for r in records:
        out["outcomes"][r["outcome"]] = out["outcomes"].get(r["outcome"], 0) + 1

    def stat(key, sel=lambda r: True):
        v = [r[key] for r in records if sel(r) and r.get(key) is not None]
        return (
            None
            if not v
            else {
                "mean": float(np.mean(v)),
                "median": float(np.median(v)),
                "max": float(np.max(v)),
                "n": len(v),
            }
        )

    if event == "long":
        valid = lambda r: r.get("valid")  # noqa: E731
        out.update(
            valid_rate=sum(bool(r.get("valid")) for r in records) / max(n, 1),
            # valid and not a running stride over the line
            jump_rate=sum(
                bool(r.get("valid")) and not r.get("stride_like") for r in records
            )
            / max(n, 1),
            foul_rate=out["outcomes"].get("foul", 0) / max(n, 1),
            best_mark_m=max([r.get("mark_m") or 0.0 for r in records] or [0.0]),
            # every attempt, fouls and endings scored 0 (the headline score; best-of grows with the attempt budget)
            mean_score_m=float(np.mean([r.get("mark_m") or 0.0 for r in records]))
            if n
            else 0.0,
            mark_m=stat("mark_m", valid),
            effective_m=stat("effective_m", valid),
            board_gap_m=stat("board_gap_m", valid),
            takeoff_vx=stat("takeoff_vx", valid),
            takeoff_vz=stat("takeoff_vz", valid),
            takeoff_angle_deg=stat("takeoff_angle_deg", valid),
            flight_time_s=stat("flight_time_s", valid),
            com_rise_m=stat("com_rise_m", valid),
            single_leg_rate=(
                sum(
                    bool(r.get("single_leg_takeoff")) for r in records if r.get("valid")
                )
                / max(1, sum(bool(r.get("valid")) for r in records))
            ),
        )
    else:
        out.update(
            clear_rate=sum(bool(r.get("cleared")) for r in records) / max(n, 1),
            knocked_by={
                k: sum(str(r.get("knocked_by")) == k for r in records)
                for k in sorted(
                    {
                        str(r.get("knocked_by"))
                        for r in records
                        if r.get("knocked_by") is not None
                    }
                )
            },
            min_bar_gap=stat("min_bar_gap"),
            com_peak_up=stat("com_peak_up"),
            com_peak_minus_bar=stat("com_peak_minus_bar", lambda r: r.get("cleared")),
            takeoff_angle_deg=stat("takeoff_angle_deg"),
            takeoff_vz=stat("takeoff_vz"),
        )
    return out
