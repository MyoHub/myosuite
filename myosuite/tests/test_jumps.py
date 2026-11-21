"""Judge tests for the long jump / high jump on a rigid kinematic body with analytic answers.

The body: two foot spheres (r 3 cm, right/left) 0.95 m below the COM and a trunk capsule (r 10 cm, half 0.3 m)
centred on the COM, translated rigidly along a prescribed run-then-ballistic path.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from myosuite.envs.myo.tasks.basic.full_body.jumps import (
    HIGH_JUMP,
    LONG_JUMP,
    JumpJudge,
    high_jump_at,
    summarize,
)

pytestmark = pytest.mark.tier1

G = 9.81
DT = 1.0 / 30.0
FOOT = np.array([1, 2, 0])
R = np.array([0.03, 0.03, 0.10])
COM_UP = 1.0
FOOT_DROP = COM_UP - 0.03  # foot sphere centre at 3 cm, lowest point on the ground


def body(com, two_feet_down=True, right_only=False):
    """Collider segments for a COM position (fwd, side, up)."""
    f, s, u = com
    foot_up = u - FOOT_DROP
    fr = [f, s - 0.1, foot_up]
    fl = [
        f - (0.0 if two_feet_down else 0.0),
        s + 0.1,
        foot_up + (0.2 if right_only else 0.0),
    ]
    a = np.array([fr, fl, [f, s, u - 0.3]])
    b = np.array([fr, fl, [f, s, u + 0.3]])
    return a, b


def run_path(judge, path, right_only=False, grounded=None):
    out = None
    for k, (com, vel) in enumerate(path):
        a, b = body(com, right_only=right_only)
        # penetrate 1 mm when on the ground (Bolt pushes only when penetrating)
        on = grounded[k] if grounded is not None else com[2] <= COM_UP + 1e-9
        if on:
            a[:2, 2] -= 0.001
            b[:2, 2] -= 0.001
        out = judge.update(
            DT,
            a[None],
            b[None],
            R,
            np.array([com]),
            np.array([vel]),
            np.array([com[2]]),
            0.6,
        )
        if out["done"][0]:
            break
    return out


def ballistic(x_off, vx, vz, line, t_run=1.0, t_after=1.5):
    """Run at vx on the ground until fwd = x_off, then a ballistic flight, then stand still where it lands."""
    path = []
    x = x_off - vx * t_run
    while x < x_off - 1e-9:
        path.append(((x, 0.0, COM_UP), (vx, 0.0, 0.0)))
        x += vx * DT
    t, z = 0.0, COM_UP
    x0 = x
    while True:
        t += DT
        z = COM_UP + vz * t - 0.5 * G * t * t
        if z <= COM_UP:
            break
        path.append(((x0 + vx * t, 0.0, z), (vx, 0.0, vz - G * t)))
    land = x0 + vx * t
    for _ in range(int(t_after / DT)):
        path.append(((land, 0.0, COM_UP), (0.0, 0.0, 0.0)))
    return path, x0, land, t


def test_long_jump_mark_matches_ballistic_flight():
    spec = replace(LONG_JUMP, line_fwd=20.0)
    j = JumpJudge(spec, 1, FOOT)
    j.reset(0)
    path, x0, land, t_flight = ballistic(19.8, 8.0, 3.0, spec.line_fwd)
    run_path(j, path)
    r = j.finalize(0)
    assert r["outcome"] == "jump" and r["valid"]
    # take-off frame = last contact (one step before the first airborne step)
    assert r["takeoff_toe_fwd"] == pytest.approx(
        x0 - 8.0 * DT + 0.03, abs=1e-6
    )  # foot front surface
    assert r["mark_m"] == pytest.approx(land - spec.line_fwd, abs=1e-6)
    # airborne control steps x dt: the landing frame is not airborne, so one step under the sampled flight
    assert r["flight_time_s"] == pytest.approx(t_flight - DT, abs=1e-9)
    # midpoint estimate of the take-off vz: within g dt / 2 of the true 3.0 m/s
    assert r["takeoff_vz"] == pytest.approx(
        3.0, abs=G * DT / 2 + 1e-3
    )  # worst case: leaves at the last contact sample
    assert r["takeoff_angle_deg"] == pytest.approx(
        math.degrees(math.atan2(3.0, 8.0)), abs=1.2
    )
    assert not r["stride_like"]
    assert r["board_gap_m"] == pytest.approx(spec.line_fwd - r["takeoff_toe_fwd"])


def test_long_jump_foul_when_running_over_the_line():
    j = JumpJudge(LONG_JUMP, 1, FOOT)
    j.reset(0)
    path = [
        ((x, 0.0, COM_UP), (8.0, 0.0, 0.0)) for x in np.arange(18.0, 22.0, 8.0 * DT)
    ]
    out = run_path(j, path)
    r = j.finalize(0)
    assert out["done"][0] and r["outcome"] == "foul" and r["mark_m"] == 0.0


def test_long_jump_stride_over_line_is_flagged():
    j = JumpJudge(LONG_JUMP, 1, FOOT)
    j.reset(0)
    path, *_ = ballistic(19.9, 5.0, 0.5, LONG_JUMP.line_fwd)
    run_path(j, path)
    r = j.finalize(0)
    assert r["valid"] and r["stride_like"]
    s = summarize([r], "long")
    assert s["valid_rate"] == 1.0 and s["jump_rate"] == 0.0


def test_long_jump_nearest_break_counts():
    """A later contact behind the first landing point (falling back) shortens the mark."""
    j = JumpJudge(LONG_JUMP, 1, FOOT)
    j.reset(0)
    path, x0, land, _ = ballistic(19.8, 8.0, 3.0, LONG_JUMP.line_fwd, t_after=0.0)
    path += [((land - 0.4, 0.0, COM_UP), (0.0, 0.0, 0.0))] * 40
    run_path(j, path)
    r = j.finalize(0)
    assert r["mark_m"] == pytest.approx(land - 0.4 - LONG_JUMP.line_fwd, abs=1e-6)


def _high_path(spec, vz, vx=4.0):
    path, x0, land, _ = ballistic(
        spec.line_fwd - 1.0, vx, vz, spec.line_fwd, t_after=0.5
    )
    return path


def test_high_jump_cleared_with_margin():
    spec = high_jump_at(0.6)
    j = JumpJudge(spec, 1, FOOT)
    j.reset(0)
    # COM peak 1 + 4^2 / 2g = 1.82 m; feet at peak 0.85 m above ground; flight centred on the bar
    run_path(j, _high_path(spec, vz=4.0, vx=2.4), grounded=None)
    r = j.finalize(0)
    # the body is two-footed in this kinematic model only if both feet touch at take-off: right foot only here
    assert r["outcome"] in ("cleared", "two_foot_takeoff")
    assert r["min_bar_gap"] > 0.0


def test_high_jump_single_leg_rule():
    spec = high_jump_at(0.6)
    j = JumpJudge(spec, 1, FOOT)
    j.reset(0)
    run_path(j, _high_path(spec, vz=4.0, vx=2.4), right_only=True)
    r = j.finalize(0)
    assert r["outcome"] == "cleared" and r["cleared"] and r["single_leg_takeoff"]
    j.reset(0)
    run_path(j, _high_path(spec, vz=4.0, vx=2.4), right_only=False)
    assert j.finalize(0)["outcome"] == "two_foot_takeoff"


def test_high_jump_knocked_when_running_through():
    j = JumpJudge(HIGH_JUMP, 1, FOOT, names=["foot_r", "foot_l", "trunk"])
    j.reset(0)
    path = [
        ((x, 0.0, COM_UP), (5.0, 0.0, 0.0)) for x in np.arange(12.0, 17.0, 5.0 * DT)
    ]
    run_path(j, path)
    r = j.finalize(0)
    assert r["outcome"] == "knocked" and r["knocked_by"] == "trunk"


def test_high_jump_under_the_bar():
    spec = high_jump_at(2.0)
    j = JumpJudge(spec, 1, FOOT)
    j.reset(0)
    path = [
        ((x, 0.0, COM_UP), (5.0, 0.0, 0.0)) for x in np.arange(12.0, 17.0, 5.0 * DT)
    ]
    run_path(j, path)
    assert j.finalize(0)["outcome"] == "under"


def test_high_jump_ducking_under_through_a_collider_gap_is_not_a_clearance():
    """Bar between the trunk top (1.4 m) and a head sphere floating above it: no contact, still under/through."""
    spec = high_jump_at(1.5)
    j = JumpJudge(
        spec, 1, np.array([1, 2, 0, 0]), names=["foot_r", "foot_l", "trunk", "head"]
    )
    j.reset(0)
    rad = np.array([0.03, 0.03, 0.10, 0.08])
    for x in np.arange(12.0, 17.0, 5.0 * DT):
        a, b = body((x, 0.0, COM_UP))
        head = np.array([[x, 0.0, 1.62]])
        a, b = np.vstack([a, head]), np.vstack([b, head])
        a[:2, 2] -= 0.001
        b[:2, 2] -= 0.001
        out = j.update(
            DT,
            a[None],
            b[None],
            rad,
            np.array([[x, 0, COM_UP]]),
            np.array([[5.0, 0, 0]]),
            np.array([COM_UP]),
            0.6,
        )
        if out["done"][0]:
            break
    r = j.finalize(0)
    assert r["outcome"] in ("under", "through") and not r["cleared"]


# regression cases from the review (judge v2) ---------------------------------------------------------------


def _one(spec, a, b, rad, com, vel=(5.0, 0.0, 0.0), foot=(1,), judge=None):
    j = judge or JumpJudge(
        spec, 1, np.array(foot), names=[f"c{i}" for i in range(len(foot))]
    )
    out = j.update(
        DT,
        a[None],
        b[None],
        np.asarray(rad, float),
        np.array([com]),
        np.array([vel]),
        np.array([com[2]]),
        0.3,
    )
    return j, out


def test_long_jump_foul_on_foot_front_surface():
    """A grounded 3 cm foot sphere centred 1 cm behind the line reaches 2 cm past it: foul (TR 30.1.1)."""
    p = np.array([[19.99, 0.0, 0.029]])
    j = JumpJudge(LONG_JUMP, 1, np.array([1]))
    j.reset(0)
    j, out = _one(LONG_JUMP, p, p, [0.03], (19.9, 0.0, 1.0), judge=j)
    assert out["done"][0] and j.finalize(0)["outcome"] == "foul"


def test_high_jump_swept_bar_contact_between_samples():
    """A 3 cm sphere moving from 14.9 to 15.1 m at 1.02 m passes through the 1 m bar between two samples."""
    j = JumpJudge(HIGH_JUMP, 1, np.array([0]), names=["s"])
    j.reset(0)
    for x in (14.9, 15.1):
        p = np.array([[x, 0.0, 1.02]])
        out = j.update(
            DT,
            p[None],
            p[None],
            np.array([0.03]),
            np.array([[x, 0, 1.02]]),
            np.array([[6.0, 0, 0]]),
            np.array([1.02]),
            0.3,
        )
        if out["done"][0]:
            break
    assert j.finalize(0)["outcome"] == "knocked"


def test_long_jump_lane_rules_suspended_in_flight():
    j = JumpJudge(LONG_JUMP, 1, FOOT)
    j.reset(0)
    path, *_ = ballistic(19.8, 8.0, 3.0, LONG_JUMP.line_fwd)
    flags = []
    for com, vel in path:
        a, b = body(com)
        if com[2] <= COM_UP + 1e-9:
            a[:2, 2] -= 0.001
            b[:2, 2] -= 0.001
        out = j.update(
            DT,
            a[None],
            b[None],
            R,
            np.array([com]),
            np.array([vel]),
            np.array([com[2]]),
            0.6,
        )
        flags.append((com[2] > COM_UP + 1e-9, bool(out["suspend_lane_rules"][0])))
        if out["done"][0]:
            break
    assert all(s for air, s in flags if air)


def test_long_jump_delayed_backward_contact_inside_window():
    j = JumpJudge(LONG_JUMP, 1, FOOT)
    j.reset(0)
    path, x0, land, _ = ballistic(19.8, 8.0, 3.0, LONG_JUMP.line_fwd, t_after=0.0)
    path += [
        ((land, 0.0, COM_UP), (0.0, 0.0, 0.0))
    ] * 26  # 0.87 s standing, then falls back 0.5 m
    path += [((land - 0.5, 0.0, COM_UP), (0.0, 0.0, 0.0))] * 10
    run_path(j, path)
    assert j.finalize(0)["mark_m"] == pytest.approx(
        land - 0.5 - LONG_JUMP.line_fwd, abs=1e-6
    )
