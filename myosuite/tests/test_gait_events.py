# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for O-GEST B-spline force-plate-free gait event detection."""

from __future__ import annotations

import numpy as np
import pytest

from myosuite.biomechanics.gait_events import (
    GaitEventConfig,
    contacts_from_gait_events,
    detect_gait_events,
)
from myosuite.biomechanics.retarget import _foot_contacts

pytestmark = pytest.mark.tier1


def _synthetic_gait_landmarks(
    *,
    fps: float = 100.0,
    duration_s: float = 3.6,
    stride_s: float = 1.0,
    stance_frac: float = 0.60,
    treadmill: bool = False,
    equinus: bool = False,
    heading_deg: float = 35.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n = int(round(duration_s * fps))
    t = np.arange(n, dtype=np.float64) / fps
    ang = np.radians(heading_deg)
    fwd = np.array([np.cos(ang), np.sin(ang), 0.0])
    lat = np.array([-np.sin(ang), np.cos(ang), 0.0])

    speed = 0.0 if treadmill else 1.25
    pelvis = (speed * t)[:, None] * fwd[None, :] + np.array([0.0, 0.0, 0.92])

    def limb_traj(offset_s: float, side_sign: float) -> tuple[np.ndarray, np.ndarray]:
        phase = np.mod((t - offset_s) / stride_s, 1.0)
        excursion = np.where(
            phase < stance_frac,
            0.22 * np.cos(np.pi * phase / stance_frac),
            -0.22 * np.cos(np.pi * (phase - stance_frac) / (1.0 - stance_frac)),
        )
        swing_z = np.where(
            phase >= stance_frac,
            0.07 * np.sin(np.pi * (phase - stance_frac) / (1.0 - stance_frac)),
            0.0,
        )
        heel_z = (0.06 if equinus else 0.02) + swing_z
        toe_z = 0.015 + swing_z
        y_off = side_sign * 0.11 * lat
        heel = (
            pelvis[:, :2] @ np.eye(2, 3)
            + y_off
            + (excursion - 0.08)[:, None] * fwd[None, :]
            + np.column_stack([np.zeros(n), np.zeros(n), heel_z])
        )
        toe = (
            pelvis[:, :2] @ np.eye(2, 3)
            + y_off
            + (excursion + 0.12)[:, None] * fwd[None, :]
            + np.column_stack([np.zeros(n), np.zeros(n), toe_z])
        )
        return heel, toe

    lh, lt = limb_traj(0.25, 1.0)
    rh, rt = limb_traj(0.75, -1.0)
    true_l_strikes = np.array(
        [
            0.25 + k * stride_s
            for k in range(4)
            if 0.25 + k * stride_s < duration_s - 0.1
        ]
    )
    true_l_offs = np.array(
        [
            0.25 + stance_frac * stride_s + k * stride_s
            for k in range(4)
            if 0.25 + stance_frac * stride_s + k * stride_s < duration_s - 0.1
        ]
    )
    return pelvis, lh, lt, rh, rt, true_l_strikes, true_l_offs


@pytest.mark.parametrize(
    ("treadmill", "equinus"),
    [(False, False), (True, False), (False, True)],
)
def test_ogest_overground_treadmill_and_equinus_events(
    treadmill: bool, equinus: bool
) -> None:
    fps = 100.0
    pel, lh, lt, rh, rt, true_ls, true_lo = _synthetic_gait_landmarks(
        fps=fps, treadmill=treadmill, equinus=equinus
    )
    rng = np.random.default_rng(7)
    lh_noisy = lh + rng.normal(0.0, 0.0015, size=lh.shape)
    lt_noisy = lt + rng.normal(0.0, 0.0015, size=lt.shape)

    res = detect_gait_events(
        pel, lh_noisy, lt_noisy, rh, rt, fps, GaitEventConfig(min_stride_s=0.6)
    )
    assert len(res.left_strike_times) == len(true_ls)
    assert len(res.left_off_times) == len(true_lo)
    np.testing.assert_allclose(res.left_strike_times, true_ls, atol=0.025)
    np.testing.assert_allclose(res.left_off_times, true_lo, atol=0.025)

    sole = np.stack([lh, lt, rh, rt], axis=1)
    sole_floor = np.percentile(sole[..., 2], 2, axis=0)
    mask_ogest = _foot_contacts(
        sole, sole_floor, fps, 2, 0.035, 0.5, mode="ogest", pelvis=pel
    )
    assert mask_ogest.shape == (len(pel), 2)
    assert 0.45 <= float(mask_ogest[:, 0].mean()) <= 0.75


def test_ogest_quiet_standing_fallback_and_contact_mask_builder() -> None:
    n = 40
    pel = np.tile([0.0, 0.0, 0.92], (n, 1))
    lh = np.tile([0.0, 0.10, 0.01], (n, 1))
    lt = np.tile([0.18, 0.10, 0.01], (n, 1))
    rh = np.tile([0.0, -0.10, 0.01], (n, 1))
    rt = np.tile([0.18, -0.10, 0.01], (n, 1))
    res = detect_gait_events(pel, lh, lt, rh, rt, 60.0)
    assert len(res.left_strikes) == 0
    assert res.stance_mask.all()

    m = contacts_from_gait_events(10, np.array([2, 7]), np.array([5]))
    np.testing.assert_array_equal(
        m, [False, False, True, True, True, False, False, True, True, True]
    )
