# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for eye-head coordination and cervical/upper-limb ergonomic metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from myosuite.biomechanics.visuomotor_metrics import (
    cervical_flexion_moment_nm,
    craniovertebral_angle_deg,
    eye_head_coupling_lag,
    eye_head_onset_latency,
    head_contribution_ratio,
    upper_limb_rula_score,
    vor_gain,
)

pytestmark = pytest.mark.tier1


def test_head_contribution_onset_latency_and_vor_gain() -> None:
    # 40 deg gaze shift where head rotates 24 deg -> HCR = 0.60
    gaze_deg = np.linspace(0.0, 40.0, 51)
    head_deg = np.linspace(0.0, 24.0, 51)
    assert head_contribution_ratio(gaze_deg, head_deg) == pytest.approx(0.60)

    # 3D unit vectors: 60 deg gaze shift, 30 deg head shift -> HCR = 0.50
    g_vec = np.array([[1.0, 0.0, 0.0], [0.5, math.sqrt(0.75), 0.0]])
    h_vec = np.array([[1.0, 0.0, 0.0], [math.sqrt(0.75), 0.5, 0.0]])
    assert head_contribution_ratio(g_vec, h_vec) == pytest.approx(0.50, abs=1e-6)

    # Bell-shaped speed profile where head lags gaze by 4 samples at dt=0.01 s -> +0.04 s
    dt = 0.01
    t = np.arange(80) * dt
    gaze_speed = 250.0 * np.exp(-0.5 * ((t - 0.20) / 0.03) ** 2)
    head_speed = 120.0 * np.exp(-0.5 * ((t - 0.24) / 0.03) ** 2)
    latency = eye_head_onset_latency(gaze_speed, head_speed, dt, threshold_ratio=0.10)
    assert latency == pytest.approx(0.04, abs=1e-9)

    lag_s, corr = eye_head_coupling_lag(gaze_speed, head_speed, dt)
    assert lag_s == pytest.approx(0.04, abs=1e-9)
    assert corr == pytest.approx(1.0, abs=1e-3)

    # VOR gain: eye_in_head = -0.92 * head_vel
    h_vel = 40.0 * np.sin(2.0 * math.pi * 1.5 * t)
    e_vel = -0.92 * h_vel
    assert vor_gain(e_vel, h_vel, min_head_speed=5.0) == pytest.approx(0.92, rel=1e-5)


def test_craniovertebral_angle_and_cervical_flexion_moment() -> None:
    c7 = np.array([0.0, 0.0, 1.40])
    # Neutral posture: tragus 8 cm forward, 10 cm above C7 -> atan2(0.10, 0.08) ~ 51.34 deg
    tragus_neutral = np.array([0.08, 0.0, 1.50])
    # Forward-head posture: tragus 14 cm forward, 8 cm above C7 -> ~29.74 deg
    tragus_fhp = np.array([0.14, 0.0, 1.48])

    cva_n = craniovertebral_angle_deg(c7, tragus_neutral)
    cva_f = craniovertebral_angle_deg(c7, tragus_fhp)
    assert isinstance(cva_n, float)
    assert cva_n == pytest.approx(math.degrees(math.atan2(0.10, 0.08)))
    assert cva_f < 35.0 < cva_n

    traj_cva = craniovertebral_angle_deg(
        np.stack([c7, c7]), np.stack([tragus_neutral, tragus_fhp])
    )
    assert isinstance(traj_cva, np.ndarray) and traj_cva.shape == (2,)

    # Cervical moment increases monotonically with anterior lever arm and braking (-x base acc)
    m_neutral = cervical_flexion_moment_nm(
        np.array([0.03, 0.0, 1.52]), c7, head_mass_kg=4.5
    )
    m_flexed = cervical_flexion_moment_nm(
        np.array([0.09, 0.0, 1.50]), c7, head_mass_kg=4.5
    )
    m_braking = cervical_flexion_moment_nm(
        np.array([0.09, 0.0, 1.50]),
        c7,
        head_mass_kg=4.5,
        base_lin_acc=(-4.0, 0.0, 0.0),
    )
    assert m_neutral == pytest.approx(4.5 * 9.81 * 0.03)
    assert m_neutral < m_flexed < m_braking


def test_upper_limb_rula_score_neutral_vs_strained() -> None:
    # Ideal neutral seated posture: UA=10 deg, LA=80 deg, Wrist=0 deg, Neck=5 deg, Trunk=0 deg
    neutral = upper_limb_rula_score(
        upper_arm_flex_deg=10.0,
        lower_arm_flex_deg=80.0,
        wrist_flex_deg=0.0,
        neck_flex_deg=5.0,
        trunk_flex_deg=0.0,
    )
    assert int(neutral.grand_scores[0]) == 1
    assert neutral.action_level == 1

    # Elevated, abducted overhead reach with flexed neck, twisted trunk, and static hold
    strained = upper_limb_rula_score(
        upper_arm_flex_deg=[95.0, 100.0],
        lower_arm_flex_deg=[30.0, 25.0],
        wrist_flex_deg=[20.0, 22.0],
        neck_flex_deg=[25.0, 28.0],
        trunk_flex_deg=[30.0, 35.0],
        shoulder_raised=True,
        upper_arm_abducted=True,
        wrist_deviated=True,
        wrist_twisted_end_range=True,
        neck_twisted=True,
        trunk_twisted=True,
        static_posture=True,
        force_load_score=1,
    )
    assert strained.grand_scores.shape == (2,)
    assert np.all(strained.grand_scores == 7)
    assert strained.action_level == 4

    with pytest.raises(ValueError, match="force_load_score"):
        upper_limb_rula_score(0.0, 80.0, 0.0, 5.0, 0.0, force_load_score=5)
