# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for physical disturbance spectra and 6-axis IMU trace loaders."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from myosuite import make_env
from myosuite.envs.wrappers import BaseExcitationWrapper
from myosuite.physics import (
    BaseExcitationProfile,
    BoundedJerkProfile,
    CompositeBaseProfile,
    DiscreteGustProfile,
    DrydenTurbulenceProfile,
    GaitOscillationProfile,
    RoadRoughnessProfile,
    load_imu_base_profile,
)

pytestmark = pytest.mark.tier1


def test_dryden_turbulence_stationary_variance_and_reproducibility() -> None:
    prof = DrydenTurbulenceProfile(
        airspeed_mps=20.0,
        sigma_vel=(2.0, 1.0, 1.5),
        scale_lengths_m=(40.0, 40.0, 30.0),
        sigma_ang_vel=(0.05, 0.0, 0.02),
        ang_scale_lengths_m=(20.0, 20.0, 20.0),
    )
    assert isinstance(prof, BaseExcitationProfile)

    dt = 0.02
    s1 = prof.reset(np.random.default_rng(42), dt)
    s2 = prof.reset(np.random.default_rng(42), dt)
    samples1 = np.array([prof.sample(k * dt, s1).lin_acc for k in range(3000)])
    samples2 = np.array([prof.sample(k * dt, s2).lin_acc for k in range(3000)])
    np.testing.assert_allclose(samples1, samples2)

    # Analytical stationary acceleration std is sigma_vel / (L / V) = [1.0, 0.5, 1.0].
    emp_std = np.std(samples1, axis=0)
    np.testing.assert_allclose(emp_std, [1.0, 0.5, 1.0], rtol=0.18)


def test_discrete_gust_velocity_integral_and_recovery() -> None:
    gust = DiscreteGustProfile(
        gust_vel=(0.0, 0.0, 4.0),
        gradient_distance_m=20.0,
        airspeed_mps=10.0,
        onset_time=0.5,
        gust_ang_vel=(0.2, 0.0, 0.0),
    )
    assert isinstance(gust, BaseExcitationProfile)
    st = gust.reset(np.random.default_rng(0), 0.001)

    # Half duration H/V = 2.0 s; full duration = 4.0 s (active on [0.5, 4.5]).
    np.testing.assert_allclose(gust.sample(0.2, st).lin_acc, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(gust.sample(4.8, st).lin_acc, [0.0, 0.0, 0.0])

    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    t_half = np.linspace(0.5, 2.5, 2001)
    acc_half = np.array([gust.sample(float(t), st).lin_acc[2] for t in t_half])
    dv_peak = float(_trapz(acc_half, t_half))
    assert dv_peak == pytest.approx(4.0, rel=1e-4)

    t_full = np.linspace(0.5, 4.5, 4001)
    acc_full = np.array([gust.sample(float(t), st).lin_acc[2] for t in t_full])
    dv_net = float(_trapz(acc_full, t_full))
    assert dv_net == pytest.approx(0.0, abs=1e-6)
    assert gust.sample(2.5, st).ang_vel[0] == pytest.approx(0.2, rel=1e-6)


def test_road_roughness_iso_classes_and_bounded_jerk() -> None:
    rms_by_class: list[float] = []
    dt = 0.01
    for cls_name in ("A", "C", "E"):
        road = RoadRoughnessProfile(iso_class=cls_name, speed_mps=15.0)
        st = road.reset(np.random.default_rng(7), dt)
        az = np.array([road.sample(k * dt, st).lin_acc[2] for k in range(500)])
        rms_by_class.append(float(np.sqrt(np.mean(az**2))))

    assert rms_by_class[0] < rms_by_class[1] < rms_by_class[2]
    # Each step of two ISO classes scales Gd(n0) by 16x -> amplitude scales by 4x.
    assert rms_by_class[1] / rms_by_class[0] == pytest.approx(4.0, rel=1e-6)

    maneuver = BoundedJerkProfile(
        peak_lin_acc=(-3.0, 0.0, 0.0),
        jerk_limit=6.0,
        onset_time=0.2,
        hold_duration=0.8,
    )
    st_m = maneuver.reset(np.random.default_rng(0), dt)
    ts = np.linspace(0.0, 2.5, 251)
    ax = np.array([maneuver.sample(float(t), st_m).lin_acc[0] for t in ts])
    jerk_fd = np.abs(np.diff(ax) / np.diff(ts))
    assert float(np.max(jerk_fd)) <= 6.0 + 1e-6
    assert maneuver.sample(1.0, st_m).lin_acc[0] == pytest.approx(-3.0)


def test_gait_oscillation_harmonics_and_angular_derivative() -> None:
    gait = GaitOscillationProfile(
        step_freq_hz=2.0,
        vert_acc_amp=1.5,
        lat_acc_amp=0.6,
        ap_acc_amp=0.7,
        roll_vel_amp=0.08,
        pitch_vel_amp=0.05,
    )
    st = gait.reset(np.random.default_rng(11), 0.005)
    # At step period T_step = 0.5 s, vertical cosine repeats every 0.5 s while
    # mediolateral sway (stride period 1.0 s) flips sign at t + 0.5 s.
    k0 = gait.sample(0.1, st)
    k_step = gait.sample(0.6, st)
    assert k_step.lin_acc[2] == pytest.approx(k0.lin_acc[2], abs=1e-9)
    assert k_step.lin_acc[1] == pytest.approx(-k0.lin_acc[1], abs=1e-9)

    eps = 1e-6
    km = gait.sample(0.23 - eps, st)
    kp = gait.sample(0.23 + eps, st)
    k_mid = gait.sample(0.23, st)
    fd_alpha = (kp.ang_vel - km.ang_vel) / (2.0 * eps)
    np.testing.assert_allclose(k_mid.ang_acc, fd_alpha, atol=1e-6)


def test_load_imu_base_profile_csv_npz_and_wrapper(tmp_path: Path) -> None:
    csv_path = tmp_path / "imu_trace.csv"
    # Sensor pitched 90 deg nose-up? Use identity quat (w=1) so g_body = [0, 0, -9.81].
    # Specific force a_meas = a_kin - g_body = [1.5, 0.0, +9.81] for level [1.5, 0, 0] acc.
    csv_lines = [
        "time,ax,ay,az,gx,gy,gz,qw,qx,qy,qz",
        "10.0,1.5,0.0,9.81,0.0,0.0,0.0,1.0,0.0,0.0,0.0",
        "10.1,1.5,0.0,9.81,0.0,0.0,18.0,1.0,0.0,0.0,0.0",
        "10.2,1.5,0.0,9.81,0.0,0.0,36.0,1.0,0.0,0.0,0.0",
    ]
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    prof_csv = load_imu_base_profile(
        csv_path,
        quat_cols=("qw", "qx", "qy", "qz"),
        gyro_in_degrees=True,
    )
    kin = prof_csv.sample(0.1, None)
    np.testing.assert_allclose(kin.lin_acc, [1.5, 0.0, 0.0], atol=1e-9)
    assert kin.ang_vel[2] == pytest.approx(math.radians(18.0), rel=1e-6)
    assert kin.ang_acc[2] == pytest.approx(math.radians(180.0), rel=1e-6)

    npz_path = tmp_path / "imu_trace.npz"
    np.savez(
        npz_path,
        time=np.array([0.0, 0.05, 0.10]),
        lin_acc=np.array([[0.0, 0.0, 9.81], [0.0, 0.0, 9.81], [2.0, 0.0, 9.81]]),
        ang_vel=np.zeros((3, 3)),
    )
    prof_npz = load_imu_base_profile(npz_path, subtract_initial_samples=2)
    np.testing.assert_allclose(prof_npz.sample(0.10, None).lin_acc, [2.0, 0.0, 0.0])

    combo = CompositeBaseProfile((prof_npz, GaitOscillationProfile(random_phase=True)))
    env = BaseExcitationWrapper(make_env("myoElbowPose1D6MFixed-v0"), profile=combo)
    try:
        obs, _ = env.reset(seed=5)
        assert obs.dtype == np.float32
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert "base_excitation" in info
        assert info["base_excitation"]["lin_acc"].shape == (3,)
    finally:
        env.close()
