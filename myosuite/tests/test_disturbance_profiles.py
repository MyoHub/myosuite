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
from scipy.signal import welch

from myosuite import make_env
from myosuite.envs.wrappers import BaseExcitationWrapper
from myosuite.integrations.capture.imu import load_imu_base_profile
from myosuite.physics import (
    BaseExcitationProfile,
    BoundedJerkProfile,
    CompositeBaseProfile,
    DiscreteGustProfile,
    DrydenTurbulenceProfile,
    GaitOscillationProfile,
    RoadRoughnessProfile,
)

pytestmark = pytest.mark.tier1


def test_dryden_turbulence_matches_mil_f_8785c_velocity_and_acceleration_psd() -> None:
    prof = DrydenTurbulenceProfile(
        airspeed_mps=20.0,
        sigma_vel=(2.0, 2.0, 1.5),
        scale_lengths_m=(10.0, 10.0, 8.0),
        accel_lag_s=0.05,
        sigma_ang_vel=(0.05, 0.0, 0.02),
        ang_scale_lengths_m=(20.0, 20.0, 20.0),
    )
    assert isinstance(prof, BaseExcitationProfile)

    dt = 0.01
    n_steps = 40000
    s1 = prof.reset(np.random.default_rng(42), dt)
    s2 = prof.reset(np.random.default_rng(42), dt)

    vel1 = np.empty((n_steps, 3), dtype=np.float64)
    acc1 = np.empty((n_steps, 3), dtype=np.float64)
    for k in range(n_steps):
        acc1[k] = prof.sample(k * dt, s1).lin_acc
        vel1[k] = s1.lin_vel
    acc2 = np.array([prof.sample(k * dt, s2).lin_acc for k in range(200)])
    np.testing.assert_allclose(acc1[:200], acc2)

    # Stationary velocity std matches sigma_vel = [2.0, 2.0, 1.5].
    np.testing.assert_allclose(np.std(vel1, axis=0), [2.0, 2.0, 1.5], rtol=0.08)

    # Welch PSD on velocity and acceleration matches analytical MIL-F-8785C spectra.
    freqs, psd_vel = welch(vel1, fs=1.0 / dt, nperseg=2048, axis=0)
    _, psd_acc = welch(acc1, fs=1.0 / dt, nperseg=2048, axis=0)
    ref_vel = prof.velocity_psd(freqs)
    ref_acc = prof.acceleration_psd(freqs)

    # Compare log10 PSD over the core Dryden passband [0.15 Hz, 4.0 Hz] (~55 frequency bins).
    band = (freqs >= 0.15) & (freqs <= 4.0)
    log_err_vel = np.sqrt(
        np.mean((np.log10(psd_vel[band]) - np.log10(ref_vel[band])) ** 2, axis=0)
    )
    log_err_acc = np.sqrt(
        np.mean((np.log10(psd_acc[band]) - np.log10(ref_acc[band])) ** 2, axis=0)
    )
    assert np.all(
        log_err_vel < 0.09
    ), f"Velocity PSD log10 RMSE too high: {log_err_vel}"
    assert np.all(log_err_acc < 0.09), f"Accel PSD log10 RMSE too high: {log_err_acc}"

    # MIL-F-8785C lateral vs longitudinal shape check at equal (sigma=2, tau=0.5 s):
    # At low freq (f -> 0), G_u(0) = 4*sigma^2*tau = 2 * G_v(0); above corner (f ~ 1 Hz),
    # lateral (1 + 3*(w*tau)^2)/(1 + (w*tau)^2)^2 exceeds longitudinal 1/(1 + (w*tau)^2).
    low_band = (freqs > 0.03) & (freqs < 0.10)
    high_band = (freqs >= 0.8) & (freqs <= 2.5)
    assert float(np.mean(psd_vel[low_band, 0])) > 1.4 * float(
        np.mean(psd_vel[low_band, 1])
    )
    assert float(np.mean(psd_vel[high_band, 1])) > 1.2 * float(
        np.mean(psd_vel[high_band, 0])
    )


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


def test_load_imu_base_profile_csv_npz_conventions_and_wrapper(tmp_path: Path) -> None:
    csv_path = tmp_path / "imu_trace.csv"
    # Support pitched +90 deg about +Y: q_wxyz = [cos(45), 0, sin(45), 0].
    # World +Z maps to support -X (R^T @ [0, 0, -9.81] = [+9.81, 0, 0]), so at rest
    # specific force a_meas = -g_body = [-9.81, 0, 0] -> kinematic lin_acc = [0, 0, 0].
    s = math.sqrt(0.5)
    csv_lines = [
        "time,ax,ay,az,gx,gy,gz,qw,qx,qy,qz",
        f"10.0,-8.31,0.0,0.0,0.0,0.0,0.0,{s},0.0,{s},0.0",
        f"10.1,-8.31,0.0,0.0,0.0,0.0,18.0,{s},0.0,{s},0.0",
        f"10.2,-8.31,0.0,0.0,0.0,0.0,36.0,{s},0.0,{s},0.0",
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

    # Non-monotonic timestamps are rejected explicitly.
    bad_npz = tmp_path / "bad_time.npz"
    np.savez(
        bad_npz,
        time=np.array([0.0, 0.10, 0.10]),
        lin_acc=np.zeros((3, 3)),
        ang_vel=np.zeros((3, 3)),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        load_imu_base_profile(bad_npz)

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
