# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Parameterized physical disturbance spectra and 6-axis IMU trace loaders."""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from myosuite.physics.base_excitation import BaseKinematics, RecordedBaseProfile


def _vec3_nonneg(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError(
            f"{name} must be a finite non-negative 3-vector, got {value!r}"
        )
    return arr


def _vec3_pos(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must be a finite positive 3-vector, got {value!r}")
    return arr


def _vec3_finite(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 3-vector, got {value!r}")
    return arr


@dataclass
class _SecondOrderTurbulenceState:
    dt: float
    last_step: int
    rng: np.random.Generator
    phi: np.ndarray
    chol: np.ndarray
    lin_state: np.ndarray
    ang_phi: np.ndarray
    ang_chol: np.ndarray
    ang_state: np.ndarray


def _discretize_critically_damped_sde(
    sigma: np.ndarray, tau: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact discrete transition and innovation Cholesky for ``(1 + tau*s)^2`` filters."""
    phi = np.zeros((3, 2, 2), dtype=np.float64)
    chol = np.zeros((3, 2, 2), dtype=np.float64)
    p_inf = np.zeros((3, 2, 2), dtype=np.float64)
    for i in range(3):
        ti = float(tau[i])
        si = float(sigma[i])
        r = dt / ti
        e = math.exp(-r)
        phi_i = e * np.array([[1.0 + r, dt], [-r / ti, 1.0 - r]], dtype=np.float64)
        phi[i] = phi_i
        if si > 0.0:
            p_i = np.array([[si * si, 0.0], [0.0, (si / ti) ** 2]], dtype=np.float64)
            p_inf[i] = p_i
            q_d = p_i - phi_i @ p_i @ phi_i.T
            q_d = 0.5 * (q_d + q_d.T)
            q_d[0, 0] = max(q_d[0, 0], 1e-18)
            q_d[1, 1] = max(q_d[1, 1], 1e-18)
            chol[i] = np.linalg.cholesky(q_d)
    return phi, chol, p_inf


@dataclass(frozen=True)
class DrydenTurbulenceProfile:
    """Continuous atmospheric turbulence forming-filter base excitation.

    Propagates exact discrete-time 2nd-order forming filters parameterized by
    airspeed ``airspeed_mps``, turbulence velocity intensities ``sigma_vel`` [m/s],
    spatial scale lengths ``scale_lengths_m`` [m], and optional rotational rate
    intensities ``sigma_ang_vel`` [rad/s].
    """

    airspeed_mps: float = 25.0
    sigma_vel: tuple[float, float, float] | np.ndarray = (1.5, 1.5, 1.2)
    scale_lengths_m: tuple[float, float, float] | np.ndarray = (200.0, 200.0, 100.0)
    sigma_ang_vel: tuple[float, float, float] | np.ndarray = (0.04, 0.04, 0.03)
    ang_scale_lengths_m: tuple[float, float, float] | np.ndarray = (80.0, 80.0, 80.0)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.airspeed_mps) and self.airspeed_mps > 0.0):
            raise ValueError("airspeed_mps must be finite and positive")
        object.__setattr__(self, "sigma_vel", _vec3_nonneg(self.sigma_vel, "sigma_vel"))
        object.__setattr__(
            self,
            "scale_lengths_m",
            _vec3_pos(self.scale_lengths_m, "scale_lengths_m"),
        )
        object.__setattr__(
            self, "sigma_ang_vel", _vec3_nonneg(self.sigma_ang_vel, "sigma_ang_vel")
        )
        object.__setattr__(
            self,
            "ang_scale_lengths_m",
            _vec3_pos(self.ang_scale_lengths_m, "ang_scale_lengths_m"),
        )

    def reset(self, rng: np.random.Generator, dt: float) -> _SecondOrderTurbulenceState:
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError("dt must be finite and positive")
        v = float(self.airspeed_mps)
        tau_lin = np.asarray(self.scale_lengths_m, dtype=np.float64) / v
        tau_ang = np.asarray(self.ang_scale_lengths_m, dtype=np.float64) / v

        phi, chol, p_inf = _discretize_critically_damped_sde(
            np.asarray(self.sigma_vel), tau_lin, dt
        )
        ang_phi, ang_chol, ang_p_inf = _discretize_critically_damped_sde(
            np.asarray(self.sigma_ang_vel), tau_ang, dt
        )

        lin_state = np.zeros((3, 2), dtype=np.float64)
        ang_state = np.zeros((3, 2), dtype=np.float64)
        for i in range(3):
            if p_inf[i, 0, 0] > 0.0:
                lin_state[i, 0] = rng.normal(0.0, math.sqrt(p_inf[i, 0, 0]))
                lin_state[i, 1] = rng.normal(0.0, math.sqrt(p_inf[i, 1, 1]))
            if ang_p_inf[i, 0, 0] > 0.0:
                ang_state[i, 0] = rng.normal(0.0, math.sqrt(ang_p_inf[i, 0, 0]))
                ang_state[i, 1] = rng.normal(0.0, math.sqrt(ang_p_inf[i, 1, 1]))

        return _SecondOrderTurbulenceState(
            dt=float(dt),
            last_step=0,
            rng=rng,
            phi=phi,
            chol=chol,
            lin_state=lin_state,
            ang_phi=ang_phi,
            ang_chol=ang_chol,
            ang_state=ang_state,
        )

    def sample(self, t: float, state: _SecondOrderTurbulenceState) -> BaseKinematics:
        target_step = max(0, int(round(float(t) / state.dt)))
        while state.last_step < target_step:
            z_lin = state.rng.standard_normal((3, 2))
            z_ang = state.rng.standard_normal((3, 2))
            for i in range(3):
                state.lin_state[i] = (
                    state.phi[i] @ state.lin_state[i] + state.chol[i] @ z_lin[i]
                )
                state.ang_state[i] = (
                    state.ang_phi[i] @ state.ang_state[i] + state.ang_chol[i] @ z_ang[i]
                )
            state.last_step += 1

        return BaseKinematics(
            lin_acc=state.lin_state[:, 1].copy(),
            ang_vel=state.ang_state[:, 0].copy(),
            ang_acc=state.ang_state[:, 1].copy(),
        )


@dataclass(frozen=True)
class DiscreteGustProfile:
    """Classical ``1 - cos`` discrete gust velocity and angular-rate excursion."""

    gust_vel: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 3.0)
    gradient_distance_m: float = 30.0
    airspeed_mps: float = 20.0
    onset_time: float = 0.5
    gust_ang_vel: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0)
    repeat_period: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "gust_vel", _vec3_finite(self.gust_vel, "gust_vel"))
        object.__setattr__(
            self, "gust_ang_vel", _vec3_finite(self.gust_ang_vel, "gust_ang_vel")
        )
        if not (
            math.isfinite(self.gradient_distance_m) and self.gradient_distance_m > 0.0
        ):
            raise ValueError("gradient_distance_m must be finite and positive")
        if not (math.isfinite(self.airspeed_mps) and self.airspeed_mps > 0.0):
            raise ValueError("airspeed_mps must be finite and positive")
        if not math.isfinite(self.onset_time) or self.onset_time < 0.0:
            raise ValueError("onset_time must be finite and non-negative")
        half_dur = self.gradient_distance_m / self.airspeed_mps
        if self.repeat_period is not None and (
            not math.isfinite(self.repeat_period) or self.repeat_period < 2.0 * half_dur
        ):
            raise ValueError("repeat_period must be >= full 2*H/V gust duration")

    def reset(self, rng: np.random.Generator, dt: float) -> None:
        del rng, dt
        return None

    def sample(self, t: float, state: None) -> BaseKinematics:
        del state
        half_t = float(self.gradient_distance_m) / float(self.airspeed_mps)
        full_t = 2.0 * half_t
        rel = float(t) - float(self.onset_time)
        if rel < 0.0:
            return BaseKinematics(lin_acc=np.zeros(3, dtype=np.float64))
        if self.repeat_period is not None:
            rel = math.fmod(rel, float(self.repeat_period))
        if rel > full_t:
            return BaseKinematics(lin_acc=np.zeros(3, dtype=np.float64))

        phase = math.pi * rel / half_t
        v_max = np.asarray(self.gust_vel, dtype=np.float64)
        w_max = np.asarray(self.gust_ang_vel, dtype=np.float64)
        lin_acc = (0.5 * math.pi / half_t) * math.sin(phase) * v_max
        ang_vel = 0.5 * (1.0 - math.cos(phase)) * w_max
        ang_acc = (0.5 * math.pi / half_t) * math.sin(phase) * w_max
        return BaseKinematics(lin_acc=lin_acc, ang_vel=ang_vel, ang_acc=ang_acc)


@dataclass(frozen=True)
class RoadRoughnessProfile:
    """ISO 8608 spatial PSD road-surface roughness excitation at travel speed ``speed_mps``."""

    ISO_GD_N0: ClassVar[dict[str, float]] = {
        "A": 16e-6,
        "B": 64e-6,
        "C": 256e-6,
        "D": 1024e-6,
        "E": 4096e-6,
        "F": 16384e-6,
        "G": 65536e-6,
        "H": 262144e-6,
    }

    iso_class: str = "C"
    speed_mps: float = 15.0
    num_harmonics: int = 32
    n_min_cpm: float = 0.02
    n_max_cpm: float = 2.5
    suspension_freq_hz: float = 1.4
    suspension_damping: float = 0.65
    roll_coupling: float = 0.12

    def __post_init__(self) -> None:
        cls_key = str(self.iso_class).strip().upper()
        if cls_key not in self.ISO_GD_N0:
            raise ValueError(f"iso_class must be one of A..H, got {self.iso_class!r}")
        object.__setattr__(self, "iso_class", cls_key)
        if not (math.isfinite(self.speed_mps) and self.speed_mps > 0.0):
            raise ValueError("speed_mps must be finite and positive")
        if isinstance(self.num_harmonics, bool) or self.num_harmonics < 4:
            raise ValueError("num_harmonics must be >= 4")
        if not (0.0 < self.n_min_cpm < self.n_max_cpm):
            raise ValueError("Require 0 < n_min_cpm < n_max_cpm")
        if self.suspension_freq_hz <= 0.0 or self.suspension_damping <= 0.0:
            raise ValueError("suspension_freq_hz and suspension_damping must be > 0")
        if not math.isfinite(self.roll_coupling) or self.roll_coupling < 0.0:
            raise ValueError("roll_coupling must be finite and non-negative")

    def reset(self, rng: np.random.Generator, dt: float) -> dict[str, np.ndarray]:
        del dt
        n_edges = np.geomspace(self.n_min_cpm, self.n_max_cpm, self.num_harmonics + 1)
        n_mid = np.sqrt(n_edges[:-1] * n_edges[1:])
        dn = np.diff(n_edges)

        gd_n0 = self.ISO_GD_N0[self.iso_class]
        n0 = 0.1
        # ISO 8608 displacement PSD: Gd(n) = Gd(n0) * (n / n0)^(-2)
        gd = gd_n0 * (n_mid / n0) ** (-2.0)
        disp_amp = np.sqrt(2.0 * gd * dn)

        freqs_hz = n_mid * float(self.speed_mps)
        omega = 2.0 * np.pi * freqs_hz
        wn = 2.0 * math.pi * float(self.suspension_freq_hz)
        zeta = float(self.suspension_damping)
        r = omega / wn
        # Base-excited Kelvin-Voigt transmissibility |H(jw)|
        trans = np.sqrt(
            (1.0 + (2.0 * zeta * r) ** 2) / ((1.0 - r * r) ** 2 + (2.0 * zeta * r) ** 2)
        )
        vert_acc_amp = disp_amp * (omega**2) * trans

        phases_z = rng.uniform(0.0, 2.0 * math.pi, size=self.num_harmonics)
        phases_roll = rng.uniform(0.0, 2.0 * math.pi, size=self.num_harmonics)
        return {
            "omega": omega,
            "vert_acc_amp": vert_acc_amp,
            "phases_z": phases_z,
            "phases_roll": phases_roll,
        }

    def sample(self, t: float, state: dict[str, np.ndarray]) -> BaseKinematics:
        omega = state["omega"]
        amp = state["vert_acc_amp"]
        arg_z = omega * float(t) + state["phases_z"]
        arg_r = omega * float(t) + state["phases_roll"]

        az = float(np.sum(amp * np.cos(arg_z)))
        roll_acc = float(self.roll_coupling * np.sum(amp * np.cos(arg_r)))
        roll_vel = float(
            self.roll_coupling * np.sum((amp / np.maximum(omega, 1e-8)) * np.sin(arg_r))
        )
        return BaseKinematics(
            lin_acc=np.array([0.0, 0.0, az], dtype=np.float64),
            ang_vel=np.array([roll_vel, 0.0, 0.0], dtype=np.float64),
            ang_acc=np.array([roll_acc, 0.0, 0.0], dtype=np.float64),
        )


@dataclass(frozen=True)
class BoundedJerkProfile:
    """Trapezoidal acceleration maneuver with bounded constant jerk [m/s^3]."""

    peak_lin_acc: tuple[float, float, float] | np.ndarray = (-2.5, 0.0, 0.0)
    jerk_limit: float = 5.0
    onset_time: float = 0.5
    hold_duration: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "peak_lin_acc", _vec3_finite(self.peak_lin_acc, "peak_lin_acc")
        )
        if not (math.isfinite(self.jerk_limit) and self.jerk_limit > 0.0):
            raise ValueError("jerk_limit must be finite and positive")
        if not math.isfinite(self.onset_time) or self.onset_time < 0.0:
            raise ValueError("onset_time must be finite and non-negative")
        if not math.isfinite(self.hold_duration) or self.hold_duration < 0.0:
            raise ValueError("hold_duration must be finite and non-negative")

    def reset(self, rng: np.random.Generator, dt: float) -> None:
        del rng, dt
        return None

    def sample(self, t: float, state: None) -> BaseKinematics:
        del state
        peak = np.asarray(self.peak_lin_acc, dtype=np.float64)
        mag = float(np.linalg.norm(peak))
        if mag <= 1e-12:
            return BaseKinematics(lin_acc=np.zeros(3, dtype=np.float64))

        ramp_t = mag / float(self.jerk_limit)
        rel = float(t) - float(self.onset_time)
        total_t = 2.0 * ramp_t + float(self.hold_duration)
        if rel <= 0.0 or rel >= total_t:
            scale = 0.0
        elif rel < ramp_t:
            scale = rel / ramp_t
        elif rel <= ramp_t + float(self.hold_duration):
            scale = 1.0
        else:
            scale = (total_t - rel) / ramp_t
        return BaseKinematics(lin_acc=scale * peak)


@dataclass(frozen=True)
class GaitOscillationProfile:
    """Harmonic bipedal trunk/pelvis oscillation parameterized by step frequency."""

    step_freq_hz: float = 1.8
    vert_acc_amp: float = 1.6
    lat_acc_amp: float = 0.7
    ap_acc_amp: float = 0.8
    roll_vel_amp: float = 0.06
    pitch_vel_amp: float = 0.05
    random_phase: bool = False

    def __post_init__(self) -> None:
        if not (math.isfinite(self.step_freq_hz) and self.step_freq_hz > 0.0):
            raise ValueError("step_freq_hz must be finite and positive")
        for label in (
            "vert_acc_amp",
            "lat_acc_amp",
            "ap_acc_amp",
            "roll_vel_amp",
            "pitch_vel_amp",
        ):
            val = float(getattr(self, label))
            if not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{label} must be finite and non-negative")

    def reset(self, rng: np.random.Generator, dt: float) -> float:
        del dt
        return float(rng.uniform(0.0, 2.0 * math.pi)) if self.random_phase else 0.0

    def sample(self, t: float, state: float) -> BaseKinematics:
        phi0 = float(state)
        w_step = 2.0 * math.pi * float(self.step_freq_hz)
        w_stride = 0.5 * w_step

        ax = float(self.ap_acc_amp) * math.sin(w_step * float(t) + phi0)
        ay = float(self.lat_acc_amp) * math.sin(w_stride * float(t) + 0.5 * phi0)
        az = float(self.vert_acc_amp) * math.cos(w_step * float(t) + phi0)

        wx = float(self.roll_vel_amp) * math.cos(w_stride * float(t) + 0.5 * phi0)
        wy = float(self.pitch_vel_amp) * math.sin(w_step * float(t) + phi0)
        alphax = (
            -w_stride
            * float(self.roll_vel_amp)
            * math.sin(w_stride * float(t) + 0.5 * phi0)
        )
        alphay = w_step * float(self.pitch_vel_amp) * math.cos(w_step * float(t) + phi0)

        return BaseKinematics(
            lin_acc=np.array([ax, ay, az], dtype=np.float64),
            ang_vel=np.array([wx, wy, 0.0], dtype=np.float64),
            ang_acc=np.array([alphax, alphay, 0.0], dtype=np.float64),
        )


def _quat_rotate_inverse(quats_wxyz: np.ndarray, world_vec: np.ndarray) -> np.ndarray:
    q = np.asarray(quats_wxyz, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("Quaternion entries must be non-zero")
    q = q / norm
    w = q[:, 0:1]
    xyz = -q[:, 1:4]
    v = np.broadcast_to(np.asarray(world_vec, dtype=np.float64), (q.shape[0], 3))
    uv = np.cross(xyz, v)
    uuv = np.cross(xyz, uv)
    return v + 2.0 * (w * uv + uuv)


def load_imu_base_profile(
    path: str | Path,
    *,
    time_col: str = "time",
    acc_cols: tuple[str, str, str] = ("ax", "ay", "az"),
    gyro_cols: tuple[str, str, str] = ("gx", "gy", "gz"),
    quat_cols: tuple[str, str, str, str] | None = None,
    acc_scale: float = 1.0,
    gyro_in_degrees: bool = False,
    world_gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    subtract_initial_samples: int = 0,
    loop: bool = False,
) -> RecordedBaseProfile:
    """Load a 6-axis IMU recording (``.npz`` or ``.csv``) into a :class:`RecordedBaseProfile`."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"IMU trace file not found: {file_path}")
    if not (math.isfinite(acc_scale) and acc_scale > 0.0):
        raise ValueError("acc_scale must be finite and positive")
    if subtract_initial_samples < 0:
        raise ValueError("subtract_initial_samples must be >= 0")

    suffix = file_path.suffix.lower()
    if suffix == ".npz":
        with np.load(file_path, allow_pickle=False) as archive:
            times = np.asarray(archive[time_col], dtype=np.float64)
            if "lin_acc" in archive and all(c not in archive for c in acc_cols):
                lin_acc = np.asarray(archive["lin_acc"], dtype=np.float64)
            else:
                lin_acc = np.column_stack(
                    [np.asarray(archive[c], dtype=np.float64) for c in acc_cols]
                )
            if "ang_vel" in archive and all(c not in archive for c in gyro_cols):
                ang_vel = np.asarray(archive["ang_vel"], dtype=np.float64)
            else:
                ang_vel = np.column_stack(
                    [np.asarray(archive[c], dtype=np.float64) for c in gyro_cols]
                )
            quats = (
                np.column_stack(
                    [np.asarray(archive[c], dtype=np.float64) for c in quat_cols]
                )
                if quat_cols is not None
                else None
            )
    elif suffix == ".csv":
        with file_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        if len(rows) < 2:
            raise ValueError("CSV IMU recording must contain at least 2 rows")
        times = np.array([float(r[time_col]) for r in rows], dtype=np.float64)
        lin_acc = np.array(
            [[float(r[c]) for c in acc_cols] for r in rows], dtype=np.float64
        )
        ang_vel = np.array(
            [[float(r[c]) for c in gyro_cols] for r in rows], dtype=np.float64
        )
        quats = (
            np.array([[float(r[c]) for c in quat_cols] for r in rows], dtype=np.float64)
            if quat_cols is not None
            else None
        )
    else:
        raise ValueError(
            f"Unsupported IMU file extension {suffix!r}; expected '.npz' or '.csv'."
        )

    times = times - float(times[0])
    lin_acc = lin_acc * float(acc_scale)
    if gyro_in_degrees:
        ang_vel = np.radians(ang_vel)

    if quats is not None:
        # Accelerometer specific force measures a_kin - g_body => a_kin = a_meas + g_body.
        g_body = _quat_rotate_inverse(
            quats, np.asarray(world_gravity, dtype=np.float64)
        )
        lin_acc = lin_acc + g_body
    elif subtract_initial_samples > 0:
        n_bias = min(int(subtract_initial_samples), lin_acc.shape[0])
        lin_acc = lin_acc - np.mean(lin_acc[:n_bias], axis=0, keepdims=True)

    ang_acc = np.asarray(np.gradient(ang_vel, times, axis=0), dtype=np.float64)
    return RecordedBaseProfile(
        times=times,
        lin_acc=lin_acc,
        ang_vel=ang_vel,
        ang_acc=ang_acc,
        loop=loop,
    )


__all__ = [
    "BoundedJerkProfile",
    "DiscreteGustProfile",
    "DrydenTurbulenceProfile",
    "GaitOscillationProfile",
    "RoadRoughnessProfile",
    "load_imu_base_profile",
]
