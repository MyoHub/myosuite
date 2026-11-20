# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""EMG signal processing, muscle activation dynamics and tendon-node projection."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType

import mujoco
import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline

_MJ_MINVAL = 1e-15


@dataclass(frozen=True)
class EMGFilterConfig:
    """Filtering and normalization parameters for raw EMG signals."""

    bandpass_hz: tuple[float, float] | None = (20.0, 450.0)
    notch_hz: float | None = None
    notch_q: float = 30.0
    lowpass_hz: float | None = 6.0
    order: int = 2
    normalize: str = "peak"


def process_emg(
    signals: np.ndarray,
    fs: float,
    config: EMGFilterConfig | None = None,
    *,
    mvc_peaks: np.ndarray | None = None,
) -> np.ndarray:
    """Convert raw EMG ``(T,)`` or ``(T, C)`` into non-negative linear envelopes.

    Args:
        signals: Raw EMG samples along axis 0.
        fs: Sampling frequency [Hz].
        config: Filtering and normalization settings.
        mvc_peaks: Per-channel MVC peaks required when ``config.normalize == "mvc"``.

    Returns:
        Processed EMG envelope with the same shape as *signals*.
    """
    cfg = config or EMGFilterConfig()
    raw = np.asarray(signals, dtype=np.float64)
    squeeze = raw.ndim == 1
    if squeeze:
        raw = raw[:, None]
    if raw.ndim != 2 or len(raw) < 8:
        raise ValueError(
            f"signals must have shape (T >= 8,) or (T >= 8, C), got {signals.shape}"
        )
    if not np.isfinite(raw).all():
        raise ValueError("signals must be finite")
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError(f"fs must be finite and positive, got {fs}")
    if cfg.normalize not in ("none", "peak", "mvc"):
        raise ValueError(
            f"normalize must be 'none', 'peak' or 'mvc', got {cfg.normalize!r}"
        )

    nyquist = 0.5 * fs
    x = raw - raw.mean(axis=0, keepdims=True)

    if cfg.notch_hz is not None:
        if not 0 < cfg.notch_hz < nyquist:
            raise ValueError(
                f"notch_hz must lie in (0, {nyquist:g}), got {cfg.notch_hz}"
            )
        b, a = signal.iirnotch(cfg.notch_hz, cfg.notch_q, fs=fs)
        pad = min(len(x) - 1, max(1, int(np.ceil(cfg.notch_q * fs / cfg.notch_hz))))
        x = signal.filtfilt(b, a, x, axis=0, padlen=pad)

    if cfg.bandpass_hz is not None:
        lo, hi = cfg.bandpass_hz
        hi_eff = min(hi, 0.95 * nyquist)
        if not (0 < lo < hi_eff < nyquist):
            raise ValueError(
                f"bandpass_hz {cfg.bandpass_hz} incompatible with Nyquist {nyquist:g} Hz"
            )
        sos = signal.butter(
            cfg.order, [lo, hi_eff], btype="bandpass", fs=fs, output="sos"
        )
        pad = min(len(x) - 1, int(np.ceil(2.0 * fs / lo)))
        x = signal.sosfiltfilt(sos, x, axis=0, padlen=pad)

    x = np.abs(x)

    if cfg.lowpass_hz is not None:
        if not 0 < cfg.lowpass_hz < nyquist:
            raise ValueError(
                f"lowpass_hz must lie in (0, {nyquist:g}), got {cfg.lowpass_hz}"
            )
        sos_lp = signal.butter(
            cfg.order, cfg.lowpass_hz, btype="lowpass", fs=fs, output="sos"
        )
        pad = min(len(x) - 1, int(np.ceil(2.0 * fs / cfg.lowpass_hz)))
        x = np.maximum(signal.sosfiltfilt(sos_lp, x, axis=0, padlen=pad), 0.0)

    if cfg.normalize == "peak":
        x = x / np.maximum(x.max(axis=0, keepdims=True), 1e-12)
    elif cfg.normalize == "mvc":
        if mvc_peaks is None:
            raise ValueError("mvc_peaks is required when normalize='mvc'")
        denom = np.broadcast_to(
            np.asarray(mvc_peaks, dtype=np.float64), (1, x.shape[1])
        )
        if np.any(denom <= 0):
            raise ValueError("mvc_peaks must be strictly positive")
        x = x / denom

    return x[:, 0] if squeeze else x


def _hill_library() -> ModuleType | None:
    try:
        return importlib.import_module("myo_sim.physics.hill_muscle")
    except ModuleNotFoundError as exc:
        if exc.name != "myo_sim.physics.hill_muscle":
            raise
        return None


def _resolve_tau(
    width: int,
    tau_act: float | np.ndarray,
    tau_deact: float | np.ndarray,
    tausmooth: float | np.ndarray,
    model: mujoco.MjModel | None,
    actuator_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if model is not None:
        ids = (
            np.arange(model.nu)
            if actuator_ids is None
            else np.asarray(actuator_ids, dtype=int)
        )
        if len(ids) != width:
            raise ValueError(
                f"expected {len(ids)} columns for selected actuators, got {width}"
            )
        ta = np.full(width, float(np.mean(tau_act)), dtype=np.float64)
        td = np.full(width, float(np.mean(tau_deact)), dtype=np.float64)
        ts = np.full(width, float(np.mean(tausmooth)), dtype=np.float64)
        muscle = model.actuator_dyntype[ids] == mujoco.mjtDyn.mjDYN_MUSCLE
        ta[muscle] = model.actuator_dynprm[ids[muscle], 0]
        td[muscle] = model.actuator_dynprm[ids[muscle], 1]
        ts[muscle] = model.actuator_dynprm[ids[muscle], 2]
        return np.maximum(ta, 1e-4), np.maximum(td, 1e-4), np.maximum(ts, 0.0)
    ta = np.broadcast_to(np.asarray(tau_act, dtype=np.float64), (width,)).copy()
    td = np.broadcast_to(np.asarray(tau_deact, dtype=np.float64), (width,)).copy()
    ts = np.broadcast_to(np.asarray(tausmooth, dtype=np.float64), (width,)).copy()
    if np.any(ta <= 0) or np.any(td <= 0):
        raise ValueError("tau_act and tau_deact must be positive")
    if np.any(ts < 0):
        raise ValueError("tausmooth must be non-negative")
    return ta, td, ts


def _muscle_timescale(
    dctrl: np.ndarray,
    act: np.ndarray,
    ta: np.ndarray,
    td: np.ndarray,
    ts: np.ndarray,
) -> np.ndarray:
    """Effective time constant matching MuJoCo's ``mju_muscleDynamicsTimescale``."""
    a_clamp = np.clip(act, 0.0, 1.0)
    scale = 0.5 + 1.5 * a_clamp
    tau_a = ta * scale
    tau_d = td / scale
    hard = ts < _MJ_MINVAL
    x = np.clip(dctrl / np.maximum(_MJ_MINVAL, ts) + 0.5, 0.0, 1.0)
    sig = x * x * x * (3.0 * x * (2.0 * x - 5.0) + 10.0)
    return np.where(
        hard,
        np.where(dctrl > 0.0, tau_a, tau_d),
        tau_d + (tau_a - tau_d) * sig,
    )


def excitation_to_activation(
    excitation: np.ndarray,
    dt: float,
    *,
    tau_act: float | np.ndarray = 0.01,
    tau_deact: float | np.ndarray = 0.04,
    tausmooth: float | np.ndarray = 0.0,
    initial: np.ndarray | None = None,
    model: mujoco.MjModel | None = None,
    actuator_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate neural excitation ``u(t)`` into ``mjDYN_MUSCLE`` activation ``a(t)``.

    Delegates per-step integration to ``myo_sim.physics.hill_muscle`` when installed,
    including smooth ``dynprm[2]`` (``tausmooth``) transitions.

    Args:
        excitation: ``(T,)`` or ``(T, M)`` excitation trajectory in ``[0, 1]``.
        dt: Sample period [s].
        tau_act: Activation time constant [s] when *model* is ``None``.
        tau_deact: Deactivation time constant [s] when *model* is ``None``.
        tausmooth: Smooth transition width ``dynprm[2]`` when *model* is ``None``.
        initial: Optional ``(M,)`` initial activation at ``t = 0``.
        model: Optional compiled model supplying ``actuator_dynprm[:, :3]``.
        actuator_ids: Optional subset of actuator indices matching *excitation* columns.

    Returns:
        Activation trajectory with the same shape as *excitation*.
    """
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError(f"dt must be finite and positive, got {dt}")
    u = np.asarray(excitation, dtype=np.float64)
    squeeze = u.ndim == 1
    if squeeze:
        u = u[:, None]
    if u.ndim != 2 or len(u) == 0:
        raise ValueError(
            f"excitation must be 1D or 2D and non-empty, got {excitation.shape}"
        )

    u_clip = np.clip(u, 0.0, 1.0)
    width = u_clip.shape[1]
    ta, td, ts = _resolve_tau(width, tau_act, tau_deact, tausmooth, model, actuator_ids)
    substeps = max(1, int(np.ceil(dt / (0.25 * float(ta.min())))))
    h = dt / substeps

    hill = _hill_library()
    params = None
    if hill is not None:
        params = hill.MuscleParams(
            tau_act=ta,
            tau_deact=td,
            tausmooth=ts,
            act_limited=np.ones(width, dtype=bool),
            act_range=np.tile(np.array([[0.0, 1.0]], dtype=np.float64), (width, 1)),
            length_range=np.zeros((width, 2), dtype=np.float64),
            acc0=np.ones(width, dtype=np.float64),
            gain_prm=np.zeros((width, 9), dtype=np.float64),
            bias_prm=np.zeros((width, 9), dtype=np.float64),
        )

    out = np.empty_like(u_clip)
    a = (
        u_clip[0].copy()
        if initial is None
        else np.clip(
            np.broadcast_to(np.asarray(initial, dtype=np.float64), (width,)),
            0.0,
            1.0,
        ).copy()
    )
    out[0] = a
    for t in range(1, len(u_clip)):
        u0, u1 = u_clip[t - 1], u_clip[t]
        for s in range(substeps):
            us = u0 + ((s + 0.5) / substeps) * (u1 - u0)
            if hill is not None and params is not None:
                a, _ = hill.step_muscle_activation(a, us, params, h, substeps=1)
            else:
                dctrl = us - a
                tau = _muscle_timescale(dctrl, a, ta, td, ts)
                a = np.clip(a + h * dctrl / np.maximum(_MJ_MINVAL, tau), 0.0, 1.0)
        out[t] = a
    return out[:, 0] if squeeze else out


def activation_to_excitation(
    activation: np.ndarray,
    dt: float,
    *,
    tau_act: float | np.ndarray = 0.01,
    tau_deact: float | np.ndarray = 0.04,
    tausmooth: float | np.ndarray = 0.0,
    model: mujoco.MjModel | None = None,
    actuator_ids: np.ndarray | None = None,
    clip_bounds: tuple[float, float] | None = (0.0, 1.0),
) -> np.ndarray:
    """Invert muscle activation ``a(t)`` into neural excitation ``u(t) = a + tau * da/dt``.

    Args:
        activation: ``(T,)`` or ``(T, M)`` activation trajectory, ``T >= 2``.
        dt: Sample period [s].
        tau_act: Activation time constant [s] when *model* is ``None``.
        tau_deact: Deactivation time constant [s] when *model* is ``None``.
        tausmooth: Smooth transition width ``dynprm[2]`` when *model* is ``None``.
        model: Optional compiled model supplying ``actuator_dynprm[:, :3]``.
        actuator_ids: Optional subset of actuator indices matching *activation* columns.
        clip_bounds: Optional ``(lo, hi)`` bounds applied to recovered excitations.

    Returns:
        Excitation trajectory with the same shape as *activation*.
    """
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError(f"dt must be finite and positive, got {dt}")
    a = np.asarray(activation, dtype=np.float64)
    squeeze = a.ndim == 1
    if squeeze:
        a = a[:, None]
    if a.ndim != 2 or len(a) < 2:
        raise ValueError(
            f"activation must have shape (T >= 2, ...), got {activation.shape}"
        )

    ta, td, ts = _resolve_tau(
        a.shape[1], tau_act, tau_deact, tausmooth, model, actuator_ids
    )
    a_clip = np.clip(a, 0.0, 1.0)
    da_dt = np.gradient(a_clip, dt, axis=0)
    scale = 0.5 + 1.5 * a_clip
    tau_a = ta[None, :] * scale
    tau_d = td[None, :] / scale

    dctrl = np.where(da_dt > 0.0, da_dt * tau_a, da_dt * tau_d)
    smooth_cols = ts >= _MJ_MINVAL
    if np.any(smooth_cols):
        ta_b = ta[None, smooth_cols]
        td_b = td[None, smooth_cols]
        ts_b = ts[None, smooth_cols]
        a_b = a_clip[:, smooth_cols]
        da_b = da_dt[:, smooth_cols]
        c1 = da_b * tau_a[:, smooth_cols]
        c2 = da_b * tau_d[:, smooth_cols]
        lo = np.minimum(c1, c2)
        hi = np.maximum(c1, c2)
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            f_mid = mid - da_b * _muscle_timescale(mid, a_b, ta_b, td_b, ts_b)
            lo = np.where(f_mid <= 0.0, mid, lo)
            hi = np.where(f_mid > 0.0, mid, hi)
        dctrl[:, smooth_cols] = 0.5 * (lo + hi)

    u = a_clip + dctrl
    if clip_bounds is not None:
        u = np.clip(u, clip_bounds[0], clip_bounds[1])
    return u[:, 0] if squeeze else u


def shift_signal(signals: np.ndarray, dt: float, lag_s: float) -> np.ndarray:
    """Shift *signals* forward in time by *lag_s* seconds with clamped boundaries."""
    arr = np.asarray(signals, dtype=np.float64)
    if lag_s == 0.0 or len(arr) < 2:
        return arr.copy()
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[:, None]
    idx = np.clip(
        np.arange(len(arr), dtype=np.float64) - lag_s / dt, 0.0, len(arr) - 1.0
    )
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, len(arr) - 1)
    frac = (idx - i0)[:, None]
    out = (1.0 - frac) * arr[i0] + frac * arr[i1]
    return out[:, 0] if squeeze else out


def phase_normalize(
    signals: np.ndarray,
    time: np.ndarray,
    events: np.ndarray,
    *,
    points: int = 101,
) -> np.ndarray:
    """Resample ``(K, 2)`` ``[t_start, t_end]`` cycles onto ``0..100 %`` phase."""
    arr = np.asarray(signals, dtype=np.float64)
    t = np.asarray(time, dtype=np.float64)
    ev = np.asarray(events, dtype=np.float64)
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[:, None]
    if ev.ndim != 2 or ev.shape[1] != 2 or len(ev) == 0:
        raise ValueError(f"events must have shape (K >= 1, 2), got {ev.shape}")
    if points < 2:
        raise ValueError(f"points must be >= 2, got {points}")
    if np.any(ev[:, 1] <= ev[:, 0]):
        raise ValueError("every cycle in events must have t_end > t_start")

    spline = CubicSpline(t, arr, axis=0)
    out = np.empty((len(ev), points, arr.shape[1]), dtype=np.float64)
    for k, (t0, t1) in enumerate(ev):
        out[k] = spline(np.linspace(t0, t1, points))
    return out[:, :, 0] if squeeze else out


def project_tendon_nodes(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """Place ``mjJNT_SLIDE`` joints tied by ``mjEQ_TENDON`` equalities onto their constraints.

    Supports models with moving tendon-node slides (e.g. lumbrical origins on FDP and
    juncturae tendinum). Returns a copy of *qpos* unchanged when no slide-tendon ties exist.

    Args:
        model: Compiled MuJoCo model.
        qpos: ``(nq,)`` or ``(T, nq)`` generalized positions.

    Returns:
        Copy of *qpos* with tied slide coordinates projected onto their tendon polynomials.
    """
    out = np.array(qpos, dtype=np.float64, copy=True)
    ties: list[tuple[int, float, int, np.ndarray]] = []
    for e in range(model.neq):
        if model.eq_type[e] != mujoco.mjtEq.mjEQ_TENDON or not model.eq_active0[e]:
            continue
        t1, t2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        if t1 < 0 or t2 < 0:
            continue
        adr = int(model.tendon_adr[t1])
        if (
            model.tendon_num[t1] != 1
            or model.wrap_type[adr] != mujoco.mjtWrap.mjWRAP_JOINT
        ):
            continue
        joint = int(model.wrap_objid[adr])
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_SLIDE:
            continue
        ties.append(
            (
                int(model.jnt_qposadr[joint]),
                float(model.wrap_prm[adr]),
                t2,
                model.eq_data[e, 4::-1].copy(),
            )
        )
    if not ties:
        return out

    squeeze = out.ndim == 1
    rows = out[None, :] if squeeze else out
    data = mujoco.MjData(model)
    for t in range(len(rows)):
        data.qpos[:] = rows[t]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_tendon(model, data)
        for qadr, prm, t2, poly in ties:
            dl = float(data.ten_length[t2] - model.tendon_length0[t2])
            rows[t, qadr] = model.qpos0[qadr] + float(np.polyval(poly, dl)) / prm
    return rows[0] if squeeze else out


__all__ = [
    "EMGFilterConfig",
    "activation_to_excitation",
    "excitation_to_activation",
    "phase_normalize",
    "process_emg",
    "project_tendon_nodes",
    "shift_signal",
]
