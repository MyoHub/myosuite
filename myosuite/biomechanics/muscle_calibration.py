# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Multi-trial EMG neuromusculoskeletal parameter calibration solver.

Implements CEINMS-style subject muscle calibration (Lloyd & Besier, *J. Biomech.*,
2003; Pizzolato et al., *J. Biomech.*, 2015): calibrates muscle group peak isometric
strength scales, tendon-slack / operating-range shifts and fiber-span scales, nonlinear
EMG-to-excitation shape factor ``A``, and electromechanical delay across one or more
recorded trials. Source C3D adaptation and hybrid static-optimization pipeline
orchestration live in :mod:`myosuite.integrations.capture.emg_calibration`.
"""

from __future__ import annotations

import copy
import importlib
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import mujoco
import numpy as np
from scipy.optimize import least_squares

from myosuite.biomechanics.emg import (
    EMGFilterConfig,
    excitation_to_activation,
    process_emg,
    shift_signal,
)
from myosuite.biomechanics.inverse_dynamics import clip_kinematics, required_qfrc
from myosuite.core.muscle_conditions import _passive_peak_force, _peak_force

OnCalibrationFailure = Literal["raise", "warn", "ignore"]
_VALID_FAILURE_MODES: frozenset[str] = frozenset({"raise", "warn", "ignore"})


class CalibrationError(RuntimeError):
    """Raised when nonlinear muscle calibration fails to converge."""


@dataclass(frozen=True)
class CalibrationTrial:
    """Synchronized kinematics, external loads, and EMG signals for one trial."""

    time: np.ndarray
    qpos: np.ndarray
    emg_signals: np.ndarray
    channel_to_actuators: Mapping[int | str, str | Sequence[str]]
    qvel: np.ndarray | None = None
    qacc: np.ndarray | None = None
    external_qfrc: np.ndarray | None = None
    emg_fs: float | None = None
    is_raw_emg: bool = False
    mvc_peaks: np.ndarray | None = None
    channel_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class MuscleCalibrationConfig:
    """Bounds, regularization, and convergence policy for multi-trial muscle calibration.

    Attributes:
        strength_bounds: Lower and upper bounds on per-group peak isometric force scale.
        length_shift_bounds_m: Bounds on actuator operating-range midpoint shift [m].
        span_scale_bounds: Bounds on actuator operating-range span multiplier.
        shape_factor_bounds: Bounds on Lloyd & Besier (2003) shape factor ``A``.
        delay_bounds_s: Bounds on electromechanical delay [s].
        calibrate_length_params: Optimize ``length_shifts_m`` and ``span_scales`` when ``True``.
        reg_weight: L2 regularization weight toward nominal parameters.
        max_nfev: Maximum nonlinear least-squares residual evaluations.
        lowpass_hz: Optional kinematics Butterworth cutoff [Hz] when ``qvel``/``qacc`` are derived.
        emg_filter: Raw-EMG bandpass/envelope configuration.
        on_failure: Policy when ``scipy.optimize.least_squares`` does not converge
            (``opt.success`` is ``False``, e.g. ``max_nfev`` exhausted): ``"raise"``
            raises :class:`CalibrationError`, ``"warn"`` emits ``RuntimeWarning`` and
            returns ``converged=False``, and ``"ignore"`` returns ``converged=False``
            without warning.
    """

    strength_bounds: tuple[float, float] = (0.5, 2.0)
    length_shift_bounds_m: tuple[float, float] = (-0.03, 0.03)
    span_scale_bounds: tuple[float, float] = (0.80, 1.25)
    shape_factor_bounds: tuple[float, float] = (-2.99, -0.01)
    delay_bounds_s: tuple[float, float] = (0.0, 0.08)
    calibrate_length_params: bool = True
    reg_weight: float = 1e-2
    max_nfev: int = 250
    lowpass_hz: float | None = None
    emg_filter: EMGFilterConfig = field(default_factory=EMGFilterConfig)
    on_failure: OnCalibrationFailure = "raise"

    def __post_init__(self) -> None:
        if self.max_nfev < 1:
            raise ValueError(f"max_nfev must be >= 1, got {self.max_nfev}")
        if self.on_failure not in _VALID_FAILURE_MODES:
            raise ValueError(
                f"on_failure must be 'raise', 'warn', or 'ignore', got {self.on_failure!r}"
            )


@dataclass(frozen=True)
class MuscleCalibrationResult:
    """Calibrated MuJoCo model, physiological parameters, and solver diagnostics."""

    calibrated_model: mujoco.MjModel
    strength_scales: dict[str, float]
    length_shifts_m: dict[str, float]
    span_scales: dict[str, float]
    shape_factor_a: float
    delay_s: float
    calibrated_dofs: tuple[int, ...]
    initial_torque_rmse: float
    final_torque_rmse: float
    evaluations: int
    converged: bool = True
    status: int = 1
    message: str = ""
    optimality: float = 0.0
    cost: float = 0.0


def nonlinear_excitation_shape(u: np.ndarray, shape_a: float) -> np.ndarray:
    """Lloyd & Besier (2003) exponential EMG-to-excitation shaping ``(exp(A u)-1)/(exp(A)-1)``."""
    u_clip = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    if abs(shape_a) < 1e-5:
        return u_clip
    return (np.expm1(shape_a * u_clip)) / np.expm1(shape_a)


def _actuator_lookup(model: mujoco.MjModel) -> dict[str, int]:
    full = [model.actuator(i).name for i in range(model.nu)]
    counts = Counter(n.split("/")[-1] for n in full if n)
    out: dict[str, int] = {}
    for i, name in enumerate(full):
        if not name:
            continue
        out[name] = i
        leaf = name.split("/")[-1]
        if counts[leaf] == 1:
            out.setdefault(leaf, i)
    return out


def prepare_calibration_trial(
    model: mujoco.MjModel,
    trial: CalibrationTrial,
    emg_filter: EMGFilterConfig,
    lowpass_hz: float | None,
) -> tuple[
    np.ndarray,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[int, int],
]:
    """Validate and synchronize kinematics, external loads, and `[0, 1]` EMG envelopes."""
    t = np.asarray(trial.time, dtype=np.float64)
    if t.ndim != 1 or len(t) < 4:
        raise ValueError("trial.time must have shape (T >= 4,)")
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Invalid trial time step dt={dt}")

    qpos = np.asarray(trial.qpos, dtype=np.float64)
    if trial.qvel is None or trial.qacc is None:
        kin = clip_kinematics(model, qpos, dt, lowpass_hz=lowpass_hz)
        qpos, qvel, qacc = kin.qpos, kin.qvel, kin.qacc
    else:
        qvel = np.asarray(trial.qvel, dtype=np.float64)
        qacc = np.asarray(trial.qacc, dtype=np.float64)

    ext = (
        np.zeros((len(t), model.nv), dtype=np.float64)
        if trial.external_qfrc is None
        else np.asarray(trial.external_qfrc, dtype=np.float64)
    )

    emg_in = np.asarray(trial.emg_signals, dtype=np.float64)
    if emg_in.ndim == 1:
        emg_in = emg_in[:, None]
    if trial.is_raw_emg:
        fs = (
            float(trial.emg_fs)
            if trial.emg_fs is not None
            else (len(emg_in) - 1) / max(float(t[-1] - t[0]), 1e-9)
        )
        env_raw = process_emg(emg_in, fs, emg_filter, mvc_peaks=trial.mvc_peaks)
    else:
        env_raw = np.clip(emg_in, 0.0, None)

    if len(env_raw) != len(t):
        t_emg = np.linspace(float(t[0]), float(t[-1]), len(env_raw))
        env = np.column_stack(
            [np.interp(t, t_emg, env_raw[:, c]) for c in range(env_raw.shape[1])]
        )
    else:
        env = env_raw.copy()
    env = np.clip(env, 0.0, 1.0)

    act_lookup = _actuator_lookup(model)
    label_lookup = {lbl: i for i, lbl in enumerate(trial.channel_labels)}
    act_to_col: dict[int, int] = {}
    for ch_key, targets in trial.channel_to_actuators.items():
        col = int(ch_key) if isinstance(ch_key, int) else label_lookup[str(ch_key)]
        names = (targets,) if isinstance(targets, str) else tuple(targets)
        for nm in names:
            if nm not in act_lookup:
                raise KeyError(f"Unknown actuator {nm!r} in channel_to_actuators")
            act_to_col[act_lookup[nm]] = col
    return t, dt, qpos, qvel, qacc, ext, env, act_to_col


def apply_muscle_parameters(
    model: mujoco.MjModel,
    *,
    strength_scales: Mapping[int, float],
    length_shifts_m: Mapping[int, float],
    span_scales: Mapping[int, float],
) -> mujoco.MjModel:
    """Return a copy of *model* with calibrated peak forces and ``actuator_lengthrange``."""
    out = copy.deepcopy(model)
    base_f = _peak_force(out)
    has_pas, base_pas = _passive_peak_force(out)
    for act_id, gamma in strength_scales.items():
        out.actuator_gainprm[act_id, 2] = base_f[act_id] * float(gamma)
        if has_pas[act_id]:
            out.actuator_biasprm[act_id, 2] = base_pas[act_id] * float(gamma)

    for act_id in set(length_shifts_m) | set(span_scales):
        lr0, lr1 = out.actuator_lengthrange[act_id]
        mid = 0.5 * (lr0 + lr1)
        half = max(0.5 * (lr1 - lr0), 1e-5)
        shift = float(length_shifts_m.get(act_id, 0.0))
        span = float(span_scales.get(act_id, 1.0))
        out.actuator_lengthrange[act_id] = [
            mid + shift - span * half,
            mid + shift + span * half,
        ]
    return out


def compute_calibrated_emg_activations(
    model: mujoco.MjModel,
    env: np.ndarray,
    dt: float,
    act_to_col: Mapping[int, int],
    *,
    shape_factor_a: float,
    delay_s: float,
) -> np.ndarray:
    """Map ``(T, C)`` EMG envelopes to ``(T, nu)`` activations (``NaN`` where unmeasured)."""
    act_full = np.full((len(env), model.nu), np.nan, dtype=np.float64)
    if not act_to_col:
        return act_full
    act_ids = np.array(sorted(act_to_col), dtype=int)
    cols = [act_to_col[i] for i in act_ids]
    shifted = shift_signal(env[:, cols], dt, delay_s)
    shaped = nonlinear_excitation_shape(shifted, shape_factor_a)
    act_tracked = excitation_to_activation(
        shaped, dt, model=model, actuator_ids=act_ids
    )
    if act_tracked.ndim == 1:
        act_tracked = act_tracked[:, None]
    act_full[:, act_ids] = np.clip(act_tracked, 0.0, 1.0)
    return act_full


def calibrate_subject_muscles(
    model: mujoco.MjModel,
    trials: Sequence[CalibrationTrial],
    *,
    strength_groups: Sequence[Sequence[str]] | None = None,
    calibrated_dofs: Sequence[int] | None = None,
    config: MuscleCalibrationConfig | None = None,
) -> MuscleCalibrationResult:
    """Calibrate muscle strength, length-range, shape factor, and delay across *trials*."""
    if not trials:
        raise ValueError("trials must contain at least one CalibrationTrial")
    cfg = config or MuscleCalibrationConfig()
    prepared = [
        prepare_calibration_trial(model, tr, cfg.emg_filter, cfg.lowpass_hz)
        for tr in trials
    ]
    tracked_ids = sorted({act_id for *_, mapping in prepared for act_id in mapping})
    if not tracked_ids:
        raise ValueError("No EMG-mapped actuators found across trials")

    act_lookup = _actuator_lookup(model)
    if strength_groups is None:
        grp_ids = [(i,) for i in tracked_ids]
    else:
        grp_ids = [tuple(act_lookup[nm] for nm in g) for g in strength_groups]

    d_nom = mujoco.MjData(model)
    tau_trials: list[np.ndarray] = []
    spanned = np.zeros(model.nv, dtype=bool)
    untracked_spanned = np.zeros(model.nv, dtype=bool)
    untracked_ids = [i for i in range(model.nu) if i not in set(tracked_ids)]
    moment_buf = np.zeros((model.nu, model.nv))

    for _, _, qpos, qvel, qacc, ext, _, _ in prepared:
        tau_tr = np.empty((len(qpos), model.nv), dtype=np.float64)
        for k in range(len(qpos)):
            d_nom.qpos[:], d_nom.qvel[:] = qpos[k], qvel[k]
            mujoco.mj_fwdPosition(model, d_nom)
            mujoco.mj_fwdVelocity(model, d_nom)
            tau_tr[k] = required_qfrc(model, d_nom, qacc[k]) - ext[k]
            if k == 0:
                mujoco.mju_sparse2dense(
                    moment_buf,
                    d_nom.actuator_moment,
                    d_nom.moment_rownnz,
                    d_nom.moment_rowadr,
                    d_nom.moment_colind,
                )
                spanned |= np.any(np.abs(moment_buf[tracked_ids]) > 1e-5, axis=0)
                if untracked_ids:
                    untracked_spanned |= np.any(
                        np.abs(moment_buf[untracked_ids]) > 1e-5, axis=0
                    )
        tau_trials.append(tau_tr)

    if calibrated_dofs is not None:
        dofs = tuple(int(i) for i in calibrated_dofs)
    else:
        pure_dofs = np.flatnonzero(spanned & ~untracked_spanned)
        dofs = tuple(
            int(i) for i in (pure_dofs if len(pure_dofs) else np.flatnonzero(spanned))
        )
    if not dofs:
        raise ValueError("Tracked actuators have zero moment arm on all DOFs")
    dof_arr = np.array(dofs, dtype=int)

    n_grp = len(grp_ids)
    n_trk = len(tracked_ids)
    x0_parts = [np.ones(n_grp), np.array([-0.5, 0.015])]
    lo_parts = [
        np.full(n_grp, cfg.strength_bounds[0]),
        np.array([cfg.shape_factor_bounds[0], cfg.delay_bounds_s[0]]),
    ]
    hi_parts = [
        np.full(n_grp, cfg.strength_bounds[1]),
        np.array([cfg.shape_factor_bounds[1], cfg.delay_bounds_s[1]]),
    ]
    if cfg.calibrate_length_params:
        x0_parts.extend([np.zeros(n_trk), np.ones(n_trk)])
        lo_parts.extend(
            [
                np.full(n_trk, cfg.length_shift_bounds_m[0]),
                np.full(n_trk, cfg.span_scale_bounds[0]),
            ]
        )
        hi_parts.extend(
            [
                np.full(n_trk, cfg.length_shift_bounds_m[1]),
                np.full(n_trk, cfg.span_scale_bounds[1]),
            ]
        )
    x0 = np.concatenate(x0_parts)
    lo = np.concatenate(lo_parts)
    hi = np.concatenate(hi_parts)
    prior = x0.copy()

    def unpack(
        theta: np.ndarray,
    ) -> tuple[dict[int, float], dict[int, float], dict[int, float], float, float]:
        s_map = {
            act_id: float(theta[g_i]) for g_i, g in enumerate(grp_ids) for act_id in g
        }
        shape_a = float(theta[n_grp])
        delay_s = float(theta[n_grp + 1])
        if cfg.calibrate_length_params:
            sh_vec = theta[n_grp + 2 : n_grp + 2 + n_trk]
            sp_vec = theta[n_grp + 2 + n_trk :]
            l_map = {act_id: float(sh_vec[i]) for i, act_id in enumerate(tracked_ids)}
            sp_map = {act_id: float(sp_vec[i]) for i, act_id in enumerate(tracked_ids)}
        else:
            l_map, sp_map = {}, {}
        return s_map, l_map, sp_map, shape_a, delay_s

    def torque_errors(theta: np.ndarray) -> tuple[mujoco.MjModel, np.ndarray]:
        s_map, l_map, sp_map, shape_a, delay_s = unpack(theta)
        m_cal = apply_muscle_parameters(
            model,
            strength_scales=s_map,
            length_shifts_m=l_map,
            span_scales=sp_map,
        )
        d_cal = mujoco.MjData(m_cal)
        stateful = m_cal.actuator_actadr >= 0
        mom = np.zeros((m_cal.nu, m_cal.nv))
        errs: list[np.ndarray] = []

        for tr_i, (_, dt, qpos, qvel, _, _, env, act_to_col) in enumerate(prepared):
            act_full = compute_calibrated_emg_activations(
                m_cal,
                env,
                dt,
                act_to_col,
                shape_factor_a=shape_a,
                delay_s=delay_s,
            )
            act_zero = np.nan_to_num(act_full, nan=0.0)
            tau_pred = np.empty((len(qpos), len(dof_arr)), dtype=np.float64)
            for k in range(len(qpos)):
                d_cal.qpos[:], d_cal.qvel[:] = qpos[k], qvel[k]
                mujoco.mj_fwdPosition(m_cal, d_cal)
                mujoco.mj_fwdVelocity(m_cal, d_cal)
                a_k = act_zero[k]
                d_cal.act[m_cal.actuator_actadr[stateful]] = a_k[stateful]
                d_cal.ctrl[~stateful] = a_k[~stateful]
                mujoco.mj_fwdActuation(m_cal, d_cal)
                mujoco.mju_sparse2dense(
                    mom,
                    d_cal.actuator_moment,
                    d_cal.moment_rownnz,
                    d_cal.moment_rowadr,
                    d_cal.moment_colind,
                )
                f_trk = np.zeros(m_cal.nu, dtype=np.float64)
                f_trk[tracked_ids] = d_cal.actuator_force[tracked_ids]
                tau_pred[k] = mom[:, dof_arr].T @ f_trk
            errs.append((tau_pred - tau_trials[tr_i][:, dof_arr]).ravel())
        return m_cal, np.concatenate(errs)

    _, init_err = torque_errors(np.clip(x0, lo, hi))
    initial_rmse = float(np.sqrt(np.mean(init_err**2)))
    eval_count = 0

    def residual(theta: np.ndarray) -> np.ndarray:
        nonlocal eval_count
        eval_count += 1
        _, err = torque_errors(theta)
        if cfg.reg_weight > 0.0:
            reg = np.sqrt(cfg.reg_weight) * (theta - prior)
            reg[n_grp : n_grp + 2] *= 0.1
            return np.concatenate([err, reg])
        return err

    opt = least_squares(
        residual,
        np.clip(x0, lo, hi),
        bounds=(lo, hi),
        method="trf",
        diff_step=1e-3,
        ftol=1e-6,
        xtol=1e-5,
        max_nfev=cfg.max_nfev,
    )
    converged = bool(opt.success)
    status = int(opt.status)
    message = str(opt.message)
    optimality = float(opt.optimality)
    cost = float(opt.cost)
    if not converged:
        diag = (
            f"Muscle calibration did not converge (status={status}, "
            f"evaluations={eval_count}/{cfg.max_nfev}, optimality={optimality:.3e}): "
            f"{message}"
        )
        if cfg.on_failure == "raise":
            raise CalibrationError(diag)
        if cfg.on_failure == "warn":
            warnings.warn(diag, RuntimeWarning, stacklevel=2)

    cal_model, final_err = torque_errors(opt.x)
    final_rmse = float(np.sqrt(np.mean(final_err**2)))
    s_map, l_map, sp_map, shape_a, delay_s = unpack(opt.x)

    return MuscleCalibrationResult(
        calibrated_model=cal_model,
        strength_scales={model.actuator(i).name: v for i, v in s_map.items()},
        length_shifts_m={model.actuator(i).name: v for i, v in l_map.items()},
        span_scales={model.actuator(i).name: v for i, v in sp_map.items()},
        shape_factor_a=shape_a,
        delay_s=delay_s,
        calibrated_dofs=dofs,
        initial_torque_rmse=initial_rmse,
        final_torque_rmse=final_rmse,
        evaluations=eval_count,
        converged=converged,
        status=status,
        message=message,
        optimality=optimality,
        cost=cost,
    )


_COMPAT_ADAPTER_EXPORTS: frozenset[str] = frozenset(
    {"EMGAssistedPipelineResult", "run_emg_assisted_pipeline", "trial_from_c3d"}
)


def __getattr__(name: str) -> Any:
    if name in _COMPAT_ADAPTER_EXPORTS:
        mod = importlib.import_module("myosuite.integrations.capture.emg_calibration")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CalibrationError",
    "CalibrationTrial",
    "MuscleCalibrationConfig",
    "MuscleCalibrationResult",
    "OnCalibrationFailure",
    "apply_muscle_parameters",
    "calibrate_subject_muscles",
    "compute_calibrated_emg_activations",
    "nonlinear_excitation_shape",
    "prepare_calibration_trial",
]
