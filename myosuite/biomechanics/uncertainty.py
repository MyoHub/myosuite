# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Uncertainty bands for muscle estimates.

Static optimization gives one answer per frame, but the tracked joint angles are
noisy and the subject's muscle strength and segment masses are estimates.
:func:`static_optimization_bands` repeats the analysis (kinematics, inverse
dynamics, static optimization) on Monte Carlo samples of all three and reports
quantile bands next to the nominal solution::

    bands = static_optimization_bands(
        model, qpos, dt, lowpass_hz=6.0,
        config=UncertaintyConfig(joint_noise_deg=0.5, strength_cv=0.2, mass_cv=0.05),
    )
    lo, mid, hi = bands.activation  # 5th, 50th, 95th percentiles, (T, nu) each

Joint noise is added to the measured angles (a parametric bootstrap): the bands
are centred on the measured motion and have the right width when the noise
level matches the tracker's. With ``n`` samples, an interpolated ``[q, 1 - q]``
band holds a new draw with probability of about ``(n - 1)(1 - 2q) / (n + 1)``
(0.86 for 50 samples and a nominal 90 % band); use a few hundred samples for
calibrated bands.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from myosuite.core.muscle_conditions import _peak_force
from myosuite.biomechanics.inverse_dynamics import clip_kinematics
from myosuite.biomechanics.static_optimization import (
    StaticOptimizationConfig,
    StaticOptimizationResult,
    StaticOptimizer,
    _solve_clip,
    static_optimization,
)

_BAND_FIELDS = (
    "activation",
    "actuator_force",
    "reserve",
    "joint_torque",
    "contact_force",
)


@dataclass(frozen=True)
class UncertaintyConfig:
    """What varies between samples.

    Attributes:
        samples: Number of Monte Carlo samples.
        joint_noise_deg: Standard deviation of Gaussian noise added to every
            hinge angle of every frame [deg].
        strength_cv: Coefficient of variation of a log-normal factor (mean 1)
            on each muscle's peak active and passive force.
        mass_cv: Coefficient of variation of a log-normal factor (mean 1) on
            each moving body's mass and inertia.
        quantiles: Quantiles reported for every quantity.
        seed: Seed of the random generator.
        allow_inaccurate: Accept OSQP's ``"solved inaccurate"`` status as well
            as ``"solved"``. Other failures and nonfinite results always raise.
    """

    samples: int = 50
    joint_noise_deg: float = 0.0
    strength_cv: float = 0.0
    mass_cv: float = 0.0
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95)
    seed: int = 0
    allow_inaccurate: bool = False


@dataclass(frozen=True)
class UncertaintyBands:
    """Nominal solution and quantile bands.

    Attributes:
        nominal: Static optimization of the unperturbed model and clip.
        quantiles: The reported quantiles ``q``.
        activation: ``(len(q), T, nu)`` activation quantiles.
        actuator_force: ``(len(q), T, nu)`` actuator force quantiles [N].
        reserve: ``(len(q), T, nv)`` reserve and residual quantiles.
        joint_torque: ``(len(q), T, nv)`` inverse-dynamics quantiles.
        contact_force: ``(len(q), T, nbody, 3)`` contact force quantiles [N].
        sample_status: ``(samples, T)`` OSQP statuses, before quantiles.
        sample_iterations: ``(samples, T)`` OSQP iteration counts.
        sample_primal_residual: ``(samples, T)`` OSQP primal residuals of the QP.
        sample_dual_residual: ``(samples, T)`` OSQP dual residuals of the QP.
    """

    nominal: StaticOptimizationResult
    quantiles: tuple[float, ...]
    activation: np.ndarray
    actuator_force: np.ndarray
    reserve: np.ndarray
    joint_torque: np.ndarray
    contact_force: np.ndarray
    sample_status: np.ndarray
    sample_iterations: np.ndarray
    sample_primal_residual: np.ndarray
    sample_dual_residual: np.ndarray


class UncertaintySolveError(RuntimeError):
    """A nominal or sampled static optimization did not meet the solve policy.

    Attributes:
        sample_index: Zero-based sample index, or ``None`` for the nominal solve.
        frame_indices: Indices of frames with unacceptable or nonfinite results.
        result: The failed solve, including its statuses and residual diagnostics.
    """

    def __init__(
        self,
        result: StaticOptimizationResult,
        sample_index: int | None,
        frame_indices: np.ndarray,
    ) -> None:
        """Keep the failed solve and identify its first failing frame."""
        self.result, self.sample_index, self.frame_indices = (
            result,
            sample_index,
            frame_indices,
        )
        label = "nominal" if sample_index is None else f"sample {sample_index}"
        frame = int(frame_indices[0])
        super().__init__(
            f"{label} static optimization failed at frame {frame}: "
            f"{result.status[frame]}, primal residual={result.primal_residual[frame]:g}, "
            f"dual residual={result.dual_residual[frame]:g}; "
            f"{len(frame_indices)} frame(s) rejected by the solve policy "
            "(including nonfinite results)"
        )


def _check_solve(
    result: StaticOptimizationResult, sample_index: int | None, allow_inaccurate: bool
) -> None:
    """Reject failed frames without dropping draws or changing the sampled prior."""
    accepted = ("solved", "solved inaccurate") if allow_inaccurate else ("solved",)
    bad = ~np.isin(result.status, accepted)
    bad |= ~np.isfinite(result.primal_residual) | ~np.isfinite(result.dual_residual)
    for name in _BAND_FIELDS:
        value = getattr(result, name)
        bad |= ~np.isfinite(value).all(axis=tuple(range(1, value.ndim)))
    if bad.any():
        raise UncertaintySolveError(result, sample_index, np.flatnonzero(bad))


def _lognormal(rng: np.random.Generator, cv: float, size: int) -> np.ndarray:
    """Log-normal factors with mean 1 and coefficient of variation *cv*."""
    sigma = np.sqrt(np.log1p(cv**2))
    return rng.lognormal(-0.5 * sigma**2, sigma, size)


def perturbed_model(
    model: mujoco.MjModel, rng: np.random.Generator, config: UncertaintyConfig
) -> mujoco.MjModel:
    """A copy of *model* with sampled muscle strength and segment masses.

    Peak forces are written explicitly first, because an automatic peak force
    (``force="-1"``) follows ``acc0``, which changes with the masses. Mass and
    inertia scale together (each body keeps its inertial frame), then
    ``mj_setConst`` refreshes the derived constants.

    Args:
        model: Compiled model (not modified).
        rng: Random generator.
        config: Coefficients of variation.

    Returns:
        The perturbed copy.
    """
    m = copy.deepcopy(model)
    muscle = m.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE
    bias = m.actuator_biasprm[:, 2].copy()
    auto = muscle & (bias < 0)
    bias[auto] = m.actuator_biasprm[auto, 3] / np.maximum(
        m.actuator_acc0[auto], mujoco.mjMINVAL
    )
    strength = _lognormal(rng, config.strength_cv, m.nu)
    m.actuator_gainprm[muscle, 2] = _peak_force(m)[muscle] * strength[muscle]
    m.actuator_biasprm[muscle, 2] = bias[muscle] * strength[muscle]
    moving = m.body_weldid != 0
    mass = _lognormal(rng, config.mass_cv, m.nbody)
    m.body_mass[moving] *= mass[moving]
    m.body_inertia[moving] *= mass[moving, None]
    mujoco.mj_setConst(m, mujoco.MjData(m))
    return m


def static_optimization_bands(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    dt: float,
    *,
    config: UncertaintyConfig | None = None,
    lowpass_hz: float | None = None,
    optimization: StaticOptimizationConfig | None = None,
) -> UncertaintyBands:
    """Quantile bands of a clip's static optimization over sampled uncertainty.

    Each sample perturbs the model (:func:`perturbed_model`) and the hinge
    angles, then runs :func:`~myosuite.biomechanics.inverse_dynamics.clip_kinematics`
    and :func:`~myosuite.biomechanics.static_optimization.static_optimization`. The
    cost is ``samples`` times that of one analysis.

    Args:
        model: Compiled model.
        qpos: ``(T, nq)`` measured positions.
        dt: Frame period [s].
        config: What varies, how much, and the quantiles.
        lowpass_hz: Kinematic filter cut-off (recommended with joint noise).
        optimization: Static-optimization settings (e.g. contact forces).

    Returns:
        The nominal solution and the bands.

    Raises:
        UncertaintySolveError: If any nominal or sampled frame fails, is nonfinite,
            or is ``"solved inaccurate"`` without ``config.allow_inaccurate``.
            Failed draws are never silently dropped or resampled.
        ValueError: If fewer than one sample is requested.
    """
    cfg = config or UncertaintyConfig()
    if cfg.samples < 1:
        raise ValueError("samples must be at least one")
    rng = np.random.default_rng(cfg.seed)
    hinges = model.jnt_qposadr[model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE]
    # Reserve, residual and contact weights stay the nominal model's in every sample.
    optimization = StaticOptimizer(model, optimization).config

    def analyse(
        m: mujoco.MjModel, q: np.ndarray, *, owned: bool = False
    ) -> StaticOptimizationResult:
        kin = clip_kinematics(m, q, dt, lowpass_hz=lowpass_hz)
        if owned:
            optimizer = StaticOptimizer._from_owned_model(m, optimization)
            return _solve_clip(optimizer, kin.qpos, kin.qvel, kin.qacc)
        return static_optimization(m, kin.qpos, kin.qvel, kin.qacc, config=optimization)

    nominal = analyse(model, qpos)
    _check_solve(nominal, None, cfg.allow_inaccurate)
    samples = []
    for _ in range(cfg.samples):
        m = perturbed_model(model, rng, cfg)
        q = np.array(qpos, dtype=np.float64)
        q[:, hinges] += np.radians(cfg.joint_noise_deg) * rng.standard_normal(
            (len(q), len(hinges))
        )
        result = analyse(m, q, owned=True)
        _check_solve(result, len(samples), cfg.allow_inaccurate)
        samples.append(result)
    bands = {
        name: np.quantile(
            np.stack([getattr(s, name) for s in samples]), cfg.quantiles, axis=0
        )
        for name in _BAND_FIELDS
    }
    diagnostics = {
        f"sample_{name}": np.stack([getattr(s, name) for s in samples])
        for name in ("status", "iterations", "primal_residual", "dual_residual")
    }
    return UncertaintyBands(
        nominal=nominal, quantiles=tuple(cfg.quantiles), **bands, **diagnostics
    )
