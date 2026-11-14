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
level matches the tracker's. ``joint_noise_deg`` is the error that remains after
the analysis' low-pass filter, so the bands do not narrow when the same motion is
sampled faster; ``joint_offset_deg`` adds a constant error per sample and joint
(marker placement, soft-tissue offset). With ``n`` samples, an interpolated ``[q, 1 - q]``
band holds a new draw with probability of about ``(n - 1)(1 - 2q) / (n + 1)``
(0.86 for 50 samples and a nominal 90 % band); use a few hundred samples for
calibrated bands.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields

import mujoco
import numpy as np
from scipy import signal

from myosuite.core.muscle_conditions import _peak_force
from myosuite.biomechanics.inverse_dynamics import clip_kinematics
from myosuite.biomechanics.static_optimization import (
    StaticOptimizationConfig,
    StaticOptimizationResult,
    StaticOptimizer,
    static_optimization,
)


@dataclass(frozen=True)
class UncertaintyConfig:
    """What varies between samples.

    Attributes:
        samples: Number of Monte Carlo samples.
        joint_noise_deg: Standard deviation of the per-frame hinge-angle error
            that remains after the analysis' low-pass filter [deg]; without a
            filter it is white per-frame noise of that size.
        joint_offset_deg: Standard deviation of a constant hinge-angle error per
            sample and joint [deg].
        strength_cv: Coefficient of variation of a log-normal factor (mean 1)
            on each muscle's peak active and passive force.
        mass_cv: Coefficient of variation of a log-normal factor (mean 1) on
            each moving body's mass and inertia.
        quantiles: Quantiles reported for every quantity.
        seed: Seed of the random generator.
    """

    samples: int = 50
    joint_noise_deg: float = 0.0
    joint_offset_deg: float = 0.0
    strength_cv: float = 0.0
    mass_cv: float = 0.0
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95)
    seed: int = 0


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
    """

    nominal: StaticOptimizationResult
    quantiles: tuple[float, ...]
    activation: np.ndarray
    actuator_force: np.ndarray
    reserve: np.ndarray
    joint_torque: np.ndarray
    contact_force: np.ndarray


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


def _filter_noise_gain(dt: float, lowpass_hz: float, filter_order: int = 2) -> float:
    """RMS gain on white noise of ``clip_kinematics``' zero-phase Butterworth filter."""
    sos = signal.butter(filter_order, lowpass_hz, fs=1.0 / dt, output="sos")
    n = 2 * int(np.ceil(20.0 / (lowpass_hz * dt))) + 1
    impulse = np.zeros(n)
    impulse[n // 2] = 1.0
    return float(np.linalg.norm(signal.sosfiltfilt(sos, impulse, padlen=0)))


def joint_noise(
    rng: np.random.Generator,
    frames: int,
    joints: int,
    dt: float,
    lowpass_hz: float | None,
    config: UncertaintyConfig,
) -> np.ndarray:
    """``(frames, joints)`` hinge-angle errors [rad] of one sample.

    White per-frame noise scaled by the filter's noise gain at this frame rate,
    so that after ``clip_kinematics(..., lowpass_hz=lowpass_hz)`` its standard
    deviation is ``joint_noise_deg``, plus a constant offset per joint.

    Args:
        rng: Random generator.
        frames: Number of frames.
        joints: Number of hinge joints.
        dt: Frame period [s].
        lowpass_hz: The analysis' filter cut-off, or ``None``.
        config: Noise levels.

    Returns:
        The errors to add to the hinge angles.
    """
    gain = 1.0 if lowpass_hz is None else _filter_noise_gain(dt, lowpass_hz)
    noise = config.joint_noise_deg / gain * rng.standard_normal((frames, joints))
    offset = config.joint_offset_deg * rng.standard_normal(joints)
    return np.radians(noise + offset)


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
    """
    cfg = config or UncertaintyConfig()
    rng = np.random.default_rng(cfg.seed)
    hinges = model.jnt_qposadr[model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE]
    # Reserve, residual and contact weights stay the nominal model's in every sample.
    optimization = StaticOptimizer(model, optimization).config

    def analyse(m: mujoco.MjModel, q: np.ndarray) -> StaticOptimizationResult:
        kin = clip_kinematics(m, q, dt, lowpass_hz=lowpass_hz)
        return static_optimization(m, kin.qpos, kin.qvel, kin.qacc, config=optimization)

    nominal = analyse(model, qpos)
    samples = []
    for _ in range(cfg.samples):
        m = perturbed_model(model, rng, cfg)
        q = np.array(qpos, dtype=np.float64)
        q[:, hinges] += joint_noise(rng, len(q), len(hinges), dt, lowpass_hz, cfg)
        samples.append(analyse(m, q))
    bands = {
        f.name: np.quantile(
            np.stack([getattr(s, f.name) for s in samples]), cfg.quantiles, axis=0
        )
        for f in fields(UncertaintyBands)
        if f.name not in ("nominal", "quantiles")
    }
    return UncertaintyBands(nominal=nominal, quantiles=tuple(cfg.quantiles), **bands)
