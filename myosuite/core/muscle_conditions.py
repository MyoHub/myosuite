# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""
Muscle condition helpers: cumulative fatigue, sarcopenia, reafferentation.

These are pure transform functions that modify MuJoCo MjSpec or MjModel
to represent physiological muscle conditions. They are used by ModelBuilder
as transform callbacks and by term functions via EnvAccessor.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import mujoco


# ---------------------------------------------------------------------------
# Per-muscle fatigue parameters, F and R in 1/s. Muscle-group, sex-specific and
# joint rows: Rakshit et al. 2021 (J Biomech 127:110695, Table 2, p. 5; their k is
# r, "-" there means no r). Joint rows (Ankle, Elbow, Hand, Knee) are the "combined
# sexes, combined FMGs" column: separate fits to the pooled data, not averages of
# the group rows. Knee equals Knee-Extensor because only knee-extensor data
# existed (Section 3). Default,
# Shoulder, Trunk, Thigh-Hamstring and Hip: Frey-Law et al. 2012 (J Biomech
# 45:1803, Table 1) with r of Looft et al. 2018 (J Biomech 77:16, Table 4). Rows
# without "r" use Default's r = 15. Validated against endurance times in
# docs/source/fatigue_validation.rst.
# ---------------------------------------------------------------------------

MUSCLE_FATIGUE_PARAMS: dict[str, dict[str, float]] = {
    "Ankle-Dorsiflexor-F": {"F": 0.00746, "R": 0.00081, "r": 4.97},
    "Ankle-Dorsiflexor-M": {"F": 0.00725, "R": 0.00096, "r": 10.36},
    "Ankle-Dorsiflexor": {"F": 0.00828, "R": 0.00204, "r": 7.07},
    "Ankle-Plantarflexor-F": {"F": 0.00702, "R": 0.00098},
    "Ankle-Plantarflexor-M": {"F": 0.00683, "R": 0.00093},
    "Ankle-Plantarflexor": {"F": 0.00695, "R": 0.00096},
    "Elbow-Extensor-F": {"F": 0.01874, "R": 0.00206, "r": 21.22},
    "Elbow-Extensor-M": {"F": 0.01269, "R": 0.00085, "r": 30.21},
    "Elbow-Extensor": {"F": 0.01559, "R": 0.00125, "r": 25.52},
    "Elbow-Flexor-F": {"F": 0.00965, "R": 0.00197, "r": 6.22},
    "Elbow-Flexor-M": {"F": 0.01302, "R": 0.00188, "r": 8.99},
    "Elbow-Flexor": {"F": 0.01703, "R": 0.00494, "r": 4.68},
    "Hand-Adductor-Pollicis-F": {"F": 0.00476, "R": 0.00093, "r": 6.62},
    "Hand-Adductor-Pollicis-M": {"F": 0.00586, "R": 0.00202, "r": 1.00},
    "Hand-Adductor-Pollicis": {"F": 0.00558, "R": 0.00283, "r": 1.00},
    "Hand-First-Dorsal-Interossei-F": {"F": 0.03999, "R": 0.03983},
    "Hand-First-Dorsal-Interossei-M": {"F": 0.01637, "R": 0.00360, "r": 3.66},
    "Hand-First-Dorsal-Interossei": {"F": 0.02686, "R": 0.00656, "r": 3.41},
    "Wrist-Flexor-F": {
        "F": 0.01159,
        "R": 0.00217,
        "r": 7.39,
    },  # from Hand G/Grip group (https://doi.org/10.1016/j.jbiomech.2021.110695)
    "Wrist-Flexor-M": {
        "F": 0.01238,
        "R": 0.00178,
        "r": 8.00,
    },  # from Hand G/Grip group (https://doi.org/10.1016/j.jbiomech.2021.110695)
    "Wrist-Flexor": {
        "F": 0.01235,
        "R": 0.00135,
        "r": 12.51,
    },  # from Hand G/Grip group (https://doi.org/10.1016/j.jbiomech.2021.110695)
    "Knee-Extensor-F": {"F": 0.01407, "R": 0.00185, "r": 6.32},
    "Knee-Extensor-M": {"F": 0.01420, "R": 0.00153, "r": 10.96},
    "Knee-Extensor": {"F": 0.00825, "R": 0.00076, "r": 14.85},
    "Ankle": {"F": 0.01485, "R": 0.00333, "r": 9.31},
    "Toe": {
        "F": 0.01485,
        "R": 0.00333,
        "r": 9.31,
    },  # from Ankle group (https://doi.org/10.1016/j.jbiomech.2021.110695)
    "Elbow": {"F": 0.01086, "R": 0.00225, "r": 4.93},
    "Hand": {"F": 0.01227, "R": 0.00134, "r": 9.10},
    "Wrist": {
        "F": 0.01227,
        "R": 0.00134,
        "r": 9.10,
    },  # from Hand group (https://doi.org/10.1016/j.jbiomech.2021.110695)
    "Finger": {
        "F": 0.01227,
        "R": 0.00134,
        "r": 9.10,
    },  # from Hand group (https://doi.org/10.1016/j.jbiomech.2021.110695)
    "Knee": {"F": 0.00825, "R": 0.00076, "r": 14.85},
    # F, R: shoulder fit of Frey-Law et al. 2012 (Table 1); r = 15: Looft & Frey-Law
    # 2020 (https://doi.org/10.1016/j.jbiomech.2020.109762), better than r = 30.
    "Shoulder": {"F": 0.01820, "R": 0.00168, "r": 15},
    # hamstrings: no knee-flexor fit is published; the knee fit of Frey-Law et al. 2012
    # (Table 1) with r = 15 of Looft et al. 2018 (Table 4), the joint-level knee parameters
    # Yang et al. 2025 (Front Physiol 16:1518847) use for the knee flexors and extensors
    "Thigh-Hamstring": {"F": 0.01500, "R": 0.00149, "r": 15},
    # hip muscles: no hip-specific fit is published (Yang et al. 2025 use the general fit);
    # the "general" F, R of Frey-Law et al. 2012 (Table 1), r of Looft et al. 2018
    "Hip": {"F": 0.00970, "R": 0.00091, "r": 15},
    # lumbar and abdominal muscles: trunk F, R of Frey-Law et al. 2012 (Table 1); r = 15
    # as for Default (Looft et al. 2018 and Looft & Frey-Law 2020 fit no trunk r)
    "Trunk": {"F": 0.00755, "R": 0.00075, "r": 15},
    # default: "general" F, R of Frey-Law et al. 2012 (Table 1), r of Looft et al. 2018
    "Default": {"F": 0.00970, "R": 0.00091, "r": 15},
    # v2.4 fallback: elbow F, R of Frey-Law et al. 2012 with R / 10 (r * 10 keeps r * R)
    "Default_v2_4": {"F": 0.00912, "R": 0.1 * 0.00094, "r": 10 * 15},
}

FATIGUE_REST_THRESHOLD = 0.01
"""Commands ``TL <= FATIGUE_REST_THRESHOLD`` count as rest: recovery runs at ``r R``.

Rakshit et al. (2021, Eq. 7) define rest as a zero command. A command through the muscle
sigmoid never reaches zero (smallest value 0.00055), so a small threshold makes a relaxed
muscle count as resting, as a muscle commanded to 0 does.
"""

_TINY = 1e-20  # floor for rates and denominators (R = 0, F = 0, empty segments)


def _nonzero(xp: Any, x: Any) -> Any:
    """``x`` with magnitudes below :data:`_TINY` replaced by ``_TINY``."""
    return xp.where(xp.abs(x) > _TINY, x, _TINY)


def _relax(
    xp: Any, MA: Any, MF: Any, L: Any, TL: Any, F: Any, rho: Any, t: Any
) -> tuple[Any, Any]:
    """Exact ``(MA, MF)`` after ``t`` with ``C = L (TL - MA)`` (MA decoupled from MF).

    Increments use ``expm1``: float32 would round ``e^{-rho t}`` close to 1.
    """
    k = L + F
    MAs = L * TL / _nonzero(xp, k)
    D = MA - MAs
    em = xp.expm1(-rho * t)
    kr = _nonzero(xp, k - rho)
    dMF = MF * em + F * (-MAs * em / rho - D * (1.0 + em) * xp.expm1(-kr * t) / kr)
    return MA + D * xp.expm1(-k * t), MF + dMF


def _limited_modes(xp: Any, L: Any, F: Any, rho: Any) -> tuple[Any, ...]:
    """Eigen-data of ``C = L MR = L (1 - MA - MF)``, a linear 2x2 system in (MA, MF).

    Returns the equilibrium ``(MAe, MFe)``, ``s = lam2 - lam1`` and the fast and slow
    eigenvalues ``lam1 ~ -L``, ``lam2 ~ -(F + rho)``.
    """
    det = _nonzero(xp, (L + F) * rho + L * F)
    disc = (L + F - rho) ** 2 - 4.0 * L * F
    s = _nonzero(xp, xp.sqrt(xp.where(disc > 0, disc, 0.0)))
    lam1 = -0.5 * (L + F + rho + s)
    lam2 = det / lam1  # product form avoids cancellation
    return L * rho / det, L * F / det, s, lam1, lam2


def _limited(
    xp: Any, MA: Any, MF: Any, L: Any, F: Any, rho: Any, t: Any, modes: tuple
) -> tuple[Any, Any]:
    """Exact ``(MA, MF)`` after ``t`` with ``C = L MR``: ``e^{At} = g I + h A``."""
    MAe, MFe, s, _, lam2 = modes
    y1, y2 = MA - MAe, MF - MFe
    em = xp.expm1(lam2 * t)
    h = (1.0 + em) * -xp.expm1(-s * t) / s  # (e^{lam2 t} - e^{lam1 t}) / (lam2 - lam1)
    g1 = em - lam2 * h  # g - 1
    Ay1 = -(L + F) * y1 - L * y2
    Ay2 = F * y1 - rho * y2
    return MA + g1 * y1 + h * Ay1, MF + g1 * y2 + h * Ay2


def _free_switch_time(
    xp: Any, MA: Any, MF: Any, L: Any, TL: Any, thr: Any, F: Any, rho: Any
) -> Any:
    """Time the slow mode of MF reaches ``thr`` under ``C = L (TL - MA)`` (inf: never).

    Past the fast transient, ``MF(t) = M + tau (F MA* - rho M)`` with
    ``tau = (1 - e^{-rho t}) / rho`` and ``M = MF + F (MA - MA*) / (L + F - rho)``.
    """
    k = L + F
    MAs = L * TL / _nonzero(xp, k)
    M = MF + F * (MA - MAs) / _nonzero(xp, k - rho)
    slope = F * MAs - rho * M
    tau = xp.where(M < thr, (thr - M) / _nonzero(xp, slope), 0.0)
    x = rho * xp.where(tau > 0, tau, 0.0)
    t = -xp.log1p(-xp.where(x < 1.0 - 1e-6, x, 1.0 - 1e-6)) / rho
    return xp.where((M >= thr) | ((slope > 0) & (x < 1.0)), t, xp.inf)


def _bounded(xp: Any, MA: Any, MF: Any) -> tuple[Any, Any, Any]:
    """``(MA, MR, MF)`` with rounding clipped: each in ``[0, 1]``, summing to one."""
    MA = xp.where(MA > 0, xp.where(MA < 1, MA, 1.0), 0.0)
    MF = xp.where(MF > 0, xp.where(MF < 1 - MA, MF, 1 - MA), 0.0)
    return MA, 1.0 - MA - MF, MF


def three_cc_r_step(
    xp: Any,
    MA: Any,
    MF: Any,
    TL: Any,
    *,
    F: Any,
    R: Any,
    r: Any,
    tauact: Any,
    taudeact: Any,
    dt: float,
) -> tuple[Any, Any, Any]:
    """Exact 3CC-r step of any length ``dt`` under a constant command ``TL``.

    Backend-agnostic (``xp`` is numpy, torch or jax.numpy) and vectorised without
    host syncs; NumPy skips the split segments when no muscle needs them.
    ``MR = 1 - MA - MF``. With the activation rates frozen at the start of the step
    (``LD = 1 / (tauact (0.5 + 1.5 MA0))``, ``LR = (0.5 + 1.5 MA0) / taudeact``, as
    in the earlier explicit step), each controller regime is a linear ODE in
    ``(MA, MF)`` with a closed form:

    - ``MA >= TL``: ``C = LR (TL - MA)``; MA relaxes to ``LR TL / (LR + F)`` and
      ``MF(t) = MF0 e^{-rho t} + F int_0^t e^{-rho (t - s)} MA(s) ds``, i.e. fatigue
      accrues with the time integral of MA, ``TL t + (MA0 - TL)(1 - e^{-L t}) / L``
      for ``F, rho << L``, and decays exponentially. Ends when MA reaches
      ``max(TL, 0)`` at ``ln((MA0 - MA*) / (TL - MA*)) / (LR + F)``.
    - ``MA < TL``, ``MF < 1 - TL``: the same with ``LD``.
    - ``MA < TL``, ``MF >= 1 - TL`` (MR limits, ``C = LD MR``): coupled 2x2 system,
      solved with its two real eigenvalues ``-LD`` (fast) and ``-(F + rho)`` (slow).

    ``C`` is continuous across ``MF = 1 - TL``, so the switch between the last two
    regimes is placed where the slow mode crosses it (closed form; the neglected
    fast transient changes the state at second order). A step has at most three
    segments. ``rho = r R`` at rest (``TL <= FATIGUE_REST_THRESHOLD``), else ``R``.
    A NaN command keeps ``C = 0``, as the masks of the explicit step did.

    The frozen rates are the remaining error against MuJoCo's MA-dependent rates:
    after a large command jump MA leads them by up to ~0.2 within a 10-25 ms step,
    and each transient shifts MF by ~1e-5 to 1e-4.

    Args:
        xp: Array module of the inputs.
        MA: Active fraction.
        MF: Fatigued fraction.
        TL: Target load (command), broadcastable with ``MA``.
        F: Fatigue rate [1/s].
        R: Recovery rate [1/s].
        r: Rest recovery multiplier.
        tauact: Activation time constants [s].
        taudeact: Deactivation time constants [s].
        dt: Step length [s].

    Returns:
        ``(MA, MR, MF)`` after ``dt``, each in ``[0, 1]`` and summing to one.
    """
    rho = xp.where(TL <= FATIGUE_REST_THRESHOLD, r * R, R)
    rho = xp.where(rho > _TINY, rho, _TINY)
    nan = xp.isnan(TL)
    TL = xp.where(nan, 0.0, TL)
    act = 0.5 + 1.5 * MA
    LD = xp.where(nan, 0.0, 1.0 / (tauact * act))
    LR = xp.where(nan, 0.0, act / taudeact)
    TLc = xp.where(TL > 0, TL, 0.0)  # MA stops at 0 under a negative command
    thr = 1.0 - TL  # MR runs out once MF >= 1 - TL
    T = MA * 0.0 + dt

    # 1) MA >= TL: deactivate until MA reaches max(TL, 0).
    falling = MA >= TL
    k1 = _nonzero(xp, LR + F)
    gap = TLc - LR * TL / k1
    ratio = (MA - LR * TL / k1) / _nonzero(xp, gap)
    t_cross = xp.where(gap > 0, xp.log(xp.where(ratio > 1, ratio, 1.0)) / k1, xp.inf)
    crosses = falling & (t_cross < T)
    if xp is np:  # NumPy skips the split segments when no muscle needs them
        t_a = _free_switch_time(xp, MA, MF, LD, TLc, thr, F, rho)
        if not (crosses | ~falling & ((MF >= thr) | (t_a < T))).any():
            L = xp.where(falling, LR, LD)
            return _bounded(xp, *_relax(xp, MA, MF, L, TL, F, rho, T))
    d1 = xp.where(falling, xp.where(crosses, t_cross, T), 0.0)
    MA1, MF1 = _relax(xp, MA, MF, LR, TL, F, rho, d1)
    MA1 = xp.where(crosses, TLc, MA1)
    T2 = T - d1

    # 2) MA < TL: free (A) or MR-limited (B) recruitment; they switch where the slow
    # mode of MF crosses 1 - TL.
    limited = MF1 >= thr
    modes = _limited_modes(xp, LD, F, rho)
    MAe, MFe, s, lam1, lam2 = modes
    p2 = (F * (MA1 - MAe) - (rho + lam1) * (MF1 - MFe)) / s  # slow-mode part of MF
    q = (thr - MFe) / _nonzero(xp, p2)
    down = (q > 0) & (q < 1)
    t_b = xp.log(xp.where(down, q, 1.0)) / lam2
    t_b = xp.where(MFe + p2 < thr, 0.0, xp.where(down, t_b, xp.inf))
    t_a = _free_switch_time(xp, MA1, MF1, LD, TLc, thr, F, rho)
    t_sw = xp.where(limited, t_b, t_a)
    d2 = xp.where(t_sw < T2, t_sw, T2)
    d3 = T2 - d2

    MA_a, MF_a = _relax(xp, MA1, MF1, LD, TLc, F, rho, d2)
    MA_b, MF_b = _limited(xp, MA1, MF1, LD, F, rho, d2, modes)
    MA2 = xp.where(limited, MA_b, MA_a)
    MF2 = xp.where(limited, MF_b, MF_a)
    MA_a, MF_a = _relax(xp, MA2, MF2, LD, TLc, F, rho, d3)
    MA_b, MF_b = _limited(xp, MA2, MF2, LD, F, rho, d3, modes)
    MA3 = xp.where(limited, MA_a, MA_b)
    return _bounded(xp, MA3, xp.where(limited, MF_a, MF_b))


_DEFAULT_F = MUSCLE_FATIGUE_PARAMS["Default"]["F"]
_DEFAULT_R = MUSCLE_FATIGUE_PARAMS["Default"]["R"]
_DEFAULT_r = MUSCLE_FATIGUE_PARAMS["Default"]["r"]
# ``use_uniform_params``: the single parameter set of the legacy (<= v2.x)
# CumulativeFatigue, also hard-coded in the JAX models (physics/fatigue_jax.py).
_UNIFORM_PARAMS = MUSCLE_FATIGUE_PARAMS["Default_v2_4"]


def _muscle_ids(
    mj_model: Any, muscles: Sequence[int | str] | None = None
) -> np.ndarray:
    """Actuator ids of *muscles* (ids or names), default every muscle in model order.

    Raises:
        ValueError: If an entry is not a muscle actuator.
    """
    import mujoco as _mujoco  # noqa: PLC0415

    is_muscle = mj_model.actuator_dyntype == _mujoco.mjtDyn.mjDYN_MUSCLE
    if muscles is None:
        return np.flatnonzero(is_muscle)
    ids = np.array(
        [mj_model.actuator(m).id if isinstance(m, str) else int(m) for m in muscles],
        dtype=int,
    )
    if not is_muscle[ids].all():
        raise ValueError(f"not muscle actuators: {ids[~is_muscle[ids]].tolist()}")
    return ids


def _muscle_time_constants(
    mj_model: Any, ids: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(tauact, taudeact)`` of the muscle actuators *ids* (default all)."""
    ids = _muscle_ids(mj_model) if ids is None else ids
    dynprm = np.asarray(mj_model.actuator_dynprm)[ids]
    return dynprm[:, 0].copy(), dynprm[:, 1].copy()


def _muscle_group(name: str) -> str:
    """Return the functional muscle group of actuator *name*, or ``"Default"``.

    Scene models prefix actuator names (mjlab: ``robot/BIClong``), two-agent scenes
    prefix the agent (``a0_BIClong``), and models mark the side in several ways
    (``ECRL_r``, ``rect_abd_l``, MuscleMimic's ``BIClong_left``, ``R:ECRL``), while
    ``MUSCLE_FMG`` keys arm, hand and torso muscles without it. Leg muscles are keyed
    with their side, so the exact name is tried first.
    """
    from myosuite.core.muscle_groups import MUSCLE_FMG  # noqa: PLC0415

    base = re.sub(r"^a\d+_", "", name.rsplit("/", 1)[-1])
    base = base.removeprefix("R:").removeprefix("L:").removesuffix("_left")
    for key in (base, base.removesuffix("_r"), base.removesuffix("_l")):
        if key in MUSCLE_FMG:
            return MUSCLE_FMG[key]
    return "Default"


def _per_muscle_params(
    mj_model: Any, sex: str | None, ids: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive per-muscle F, R, r arrays from MFG lookup.

    Returns arrays of shape ``(na,)``, one entry per muscle actuator in *ids*
    (default: every muscle actuator of *mj_model*).
    """
    ids = _muscle_ids(mj_model) if ids is None else ids
    na = int(ids.size)
    actuator_names = [mj_model.actuator(int(i)).name for i in ids]

    F_arr = np.zeros(na)
    R_arr = np.zeros(na)
    r_arr = np.zeros(na)

    for idx, name in enumerate(actuator_names):
        mfg = _muscle_group(name)
        if sex is not None:
            mfg_sex = f"{mfg}-{sex}"
            if mfg_sex not in MUSCLE_FATIGUE_PARAMS:
                mfg_sex = mfg
        else:
            mfg_sex = mfg
        # Walk up to parent group if needed
        group = mfg_sex if mfg_sex in MUSCLE_FATIGUE_PARAMS else mfg_sex.split("-")[0]
        if group not in MUSCLE_FATIGUE_PARAMS:
            group = "Default"
        p = MUSCLE_FATIGUE_PARAMS[group]
        F_arr[idx] = p.get("F", _DEFAULT_F)
        R_arr[idx] = p.get("R", _DEFAULT_R)
        r_arr[idx] = p.get("r", _DEFAULT_r)

    return F_arr, R_arr, r_arr


# ---------------------------------------------------------------------------
# Sarcopenia helpers
# ---------------------------------------------------------------------------


def _peak_force(model: mujoco.MjModel) -> np.ndarray:
    """Per-actuator peak force ``gainprm[:, 2]`` as MuJoCo applies it.

    A negative muscle peak force (``force="-1"``) means automatic: MuJoCo
    uses ``gainprm[3] / acc0`` at runtime, so scaling the stored value would
    leave the force unchanged.
    """
    import mujoco as _mujoco  # noqa: PLC0415

    force = np.array(model.actuator_gainprm[:, 2], dtype=np.float64)
    auto = (force < 0) & (model.actuator_gaintype == _mujoco.mjtGain.mjGAIN_MUSCLE)
    if np.any(auto):
        force[auto] = model.actuator_gainprm[auto, 3] / np.maximum(
            model.actuator_acc0[auto], _mujoco.mjMINVAL
        )
    return force


def apply_sarcopenia_to_spec(
    spec: mujoco.MjSpec, force_scale: float = 0.5
) -> mujoco.MjSpec:
    """Scale muscle peak forces to simulate sarcopenia (age-related muscle loss).

    Spec-level equivalent of :func:`apply_sarcopenia_to_model`: scales the
    *compiled* peak force ``gainprm[2]`` of every actuator. Muscles declared
    with ``force="-1"`` keep ``gainprm[2] = -1`` after compilation and get
    their peak force ``scale / acc0`` from the compiled model, so the spec is
    compiled once to resolve the peak forces, which are then written back
    explicitly (scaled).

    Args:
        spec: MuJoCo model spec to modify in-place.
        force_scale: Fraction of original peak force to retain (0–1).

    Returns:
        The modified spec (same object, modified in-place).

    Example:
        >>> import mujoco
        >>> spec = mujoco.MjSpec.from_file("elbow.xml")
        >>> apply_sarcopenia_to_spec(spec, force_scale=0.5)
    """
    peak_force = _peak_force(spec.compile())
    for actuator, force in zip(spec.actuators, peak_force, strict=True):
        actuator.gainprm[2] = float(force) * force_scale
    return spec


def apply_sarcopenia_to_model(model: mujoco.MjModel, force_scale: float = 0.5) -> None:
    """Scale muscle peak forces on a compiled MjModel in-place.

    Multiplies ``actuator_gainprm[:, 2]`` (the maximum isometric force Fmax)
    by *force_scale* for every actuator, as the original
    ``BaseV0.initializeConditions`` does. Muscles declared with
    ``force="-1"`` are first resolved to their peak force ``scale / acc0``.

    Args:
        model: Compiled MuJoCo model to modify.
        force_scale: Fraction of original peak force to retain (0–1).
    """
    model.actuator_gainprm[:, 2] = _peak_force(model) * force_scale


# ---------------------------------------------------------------------------
# CumulativeFatigue — 3CC-r model (numpy, CPU)
# ---------------------------------------------------------------------------


class CumulativeFatigue:
    """3CC-r cumulative fatigue model (Xia & Frey Law 2008, Looft et al. 2018).

    Tracks the active (MA), fatigued (MF), and resting (MR) compartments
    for each muscle actuator.  By default uses per-muscle fatigue / recovery
    constants derived from biomechanical functional muscle groups (FMG).

    The recovery multiplier ``r`` acts only at rest, as in Rakshit et al.
    (2021, Eqs. 1, 3 and 7; their ``k`` is ``r`` here):
    ``dMR/dt = -C(t) + r(k, TL) x R x MF``,
    ``dMF/dt = F x MA - r(k, TL) x R x MF`` with
    ``r = k if TL = 0`` and ``r = 1 if TL > 0``.  Rest is
    ``TL <= FATIGUE_REST_THRESHOLD`` (0.01): a command of (nearly) zero, plus the
    negative commands of ``[-1, 1]`` control ranges, which MuJoCo's muscle dynamics
    clamp to zero excitation.  The threshold lets a command mapped through the
    muscle sigmoid, which never reaches exactly zero, rest.

    Args:
        mj_model: Compiled MuJoCo model.  Used to read muscle actuator
            time constants (tauact / taudeact) and to look up per-muscle
            fatigue parameters via :data:`MUSCLE_FATIGUE_PARAMS`.
        frame_skip: Number of physics steps per control step.  Together
            with ``mj_model.opt.timestep`` this defines the stored ``_dt``
            used when *dt* is not provided to :meth:`compute_act`.
        sex: Optional sex specifier (``"F"`` or ``"M"``) selects
            sex-specific rows in :data:`MUSCLE_FATIGUE_PARAMS`.
        seed: Random seed for stochastic resets via
            :meth:`reset(fatigue_reset_random=True)` when no ``np_random``
            generator is passed to it.
        use_uniform_params: If ``True``, bypass per-muscle lookup and use
            the ``"Default_v2_4"`` row of :data:`MUSCLE_FATIGUE_PARAMS` for
            all muscles: the single F/R/r of the legacy (v2.x) model, which the
            JAX models in ``physics/fatigue_jax.py`` also use.
        muscles: Muscle actuators to track (ids or names), in the order the
            excitations are passed; default every muscle in model order.

    Example:
        >>> fatigue = CumulativeFatigue(mj_model, frame_skip=5)
        >>> ma, _, _ = fatigue.compute_act(excitation)
        >>> # or use the convenience wrapper:
        >>> eff = fatigue.step(excitation, dt=0.01)
    """

    def __init__(
        self,
        mj_model: Any,
        frame_skip: int = 1,
        sex: str | None = None,
        seed: int | None = None,
        use_uniform_params: bool = False,
        muscles: Sequence[int | str] | None = None,
    ) -> None:
        self._dt = float(mj_model.opt.timestep) * int(frame_skip)
        self.muscle_ids = _muscle_ids(mj_model, muscles)  # actuator ids, state order
        self._tauact, self._taudeact = _muscle_time_constants(mj_model, self.muscle_ids)
        self.na: int = int(self._tauact.size)
        self._MA = np.zeros(self.na)
        self._MR = np.ones(self.na)
        self._MF = np.zeros(self.na)
        self.TL = np.zeros(self.na)

        if use_uniform_params:
            self._F = _UNIFORM_PARAMS["F"] * np.ones(self.na)
            self._R = _UNIFORM_PARAMS["R"] * np.ones(self.na)
            self._r = _UNIFORM_PARAMS["r"] * np.ones(self.na)
        else:
            self._F, self._R, self._r = _per_muscle_params(
                mj_model, sex, self.muscle_ids
            )

        self.seed(seed)

    # ------------------------------------------------------------------
    # Pipeline interface
    # ------------------------------------------------------------------

    def step(self, excitation: np.ndarray, dt: float | None = None) -> np.ndarray:
        """One control-step wrapper: returns effective muscle activation (MA).

        This is the interface used by ``ModularTaskEnv`` and other pipeline
        callers that just want the effective excitation after fatigue.

        Args:
            excitation: Commanded muscle excitations, shape ``(na,)``.
            dt: Control timestep in seconds.  If ``None``, uses ``self._dt``.

        Returns:
            Effective excitation (= MA after update), shape ``(na,)``.
        """
        ma, _, _ = self.compute_act(excitation, dt=dt)
        return ma

    # ------------------------------------------------------------------
    # Core 3CC-r dynamics
    # ------------------------------------------------------------------

    def compute_act(
        self, act: np.ndarray, dt: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance fatigue state by one step and return updated compartments.

        The step is exact for any ``dt`` under a constant command (:func:`three_cc_r_step`).

        Args:
            act: Target activation (commanded excitation), shape ``(na,)``.
            dt: Timestep in seconds.  Defaults to ``self._dt``.

        Returns:
            ``(MA, MR, MF)`` — updated active, resting, and fatigued
            compartments, each shape ``(na,)``.
        """
        _dt = dt if dt is not None else self._dt
        self.TL = act.copy()
        MA, MR, MF = three_cc_r_step(
            np,
            self._MA,
            self._MF,
            self.TL,
            F=self._F,
            R=self._R,
            r=self._r,
            tauact=self._tauact,
            taudeact=self._taudeact,
            dt=_dt,
        )
        self._MA[:], self._MR[:], self._MF[:] = MA, MR, MF
        return self._MA, self._MR, self._MF

    def get_effort(self) -> float:
        """Return the effort ``||MA - TL||`` of the last :meth:`compute_act` call.

        Same as the legacy model and the JAX ``get_effort``: how far the active
        compartment falls short of (or exceeds) the commanded target load.
        """
        return float(np.linalg.norm(self._MA - self.TL))

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        fatigue_reset_vec: np.ndarray | None = None,
        fatigue_reset_random: bool = False,
        np_random: np.random.Generator | None = None,
    ) -> None:
        """Reset fatigue state.

        Args:
            fatigue_reset_vec: If provided, set MF to this vector and
                MR = 1 - fatigue_reset_vec. Shape ``(na,)``. The vector is copied.
            fatigue_reset_random: If ``True``, sample a random fatigue
                state.  Cannot be combined with *fatigue_reset_vec*.
            np_random: Generator for the random state, typically the env's
                seeded ``np_random`` so that ``env.reset(seed=...)`` is
                reproducible.  Defaults to the model's own RNG (see
                :meth:`seed`).
        """
        if fatigue_reset_random:
            if fatigue_reset_vec is not None:
                raise ValueError(
                    "Cannot pass fatigue_reset_vec when fatigue_reset_random=True."
                )
            rng = self.np_random if np_random is None else np_random
            nf = rng.random(size=(self.na,))
            ap = rng.random(size=(self.na,))
            # Write in place. Rebinding ``1.0 - nf`` is typed as an N-D array
            # union, which mypy rejects for these 1-D buffers.
            self._MA[:] = nf * ap
            self._MR[:] = nf * (1.0 - ap)
            self._MF[:] = 1.0 - nf
        elif fatigue_reset_vec is not None:
            if len(fatigue_reset_vec) != self.na:
                raise ValueError(
                    f"fatigue_reset_vec length {len(fatigue_reset_vec)} != na={self.na}"
                )
            self._MF = np.array(fatigue_reset_vec, dtype=float, copy=True)  # type: ignore[assignment]
            self._MR = 1.0 - self._MF  # type: ignore[assignment]
            self._MA = np.zeros(self.na)
        else:
            self._MA[:] = 0.0
            self._MR[:] = 1.0
            self._MF[:] = 0.0

    def state_dict(self) -> dict[str, list]:
        """Return serialisable snapshot of the fatigue compartments.

        Returns:
            Dict with keys ``"MA"``, ``"MR"``, ``"MF"``, each a flat list of
            length ``na``.
        """
        return {
            "MA": self._MA.tolist(),
            "MR": self._MR.tolist(),
            "MF": self._MF.tolist(),
        }

    def load_state_dict(self, state: dict[str, list]) -> None:
        """Restore fatigue compartments from a :meth:`state_dict` snapshot.

        Args:
            state: Dict produced by :meth:`state_dict`.
        """
        self._MA = np.array(state["MA"], dtype=float)  # type: ignore[assignment]
        self._MR = np.array(state["MR"], dtype=float)  # type: ignore[assignment]
        self._MF = np.array(state["MF"], dtype=float)  # type: ignore[assignment]

    def seed(self, seed: int | None = None) -> list[int]:
        """Set random seed used by stochastic reset."""
        from myosuite.utils import gym  # noqa: PLC0415

        self.input_seed = seed
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]  # type: ignore[list-item]

    # ------------------------------------------------------------------
    # Properties (legacy compatibility)
    # ------------------------------------------------------------------

    @property
    def n_muscles(self) -> int:
        return self.na

    @property
    def MA(self) -> np.ndarray:
        return self._MA

    @property
    def MF(self) -> np.ndarray:
        return self._MF

    @property
    def MR(self) -> np.ndarray:
        return self._MR

    @property
    def F(self) -> np.ndarray:
        return self._F

    @property
    def R(self) -> np.ndarray:
        return self._R

    @property
    def r(self) -> np.ndarray:
        return self._r

    # ------------------------------------------------------------------
    # Legacy coefficient setters
    # ------------------------------------------------------------------

    def set_FatigueCoefficient(self, F: float | np.ndarray) -> None:
        self._F = F * np.ones(self.na) if np.isscalar(F) else np.asarray(F)  # type: ignore[assignment, operator]

    def set_RecoveryCoefficient(self, R: float | np.ndarray) -> None:
        self._R = R * np.ones(self.na) if np.isscalar(R) else np.asarray(R)  # type: ignore[assignment, operator]

    def set_RecoveryMultiplier(self, r: float | np.ndarray) -> None:
        self._r = r * np.ones(self.na) if np.isscalar(r) else np.asarray(r)  # type: ignore[assignment, operator]


# ---------------------------------------------------------------------------
# TorchFatigueState — batched 3CC-r model (torch, mjlab)
# ---------------------------------------------------------------------------


class TorchFatigueState:
    """Batched 3CC-r cumulative fatigue state for torch action terms.

    Mirrors :class:`CumulativeFatigue` but operates on torch tensors of shape
    ``(num_envs, n_muscles)`` so it can be used directly in mjlab action terms
    without looping over environments.  Uses the same per-muscle
    :data:`MUSCLE_FATIGUE_PARAMS` and recovery-multiplier dynamics as the
    numpy class.

    Prefer :meth:`from_mj_model` when a MuJoCo model is available; the plain
    constructor accepts scalar or per-muscle-array F / R / r.

    Args:
        num_envs: Number of parallel environments.
        n_muscles: Number of muscle actuators.
        device: Torch device string (e.g. ``"cpu"``, ``"cuda:0"``).
        F: Fatigue rate — scalar or array of length *n_muscles*.
        R: Recovery rate — scalar or array of length *n_muscles*.
        r: Recovery multiplier, applied only at rest (``TL <= FATIGUE_REST_THRESHOLD``, Rakshit
            et al. 2021, Eq. 7; see :class:`CumulativeFatigue`) — scalar or array.
    """

    def __init__(
        self,
        num_envs: int,
        n_muscles: int,
        device: str = "cpu",
        F: float | np.ndarray = _DEFAULT_F,
        R: float | np.ndarray = _DEFAULT_R,
        r: float | np.ndarray = _DEFAULT_r,
        tauact: float | np.ndarray = 0.01,
        taudeact: float | np.ndarray = 0.04,
    ) -> None:
        import torch  # noqa: PLC0415

        def _to_tensor(v: float | np.ndarray) -> Any:
            arr = np.broadcast_to(np.asarray(v, dtype=np.float32), (n_muscles,)).copy()
            return torch.tensor(arr, device=device)

        self._F: Any = _to_tensor(F)  # (n_muscles,)
        self._R: Any = _to_tensor(R)  # (n_muscles,)
        self._r: Any = _to_tensor(r)  # (n_muscles,)
        # The state stays float32 whatever the dtype of the excitation.
        shape = (num_envs, n_muscles)
        self.MA: Any = torch.zeros(shape, dtype=torch.float32, device=device)
        self.MF: Any = torch.zeros(shape, dtype=torch.float32, device=device)
        self.MR: Any = torch.ones(shape, dtype=torch.float32, device=device)
        self._tauact: Any = _to_tensor(tauact)  # (n_muscles,)
        self._taudeact: Any = _to_tensor(taudeact)  # (n_muscles,)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_mj_model(
        cls,
        mj_model: Any,
        num_envs: int,
        device: str = "cpu",
        sex: str | None = None,
        use_uniform_params: bool = False,
        muscles: Sequence[int | str] | None = None,
    ) -> TorchFatigueState:
        """Build a :class:`TorchFatigueState` from a compiled MuJoCo model.

        Extracts per-muscle F / R / r from :data:`MUSCLE_FATIGUE_PARAMS` via
        the muscle functional group (MFG) lookup, and the muscle time constants
        from ``actuator_dynprm`` — the same logic used by
        :class:`CumulativeFatigue`.

        Args:
            mj_model: Compiled MuJoCo model (``mujoco.MjModel``).
            num_envs: Number of parallel environments.
            device: Torch device string.
            sex: Optional ``"F"`` / ``"M"`` for sex-specific parameters.
            use_uniform_params: Use the uniform ``"Default_v2_4"`` F/R/r for
                every muscle, as :class:`CumulativeFatigue` does.
            muscles: Muscle actuators to track (ids or names), in column order;
                default every muscle in model order.
        """
        ids = _muscle_ids(mj_model, muscles)
        tauact, taudeact = _muscle_time_constants(mj_model, ids)
        na = int(tauact.size)
        if na == 0:
            return cls(num_envs=num_envs, n_muscles=mj_model.nu, device=device)
        if use_uniform_params:
            F_arr = np.full(na, _UNIFORM_PARAMS["F"])
            R_arr = np.full(na, _UNIFORM_PARAMS["R"])
            r_arr = np.full(na, _UNIFORM_PARAMS["r"])
        else:
            F_arr, R_arr, r_arr = _per_muscle_params(mj_model, sex, ids)
        return cls(
            num_envs=num_envs,
            n_muscles=na,
            device=device,
            F=F_arr,
            R=R_arr,
            r=r_arr,
            tauact=tauact,
            taudeact=taudeact,
        )

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, excitation: Any, dt: float) -> Any:
        """Advance fatigue state by one control step.

        Same exact step as :meth:`CumulativeFatigue.compute_act` (:func:`three_cc_r_step`).

        Args:
            excitation: Muscle excitation tensor, shape ``(num_envs, n_muscles)``;
                cast to the float32 state.
            dt: Control timestep in seconds.

        Returns:
            Active compartment ``MA`` (the effective muscle ctrl), same shape as
            *excitation*.
        """
        excitation = excitation.to(self.MA.dtype)
        import torch  # noqa: PLC0415

        self.MA, self.MR, self.MF = three_cc_r_step(
            torch,
            self.MA,
            self.MF,
            excitation,
            F=self._F,
            R=self._R,
            r=self._r,
            tauact=self._tauact,
            taudeact=self._taudeact,
            dt=dt,
        )
        # Same as CPU CumulativeFatigue: the active compartment is the ctrl.
        return self.MA

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        env_ids: Any = None,
        fatigue_reset_vec: Any = None,
        fatigue_reset_random: bool = False,
        generator: Any = None,
    ) -> None:
        """Reset the fatigue state of the given environments.

        Same three modes as :meth:`CumulativeFatigue.reset`, vectorised over
        the environments: fresh (``MA = MF = 0``, ``MR = 1``), a fixed fatigued
        fraction, or a random state drawn independently per env and muscle.

        Args:
            env_ids: Indices or slice of environments to reset.  If ``None``,
                resets all environments.
            fatigue_reset_vec: If provided, ``MF = fatigue_reset_vec``,
                ``MR = 1 - fatigue_reset_vec`` and ``MA = 0`` in every reset
                env.  Shape ``(n_muscles,)``.
            fatigue_reset_random: If ``True``, draw ``nf, ap ~ U[0, 1)`` and set
                ``MA = nf * ap``, ``MR = nf * (1 - ap)``, ``MF = 1 - nf``.
                Cannot be combined with *fatigue_reset_vec*.
            generator: ``torch.Generator`` for the random state.  Defaults to
                torch's global generator, which mjlab seeds
                (``env.reset(seed=...)``).

        Raises:
            ValueError: If both reset options are given, or the length of
                *fatigue_reset_vec* is not ``n_muscles``.
        """
        import torch  # noqa: PLC0415

        rows = slice(None) if env_ids is None else env_ids
        if fatigue_reset_random:
            if fatigue_reset_vec is not None:
                raise ValueError(
                    "Cannot pass fatigue_reset_vec when fatigue_reset_random=True."
                )
            shape = self.MA[rows].shape
            nf = torch.rand(shape, generator=generator, device=self.MA.device)
            ap = torch.rand(shape, generator=generator, device=self.MA.device)
            self.MA[rows] = nf * ap
            self.MR[rows] = nf * (1.0 - ap)
            self.MF[rows] = 1.0 - nf
        elif fatigue_reset_vec is not None:
            mf = torch.as_tensor(
                np.asarray(fatigue_reset_vec, dtype=np.float32), device=self.MA.device
            )
            if mf.shape != self.MA.shape[1:]:
                raise ValueError(
                    f"fatigue_reset_vec length {mf.numel()} != "
                    f"n_muscles={self.MA.shape[1]}"
                )
            self.MA[rows] = 0.0
            self.MF[rows] = mf
            self.MR[rows] = 1.0 - mf
        else:
            self.MA[rows] = 0.0
            self.MF[rows] = 0.0
            self.MR[rows] = 1.0

    def state_dict(self) -> dict[str, list]:
        """Return serialisable snapshot of the fatigue compartments.

        Returns:
            Dict with keys ``"MA"``, ``"MR"``, ``"MF"``, each a nested list
            of shape ``(num_envs, n_muscles)``.
        """
        return {
            "MA": self.MA.cpu().numpy().tolist(),
            "MR": self.MR.cpu().numpy().tolist(),
            "MF": self.MF.cpu().numpy().tolist(),
        }

    def load_state_dict(self, state: dict[str, list]) -> None:
        """Restore fatigue compartments from a :meth:`state_dict` snapshot.

        Args:
            state: Dict produced by :meth:`state_dict`.  Shape must be
                compatible with the current ``(num_envs, n_muscles)`` tensors.
        """
        import torch  # noqa: PLC0415

        self.MA.copy_(
            torch.tensor(state["MA"], dtype=torch.float32, device=self.MA.device)
        )
        self.MR.copy_(
            torch.tensor(state["MR"], dtype=torch.float32, device=self.MR.device)
        )
        self.MF.copy_(
            torch.tensor(state["MF"], dtype=torch.float32, device=self.MF.device)
        )
