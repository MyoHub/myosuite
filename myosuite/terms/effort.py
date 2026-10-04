# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Effort and ergonomics term functions (opt-in; no env uses them by default).

Pure, backend-agnostic terms built on the :class:`~myosuite.core.protocols.EnvAccessor`
muscle and joint-space state, for HCI / ergonomics studies:

- :func:`muscle_mechanical_power`: muscle mechanical power ``F * v``.
- :func:`metabolic_energy_rate`: Umberger et al. (2003) muscle energetics
  (or the Umberger 2010 variant), as implemented by OpenSim.
- :func:`consumed_endurance`: shoulder Consumed Endurance (Hincapié-Ramos et
  al. 2014), with :func:`endurance_time` and :func:`consumed_endurance_episode`.
- :func:`fatigue_effort`: effort read from the 3CC-r fatigue state.
- :func:`joint_limit_discomfort`: smooth penalty near the joint limits.

Every reward term returns its named components plus ``dense`` (``-weight *``
the effort measure), ``solved`` and ``done``; leading axes are batch axes
(none on CPU, ``(N,)`` on mjlab).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from myosuite.physics.muscle import (
    active_force_length,
    normalized_fiber_length,
    passive_force,
)

if TYPE_CHECKING:
    from myosuite.core.protocols import EnvAccessor

# ── Umberger et al. (2003) energetics, as in OpenSim Umberger2010MuscleMetabolicsProbe ──
UMBERGER_SPECIFIC_TENSION: float = 0.25e6  # Pa; OpenSim default for the muscle mass
UMBERGER_DENSITY: float = (
    1059.7  # kg/m^3, mammalian muscle (Mendez & Keys 1960); OpenSim default
)
UMBERGER_AEROBIC_FACTOR: float = 1.5  # S: 1.5 mainly aerobic, 1.0 mainly anaerobic
UMBERGER_FAST_TWITCH_FRACTION: float = 0.5  # OpenSim default (ratio_slow_twitch = 0.5)
_AM_SLOW: float = 25.0  # W/kg, activation + maintenance heat of slow-twitch fibres
_AM_FAST_SLOPE: float = 128.0  # W/kg per unit fast-twitch fraction (1.28 W/kg per %FT)
_SHORTEN_SLOW: float = 100.0  # W/kg, slow-twitch shortening heat at v = vmax_ST
_SHORTEN_FAST: float = 153.0  # W/kg, fast-twitch shortening heat at v = vmax_FT
_VMAX_FAST_OVER_SLOW: float = 2.5  # vmax_FT / vmax_ST
_LENGTHEN_2003: float = 4.0  # alpha_L / alpha_S(ST), Umberger et al. 2003
_LENGTHEN_2010: float = 0.3  # alpha_L / alpha_S(ST), Umberger 2010
_MIN_HEAT_RATE: float = 1.0  # W/kg, minimum heat rate per muscle

# ── Consumed Endurance (Hincapié-Ramos et al. 2014, Eqs. 1-2) ──
_CE_THRESHOLD_PCT: float = 15.0  # % of max torque below which endurance is infinite
_CE_GAIN: float = 1236.5
_CE_EXPONENT: float = 0.618
_CE_OFFSET: float = 72.5
# Constant shoulder Max_Torque of the original CE as reported by Li et al. (2024):
# Tan et al.'s maximal shoulder force at the elbow (101.6 N male, 87.2 N female)
# times the upper-arm length, with the arm-weight torque removed.
CE_MAX_SHOULDER_TORQUE_MALE: float = 22.94  # N m
CE_MAX_SHOULDER_TORQUE_FEMALE: float = 18.57  # N m


def _as_array(xp: Any, value: Any, like: Any) -> Any:
    """*value* as an array of *like*'s backend, dtype and device."""
    if getattr(xp, "__name__", "") == "torch":
        return xp.as_tensor(value, dtype=like.dtype, device=like.device)
    return xp.asarray(value, dtype=like.dtype)


def _effort_dict(
    xp: Any, measure: Any, weight: float, **components: Any
) -> dict[str, Any]:
    """Reward dict with ``dense = -weight * measure`` and no solved/done signal."""
    return {
        **components,
        "dense": -weight * measure,
        "solved": xp.zeros_like(measure, dtype=bool),
        "done": xp.zeros_like(measure, dtype=bool),
    }


def muscle_mechanical_power(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    mode: str = "abs",
    weight: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Mechanical power of the muscles, ``P_i = F_i * v_i``.

    ``F_i`` is the MuJoCo muscle force (tension < 0) and ``v_i`` the fiber
    velocity (MuJoCo muscles have rigid tendons, so it equals the actuator
    velocity, + = lengthening), so ``P_i > 0`` while a muscle shortens under
    tension (concentric, positive work on the skeleton) and ``P_i < 0`` while it
    is stretched (eccentric, negative work). ``mode`` selects the effort
    measure: ``"abs"`` sums ``|P_i|`` (the "absolute work" cost of Berret et
    al. 2011, PLoS Comput Biol 7(10):e1002183, at muscle level, counting
    concentric and eccentric work alike); ``"positive"`` sums ``max(P_i, 0)``
    (positive work only; Margaria 1968, Int Z angew Physiol 25:339-351, found
    negative work about five times cheaper metabolically).

    Args:
        accessor: Environment state accessor.
        task_state: Unused; present for uniform call signature.
        mode: ``"abs"`` or ``"positive"``.
        weight: Scale of the ``dense`` penalty.
        **kwargs: Unused extra keyword arguments.

    Returns:
        Dict with ``muscle_power_abs``, ``muscle_power_positive`` and
        ``muscle_power_net`` (W, summed over muscles), ``dense``, ``solved``,
        ``done``.

    Raises:
        ValueError: If *mode* is unknown.
    """
    if mode not in ("abs", "positive"):
        raise ValueError(f"mode must be 'abs' or 'positive', got {mode!r}")
    xp = accessor.array_module()
    power = accessor.muscle_force() * accessor.muscle_velocity()
    p_abs = xp.sum(xp.abs(power), axis=-1)
    p_pos = xp.sum(xp.clip(power, 0.0, None), axis=-1)
    return _effort_dict(
        xp,
        p_abs if mode == "abs" else p_pos,
        weight,
        muscle_power_abs=p_abs,
        muscle_power_positive=p_pos,
        muscle_power_net=xp.sum(power, axis=-1),
    )


def metabolic_energy_rate(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    version: str = "2003",
    fast_twitch_fraction: Any = UMBERGER_FAST_TWITCH_FRACTION,
    specific_tension: float = UMBERGER_SPECIFIC_TENSION,
    density: float = UMBERGER_DENSITY,
    aerobic_factor: float = UMBERGER_AEROBIC_FACTOR,
    vmax_fast: Any = None,
    enforce_min_heat_rate: bool = True,
    forbid_negative_total_rate: bool = True,
    weight: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Muscle metabolic energy rate of Umberger, Gerritsen & Martin (2003).

    Per muscle (rates in W/kg; Umberger et al. 2003, Comput Methods Biomech
    Biomed Engin 6(2):99-111), as in the reference implementation
    ``OpenSim::Umberger2010MuscleMetabolicsProbe`` (Uchida et al. 2016, PLoS
    One 11(3):e0150378) without fiber recruitment or basal rate:

    - ``A = u`` if ``u > a`` else ``(u + a) / 2`` (excitation ``u``, activation ``a``);
    - activation + maintenance: ``h_AM = S A^0.6 (128 f_FT + 25)``, scaled by
      ``0.4 + 0.6 F_iso`` when ``L~ > 1``;
    - shortening (``v~ <= 0``): ``h_SL = S A^2 [min(-a_ST v~, 100)(1 - f_FT) - a_FT v~ f_FT]``
      with ``a_ST = 100 / vmax_ST``, ``a_FT = 153 / vmax_FT``, ``vmax_ST = vmax_FT / 2.5``;
      lengthening (``v~ > 0``): ``h_SL = S A a_L v~`` with ``a_L = 4 a_ST`` (2003)
      or ``0.3 a_ST`` (Umberger 2010, J R Soc Interface 7:1329-1340); both
      scaled by ``F_iso`` when ``L~ > 1``;
    - work: ``w = -F_CE v / m`` (2003: positive and negative; 2010: ``v <= 0`` only);
    - a negative total raises ``h_SL`` to make it zero, then the heat
      ``h_AM + h_SL`` is at least 1 W/kg (OpenSim defaults);
    - ``E_i = m_i (h_AM + h_SL + w)``, ``E = sum_i E_i`` (W).

    MuJoCo assumptions: fiber length and velocity from the rigid-tendon muscle
    (:mod:`myosuite.physics.muscle`, ``v~ = v / L0``); ``F_iso`` is MuJoCo's
    active force-length curve; the active fiber force ``F_CE`` is the muscle
    force minus MuJoCo's passive force; ``vmax_FT`` is the muscle's ``vmax``
    (MyoSuite: 10-15 L0/s) unless *vmax_fast* is given; the muscle mass is
    ``m = F0 / specific_tension * density * L0``, so the heat terms scale with
    ``1 / specific_tension``: match it to the model's forces (the default
    0.25 MPa gives the MyoSuite leg 45 kg of muscle; the 0.6 MPa of Rajagopal
    et al. 2016, whose forces it uses, gives about 19 kg); without
    ``task_state["muscle_excitation"]`` the excitation equals the activation
    (``A = a``; MuJoCo's activation lags excitation by tau_act = 10 ms /
    tau_deact = 40 ms in MyoSuite models).

    Args:
        accessor: Environment state accessor.
        task_state: Optional ``"muscle_excitation"`` (muscle order, ``u`` in 0..1).
        version: ``"2003"`` (Umberger et al. 2003) or ``"2010"`` (Umberger 2010).
        fast_twitch_fraction: ``f_FT`` in 0..1, scalar or per muscle.
        specific_tension: Pa, for the muscle mass.
        density: kg/m^3, for the muscle mass.
        aerobic_factor: ``S`` (1.5 aerobic, 1.0 anaerobic).
        vmax_fast: Fast-twitch maximal shortening velocity (L0/s), scalar or per
            muscle; default: the muscle's ``vmax``.
        enforce_min_heat_rate: Keep the heat rate of every muscle >= 1 W/kg.
        forbid_negative_total_rate: Clamp a negative muscle total to zero.
        weight: Scale of the ``dense`` penalty.
        **kwargs: Unused extra keyword arguments.

    Returns:
        Dict with ``metabolic_rate`` (W), ``metabolic_heat_rate`` (W),
        ``metabolic_work_rate`` (W), ``metabolic_rate_per_muscle`` (W, muscle
        axis last), ``dense``, ``solved``, ``done``.

    Raises:
        ValueError: If *version* is unknown.
    """
    if version not in ("2003", "2010"):
        raise ValueError(f"version must be '2003' or '2010', got {version!r}")
    xp = accessor.array_module()
    p = accessor.muscle_params()
    force = accessor.muscle_force()
    vel = accessor.muscle_velocity()
    mtu_length = accessor.muscle_length()
    act = accessor.muscle_act()[..., p.act_ids]
    exc = task_state.get("muscle_excitation") if task_state else None
    exc = act if exc is None else _as_array(xp, exc, act)

    f_ft = _as_array(xp, fast_twitch_fraction, force)
    s = aerobic_factor
    a_eff = xp.where(exc > act, exc, 0.5 * (exc + act))
    length = normalized_fiber_length(mtu_length, p)
    v_norm = vel / p.optimal_length
    f_iso = active_force_length(length, p.lmin, p.lmax, xp)
    stretched = length > 1.0
    mass = p.peak_force / specific_tension * density * p.optimal_length

    # Activation + maintenance heat (W/kg).
    am = _AM_FAST_SLOPE * f_ft + _AM_SLOW
    am = s * a_eff**0.6 * xp.where(stretched, am * (0.4 + 0.6 * f_iso), am * 1.0)

    # Shortening / lengthening heat (W/kg).
    v_fast = p.vmax if vmax_fast is None else _as_array(xp, vmax_fast, force)
    alpha_fast = _SHORTEN_FAST / v_fast
    alpha_slow = _SHORTEN_SLOW / (v_fast / _VMAX_FAST_OVER_SLOW)
    slow_part = xp.clip(-alpha_slow * v_norm, None, _SHORTEN_SLOW)
    shorten = s * a_eff**2 * (slow_part * (1.0 - f_ft) - alpha_fast * v_norm * f_ft)
    ratio = _LENGTHEN_2003 if version == "2003" else _LENGTHEN_2010
    lengthen = s * a_eff * ratio * alpha_slow * v_norm
    sl = xp.where(v_norm <= 0.0, shorten, lengthen)
    sl = xp.where(stretched, sl * f_iso, sl)

    # Contractile-element work rate (W/kg): active tension times shortening speed.
    active_tension = xp.clip(-force - passive_force(mtu_length, p, xp), 0.0, None)
    work = -active_tension * vel / mass
    if version == "2010":
        work = xp.where(vel <= 0.0, work, xp.zeros_like(work))

    if forbid_negative_total_rate:
        total = am + sl + work
        sl = xp.where(total < 0.0, sl - total, sl)
    heat = am + sl
    if enforce_min_heat_rate:
        heat = xp.clip(heat, _MIN_HEAT_RATE, None)

    per_muscle = mass * (heat + work)
    rate = xp.sum(per_muscle, axis=-1)
    return _effort_dict(
        xp,
        rate,
        weight,
        metabolic_rate=rate,
        metabolic_heat_rate=xp.sum(mass * heat, axis=-1),
        metabolic_work_rate=xp.sum(mass * work, axis=-1),
        metabolic_rate_per_muscle=per_muscle,
    )


def endurance_time(strength: Any, xp: Any) -> Any:
    """Rohmert-type endurance time of Consumed Endurance (Hincapié-Ramos et al. 2014).

    Rohmert's endurance (Eq. 1 of Hincapié-Ramos, Guo, Moghadasian & Irani,
    "Consumed Endurance: a metric to quantify arm fatigue of mid-air
    interactions", CHI 2014, doi:10.1145/2556288.2557130), written for the
    shoulder torque (their Eq. 2; constants as restated by Li et al. 2024)::

        E = 1236.5 / (Torque / Max_Torque * 100 - 15) ** 0.618 - 72.5   [s]

    Contractions at or below 15 % of the maximum can be held indefinitely
    (``E = inf``; Eq. 1 is asymptotic at 15 %). The strength is clipped to 1,
    where ``E`` is about 6.9 s (the formula turns negative above ~113 %).

    Args:
        strength: ``Torque / Max_Torque`` (fraction, not percent).
        xp: Array module.

    Returns:
        Endurance time in seconds, same shape as *strength*.
    """
    pct = 100.0 * xp.clip(strength, 0.0, 1.0)
    excess = xp.clip(pct - _CE_THRESHOLD_PCT, 1e-6, None)
    endurance = _CE_GAIN / excess**_CE_EXPONENT - _CE_OFFSET
    return xp.where(
        pct > _CE_THRESHOLD_PCT, endurance, xp.full_like(endurance, math.inf)
    )


def consumed_endurance(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    shoulder_dof_ids: Any,
    max_shoulder_torque: float = CE_MAX_SHOULDER_TORQUE_MALE,
    weight: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Per-step Consumed Endurance of the shoulder (Hincapié-Ramos et al. 2014).

    The shoulder torque is the norm of the actuator generalized force
    ``qfrc_actuator`` (the net muscle moment) at *shoulder_dof_ids*: the torque
    the shoulder muscles exert, which balances gravity and the arm's inertia as
    in the paper's Eq. 6. For non-orthogonal shoulder coordinates (e.g. the
    elevation plane / elevation / rotation of the MyoSuite arm) the norm only
    approximates the magnitude of the 3-D torque; pass only the elevation dof
    for a single-axis measure.

    ``ce_step = 100 * dt / E(strength_t)`` is the percent of endurance spent
    this step; summed over an episode it is a Miner's-rule accumulation, not the
    paper's metric, which divides the interaction time by the endurance at the
    *average* torque: use :func:`consumed_endurance_episode` on the
    ``shoulder_strength`` history for that (the CE workbench reports it online
    from running averages).

    Args:
        accessor: Environment state accessor.
        task_state: Unused; present for uniform call signature.
        shoulder_dof_ids: Indices of the shoulder dofs in ``joint_vel()`` layout.
        max_shoulder_torque: ``Max_Torque`` (N m); default: the male constant of
            the original CE (22.94 N m, female 18.57 N m, as reported by Li et
            al. 2024, ACM TOCHI, doi:10.1145/3658230).
        weight: Scale of the ``dense`` penalty (on ``ce_step``).
        **kwargs: Unused extra keyword arguments.

    Returns:
        Dict with ``shoulder_torque`` (N m), ``shoulder_strength`` (fraction),
        ``endurance_time`` (s), ``ce_step`` (%), ``dense``, ``solved``, ``done``.
    """
    xp = accessor.array_module()
    torque = xp.linalg.norm(accessor.qfrc_actuator()[..., shoulder_dof_ids], axis=-1)
    strength = torque / max_shoulder_torque
    endurance = endurance_time(strength, xp)
    ce_step = 100.0 * accessor.dt() / endurance
    return _effort_dict(
        xp,
        ce_step,
        weight,
        shoulder_torque=torque,
        shoulder_strength=strength,
        endurance_time=endurance,
        ce_step=ce_step,
    )


def consumed_endurance_episode(
    shoulder_strength: Any, dt: float, xp: Any
) -> dict[str, Any]:
    """Consumed Endurance of an interaction (Hincapié-Ramos et al. 2014, Eq. 7).

    ``CE = interaction_time / E(mean strength) * 100``, with the strength
    ``S = average torque / Max_Torque`` and ``E`` from :func:`endurance_time`.

    Args:
        shoulder_strength: Per-step ``shoulder_strength`` from
            :func:`consumed_endurance`, time on axis 0 (``(T,)`` or ``(T, N)``).
        dt: Control timestep (s).
        xp: Array module.

    Returns:
        Dict with ``mean_strength``, ``endurance_time`` (s),
        ``interaction_time`` (s) and ``consumed_endurance`` (%).
    """
    mean_strength = xp.mean(shoulder_strength, axis=0)
    endurance = endurance_time(mean_strength, xp)
    interaction_time = shoulder_strength.shape[0] * dt
    return {
        "mean_strength": mean_strength,
        "endurance_time": endurance,
        "interaction_time": interaction_time,
        "consumed_endurance": 100.0 * interaction_time / endurance,
    }


def fatigue_effort(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    weight: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Effort from the 3CC-r fatigue state (Xia & Frey-Law 2008; Looft et al. 2018).

    The three-compartment model splits each muscle's motor units into active
    (``MA``), resting (``MR``) and fatigued (``MF``) fractions (Xia & Frey-Law
    2008, J Biomech 41:3046-3052; recovery multiplier ``r``: Looft, Herkert &
    Frey-Law 2018, J Biomech 77:16-25). The fatigued fraction is the lost
    capacity (residual capacity ``1 - MF``; Frey-Law et al. 2012, Med Sci
    Sports Exerc 44:1371-1382). ``fatigue_effort = ||MA - TL||`` is the shortfall
    of the active compartment from the commanded target load, as
    :meth:`~myosuite.core.muscle_conditions.CumulativeFatigue.get_effort`.

    Args:
        accessor: Environment state accessor (array backend only).
        task_state: ``"fatigue"``: the fatigue state, an object with ``MA`` /
            ``MF`` arrays (CPU ``env.muscle_fatigue``, a ``CumulativeFatigue``;
            mjlab ``MyoAction.fatigue_state``, a ``TorchFatigueState``).
            Optional ``"fatigue_target"``: the target load ``TL`` (default: the
            state's ``TL`` attribute, if any).
        weight: Scale of the ``dense`` penalty (on the mean ``MF``).
        **kwargs: Unused extra keyword arguments.

    Returns:
        Dict with ``fatigue_mf`` (mean MF), ``fatigue_mf_max``, ``fatigue_effort``
        (only when a target load is known), ``dense``, ``solved``, ``done``.
    """
    xp = accessor.array_module()
    state = task_state["fatigue"]
    mf = state.MF
    out: dict[str, Any] = {
        "fatigue_mf": xp.mean(mf, axis=-1),
        "fatigue_mf_max": xp.amax(mf, axis=-1),
    }
    target = task_state.get("fatigue_target", getattr(state, "TL", None))
    if target is not None:
        out["fatigue_effort"] = xp.linalg.norm(
            state.MA - _as_array(xp, target, state.MA), axis=-1
        )
    return _effort_dict(xp, out["fatigue_mf"], weight, **out)


def joint_limit_discomfort(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    margin: float = 0.1,
    weight: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Smooth (C1) discomfort that rises as joints approach their limits.

    With ``x`` the position of each limited hinge/slide joint as a fraction of
    its range, the discomfort is ``((margin - x)_+ / margin)^2 +
    ((x - 1 + margin)_+ / margin)^2``: zero in the inner ``1 - 2 margin`` of
    the range, 1 at a limit, averaged over joints. Joint-limit terms of this
    kind appear in posture-prediction discomfort functions (Marler et al.
    2005, SAE 2005-01-2680); this squared hinge is a simpler form, not theirs.
    Unlike :func:`~myosuite.terms.base_reward.joint_penalty` it is
    normalised by the range and has a continuous gradient.

    Args:
        accessor: Environment state accessor.
        task_state: Unused; present for uniform call signature.
        margin: Fraction of the range, at each end, where discomfort rises.
        weight: Scale of the ``dense`` penalty.
        **kwargs: Unused extra keyword arguments.

    Returns:
        Dict with ``joint_limit_discomfort`` (0 to 1 inside the range),
        ``dense``, ``solved``, ``done``.
    """
    xp = accessor.array_module()
    qpos_ids, ranges = accessor.joint_range()
    qpos = accessor.joint_pos()[..., qpos_ids]
    x = (qpos - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0])
    low = xp.clip(margin - x, 0.0, None) / margin
    high = xp.clip(x - (1.0 - margin), 0.0, None) / margin
    discomfort = xp.mean(low * low + high * high, axis=-1)
    return _effort_dict(xp, discomfort, weight, joint_limit_discomfort=discomfort)
