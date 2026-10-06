# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""MuJoCo muscle parameters and Hill-type curves, shared by CPU and mjlab.

MuJoCo muscles (``gaintype="muscle"``) have a rigid tendon and no pennation,
so the fiber length is an affine function of the actuator (MTU) length and the
fiber velocity equals the actuator velocity (MuJoCo documentation, "Muscles";
``mju_muscleGain`` / ``mju_muscleBias`` in ``engine_util_misc.c``):

    L0 = (lengthrange[1] - lengthrange[0]) / (range[1] - range[0])   (optimal length, m)
    L~ = range[0] + (length - lengthrange[0]) / L0                    (normalised length)

The active force uses ``gainprm`` and the passive force ``biasprm``; some
MyoSuite models (the arm) give them different ``range`` / ``force`` / ``fpmax``.
The curve helpers take the array module ``xp`` (numpy or torch) so term
functions can call them on either backend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import mujoco
import numpy as np

from myosuite.core.muscle_conditions import _peak_force

_MJMINVAL = mujoco.mjMINVAL


@dataclass(frozen=True)
class MuscleParams:
    """Static parameters of the muscle actuators, one entry per muscle.

    Arrays are numpy on CPU and torch on mjlab; the order matches
    ``EnvAccessor.muscle_force()`` / ``muscle_length()`` / ``muscle_velocity()``.

    Args:
        peak_force: Peak isometric force F0 (N, ``gainprm``); ``force="-1"`` is
            resolved to ``scale / acc0`` as MuJoCo does at runtime.
        optimal_length: Optimal fiber length L0 (m).
        length_range_lo: Actuator length at ``range[0]``, ``lengthrange[0]`` (m).
        range_lo: Normalised fiber length at ``lengthrange[0]``, ``gainprm[0]``.
        lmin: Lower end of the active force-length curve (L0 units).
        lmax: Upper end of the active force-length curve (L0 units).
        vmax: Maximal shortening velocity (L0 / s).
        fvmax: Eccentric force plateau (F0 units).
        passive_force: Force scale of the passive curve (N, ``biasprm``); 0 when
            the actuator has no muscle bias.
        passive_optimal_length: L0 of the passive curve (m, ``biasprm`` range).
        passive_range_lo: Normalised length at ``lengthrange[0]``, ``biasprm[0]``.
        passive_lmax: ``biasprm[5]``, sets where the passive curve turns linear.
        fpmax: Passive force at ``passive_lmax`` (``passive_force`` units).
        act_ids: Index of each muscle in ``EnvAccessor.muscle_act()``.
    """

    peak_force: Any
    optimal_length: Any
    length_range_lo: Any
    range_lo: Any
    lmin: Any
    lmax: Any
    vmax: Any
    fvmax: Any
    passive_force: Any
    passive_optimal_length: Any
    passive_range_lo: Any
    passive_lmax: Any
    fpmax: Any
    act_ids: Any

    def map(self, fn: Callable[[str, Any], Any]) -> MuscleParams:
        """Return a copy with ``fn(name, value)`` applied to every field."""
        return MuscleParams(
            **{f.name: fn(f.name, getattr(self, f.name)) for f in fields(self)}
        )


def muscle_columns(mj_model: mujoco.MjModel, actuator_ids: Any = None) -> np.ndarray:
    """Positions in *actuator_ids* of the MuJoCo muscles (``gaintype`` muscle).

    Args:
        mj_model: Compiled model.
        actuator_ids: Actuator ids to filter (default: all, so positions are ids).

    Returns:
        Integer array, in ascending position order.
    """
    ids = np.arange(mj_model.nu) if actuator_ids is None else np.asarray(actuator_ids)
    gain = np.asarray(mj_model.actuator_gaintype)[ids.astype(int)]
    return np.flatnonzero(gain == mujoco.mjtGain.mjGAIN_MUSCLE)


def muscle_params_from_model(
    mj_model: mujoco.MjModel, actuator_ids: Any = None
) -> MuscleParams:
    """Read :class:`MuscleParams` of the muscles among *actuator_ids*.

    Args:
        mj_model: Compiled model.
        actuator_ids: Actuator ids to consider, in order (default: all).

    Returns:
        Numpy :class:`MuscleParams` in the order of :func:`muscle_columns`.
    """
    ids = np.arange(mj_model.nu) if actuator_ids is None else np.asarray(actuator_ids)
    ids = ids.astype(int)[muscle_columns(mj_model, ids)]
    lr = np.asarray(mj_model.actuator_lengthrange, dtype=np.float64)[ids]
    gain = np.asarray(mj_model.actuator_gainprm, dtype=np.float64)[ids]
    bias = np.asarray(mj_model.actuator_biasprm, dtype=np.float64)[ids]
    acc0 = np.asarray(mj_model.actuator_acc0, dtype=np.float64)[ids]
    # mju_muscleBias: a negative force means scale / acc0.
    passive_force = np.where(
        bias[:, 2] < 0, bias[:, 3] / np.maximum(acc0, _MJMINVAL), bias[:, 2]
    )
    has_bias = (
        np.asarray(mj_model.actuator_biastype)[ids] == mujoco.mjtBias.mjBIAS_MUSCLE
    )

    def optimal_length(prm: np.ndarray) -> np.ndarray:
        return (lr[:, 1] - lr[:, 0]) / np.maximum(prm[:, 1] - prm[:, 0], _MJMINVAL)

    return MuscleParams(
        peak_force=_peak_force(mj_model)[ids],
        optimal_length=optimal_length(gain),
        length_range_lo=lr[:, 0].copy(),
        range_lo=gain[:, 0].copy(),
        lmin=gain[:, 4].copy(),
        lmax=gain[:, 5].copy(),
        vmax=gain[:, 6].copy(),
        fvmax=gain[:, 8].copy(),
        passive_force=np.where(has_bias, passive_force, 0.0),
        passive_optimal_length=optimal_length(bias),
        passive_range_lo=bias[:, 0].copy(),
        passive_lmax=bias[:, 5].copy(),
        fpmax=bias[:, 7].copy(),
        act_ids=np.asarray(mj_model.actuator_actadr)[ids].astype(np.int64),
    )


def normalized_fiber_length(length: Any, params: MuscleParams) -> Any:
    """Normalised fiber length L~ (L0 units) from the actuator length (m)."""
    return params.range_lo + (length - params.length_range_lo) / params.optimal_length


def active_force_length(length: Any, lmin: Any, lmax: Any, xp: Any) -> Any:
    """MuJoCo active force-length multiplier ``mju_muscleGainLength`` (0..1).

    Four half-quadratics joined at ``lmin``, ``(lmin+1)/2``, 1, ``(1+lmax)/2``
    and ``lmax``; zero outside ``[lmin, lmax]``.

    Args:
        length: Normalised fiber length L~.
        lmin: Lower end of the curve.
        lmax: Upper end of the curve.
        xp: Array module.

    Returns:
        FL(L~), same shape as *length*.
    """
    a = 0.5 * (lmin + 1.0)
    b = 0.5 * (1.0 + lmax)
    x1 = (length - lmin) / xp.clip(a - lmin, _MJMINVAL, None)
    x2 = (1.0 - length) / xp.clip(1.0 - a, _MJMINVAL, None)
    x3 = (length - 1.0) / xp.clip(b - 1.0, _MJMINVAL, None)
    x4 = (lmax - length) / xp.clip(lmax - b, _MJMINVAL, None)
    fl = xp.where(
        length <= a,
        0.5 * x1 * x1,
        xp.where(
            length <= 1.0,
            1.0 - 0.5 * x2 * x2,
            xp.where(length <= b, 1.0 - 0.5 * x3 * x3, 0.5 * x4 * x4),
        ),
    )
    outside = (length < lmin) | (length > lmax)
    return xp.where(outside, xp.zeros_like(fl), fl)


def passive_force_length(length: Any, lmax: Any, fpmax: Any, xp: Any) -> Any:
    """MuJoCo passive force multiplier (``mju_muscleBias`` / force, positive).

    Zero below the optimal length, half-quadratic up to ``(1+lmax)/2``, then
    linear.

    Args:
        length: Normalised length L~ of the passive curve.
        lmax: ``biasprm[5]``.
        fpmax: Passive force at ``lmax`` (force units).
        xp: Array module.

    Returns:
        FP(L~) in force units, same shape as *length*.
    """
    b = 0.5 * (1.0 + lmax)
    width = xp.clip(b - 1.0, _MJMINVAL, None)
    x_quad = (length - 1.0) / width
    x_lin = (length - b) / width
    fp = xp.where(length <= b, fpmax * 0.5 * x_quad * x_quad, fpmax * (0.5 + x_lin))
    return xp.where(length <= 1.0, xp.zeros_like(fp), fp)


def passive_force(length: Any, params: MuscleParams, xp: Any) -> Any:
    """Passive muscle tension (N, positive) at the actuator length *length* (m)."""
    norm = (
        params.passive_range_lo
        + (length - params.length_range_lo) / params.passive_optimal_length
    )
    return params.passive_force * passive_force_length(
        norm, params.passive_lmax, params.fpmax, xp
    )
