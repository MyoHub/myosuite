# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Muscle effort metrics of recorded rollouts.

Pure functions on recorded arrays, time-major with an optional leading batch:
activations, excitations or actions ``(..., T, M)``, muscle forces ``(..., T, M)``
and moment arms ``(..., T, M, D)`` or ``(M, D)``. NumPy arrays and torch tensors
are accepted; the result has the input's backend.

- :func:`squared_activation`: mean over time of ``sum_m a_m^2``.
- :func:`wasted_torque` and :func:`cocontraction_index`: torque that antagonists
  cancel, ``sum_m |tau_mj| - |sum_m tau_mj|`` per DOF ``j`` with
  ``tau_mj = moment_arm_mj * force_m``, the definition of the opt-in co-contraction
  cost (MyoHub/myosuite#607). Divided by ``sum_m |tau_mj|`` it is the
  co-contraction index; for one agonist-antagonist pair with equal moment arms it
  is ``2 min(F_1, F_2) / (F_1 + F_2)`` (Falconer & Winter, 1985).
- :func:`muscle_saturation`: fraction of muscle commands within a margin of 0 or 1
  excitation, or clamped, where the action map is flat and the policy gradient
  vanishes. The saturation metric is from #588 by Mark Colley (MyoHub/myosuite#566).
"""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np

# Excitation distance from 0 or 1 below which a muscle command counts as saturated.
MUSCLE_SATURATION_MARGIN: float = 0.05


def _array_module(x: Any) -> Any:
    """``torch`` for tensors, else ``numpy`` (never imports torch)."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.is_tensor(x):
        return torch
    return np


def _as_array(x: Any) -> Any:
    return x if _array_module(x) is not np else np.asarray(x)


def squared_activation(activations: Any) -> Any:
    """Mean over time of the summed squared activations, ``mean_t sum_m a_tm^2``.

    Args:
        activations: Muscle activations, shape ``(..., T, M)``.

    Returns:
        Effort, shape ``(...)``.
    """
    a = _as_array(activations)
    return (a * a).sum(-1).mean(-1)


def _muscle_torques(moment_arms: Any, forces: Any) -> tuple[Any, Any]:
    """Per DOF: summed absolute torque and absolute net torque, shape ``(..., D)``."""
    forces = _as_array(forces)
    xp = _array_module(forces)
    tau = _as_array(moment_arms) * forces[..., :, None]
    return xp.abs(tau).sum(-2), xp.abs(tau.sum(-2))


def wasted_torque(moment_arms: Any, forces: Any) -> Any:
    """Torque the muscles spend against each other, summed over DOFs.

    Args:
        moment_arms: Moment arms ``dL/dq`` (m for hinge DOFs), shape ``(..., M, D)``,
            broadcastable against *forces*.
        forces: Muscle forces in N, shape ``(..., M)``.

    Returns:
        ``sum_j (sum_m |tau_mj| - |sum_m tau_mj|)`` in N·m, shape ``(...)``.
    """
    total, net = _muscle_torques(moment_arms, forces)
    return (total - net).sum(-1)


def cocontraction_index(moment_arms: Any, forces: Any) -> Any:
    """Share of the muscle torque about each DOF that antagonists cancel.

    Torques are summed over time before the ratio, so steps with little muscle
    torque weigh little.

    Args:
        moment_arms: Moment arms, shape ``(..., T, M, D)`` or ``(M, D)``.
        forces: Muscle forces in N, shape ``(..., T, M)``.

    Returns:
        ``sum_t wasted_tj / sum_t sum_m |tau_tmj|`` in ``[0, 1]``, shape ``(..., D)``;
        0 for a DOF without muscle torque.
    """
    total, net = _muscle_torques(moment_arms, forces)
    total, net = total.sum(-2), net.sum(-2)
    return (total - net) / (total + (total == 0))


def muscle_saturation_bounds(
    *,
    sigmoid: bool,
    clip: tuple[float, float] | None = None,
    margin: float = MUSCLE_SATURATION_MARGIN,
) -> tuple[float, float]:
    """Policy-action bounds outside which a muscle command is saturated.

    A command is saturated when its excitation is within *margin* of 0 or 1, or
    when the action is clamped. MuJoCo clamps muscle ctrl to ``[0, 1]``, so an
    identity map (``sigmoid=False``) saturates at its bounds too.

    Args:
        sigmoid: The map is ``sigmoid_muscle_activation``
            (``1 / (1 + exp(-5 (a - 0.5)))``); otherwise the excitation is the action.
        clip: Clamp applied to the action before the map, if any.
        margin: Excitation distance from 0 or 1, in ``(0, 0.5)``.

    Returns:
        ``(lo, hi)``: actions ``<= lo`` or ``>= hi`` are saturated.

    Raises:
        ValueError: If *margin* is outside ``(0, 0.5)``.
    """
    if not 0.0 < margin < 0.5:
        raise ValueError(f"margin must be in (0, 0.5), got {margin}.")
    if sigmoid:
        half = math.log((1.0 - margin) / margin) / 5.0
        lo, hi = 0.5 - half, 0.5 + half
    else:
        lo, hi = margin, 1.0 - margin
    if clip is not None:
        lo, hi = max(lo, clip[0]), min(hi, clip[1])
    return lo, hi


def muscle_saturated(action: Any, lo: Any, hi: Any) -> Any:
    """Boolean mask of saturated muscle commands.

    Args:
        action: Policy actions of the muscles, any shape.
        lo: Lower bound, broadcastable to *action*.
        hi: Upper bound, broadcastable to *action*.

    Returns:
        ``(action <= lo) | (action >= hi)``.
    """
    action = _as_array(action)
    return (action <= lo) | (action >= hi)


def muscle_saturation(
    actions: Any,
    bounds: tuple[float, float] | None = None,
    margin: float = MUSCLE_SATURATION_MARGIN,
) -> Any:
    """Fraction of saturated muscle commands over time and muscles.

    Args:
        actions: Muscle commands, shape ``(..., T, M)``.
        bounds: ``(lo, hi)`` from :func:`muscle_saturation_bounds` for the action
            map; ``None`` treats *actions* as excitations (identity map).
        margin: Excitation margin used when *bounds* is ``None``.

    Returns:
        Saturated fraction in ``[0, 1]``, shape ``(...)``.
    """
    lo, hi = bounds or muscle_saturation_bounds(sigmoid=False, margin=margin)
    mask = muscle_saturated(actions, lo, hi)
    return (mask.reshape(*mask.shape[:-2], -1) * 1.0).mean(-1)


__all__ = [
    "MUSCLE_SATURATION_MARGIN",
    "cocontraction_index",
    "muscle_saturated",
    "muscle_saturation",
    "muscle_saturation_bounds",
    "squared_activation",
    "wasted_torque",
]
