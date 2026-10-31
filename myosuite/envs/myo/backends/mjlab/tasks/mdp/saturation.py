# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Muscle saturation metric and opt-in penalty (MyoHub/myosuite#566).

A muscle command is saturated when its excitation is within a margin of 0 or 1,
or its action is clamped (:func:`~myosuite.terms.base_action.muscle_saturation_bounds`).
There the action map is flat, so the policy gradient through the action vanishes
and exploration only switches the muscle fully on or off. Action terms that drive
muscles expose ``muscle_action_bounds(margin)``; these terms read every such term.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from myosuite.terms.base_action import (
    MUSCLE_SATURATION_MARGIN,
    muscle_saturated,
    muscle_saturation_excess,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

# Key of the saturation metric and of the opt-in penalty reward.
SATURATION_TERM = "muscle_saturation"


def _muscle_actions(
    env: ManagerBasedRlEnv, margin: float
) -> list[tuple[torch.Tensor, Any, Any]]:
    """``muscle_action_bounds(margin)`` of every action term that drives muscles."""
    manager = env.action_manager
    terms = (manager.get_term(name) for name in manager.active_terms)
    parts = [
        term.muscle_action_bounds(margin)
        for term in terms
        if hasattr(term, "muscle_action_bounds")
    ]
    parts = [part for part in parts if part[0].shape[-1] > 0]
    if not parts:
        raise ValueError("No action term of this env reports muscle actions.")
    return parts


def muscle_saturation(
    env: ManagerBasedRlEnv, margin: float = MUSCLE_SATURATION_MARGIN
) -> torch.Tensor:
    """Fraction of muscle commands that are saturated (metric).

    During training the actions include the exploration noise, so this counts
    the commands the env receives; with the mean-action policy (evaluation) it
    is the saturation of the deterministic policy.

    Args:
        env: The environment.
        margin: Excitation distance from 0 or 1 that counts as saturated.

    Returns:
        Saturated fraction in ``[0, 1]``, shape ``(num_envs,)``.
    """
    masks = [muscle_saturated(a, lo, hi) for a, lo, hi in _muscle_actions(env, margin)]
    return torch.cat(masks, dim=-1).float().mean(dim=-1)


def muscle_saturation_l2(
    env: ManagerBasedRlEnv, margin: float = MUSCLE_SATURATION_MARGIN
) -> torch.Tensor:
    """Mean squared distance of the muscle actions beyond their saturation bounds.

    A reward term; give it a negative weight (:func:`add_muscle_saturation_penalty`).

    Args:
        env: The environment.
        margin: Excitation distance from 0 or 1 that counts as saturated.

    Returns:
        Penalty, shape ``(num_envs,)``.
    """
    excess = [
        muscle_saturation_excess(a, lo, hi, torch)
        for a, lo, hi in _muscle_actions(env, margin)
    ]
    return torch.cat(excess, dim=-1).square().mean(dim=-1)


def muscle_saturation_metric() -> MetricsTermCfg:
    """The ``muscle_saturation`` metric (logged as ``Episode_Metrics/muscle_saturation``)."""
    return MetricsTermCfg(func=muscle_saturation)


def add_muscle_saturation_penalty(
    env_cfg: ManagerBasedRlEnvCfg,
    penalty: float,
    margin: float = MUSCLE_SATURATION_MARGIN,
) -> None:
    """Add the opt-in saturation penalty to *env_cfg* (reward key ``muscle_saturation``).

    It changes the task's reward, so it is off in every registered task
    (``scripts/train_mjlab.py --saturation-penalty`` turns it on).

    Args:
        env_cfg: Env config, changed in place.
        penalty: Magnitude of the penalty (the term's weight is ``-penalty``).
        margin: Excitation distance from 0 or 1 that counts as saturated.

    Raises:
        ValueError: If *penalty* is negative or the reward key is taken.
    """
    if penalty < 0.0:
        raise ValueError(f"penalty must be >= 0, got {penalty}.")
    if SATURATION_TERM in env_cfg.rewards:
        raise ValueError(f"The env already has a {SATURATION_TERM!r} reward.")
    env_cfg.rewards[SATURATION_TERM] = RewardTermCfg(
        func=muscle_saturation_l2, weight=-penalty, params={"margin": margin}
    )
