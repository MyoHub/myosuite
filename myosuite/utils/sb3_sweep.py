# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Helpers for the SB3 all-environment learnability sweep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def is_excluded_env_id(env_id: str) -> bool:
    """Return True for fatigue / sarcopenia / reafferentation variants."""
    return env_id.startswith(("myoSarc", "myoFati", "myoReaf"))


def budget_for(act_dim: int, base: int, *, escalate: bool = False) -> int:
    """Scale PPO timesteps with action dimension.

    High-dimensional tasks (act > 80) escalate rather than shrink: the
    historical ``base // 2`` schedule left boxing/mimic/torso under-trained.

    Args:
        act_dim: Environment action dimension.
        base: Diagnostic timestep budget (typically 100_000).
        escalate: When True, raise remaining hard tasks to the 1–3M range.

    Returns:
        Integer timestep budget.
    """
    if act_dim <= 10:
        budget = base
    elif act_dim <= 40:
        budget = max(base, int(base * 1.5))
    elif act_dim <= 80:
        budget = max(base, int(base * 2.0))
    else:
        budget = max(base, 1_000_000)

    if escalate:
        if act_dim <= 10:
            budget = max(budget, 1_000_000)
        elif act_dim <= 40:
            budget = max(budget, 1_500_000)
        elif act_dim <= 80:
            budget = max(budget, 2_000_000)
        else:
            budget = max(budget, 3_000_000)
    return int(budget)


def improved(
    before: float,
    after: float,
    eps_abs: float = 0.5,
    eps_rel: float = 0.05,
) -> bool:
    """Legacy SB3 smoke pass rule.

    Pass when ``after - before > max(eps_abs, eps_rel * |before|)``.
    """
    delta = after - before
    thr = max(eps_abs, abs(before) * eps_rel)
    return delta > thr


def improvement_threshold(
    before: float,
    eps_abs: float = 0.5,
    eps_rel: float = 0.05,
) -> float:
    """Return the absolute Δ required to pass for a given baseline reward."""
    return max(eps_abs, abs(before) * eps_rel)


def parse_act_dim(probe: str) -> int:
    """Parse action dim from a probe string ``obs=(..) act=(N,)``."""
    return int(probe.split("act=(")[1].split(",")[0].rstrip(")"))


@dataclass(frozen=True)
class WinnerPpoConfig:
    """CPU PPO subset of Lattice Relocate / official 2025 challenge baselines.

    Omits RecurrentPPO, Lattice exploration, 250 GPU envs, imitation
    priors, and billion-step curricula. Keeps the pieces that still run
    on DummyVecEnv CPU: VecNormalize, a wider MLP, low LR, tiny entropy,
    GAE 0.9, and no reward-Δ early-stop.
    """

    learning_rate: float = 2e-5
    n_steps: int = 1024
    batch_size: int = 512
    n_epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.9
    clip_range: float = 0.2
    ent_coef: float = 1e-5
    max_grad_norm: float = 0.7
    net_arch: tuple[int, ...] = (256, 256)
    vec_normalize: bool = True
    no_early_stop: bool = True

    @classmethod
    def for_n_envs(cls, n_envs: int) -> WinnerPpoConfig:
        """Pick a rollout/batch size that divides ``n_envs * n_steps``."""
        # DummyVecEnv is sequential; short rollouts just multiply PPO updates.
        n_steps = 1024
        rollout = n_envs * n_steps
        batch_size = min(1024, rollout)
        while batch_size > 1 and rollout % batch_size != 0:
            batch_size //= 2
        return cls(n_steps=n_steps, batch_size=int(batch_size))


def solved_from_info(info: dict[str, Any]) -> bool:
    """Extract a boolean solved flag from a Gymnasium ``info`` dict."""
    solved = info.get("solved")
    if solved is None:
        rwd = info.get("rwd_dict")
        if isinstance(rwd, dict):
            solved = rwd.get("solved")
    if solved is None:
        return False
    try:
        import numpy as np

        return bool(np.asarray(solved).astype(bool).any())
    except Exception:  # noqa: BLE001
        return bool(solved)
