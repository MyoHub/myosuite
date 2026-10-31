# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Frozen-opponent self-play for two-player mjlab twins.

A task's action term owns an :class:`OpponentPool` (attribute
``opponent_pool``) and drives its opponent with :meth:`OpponentPool.act` on an
``opponent`` observation group that mirrors the agent's (the actor must read
a single observation group, whose layout the opponent group repeats). During training,
:func:`install_self_play` adds a frozen copy of the learner to the pool every
``snapshot_every`` iterations; each env plays one snapshot per episode.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SelfPlayCfg:
    """Snapshot pool settings.

    Attributes:
        snapshot_every: Training iterations between snapshots of the learner.
        pool_size: Snapshots kept; the oldest is replaced when full.
        latest_prob: Chance that an episode plays the newest snapshot (else a
            uniformly drawn one), which keeps old strategies beaten.
        obs_group: Observation group the opponent's policy reads.
    """

    snapshot_every: int = 50
    pool_size: int = 8
    latest_prob: float = 0.5
    obs_group: str = "opponent"


class OpponentPool:
    """Frozen policy snapshots driving the opponent; one is drawn per env at reset.

    Every snapshot runs on every env and each env keeps its own snapshot's
    action, so a step has no host sync.

    Args:
        cfg: Pool settings.
        num_envs: Number of envs.
        device: Env device.
    """

    def __init__(
        self, cfg: SelfPlayCfg, num_envs: int, device: str | torch.device
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.policies: list[torch.nn.Module] = []
        self.latest = 0  # slot of the newest snapshot
        self.slot = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._envs = torch.arange(num_envs, device=device)

    def __len__(self) -> int:
        return len(self.policies)

    def add(self, policy: torch.nn.Module) -> None:
        """Store a frozen copy of *policy*, a module mapping flat observations to actions.

        An RSL-RL actor is copied through ``as_jit()`` (its deterministic export,
        normalizer included).
        """
        frozen = policy.as_jit() if hasattr(policy, "as_jit") else copy.deepcopy(policy)
        frozen = frozen.to(self.device).eval()
        frozen.requires_grad_(False)
        if len(self.policies) < self.cfg.pool_size:
            self.policies.append(frozen)
            self.latest = len(self.policies) - 1
        else:
            self.latest = (self.latest + 1) % self.cfg.pool_size
            self.policies[self.latest] = frozen

    def reset(self, env_ids: torch.Tensor | slice) -> None:
        """Draw the snapshot each of *env_ids* plays this episode."""
        n = self._envs[env_ids].shape[0]
        if not self.policies or n == 0:
            return
        uniform = torch.randint(len(self.policies), (n,), device=self.device)
        newest = torch.rand(n, device=self.device) < self.cfg.latest_prob
        self.slot[env_ids] = torch.where(newest, self.latest, uniform)

    def act(self, obs: torch.Tensor) -> torch.Tensor | None:
        """Each env's snapshot action (mean) for its opponent *obs*; ``None`` while empty."""
        if not self.policies:
            return None
        actions = torch.stack([p(obs) for p in self.policies])
        return actions[self.slot.clamp(max=len(self.policies) - 1), self._envs]


def find_opponent_pool(env: Any) -> OpponentPool | None:
    """The :class:`OpponentPool` of an action term of *env* (unwrapped), if any."""
    for term in env.action_manager._terms.values():
        pool = getattr(term, "opponent_pool", None)
        if pool is not None:
            return pool
    return None


def install_self_play(runner: Any) -> OpponentPool | None:
    """Feed the learner's snapshots to the env's opponent during ``runner.learn``.

    Adds the current policy right away, then one snapshot every
    ``snapshot_every`` iterations. RSL-RL has no callback hook, so this wraps
    ``runner.logger.log`` (called once per iteration, after the update), as
    ``stop_on_success`` in ``scripts/train_mjlab.py`` does.

    Args:
        runner: An RSL-RL ``OnPolicyRunner`` on an mjlab ``RslRlVecEnvWrapper``.

    Returns:
        The pool, or ``None`` if the task has no opponent pool (nothing installed).
    """
    pool = find_opponent_pool(runner.env.unwrapped)
    if pool is None:
        return None
    pool.add(runner.alg.get_policy())
    log = runner.logger.log

    def log_and_snapshot(*args: Any, **kwargs: Any) -> None:
        log(*args, **kwargs)
        if (kwargs["it"] + 1) % pool.cfg.snapshot_every == 0:
            pool.add(runner.alg.get_policy())

    runner.logger.log = log_and_snapshot
    return pool
