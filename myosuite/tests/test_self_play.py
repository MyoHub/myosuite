# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Frozen-opponent self-play: the snapshot pool.

The training hook (``install_self_play``) is exercised on the ChaseTag train
env in ``test_chasetag_fbvs_train_parity.py``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

from myosuite.envs.myo.backends.mjlab.tasks.mdp.self_play import (  # noqa: E402
    OpponentPool,
    SelfPlayCfg,
)

pytestmark = pytest.mark.tier2


class _Const(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.full((obs.shape[0], 2), self.value)


def test_pool_keeps_the_newest_snapshots_and_one_per_env() -> None:
    pool = OpponentPool(
        SelfPlayCfg(pool_size=2, latest_prob=1.0), num_envs=3, device="cpu"
    )
    obs = torch.zeros(3, 4)
    assert pool.act(obs) is None
    for value in (1.0, 2.0, 3.0):  # 3.0 replaces the oldest (1.0)
        pool.add(_Const(value))
    assert len(pool) == 2
    pool.reset(slice(None))  # latest_prob=1: everyone plays the newest
    assert torch.all(pool.act(obs) == 3.0)
    pool.slot[:] = torch.tensor([0, 1, 0])
    assert pool.act(obs)[:, 0].tolist() == [3.0, 2.0, 3.0]


def test_snapshots_are_frozen_copies() -> None:
    """A snapshot keeps acting like the learner did when it was taken."""
    learner = torch.nn.Linear(4, 2)
    pool = OpponentPool(SelfPlayCfg(), num_envs=2, device="cpu")
    pool.add(learner)
    obs = torch.ones(2, 4)
    before = pool.act(obs).clone()
    with torch.no_grad():
        learner.weight.add_(1.0)
    assert torch.equal(pool.act(obs), before)
    assert not any(p.requires_grad for p in pool.policies[0].parameters())
