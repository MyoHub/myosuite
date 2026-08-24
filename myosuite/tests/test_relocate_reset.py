# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Relocate reset must not hang under P2eval randomization."""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.tier1


def test_relocate_p2eval_reset_bounded() -> None:
    """P2eval used an unbounded ``while ncon > 0`` loop that hung SB3."""
    import myosuite

    myosuite.register_all_envs()
    from myosuite.utils import gym

    env = gym.make("myoChallengeRelocateP2eval-v0")
    try:
        t0 = time.perf_counter()
        for seed in range(8):
            env.reset(seed=seed)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"reset hung or too slow: {elapsed:.2f}s for 8 seeds"
    finally:
        env.close()
