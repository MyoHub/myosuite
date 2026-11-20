# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for LegReachEnvV0._generate_qpos preserving freejoint position."""

from __future__ import annotations

import pytest

from myosuite import make_env

pytestmark = pytest.mark.tier1


def test_leg_reach_preserves_freejoint_position_under_qpos_generation() -> None:
    env = make_env("myoLegStandRandom-v0").unwrapped
    try:
        env.reset(seed=0)
        env._init_qpos[0] = 1.25
        qpos = env._generate_qpos()
        assert qpos[0] == pytest.approx(1.25)
    finally:
        env.close()
