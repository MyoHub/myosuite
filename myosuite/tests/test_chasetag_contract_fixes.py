# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for HeightField, ChaseTagField patch axes, and ChaseTag resets."""

from __future__ import annotations

import numpy as np
import pytest

from myosuite import make_env
from myosuite.envs.heightfields import ChaseTagField, TerrainTypes

pytestmark = pytest.mark.tier1


def test_heightfield_size_and_chasetag_flatten_agent_patch_axes() -> None:
    env = make_env("myoChallengeChaseTagP2-v0").unwrapped
    try:
        hf: ChaseTagField = env.heightfield
        assert hf.size == hf.nrow * hf.ncol

        hf.hfield.data[:, :] = 1.0
        qpos = np.zeros(7, dtype=np.float64)
        qpos[0] = 4.5
        qpos[1] = -4.5
        hf.flatten_agent_patch(qpos)
        ps = hf.patch_size
        np.testing.assert_array_equal(hf.hfield.data[0:ps, 2 * ps : 3 * ps], 0.0)
        assert float(hf.hfield.data[2 * ps : 3 * ps, 0:ps].sum()) > 0.0
        assert TerrainTypes.FLAT is not None
    finally:
        env.close()


def test_chasetag_repeller_equilibrium_fallback_and_single_keyframe_clamp() -> None:
    env = make_env("myoChallengeChaseTagP2eval-v0").unwrapped
    try:
        opp = env.opponent
        opp.opponent_probabilities = (0.1, 0.2, 0.2, 0.1)
        opp.rng = np.random.default_rng(0)
        orig_rng = opp.rng

        class _HighRng:
            def uniform(self, low=0.0, high=1.0, size=None):
                if low == 0.0 and high == 1.0 and size is None:
                    return 0.99
                return orig_rng.uniform(low=low, high=high, size=size)

        opp.rng = _HighRng()
        opp.sample_opponent_policy()
        assert opp.opponent_policy == "repeller"
        opp.rng = orig_rng

        env.reset(seed=0)
        agent_xy = opp.get_agent_pos().copy()
        opp.set_opponent_pose([float(agent_xy[0]), float(agent_xy[1]), 0.0])
        vel = opp.repeller_stochastic()
        assert vel.shape == (2,)
        assert np.all(np.isfinite(vel))
    finally:
        env.close()

    fb_env = make_env("myoChallengeChaseTagFBP2-v0").unwrapped
    try:
        assert fb_env.model.nkey == 1
        fb_env.reset_type = "init"
        qpos_init, _ = fb_env._get_reset_state()
        assert qpos_init.shape == (fb_env.model.nq,)
        fb_env.reset_type = "random"
        qpos_rand, _ = fb_env._get_reset_state()
        assert qpos_rand.shape == (fb_env.model.nq,)
    finally:
        fb_env.close()
