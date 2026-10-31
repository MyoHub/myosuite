# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Pen twirl success indicator sits on the target pen (issue 136)."""

from __future__ import annotations

import numpy as np
import pytest

from myosuite import make_env

pytestmark = pytest.mark.tier1


@pytest.mark.parametrize(
    "env_id", ["myoHandPenTwirlFixed-v0", "myoHandPenTwirlRandom-v0"]
)
def test_indicator_on_target_turns_green_only_when_solved(env_id: str) -> None:
    env = make_env(env_id)
    u = env.unwrapped
    try:
        env.reset(seed=0)
        model = u.model
        sid = u.success_indicator_sid
        assert model.site_bodyid[sid] == model.body("target").id
        # The position goal stays hidden so it no longer covers the pen.
        assert model.site_group[model.site("eps_ball").id] == 4

        obs = dict(u._get_obs_dict(u._accessor))
        solved = {**obs, "obj_rot": obs["obj_des_rot"], "obj_err_pos": np.zeros(3)}
        assert u.get_reward_dict(solved)["solved"]
        np.testing.assert_array_equal(model.site_rgba[sid, :2], [0, 2])

        flipped = {**solved, "obj_rot": -np.asarray(obs["obj_des_rot"])}
        assert not u.get_reward_dict(flipped)["solved"]
        np.testing.assert_array_equal(model.site_rgba[sid, :2], [2, 0])
    finally:
        env.close()


def test_rewards_without_indicator_leave_site_colours_alone() -> None:
    """A model lacking ``target_ball`` (index -1) still computes rewards."""
    env = make_env("myoHandPenTwirlFixed-v0")
    u = env.unwrapped
    try:
        env.reset(seed=0)
        u.success_indicator_sid = -1
        before = u.model.site_rgba.copy()
        rewards = u.get_reward_dict(dict(u._get_obs_dict(u._accessor)))
        assert np.isfinite(rewards["dense"])
        np.testing.assert_array_equal(u.model.site_rgba, before)
    finally:
        env.close()
