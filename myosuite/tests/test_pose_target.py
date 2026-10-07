# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""PoseEnvV0 observes and rewards one target, however the target is changed."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import myosuite  # noqa: F401
from myosuite import make_env

pytestmark = pytest.mark.tier1


def _update_target(env) -> None:
    with pytest.warns(DeprecationWarning):
        env.update_target()


def _assign_target(env) -> None:
    env.target_jnt_value = env.target_jnt_value + 0.5


@pytest.mark.parametrize(
    "change_target",
    [_update_target, lambda env: env._update_target(restore_sim=True), _assign_target],
    ids=["update_target", "_update_target", "assign"],
)
def test_reward_scores_the_observed_target(change_target: Callable) -> None:
    env = make_env("myoElbowPose1D6MRandom-v0")
    u = env.unwrapped
    env.reset(seed=0)
    old_target = u.target_jnt_value.copy()
    change_target(u)
    assert not np.allclose(u.target_jnt_value, old_target)
    _, _, _, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
    pose_err = info["obs_dict"]["pose_err"]
    np.testing.assert_allclose(pose_err, u.target_jnt_value - u.data.qpos[:1])
    # pose = -|target - qpos| of the target the policy observes.
    np.testing.assert_allclose(info["rwd_dict"]["pose"], -np.linalg.norm(pose_err))
