# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""A step's ``info["obs_dict"]`` keeps that step's values after later steps.

Rollout loggers (``examine_policy``) store ``info`` by reference, so an entry
that views ``MjData`` (a ``site_xpos`` row, a ``sensordata`` slice) would show
the final state at every logged step.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from myosuite import make_env

pytestmark = pytest.mark.tier1


@pytest.mark.parametrize(
    "env_id",
    [
        "myoChallengeDieReorientP2-v0",  # obj_pos, goal_pos
        "myoChallengeBimanual-v0",  # palm_pos, fin0..4, obj_pos
        "myoChallengeTableTennisP1-v0",  # ball / paddle / pelvis / palm entries
        "myoElbowPose1D6MRandom-v0",
    ],
)
def test_info_obs_dict_is_a_snapshot(env_id: str) -> None:
    env = make_env(env_id)
    try:
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        space = env.action_space
        info = env.step(rng.uniform(space.low, space.high).astype(space.dtype))[4]
        snapshot = copy.deepcopy(info["obs_dict"])
        for _ in range(3):
            env.step(rng.uniform(space.low, space.high).astype(space.dtype))
        changed = [
            key
            for key, value in info["obs_dict"].items()
            if not np.array_equal(value, snapshot[key])
        ]
        assert changed == []
    finally:
        env.close()
