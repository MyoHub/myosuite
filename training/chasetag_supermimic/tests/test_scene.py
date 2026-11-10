# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The scored scene without data: the players have the tracker's body, and a yaw rotation is a symmetry."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import pytest
import torch

from chasetag_supermimic.distill import yaw_rotate
from chasetag_supermimic.match import AGENTS, make
from chasetag_supermimic.tracker import check_same_body

pytestmark = pytest.mark.tier2


@pytest.fixture(scope="module", name="env")
def env_fixture() -> Any:
    env = make()
    yield env
    env.close()


def test_players_have_the_tracker_body(env: Any) -> None:
    from myosuite.envs.myo.tasks.mimic.tracking.models import compiled_body

    joints = check_same_body(env.unwrapped.model, compiled_body("myofullbody"))
    assert len(joints) > 0


def test_yaw_rotate_equals_rotating_the_scene(env: Any) -> None:
    """Rotating every free body about the world z axis rotates each player's observation as ``yaw_rotate`` does."""
    u = env.unwrapped
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(10):
        env.step({a: rng.uniform(0, 0.3, 354) for a in AGENTS})
    m, d = u.model, u.data
    before = {a: u._get_obs()[a].copy() for a in AGENTS}
    th = 0.7
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    qz = np.array([np.cos(th / 2), 0, 0, np.sin(th / 2)])
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
            d.qpos[qa : qa + 3] = R @ d.qpos[qa : qa + 3]
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, qz, d.qpos[qa + 3 : qa + 7])
            d.qpos[qa + 3 : qa + 7] = q
            d.qvel[va : va + 3] = R @ d.qvel[va : va + 3]
    mujoco.mj_forward(m, d)
    after = {a: u._get_obs()[a] for a in AGENTS}
    for a in AGENTS:
        want = yaw_rotate(
            torch.as_tensor(before[a], dtype=torch.float64)[None],
            torch.tensor([th], dtype=torch.float64),
        )
        np.testing.assert_allclose(after[a], want[0].numpy(), atol=1e-4)
