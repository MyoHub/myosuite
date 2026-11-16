# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Die Reorient and Relocate score the rotation angle between object and goal.

The norm of Euler-angle differences jumps by up to 2 pi at the +-pi wrap and
near gimbal lock, so a die 0.05 rad from its goal could miss the 0.262 rad
tolerance.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from myosuite import make_env

pytestmark = pytest.mark.tier1

ANGLE = 0.05  # rad


def _place_object(env: Any, rotation: Rotation) -> None:
    """Put the object site on the goal (position reward 0), rotated by *rotation*.

    The object hangs on three slide joints and then three hinges about its x, y
    and z axes, i.e. intrinsic XYZ Euler angles in its rest frame.
    """
    model, data = env.model, env.data
    obj_sid, goal_sid = model.site("object_o").id, model.site("target_o").id
    body = model.site_bodyid[obj_sid]
    adr = model.jnt_qposadr[model.body_jntadr[body]]
    data.qpos[adr : adr + 6] = 0.0
    mujoco.mj_kinematics(model, data)
    rest_pos, rest_rot = data.xpos[body].copy(), data.xmat[body].reshape(3, 3)
    site_rel = model.site_pos[obj_sid]
    site_local = np.zeros(9)
    mujoco.mju_quat2Mat(site_local, model.site_quat[obj_sid])
    goal_rot = data.site_xmat[goal_sid].reshape(3, 3)
    body_rot = goal_rot @ rotation.as_matrix() @ site_local.reshape(3, 3).T
    offset = getattr(env, "goal_obj_offset", np.zeros(3))
    site_pos = data.site_xpos[goal_sid] - offset
    body_pos = site_pos - body_rot @ site_rel
    data.qpos[adr : adr + 3] = rest_rot.T @ (body_pos - rest_pos)
    chain = Rotation.from_matrix(rest_rot.T @ body_rot)
    data.qpos[adr + 3 : adr + 6] = chain.as_euler("XYZ")
    mujoco.mj_forward(model, data)


@pytest.mark.parametrize(
    "env_id",
    [
        "myoChallengeDieReorientP1-v0",
        "myoChallengeDieReorientP2-v0",
        "myoChallengeRelocateP2-v0",
    ],
)
def test_rotation_error_is_the_geodesic_angle(env_id: str) -> None:
    env = make_env(env_id).unwrapped
    rng = np.random.default_rng(0)
    try:
        for seed in range(40):
            env.reset(seed=seed)
            axis = rng.normal(size=3)
            _place_object(
                env, Rotation.from_rotvec(ANGLE * axis / np.linalg.norm(axis))
            )
            rwd = env.get_reward_dict(env._get_obs_dict(env._accessor))
            assert rwd["rot_dist"] == pytest.approx(-ANGLE, abs=1e-6)
            if "Reorient" in env_id:
                assert rwd["solved"]
    finally:
        env.close()
