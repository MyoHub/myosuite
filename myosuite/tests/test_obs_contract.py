# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Step contract of the CPU envs: ``MyoGymnasiumEnv`` and its ``step()`` overrides.

The registry-wide dtype / bounds checks live in ``test_registry.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from myosuite.utils import gym

pytestmark = pytest.mark.tier1

# One env id per MyoGymnasiumEnv class with its own step(), plus the base step().
STEP_OVERRIDE_IDS = (
    "myoElbowPose1D6MRandom-v0",  # arm.pose.PoseEnvV0
    "myoHandReachFixed-v0",  # arm.reach.ReachEnvV0
    "myoHandKeyTurnFixed-v0",  # arm.key_turn.KeyTurnEnvV0
    "myoHandObjHoldFixed-v0",  # hand.obj_hold
    "myoHandPenTwirlFixed-v0",  # hand.pen
    "myoTorsoPoseFixed-v0",  # torso.pose.TorsoEnvV0
    "myoLegStandRandom-v0",  # leg.reach.LegReachEnvV0
    "myoLegWalk-v0",  # leg.walk.LegWalkEnvV0
    "myoChallengeBaodingP1-v1",
    "myoChallengeBimanual-v0",
    "myoChallengeChaseTagP1-v0",
    "myoChallengeDieReorientP1-v0",
    "myoChallengeOslRunFixed-v0",
    "myoChallengeRelocateP1-v0",
    "myoChallengeSoccerP1-v0",
    "myoChallengeTableTennisP0-v0",
    "myoElbowPoseTaskFixed-v0",  # ModularTaskEnv
    "myoHandReorient8-v0",  # base MyoGymnasiumEnv.step
)


@pytest.mark.parametrize("env_id", STEP_OVERRIDE_IDS)
def test_step_keeps_base_contract(env_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every step() renders on request and rejects a reward dict without "done"."""
    env = gym.make(env_id).unwrapped
    try:
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)

        frames: list[int] = []
        monkeypatch.setattr(env, "mj_render", lambda: frames.append(1))
        env.mujoco_render_frames = True
        env.step(action)
        assert frames == [1]
        env.mujoco_render_frames = False

        reward_dict = env.get_reward_dict
        monkeypatch.setattr(
            env,
            "get_reward_dict",
            lambda obs_dict: {
                k: v for k, v in reward_dict(obs_dict).items() if k != "done"
            },
        )
        with pytest.raises(KeyError, match="done"):
            env.step(action)
    finally:
        env.close()
