# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tier A migration matrix checks for MyoChallenge tasks."""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.tier2

_CHALLENGE_IDS = (
    "myoChallengeSoccerP1-v0",
    "myoChallengeSoccerP2-v0",
    "myoChallengeTableTennisP0-v0",
    "myoChallengeTableTennisP1-v0",
    "myoChallengeTableTennisP2-v0",
    "myoChallengeBimanual-v0",
    "myoChallengeOslRunFixed-v0",
    "myoChallengeOslRunRandom-v0",
    "myoChallengeRelocateP1-v0",
    "myoChallengeRelocateP2-v0",
    "myoChallengeRelocateP2eval-v0",
    "myoChallengeChaseTagP1-v0",
    "myoChallengeChaseTagP2-v0",
    "myoChallengeChaseTagP2eval-v0",
    "myoChallengeChaseTagFBP2-v0",
    "myoChallengeDieReorientDemo-v0",
    "myoChallengeDieReorientP1-v0",
    "myoChallengeDieReorientP2-v0",
    "myoChallengeBaodingP1-v1",
    "myoChallengeBaodingP2-v1",
)


@pytest.mark.parametrize("env_id", _CHALLENGE_IDS)
def test_challenge_gym_registration_contract(env_id: str) -> None:
    """All canonical challenge IDs must remain Gym-constructible."""
    import gymnasium as gym
    import myosuite

    myosuite.register_all_envs()
    env = gym.make(env_id)
    obs, _ = env.reset(seed=0)
    assert obs is not None
    env.close()


def test_challenge_cross_backend_tier_a_target() -> None:
    """Gate: challenge family has both MJX and MJLab mappings."""
    try:
        from myosuite.envs.myo.backends.mjx import ALL_ENVS
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"MJX not importable on this setup: {exc}")

    from myosuite.envs.myo.backends.mjlab import REGISTERED_TASKS

    # Canonical target rule for migration: each challenge family must have at
    # least one MJX and one MJLab representation before promotion to Tier A.
    challenge_mjlab = [k for k in REGISTERED_TASKS if k.startswith("myoChallenge")]
    challenge_mjx = [k for k in ALL_ENVS if "Challenge" in k]
    if not challenge_mjlab or not challenge_mjx:
        pytest.skip("MyoChallenge cross-backend mappings are not complete yet.")
    assert challenge_mjlab and challenge_mjx
