# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Smoke tests for MyoChallenge boxing environment registration."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.error import NameNotFound, VersionNotFound

from myosuite.tests.support.optional_deps import require_musclemimic_models


pytestmark = pytest.mark.tier1

_REMOVED_BOXING_IDS = (
    "myoChallengeBoxingDrill-v0",
    "myoChallengeBoxingMannequinMulti-v0",
    "myoChallengeBoxingMannequin-v1",
)


class TestBoxingRegistry:
    """Guards against accidental re-registration of removed boxing envs."""

    def test_boxing_p0_and_removed_ids(self) -> None:
        """Boxing P0 (six-target pads) must exist; deprecated IDs must not."""
        import myosuite

        myosuite.register_all_envs()

        assert gym.spec("myoChallengeBoxingP0-v0") is not None
        assert gym.spec("myoChallengeBoxingMannequin-v0") is not None
        assert gym.spec("myoChallengeBoxingVs-v0") is not None

        for env_id in _REMOVED_BOXING_IDS:
            # Versioned envs under an existing namespace may raise VersionNotFound.
            with pytest.raises((NameNotFound, VersionNotFound)):
                gym.spec(env_id)

    def test_boxing_p0_reset_step(self) -> None:
        """Minimal reset/step on Boxing P0 after registration."""
        require_musclemimic_models()
        import myosuite

        myosuite.register_all_envs()
        env = gym.make("myoChallengeBoxingP0-v0")
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        obs2, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert obs2.shape == obs.shape
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        env.close()

    def test_boxing_p0_has_six_scored_targets(self) -> None:
        """P0 contract is fixed to six scored pads."""
        require_musclemimic_models()
        import myosuite

        myosuite.register_all_envs()
        env = gym.make("myoChallengeBoxingP0-v0")
        base = env.unwrapped
        assert len(base._meta.target_site_ids) == 6
        assert len(base._meta.target_geom_ids) == 6
        env.close()


def test_boxing_visuals_quat_uses_intrinsic_euler_convention() -> None:
    """Glove/helmet placement must use intrinsic RPY, not legacy ``euler2quat``."""
    from myosuite.envs.myo.tasks.challenge.boxing_visuals import (
        HELMET_EULER_DEG,
        quat_from_euler_xyz_deg,
    )
    from myosuite.physics.quat_math import intrinsic_euler2quat

    euler_rad = np.deg2rad(np.asarray(HELMET_EULER_DEG, dtype=np.float64))
    expected = intrinsic_euler2quat(euler_rad)
    got = np.asarray(quat_from_euler_xyz_deg(list(HELMET_EULER_DEG)), dtype=np.float64)
    np.testing.assert_allclose(got, expected, atol=1e-9)
