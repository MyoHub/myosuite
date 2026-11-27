# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for mjlab ``condition_variants`` across ``myo*`` and non-``myo`` task ids."""

from __future__ import annotations

import gymnasium as gym
import pytest

pytest.importorskip("mjlab")

from myosuite.core import registry as myo_registry  # noqa: E402
from myosuite.envs.myo.backends.mjlab.tasks.registration import (  # noqa: E402
    condition_variants,
)

pytestmark = pytest.mark.tier2


def test_condition_variants_matches_myo_and_non_myo_ids() -> None:
    """``condition_variants`` must match ``_register_task_variants`` naming for all ids."""
    import myosuite  # noqa: F401, PLC0415

    assert "myoSarcElbowPose1D6MRandom-v0" in condition_variants(
        "myoElbowPose1D6MRandom-v0"
    )

    temp_ids = (
        "ElbowPoseCustomTest-v0",
        "ElbowPoseCustomTestSarc-v0",
        "ElbowPoseCustomTestFati-v0",
    )
    entry = "myosuite.envs.myo.tasks.basic.arm.pose:PoseEnvV0"
    try:
        for tid in temp_ids:
            myo_registry.register_env(tid, entry, max_episode_steps=100)
        assert condition_variants("ElbowPoseCustomTest-v0") == [
            "ElbowPoseCustomTestSarc-v0",
            "ElbowPoseCustomTestFati-v0",
        ]
    finally:
        for tid in temp_ids:
            gym.envs.registry.pop(tid, None)
            myo_registry._ENV_REGISTRY.pop(tid, None)
            myo_registry._REGISTERED_BY.pop(tid, None)
