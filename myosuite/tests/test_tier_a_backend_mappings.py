# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tier A backend mapping checks for newly promoted/targeted tasks."""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.tier2


def test_mjlab_mapping_for_finger_hand_targets() -> None:
    """MJLab registry should expose expected task config mappings."""
    from myosuite.envs.myo.backends.mjlab import REGISTERED_TASKS
    from myosuite.envs.myo.backends.mjlab.configs.finger_pose_cfg import FingerPoseCfg
    from myosuite.envs.myo.backends.mjlab.configs.finger_reach_cfg import FingerReachCfg
    from myosuite.envs.myo.backends.mjlab.configs.hand_pose_cfg import HandPoseCfg

    assert REGISTERED_TASKS["myoFingerPoseFixed-v0"] is FingerPoseCfg
    assert REGISTERED_TASKS["myoFingerPoseRandom-v0"] is FingerPoseCfg
    assert REGISTERED_TASKS["myoFingerReachRandom-v0"] is FingerReachCfg
    assert REGISTERED_TASKS["myoHandPoseRandom-v0"] is HandPoseCfg


def test_mjlab_mapping_for_leg_walk_sarc_variant() -> None:
    """MJLab registry should include leg walk and sarcopenia variant IDs."""
    from myosuite.envs.myo.backends.mjlab import REGISTERED_TASKS
    from myosuite.envs.myo.backends.mjlab.configs.walk_cfg import WalkCfg

    assert REGISTERED_TASKS["myoLegWalk-v0"] is WalkCfg
    assert REGISTERED_TASKS["myoSarcLegWalk-v0"] is WalkCfg


def test_mjx_mapping_for_finger_hand_targets() -> None:
    """MJX env list should include newly added finger/hand IDs."""
    try:
        from myosuite.envs.myo.backends.mjx import ALL_ENVS
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"MJX not importable on this setup: {exc}")

    assert "MjxFingerReachRandom-v0" in ALL_ENVS
    assert "MjxHandPoseRandom-v0" in ALL_ENVS
