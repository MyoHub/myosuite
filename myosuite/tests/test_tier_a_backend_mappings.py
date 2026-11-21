# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tier A backend mapping checks for newly promoted/targeted tasks."""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.tier2


def test_mjlab_mapping_for_finger_hand_targets() -> None:
    """MJLab registry should expose expected finger and hand task IDs."""
    pytest.importorskip("mjlab")
    from myosuite.envs.myo.backends.mjlab import list_mjlab_tasks

    registered = set(list_mjlab_tasks())
    for env_id in (
        "myoFingerPoseFixed-v0",
        "myoFingerPoseRandom-v0",
        "myoFingerReachRandom-v0",
        "myoHandPoseRandom-v0",
        "myoHandReachRandom-v0",
    ):
        assert env_id in registered


def test_mjlab_mapping_for_leg_walk_sarc_variant() -> None:
    """MJLab registry should include leg walk and sarcopenia variant IDs."""
    pytest.importorskip("mjlab")
    from myosuite.envs.myo.backends.mjlab import list_mjlab_tasks

    registered = set(list_mjlab_tasks())
    assert "myoLegWalk-v0" in registered
    assert "myoSarcLegWalk-v0" in registered


def test_mjx_mapping_for_finger_hand_targets() -> None:
    """MJX env list should include newly added finger/hand IDs."""
    try:
        from myosuite.envs.myo.backends.mjx import ALL_ENVS
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"MJX not importable on this setup: {exc}")

    assert "MjxFingerReachRandom-v0" in ALL_ENVS
    assert "MjxHandPoseRandom-v0" in ALL_ENVS
