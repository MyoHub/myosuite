# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for core playback contract dataclasses."""

from __future__ import annotations

from pathlib import Path

from myosuite.core.playback_contract import (
    PlaybackArtifacts,
    PlaybackRequest,
    PlaybackResult,
)


def test_playback_contract_objects_construct() -> None:
    """Contract dataclasses should expose expected fields."""
    req = PlaybackRequest(checkpoint_ref="hf://a/b", motion_ref="KIT/1/x")
    artifacts = PlaybackArtifacts(
        checkpoint_path=Path("/tmp/ckpt"),
        motion_path=Path("/tmp/motion.npz"),
    )
    result = PlaybackResult(frames_executed=10, mean_tracking_error=0.1)
    assert req.backend == "gym"
    assert artifacts.checkpoint_path.name == "ckpt"
    assert result.frames_executed == 10
