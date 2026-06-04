# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for :mod:`myosuite.integrations.musclemimic.hf_demo_cache`."""

from __future__ import annotations

from myosuite.integrations.musclemimic.hf_demo_cache import get_demo_motions


def test_get_demo_motions_includes_fullbody_demos() -> None:
    """MyoFullBody demo list should match upstream demo_cache length."""
    motions = get_demo_motions()
    assert "MyoFullBody" in motions
    assert len(motions["MyoFullBody"]) == 3
