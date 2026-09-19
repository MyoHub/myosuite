# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests that REGISTERED_TASKS in the mjlab backend stays in sync."""

from __future__ import annotations

import gymnasium as gym

from myosuite.envs.myo.backends.mjlab import REGISTERED_TASKS


def test_registered_tasks_non_empty() -> None:
    """REGISTERED_TASKS must have at least one entry."""
    assert len(REGISTERED_TASKS) > 0, "REGISTERED_TASKS is empty"


def test_registered_tasks_keys_in_gymnasium_registry() -> None:
    """Every key in REGISTERED_TASKS must be registered in the Gymnasium env registry."""
    gym_ids = set(gym.envs.registry.keys())
    missing = [env_id for env_id in REGISTERED_TASKS if env_id not in gym_ids]
    assert not missing, (
        f"The following env IDs are in REGISTERED_TASKS but not in gym.envs.registry: "
        f"{missing}. Either register them via myosuite.__init__ or remove them from "
        f"REGISTERED_TASKS."
    )
