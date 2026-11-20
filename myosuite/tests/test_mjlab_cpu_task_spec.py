# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for mjlab ``cpu_task_spec`` when ``register_env(..., time_limit=False)`` is used."""

from __future__ import annotations

import gymnasium as gym
import pytest

pytest.importorskip("mjlab")

from myosuite.core import registry as myo_registry  # noqa: E402
from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import (  # noqa: E402
    cpu_task_spec,
)

pytestmark = pytest.mark.tier2


def test_cpu_task_spec_reads_registry_horizon_without_time_limit() -> None:
    """``cpu_task_spec`` must fall back to ``_ENV_REGISTRY`` when ``time_limit=False``."""
    env_id = "myoCustomNoTimeLimitTest-v0"
    entry = "myosuite.envs.myo.tasks.basic.arm.pose:PoseEnvV0"
    try:
        myo_registry.register_env(
            env_id,
            entry,
            max_episode_steps=350,
            time_limit=False,
            kwargs={"frame_skip": 5},
        )
        assert gym.spec(env_id).max_episode_steps is None
        task = cpu_task_spec(env_id)
        assert task.max_episode_steps == 350
        assert task.frame_skip == 5
    finally:
        gym.envs.registry.pop(env_id, None)
        myo_registry._ENV_REGISTRY.pop(env_id, None)
        myo_registry._REGISTERED_BY.pop(env_id, None)


def test_cpu_task_spec_raises_when_no_horizon_recorded() -> None:
    """Unbounded bare registrations without ``max_episode_steps`` raise ``ValueError``."""
    env_id = "myoBareUnboundedTest-v0"
    entry = "myosuite.envs.myo.tasks.basic.arm.pose:PoseEnvV0"
    try:
        gym.register(id=env_id, entry_point=entry, max_episode_steps=None)
        with pytest.raises(ValueError, match="max_episode_steps"):
            cpu_task_spec(env_id)
    finally:
        gym.envs.registry.pop(env_id, None)
