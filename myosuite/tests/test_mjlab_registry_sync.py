# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests that MyoSuite's mjlab task registry stays in sync and fails loudly."""

from __future__ import annotations

import subprocess
import sys

import gymnasium as gym
import pytest

pytestmark = pytest.mark.tier2


def test_list_mjlab_tasks_non_empty() -> None:
    """``list_mjlab_tasks()`` must report all registered MyoSuite mjlab tasks."""
    pytest.importorskip("mjlab")
    from myosuite.envs.myo.backends.mjlab import list_mjlab_tasks

    tasks = list_mjlab_tasks()
    assert len(tasks) > 0, "list_mjlab_tasks() is empty"


def test_mjlab_tasks_keys_in_gymnasium_registry() -> None:
    """Every MyoSuite mjlab task ID must also exist in the Gymnasium env registry."""
    pytest.importorskip("mjlab")
    import myosuite
    from myosuite.envs.myo.backends.mjlab import list_mjlab_tasks

    myosuite.register_all_envs()
    gym_ids = set(gym.envs.registry.keys())
    missing = [env_id for env_id in list_mjlab_tasks() if env_id not in gym_ids]
    assert not missing, (
        f"The following env IDs are in list_mjlab_tasks() but not in gym.envs.registry: "
        f"{missing}. Register them on CPU under the same env_id."
    )


def test_deprecated_registered_tasks_shim_matches_registry() -> None:
    """Accessing ``REGISTERED_TASKS`` emits a DeprecationWarning and mirrors ``list_mjlab_tasks``."""
    import myosuite.envs.myo.backends.mjlab as mjlab_backend

    with pytest.deprecated_call(match="REGISTERED_TASKS is deprecated"):
        mapping = mjlab_backend.REGISTERED_TASKS
    assert set(mapping) == set(mjlab_backend.list_mjlab_tasks())


def test_register_cpu_twins_propagates_config_failure() -> None:
    """``register_cpu_twins`` must raise instead of warning and skipping broken configs."""
    pytest.importorskip("mjlab")
    from myosuite.envs.myo.backends.mjlab.tasks.registration import register_cpu_twins

    def _boom(_env_id: str, play: bool = False):
        raise RuntimeError(f"broken config (play={play})")

    with pytest.raises(RuntimeError, match="broken config"):
        register_cpu_twins(("__unregistered_broken_test_env-v0",), _boom, lambda: None)  # type: ignore[arg-type]


def test_importing_mjlab_env_base_before_mjlab_avoids_circular_import() -> None:
    """Importing ``mjlab_env_base`` before ``mjlab`` must not trigger a circular ImportError."""
    code = (
        "from myosuite.envs.myo.backends.mjlab.mjlab_env_base import "
        "MjlabEnvAccessor, normalize_mjlab_env_ids\n"
        "import myosuite.envs.myo.backends.mjlab as b\n"
        "assert callable(normalize_mjlab_env_ids)\n"
        "assert isinstance(b.list_mjlab_tasks(), list)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
