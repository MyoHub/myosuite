# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Smoke tests for the BoxingP0 mjlab backend registration.

If mjlab is not installed, tests are skipped automatically.
Note: These tests verify that the env cfg builds without error, but do NOT
instantiate a live ManagerBasedRlEnv (which requires a GPU / Warp runtime).
"""

from __future__ import annotations

import pytest

mjlab = pytest.importorskip("mjlab", reason="mjlab not installed")


def test_boxing_p0_mjlab_cfg_builds() -> None:
    """Smoke test: the env cfg builds without error."""
    from myosuite.envs.myo.backends.mjlab.register_mjlab_boxing import (
        _make_boxing_p0_env_cfg,
    )

    cfg = _make_boxing_p0_env_cfg(play=False)
    assert cfg is not None


def test_boxing_p0_mjlab_play_cfg_builds() -> None:
    """Smoke test: the play env cfg builds without error."""
    from myosuite.envs.myo.backends.mjlab.register_mjlab_boxing import (
        _make_boxing_p0_env_cfg,
    )

    cfg = _make_boxing_p0_env_cfg(play=True)
    assert cfg is not None


def test_boxing_p0_mjlab_env_id_constant() -> None:
    """Ensure the env id constant matches the expected string."""
    from myosuite.envs.myo.backends.mjlab.register_mjlab_boxing import (
        BOXING_P0_MJLAB_ENV_ID,
    )

    assert BOXING_P0_MJLAB_ENV_ID == "myoChallengeBoxingP0Mjlab-v0"


def test_boxing_task_spec_imports() -> None:
    """Smoke test: boxing_task_spec module is importable."""
    from myosuite.envs.myo.tasks.challenge.boxing_mannequin.boxing_task_spec import (
        BOXING_P0_MJLAB_ENV_ID,
        get_boxing_p0_task_spec,
        get_boxing_p0_env_spec,
    )

    assert BOXING_P0_MJLAB_ENV_ID == "myoChallengeBoxingP0Mjlab-v0"
    spec = get_boxing_p0_task_spec()
    assert spec is not None
    env_spec = get_boxing_p0_env_spec()
    assert env_spec is not None
