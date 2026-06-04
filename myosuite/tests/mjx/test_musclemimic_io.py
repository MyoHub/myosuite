# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke tests for the MuscleMimic-compatible MJX bimanual task."""

from __future__ import annotations

import pytest

try:
    import jax
    from mujoco import mjx as _mjx  # noqa: F401
    import mujoco_playground as _mp  # noqa: F401

    _MJX_AVAILABLE = True
except (ImportError, AttributeError):
    _MJX_AVAILABLE = False


pytestmark = pytest.mark.tier2


@pytest.mark.skipif(not _MJX_AVAILABLE, reason="MJX/mujoco_playground not available")
def test_musclemimic_bimanual_smoke_reset() -> None:
    from myosuite.envs.myo.backends.mjx import make

    env = make("MjxMimicBimanual-v0")
    state = env.reset(jax.random.PRNGKey(0))

    assert "state" in state.obs
    assert state.obs["state"].ndim == 1


@pytest.mark.skipif(not _MJX_AVAILABLE, reason="MJX/mujoco_playground not available")
def test_musclemimic_bimanual_default_sites() -> None:
    from myosuite.envs.myo.backends.mjx import make

    env = make("MjxMimicBimanual-v0")
    assert tuple(env._config.mimic_site_names) == (
        "upper_body_mimic",
        "left_shoulder_mimic",
        "left_elbow_mimic",
        "left_hand_mimic",
        "right_shoulder_mimic",
        "right_elbow_mimic",
        "right_hand_mimic",
    )


@pytest.mark.skipif(not _MJX_AVAILABLE, reason="MJX/mujoco_playground not available")
def test_musclemimic_fullbody_smoke_reset() -> None:
    pytest.importorskip("musclemimic_models")
    import jax

    from myosuite.envs.myo.backends.mjx import make

    env = make("MjxMimicFullbody-v0")
    state = env.reset(jax.random.PRNGKey(0))

    assert "state" in state.obs
    assert state.obs["state"].ndim == 1


@pytest.mark.skipif(not _MJX_AVAILABLE, reason="MJX/mujoco_playground not available")
def test_musclemimic_fullbody_default_site_count() -> None:
    pytest.importorskip("musclemimic_models")
    from myosuite.envs.myo.backends.mjx import make

    env = make("MjxMimicFullbody-v0")
    assert len(tuple(env._config.mimic_site_names)) == 17
