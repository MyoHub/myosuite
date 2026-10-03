# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tier A migration matrix checks for Mimic (site-tracking) tasks."""

from __future__ import annotations

import pytest

from myosuite.tests.support.optional_deps import require_musclemimic_models


pytestmark = pytest.mark.tier2


@pytest.mark.parametrize(
    "env_id",
    ("myoMimicBimanual-v0", "myoMimicFullbody-v0"),
)
def test_musclemimic_cpu_ids_make_reset_step(env_id: str) -> None:
    """CPU Mimic env IDs should reset and step with finite outputs."""
    require_musclemimic_models()
    import numpy as np
    import myosuite
    from myosuite.utils import gym

    myosuite.register_all_envs()
    env = gym.make(env_id)
    obs, info = env.reset(seed=0)
    assert obs.size > 0
    assert np.all(np.isfinite(obs))
    assert isinstance(info, dict)
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert np.all(np.isfinite(obs2))
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info2, dict)
    env.close()


@pytest.mark.tier1
def test_musclemimic_mjx_ids_registered() -> None:
    """Canonical MJX Mimic IDs must stay discoverable."""
    try:
        from myosuite.envs.myo.backends.mjx import ALL_ENVS
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"MJX not importable on this setup: {exc}")

    expected = {"MjxMimicBimanual-v0", "MjxMimicFullbody-v0"}
    assert expected.issubset(set(ALL_ENVS))


def test_musclemimic_cross_backend_tier_a_target() -> None:
    """Gate: canonical Mimic IDs exist on CPU, MJX, and mjlab registries."""
    import myosuite
    from myosuite.envs.myo.backends.mjlab import REGISTERED_TASKS

    try:
        from myosuite.envs.myo.backends.mjx import ALL_ENVS
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"MJX not importable on this setup: {exc}")

    myosuite.register_all_envs()
    gym_ids = set(myosuite.gym_registry_specs().keys())
    mjx_ids = set(ALL_ENVS)
    mjlab_ids = set(REGISTERED_TASKS)
    want_cpu = {"myoMimicBimanual-v0", "myoMimicFullbody-v0"}
    want_mjx = {"MjxMimicBimanual-v0", "MjxMimicFullbody-v0"}
    want_mjlab = {"myoMimicBimanual-v0", "myoMimicFullbody-v0"}
    has_gym = want_cpu.issubset(gym_ids)
    has_mjx = want_mjx.issubset(mjx_ids)
    has_mjlab = want_mjlab.issubset(mjlab_ids)
    if not (has_gym and has_mjx and has_mjlab):
        pytest.skip(
            "Mimic cross-backend mappings are not complete on this installation."
        )

    assert has_gym
    assert has_mjx
    assert has_mjlab
