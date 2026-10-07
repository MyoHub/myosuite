# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for SB3 rollout rendering camera and scene selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from myosuite import make_env

pytestmark = pytest.mark.tier1


def test_scene_option_hides_internal_wrap_geoms() -> None:
    """Visible anatomy stays on while wrapping/debug groups stay hidden."""
    from scripts.render_sb3_solutions import _scene_option

    option = _scene_option()
    np.testing.assert_array_equal(option.geomgroup[:3], np.ones(3))
    np.testing.assert_array_equal(option.geomgroup[3:], np.zeros(3))
    assert np.all(option.tendongroup == 1)


def test_actor_camera_ignores_full_model_extent() -> None:
    """Distant targets must not force the camera far from a MyoArm actor."""
    import myosuite  # noqa: F401
    from scripts.render_sb3_solutions import _actor_body_ids, _actor_camera

    env = make_env("myoArmReachFixed-v0")
    try:
        env.reset(seed=0)
        model = env.unwrapped.model
        data = env.unwrapped.data
        # The arm-reach model frames its own default camera (small extent); emulate
        # the huge automatic extent of a full scene, which the camera must ignore.
        model.stat.extent = 25.0
        actor_body_ids = _actor_body_ids(model)
        camera = _actor_camera(data, actor_body_ids)

        assert actor_body_ids.size > 20
        assert camera.distance < 3.0
        assert 0.5 < camera.lookat[2] < 1.6
    finally:
        env.close()


def test_sb3_policy_normalizes_with_the_saved_vec_normalize(tmp_path: Path) -> None:
    """The rendered PPO policy sees observations normalized as in training."""
    from myosuite.tests.support.sb3_models import (
        raw_observations,
        vec_normalized_model,
    )
    from scripts.render_sb3_solutions import _sb3_policy

    model, venv = vec_normalized_model("PPO")
    try:
        raw = raw_observations(venv)
        expected, _ = model.predict(venv.normalize_obs(raw), deterministic=True)
        unnormalized, _ = model.predict(raw, deterministic=True)
        model.save(tmp_path / "ppo_final.zip")
        venv.save(tmp_path / "stats.pkl")
    finally:
        venv.close()
    assert np.abs(expected - unnormalized).max() > 1e-2

    policy, stats = _sb3_policy(tmp_path / "ppo_final.zip")
    assert stats is None
    np.testing.assert_allclose(policy(raw), unnormalized, atol=1e-6)

    (tmp_path / "stats.pkl").replace(tmp_path / "vecnormalize.pkl")
    policy, stats = _sb3_policy(tmp_path / "ppo_final.zip")
    assert stats == tmp_path / "vecnormalize.pkl"
    np.testing.assert_allclose(policy(raw), expected, atol=1e-6)
