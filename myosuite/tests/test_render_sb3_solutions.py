# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for SB3 rollout rendering camera and scene selection."""

from __future__ import annotations

import numpy as np
import pytest

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
    import myosuite
    from scripts.render_sb3_solutions import _actor_body_ids, _actor_camera

    myosuite.register_all_envs()
    from myosuite.utils import gym

    env = gym.make("myoArmReachFixed-v0")
    try:
        env.reset(seed=0)
        model = env.unwrapped.model
        data = env.unwrapped.data
        actor_body_ids = _actor_body_ids(model)
        camera = _actor_camera(data, actor_body_ids)

        assert actor_body_ids.size > 20
        assert model.stat.extent > 20.0
        assert camera.distance < 3.0
        assert 0.5 < camera.lookat[2] < 1.6
    finally:
        env.close()
