# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""The MyoChallenge 2025 envs render a camera that shows the agent by default (#323)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite import make_env

pytestmark = pytest.mark.tier2

ENVS = {
    "myoChallengeSoccerP1-v0": "agent_view",
    "myoChallengeTableTennisP0-v0": "default",
    "myoChallengeBimanual-v0": "front_view",
}


@pytest.mark.parametrize(("env_id", "camera"), ENVS.items())
def test_default_render_uses_the_named_camera(env_id: str, camera: str) -> None:
    """``render()`` without arguments shows the model camera of the env, not the far free camera."""
    env = make_env(env_id, render_mode="rgb_array")
    try:
        env.reset(seed=0)
        unwrapped = env.unwrapped
        assert unwrapped.render_camera == camera
        names = [unwrapped.model.camera(i).name for i in range(unwrapped.model.ncam)]
        assert camera in names
        frame = env.render()
        assert frame.shape == (480, 640, 3)
        renderer = mujoco.Renderer(unwrapped.model, 480, 640)
        renderer.update_scene(unwrapped.data, camera=camera)
        expected = renderer.render()
        renderer.close()
        assert np.mean(np.abs(frame.astype(int) - expected.astype(int))) < 3.0
    finally:
        env.close()


def test_other_envs_keep_the_free_camera() -> None:
    """Envs without a ``render_camera`` still render the free camera."""
    env = make_env("myoElbowPose1D6MFixed-v0", render_mode="rgb_array")
    try:
        env.reset(seed=0)
        assert env.unwrapped.render_camera is None
        assert env.render().shape == (480, 640, 3)
    finally:
        env.close()
