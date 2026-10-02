# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""examine_policy record layout, offscreen frames and evaluate_success counting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import myosuite.utils.video_io as video_io
import myosuite.viz.mj_renderer as mj_renderer
from myosuite.utils import gym
from myosuite.utils.path_utils import evaluate_success
from myosuite.utils.policy_utils import examine_policy

pytestmark = pytest.mark.tier1


class _CounterEnv:
    """State ``s`` counts steps; solved from ``s >= 3``; terminates at ``s == 5``."""

    id = "Counter-v0"
    dt = 0.1
    model = data = None  # handed to the (stubbed) offscreen renderer

    def __init__(self) -> None:
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.s = 0

    def _out(self) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        info = {
            "solved": self.s >= 3,
            "rwd_dense": float(self.s),
            "obs_dict": {"s": np.array([float(self.s)])},
        }
        return np.array([self.s], np.float32), float(self.s), self.s >= 5, False, info

    def reset(self) -> tuple[np.ndarray, dict]:
        self.s = 0
        return self._out()[0], {}

    def forward(self) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return self._out()

    def step(self, act: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.s += 1
        return self._out()

    def render(self) -> None:
        return None  # like MyoGymnasiumEnv without a render_mode


class _ZeroPolicy:
    def __init__(self, action_space: Any) -> None:
        self.action_space = action_space

    def get_action(self, obs: np.ndarray) -> tuple[np.ndarray, dict]:
        act = self.action_space.sample() * 0.0
        return act, {"evaluation": act}


def test_records_are_time_aligned_and_final_state_logged_once() -> None:
    env = _CounterEnv()
    path = examine_policy(env, _ZeroPolicy(env.action_space), horizon=10)["Trial0"]
    states = np.arange(6.0)
    np.testing.assert_array_equal(path["observations"][:, 0], states)
    # env_infos[t] describes the same state as observations[t] (was s_{t+1}, last one twice)
    np.testing.assert_array_equal(path["env_infos"]["obs_dict"]["s"][:, 0], states)
    np.testing.assert_array_equal(path["rewards"], states)
    np.testing.assert_array_equal(path["done"], [False] * 5 + [True])
    np.testing.assert_allclose(path["time"], 0.1 * states)
    assert (
        np.isnan(path["actions"][-1]).all() and np.isfinite(path["actions"][:-1]).all()
    )


def test_evaluate_success_counts_each_solved_state_once() -> None:
    """Solved states s=3,4,5 -> 3 solved steps (the duplicated final info made it 4)."""
    env = _CounterEnv()
    trace = examine_policy(env, _ZeroPolicy(env.action_space), horizon=10)
    assert evaluate_success(env, trace, successful_steps=2) == 100.0
    assert evaluate_success(env, trace, successful_steps=3) == 0.0


def test_offscreen_frames_use_frame_size_and_camera(monkeypatch, tmp_path) -> None:
    """Frames come from an offscreen renderer at frame_size / camera_name (were black)."""

    class _StubRenderer:
        instances: list[_StubRenderer] = []

        def __init__(self, model: Any, data: Any) -> None:
            self.calls: list[tuple[int, int, Any]] = []
            self.closed = False
            _StubRenderer.instances.append(self)

        def render_offscreen(
            self, width: int, height: int, camera_id: Any = -1, **_: Any
        ) -> np.ndarray:
            self.calls.append((width, height, camera_id))
            return np.full((height, width, 3), 7, np.uint8)

        def close(self) -> None:
            self.closed = True

    videos: dict[str, np.ndarray] = {}
    monkeypatch.setattr(mj_renderer, "MJRenderer", _StubRenderer)
    monkeypatch.setattr(
        video_io,
        "write_video",
        lambda name, frames, **kw: videos.update({name: frames}),
    )
    env = _CounterEnv()
    examine_policy(
        env,
        _ZeroPolicy(env.action_space),
        horizon=10,
        render="offscreen",
        camera_name="side",
        frame_size=(96, 64),
        output_dir=str(tmp_path),
    )
    (frames,) = videos.values()
    assert frames.shape == (5, 64, 96, 3)
    assert (frames == 7).all()
    (renderer,) = _StubRenderer.instances
    assert renderer.calls == [(96, 64, "side")] * 5
    assert renderer.closed


def test_myo_env_observations_match_logged_obs_dict() -> None:
    """On a real env, observations[t] is the obs vector of env_infos.obs_dict[t]."""
    env = gym.make("myoElbowPose1D6MRandom-v0").unwrapped
    env.seed(0)
    env.action_space.seed(0)
    path = examine_policy(env, _ZeroPolicy(env.action_space), horizon=6)["Trial0"]
    obs_dict = path["env_infos"]["obs_dict"]
    for t, obs in enumerate(path["observations"]):
        expected = env._obs_dict_to_vec({k: v[t] for k, v in obs_dict.items()})
        np.testing.assert_allclose(obs, expected.astype(np.float32), atol=1e-6)
    env.close()
