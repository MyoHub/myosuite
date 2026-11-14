# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""``solved`` of the directional walks means sustained walking, not one step on target.

As in ``test_walk_solved_sustained.py``, the scripted episodes put the body back on
its reset pose before every control step, translating rigidly at the commanded
velocity, and then call the env's own ``step``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from myosuite import make_env
from myosuite.terms.base_reward import locomotion_sustained

pytestmark = pytest.mark.tier1

DIRECTIONAL_IDS = (
    "myoLegDirectionalForward-v0",
    "myoLegDirectionalBackward-v0",
    "myoLegDirectionalRandom-v0",
)
# Control steps in one second: the directional walks step at 0.005 s.
WINDOW = 200


def _zero(env: gym.Env) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=np.float32)


def _command(env: gym.Env) -> np.ndarray:
    """Commanded planar velocity of the episode (``heading_reward``'s target)."""
    base = env.unwrapped
    extra = base._task_config.reward.extra
    heading = base._task_state.get("heading_dir", extra["heading_dir"])
    return float(extra["target_speed"]) * np.asarray(heading)


def _held_step(env: gym.Env, qpos: np.ndarray, velocity: np.ndarray) -> dict:
    """One control step from the pose *qpos* moving at the planar *velocity*."""
    base = env.unwrapped
    base.data.qpos[:] = qpos
    base.data.qvel[:] = 0.0
    base.data.qvel[:2] = velocity
    _, _, terminated, _, info = env.step(_zero(env))
    assert not terminated
    return info


def test_a_zero_window_keeps_the_per_step_flag() -> None:
    steps = np.array([0, 1, 2])
    np.testing.assert_array_equal(
        locomotion_sustained(steps, 0.005, 0.0), [False, True, True]
    )
    np.testing.assert_array_equal(
        locomotion_sustained(np.array([199, 200]), 0.005), [False, True]
    )


@pytest.mark.parametrize("env_id", DIRECTIONAL_IDS)
def test_falling_is_not_a_success(env_id: str) -> None:
    """Random-action rollouts fall within about a second: no solved step, 0 %."""
    env = make_env(env_id)
    base = env.unwrapped
    assert base._ctrl_dt == pytest.approx(1.0 / WINDOW)
    rng = np.random.default_rng(0)
    paths = []
    for episode in range(3):
        env.reset(seed=episode)
        solved, on_target, per_term = [], [], []
        terminated = truncated = False
        while not (terminated or truncated):
            action = rng.uniform(0.0, 1.0, _zero(env).shape).astype(np.float32)
            _, _, terminated, truncated, info = env.step(action)
            solved.append(bool(info["solved"]))
            on_target.append(info.get("on_target"))
            per_term.append(bool(info["heading/solved"]))
        assert terminated, "the rollout was expected to fall"
        assert not any(solved), f"{sum(solved)} solved steps in a fall"
        # The per-step flag is the one of the heading term.
        assert on_target == per_term
        paths.append({"env_infos": {"solved": np.array(solved)}})
    assert base.evaluate_success(paths) == 0.0
    env.close()


@pytest.mark.parametrize("env_id", DIRECTIONAL_IDS)
def test_solved_needs_one_second_on_target_without_a_break(env_id: str) -> None:
    """``solved`` turns true after ``WINDOW`` consecutive on-target steps of one episode."""
    env = make_env(env_id)
    env.reset(seed=0)
    qpos = env.unwrapped.data.qpos.copy()
    command = _command(env)

    # Off target on step 30 only: the count starts again.
    off = 30
    infos = [
        _held_step(env, qpos, 0.0 * command if step == off else command)
        for step in range(1, off + WINDOW + 6)
    ]
    solved = np.array([bool(info["solved"]) for info in infos])
    on_target = np.array([info.get("on_target") for info in infos])
    steps = np.arange(1, len(infos) + 1)
    np.testing.assert_array_equal(solved, steps >= off + WINDOW)
    np.testing.assert_array_equal(on_target, steps != off)

    # A reset clears the count, also in the middle of a run on target.
    env.reset(seed=0)
    for _ in range(WINDOW // 2):
        assert not _held_step(env, qpos, command)["solved"]
    env.reset(seed=0)
    command = _command(env)
    solved = [bool(_held_step(env, qpos, command)["solved"]) for _ in range(WINDOW)]
    assert solved == [False] * (WINDOW - 1) + [True]
    env.close()
