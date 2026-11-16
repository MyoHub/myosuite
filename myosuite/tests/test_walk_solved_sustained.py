# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""``solved`` of the walk envs means sustained walking, not one step on target.

No walking policy is needed: the scripted episodes put the body back on its reset pose
before every control step, translating rigidly at a chosen velocity (zero joint and
angular velocity, so every definition of the centre-of-mass velocity agrees), and then
call the env's own ``step``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from myosuite import make_env
from myosuite.terms.base_reward import LOCOMOTION_VEL_TOL

pytestmark = pytest.mark.tier1

WALK_IDS = (
    "myoLegWalk-v0",
    "myoLegRoughTerrainWalk-v0",
    "myoLegHillyTerrainWalk-v0",
    "myoLegStairTerrainWalk-v0",
)
# Control steps in one second: the registered walk ids step at 0.01 s.
WINDOW = 100


def _zero(env: gym.Env) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=np.float32)


def _held_step(env: gym.Env, qpos: np.ndarray, velocity: tuple[float, float]) -> dict:
    """One control step from the pose *qpos* moving at *velocity* (env convention)."""
    base = env.unwrapped
    base.data.qpos[:] = qpos
    base.data.qvel[:] = 0.0
    base.data.qvel[:2] = -np.asarray(velocity)  # the model faces -y
    _, _, terminated, _, info = env.step(_zero(env))
    assert not terminated
    return info


@pytest.mark.parametrize("env_id", WALK_IDS)
def test_falling_is_not_a_success(env_id: str) -> None:
    """Zero-action and random rollouts fall within a second: no solved step, 0 %."""
    env = make_env(env_id)
    base = env.unwrapped
    assert base.dt == pytest.approx(1.0 / WINDOW)
    target = np.array([base.target_x_vel, base.target_y_vel])
    rng = np.random.default_rng(0)
    paths = []
    for episode in range(4):
        env.reset(seed=episode)
        solved, on_target, expected = [], [], []
        for _ in range(WINDOW):
            action = rng.uniform(0.0, 1.0, _zero(env).shape) * (episode % 2)
            _, _, terminated, _, info = env.step(action.astype(np.float32))
            solved.append(bool(info["solved"]))
            on_target.append(info.get("on_target"))
            error = np.linalg.norm(target - info["obs_dict"]["com_vel"][0])
            expected.append(bool(error < LOCOMOTION_VEL_TOL and not terminated))
            if terminated:
                break
        assert terminated, "the rollout was expected to fall"
        assert not any(
            solved
        ), f"{sum(solved)} solved steps in a {len(solved)}-step fall"
        # The instantaneous flag: upright and within the velocity tolerance.
        assert on_target == expected
        paths.append({"env_infos": {"solved": np.array(solved)}})
    assert base.evaluate_success(paths) == 0.0
    env.close()


@pytest.mark.parametrize("env_id", WALK_IDS)
def test_solved_needs_one_second_on_target_without_a_break(env_id: str) -> None:
    """``solved`` turns true after ``WINDOW`` consecutive on-target steps of one episode."""
    env = make_env(env_id)
    base = env.unwrapped
    target = (base.target_x_vel, base.target_y_vel)
    env.reset(seed=0)
    qpos = base.data.qpos.copy()

    # Off target on step 30 only: the count starts again.
    off = 30
    infos = [
        _held_step(env, qpos, (0.0, 0.0) if step == off else target)
        for step in range(1, off + WINDOW + 6)
    ]
    solved = np.array([bool(info["solved"]) for info in infos])
    on_target = np.array([info.get("on_target") for info in infos])
    steps = np.arange(1, len(infos) + 1)
    np.testing.assert_array_equal(solved, steps >= off + WINDOW)
    np.testing.assert_array_equal(on_target, steps != off)
    assert base.evaluate_success([{"env_infos": {"solved": solved}}]) == 100.0

    # A fall ends it: the terminal step is not solved.
    for _ in range(WINDOW):
        _, _, terminated, _, info = env.step(_zero(env))
        if terminated:
            break
    assert terminated and not info["solved"] and not info["on_target"]

    # A reset clears the count, also in the middle of a run on target.
    env.reset(seed=0)
    for _ in range(WINDOW // 2):
        assert not _held_step(env, qpos, target)["solved"]
    env.reset(seed=0)
    solved = [bool(_held_step(env, qpos, target)["solved"]) for _ in range(WINDOW)]
    assert solved == [False] * (WINDOW - 1) + [True]
    env.close()
