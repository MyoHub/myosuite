# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""CPU TableTennis: failed-rally termination, ball relaunch, muscle conditions."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import mujoco
import numpy as np
import pytest

import myosuite  # noqa: F401  (registers the envs)
from myosuite.core.muscle_conditions import _peak_force
from myosuite.envs.myo.tasks.challenge.tabletennis import (
    ContactTrajIssue,
    PingpongContactLabels,
    evaluate_pingpong_trajectory,
)
from myosuite.utils import gym

pytestmark = pytest.mark.tier1

P0 = "myoChallengeTableTennisP0-v0"
P1 = "myoChallengeTableTennisP1-v0"
OWN = PingpongContactLabels.OWN
OPPONENT = PingpongContactLabels.OPPONENT
PADDLE = PingpongContactLabels.PADDLE
RETURNED_BALL = [{OWN}, set(), {PADDLE}, set(), {OPPONENT}]


@pytest.fixture(scope="module")
def make_env() -> Iterator[Callable[[str], gym.Env]]:
    """Build each env id once per module (a TableTennis make takes seconds)."""
    envs: dict[str, gym.Env] = {}

    def _get(env_id: str) -> gym.Env:
        if env_id not in envs:
            envs[env_id] = gym.make(env_id)
        return envs[env_id]

    yield _get
    for env in envs.values():
        env.close()


def _zero(env: gym.Env) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=np.float32)


@pytest.mark.parametrize(
    ("trajectory", "issue"),
    [
        ([{OWN}, set(), {OPPONENT}], ContactTrajIssue.NO_PADDLE),
        ([{OWN}, set(), {OWN}], ContactTrajIssue.OWN_HALF),
        ([{OWN}, {OWN}, {OWN}], ContactTrajIssue.OWN_HALF),
        ([{PADDLE}, set(), {PADDLE}], ContactTrajIssue.DOUBLE_TOUCH),
    ],
)
def test_failed_rally_ends_episode(
    make_env: Callable[[str], gym.Env], trajectory: list[set], issue: ContactTrajIssue
) -> None:
    """OWN_HALF / NO_PADDLE / DOUBLE_TOUCH terminate and pay the done penalty."""
    assert evaluate_pingpong_trajectory(trajectory) is issue
    env = make_env(P0)
    env.reset(seed=0)
    env.unwrapped.contact_trajectory = [set(s) for s in trajectory]
    *_, terminated, _, info = env.step(_zero(env))
    assert terminated
    assert info["rwd_dict"]["done"]


def test_ball_in_play_does_not_end_episode(make_env: Callable[[str], gym.Env]) -> None:
    """A serve that has only bounced on the own half (MISS so far) plays on."""
    env = make_env(P0)
    env.reset(seed=0)
    env.unwrapped.contact_trajectory = [{OWN}, set()]
    *_, terminated, _, info = env.step(_zero(env))
    trajectory = env.unwrapped.contact_trajectory
    assert evaluate_pingpong_trajectory(trajectory) is ContactTrajIssue.MISS
    assert not terminated
    assert not info["rwd_dict"]["done"]


@pytest.mark.parametrize("env_id", [P0, P1])
def test_relaunch_serves_again_between_rallies(
    make_env: Callable[[str], gym.Env], env_id: str
) -> None:
    """With rally_count=2 a returned ball is served again; the 2nd return ends."""
    env = make_env(env_id)
    u = env.unwrapped
    u.rally_count = 2
    try:
        env.reset(seed=0)
        pos, dof = u.ball_posadr, u.ball_dofadr
        u.data.qpos[pos + 3 : pos + 7] = [0.0, 1.0, 0.0, 0.0]  # turn the ball
        u.contact_trajectory = [set(s) for s in RETURNED_BALL]
        *_, terminated, _, _ = env.step(_zero(env))
        assert not terminated and u.cur_rally == 1 and u.contact_trajectory == []
        np.testing.assert_array_equal(
            u.data.qpos[pos : pos + 7], u._init_qpos[pos : pos + 7]
        )
        np.testing.assert_array_equal(
            u.data.qpos[pos + 3 : pos + 7], u.model.key_qpos[0][pos + 3 : pos + 7]
        )
        np.testing.assert_array_equal(
            u.data.qvel[dof : dof + 6], u._init_qvel[dof : dof + 6]
        )

        u.contact_trajectory = [set(s) for s in RETURNED_BALL]
        *_, terminated, _, info = env.step(_zero(env))
        assert terminated and info["rwd_dict"]["solved"]
    finally:
        u.rally_count = 1


def test_sarcopenia_halves_muscle_peak_force(
    make_env: Callable[[str], gym.Env],
) -> None:
    """myoSarc* halves the muscles' peak force and leaves the pelvis motors."""
    base = make_env(P0).unwrapped.model
    sarc = make_env("myoSarc" + P0[3:]).unwrapped.model
    muscle = base.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
    np.testing.assert_allclose(
        sarc.actuator_gainprm[muscle, 2], 0.5 * _peak_force(base)[muscle]
    )
    np.testing.assert_array_equal(
        sarc.actuator_gainprm[~muscle], base.actuator_gainprm[~muscle]
    )


def test_fatigue_filters_muscle_controls_and_resets(
    make_env: Callable[[str], gym.Env],
) -> None:
    """myoFati* drives the muscles through the fatigue model and resets it."""
    env = make_env("myoFati" + P0[3:])
    u = env.unwrapped
    muscle = u.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
    env.reset(seed=0)
    action = np.ones(env.action_space.shape, dtype=np.float32)
    for _ in range(5):
        env.step(action)
    fatigue = u.muscle_fatigue
    np.testing.assert_array_equal(u.data.ctrl[muscle], fatigue.MA)
    assert 0.1 < fatigue.MA.max() < 1.0
    env.reset(seed=0)
    np.testing.assert_array_equal(fatigue.MA, 0.0)
    np.testing.assert_array_equal(fatigue.MR, 1.0)
    np.testing.assert_array_equal(fatigue.MF, 0.0)


def test_unknown_muscle_condition_raises() -> None:
    """A condition TableTennis does not implement must not run the healthy env."""
    with pytest.raises(ValueError, match="muscle_condition"):
        gym.make(P0, muscle_condition="reafferentation")
