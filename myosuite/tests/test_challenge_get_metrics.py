# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Challenge ``get_metrics`` keeps the legacy scoring of lost episodes.

Legacy ``run_track_v0`` and ``chasetag_v0`` (CHASE) reported ``maxTime`` as the
episode time of a lost episode, so falling early cannot earn a short time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import myosuite  # noqa: F401
from myosuite.utils import gym

pytestmark = pytest.mark.tier1


def _lost_episode(env_id: str, **kwargs: Any) -> tuple[Any, dict, dict]:
    """Roll out zero actions (the model falls) and stack the infos into a path."""
    env = gym.make(env_id, **kwargs)
    env.reset(seed=0)
    infos = []
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape, np.float32)
        )
        infos.append(info)
    assert terminated and not info["rwd_dict"]["solved"], "the episode must be lost"
    env_infos = {
        group: {k: np.array([i[group][k] for i in infos]) for k in info[group]}
        for group in ("obs_dict", "rwd_dict")
    }
    return env.unwrapped, {"env_infos": env_infos}, info


def test_run_track_lost_episode_reports_max_time() -> None:
    env, path, info = _lost_episode("myoChallengeOslRunFixed-v0")
    sim_time = float(np.ravel(info["obs_dict"]["time"])[-1])
    assert sim_time < env.maxTime  # the observation keeps the simulated time
    assert env.get_metrics([path])["time"] == pytest.approx(env.maxTime)


@pytest.mark.parametrize("task", ["CHASE", "EVADE"])
def test_chasetag_lost_episode_time(task: str) -> None:
    env, path, info = _lost_episode("myoChallengeChaseTagP1-v0", task_choice=task)
    sim_time = float(np.ravel(info["obs_dict"]["time"])[-1])
    assert sim_time < env.maxTime
    # A lost CHASE scores maxTime; an EVADE episode scores the time survived.
    expected = env.maxTime if task == "CHASE" else round(sim_time, 2)
    assert env.get_metrics([path])["times"] == pytest.approx(expected)


def _path(**groups: dict[str, Any]) -> dict:
    """A rollout path whose ``env_infos`` hold the given per-step arrays."""
    env_infos = {
        group: {k: np.asarray(v, float) for k, v in values.items()}
        for group, values in groups.items()
        if group != "touch_history"
    }
    if "touch_history" in groups:
        env_infos["touch_history"] = groups["touch_history"]
    return {"env_infos": env_infos}


@pytest.mark.parametrize(
    "env_id", ["myoChallengeRelocateP1-v0", "myoChallengeDieReorientP1-v0"]
)
def test_relocate_and_reorient_score_by_solved_steps(env_id: str) -> None:
    env = gym.make(env_id).unwrapped
    act_reg = np.full(20, -0.25)
    solved = lambda n: np.r_[np.ones(n), np.zeros(20 - n)]  # noqa: E731
    paths = [
        _path(rwd_dict={"solved": solved(6), "act_reg": act_reg}),
        _path(rwd_dict={"solved": solved(5), "act_reg": act_reg}),
    ]
    metrics = env.get_metrics(paths)  # legacy: solved on more than 5 steps
    assert metrics == {"score": 0.5, "effort": pytest.approx(0.25)}


def test_tabletennis_score_needs_one_solved_step() -> None:
    env = gym.make("myoChallengeTableTennisP0-v0").unwrapped
    act_reg = np.full(10, -0.5)
    paths = [
        _path(rwd_dict={"solved": np.r_[np.zeros(9), 1.0], "act_reg": act_reg}),
        _path(rwd_dict={"solved": np.zeros(10), "act_reg": act_reg}),
    ]
    assert env.get_metrics(paths) == {"score": 0.5, "effort": pytest.approx(0.5)}


def test_baoding_score_is_the_solved_fraction_of_the_horizon() -> None:
    env = gym.make("myoChallengeBaodingP1-v1")
    horizon = env.spec.max_episode_steps
    solved = np.zeros(horizon)
    solved[: horizon // 4] = 1.0
    path = _path(rwd_dict={"solved": solved, "act_reg": np.full(horizon, -0.1)})
    metrics = env.unwrapped.get_metrics([path])
    assert metrics == {"score": pytest.approx(0.25), "effort": pytest.approx(0.1)}


def test_bimanual_score_requires_a_clean_contact_history() -> None:
    from myosuite.envs.myo.tasks.challenge.bimanual import (
        CONTACT_TRAJ_MIN_LENGTH,
        GOAL_CONTACT,
        ContactTrajIssue,
        ObjLabels,
        evaluate_contact_trajectory,
    )

    n = CONTACT_TRAJ_MIN_LENGTH + GOAL_CONTACT
    good = [{ObjLabels.MYO, ObjLabels.PROSTH}] * CONTACT_TRAJ_MIN_LENGTH
    good += [{ObjLabels.GOAL}] * GOAL_CONTACT
    assert evaluate_contact_trajectory(good) is None
    assert evaluate_contact_trajectory(good + [{ObjLabels.ENV}]) == (
        ContactTrajIssue.ENV_CONTACT
    )
    assert evaluate_contact_trajectory(good[10:]) == ContactTrajIssue.MYO_SHORT
    assert evaluate_contact_trajectory(good[:-GOAL_CONTACT]) == (
        ContactTrajIssue.NO_GOAL
    )

    env = gym.make("myoChallengeBimanual-v0").unwrapped
    base = {
        "obs_dict": {"time": np.linspace(0, 2, n), "max_force": np.linspace(0, 30, n)},
        "rwd_dict": {
            "solved": np.r_[np.zeros(n - 6), np.ones(6)],
            "act": np.full(n, 0.3),
            "goal_dist": np.full(n, 0.1),
        },
    }
    ok = _path(**base, touch_history=good)
    bad = _path(**base, touch_history=good + [{ObjLabels.ENV}])
    metrics = env.get_metrics([ok, bad])
    assert metrics == {
        "score": 0.5,
        "time": pytest.approx(2.0),
        "effort": pytest.approx(0.3),
        "peak force": pytest.approx(30.0),
        "goal dist": pytest.approx(0.1),
    }
