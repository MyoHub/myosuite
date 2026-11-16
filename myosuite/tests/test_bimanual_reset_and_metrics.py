# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Bimanual: the first reset draws the box size, and a solved rollout scores.

The handover is scripted: the box is held at the myoHand palm, then at the
prosthesis palm, then released above the goal pillar. The env ends the episode
on the step after the first solved one, so the path has two solved steps.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from myosuite import make_env
from myosuite.envs.myo.tasks.challenge.bimanual import (
    CONTACT_TRAJ_MIN_LENGTH,
    evaluate_contact_trajectory,
)
from myosuite.utils.tensor_utils import stack_tensor_dict_list

pytestmark = pytest.mark.tier1

ENV_ID = "myoChallengeBimanual-v0"


def _box_and_rollout(env: Any, steps: int = 25) -> tuple[np.ndarray, ...]:
    """Reset with seed 0, let the box drop on the start pillar; box and observations."""
    unwrapped = env.unwrapped
    observations = [env.reset(seed=0)[0]]
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    observations += [env.step(action)[0] for _ in range(steps)]
    model, body = unwrapped.model, unwrapped.obj_bid
    return (
        model.geom_size[unwrapped.obj_gid].copy(),
        model.body_mass[body].copy(),
        model.body_inertia[body].copy(),
        np.array(observations),
    )


@pytest.mark.parametrize("env_id", [ENV_ID, "myoFatiChallengeBimanual-v0"])
def test_first_reset_draws_the_box_like_later_resets(env_id: str) -> None:
    episodes = []
    for _ in range(2):  # two fresh instances, two resets each
        env = make_env(env_id)
        nominal = env.unwrapped.model.geom_size[env.unwrapped.obj_gid].copy()
        episodes += [_box_and_rollout(env), _box_and_rollout(env)]
        env.close()
    first = episodes[0]
    assert not np.allclose(first[0], nominal)
    for episode in episodes[1:]:
        for value, expected in zip(episode, first):
            np.testing.assert_array_equal(value, expected)


@pytest.fixture(scope="module")
def handover() -> Iterator[tuple[Any, list[dict], bool]]:
    """Infos of a scripted handover that ends on the goal pillar."""
    env = make_env(ENV_ID)
    unwrapped = env.unwrapped
    env.reset(seed=0)
    joint = int(unwrapped.model.body(unwrapped.obj_bid).jntadr[0])
    qpos_adr = int(unwrapped.model.jnt_qposadr[joint])
    dof_adr = int(unwrapped.model.jnt_dofadr[joint])
    data = unwrapped.data
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    infos: list[dict] = []

    def place(position: np.ndarray) -> None:
        data.qpos[qpos_adr : qpos_adr + 7] = [*position, 1.0, 0.0, 0.0, 0.0]
        data.qvel[dof_adr : dof_adr + 6] = 0.0

    for _ in range(CONTACT_TRAJ_MIN_LENGTH + 10):
        place(data.site_xpos[unwrapped.palm_sid])
        infos.append(env.step(action)[4])
    for _ in range(CONTACT_TRAJ_MIN_LENGTH + 10):
        palm = data.site_xpos[[unwrapped.rpalm1_sid, unwrapped.rpalm2_sid]].mean(axis=0)
        place(palm)
        infos.append(env.step(action)[4])
    place(np.r_[unwrapped.goal_pos[:2], unwrapped.PILLAR_HEIGHT + 0.02])
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(action)
        infos.append(info)
    yield unwrapped, infos, terminated
    env.close()


def test_solved_rollout_ends_after_two_solved_steps_and_stacks(handover) -> None:
    _, infos, terminated = handover
    assert terminated
    assert {type(info["rwd_dict"]["solved"]) for info in infos} == {bool}
    solved = stack_tensor_dict_list([info["rwd_dict"] for info in infos])["solved"]
    assert solved[-1] and solved.sum() == 2


def test_solved_rollout_scores_in_get_metrics(handover) -> None:
    unwrapped, infos, _ = handover
    history = infos[-1]["touch_history"]
    assert evaluate_contact_trajectory(history) is None
    path = {
        "env_infos": {
            "obs_dict": {
                k: np.array([np.ravel(info["obs_dict"][k]) for info in infos])
                for k in ("time", "max_force")
            },
            "rwd_dict": {
                k: np.array([float(np.ravel(info["rwd_dict"][k])[0]) for info in infos])
                for k in ("solved", "act", "goal_dist")
            },
            "touch_history": history,
        }
    }
    metrics = unwrapped.get_metrics([path])
    assert metrics["score"] == 1.0
    assert metrics["time"] == pytest.approx(unwrapped.data.time, abs=1e-5)
