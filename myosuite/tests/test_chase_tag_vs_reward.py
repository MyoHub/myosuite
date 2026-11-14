# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The 1v1 ChaseTag reward pays the tag, not staying next to the runner."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_task_config import (
    ChaseTagVsTaskConfig,
)
from myosuite.terms.multiplayer.chase_tag_vs_reward import CHASER_ID, RUNNER_ID

pytestmark = pytest.mark.tier1

_META = SimpleNamespace(
    site_ids={CHASER_ID: {"pelvis_site": 0}, RUNNER_ID: {"pelvis_site": 1}}
)


def _rewards(
    config: ChaseTagVsTaskConfig, distances: list[float], tag_last: bool
) -> np.ndarray:
    """Per-step ``(chaser, runner)`` rewards of a CHASE episode at the given distances."""
    config.on_reset(None, None, None, np.random.default_rng(0))
    out = []
    for step, d in enumerate(distances, start=1):
        data = SimpleNamespace(
            time=0.01 * step, site_xpos=np.array([[0.0, 0, 1], [d, 0, 1]])
        )
        tagged = tag_last and step == len(distances)
        ko = {RUNNER_ID: tagged, CHASER_ID: False}
        out.append(
            [
                config.compute_reward(data, _META, agent, {}, {}, np.zeros(3), {}, ko)
                for agent in (CHASER_ID, RUNNER_ID)
            ]
        )
    return np.array(out)


def _discounted(rewards: np.ndarray, gamma: float) -> float:
    return float(np.sum(rewards * gamma ** np.arange(len(rewards))))


@pytest.mark.parametrize("gamma", [0.99, 1.0])
def test_tagging_now_beats_hovering_outside_the_tag_radius(gamma: float) -> None:
    config = ChaseTagVsTaskConfig(task_choice="CHASE")
    hover = config.tag_radius_m + 0.1
    hovering = _rewards(config, [1.0] + [hover] * (config.max_episode_steps - 1), False)
    tagging = _rewards(config, [1.0, config.tag_radius_m - 0.1], True)
    assert _discounted(tagging[:, 0], gamma) > _discounted(hovering[:, 0], gamma)
    assert _discounted(tagging[:, 1], gamma) < _discounted(hovering[:, 1], gamma)


def test_proximity_terms_telescope_to_the_change_of_distance() -> None:
    config = ChaseTagVsTaskConfig(
        task_choice="CHASE", r_survival_per_step=0.0, r_act_reg=0.0
    )
    distances = [3.0, 2.0, 2.5, 1.0, 1.5]
    total = _rewards(config, distances, False).sum(0)
    change = np.exp(-distances[-1]) - np.exp(-distances[0])
    np.testing.assert_allclose(total, [0.5 * change, -0.5 * change], atol=1e-12)
