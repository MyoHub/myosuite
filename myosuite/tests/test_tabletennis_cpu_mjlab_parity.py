# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""CPU / mjlab parity of the TableTennis episode termination.

The mjlab twin documents ``_dense_channel_done`` as matching the CPU ``_get_done``. They
must agree for every contact-trajectory issue and every combination of time, ball height
and solved flag: a failed rally ends the episode on both backends, a missed ball (MISS)
plays on on both.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest

pytest.importorskip("mjlab")

from myosuite.envs.myo.backends.mjlab import (  # noqa: E402
    register_mjlab_tabletennis as tt_mjlab,
)
from myosuite.envs.myo.tasks.challenge import tabletennis as tt_cpu  # noqa: E402

pytestmark = pytest.mark.tier1

_LABEL = tt_cpu.PingpongContactLabels
_ISSUE = tt_cpu.ContactTrajIssue
# Contact trajectories covering every outcome of evaluate_pingpong_trajectory.
_TRAJECTORIES = {
    "empty": [],  # MISS
    "paddle_only": [{_LABEL.PADDLE}],  # MISS
    "own_only": [{_LABEL.OWN}],  # MISS
    "paddle_twice": [{_LABEL.PADDLE}, set(), {_LABEL.PADDLE}],  # DOUBLE_TOUCH
    "own_half_twice": [{_LABEL.OWN}, set(), {_LABEL.OWN}],  # OWN_HALF
    "opponent_without_paddle": [{_LABEL.OPPONENT}],  # NO_PADDLE
    "returned_ball": [{_LABEL.PADDLE}, {_LABEL.OPPONENT}],  # None: a successful return
}
_FAILURES = {_ISSUE.OWN_HALF, _ISSUE.NO_PADDLE, _ISSUE.DOUBLE_TOUCH}


def _cpu_done(time: float, z: float, solved: bool, trajectory: list) -> bool:
    env = SimpleNamespace(obs_dict={"time": time}, contact_trajectory=trajectory)
    return bool(tt_cpu.TableTennisEnv._get_done(env, z, solved))


@pytest.mark.parametrize("name", sorted(_TRAJECTORIES))
@pytest.mark.parametrize(
    "time,z,solved",
    list(itertools.product((0.5, tt_cpu.MAX_TIME + 0.1), (0.1, 0.9), (False, True))),
)
def test_done_flag_matches_mjlab(
    name: str, time: float, z: float, solved: bool
) -> None:
    trajectory = _TRAJECTORIES[name]
    cpu = _cpu_done(time, z, solved, trajectory)
    mjlab = bool(tt_mjlab._dense_channel_done(time, z, solved, trajectory))
    assert cpu == mjlab, (name, time, z, solved)


def test_trajectories_cover_every_issue_and_failures_terminate() -> None:
    """Guards the table above, and pins the intended behaviour on the CPU env."""
    issues = {tt_cpu.evaluate_pingpong_trajectory(t) for t in _TRAJECTORIES.values()}
    assert issues == _FAILURES | {_ISSUE.MISS, None}
    for trajectory in _TRAJECTORIES.values():
        failed = tt_cpu.evaluate_pingpong_trajectory(trajectory) in _FAILURES
        assert _cpu_done(0.5, 0.9, False, trajectory) == failed
