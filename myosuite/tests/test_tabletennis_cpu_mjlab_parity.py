# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""CPU / mjlab parity of the TableTennis episode termination.

The mjlab rally term (``_rally_done`` on the ``_PingpongTrajectory`` outcome) must agree
with the CPU ``_get_done`` for every contact-trajectory issue and every combination of
time, ball height and solved flag: a failed rally ends the episode on both backends, a
missed ball (MISS) plays on on both.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

from myosuite.envs.myo.backends.mjlab import (  # noqa: E402
    register_mjlab_tabletennis as tt_mjlab,
)
from myosuite.envs.myo.tasks.challenge import tabletennis as tt_cpu  # noqa: E402
from myosuite import make_env  # noqa: E402

pytestmark = pytest.mark.tier1

_CTRL_DT = 0.01  # control step of myoChallengeTableTennisP{0,1,2}-v0
_LABEL = tt_cpu.PingpongContactLabels
# Column of each label in the ``touching_info`` observation / mjlab label flags.
_COLUMN = {
    _LABEL.PADDLE: 0,
    _LABEL.OWN: 1,
    _LABEL.OPPONENT: 2,
    _LABEL.NET: 3,
    _LABEL.GROUND: 4,
    _LABEL.ENV: 5,
}
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


def _mjlab_done(time: float, z: float, solved: bool, trajectory: list) -> bool:
    """The mjlab rally term's ``done`` for the same inputs (one env)."""
    state = tt_mjlab._PingpongTrajectory(1, "cpu")
    for contacts in trajectory:
        labels = torch.zeros(1, 6, dtype=torch.bool)
        for label in contacts:
            labels[0, _COLUMN[label]] = True
        state.update(labels)
    # mjlab counts control steps where CPU compares the sim time with MAX_TIME.
    timed_out = round(time / _CTRL_DT) > round(tt_cpu.MAX_TIME / _CTRL_DT)
    done = tt_mjlab._rally_done(
        torch.tensor([timed_out]),
        torch.tensor([z]),
        torch.tensor([solved]),
        state.outcome,
    )
    return bool(done[0])


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
    mjlab = _mjlab_done(time, z, solved, trajectory)
    assert cpu == mjlab, (name, time, z, solved)


def test_trajectories_cover_every_issue_and_failures_terminate() -> None:
    """Guards the table above, and pins the intended behaviour on the CPU env."""
    issues = {tt_cpu.evaluate_pingpong_trajectory(t) for t in _TRAJECTORIES.values()}
    assert issues == _FAILURES | {_ISSUE.MISS, None}
    for trajectory in _TRAJECTORIES.values():
        failed = tt_cpu.evaluate_pingpong_trajectory(trajectory) in _FAILURES
        assert _cpu_done(0.5, 0.9, False, trajectory) == failed


def test_paddle_target_orientation_is_the_keyframe_orientation() -> None:
    """The paddle_quat reward target equals the held paddle's keyframe orientation."""
    import numpy as np  # noqa: PLC0415

    import myosuite  # noqa: F401, PLC0415

    keyframe = np.asarray(tt_mjlab._tt_reference().paddle_pose[3:7])
    mjlab_target = np.asarray(tt_mjlab._tt_reference().init_paddle_quat)
    env = make_env("myoChallengeTableTennisP0-v0")
    try:
        cpu_target = np.asarray(env.unwrapped.init_paddle_quat)
    finally:
        env.close()
    for target in (mjlab_target, cpu_target):  # q and -q are the same rotation
        assert min(
            np.linalg.norm(target - keyframe), np.linalg.norm(target + keyframe)
        ) == pytest.approx(0.0, abs=1e-3)


def test_ball_and_paddle_inertia_match_the_cpu_model() -> None:
    """The free-body specs keep the CPU model's mass and inertia (no 1e-4 clamp)."""
    import numpy as np  # noqa: PLC0415

    ref = tt_mjlab._reference_model()
    for name, spec_fn in (
        ("pingpong", tt_mjlab._pingpong_spec_fn),
        ("paddle", tt_mjlab._paddle_spec_fn),
    ):
        body = spec_fn().compile().body(name)
        np.testing.assert_allclose(body.mass, ref.body(name).mass, rtol=1e-6)
        np.testing.assert_allclose(body.inertia, ref.body(name).inertia, rtol=1e-6)
