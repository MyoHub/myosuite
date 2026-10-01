"""Random finger-reach targets are sampled from the reachable workspace on both backends."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from myosuite.utils.reach_workspace import reachable_target_points

pytestmark = pytest.mark.tier1

_IDS = ("myoFingerReachRandom-v0", "motorFingerReachRandom-v0")
_BOX = (np.array([[0.1, -0.1, 0.1]]), np.array([[0.27, 0.1, 0.3]]))


@pytest.mark.parametrize("env_id", _IDS)
def test_cpu_targets_come_from_the_reachable_table(env_id: str) -> None:
    env = gym.make(env_id)
    u = env.unwrapped
    assert u.target_sampling == "workspace"
    points = u._workspace_points.reshape(-1, 3)
    low, high = _BOX
    assert ((points >= low) & (points <= high)).all()
    for seed in range(5):
        env.reset(seed=seed)
        target = u.data.site_xpos[u.target_sids[0]]
        assert np.abs(points - target).sum(axis=1).min() < 1e-9
    env.close()


def test_table_is_deterministic_and_covers_most_of_the_box() -> None:
    u = gym.make("motorFingerReachRandom-v0").unwrapped
    sites = [u.model.site("IFtip").id]
    a = reachable_target_points(u.model, sites, *_BOX)
    b = reachable_target_points(u.model, sites, *_BOX, seed=0)
    np.testing.assert_array_equal(a, b)
    assert 20_000 < len(a)  # a large share of the joint space lands in the box


@pytest.mark.parametrize("env_id", _IDS)
def test_mjlab_twin_samples_the_same_table(env_id: str) -> None:
    pytest.importorskip("mjlab")
    from myosuite.tests.test_mjlab_cpu_twins import _make_pair

    cpu, mj = _make_pair(env_id)
    command = mj.command_manager.get_term("reach")
    assert type(command).__name__ == "WorkspaceReachTargetCommand"
    cpu.reset(seed=0)
    table = cpu.unwrapped._workspace_points.reshape(-1, 3)
    np.testing.assert_allclose(command._points.cpu().numpy(), table, atol=1e-5)
    mj.reset()
    target = command.command.cpu().numpy()[0]
    assert np.abs(table - target).sum(axis=1).min() < 1e-5


def test_fixed_and_other_random_reach_envs_keep_box_sampling() -> None:
    for env_id in (
        "myoFingerReachFixed-v0",
        "myoHandReachRandom-v0",
        "myoArmReachRandom-v0",
    ):
        assert gym.make(env_id).unwrapped.target_sampling == "box"
