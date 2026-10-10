# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache 2 license in the root LICENSE file.
"""Goal, reward and reset contract of the CPU waypoint env and its shared terms."""

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from myosuite.envs.waypoint import WaypointEnv, WaypointTaskCfg
from myosuite.terms.waypoint import (
    WaypointRouteCfg,
    sample_route,
    select_waypoints,
    waypoint_progress,
    waypoint_targets_obs,
)

pytestmark = pytest.mark.tier1

SLIDER = """
<mujoco><option timestep="0.01" gravity="0 0 0"/>
  <worldbody><body name="actor" pos="0 0 0.1">
    <joint name="x" type="slide" axis="1 0 0" damping="1"/>
    <joint name="y" type="slide" axis="0 1 0" damping="1"/>
    <geom type="sphere" size="0.05" mass="1"/><site name="tip"/>
  </body></worldbody>
  <actuator><motor joint="x" ctrlrange="-10 10"/>
    <motor joint="y" ctrlrange="-10 10"/></actuator>
</mujoco>"""

FREE = """
<mujoco><option timestep="0.01"/>
  <worldbody><geom type="plane" size="5 5 .1"/>
    <body name="root" pos="0 0 1"><freejoint/><geom type="sphere" size=".1"/>
      <site name="tip"/>
      <body pos="0 0 -.5"><joint name="hinge" range="-1 1"/><geom type="capsule" size=".05" fromto="0 0 0 0 0 -.3"/></body>
    </body></worldbody>
  <actuator><motor joint="hinge" ctrlrange="-1 1"/></actuator>
</mujoco>"""


@pytest.fixture
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(SLIDER)


def make(model: mujoco.MjModel, **task) -> WaypointEnv:
    cfg = WaypointTaskCfg(site_name="tip", lookahead=1, **task)
    return WaypointEnv(model, task=cfg, frame_skip=1)


def pd(env: WaypointEnv) -> np.ndarray:
    return 20 * env.get_obs_dict()["waypoint_targets"][:2] - 7 * env.data.qvel


def test_controlled_route_solves_and_resets(model: mujoco.MjModel) -> None:
    env = make(model, waypoints=((0.3, 0), (0.3, 0.3), (0, 0.3)), arrival_radius=0.025)
    first, _ = env.reset(seed=12)
    order, ret = [], 0.0
    for _ in range(1500):
        _, r, done, _, info = env.step(pd(env))
        order.append(info["next_waypoint"])
        ret += r
        if done:
            break
    assert info["solved"] and set(order) == {0, 1, 2, 3}
    # Progress telescopes to the 0.9 m route minus each arrival distance, plus 3 arrivals.
    assert 3 + 0.9 - 3 * 0.025 <= ret <= 3 + 0.9
    with pytest.raises(gym.error.ResetNeeded):
        env.step(np.zeros(2))
    again, _ = env.reset(seed=12)
    np.testing.assert_array_equal(first, again)
    assert env.next_waypoint == 0 and np.all(env.data.qvel == 0)


def test_failing_never_beats_progress(model: mujoco.MjModel) -> None:
    """Ending early by failing must not pay more than heading for the target."""
    env = make(model, waypoints=((1, 0),))
    env.reset()
    _, r_move, *_ = env.step(np.array([10.0, 0]))
    env.reset()
    env.data.warning[mujoco.mjtWarning.mjWARN_BADQPOS].number = 1
    _, r_fail, terminated, _, info = env.step(np.zeros(2))
    assert terminated and info["failed"] and not info["solved"]
    assert r_fail < 0 < r_move


def test_no_reward_jump_on_arrival(model: mujoco.MjModel) -> None:
    env = make(model, waypoints=((0.3, 0), (0.3, 1.0)), arrival_radius=0.025)
    env.reset()
    rewards = []
    while True:
        _, r, _, _, info = env.step(pd(env))
        rewards.append(r - info["arrived"])
        if info["next_waypoint"] == 1:
            break
    for _ in range(3):
        rewards.append(env.step(pd(env))[1])
    assert max(abs(r) for r in rewards[-5:]) < 0.05


def test_distance_reported_at_reset(model: mujoco.MjModel) -> None:
    env = make(model, waypoints=((3, 4),))
    env.reset()
    _, r, *_, info = env.forward()
    assert info["distance"] == pytest.approx(5.0) and r == 0


def test_skipped_target_does_not_count(model: mujoco.MjModel) -> None:
    env = WaypointEnv(
        model,
        task=WaypointTaskCfg(
            site_name="tip", waypoints=((0.2, 0), (1, 0)), arrival_radius=0.05
        ),
        initial_qpos=[1, 0],
        frame_skip=1,
    )
    env.reset()
    _, _, done, _, info = env.step(np.zeros(2))
    assert not done and info["next_waypoint"] == 0
    assert info["distance"] == pytest.approx(0.8)


def test_boundary_and_reads_do_not_advance(model: mujoco.MjModel) -> None:
    env = make(model, waypoints=((0.125, 0), (0.125, 0)), arrival_radius=0.125)
    env.reset()
    env.forward()
    env.get_reward_dict(env.get_obs_dict())
    assert env.next_waypoint == 0
    _, _, done, _, info = env.step(np.zeros(2))
    assert not done and info["next_waypoint"] == 1
    assert env.step(np.zeros(2))[2]


def test_step_kwargs_and_validation(model: mujoco.MjModel) -> None:
    env = make(model, waypoints=((10, 0),))
    env.reset()
    for action in ([1], [np.nan, 0]):
        with pytest.raises(ValueError):
            env.step(np.asarray(action))
    env.step(np.array([100, -100]), update_exteroception=True)
    np.testing.assert_array_equal(env.data.ctrl, [10, -10])


def test_random_routes_follow_the_seed(model: mujoco.MjModel) -> None:
    env = make(model, route=WaypointRouteCfg(3, (0.5, 1.0)), reset_noise=0.05)
    env.reset(seed=3)
    a = env.waypoints.copy(), env.data.qpos.copy()
    env.reset(seed=3)
    np.testing.assert_array_equal(a[0], env.waypoints)
    np.testing.assert_array_equal(a[1], env.data.qpos)
    env.reset(seed=4)
    assert not np.array_equal(a[0], env.waypoints)
    steps = np.linalg.norm(np.diff(np.vstack([[0, 0], env.waypoints]), axis=0), axis=1)
    assert np.all((steps >= 0.5) & (steps <= 1.0))


def test_translation_invariant_heading_frame_obs() -> None:
    """The obs omit the root's world x, y and give targets in the root's heading frame."""
    m = mujoco.MjModel.from_xml_string(FREE)
    env = WaypointEnv(m, task=WaypointTaskCfg(site_name="tip", waypoints=((1, 0),)))
    env.reset(seed=0)
    base = env.get_obs_dict()
    env.data.qpos[:2] += 0.5
    env.data.qpos[3:7] = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]  # yaw 90 deg
    env.waypoints.flags.writeable = True
    env.waypoints[:] = [[1.5, 0.5]]
    env.forward()
    moved = env.get_obs_dict()
    assert moved["qpos"].shape == (m.nq - 2,)
    # Target 1 m along world x = 1 m to the root's right after a 90 deg yaw.
    np.testing.assert_allclose(base["waypoint_targets"][:2], [1, 0], atol=1e-9)
    np.testing.assert_allclose(moved["waypoint_targets"][:2], [0, -1], atol=1e-9)


def test_falling_fails() -> None:
    m = mujoco.MjModel.from_xml_string(FREE)
    task = WaypointTaskCfg(site_name="tip", waypoints=((5, 0),), min_site_height=0.95)
    env = WaypointEnv(m, task=task)
    env.reset()
    for _ in range(200):
        _, r, terminated, _, info = env.step(np.zeros(1))
        if terminated:
            break
    assert info["failed"] and r == pytest.approx(-10, abs=0.1)


@pytest.mark.parametrize(
    "task",
    [
        {"waypoints": ()},
        {"waypoints": ((0, 0, 0),)},
        {"waypoints": ((np.nan, 0),)},
        {"arrival_radius": 0},
        {"arrival_radius": np.inf},
        {"lookahead": 0},
        {"reset_noise": -1},
        {"obs_keys": ("bogus",)},
    ],
)
def test_invalid_task(task: dict) -> None:
    with pytest.raises(ValueError):
        WaypointTaskCfg(**task)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_skip": 0},
        {"frame_skip": 1.5},
        {"initial_qpos": [0]},
        {"initial_qpos": [np.nan, 0]},
        {"task": WaypointTaskCfg(site_name="missing")},
        {"model_recipe": "musclemimic_fullbody"},
    ],
)
def test_invalid_inputs(model: mujoco.MjModel, kwargs: dict) -> None:
    args = {"task": WaypointTaskCfg(site_name="tip"), **kwargs}
    with pytest.raises(ValueError):
        WaypointEnv(model, **args)


def test_gymnasium_contract(model: mujoco.MjModel) -> None:
    check_env(make(model, waypoints=((1, 0),)), skip_render_check=True)


def test_terms_batch_matches_single() -> None:
    """The shared terms give the same result per env with a leading batch axis."""
    rng = np.random.default_rng(0)
    route = WaypointRouteCfg(4)
    u = rng.random((2, 3, 4))
    start, yaw = rng.normal(size=(3, 2)), rng.normal(size=3)
    batch = sample_route(np, start, yaw, u[0], u[1], route)
    index = np.array([0, 2, 4])
    pos = rng.normal(size=(3, 2))
    obs = waypoint_targets_obs(np, pos, yaw, batch, index, 2)
    res = waypoint_progress(np, pos, batch, index, np.ones(3), 0.5, np.zeros(3, bool))
    for i in range(3):
        single = sample_route(np, start[i], yaw[i], u[0, i], u[1, i], route)
        np.testing.assert_allclose(batch[i], single)
        np.testing.assert_allclose(
            obs[i], waypoint_targets_obs(np, pos[i], yaw[i], single, index[i], 2)
        )
        one = waypoint_progress(
            np, pos[i], single, index[i], np.asarray(1.0), 0.5, np.asarray(False)
        )
        for key, value in res.items():
            np.testing.assert_allclose(value[i], one[key])
    np.testing.assert_allclose(
        select_waypoints(np, batch, index[:, None])[1, 0], batch[1, 2]
    )
