# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache 2 license in the root LICENSE file.
"""Physical and goal-contract tests for the unregistered waypoint prototype."""

import gymnasium as gym
import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from myosuite.envs.waypoint import WaypointEnv

pytestmark = pytest.mark.tier1


@pytest.fixture
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep="0.01" gravity="0 0 0"/>
      <worldbody><body name="actor" pos="0 0 0.1">
        <joint name="x" type="slide" axis="1 0 0" damping="1"/>
        <joint name="y" type="slide" axis="0 1 0" damping="1"/>
        <geom type="sphere" size="0.05" mass="1"/>
      </body></worldbody>
      <actuator><motor joint="x" ctrlrange="-10 10"/>
        <motor joint="y" ctrlrange="-10 10"/></actuator>
    </mujoco>""")


def make(model: mujoco.MjModel, **kwargs) -> WaypointEnv:
    return WaypointEnv(model, body_name="actor", frame_skip=1, **kwargs)


def test_controlled_route_and_reset(model: mujoco.MjModel) -> None:
    env = make(model, waypoints=[[0.3, 0], [0.3, 0.3], [0, 0.3]], arrival_radius=0.025)
    first, _ = env.reset(seed=12)
    order = []
    for _ in range(1500):
        delta = env.get_obs_dict()["target_delta"]
        _, _, done, truncated, info = env.step(20 * delta - 7 * env.data.qvel)
        order.append(info["next_waypoint"])
        if done:
            break
    assert info["solved"] and not truncated
    assert set(order) == {0, 1, 2, 3}
    assert env.data.time > 0
    with pytest.raises(gym.error.ResetNeeded):
        env.step(np.zeros(2))
    again, _ = env.reset(seed=12)
    np.testing.assert_array_equal(first, again)
    assert env.next_waypoint == 0 and np.all(env.data.qvel == 0)
    env.close()


def test_skipped_target_does_not_succeed(model: mujoco.MjModel) -> None:
    env = make(
        model, waypoints=[[0.2, 0], [1, 0]], initial_qpos=[1, 0], arrival_radius=0.05
    )
    env.reset()
    _, _, done, _, info = env.step(np.zeros(2))
    assert not done and info["next_waypoint"] == 0
    assert info["distance"] == pytest.approx(0.8)


def test_boundary_and_reads_do_not_advance(model: mujoco.MjModel) -> None:
    env = make(model, waypoints=[[0.125, 0], [0.125, 0]], arrival_radius=0.125)
    env.reset()
    env.forward()
    env.get_reward_dict(env.get_obs_dict())
    assert env.next_waypoint == 0
    _, _, done, _, info = env.step(np.zeros(2))
    assert not done and info["next_waypoint"] == 1
    assert env.step(np.zeros(2))[2]


def test_timeout_truncates(model: mujoco.MjModel) -> None:
    env = gym.wrappers.TimeLimit(make(model, waypoints=[[10, 0]]), max_episode_steps=2)
    env.reset(seed=0)
    assert env.step(np.zeros(2))[2:4] == (False, False)
    assert env.step(np.zeros(2))[2:4] == (False, True)


def test_action_validation_and_clipping(model: mujoco.MjModel) -> None:
    env = make(model, waypoints=[[10, 0]])
    env.reset()
    for action in ([1], [np.nan, 0], [np.inf, 0]):
        with pytest.raises(ValueError):
            env.step(np.asarray(action))
    assert env.data.time == 0
    env.step(np.array([100, -100]))
    np.testing.assert_array_equal(env.data.ctrl, [10, -10])


def test_unlimited_controls(model: mujoco.MjModel) -> None:
    model.actuator_ctrllimited[:] = 0
    env = make(model, waypoints=[[10, 0]])
    assert np.isinf(env.action_space.high).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"waypoints": []},
        {"waypoints": [[0, 0, 0]]},
        {"waypoints": [[np.nan, 0]]},
        {"arrival_radius": 0},
        {"arrival_radius": np.inf},
        {"frame_skip": 0},
        {"frame_skip": 1.5},
        {"body_name": "missing"},
        {"body_name": "world"},
        {"initial_qpos": [0]},
        {"initial_qpos": [np.nan, 0]},
    ],
)
def test_invalid_inputs(model: mujoco.MjModel, kwargs: dict) -> None:
    args = {"waypoints": [[1, 0]], "body_name": "actor", **kwargs}
    with pytest.raises(ValueError):
        WaypointEnv(model, **args)


def test_inputs_copied_and_gymnasium_contract(model: mujoco.MjModel) -> None:
    targets = np.array([[1.0, 0.0]])
    env = make(model, waypoints=targets)
    targets[:] = 9
    np.testing.assert_array_equal(env.waypoints, [[1, 0]])
    check_env(env, skip_render_check=True)


def test_instability_cannot_count_as_success(model: mujoco.MjModel) -> None:
    env = make(model, waypoints=[[0, 0]])
    env.reset()
    env.data.warning[mujoco.mjtWarning.mjWARN_BADQPOS].number = 1
    _, _, terminated, _, info = env.step(np.zeros(2))
    assert terminated and info["failed"] and not info["solved"]
