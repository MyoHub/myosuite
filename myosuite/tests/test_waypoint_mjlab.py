# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache 2 license in the root LICENSE file.
"""CPU <-> mjlab parity of the waypoint twin (``myoFullBodyWaypoint-v0``).

As for the basic-suite twins, parity is checked one step at a time: the twin is
synced to the CPU state and route, both take the same action, and observation,
reward, termination and route progress are compared.
"""

from __future__ import annotations

import numpy as np
import mujoco
import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.tier2

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

from myosuite import make_env  # noqa: E402
from myosuite.envs.myo.backends.mjlab.tasks.mdp import write_cpu_state  # noqa: E402
from myosuite.envs.myo.backends.mjlab.tasks.waypoint.mdp import COMMAND  # noqa: E402

ENV_ID = "myoFullBodyWaypoint-v0"
# Float32 Warp vs float64 MuJoCo with foot contacts after one 10 ms step.
OBS_ATOL, REW_ATOL = 5e-3, 5e-3


def _t(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x)[None], dtype=torch.float32)


@pytest.fixture(scope="module")
def pair():
    import myosuite.envs.myo.backends.mjlab  # noqa: F401, PLC0415 (registers twins)

    cpu = make_env(ENV_ID).unwrapped
    mj = ManagerBasedRlEnv(cfg=load_env_cfg(ENV_ID), device="cpu")
    return cpu, mj


def _sync(cpu, mj: ManagerBasedRlEnv) -> None:
    write_cpu_state(
        mj,
        "robot",
        torch.zeros(1, dtype=torch.long),
        _t(cpu.data.qpos),
        _t(cpu.data.qvel),
    )
    mj.sim.data.act[:] = _t(cpu.data.act)
    route = mj.command_manager.get_term(COMMAND)
    route.waypoints[:] = _t(cpu.waypoints)
    route.next_index[:] = int(cpu._task_state["next_index"])
    route.prev_distance[:] = float(cpu._task_state["prev_distance"])
    mj.episode_length_buf[:] = round(cpu.data.time / mj.step_dt)
    mj.sim.forward()


def test_contract_matches(pair) -> None:
    cpu, mj = pair
    assert mj.action_manager.total_action_dim == cpu.action_space.shape[0]
    assert (
        mj.observation_manager.group_obs_dim["actor"][0]
        == cpu.observation_space.shape[0]
    )
    assert cpu._ctrl_dt == pytest.approx(float(mj.step_dt))
    assert mj.max_episode_length == make_env(ENV_ID).spec.max_episode_steps
    route = mj.command_manager.get_term(COMMAND)
    np.testing.assert_allclose(route._start.numpy(), cpu._start, atol=1e-6)
    assert float(route._start_yaw) == pytest.approx(cpu._start_yaw)


def test_reset_routes_follow_the_cpu_distribution(pair) -> None:
    """Same start, segment lengths and turns (the backends agree in distribution)."""
    cpu, mj = pair
    mj.reset()
    route = mj.command_manager.get_term(COMMAND).waypoints.numpy()[0]
    cfg = cpu.task.route
    steps = np.diff(np.vstack([cpu._start, route]), axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    assert np.all(
        (lengths >= cfg.segment_length[0] - 1e-5)
        & (lengths <= cfg.segment_length[1] + 1e-5)
    )
    first = np.arctan2(steps[0, 1], steps[0, 0]) - cpu._start_yaw - cfg.heading_offset
    assert (
        cfg.turn_angle[0] - 1e-5
        <= (first + np.pi) % (2 * np.pi) - np.pi
        <= cfg.turn_angle[1] + 1e-5
    )


def test_one_step_parity(pair) -> None:
    """Same state + same action -> same obs, reward, termination and progress."""
    cpu, mj = pair
    cpu.reset(seed=0)
    mj.reset()
    _sync(cpu, mj)
    obs0 = mj.observation_manager.compute_group("actor")[0].numpy()
    np.testing.assert_allclose(
        obs0, cpu._obs_dict_to_vec(cpu.get_obs_dict()), atol=1e-5
    )
    rng = np.random.default_rng(0)
    for step in range(15):
        if step == 5:  # put the next waypoint under the pelvis: an arrival step
            cpu._waypoints.flags.writeable = True
            cpu._waypoints[0] = cpu.data.site_xpos[cpu._site_id, :2]
        _sync(cpu, mj)
        action = rng.uniform(-1, 1, cpu.action_space.shape).astype(np.float32)
        cpu_obs, cpu_rew, cpu_term, _, info = cpu.step(action)
        mj_obs, mj_rew, mj_term, _, _ = mj.step(torch.as_tensor(action[None]))
        assert bool(mj_term[0]) == cpu_term
        if cpu_term:
            break
        route = mj.command_manager.get_term(COMMAND)
        assert int(route.next_index[0]) == info["next_waypoint"]
        np.testing.assert_allclose(mj_obs["actor"][0].numpy(), cpu_obs, atol=OBS_ATOL)
        np.testing.assert_allclose(float(mj_rew[0]), cpu_rew, atol=REW_ATOL)
    assert cpu.next_waypoint == 1


def test_fall_fails_on_both(pair) -> None:
    """Zero drive: both backends fail on the same step, with the -10 penalty."""
    cpu, mj = pair
    cpu.reset(seed=0)
    mj.reset()
    action = np.zeros(cpu.action_space.shape, dtype=np.float32)
    for _ in range(200):
        _sync(cpu, mj)
        _, cpu_rew, cpu_term, _, info = cpu.step(action)
        _, mj_rew, mj_term, _, _ = mj.step(torch.as_tensor(action[None]))
        assert bool(mj_term[0]) == cpu_term
        if cpu_term:
            break
    assert info["failed"]
    np.testing.assert_allclose(float(mj_rew[0]), cpu_rew, atol=REW_ATOL)


def _slower_scene(spec: mujoco.MjSpec) -> None:
    spec.option.timestep = 0.004


def test_instance_overrides_share_one_id() -> None:
    from myosuite.core.config import EnvConfig
    from myosuite.envs.waypoint import WaypointTaskCfg

    task = WaypointTaskCfg(waypoints=((0.0, -1.0),), arrival_radius=0.12)
    config = EnvConfig(
        ENV_ID,
        task_kwargs={"task": task, "edit_fn": _slower_scene},
        ctrl_dt=0.012,
        max_episode_steps=20,
    )
    cpu = make_env(config)
    mj = make_env(config, backend="mjlab", num_envs=2, device="cpu")
    route = mj.command_manager.get_term(COMMAND)
    assert route.task == cpu.unwrapped.task == task
    assert mj.step_dt == pytest.approx(cpu.unwrapped._ctrl_dt)
    assert mj.max_episode_length == 20
    mj.reset()
    np.testing.assert_allclose(route.waypoints.numpy(), [[[0.0, -1.0]], [[0.0, -1.0]]])
    # Defaults are unchanged for the next instance.
    assert make_env(ENV_ID).unwrapped.task.waypoints is None
    cpu.close()
    mj.close()
