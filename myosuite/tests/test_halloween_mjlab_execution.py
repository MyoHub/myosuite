"""Release checks for the actual MJLab task, executable through Warp on CPU."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

pytestmark = pytest.mark.tier2


def test_vectorized_mjlab_observation_parity_and_physical_hit() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("mjlab")
    from mjlab.envs import ManagerBasedRlEnv

    from myosuite import make_env
    from myosuite.envs.myo.backends.mjlab.register_mjlab_halloween import (
        make_halloween_env_cfg,
    )

    cpu = make_env("myoHalloween-v0")
    env = ManagerBasedRlEnv(make_halloween_env_cfg(num_envs=2), device="cpu")
    try:
        initial, _ = cpu.reset(seed=13)
        observations, _ = env.reset(seed=13)
        np.testing.assert_allclose(
            initial, observations["policy"][0].numpy(), atol=1e-6, rtol=0
        )
        for _ in range(5):
            observations, rewards, terminated, _, _ = env.step(
                torch.full((2, 354), 0.04)
            )
            assert torch.isfinite(rewards).all() and not terminated.any()
            # Compare identical states; float32 Warp and float64 native trajectories
            # can diverge numerically, which is a different release measurement.
            state = env.sim.data
            raw = cpu.unwrapped
            raw.data.qpos[:] = state.qpos[0].numpy()
            raw.data.qvel[:] = state.qvel[0].numpy()
            raw.data.act[:] = state.act[0].numpy()
            mujoco.mj_forward(raw.model, raw.data)
            obs = raw._obs_dict_to_vec(raw._get_obs_dict(None)).astype(np.float32)
            np.testing.assert_allclose(
                obs, observations["policy"][0].numpy(), atol=1e-6, rtol=0
            )
        robot, pumpkin = env.scene["halloween_fullbody"], env.scene["halloween_pumpkin"]
        head = robot.find_bodies("^head$")[0][0]
        state = torch.zeros(2, 13)
        state[:, :3] = robot.data.body_link_pos_w[:, head]
        state[:, 3] = 1.0
        pumpkin.write_root_state_to_sim(state)
        reset_obs, _, terminated, _, _ = env.step(torch.zeros(2, 354))
        assert terminated.tolist() == [True, True]
        np.testing.assert_allclose(
            reset_obs["policy"][0].numpy(), initial, atol=1e-6, rtol=0
        )
        _, _, terminated, _, _ = env.step(torch.zeros(2, 354))
        assert not terminated.any()
    finally:
        env.close()
        cpu.close()


@pytest.mark.parametrize("custom_spawn", [False, True])
def test_partial_episode_reset_restores_robot_and_keeps_other_world(
    custom_spawn: bool,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("mjlab")
    from mjlab.envs import ManagerBasedRlEnv

    from myosuite.envs.myo.backends.mjlab.register_mjlab_halloween import (
        make_halloween_env_cfg,
    )
    from myosuite.envs.myo.tasks.halloween.config import HALLOWEEN

    cfg = make_halloween_env_cfg(num_envs=2)
    if custom_spawn:
        initial = cfg.scene.entities["halloween_fullbody"].init_state
        initial.pos = (initial.pos[0] + 0.25, initial.pos[1] - 0.25, initial.pos[2])
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        env.reset(seed=13)
        robot = env.scene["halloween_fullbody"]
        ids = torch.tensor([1])
        for _ in range(2):
            env.step(torch.full((2, 354), 0.4))
            fallen = robot.data.default_root_state[ids].clone()
            fallen[:, :3] += env.scene.env_origins[ids]
            fallen[:, 2] = 0.2
            fallen[:, 7:] = 0.5
            robot.write_root_state_to_sim(fallen, env_ids=ids)
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos[ids] + 0.1,
                torch.full_like(robot.data.default_joint_vel[ids], 0.5),
                env_ids=ids,
            )
            untouched = {
                key: getattr(env.sim.data, key)[0].clone()
                for key in ("qpos", "qvel", "act", "ctrl")
            }
            env.reset(env_ids=ids)
            expected = robot.data.default_root_state[ids].clone()
            expected[:, :3] += env.scene.env_origins[ids]
            torch.testing.assert_close(
                robot.data.root_link_pose_w[ids], expected[:, :7]
            )
            torch.testing.assert_close(robot.data.root_link_vel_w[ids], expected[:, 7:])
            torch.testing.assert_close(
                robot.data.joint_pos[ids], robot.data.default_joint_pos[ids]
            )
            torch.testing.assert_close(
                robot.data.joint_vel[ids], robot.data.default_joint_vel[ids]
            )
            assert torch.count_nonzero(env.sim.data.act[ids]) == 0
            assert torch.count_nonzero(env.sim.data.ctrl[ids]) == 0
            for key, state in untouched.items():
                torch.testing.assert_close(
                    getattr(env.sim.data, key)[0], state, rtol=0, atol=0
                )
            pumpkin = env.scene["halloween_pumpkin"].data
            torch.testing.assert_close(
                pumpkin.root_link_pos_w[ids],
                env.scene.env_origins[ids] + torch.tensor(HALLOWEEN.park_pos),
            )
    finally:
        env.close()
