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
        _, _, terminated, _, _ = env.step(torch.zeros(2, 354))
        assert terminated.tolist() == [True, True]
    finally:
        env.close()
        cpu.close()
