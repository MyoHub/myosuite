# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for experimental Newton torso pose, leg stand-and-reach, and terrain locomotion."""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest

from myosuite import make_env

pytestmark = pytest.mark.tier2


def _require_newton_and_torch():
    if os.environ.get("MYOSUITE_REQUIRE_NEWTON") == "1":
        importlib.import_module("newton")
        wp = importlib.import_module("warp")
        torch = importlib.import_module("torch")
    else:
        pytest.importorskip("newton")
        wp = pytest.importorskip("warp")
        torch = pytest.importorskip("torch")
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    return torch, device


@pytest.mark.parametrize("env_id", ["myoTorsoPoseFixed-v0", "myoTorsoExoPoseFixed-v0"])
def test_torso_pose_matches_cpu(env_id: str) -> None:
    torch, device = _require_newton_and_torch()
    from myosuite.integrations.newton import NewtonTaskEnv

    cpu_env = make_env(env_id)
    n_env = NewtonTaskEnv(env_id, num_envs=2, device=device, seed=13, auto_reset=False)
    try:
        obs_c, _ = cpu_env.reset(seed=13)
        obs_n, _ = n_env.reset_torch(seed=13)
        np.testing.assert_allclose(obs_n[0].detach().cpu().numpy(), obs_c, atol=1e-4)

        rng = np.random.default_rng(17)
        for _ in range(6):
            act = rng.uniform(-1.0, 1.0, size=(n_env.num_actions,)).astype(np.float32)
            obs_c, r_c, term_c, trunc_c, info_c = cpu_env.step(act)
            act_batch = np.stack([act, -0.5 * act], axis=0)
            obs_n, r_n, term_n, trunc_n, info_n = n_env.step_torch(
                torch.as_tensor(
                    act_batch, dtype=torch.float32, device=n_env.torch_device
                )
            )
            np.testing.assert_allclose(
                obs_n[0].detach().cpu().numpy(), obs_c, atol=2e-3, rtol=2e-3
            )
            np.testing.assert_allclose(
                float(r_n[0].detach().cpu().item()), float(r_c), atol=5e-3, rtol=5e-3
            )
            assert bool(term_n[0].detach().cpu().item()) == bool(term_c)
            assert bool(trunc_n[0].detach().cpu().item()) == bool(trunc_c)
            assert bool(info_n["solved"][0].detach().cpu().item()) == bool(
                info_c.get("solved", False)
            )
    finally:
        cpu_env.close()


def test_leg_stand_random_matches_cpu() -> None:
    torch, device = _require_newton_and_torch()
    from myosuite.integrations.newton import NewtonTaskEnv

    env_id = "myoLegStandRandom-v0"
    cpu_env = make_env(env_id)
    n_env = NewtonTaskEnv(env_id, num_envs=2, device=device, seed=29, auto_reset=False)
    try:
        obs_c, _ = cpu_env.reset(seed=29)
        obs_n, _ = n_env.reset_torch(seed=29)
        np.testing.assert_allclose(
            obs_n[0].detach().cpu().numpy(), obs_c, atol=1e-4, rtol=1e-4
        )
        # Re-seeding with the same seed reproduces the exact initial state and target.
        obs_n_rep, _ = n_env.reset_torch(seed=29)
        np.testing.assert_allclose(
            obs_n_rep.detach().cpu().numpy(), obs_n.detach().cpu().numpy(), atol=1e-6
        )

        rng = np.random.default_rng(41)
        for _ in range(6):
            act = rng.uniform(-1.0, 1.0, size=(n_env.num_actions,)).astype(np.float32)
            obs_c, r_c, term_c, trunc_c, info_c = cpu_env.step(act)
            act_batch = np.stack([act, 0.25 * act], axis=0)
            obs_n, r_n, term_n, trunc_n, info_n = n_env.step_torch(
                torch.as_tensor(
                    act_batch, dtype=torch.float32, device=n_env.torch_device
                )
            )
            np.testing.assert_allclose(
                obs_n[0].detach().cpu().numpy(), obs_c, atol=2e-2, rtol=2e-2
            )
            np.testing.assert_allclose(
                float(r_n[0].detach().cpu().item()), float(r_c), atol=0.15, rtol=2e-2
            )
            assert bool(term_n[0].detach().cpu().item()) == bool(term_c)
            assert bool(trunc_n[0].detach().cpu().item()) == bool(trunc_c)
            assert bool(info_n["solved"][0].detach().cpu().item()) == bool(
                info_c.get("solved", False)
            )
    finally:
        cpu_env.close()


@pytest.mark.parametrize(
    "env_id",
    [
        "myoLegRoughTerrainWalk-v0",
        "myoLegHillyTerrainWalk-v0",
        "myoLegStairTerrainWalk-v0",
    ],
)
def test_leg_terrain_walk_matches_cpu_and_knee_termination(env_id: str) -> None:
    torch, device = _require_newton_and_torch()
    from myosuite.integrations.newton import NewtonTaskEnv

    cpu_env = make_env(env_id)
    n_env = NewtonTaskEnv(env_id, num_envs=1, device=device, seed=7, auto_reset=False)
    try:
        obs_c, _ = cpu_env.reset(seed=7)
        obs_n, _ = n_env.reset_torch(seed=7)
        np.testing.assert_allclose(
            n_env.sim.solver.mjw_model.hfield_data.numpy(),
            cpu_env.unwrapped.model.hfield_data,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            obs_n[0].detach().cpu().numpy(), obs_c, atol=1e-3, rtol=1e-3
        )

        rng = np.random.default_rng(19)
        for _ in range(5):
            act = rng.uniform(0.0, 1.0, size=(n_env.num_actions,)).astype(np.float32)
            obs_c, r_c, term_c, trunc_c, _ = cpu_env.step(act)
            obs_n, r_n, term_n, trunc_n, _ = n_env.step_torch(
                torch.as_tensor(
                    act[None, :], dtype=torch.float32, device=n_env.torch_device
                )
            )
            obs_n0 = obs_n[0].detach().cpu().numpy()
            # MuJoCo-Warp reduces heightfield contacts per pair (4-8 vs full prism fan on CPU),
            # keeping proximal pose within 2e-2 and distal ankle/toe heightfield contact within 0.12.
            np.testing.assert_allclose(obs_n0[:20], obs_c[:20], atol=2e-2, rtol=2e-2)
            np.testing.assert_allclose(obs_n0[:163], obs_c[:163], atol=0.12, rtol=0.12)
            np.testing.assert_allclose(obs_n0[323:], obs_c[323:], atol=1e-2, rtol=1e-2)
            np.testing.assert_allclose(
                float(r_n[0].detach().cpu().item()), float(r_c), atol=0.25, rtol=5e-2
            )
            assert bool(term_n[0].detach().cpu().item()) == bool(term_c)
            assert bool(trunc_n[0].detach().cpu().item()) == bool(trunc_c)

        # Verify the 0.61 m knee-margin termination triggers even when COM height >= min_height (0.8 m).
        if env_id == "myoLegStairTerrainWalk-v0":
            # Elevate the entire heightfield and pose the humanoid squatting with COM > 0.8 m
            # while feet are elevated close to COM (com_z - mean(feet_z) < 0.61 m).
            qp_crouch = cpu_env.unwrapped.data.qpos.copy()
            qp_crouch[2] = 1.15
            for jname in ("hip_flexion_l", "hip_flexion_r"):
                qp_crouch[cpu_env.unwrapped.model.joint(jname).qposadr[0]] = 1.6
            for jname in ("knee_angle_l", "knee_angle_r"):
                qp_crouch[cpu_env.unwrapped.model.joint(jname).qposadr[0]] = 2.0
            n_env.sim.reset(qpos=qp_crouch[None, :])
            com, _, _ = n_env._walk_kinematics_torch()
            feet_z = n_env.sim.torch_accessor.body_xpos()[:, n_env._feet_bids, 2]
            assert float(com[0, 2].item()) > n_env.min_height
            assert float((com[0, 2] - feet_z[0].mean()).item()) < 0.61
            rwd_d = n_env._compute_reward_dict_torch()
            assert bool((rwd_d["done"][0] > 0.5).item())
    finally:
        cpu_env.close()
