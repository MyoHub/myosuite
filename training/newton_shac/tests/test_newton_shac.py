# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for differentiable musculoskeletal PoseEnv, SHAC, and matched PPO."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

pytest.importorskip("mujoco_warp")
pytest.importorskip("warp")

from newton_shac import (  # noqa: E402
    DifferentiablePoseEnv,
    PpoConfig,
    ShacConfig,
    train_ppo,
    train_shac,
)


def test_differentiable_muscle_step_matches_finite_differences() -> None:
    env = DifferentiablePoseEnv("myoElbowPose1D6MRandom-v0", num_envs=2, seed=3)
    env.reset(seed=3)
    q0 = torch.tensor(
        [[0.75], [1.15]], dtype=torch.float32, device=env.device, requires_grad=True
    )
    v0 = torch.tensor(
        [[0.10], [-0.15]], dtype=torch.float32, device=env.device, requires_grad=True
    )
    a0 = torch.full(
        (2, env.na), 0.30, dtype=torch.float32, device=env.device, requires_grad=True
    )
    raw_act = (
        torch.linspace(-0.6, 0.8, 2 * env.nu, dtype=torch.float32, device=env.device)
        .reshape(2, env.nu)
        .requires_grad_(True)
    )

    env.qpos = q0
    env.qvel = v0
    env.act = a0
    env.target_angles = torch.tensor(
        [[1.35], [0.45]], dtype=torch.float32, device=env.device
    )

    _, rwd_smooth, _, _ = env.step_diff(raw_act)
    loss = -rwd_smooth.sum()
    loss.backward()
    grad_analytic = raw_act.grad.detach().cpu().numpy()

    eps = 2e-4
    grad_fd = np.zeros_like(grad_analytic)
    base_act = raw_act.detach().clone()
    for w in range(2):
        for j in range(env.nu):
            vals = []
            for sgn in (1.0, -1.0):
                pert = base_act.clone()
                pert[w, j] += sgn * eps
                with torch.no_grad():
                    env.qpos = q0.detach().clone()
                    env.qvel = v0.detach().clone()
                    env.act = a0.detach().clone()
                    env.step_count.zero_()
                    _, r_p, _, _ = env.step_diff(pert)
                    vals.append(float(-r_p.sum().cpu().item()))
            grad_fd[w, j] = (vals[0] - vals[1]) / (2.0 * eps)

    np.testing.assert_allclose(grad_analytic, grad_fd, atol=2e-3, rtol=2e-2)

    # Verify forward trajectory matches CPU MuJoCo over frame_skip substeps.
    d_cpu = mujoco.MjData(env.mj_model)
    d_cpu.qpos[:] = q0[0].detach().cpu().numpy()
    d_cpu.qvel[:] = v0[0].detach().cpu().numpy()
    d_cpu.act[:] = a0[0].detach().cpu().numpy()
    ctrl0 = env.actions_to_ctrl(base_act)[0].detach().cpu().numpy()
    for _ in range(env.frame_skip):
        d_cpu.ctrl[:] = ctrl0
        mujoco.mj_step(env.mj_model, d_cpu)
    with torch.no_grad():
        q_w, _, _ = env._forward_substeps(
            q0.detach(), v0.detach(), a0.detach(), env.actions_to_ctrl(base_act)
        )
    np.testing.assert_allclose(q_w[0].cpu().numpy(), d_cpu.qpos, atol=1e-4)


def test_shac_and_ppo_smoke_and_cpu_playback() -> None:
    shac_cfg = ShacConfig(
        num_envs=16,
        horizon=4,
        total_env_steps=128,
        eval_interval_steps=64,
        eval_episodes=8,
    )
    shac_model, shac_metrics = train_shac("myoElbowPose1D6MRandom-v0", shac_cfg, seed=0)
    assert len(shac_metrics.eval_return_mean) == 3
    assert np.isfinite(shac_metrics.eval_return_mean).all()
    assert np.isfinite(shac_metrics.cpu_eval_return)
    for p in shac_model.parameters():
        assert torch.isfinite(p).all()

    ppo_cfg = PpoConfig(
        num_envs=16,
        horizon=4,
        total_env_steps=128,
        eval_interval_steps=64,
        eval_episodes=8,
    )
    ppo_model, ppo_metrics = train_ppo("myoElbowPose1D6MRandom-v0", ppo_cfg, seed=0)
    assert len(ppo_metrics.eval_return_mean) == 3
    assert np.isfinite(ppo_metrics.eval_return_mean).all()
    assert np.isfinite(ppo_metrics.cpu_eval_return)
    for p in ppo_model.parameters():
        assert torch.isfinite(p).all()
