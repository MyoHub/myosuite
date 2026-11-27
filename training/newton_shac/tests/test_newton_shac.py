# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for differentiable musculoskeletal PoseEnv, CUDA-graph capture, SHAC, and ONNX export."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from myosuite import make_env


def _require_newton_shac():
    pytest.importorskip("mujoco_warp")
    pytest.importorskip("warp")
    return pytest.importorskip("newton_shac")


def test_differentiable_muscle_step_matches_finite_differences() -> None:
    ns = _require_newton_shac()
    env = ns.DifferentiablePoseEnv("myoElbowPose1D6MRandom-v0", num_envs=2, seed=3)
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


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for graph capture"
)
def test_cuda_graph_matches_eager_forward_and_backward() -> None:
    ns = _require_newton_shac()
    env_graph = ns.DifferentiablePoseEnv(
        "myoElbowPose1D6MRandom-v0", num_envs=8, seed=7, use_cuda_graph=True
    )
    env_eager = ns.DifferentiablePoseEnv(
        "myoElbowPose1D6MRandom-v0", num_envs=8, seed=7, use_cuda_graph=False
    )
    assert env_graph.uses_cuda_graph
    assert not env_eager.uses_cuda_graph

    obs_g = env_graph.reset(seed=7)
    obs_e = env_eager.reset(seed=7)
    np.testing.assert_allclose(obs_g.cpu().numpy(), obs_e.cpu().numpy(), atol=1e-6)

    act_g = (
        torch.linspace(
            -0.5, 0.7, 8 * env_graph.nu, dtype=torch.float32, device=env_graph.device
        )
        .reshape(8, env_graph.nu)
        .requires_grad_(True)
    )
    act_e = act_g.detach().clone().requires_grad_(True)

    next_g, rwd_g, _, _ = env_graph.step_diff(act_g)
    next_e, rwd_e, _, _ = env_eager.step_diff(act_e)
    np.testing.assert_allclose(
        next_g.detach().cpu().numpy(), next_e.detach().cpu().numpy(), atol=1e-5
    )
    np.testing.assert_allclose(
        rwd_g.detach().cpu().numpy(), rwd_e.detach().cpu().numpy(), atol=1e-5
    )

    (-rwd_g.sum()).backward()
    (-rwd_e.sum()).backward()
    np.testing.assert_allclose(
        act_g.grad.detach().cpu().numpy(), act_e.grad.detach().cpu().numpy(), atol=1e-5
    )


def test_shac_smoke_cpu_playback_and_onnx_export(tmp_path: Path) -> None:
    ns = _require_newton_shac()
    shac_cfg = ns.ShacConfig(
        num_envs=16,
        horizon=4,
        total_env_steps=128,
        eval_interval_steps=64,
        eval_episodes=8,
    )
    model, metrics = ns.train_shac("myoElbowPose1D6MRandom-v0", shac_cfg, seed=0)
    assert len(metrics.eval_return_mean) == 3
    assert np.isfinite(metrics.eval_return_mean).all()
    assert np.isfinite(metrics.cpu_eval_return)
    for p in model.parameters():
        assert torch.isfinite(p).all()

    ort = pytest.importorskip("onnxruntime")
    onnx_path = ns.export_shac_onnx(model, tmp_path / "shac_policy.onnx")
    assert onnx_path.is_file()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    cpu_env = make_env("myoElbowPose1D6MRandom-v0")
    try:
        obs_np, _ = cpu_env.reset(seed=42)
        obs_batch = obs_np.astype(np.float32)[None, :]
        onnx_act = sess.run(["action"], {"obs": obs_batch})[0]
        with torch.no_grad():
            dev = next(model.parameters()).device
            ref_act = (
                torch.clamp(
                    model.act_deterministic(torch.as_tensor(obs_batch, device=dev)),
                    -1.0,
                    1.0,
                )
                .cpu()
                .numpy()
            )
        np.testing.assert_allclose(onnx_act, ref_act, atol=1e-6)
        next_obs, rwd, _, _, _ = cpu_env.step(onnx_act[0])
        assert np.isfinite(next_obs).all()
        assert np.isfinite(rwd)
    finally:
        cpu_env.close()

    from myosuite.utils.export_onnx import verify_onnx_on_cpu

    ver = verify_onnx_on_cpu(onnx_path, n_steps=5, env_id="myoElbowPose1D6MRandom-v0")
    assert ver["n_steps"] == 5
    assert np.isfinite(ver["total_reward"])
