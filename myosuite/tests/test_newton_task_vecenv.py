# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for experimental NewtonTaskEnv, closed-loop ONNX playback, and rsl_rl VecEnv."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

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


@pytest.mark.parametrize(
    "env_id",
    [
        "myoElbowPose1D6MRandom-v0",
        "myoHandPoseRandom-v0",
        "myoArmReachRandom-v0",
        "myoFatiElbowPose1D6MRandom-v0",
    ],
)
def test_newton_task_env_matches_cpu_step_by_step(env_id: str) -> None:
    torch, device = _require_newton_and_torch()
    from myosuite.integrations.newton import NewtonTaskEnv

    cpu_env = make_env(env_id)
    n_env = NewtonTaskEnv(env_id, num_envs=2, device=device, seed=17, auto_reset=False)
    try:
        obs_cpu, _ = cpu_env.reset(seed=17)
        obs_n, _ = n_env.reset_torch(seed=17)
        np.testing.assert_allclose(
            obs_n[0].detach().cpu().numpy(), obs_cpu, atol=1e-4, rtol=1e-4
        )

        rng = np.random.default_rng(29)
        for _ in range(12):
            act = rng.uniform(-1.0, 1.0, size=(n_env.num_actions,)).astype(np.float32)
            obs_cpu, r_cpu, term_cpu, trunc_cpu, _ = cpu_env.step(act)
            act_batch = np.stack([act, -0.5 * act], axis=0)
            act_t = torch.as_tensor(
                act_batch, dtype=torch.float32, device=n_env.torch_device
            )
            obs_n, r_n, term_n, trunc_n, _ = n_env.step_torch(act_t)

            np.testing.assert_allclose(
                obs_n[0].detach().cpu().numpy(), obs_cpu, atol=5e-4, rtol=5e-4
            )
            np.testing.assert_allclose(
                float(r_n[0].detach().cpu().item()), float(r_cpu), atol=1e-3, rtol=1e-3
            )
            assert bool(term_n[0].detach().cpu().item()) == bool(term_cpu)
            assert bool(trunc_n[0].detach().cpu().item()) == bool(trunc_cpu)
    finally:
        cpu_env.close()


def test_closed_loop_onnx_and_torch_policy_parity(tmp_path: Path) -> None:
    torch, device = _require_newton_and_torch()
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from myosuite.integrations.newton import rollout_closed_loop_parity

    torch.manual_seed(7)
    elbow_policy = torch.nn.Sequential(
        torch.nn.Linear(9, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 6),
        torch.nn.Tanh(),
    )
    onnx_file = tmp_path / "elbow_policy.onnx"
    dummy_in = torch.zeros((1, 9), dtype=torch.float32)
    torch.onnx.export(
        elbow_policy,
        dummy_in,
        str(onnx_file),
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        dynamo=False,
    )

    onnx_report = rollout_closed_loop_parity(
        "myoElbowPose1D6MRandom-v0", onnx_file, steps=25, seed=23, device=device
    )
    assert onnx_report.steps == 25
    assert onnx_report.termination_matches
    assert onnx_report.max_abs_obs_diff < 1e-3
    assert onnx_report.max_abs_action_diff < 1e-3
    assert onnx_report.max_abs_reward_diff < 1e-3

    torch.manual_seed(13)
    hand_policy = torch.nn.Sequential(
        torch.nn.Linear(108, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 39),
        torch.nn.Tanh(),
    )
    hand_report = rollout_closed_loop_parity(
        "myoHandPoseRandom-v0", hand_policy, steps=15, seed=11, device=device
    )
    assert hand_report.steps == 15
    assert hand_report.termination_matches
    assert hand_report.max_abs_obs_diff < 1e-3
    assert hand_report.max_abs_reward_diff < 1e-3


def test_newton_rslrl_vecenv_ppo_smoke() -> None:
    torch, device = _require_newton_and_torch()
    rsl_env_mod = pytest.importorskip("rsl_rl.env")
    rsl_runners = pytest.importorskip("rsl_rl.runners")
    from myosuite.integrations.newton import NewtonRslRlVecEnv

    vec_env = NewtonRslRlVecEnv(
        "myoElbowPose1D6MRandom-v0", num_envs=8, device=device, seed=5
    )
    assert isinstance(vec_env, rsl_env_mod.VecEnv)

    train_cfg = {
        "seed": 5,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": 16,
        "save_interval": 100,
        "empirical_normalization": False,
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
            "obs_normalization": True,
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.001,
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "learning_rate": 3e-4,
            "schedule": "adaptive",
            "gamma": 0.97,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }
    runner = rsl_runners.OnPolicyRunner(
        vec_env, train_cfg, log_dir=None, device=str(vec_env.device)
    )
    runner.learn(num_learning_iterations=2)
    obs_td = vec_env.get_observations()
    with torch.no_grad():
        acts = runner.alg.actor(obs_td)
    assert acts.shape == (8, vec_env.num_actions)
    assert torch.isfinite(acts).all()
