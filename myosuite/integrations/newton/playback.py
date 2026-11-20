# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Experimental closed-loop ONNX / PyTorch policy playback parity on Newton.

.. warning::
   **Experimental validation probe only.** CPU ``MyoGymnasiumEnv`` and GPU
   ``mjlab`` are the only two supported MyoSuite backends. Like the experimental
   MJX route, this module exists solely for solver compatibility diagnostics and
   should not be used as a foundation for new MyoSuite tasks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from myosuite import make_env
from myosuite.integrations.newton.task_env import NewtonTaskEnv


@dataclass(frozen=True)
class ClosedLoopParityReport:
    """Summary of an independent closed-loop policy rollout across CPU and Newton."""

    env_id: str
    steps: int
    seed: int
    max_abs_obs_diff: float
    max_abs_action_diff: float
    max_abs_reward_diff: float
    termination_matches: bool
    cpu_return: float
    newton_return: float


def _build_policy_runners(
    policy: str | Path | torch.nn.Module | Callable[[np.ndarray], np.ndarray],
    torch_device: torch.device,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[torch.Tensor], torch.Tensor]]:
    if isinstance(policy, str | Path):
        import onnxruntime as ort  # noqa: PLC0415

        sess = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name

        def _run_np(obs_np: np.ndarray) -> np.ndarray:
            inp = np.asarray(obs_np, dtype=np.float32).reshape(1, -1)
            out = sess.run(None, {in_name: inp})[0]
            return np.asarray(out, dtype=np.float32).reshape(-1)

        def _run_torch(obs_t: torch.Tensor) -> torch.Tensor:
            inp = obs_t.detach().cpu().numpy().astype(np.float32)
            out = sess.run(None, {in_name: inp})[0]
            return torch.as_tensor(out, dtype=torch.float32, device=torch_device)

        return _run_np, _run_torch

    if isinstance(policy, torch.nn.Module):
        policy_eval = policy.to(device=torch_device).eval()

        def _mod_np(obs_np: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                inp = torch.as_tensor(
                    obs_np, dtype=torch.float32, device=torch_device
                ).unsqueeze(0)
                return policy_eval(inp).squeeze(0).detach().cpu().numpy()

        def _mod_torch(obs_t: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return policy_eval(obs_t.to(device=torch_device, dtype=torch.float32))

        return _mod_np, _mod_torch

    def _fn_np(obs_np: np.ndarray) -> np.ndarray:
        return np.asarray(policy(obs_np), dtype=np.float32)

    def _fn_torch(obs_t: torch.Tensor) -> torch.Tensor:
        arr = np.stack([policy(row) for row in obs_t.detach().cpu().numpy()], axis=0)
        return torch.as_tensor(arr, dtype=torch.float32, device=torch_device)

    return _fn_np, _fn_torch


def rollout_closed_loop_parity(
    env_id: str,
    policy: str | Path | torch.nn.Module | Callable[[np.ndarray], np.ndarray],
    *,
    steps: int = 25,
    seed: int = 0,
    device: str = "cuda:0",
) -> ClosedLoopParityReport:
    """Run independent closed-loop trajectories on CPU and Newton and measure drift."""
    cpu_env = make_env(env_id)
    newton_env = NewtonTaskEnv(
        env_id, num_envs=1, device=device, seed=seed, auto_reset=False
    )
    run_np, run_torch = _build_policy_runners(policy, newton_env.torch_device)

    try:
        obs_cpu, _ = cpu_env.reset(seed=seed)
        obs_n, _ = newton_env.reset_torch(seed=seed)

        max_obs_diff = float(np.max(np.abs(obs_cpu - obs_n[0].detach().cpu().numpy())))
        max_act_diff = 0.0
        max_rwd_diff = 0.0
        term_match = True
        cpu_ret = 0.0
        newton_ret = 0.0
        executed = 0

        for _ in range(steps):
            act_cpu = run_np(obs_cpu)
            act_n = run_torch(obs_n)
            act_diff = float(np.max(np.abs(act_cpu - act_n[0].detach().cpu().numpy())))
            max_act_diff = max(max_act_diff, act_diff)

            obs_cpu, r_cpu, term_cpu, trunc_cpu, _ = cpu_env.step(act_cpu)
            obs_n, r_n, term_n, trunc_n, _ = newton_env.step_torch(act_n)
            executed += 1

            obs_n_np = obs_n[0].detach().cpu().numpy()
            r_n_val = float(r_n[0].detach().cpu().item())
            done_cpu = bool(term_cpu or trunc_cpu)
            done_n = bool((term_n[0] | trunc_n[0]).detach().cpu().item())

            max_obs_diff = max(max_obs_diff, float(np.max(np.abs(obs_cpu - obs_n_np))))
            max_rwd_diff = max(max_rwd_diff, abs(float(r_cpu) - r_n_val))
            cpu_ret += float(r_cpu)
            newton_ret += r_n_val
            if done_cpu != done_n:
                term_match = False
                break
            if done_cpu:
                break

        return ClosedLoopParityReport(
            env_id=env_id,
            steps=executed,
            seed=seed,
            max_abs_obs_diff=max_obs_diff,
            max_abs_action_diff=max_act_diff,
            max_abs_reward_diff=max_rwd_diff,
            termination_matches=term_match,
            cpu_return=cpu_ret,
            newton_return=newton_ret,
        )
    finally:
        cpu_env.close()
