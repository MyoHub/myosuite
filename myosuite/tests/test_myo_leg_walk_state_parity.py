# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Diagnostics for myoLegWalk-v0 CPU vs mjlab parity.

This module contains tests that:

- Drive CPU and mjlab with the same actions and compare observations
  (documenting the current physics mismatch).
- Compare initial states after reset.
- Compare the effective control signals written into MuJoCo ctrl slots.

The goal is to collect quantitative evidence about where CPU and mjlab diverge:
initialisation, action→ctrl mapping, or physics/integration.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


pytestmark = pytest.mark.tier2

_GPU_REQUESTED = os.environ.get("MYOSUITE_GPU_TESTS", "").strip().lower() in (
    "1",
    "true",
    "yes",
)


try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _TORCH_AVAILABLE = False

try:
    import mjlab  # noqa: F401

    _MJLAB_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _MJLAB_AVAILABLE = False


@pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _MJLAB_AVAILABLE and _GPU_REQUESTED),
    reason=(
        "mjlab walk state tests require mjlab+torch and a GPU run "
        "(set MYOSUITE_GPU_TESTS=1 on a CUDA-enabled machine)."
    ),
)
def test_myo_leg_walk_state_parity_cpu_vs_mjlab() -> None:
    """Drive CPU and mjlab myoLegWalk-v0 with same actions and compare obs."""
    import gymnasium as gym

    import myosuite

    myosuite.register_all_envs()
    from myosuite.core.registry import make_env

    seed = 0
    num_steps = 10
    rng = np.random.default_rng(seed)

    # ----- CPU env (legacy WalkEnvV0) -----
    cpu_env = gym.make("myoLegWalk-v0", reset_type="init")
    obs_cpu, _ = cpu_env.reset(seed=seed)
    obs_cpu = np.asarray(obs_cpu, dtype=np.float32)

    # ----- mjlab env -----
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mj_env = make_env("myoLegWalk-v0", backend="mjlab", device=device)
    obs_mj, _ = mj_env.reset(seed=seed)
    # mjlab observations are dict[group_name] → array; use "policy" group.
    if isinstance(obs_mj, dict):
        obs_mj_arr = obs_mj["policy"]
        if isinstance(obs_mj_arr, torch.Tensor):
            obs_mj_arr = obs_mj_arr.detach().cpu().numpy()
        obs_mj = obs_mj_arr[0] if obs_mj_arr.ndim > 1 else obs_mj_arr
    elif isinstance(obs_mj, torch.Tensor):
        obs_mj = obs_mj.detach().cpu().numpy()
    obs_mj = np.asarray(obs_mj, dtype=np.float32)

    act_dim = cpu_env.action_space.shape[0]

    max_diffs: list[float] = []

    for _ in range(num_steps):
        # Same normalised action for both backends, in [-1, 1].
        action = rng.uniform(low=-1.0, high=1.0, size=(act_dim,)).astype(np.float32)

        # CPU step.
        obs_cpu, _, term_cpu, trunc_cpu, _ = cpu_env.step(action)
        obs_cpu = np.asarray(obs_cpu, dtype=np.float32)

        # mjlab step.
        action_t = torch.as_tensor(
            action, dtype=torch.float32, device=device
        ).unsqueeze(0)
        obs_mj, _, term_mj, trunc_mj, _ = mj_env.step(action_t)
        if isinstance(obs_mj, dict):
            obs_mj_arr = obs_mj["policy"]
            if isinstance(obs_mj_arr, torch.Tensor):
                obs_mj_arr = obs_mj_arr.detach().cpu().numpy()
            obs_mj = obs_mj_arr[0] if obs_mj_arr.ndim > 1 else obs_mj_arr
        elif isinstance(obs_mj, torch.Tensor):
            obs_mj = obs_mj.detach().cpu().numpy()
        obs_mj = np.asarray(obs_mj, dtype=np.float32)

        # Compare observation vectors up to the mjlab length (CPU obs may be longer).
        length = min(obs_cpu.size, obs_mj.size)
        max_diff = float(np.max(np.abs(obs_cpu[:length] - obs_mj[:length])))
        max_diffs.append(max_diff)

        if term_cpu or trunc_cpu:
            obs_cpu, _ = cpu_env.reset()
            obs_cpu = np.asarray(obs_cpu, dtype=np.float32)
        if bool(term_mj.any()) or bool(trunc_mj.any()):
            obs_mj, _ = mj_env.reset()
            if isinstance(obs_mj, dict):
                obs_mj_arr = obs_mj["policy"]
                if isinstance(obs_mj_arr, torch.Tensor):
                    obs_mj_arr = obs_mj_arr.detach().cpu().numpy()
                obs_mj = obs_mj_arr[0] if obs_mj_arr.ndim > 1 else obs_mj_arr
            elif isinstance(obs_mj, torch.Tensor):
                obs_mj = obs_mj.detach().cpu().numpy()
            obs_mj = np.asarray(obs_mj, dtype=np.float32)

    # Bounded-parity guardrail: trajectories need not be identical yet, but
    # divergence must remain finite and below a broad ceiling.
    assert np.all(np.isfinite(max_diffs)), f"Non-finite diffs: {max_diffs}"
    assert max(max_diffs) < 50.0, f"Excessive CPU↔mjlab obs divergence: {max_diffs}"

    cpu_env.close()
    if hasattr(mj_env, "close"):
        mj_env.close()


@pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _MJLAB_AVAILABLE and _GPU_REQUESTED),
    reason=(
        "mjlab walk state tests require mjlab+torch and a GPU run "
        "(set MYOSUITE_GPU_TESTS=1 on a CUDA-enabled machine)."
    ),
)
def test_myo_leg_walk_initial_state_difference_cpu_vs_mjlab() -> None:
    """Compare initial qpos/qvel at reset for CPU vs mjlab."""
    import gymnasium as gym

    import myosuite

    myosuite.register_all_envs()
    from myosuite.core.registry import make_env

    seed = 0

    cpu_env = gym.make("myoLegWalk-v0", reset_type="init")
    obs_cpu, _ = cpu_env.reset(seed=seed)
    del obs_cpu
    cpu_impl = cpu_env.unwrapped
    qpos_cpu = cpu_impl.mj_data.qpos.copy()
    qvel_cpu = cpu_impl.mj_data.qvel.copy()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mj_env = make_env("myoLegWalk-v0", backend="mjlab", device=device)
    obs_mj, _ = mj_env.reset(seed=seed)
    del obs_mj

    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        _WALK_ENTITY_NAME,
    )

    entity = mj_env.scene[_WALK_ENTITY_NAME]
    raw_data = entity.data.data  # torch-backed mjlab data
    qpos_mj = raw_data.qpos[0].detach().cpu().numpy()
    qvel_mj = raw_data.qvel[0].detach().cpu().numpy()

    # Compare shapes and norms; print diagnostics for inspection.
    assert qpos_cpu.shape == qpos_mj.shape
    assert qvel_cpu.shape == qvel_mj.shape

    qpos_max_diff = float(np.max(np.abs(qpos_cpu - qpos_mj)))
    qvel_max_diff = float(np.max(np.abs(qvel_cpu - qvel_mj)))
    print(f"initial qpos max|cpu-mjlab|: {qpos_max_diff:.3e}")
    print(f"initial qvel max|cpu-mjlab|: {qvel_max_diff:.3e}")

    # No strict assertion on magnitude yet; this test is primarily diagnostic.
    cpu_env.close()
    if hasattr(mj_env, "close"):
        mj_env.close()


@pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _MJLAB_AVAILABLE and _GPU_REQUESTED),
    reason=(
        "mjlab walk state tests require mjlab+torch and a GPU run "
        "(set MYOSUITE_GPU_TESTS=1 on a CUDA-enabled machine)."
    ),
)
def test_myo_leg_walk_ctrl_mapping_cpu_vs_mjlab() -> None:
    """Compare the effective ctrl signals applied for the same actions."""
    import gymnasium as gym

    import myosuite

    myosuite.register_all_envs()
    from myosuite.core.registry import make_env
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        _WALK_ENTITY_NAME,
    )

    seed = 0
    rng = np.random.default_rng(seed)

    cpu_env = gym.make("myoLegWalk-v0", reset_type="init")
    obs_cpu, _ = cpu_env.reset(seed=seed)
    del obs_cpu
    cpu_impl = cpu_env.unwrapped

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mj_env = make_env("myoLegWalk-v0", backend="mjlab", device=device)
    obs_mj, _ = mj_env.reset(seed=seed)
    del obs_mj

    entity = mj_env.scene[_WALK_ENTITY_NAME]

    act_dim = cpu_env.action_space.shape[0]

    for _ in range(3):
        action = rng.uniform(low=-1.0, high=1.0, size=(act_dim,)).astype(np.float32)

        # CPU step and read resulting ctrl.
        _, _, term_cpu, trunc_cpu, _ = cpu_env.step(action)
        ctrl_cpu = cpu_impl.mj_data.ctrl.copy()

        # mjlab step and read resulting ctrl.
        action_t = torch.as_tensor(
            action, dtype=torch.float32, device=device
        ).unsqueeze(0)
        _, _, term_mj, trunc_mj, _ = mj_env.step(action_t)
        ctrl_mj = entity.data.data.ctrl[0].detach().cpu().numpy()

        assert ctrl_cpu.shape == ctrl_mj.shape
        max_diff = float(np.max(np.abs(ctrl_cpu - ctrl_mj)))
        print(f"ctrl max|cpu-mjlab|: {max_diff:.3e}")

        if term_cpu or trunc_cpu:
            _, _ = cpu_env.reset()
        if bool(term_mj.any()) or bool(trunc_mj.any()):
            _, _ = mj_env.reset()

    cpu_env.close()
    if hasattr(mj_env, "close"):
        mj_env.close()


@pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _MJLAB_AVAILABLE and _GPU_REQUESTED),
    reason=(
        "mjlab walk state tests require mjlab+torch and a GPU run "
        "(set MYOSUITE_GPU_TESTS=1 on a CUDA-enabled machine)."
    ),
)
def test_myo_leg_walk_forced_initial_parity_one_step_stats() -> None:
    """Force identical initial (qpos, qvel) and collect one-step divergence stats.

    This test:
    - Resets CPU and mjlab myoLegWalk-v0.
    - Copies CPU qpos/qvel into mjlab's underlying data arrays for env 0.
    - Verifies the initial states match numerically.
    - Applies one shared random action and records max|obs_cpu - obs_mjlab|
      after that step (printed for inspection).

    It validates that we can align initial conditions exactly, and provides
    concrete one-step observation differences starting from that shared state.
    """
    import gymnasium as gym

    import myosuite

    myosuite.register_all_envs()
    from myosuite.core.registry import make_env
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        _WALK_ENTITY_NAME,
    )

    seed = 0
    rng = np.random.default_rng(seed)

    # CPU env and initial state (reset_type="init" → keyframe 2 pose).
    cpu_env = gym.make("myoLegWalk-v0", reset_type="init")
    obs_cpu, _ = cpu_env.reset(seed=seed)
    obs_cpu = np.asarray(obs_cpu, dtype=np.float32)
    cpu_impl = cpu_env.unwrapped
    qpos_cpu = cpu_impl.mj_data.qpos.copy()
    qvel_cpu = cpu_impl.mj_data.qvel.copy()

    # mjlab env and scene.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mj_env = make_env("myoLegWalk-v0", backend="mjlab", device=device)
    obs_mj, _ = mj_env.reset(seed=seed)
    if isinstance(obs_mj, dict):
        obs_mj_arr = obs_mj["policy"]
        if isinstance(obs_mj_arr, torch.Tensor):
            obs_mj_arr = obs_mj_arr.detach().cpu().numpy()
        obs_mj = obs_mj_arr[0] if obs_mj_arr.ndim > 1 else obs_mj_arr
    elif isinstance(obs_mj, torch.Tensor):
        obs_mj = obs_mj.detach().cpu().numpy()
    obs_mj = np.asarray(obs_mj, dtype=np.float32)

    entity = mj_env.scene[_WALK_ENTITY_NAME]
    raw_data = entity.data.data  # torch-backed mjlab data

    # Force mjlab qpos/qvel to match CPU for env 0.
    raw_data.qpos[0, :] = torch.as_tensor(
        qpos_cpu, device=raw_data.qpos.device, dtype=raw_data.qpos.dtype
    )
    raw_data.qvel[0, :] = torch.as_tensor(
        qvel_cpu, device=raw_data.qvel.device, dtype=raw_data.qvel.dtype
    )

    qpos_mj0 = raw_data.qpos[0].detach().cpu().numpy()
    qvel_mj0 = raw_data.qvel[0].detach().cpu().numpy()

    # Initial states must now match numerically.
    assert np.allclose(qpos_mj0, qpos_cpu, atol=1e-7)
    assert np.allclose(qvel_mj0, qvel_cpu, atol=1e-7)

    # One shared action step from this common initial state.
    act_dim = cpu_env.action_space.shape[0]
    action = rng.uniform(low=-1.0, high=1.0, size=(act_dim,)).astype(np.float32)

    obs_cpu_1, _, _, _, _ = cpu_env.step(action)
    obs_cpu_1 = np.asarray(obs_cpu_1, dtype=np.float32)

    action_t = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
    obs_mj_1, _, _, _, _ = mj_env.step(action_t)
    if isinstance(obs_mj_1, dict):
        obs_mj_arr = obs_mj_1["policy"]
        if isinstance(obs_mj_arr, torch.Tensor):
            obs_mj_arr = obs_mj_arr.detach().cpu().numpy()
        obs_mj_1 = obs_mj_arr[0] if obs_mj_arr.ndim > 1 else obs_mj_arr
    elif isinstance(obs_mj_1, torch.Tensor):
        obs_mj_1 = obs_mj_1.detach().cpu().numpy()
    obs_mj_1 = np.asarray(obs_mj_1, dtype=np.float32)

    length = min(obs_cpu_1.size, obs_mj_1.size)
    max_diff = float(np.max(np.abs(obs_cpu_1[:length] - obs_mj_1[:length])))
    print(f"one-step max|obs_cpu-obs_mjlab| from shared init: {max_diff:.3e}")

    cpu_env.close()
    if hasattr(mj_env, "close"):
        mj_env.close()
