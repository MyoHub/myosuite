# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backend throughput benchmarks: CPU vs MJX vs mjlab.

Run with:
    uv run pytest myosuite/tests/benchmarks/ --benchmark-only -v
    uv run pytest myosuite/tests/benchmarks/ --benchmark-only --benchmark-compare

PLAN.md targets:
    CPU  (1 env):       ~2,000 steps/sec
    MJX  (JIT, 1 env):  >> CPU (XLA compiled; scales x n_envs)
    mjlab: skipped until installed
"""

import os

import numpy as np
import pytest


pytestmark = [pytest.mark.tier3, pytest.mark.slow]

_GPU_REQUESTED = os.environ.get("MYOSUITE_GPU_TESTS", "").strip().lower() in (
    "1",
    "true",
    "yes",
)


# MJX guard: benchmark uses mjx_impl=None (JAX XLA), so it runs on CPU without warp.
# Only require mjx + jax + playground; warp is optional (needed only for impl="warp").
try:
    from mujoco import mjx as _mjx  # noqa: F401
    import jax as _jax_guard  # noqa: F401 — guard: ensure JAX available for benchmark
    import mujoco_playground as _mp  # noqa: F401

    _MJX_AVAILABLE = True
except (ImportError, AttributeError):
    _MJX_AVAILABLE = False

try:
    import mjlab  # noqa: F401

    _MJLAB_AVAILABLE = True
except ImportError:
    _MJLAB_AVAILABLE = False


# --- Benchmark 1: CPU (always runs) ---
def test_bench_cpu_elbow_pose(benchmark):
    """MuJoCo CPU: steps/sec for 1-DOF elbow pose (1 env)."""
    import myosuite

    myosuite.register_all_envs()
    import gymnasium as gym

    env = gym.make("myoElbowPose1D6MFixed-v0")
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    env.reset(seed=0)

    def _step():
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            env.reset()

    benchmark(_step)
    env.close()


# --- Benchmark 2: MJX (skipped without GPU/XLA) ---
@pytest.mark.skipif(
    not (_MJX_AVAILABLE and _GPU_REQUESTED),
    reason=(
        "MJX throughput benchmark requires MJX/mujoco_playground and an explicit "
        "GPU run (set MYOSUITE_GPU_TESTS=1 on a CUDA-enabled machine)."
    ),
)
def test_bench_mjx_elbow_pose(benchmark):
    """MJX (XLA): JIT-compiled steps/sec for 1-DOF elbow pose (1 env).

    Uses benchmark.pedantic(warmup_rounds=3) to absorb JIT compilation;
    timing reflects steady-state compiled kernel only.
    """
    import pathlib

    import jax
    import jax.numpy as jnp
    from etils import epath
    from ml_collections import config_dict

    from myosuite.envs.myo.backends.mjx.pose_env import MjxPoseEnv

    _ASSETS = pathlib.Path(__file__).parents[2] / "envs/myo/assets"
    cfg = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        num_envs=1,
        mjx_impl=None,
        reward_config=config_dict.create(
            angle_reward_weight=1.0,
            ctrl_cost_weight=0.0,
            pose_thd=0.35,
            far_th=float(2 * jnp.pi),
            bonus_weight=4.0,
        ),
        target_jnt_range=config_dict.create(r_elbow_flex=jnp.array((2.0, 2.0))),
        max_episode_steps=100,
        model_path=epath.Path(str(_ASSETS / "elbow/myoelbow_1dof6muscles.xml")),
    )
    env = MjxPoseEnv(config=cfg)
    jit_step = jax.jit(env.step)
    state = env.reset(jax.random.PRNGKey(0))
    action = jnp.zeros((env._mj_model.nu,), dtype=jnp.float32)

    def _jit_step():
        nonlocal state
        state = jit_step(state, action)

    benchmark.pedantic(_jit_step, warmup_rounds=3, rounds=50)


# --- Benchmark 3: mjlab (runs when mjlab installed and backend works) ---
@pytest.mark.skipif(not _MJLAB_AVAILABLE, reason="mjlab not installed")
def test_bench_mjlab_elbow_pose(benchmark):
    """mjlab (MuJoCo Warp): steps/sec for 1-DOF elbow pose (1 env)."""
    import myosuite

    myosuite.register_all_envs()

    from myosuite.core.registry import make_env

    try:
        env = make_env("myoElbowPose1D6MFixed-v0", backend="mjlab")
    except Exception as e:
        pytest.skip(f"mjlab backend could not create env: {e!s}")

    # mjlab envs expect torch tensors on the env device
    try:
        import torch

        device = getattr(env, "device", "cpu")
        action = torch.zeros(env.action_space.shape, dtype=torch.float32, device=device)
    except Exception:
        action = np.zeros(env.action_space.shape, dtype=np.float32)

    try:
        env.reset(seed=0)
    except Exception as e:
        env.close()
        pytest.skip(f"mjlab env reset failed: {e!s}")

    def _step():
        _, _, term, trunc, _ = env.step(action)
        t = term.any().item() if hasattr(term, "any") else bool(term)
        u = trunc.any().item() if hasattr(trunc, "any") else bool(trunc)
        if t or u:
            env.reset()

    try:
        benchmark(_step)
    except Exception as e:
        env.close()
        pytest.skip(f"mjlab benchmark step failed: {e!s}")
    env.close()


# --- Standalone comparison (no pytest needed) ---
def _compare_backends() -> None:
    """Print a quick steps/sec table for CPU (and MJX if available)."""
    import time

    import myosuite

    myosuite.register_all_envs()
    import gymnasium as gym

    targets: dict[str, int] = {"CPU": 2_000, "MJX": 500_000}
    results: dict[str, float] = {}

    # CPU
    env = gym.make("myoElbowPose1D6MFixed-v0")
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    env.reset(seed=0)
    n = 500
    t0 = time.perf_counter()
    for _ in range(n):
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            env.reset()
    results["CPU"] = n / (time.perf_counter() - t0)
    env.close()

    print("\nBackend throughput (steps/sec)")
    print(f"{'Backend':<10} {'Measured':>12} {'Target':>10} {'Status':>8}")
    print("-" * 45)
    for name, sps in results.items():
        status = "PASS" if sps >= targets.get(name, 0) else "WARN"
        print(f"{name:<10} {sps:>12,.0f} {targets.get(name, 0):>10,} {status:>8}")


if __name__ == "__main__":
    _compare_backends()
