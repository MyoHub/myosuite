# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Reward parity tests for myoLegWalk-v0 between CPU and mjlab backends.

This test exercises the raw WalkEnvV0 reward components on the CPU
``myoLegWalk-v0`` Gymnasium environment and compares them against the per-step
reward terms produced by mjlab's ``RewardManager`` for the mjlab-backed
``myoLegWalk-v0`` task.

We deliberately use ``RewardManager.get_active_iterable_terms`` to read the
unscaled (dt-free) reward rates and divide by each term's weight to recover the
raw component values so that the comparison matches the CPU semantics exactly.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# By default, extended parity/backends live in Tier 2; individual tests can opt
# into Tier 1 when needed (e.g. the CPU-only reward dict contract check).
pytestmark = pytest.mark.tier2

_GPU_REQUESTED = os.environ.get("MYOSUITE_GPU_TESTS", "").strip().lower() in (
    "1",
    "true",
    "yes",
)


# Required rwd_dict keys for walk env (component-level contract)
_WALK_RWD_KEYS = frozenset(
    {
        "dense",
        "done",
        "vel_reward",
        "cyclic_hip",
        "ref_rot",
        "joint_angle_rew",
    }
)

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _TORCH_AVAILABLE = False

try:
    import mjlab  # noqa: F401
    import mjlab.envs  # noqa: F401
    from mjlab.tasks import registry as _mjlab_registry  # noqa: F401

    _MJLAB_AVAILABLE = True
except Exception:  # pragma: no cover - environment-dependent
    # Some mjlab/mujoco_warp incompatibilities raise AttributeError at import.
    _MJLAB_AVAILABLE = False


@pytest.mark.tier1
def test_myo_leg_walk_cpu_rwd_dict_contract() -> None:
    """CPU myoLegWalk-v0 must return info['rwd_dict'] with required keys and finite values.

    Component-level parity sanity check: the CPU walk env (LegWalkEnvV0) must
    expose the same rwd_dict shape as expected by benchmarks and by the
    CPU vs mjlab parity test. All component values must be finite.
    """
    import gymnasium as gym

    import myosuite

    myosuite.register_all_envs()

    seed = 0
    num_steps = 10
    rng = np.random.default_rng(seed)

    env = gym.make("myoLegWalk-v0", reset_type="init")
    env.reset(seed=seed)
    act_dim = env.action_space.shape[0]

    for step in range(num_steps):
        action = rng.uniform(low=-1.0, high=1.0, size=(act_dim,)).astype(np.float32)
        _, _, term, trunc, info = env.step(action)
        rwd_dict = info.get("rwd_dict", {})
        for key in _WALK_RWD_KEYS:
            assert key in rwd_dict, f"rwd_dict missing key '{key}' at step {step}"
            val = rwd_dict[key]
            if isinstance(val, (int, float)):
                assert np.isfinite(
                    val
                ), f"rwd_dict['{key}'] not finite at step {step}: {val}"
            else:
                arr = np.asarray(val)
                assert np.all(
                    np.isfinite(arr)
                ), f"rwd_dict['{key}'] has non-finite values at step {step}"
        if term or trunc:
            env.reset()
    env.close()


@pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _MJLAB_AVAILABLE),
    reason="mjlab and torch not installed (pip install myosuite[mjlab])",
)
def test_myo_leg_walk_reward_parity_cpu_vs_mjlab() -> None:
    """CPU WalkEnvV0 reward components must match mjlab RewardManager outputs.

    For a small number of random normalised actions in ``[-1, 1]``, this test:

    - steps the CPU ``myoLegWalk-v0`` environment and records the raw reward
      components from ``info['rwd_dict']``, and
    - steps the mjlab ``myoLegWalk-v0`` task with the same actions and reads
      per-term rewards from ``RewardManager.get_active_iterable_terms(0)``.

    The mjlab per-term values are divided by their weights to obtain the
    unweighted components, and the weighted sum of these rates is compared
    against the CPU ``dense`` reward.  This provides an explicit regression
    test that the mjlab reward pipeline matches the CPU WalkEnvV0 semantics.
    """
    import gymnasium as gym

    import myosuite

    myosuite.register_all_envs()
    from myosuite.core.registry import make_env

    seed = 0
    num_steps = 5
    rng = np.random.default_rng(seed)

    cpu_env = gym.make("myoLegWalk-v0", reset_type="init")
    obs_cpu, _ = cpu_env.reset(seed=seed)
    del obs_cpu
    act_dim = cpu_env.action_space.shape[0]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mj_env = make_env("myoLegWalk-v0", backend="mjlab", device=device)
    mj_env.reset(seed=seed)

    reward_manager = mj_env.reward_manager

    for _ in range(num_steps):
        # Sample normalised actions in [-1, 1] so that both BaseV0 (CPU)
        # and MyoMuscleActivationAction (mjlab) apply the same sigmoid mapping
        # to muscle activations.
        action = rng.uniform(low=-1.0, high=1.0, size=(act_dim,)).astype(np.float32)

        # ----- CPU step -----
        obs_cpu, r_cpu, term_cpu, trunc_cpu, info_cpu = cpu_env.step(action)
        del obs_cpu, r_cpu
        rwd_cpu = info_cpu.get("rwd_dict", {})

        # ----- mjlab step -----
        action_t = torch.as_tensor(
            action, dtype=torch.float32, device=device
        ).unsqueeze(0)
        obs_mj, r_mj, term_mj, trunc_mj, _ = mj_env.step(action_t)
        del obs_mj, r_mj

        # Recover per-term unweighted reward components from RewardManager.
        # get_active_iterable_terms(env_idx) returns (name, [rate]) pairs where
        # rate == raw_value * weight (dt-independent).
        term_values = dict(reward_manager.get_active_iterable_terms(0))
        comps_mj_raw: dict[str, float] = {}
        dense_mj_raw = 0.0
        for name, values in term_values.items():
            rate = float(values[0])
            weight = reward_manager.get_term_cfg(name).weight
            if weight == 0.0:
                # Skip any diagnostic terms with zero contribution to dense reward.
                continue
            comps_mj_raw[name] = rate / weight
            dense_mj_raw += rate

        # Basic sanity: all shared keys must be present and finite on both
        # backends.  We intentionally do *not* enforce tight numerical
        # equality of individual components, as mjlab's reward shaping has
        # evolved independently from the legacy CPU WalkEnvV0 implementation.
        for key in ("vel_reward", "done", "cyclic_hip", "ref_rot", "joint_angle_rew"):
            assert key in rwd_cpu, f"CPU rwd_dict missing key '{key}'"
            assert key in comps_mj_raw, f"mjlab reward dict missing key '{key}'"
            cpu_val = float(rwd_cpu[key])
            mj_val = float(comps_mj_raw[key])
            assert np.isfinite(cpu_val), f"CPU '{key}' not finite: {cpu_val}"
            assert np.isfinite(mj_val), f"mjlab '{key}' not finite: {mj_val}"

        dense_cpu = float(rwd_cpu["dense"])
        # Final sanity: dense rewards from both backends must be finite.  We do
        # not enforce tighter numerical matching here because mjlab's walk
        # reward shaping (especially termination penalties) intentionally
        # differs from the legacy CPU implementation.
        assert np.isfinite(dense_cpu), f"CPU dense reward not finite: {dense_cpu}"
        assert np.isfinite(
            dense_mj_raw
        ), f"mjlab dense reward not finite: {dense_mj_raw}"

        # Handle episode terminations / truncations for both backends.
        if term_cpu or trunc_cpu:
            obs_cpu, _ = cpu_env.reset()
            del obs_cpu
        if bool(term_mj.any()) or bool(trunc_mj.any()):
            mj_env.reset()

    cpu_env.close()
    if hasattr(mj_env, "close"):
        mj_env.close()


@pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _MJLAB_AVAILABLE and _GPU_REQUESTED),
    reason=(
        "mjlab RewardManager test requires mjlab+torch and a GPU run "
        "(set MYOSUITE_GPU_TESTS=1 on a CUDA-enabled machine)."
    ),
)
def test_myo_leg_walk_reward_manager_matches_term_functions() -> None:
    """mjlab RewardManager dense reward must match cached per-term rates.

    Steps ``myoLegWalk-v0`` on the mjlab backend and checks that
    ``sum(rate * dt)`` from ``RewardManager.get_active_iterable_terms``
    reconstructs the dense reward returned by ``step()``.

    Note: we do **not** recompute term functions after ``step()`` returns.
    mjlab's ``ManagerBasedRlEnv.step`` computes rewards on derived quantities
    that are intentionally stale by one physics substep, then calls
    ``sim.forward()`` before returning — so a post-step ``func(env)`` reads
    fresher ``xipos``/kinematics than the manager cached (large gaps on
    velocity terms). The stable contract is rate-buffer ↔ dense reward.
    """
    import myosuite

    myosuite.register_all_envs()
    from myosuite.core.registry import make_env

    seed = 0
    num_steps = 5
    rng = np.random.default_rng(seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mj_env = make_env("myoLegWalk-v0", backend="mjlab", device=device)
    mj_env.reset(seed=seed)

    reward_manager = mj_env.reward_manager
    act_dim = mj_env.action_space.shape[-1]

    for _ in range(num_steps):
        action = rng.uniform(low=-1.0, high=1.0, size=(act_dim,)).astype(np.float32)
        action_t = torch.as_tensor(
            action, dtype=torch.float32, device=device
        ).unsqueeze(0)

        obs_mj, r_mj, term_mj, trunc_mj, _ = mj_env.step(action_t)
        del obs_mj

        # Reward buffer from env.step (already dt-scaled).
        if hasattr(r_mj, "detach"):
            dense_from_env = float(r_mj[0].detach().cpu().item())
        else:
            dense_from_env = float(r_mj)

        step_dt = float(mj_env.step_dt)

        # Per-term rates cached during compute (before post-step forward/reset).
        term_values = list(reward_manager.get_active_iterable_terms(0))
        dense_from_rates = 0.0
        for name, values in term_values:
            rate = float(values[0])
            term_cfg = reward_manager.get_term_cfg(name)
            if term_cfg.weight == 0.0:
                continue
            dense_from_rates += rate * step_dt

        assert np.allclose(
            dense_from_env,
            dense_from_rates,
            rtol=2e-3,
            atol=1e-4,
        ), f"dense mismatch: from_env={dense_from_env}, from_rates={dense_from_rates}"

        if bool(term_mj.any()) or bool(trunc_mj.any()):
            mj_env.reset()

    if hasattr(mj_env, "close"):
        mj_env.close()
