# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tier A contract checks for myoLegWalk-v0.

This module encodes promotion gates for myoLegWalk-v0. The first checks are
hard contract checks. Tight numerical parity checks are tracked as xfail until
action/termination semantics are fully aligned across backends.
"""

from __future__ import annotations

import numpy as np
import pytest


pytestmark = pytest.mark.tier2


def _canonical_action(action: np.ndarray) -> np.ndarray:
    """Map canonical policy action to muscle activation domain.

    Current cross-backend policy convention for walk parity is to sample in
    [-1, 1] and map to [0, 1] before sending to CPU path.
    """
    return np.clip(0.5 * (action + 1.0), 0.0, 1.0).astype(np.float32)


@pytest.mark.tier1
def test_walk_cpu_contract_smoke() -> None:
    """CPU walk env must provide finite obs/reward under canonical actions."""
    import gymnasium as gym

    import myosuite

    myosuite.register_all_envs()

    env = gym.make("myoLegWalk-v0", reset_type="init")
    obs, _ = env.reset(seed=0)
    assert np.all(np.isfinite(np.asarray(obs)))

    rng = np.random.default_rng(0)
    act_dim = env.action_space.shape[0]
    for _ in range(5):
        action = _canonical_action(rng.uniform(-1.0, 1.0, size=(act_dim,)))
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(np.asarray(obs)))
        assert np.isfinite(float(reward))
        assert "rwd_dict" in info
        if terminated or truncated:
            env.reset()
    env.close()


def test_walk_tier_a_dense_reward_gate_cpu_vs_mjlab() -> None:
    """Future Tier A gate: dense reward parity within 10%."""
    try:
        import torch
    except ImportError as exc:
        pytest.skip(f"mjlab/torch not available for walk parity gate: {exc}")
    try:
        import mjlab  # noqa: F401
        import mjlab.envs  # noqa: F401
    except Exception as exc:
        pytest.skip(f"mjlab unavailable for walk parity gate: {exc}")

    import myosuite
    from myosuite.core.registry import make_env
    from myosuite.utils import gym

    myosuite.register_all_envs()

    cpu_env = gym.make("myoLegWalk-v0", reset_type="init")
    cpu_env.reset(seed=0)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        mj_env = make_env("myoLegWalk-v0", backend="mjlab", device=device)
    except Exception as exc:
        pytest.skip(f"could not build mjlab walk env for parity gate: {exc}")
    mj_env.reset(seed=0)

    rng = np.random.default_rng(0)
    act_dim = cpu_env.action_space.shape[0]
    cpu_total = 0.0
    mj_total = 0.0

    for _ in range(20):
        policy_action = rng.uniform(-1.0, 1.0, size=(act_dim,)).astype(np.float32)
        cpu_action = _canonical_action(policy_action)
        _, r_cpu, term_cpu, trunc_cpu, _ = cpu_env.step(cpu_action)
        cpu_total += float(r_cpu)

        action_t = torch.as_tensor(policy_action, dtype=torch.float32, device=device)
        action_t = action_t.unsqueeze(0)
        _, r_mj, term_mj, trunc_mj, _ = mj_env.step(action_t)
        mj_total += float(r_mj[0].detach().cpu().item())

        if term_cpu or trunc_cpu:
            cpu_env.reset()
        if bool(term_mj.any()) or bool(trunc_mj.any()):
            mj_env.reset()

    denom = abs(cpu_total) + 1e-6
    rel_diff = abs(cpu_total - mj_total) / denom
    assert rel_diff <= 0.10, f"walk dense reward diff {rel_diff:.3f} > 0.10"
