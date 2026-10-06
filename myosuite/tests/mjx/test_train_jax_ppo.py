#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Smoke tests for MJX Brax PPO training helper.

This module validates that the lightweight MJX training script
``myosuite.envs.myo.backends.mjx.train_jax_ppo`` can:

* Construct an MJX environment from a MyoSuite MJX env ID.
* Produce a valid initial state via ``reset()``.
* Step the environment once with a zero action.

The test is deliberately small and does **not** run a full PPO training loop;
that would be too slow for routine CI. Instead it checks that the key wiring
between:

* MyoSuite MJX envs (``myosuite.envs.myo.backends.mjx.make``),
* `mujoco_playground`'s `MjxEnv` interface, and
* Brax PPO configs (`ppo_config`)

is correct. When the optional MJX stack (``myosuite[mjx]``) is not installed,
the entire module is skipped.
"""

from __future__ import annotations

import os

import pytest


pytestmark = [pytest.mark.tier3, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Module-level guard: require MJX + mujoco_playground stack
# ---------------------------------------------------------------------------

try:
    from mujoco import mjx as _mjx_mod  # noqa: F401
    import jax  # noqa: F401
    import mujoco_playground as _mp  # noqa: F401
except (ImportError, AttributeError) as _err:  # pragma: no cover - defensive
    pytest.skip(
        f"MJX / mujoco_playground stack not available ({_err}); "
        "install extras with `pip install -e '.[mjx]'` or "
        "`uv sync --extra mjx-cuda` to run MJX training tests.",
        allow_module_level=True,
    )


def test_train_jax_ppo_env_reset_and_step(tmp_path, monkeypatch) -> None:
    """Environment built by ``train_jax_ppo`` must reset and step once.

    This is a minimal end-to-end smoke test:

    1. Use ``load_env_and_network_factory`` to construct an MJX env for
       ``MjxFingerPoseRandom-v0``.
    2. Call ``reset(rng)`` and verify the returned state has finite qpos.
    3. Call ``step(state, action)`` once with a zero action and verify
       the next state's qpos and reward are finite.

    The test does *not* invoke the full PPO training loop; that is covered
    by manual runs on GPU machines using the same helper.
    """
    import jax
    import jax.numpy as jnp

    from myosuite.envs.myo.backends.mjx import train_jax_ppo

    # Ensure any incidental wandb usage in downstream pipelines stays offline
    # and writes logs into a temporary directory.
    monkeypatch.setenv("WANDB_MODE", os.environ.get("WANDB_MODE", "offline"))
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))

    env_name = "MjxFingerPoseRandom-v0"
    env, ppo_params, network_factory = train_jax_ppo.load_env_and_network_factory(
        env_name
    )

    # Basic sanity checks on returned objects.
    assert env is not None
    assert callable(network_factory)
    assert "num_timesteps" in ppo_params

    # Reset the MJX env and check that qpos is finite.
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    qpos0 = jnp.asarray(state.data.qpos)
    assert jnp.all(jnp.isfinite(qpos0)), "Initial MJX qpos contains NaN/Inf"

    # Single zero-action step.
    action = jnp.zeros((env.mjx_model.nu,), dtype=jnp.float32)
    next_state = env.step(state, action)
    qpos1 = jnp.asarray(next_state.data.qpos)
    reward1 = jnp.asarray(next_state.reward)

    assert jnp.all(jnp.isfinite(qpos1)), "Next-step MJX qpos contains NaN/Inf"
    assert jnp.all(jnp.isfinite(reward1)), "Next-step MJX reward contains NaN/Inf"
