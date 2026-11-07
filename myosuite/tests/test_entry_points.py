# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Integration tests for the two main MyoSuite entry-point patterns.

CPU pattern::

    import myosuite
    myosuite.register_all_envs()
    env = make_env("myoElbowPose1D6MRandom-v0")

MJX + Brax pattern::

    from myosuite.envs.myo.backends.mjx.brax_env import wrap_for_brax_training
    env = make_env("MjxElbowPoseRandom-v0", backend="mjx")
    brax_env = wrap_for_brax_training(env)

The MJX tests are skipped automatically when ``mujoco.mjx`` or ``brax``
are not available (version mismatch / not installed).
Run the CPU tests with any Python environment; run the MJX tests via
``uv run pytest myosuite/tests/test_entry_points.py::TestMjxEntryPoint``.
"""

from __future__ import annotations

import numpy as np
import pytest
from myosuite import make_env


pytestmark = pytest.mark.tier1


# ---------------------------------------------------------------------------
# CPU entry-point tests
# ---------------------------------------------------------------------------


class TestCpuEntryPoint:
    """Verify the CPU / Gymnasium entry point works end-to-end."""

    def test_make_cpu_env(self):
        """make_env() succeeds after explicit registration; reset+step return correct shapes."""
        import myosuite

        myosuite.register_all_envs()

        env = make_env("myoElbowPose1D6MRandom-v0")
        obs, info = env.reset()

        assert obs is not None, "reset() returned None obs"
        assert isinstance(obs, np.ndarray), f"obs is not ndarray: {type(obs)}"
        assert np.all(np.isfinite(obs)), "obs contains NaN/Inf after reset"

        obs2, rwd, terminated, truncated, info2 = env.step(env.action_space.sample())
        assert isinstance(obs2, np.ndarray), "step() obs is not ndarray"
        assert np.isfinite(rwd), f"reward is not finite: {rwd}"
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

        env.close()


# ---------------------------------------------------------------------------
# MJX entry-point tests
# ---------------------------------------------------------------------------

# Guard: skip entire class if MJX / brax are unavailable.
# Use try/except rather than pytest.importorskip because some environments
# raise AttributeError (mujoco/mjx version mismatch) rather than ImportError.
# The default mjx_impl=None uses JAX/XLA and works on CPU-only machines.
# Only GPU-specific impl (e.g. "warp") requires CUDA hardware.
try:
    from mujoco import mjx as _mjx_mod  # noqa: F401
    import brax as _brax_check  # noqa: F401

    _MJX_AVAILABLE = True
    _MJX_SKIP_REASON = None
except (ImportError, AttributeError) as _mjx_err:
    _MJX_AVAILABLE = False
    _MJX_SKIP_REASON = str(_mjx_err)


@pytest.mark.skipif(
    not _MJX_AVAILABLE,
    reason=f"MJX/brax not available: {_MJX_SKIP_REASON}",
)
class TestMjxEntryPoint:
    """Verify the MJX + Brax entry point works end-to-end.

    Skipped automatically when ``mujoco.mjx`` or ``brax`` are not
    importable (e.g., version mismatch on macOS dev machines, or simply not
    installed).  All tests pass on a GPU machine with the correct package set.
    """

    def test_make_mjx_env(self):
        """make() returns a MyoMjxEnvBase subclass for MjxElbowPoseRandom-v0."""
        from myosuite.envs.myo.backends.mjx import make
        from myosuite.envs.myo.backends.mjx.mjx_env_base import MyoMjxEnvBase

        try:
            env = make("MjxElbowPoseRandom-v0")
        except NotImplementedError as exc:
            # Upstream MJX currently does not implement some collision pairs
            # (e.g. cylinder–mesh); treat this as an environment limitation
            # rather than a hard failure for the entry-point test.
            pytest.skip(f"MJX env not supported on this mujoco.mjx build: {exc}")
        assert isinstance(
            env, MyoMjxEnvBase
        ), f"make() returned {type(env)}, expected MyoMjxEnvBase subclass"

    def test_brax_wrap(self):
        """wrap_for_brax_training() returns an object with reset and step attributes."""
        import jax

        from myosuite.envs.myo.backends.mjx import make
        from myosuite.envs.myo.backends.mjx.brax_env import (
            wrap_for_brax_training,
        )

        try:
            env = make("MjxElbowPoseRandom-v0")
        except NotImplementedError as exc:
            pytest.skip(f"MJX env not supported on this mujoco.mjx build: {exc}")
        brax_env = wrap_for_brax_training(env)

        assert hasattr(brax_env, "reset"), "brax env missing reset()"
        assert hasattr(brax_env, "step"), "brax env missing step()"
        obs = env.reset(jax.random.PRNGKey(0)).obs
        assert brax_env.observation_size == {"state": obs["state"].shape}

    def test_brax_auto_reset_restores_the_first_state(self):
        """Envs that reach the episode length restart from their first reset."""
        import jax
        import jax.numpy as jp

        from myosuite.envs.myo.backends.mjx.brax_env import (
            wrap_for_brax_training,
        )

        env = make_env("MjxElbowPoseRandom-v0", backend="mjx")
        brax_env = wrap_for_brax_training(env, episode_length=2)
        state = jax.jit(brax_env.reset)(jax.random.split(jax.random.PRNGKey(0), 2))
        first_qpos = state.data.qpos
        step = jax.jit(brax_env.step)
        action = jp.ones((2, env.action_size))
        state = step(state, action)
        assert not np.any(state.done)
        assert np.all(np.asarray(state.data.qvel) != 0.0)
        state = step(state, action)
        assert np.all(state.done == 1.0)
        np.testing.assert_array_equal(state.data.qpos, first_qpos)
        np.testing.assert_array_equal(state.info["AutoResetWrapper_done_count"], 1)

    def test_make_env_builds_named_mjx_env(self):
        """make_env(<Mjx id>, backend="mjx") builds the hand-written MJX env."""
        env = make_env("MjxElbowPoseRandom-v0", backend="mjx", num_envs=2)
        assert type(env).__name__ == "MjxPoseEnv"
        assert env._config.num_envs == 2

    def test_all_mjx_env_ids_listed(self):
        """ALL_ENVS contains at least the 6 expected environment names."""
        from myosuite.envs.myo.backends.mjx import ALL_ENVS

        expected = {
            "MjxElbowPoseFixed-v0",
            "MjxElbowPoseRandom-v0",
            "MjxFingerPoseFixed-v0",
            "MjxFingerPoseRandom-v0",
            "MjxHandReachFixed-v0",
            "MjxHandReachRandom-v0",
        }
        missing = expected - set(ALL_ENVS)
        assert not missing, f"ALL_ENVS is missing expected entries: {missing}"

    def test_make_all_mjx_envs(self):
        """make() succeeds for every name in ALL_ENVS without raising."""
        from myosuite.envs.myo.backends.mjx import make, ALL_ENVS

        for name in ALL_ENVS:
            try:
                env = make(name)
            except NotImplementedError as exc:
                pytest.skip(
                    f"MJX env '{name}' not supported on this mujoco.mjx build: {exc}"
                )
            assert env is not None, f"make('{name}') returned None"
