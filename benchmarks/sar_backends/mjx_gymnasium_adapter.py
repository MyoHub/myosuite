# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Gymnasium adapter for MJX (JAX-based) environments.

Wraps a ``MyoMjxEnvBase`` instance as a standard ``gymnasium.Env`` so that
CPU-side RL libraries (e.g. Stable-Baselines3) can train on MJX environments
without modification.

Key properties:
  - Actions are numpy arrays; the adapter converts them to JAX at each step.
  - Observations are numpy arrays converted from JAX ``state.obs["state"]``.
  - The ``info`` dict contains reward component metrics from ``state.metrics``
    plus episode-level debug info (qpos snapshot, com_vel).
  - ``jax.jit`` is applied to ``reset`` and ``step`` once at construction.

Usage::

    from myosuite.envs.myo.backends.mjx import make
    from benchmarks.sar_backends.mjx_gymnasium_adapter import MJXGymnasiumAdapter

    mjx_env = make("MjxLegWalk-v0")
    env = MJXGymnasiumAdapter(mjx_env, seed=0)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import gymnasium

    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False


class MJXGymnasiumAdapter(gymnasium.Env if _DEPS_OK else object):
    """Adapt a ``MyoMjxEnvBase`` to the standard ``gymnasium.Env`` API.

    The adapter JIT-compiles ``reset`` and ``step`` at construction time so
    that subsequent calls are fast (after the first warm-up call).

    Args:
        mjx_env: An instantiated ``MyoMjxEnvBase`` environment.
        seed: Initial RNG seed (default: 0).
    """

    metadata = {"render_modes": []}

    def __init__(self, mjx_env, seed: int = 0) -> None:
        if not _DEPS_OK:
            raise ImportError("jax and gymnasium are required for MJXGymnasiumAdapter")

        super().__init__()
        self._env = mjx_env
        self._rng = jax.random.PRNGKey(seed)
        self._state = None

        # JIT-compile reset and step once
        self._jit_reset = jax.jit(mjx_env.reset)
        self._jit_step = jax.jit(mjx_env.step)

        # Determine action and observation dimensions via a dummy reset
        rng_dummy, rng_init = jax.random.split(self._rng)
        dummy_state = self._jit_reset(rng_init)
        dummy_obs = np.array(dummy_state.obs["state"])
        self._rng = rng_dummy

        nu = mjx_env.action_size
        obs_size = dummy_obs.shape[0]

        self.action_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(nu,), dtype=np.float32
        )
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset to a new episode.

        Args:
            seed: If provided, reset the RNG to ``jax.random.PRNGKey(seed)``.
            options: Ignored.

        Returns:
            ``(obs, info)`` where obs is a float32 numpy array.
        """
        if seed is not None:
            self._rng = jax.random.PRNGKey(seed)

        self._rng, rng_reset = jax.random.split(self._rng)
        self._state = self._jit_reset(rng_reset)

        obs = np.array(self._state.obs["state"], dtype=np.float32)
        info = self._extract_info()
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one control step.

        Args:
            action: Muscle activation vector (float32, shape ``(nu,)``).

        Returns:
            ``(obs, reward, terminated, truncated, info)`` in standard
            gymnasium convention.  ``terminated`` is True when the episode
            ends due to falling / rotation; ``truncated`` is never raised here
            (truncation is handled by the inner env's ``max_episode_steps``).
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        action_jax = jnp.array(action, dtype=jnp.float32)
        self._state = self._jit_step(self._state, action_jax)

        obs = np.array(self._state.obs["state"], dtype=np.float32)
        reward = float(self._state.reward)
        terminated = bool(self._state.done)
        truncated = False  # inner env handles truncation via auto-reset
        info = self._extract_info()
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """No resources to release."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_info(self) -> dict:
        """Extract diagnostic info from the current state.

        Returns a dict with:
          - All ``state.metrics`` keys (reward components, solved_frac, …)
          - ``qpos``: current joint position snapshot (numpy)
          - ``com_vel_y``: y-axis COM velocity (extracted from metrics if available)

        Returns:
            Info dict (all values are plain Python floats or numpy arrays).
        """
        info: dict = {}

        # Reward component metrics (populated by _update_metrics in MjxWalkEnv)
        for k, v in self._state.metrics.items():
            info[k] = float(v)

        # Physics snapshot for detailed diagnostics
        info["qpos"] = np.array(self._state.data.qpos, dtype=np.float32)
        info["qvel"] = np.array(self._state.data.qvel, dtype=np.float32)
        info["act"] = np.array(self._state.data.act, dtype=np.float32)

        # Mimic the "solved" key that WalkEnvV0 places in info (for SB3 Monitor)
        if "solved_frac" in info:
            info["solved"] = info["solved_frac"] > 0

        return info
