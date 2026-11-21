# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Batched GPU/JAX driver for MyoSuite MJX / Warp environments."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import elements
import numpy as np


class MJXDriver:
    """Batched GPU/JAX driver for MyoSuite MJX environments.

    Executes ``num_envs`` MJX environments in lockstep inside a single
    ``jax.jit(jax.vmap(...))`` kernel while preserving terminal observations
    (``is_last=True``, ``is_terminal``) prior to episode reset.

    Args:
        env_or_ctor: Instantiated MJX environment or zero-arg constructor.
        num_envs: Number of parallel environments stepped on device.
        max_episode_steps: Optional truncation horizon override.
        seed: Base PRNG seed for environment initialization and resets.
        act_key: Key of the continuous action tensor in ``act_space``.
        log_metrics: Expose scalar entries in ``state.metrics`` as ``log/<key>``.
        **kwargs: Forwarded to policy and step callbacks.
    """

    def __init__(
        self,
        env_or_ctor: Any,
        num_envs: int,
        max_episode_steps: int | None = None,
        seed: int = 0,
        act_key: str = "action",
        log_metrics: bool = True,
        **kwargs: Any,
    ) -> None:
        import jax
        import jax.numpy as jnp

        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self._jax = jax
        self._jnp = jnp

        with jax.transfer_guard("allow"):
            self.env = env_or_ctor() if callable(env_or_ctor) else env_or_ctor
            self.length = int(num_envs)
            self.act_key = act_key
            self.log_metrics = bool(log_metrics)
            self.kwargs = kwargs

            if max_episode_steps is None:
                cfg = getattr(self.env, "_config", None)
                max_episode_steps = int(getattr(cfg, "max_episode_steps", 1000))
            self.max_episode_steps = int(max_episode_steps)

            sample_state = self.env.reset(jax.random.PRNGKey(int(seed)))
            self._obs_is_dict = isinstance(sample_state.obs, Mapping)
            self._metric_keys = (
                sorted(sample_state.metrics.keys())
                if self.log_metrics and isinstance(sample_state.metrics, Mapping)
                else []
            )

            obs_spaces: dict[str, elements.Space] = {
                "reward": elements.Space(np.float32),
                "is_first": elements.Space(bool),
                "is_last": elements.Space(bool),
                "is_terminal": elements.Space(bool),
            }
            if self._obs_is_dict:
                for k, v in sample_state.obs.items():
                    obs_spaces[k.replace("/", "_")] = elements.Space(
                        np.float32, v.shape
                    )
            else:
                obs_spaces["proprio"] = elements.Space(
                    np.float32, sample_state.obs.shape
                )
            for k in self._metric_keys:
                obs_spaces[f"log/{k}"] = elements.Space(np.float32)
            self.obs_space = obs_spaces

            act_size = int(self.env.action_size)
            self.act_space: dict[str, elements.Space] = {
                self.act_key: elements.Space(np.float32, (act_size,), -1.0, 1.0),
                "reset": elements.Space(bool),
            }

            env = self.env
            horizon = self.max_episode_steps

            def _step_or_reset_one(
                state: Any, ep_step: Any, rng: Any, act: Any, do_reset: Any
            ) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
                rng, reset_key = jax.random.split(rng)

                def _do_reset(_: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
                    s0 = env.reset(reset_key)
                    return (
                        s0,
                        jnp.int32(0),
                        jnp.float32(0.0),
                        jnp.bool_(True),
                        jnp.bool_(False),
                        jnp.bool_(False),
                    )

                def _do_step(_: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
                    clipped = jnp.clip(act, -1.0, 1.0)
                    s1 = env.step(state, clipped)
                    next_step = ep_step + jnp.int32(1)
                    terminated = jnp.asarray(s1.done) > 0.5
                    truncated = (
                        (next_step >= horizon) & ~terminated
                        if horizon > 0
                        else jnp.bool_(False)
                    )
                    is_last = terminated | truncated
                    return (
                        s1,
                        next_step,
                        jnp.float32(s1.reward),
                        jnp.bool_(False),
                        is_last,
                        terminated,
                    )

                new_state, new_ep_step, rew, is_first, is_last, is_term = jax.lax.cond(
                    do_reset, _do_reset, _do_step, operand=None
                )
                return new_state, new_ep_step, rng, rew, is_first, is_last, is_term

            self._init_batch = jax.jit(jax.vmap(env.reset))
            self._advance_batch = jax.jit(jax.vmap(_step_or_reset_one))

            base_key = jax.random.PRNGKey(int(seed))
            keys = jax.random.split(base_key, self.length * 2)
            self._rngs = keys[: self.length]
            self._states = self._init_batch(keys[self.length :])
            self._ep_steps = jnp.zeros((self.length,), dtype=jnp.int32)

        self.callbacks: list[Callable[..., Any]] = []
        self.acts: dict[str, np.ndarray] = {}
        self.carry: Any = None
        self.reset()

    def reset(self, init_policy: Callable[[int], Any] | None = None) -> None:
        """Reset driver action buffers and policy recurrent state."""
        self.acts = {
            k: np.zeros((self.length, *v.shape), v.dtype)
            for k, v in self.act_space.items()
        }
        self.acts["reset"] = np.ones(self.length, bool)
        self.carry = init_policy(self.length) if init_policy is not None else None

    def close(self) -> None:
        """No-op close hook matching ``embodied.Driver``."""

    def on_step(self, callback: Callable[..., Any]) -> None:
        """Register a per-transition callback ``fn(tran, worker_idx)``."""
        self.callbacks.append(callback)

    def __call__(
        self, policy: Callable[..., Any], steps: int = 0, episodes: int = 0
    ) -> None:
        """Roll out ``policy`` for at least ``steps`` transitions or ``episodes``."""
        step, episode = 0, 0
        while step < steps or episode < episodes:
            step, episode = self._step(policy, step, episode)

    def _step(
        self, policy: Callable[..., Any], step: int, episode: int
    ) -> tuple[int, int]:
        jax, jnp = self._jax, self._jnp
        with jax.transfer_guard("allow"):
            act_arr = jax.device_put(
                np.asarray(self.acts[self.act_key], dtype=np.float32)
            )
            do_reset = jax.device_put(np.asarray(self.acts["reset"], dtype=bool))

            (
                self._states,
                self._ep_steps,
                self._rngs,
                rew,
                is_first,
                is_last,
                is_term,
            ) = self._advance_batch(
                self._states, self._ep_steps, self._rngs, act_arr, do_reset
            )

            obs_dev: dict[str, Any] = {
                "reward": rew,
                "is_first": is_first,
                "is_last": is_last,
                "is_terminal": is_term,
            }
            if self._obs_is_dict:
                for k, v in self._states.obs.items():
                    obs_dev[k.replace("/", "_")] = jnp.asarray(v, jnp.float32)
            else:
                obs_dev["proprio"] = jnp.asarray(self._states.obs, jnp.float32)

            logs_dev = {
                f"log/{k}": jnp.asarray(self._states.metrics[k], jnp.float32)
                for k in self._metric_keys
            }
            obs, logs = jax.device_get((obs_dev, logs_dev))

        self.carry, acts, outs = policy(self.carry, obs, **self.kwargs)
        if obs["is_last"].any():
            mask = ~obs["is_last"]
            acts = {k: self._mask(v, mask) for k, v in acts.items()}
        self.acts = {**acts, "reset": obs["is_last"].copy()}
        trans = {**obs, **acts, **outs, **logs}
        for i in range(self.length):
            trn = elements.tree.map(lambda x, idx=i: x[idx], trans)
            for fn in self.callbacks:
                fn(trn, i, **self.kwargs)
        step += len(obs["is_first"])
        episode += int(obs["is_last"].sum())
        return step, episode

    @staticmethod
    def _mask(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
        while mask.ndim < value.ndim:
            mask = mask[..., None]
        return value * mask.astype(value.dtype)
