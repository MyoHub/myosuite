# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MjxModularTaskEnv — MJX backend driven by TaskConfig (Phase 5).

Usage::

    from myosuite.core.config import TaskConfig, GoalSpec
    from myosuite.envs.myo.backends.mjx.mjx_modular_env import MjxModularTaskEnv, modular_task_config

    env = MjxModularTaskEnv(TaskConfig())
    cfg = modular_task_config(TaskConfig())   # get the ConfigDict directly
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
from mujoco import mjx

from myosuite.core.config import GoalSpec, TaskConfig
from myosuite.envs.modular_env import _joint_qpos_width
from myosuite.envs.myo.backends.mjx.mjx_env_base import MjxEnvAccessor, MyoMjxEnvBase
from myosuite.envs.myo.backends.mjx.mjx_spec_preprocess import preprocess_mjx_spec


def modular_task_config(task_config: TaskConfig) -> config_dict.ConfigDict:
    """Build a mujoco_playground-compatible ConfigDict from a TaskConfig.

    Args:
        task_config: High-level task specification.

    Returns:
        Frozen ConfigDict with fields expected by MyoMjxEnvBase.
    """
    extra = task_config.backend.extra
    return config_dict.create(
        ctrl_dt=task_config.backend.ctrl_dt,
        sim_dt=task_config.backend.sim_dt,
        max_episode_steps=task_config.max_episode_steps,
        num_envs=int(extra.get("num_envs", 4096)),
        mjx_impl=extra.get("mjx_impl", None),
        norm_actions=extra.get("norm_actions", True),
    )


def _resolve_obs_term(key: str | Any) -> tuple[str, Any]:
    """Resolve an obs term key (string or callable) to a (label, fn) pair.

    Args:
        key: Observation term name string or a callable directly.

    Returns:
        Tuple of (label, callable).
    """
    if callable(key):
        return getattr(key, "__name__", repr(key)), key
    from myosuite.envs.modular_env import _load_obs_fn

    return key, _load_obs_fn(key)


def _resolve_reward_term(term: str | Any) -> tuple[str, Any]:
    """Resolve a reward term (string or callable) to a (label, fn) pair.

    Args:
        term: Reward term name string or a callable directly.

    Returns:
        Tuple of (label, callable).
    """
    if callable(term):
        return getattr(term, "__name__", repr(term)), term
    from myosuite.envs.modular_env import _load_reward_fn

    return term, _load_reward_fn(term)


def _sample_task_jax(
    goal: GoalSpec,
    mj_model: mujoco.MjModel,
    qpos_init: jax.Array,
    rng: jax.Array,
) -> dict[str, jax.Array]:
    """Sample a JAX task state from a GoalSpec.

    Args:
        goal: Goal specification.
        mj_model: MuJoCo model (joint qpos layout).
        qpos_init: Baseline ``qpos`` (shape ``(nq,)``); unlisted joints keep
            these values so ``target_angles`` matches ``joint_pos`` length.
        rng: JAX random key.

    Returns:
        Dict of JAX arrays forming the task state.
    """
    if goal.target_type == "joint_angles":
        nq = mj_model.nq
        target = jp.array(qpos_init, dtype=jp.float32).reshape((nq,))
        for j in range(mj_model.njnt):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if name not in goal.range:
                continue
            lo, hi = goal.range[name]
            qadr = int(mj_model.jnt_qposadr[j])
            span = _joint_qpos_width(mj_model, j)
            rng, subkey = jax.random.split(rng)
            if goal.randomize:
                vals = jax.random.uniform(
                    subkey, shape=(span,), minval=float(lo), maxval=float(hi)
                )
            else:
                vals = jp.full((span,), float(lo))
            target = target.at[qadr : qadr + span].set(vals)
        return {"target_angles": target}
    # site_positions — return from goal.extra if provided
    return {k: jp.array(v) for k, v in goal.extra.items()}


class MjxModularTaskEnv(MyoMjxEnvBase):
    """MJX (JAX-accelerated) environment driven by a TaskConfig.

    Mirrors :class:`~myosuite.envs.modular_env.ModularTaskEnv` for the MJX
    backend.  No subclassing needed — observation channels, reward terms, and
    goal sampling are fully specified by *task_config*.

    Args:
        task_config: Data-driven task specification.
        config_overrides: Optional key→value overrides applied on top of the
            derived ConfigDict.

    Example::

        from myosuite.core.config import TaskConfig
        from myosuite.envs.myo.backends.mjx.mjx_modular_env import MjxModularTaskEnv

        env = MjxModularTaskEnv(TaskConfig())
        state = env.reset(jax.random.PRNGKey(0))
    """

    _preprocess_spec = staticmethod(preprocess_mjx_spec)

    def __init__(
        self,
        task_config: TaskConfig,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        # Must set _task_config BEFORE calling super().__init__() because
        # MyoMjxEnvBase.__init__ calls self.setup_model(config) synchronously.
        self._task_config = task_config
        cfg = modular_task_config(task_config)
        super().__init__(cfg, config_overrides)

        # Resolve term functions now that setup_model has run
        self._obs_fns: list[tuple[str, Any]] = [
            _resolve_obs_term(k) for k in task_config.obs.keys
        ]
        self._reward_fns: list[tuple[str, Any]] = [
            _resolve_reward_term(t) for t in task_config.reward.terms
        ]

    # ------------------------------------------------------------------
    # MyoMjxEnvBase template methods
    # ------------------------------------------------------------------

    def setup_model(self, config: config_dict.ConfigDict) -> None:
        """Build MJ/MJX model from the task recipe.

        Args:
            config: Frozen ConfigDict derived from the TaskConfig.
        """
        from myosuite.core.model_builder import build_from_recipe
        from myosuite.core.muscle_conditions import apply_sarcopenia_to_spec

        impl = getattr(config, "mjx_impl", None) or None
        mj_model, spec = build_from_recipe(self._task_config.model)

        # Apply physiological conditions declared in actuator groups
        if any(g.condition == "sarcopenia" for g in self._task_config.actuators):
            spec = apply_sarcopenia_to_spec(spec)

        # Keep MJX JAX backend compatible with plane/mesh/hfield margins and
        # unsupported cylinder/ellipsoid contacts used by modular recipes.
        spec = self._preprocess_spec(spec, impl=impl)
        mj_model = spec.compile()

        mj_model.opt.timestep = config.sim_dt
        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(self._mj_model, impl=impl)
        self._xml_path = ""
        self._n_substeps = int(config.ctrl_dt / config.sim_dt)

    def sample_task(self, rng: jax.Array) -> dict[str, jax.Array]:
        """Sample a new episode goal from the configured GoalSpec.

        Args:
            rng: JAX random key.

        Returns:
            Dict of named JAX arrays stored in state.info.
        """
        qpos0 = jp.array(self._mj_model.qpos0, dtype=jp.float32)
        return _sample_task_jax(self._task_config.goal, self._mj_model, qpos0, rng)

    def get_obs_dict(
        self,
        accessor: MjxEnvAccessor,
        task_state: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Assemble obs dict by calling each configured obs term.

        Args:
            accessor: MJX physics state accessor.
            task_state: Current episode task state (goal, etc.).

        Returns:
            Ordered obs dict of JAX arrays.
        """
        extra = {**self._task_config.obs.extra, **task_state}
        return {key: jp.atleast_1d(fn(accessor, **extra)) for key, fn in self._obs_fns}

    def get_reward_dict(
        self,
        accessor: MjxEnvAccessor,
        task_state: dict[str, jax.Array],
        obs_dict: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Compute weighted reward from all configured reward terms.

        Args:
            accessor: MJX physics state accessor.
            task_state: Current episode task state.
            obs_dict: Output of get_obs_dict (not used directly).

        Returns:
            Dict with 'dense' (scalar), 'done' (0/1 float), and per-term keys.
        """
        extra = self._task_config.reward.extra
        combined: dict[str, jax.Array] = {}
        dense_total = jp.array(0.0)
        done = jp.array(0.0)

        for term, fn in self._reward_fns:
            result = fn(accessor, task_state, **extra)
            weight = self._task_config.reward.weight_for(term)
            dense_total = dense_total + weight * result.get("dense", jp.array(0.0))
            done = jp.maximum(
                done, jp.array(result.get("done", jp.array(0.0)), dtype=jp.float32)
            )
            for k, v in result.items():
                if k not in ("dense", "done"):
                    combined[f"{term}/{k}"] = v

        combined["dense"] = dense_total
        combined["done"] = done
        return combined
