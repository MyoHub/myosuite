# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MyoMjxEnvBase — abstract base for MyoSuite MJX (JAX) environments.

Provides:
- ``MjxEnvAccessor``: wraps ``mjx.Data`` and implements the ``EnvAccessor``
  protocol so that term functions in ``myosuite/terms/`` work on the MJX path.
- ``MyoMjxEnvBase``: thin template base that handles ``reset()`` / ``step()``
  boilerplate (physics step, auto-reset, obs/reward assembly) and delegates
  task logic to subclass template methods.

Subclasses must implement:
    ``setup_model(config)``
        Build ``self._mj_model``, ``self._mjx_model``, ``self._xml_path``, and
        ``self._n_substeps``.
    ``sample_task(rng)``
        Sample a new task config for one episode; return as a flat dict of
        JAX arrays that is merged into ``state.info``.
    ``get_obs_dict(accessor, task_state)``
        Compute observation dict from current physics state and task config.
    ``get_reward_dict(accessor, task_state, obs_dict)``
        Compute reward dict; must include ``"dense"`` (scalar) and ``"done"``
        (0/1 float).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco_playground import State
from mujoco_playground._src import mjx_env

from myosuite.core.protocols import PhysicsPath
from myosuite.terms.base_action import sigmoid_muscle_activation


class MjxEnvAccessor:
    """``EnvAccessor`` implementation for the MJX (JAX) backend.

    Wraps a read-only snapshot of ``mjx.Model`` + ``mjx.Data`` and returns
    ``jax.Array`` from every accessor method so that term functions in
    ``myosuite/terms/`` remain backend-agnostic.

    Args:
        model: Compiled MJX model (device-side).
        data: Current MJX simulation data (device-side).
        ctrl_dt: Control timestep in seconds (``sim_dt × n_substeps``).
    """

    def __init__(self, model: mjx.Model, data: mjx.Data, ctrl_dt: float) -> None:
        self._model = model
        self._data = data
        self._ctrl_dt = ctrl_dt

    @property
    def physics_path(self) -> PhysicsPath:
        """MJX backend identifier."""
        return PhysicsPath.MJX

    @property
    def model(self) -> mjx.Model:
        """The device-side MJX model."""
        return self._model

    @property
    def data(self) -> mjx.Data:
        """The device-side MJX data snapshot."""
        return self._data

    def joint_pos(self) -> jax.Array:
        """Joint position vector, shape ``(nq,)``."""
        return self._data.qpos

    def joint_vel(self) -> jax.Array:
        """Joint velocity vector scaled by ctrl_dt, shape ``(nv,)``."""
        return self._data.qvel * self._ctrl_dt

    def muscle_act(self) -> jax.Array:
        """Muscle activation vector, shape ``(na,)``."""
        return self._data.act

    def site_xpos(self, site_ids: Any) -> jax.Array:
        """Cartesian positions of requested sites, shape ``(len, 3)``."""
        return self._data.site_xpos[site_ids]

    def time(self) -> jax.Array:
        """Current simulation time as a scalar JAX array."""
        return self._data.time

    def ctrl_range(self) -> jax.Array:
        """Actuator control range, shape ``(nu, 2)``."""
        return self._model.actuator_ctrlrange

    def dt(self) -> float:
        """Control timestep in seconds."""
        return self._ctrl_dt

    def array_module(self):
        """Return the JAX numpy module for this backend."""
        return jp


# ---------------------------------------------------------------------------
# Helper: allocate MJX data from host arrays
# ---------------------------------------------------------------------------


def _make_mjx_data(
    mj_model: mujoco.MjModel,
    *,
    qpos: jax.Array,
    qvel: jax.Array,
    ctrl: jax.Array,
    impl: str | None = None,
    nconmax: int = 512,
) -> mjx.Data:
    """Allocate MJX data and fill initial arrays.

    Args:
        mj_model: Host-side MuJoCo model (used for shape metadata).
        qpos: Initial joint positions.
        qvel: Initial joint velocities.
        ctrl: Initial ctrl vector.
        impl: MJX backend impl string.  ``None`` (default) selects the JAX/XLA
            backend and works on CPU, GPU, and TPU.  Pass ``"warp"`` to
            explicitly request the MuJoCo Warp backend (GPU only).
        nconmax: Max contacts — only used by the ``"warp"`` backend.

    Returns:
        Initialised ``mjx.Data`` on device.
    """
    if impl == "warp":
        data = mjx.make_data(mj_model, impl=impl, nconmax=nconmax, njmax=mj_model.njmax)
    else:
        # XLA backend (CPU / GPU / TPU): nconmax / njmax not supported
        data = mjx.make_data(mj_model, impl=impl)
    data = data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
    return data


def _obs_dict_to_vec(obs_dict: dict[str, jax.Array]) -> jax.Array:
    """Concatenate an obs dict into a 1-D JAX array.

    Args:
        obs_dict: Dict of named observation arrays.

    Returns:
        Flat 1-D JAX array.
    """
    return jp.concatenate([jp.atleast_1d(v).ravel() for v in obs_dict.values()])


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class MyoMjxEnvBase(mjx_env.MjxEnv):
    """Abstract base for MyoSuite MJX environments.

    Handles the ``reset()`` / ``step()`` boilerplate: model loading, data
    allocation, accessor construction, obs/reward assembly, auto-reset.
    Subclasses focus on task logic only.

    Subclasses must implement:
        ``setup_model(config)``
        ``sample_task(rng)``
        ``get_obs_dict(accessor, task_state)``
        ``get_reward_dict(accessor, task_state, obs_dict)``

    Optionally override:
        ``initial_qpos_qvel(rng)``  — default: ``(model.qpos0, zeros)``
        ``initial_metrics()``       — default: ``{}``
        ``_normalize_action(action)`` — default: muscle sigmoid
        ``_update_metrics(metrics, rwd_dict, step_count)`` — default: no-op
    """

    def __init__(self, config, config_overrides=None) -> None:
        super().__init__(config, config_overrides)
        self._mj_model: mujoco.MjModel | None = None
        self._mjx_model: mjx.Model | None = None
        self._xml_path: str = ""
        self._n_substeps: int = 1
        self.setup_model(self._config)

    # ------------------------------------------------------------------
    # Template methods — subclasses must override
    # ------------------------------------------------------------------

    def setup_model(self, config) -> None:
        """Build MJ/MJX model and set ``_n_substeps`` from *config*.

        Args:
            config: Frozen ``ConfigDict`` passed to the constructor.

        After this call, the following attributes must be set:
            ``self._mj_model``, ``self._mjx_model``,
            ``self._xml_path``, ``self._n_substeps``.
        """
        raise NotImplementedError

    def sample_task(self, rng: jax.Array) -> dict[str, jax.Array]:
        """Sample a new episode task configuration.

        Args:
            rng: JAX random key.

        Returns:
            Flat dict of named JAX arrays stored verbatim in ``state.info``.
            Each value must have a fixed shape across episodes (required for
            the ``jp.where`` auto-reset logic in ``step()``).
        """
        raise NotImplementedError

    def get_obs_dict(
        self,
        accessor: MjxEnvAccessor,
        task_state: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Compute the observation dictionary.

        Args:
            accessor: Read-only accessor for the current physics state.
            task_state: Episode task configuration (from ``sample_task``).

        Returns:
            Dict of named JAX arrays; values are concatenated into the obs
            vector by ``_obs_dict_to_vec``.
        """
        raise NotImplementedError

    def get_reward_dict(
        self,
        accessor: MjxEnvAccessor,
        task_state: dict[str, jax.Array],
        obs_dict: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Compute the reward dictionary.

        Args:
            accessor: Read-only accessor for the current physics state.
            task_state: Episode task configuration.
            obs_dict: Output of ``get_obs_dict``.

        Returns:
            Dict that must include at least ``"dense"`` (scalar reward) and
            ``"done"`` (0.0 or 1.0 float scalar).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def initial_qpos_qvel(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return initial joint positions and velocities.

        Args:
            rng: JAX random key (available for randomised initialisation).

        Returns:
            Tuple ``(qpos, qvel)`` as device JAX arrays.
        """
        return jp.array(self._mj_model.qpos0), jp.zeros(self._mjx_model.nv)

    def initial_metrics(self) -> dict[str, jax.Array]:
        """Return a zero-valued metrics dict for ``State`` initialisation.

        Returns:
            Dict of named zero scalars; keys must match those updated in
            ``_update_metrics``.
        """
        return {}

    def _normalize_action(self, action: jax.Array) -> jax.Array:
        """Map normalised action ``[-1, 1]`` to muscle activation ``[0, 1]``.

        Default: sigmoid ``1 / (1 + exp(-5 * (a - 0.5)))``.  Can be disabled
        by setting ``config.norm_actions = False``, in which case the action is
        passed through unchanged (useful for wrappers like ``FatigueWrapper``
        that apply their own normalisation).

        Args:
            action: Raw action from the policy.

        Returns:
            Normalised ctrl vector passed to the physics step.
        """
        if not getattr(self._config, "norm_actions", True):
            return action
        return sigmoid_muscle_activation(action, jp)

    def _update_metrics(
        self,
        metrics: dict[str, jax.Array],
        rwd_dict: dict[str, jax.Array],
        step_count: jax.Array,
    ) -> dict[str, jax.Array]:
        """Update the metrics dict after a physics step.

        Args:
            metrics: Current metrics from ``state.metrics``.
            rwd_dict: Output of ``get_reward_dict``.
            step_count: Current step index (post-increment).

        Returns:
            Updated metrics dict (mutated in place via ``dict.update``).
        """
        return metrics

    # ------------------------------------------------------------------
    # Gymnasium-style interface (reset / step)
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> State:
        """Reset to a new episode.

        Args:
            rng: JAX random key for episode initialisation.

        Returns:
            Initial ``State``.
        """
        rng, rng1, rng2 = jax.random.split(rng, 3)
        qpos, qvel = self.initial_qpos_qvel(rng1)
        task_state = self.sample_task(rng2)

        impl = getattr(self._config, "mjx_impl", "warp") or None
        nconmax = int(getattr(self._config, "nconmax", 125 * self._config.num_envs))
        data = _make_mjx_data(
            self._mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=jp.zeros((self._mjx_model.nu,)),
            impl=impl,
            nconmax=nconmax,
        )
        # Compute derived quantities (xpos, xipos, cvel, actuator_length, etc.)
        # so initial observation matches CPU env after mj_forward.
        data = mjx.forward(self._mjx_model, data)

        ctrl_dt = float(self._config.ctrl_dt)
        accessor = MjxEnvAccessor(self._mjx_model, data, ctrl_dt)
        obs_dict = self.get_obs_dict(accessor, task_state)
        obs = _obs_dict_to_vec(obs_dict)

        info = {
            "rng": rng,
            "step_count": jp.array(0, dtype=jp.int32),
            **task_state,
        }
        reward, done = jp.zeros(2)
        return State(data, {"state": obs}, reward, done, self.initial_metrics(), info)

    def step(self, state: State, action: jax.Array) -> State:
        """Advance one control step.

        Applies action normalisation, steps MJX physics, computes obs/reward
        via term functions, and handles the auto-reset (re-sample task on
        episode termination or truncation).

        Args:
            state: Current environment ``State``.
            action: Policy action (normalised to ``[-1, 1]``).

        Returns:
            Next ``State``.
        """
        norm_action = self._normalize_action(action)
        data = mjx_env.step(self._mjx_model, state.data, norm_action, self._n_substeps)

        ctrl_dt = float(self._config.ctrl_dt)
        task_state = {
            k: v for k, v in state.info.items() if k not in ("rng", "step_count")
        }
        accessor = MjxEnvAccessor(self._mjx_model, data, ctrl_dt)
        obs_dict = self.get_obs_dict(accessor, task_state)
        rwd_dict = self.get_reward_dict(accessor, task_state, obs_dict)

        obs = _obs_dict_to_vec(obs_dict)
        reward = rwd_dict["dense"]
        done = jp.array(rwd_dict["done"], dtype=jp.float32)

        # Auto-reset: re-sample task when done or truncated
        step_count = state.info["step_count"] + 1
        truncation = jp.where(
            step_count >= self._config.max_episode_steps,
            1.0 - done,
            jp.array(0.0),
        )
        reset_flag = jp.logical_or(done, truncation)

        rng, rng1 = jax.random.split(state.info["rng"])
        new_task = self.sample_task(rng1)

        # Merge task_state with new_task on reset; keep keys only in task_state
        # (e.g. wrapper-added episode_done) unchanged so we don't require
        # sample_task() to return every key that may appear in state.info.
        task_keys_in_new = {
            k: jp.where(reset_flag, new_task[k], task_state[k])
            for k in task_state
            if k in new_task
        }
        task_keys_other = {k: task_state[k] for k in task_state if k not in new_task}

        new_info = {
            "rng": rng,
            "step_count": jp.where(reset_flag, jp.array(0, dtype=jp.int32), step_count),
            **task_keys_in_new,
            **task_keys_other,
        }

        metrics = self._update_metrics(state.metrics, rwd_dict, step_count)
        return state.replace(
            data=data,
            obs={"state": obs},
            reward=reward,
            done=done,
            metrics=metrics,
            info=new_info,
        )

    # ------------------------------------------------------------------
    # MjxEnv interface (required properties)
    # ------------------------------------------------------------------

    @property
    def xml_path(self) -> str:
        """Path to the source MJCF file."""
        return self._xml_path

    @property
    def action_size(self) -> int:
        """Number of actuators."""
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        """Host-side MuJoCo model."""
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        """Device-side MJX model."""
        return self._mjx_model

    @property
    def impl(self) -> str | None:
        """MJX backend impl string (``"jax"``, ``"warp"``, or ``None`` for XLA default)."""
        return getattr(self._config, "mjx_impl", None) or None

    @property
    def n_substeps(self) -> int:
        """Number of physics substeps per control step."""
        return self._n_substeps
