# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Shared base class for MJX MuscleMimic environments.

Both the bimanual-arm and full-body MuscleMimic tasks share identical
observation, reward, and task-sampling logic.  Only ``setup_model`` differs
(different compile functions, different solver tuning).

Subclasses must implement :meth:`setup_model` and call
:meth:`_init_tracking_state` at the end of that method after
``self._mj_model`` has been set.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict

from myosuite.envs.myo.backends.mjx.mjx_env_base import MjxEnvAccessor, MyoMjxEnvBase


class MjxMuscleMimicBase(MyoMjxEnvBase):
    """Shared base for bimanual and full-body MuscleMimic MJX environments.

    Provides ``sample_task``, ``get_obs_dict``, ``get_reward_dict``,
    ``initial_metrics``, and ``_update_metrics``.  Subclasses only need to
    implement :meth:`setup_model` and call :meth:`_init_tracking_state`
    after compiling the MuJoCo model.
    """

    # ------------------------------------------------------------------
    # Shared setup helper
    # ------------------------------------------------------------------

    def _init_tracking_state(self, config: config_dict.ConfigDict) -> None:
        """Populate actuator and site-tracking indices from the compiled model.

        Must be called at the end of ``setup_model`` after ``self._mj_model``
        is set.

        Args:
            config: Frozen environment config dict; must contain
                ``mimic_site_names``.
        """
        self._actuator_ids = tuple(range(self._mj_model.nu))
        self._actuator_names = tuple(
            mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx)
            for idx in self._actuator_ids
        )
        self._track_site_ids = jp.array(
            [
                mujoco.mj_name2id(
                    self._mj_model,
                    mujoco.mjtObj.mjOBJ_SITE.value,
                    site_name,
                )
                for site_name in config.mimic_site_names
            ],
            dtype=jp.int32,
        )

    # ------------------------------------------------------------------
    # Shared template methods
    # ------------------------------------------------------------------

    def sample_task(self, rng: jax.Array) -> dict[str, jax.Array]:
        """Sample random target sites for the tracking task.

        Args:
            rng: JAX random key.

        Returns:
            Dict with ``"target_sites"`` — array of shape ``(n_sites, 3)``.
        """
        if self._track_site_ids.size == 0:
            return {"target_sites": jp.zeros((0, 3), dtype=jp.float32)}

        lo = jp.asarray(self._config.target_site_range.low, dtype=jp.float32)
        hi = jp.asarray(self._config.target_site_range.high, dtype=jp.float32)
        target_sites = jax.random.uniform(
            rng,
            (self._track_site_ids.size, 3),
            minval=lo,
            maxval=hi,
        )
        return {"target_sites": target_sites}

    def get_obs_dict(
        self,
        accessor: MjxEnvAccessor,
        task_state: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Build the observation dict from physics state and task config.

        Enabled channels are controlled by config flags:
        ``enable_joint_pos_observations``, ``enable_muscle_*_observations``,
        etc.

        Args:
            accessor: MJX env accessor wrapping current mjx.Data.
            task_state: Dict produced by :meth:`sample_task`.

        Returns:
            Flat observation dict mapping string keys to JAX scalars/arrays.
        """
        data = accessor.data
        obs_dict: dict[str, jax.Array] = {}

        if self._config.enable_joint_pos_observations:
            obs_dict["q_all_pos"] = data.qpos
        if self._config.enable_joint_vel_observations:
            obs_dict["dq_all_vel"] = data.qvel

        for act_idx, actuator_name in zip(self._actuator_ids, self._actuator_names):
            suffix = actuator_name.lower()
            if self._config.enable_muscle_length_observations:
                obs_dict[f"muscle_length_{suffix}"] = data.actuator_length[act_idx]
            if self._config.enable_muscle_velocity_observations:
                obs_dict[f"muscle_velocity_{suffix}"] = data.actuator_velocity[act_idx]
            if self._config.enable_muscle_force_observations:
                obs_dict[f"muscle_force_{suffix}"] = data.actuator_force[act_idx]
            if self._config.enable_muscle_excitation_observations:
                obs_dict[f"muscle_excitation_{suffix}"] = data.ctrl[act_idx]
            if self._config.enable_muscle_activation_observations:
                obs_dict[f"muscle_activation_{suffix}"] = data.act[act_idx]

        if self._track_site_ids.size:
            site_pos = data.site_xpos[self._track_site_ids]
            target_sites = task_state["target_sites"]
            obs_dict["mimic_site_pos"] = site_pos.ravel()
            obs_dict["mimic_site_target"] = target_sites.ravel()
            obs_dict["mimic_site_error"] = (target_sites - site_pos).ravel()

        return obs_dict

    def get_reward_dict(
        self,
        accessor: MjxEnvAccessor,
        task_state: dict[str, jax.Array],
        obs_dict: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Compute site-tracking reward.

        Args:
            accessor: MJX env accessor (unused; observations already in obs_dict).
            task_state: Current task state dict.
            obs_dict: Observation dict from :meth:`get_obs_dict`.

        Returns:
            Dict with keys ``"dense"``, ``"done"``, ``"tracking"``,
            ``"solved"``.
        """
        if self._track_site_ids.size == 0:
            dense = jp.array(0.0, dtype=jp.float32)
            solved = jp.array(0.0, dtype=jp.float32)
        else:
            site_err = obs_dict["mimic_site_error"].reshape((-1, 3))
            site_dist = jp.mean(jp.linalg.norm(site_err, axis=-1))
            dense = jp.exp(-float(self._config.tracking_reward_scale) * site_dist)
            solved = jp.array(
                site_dist < float(self._config.tracking_success_threshold),
                dtype=jp.float32,
            )

        done = jp.array(0.0, dtype=jp.float32)
        return {
            "dense": dense,
            "done": done,
            "tracking": dense,
            "solved": solved,
        }

    def initial_metrics(self) -> dict[str, jax.Array]:
        """Return zeroed episode metrics dict.

        Returns:
            Dict with ``"tracking_reward"`` and ``"solved_frac"`` set to zero.
        """
        zero = jp.array(0.0, dtype=jp.float32)
        return {"tracking_reward": zero, "solved_frac": zero}

    def _update_metrics(
        self,
        metrics: dict[str, jax.Array],
        rwd_dict: dict[str, jax.Array],
        step_count: jax.Array,
    ) -> dict[str, jax.Array]:
        """Update running episode metrics from the latest reward dict.

        Args:
            metrics: Current metrics dict (mutated in-place).
            rwd_dict: Latest reward dict from :meth:`get_reward_dict`.
            step_count: Current step index within the episode.

        Returns:
            Updated metrics dict.
        """
        metrics.update(
            tracking_reward=rwd_dict["tracking"],
            solved_frac=rwd_dict["solved"] / self._config.max_episode_steps,
        )
        return metrics


__all__ = ["MjxMuscleMimicBase"]
