# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MJX reach environment — fingertip reaching task.

Replaces ``playground_reach_v0.py``.  Delegates obs and reward computation
to the shared term functions in ``myosuite/terms/``, accessed via
``MjxEnvAccessor``.

Config keys (in addition to ``MyoMjxEnvBase`` defaults):
    ``model_path``         Path to the MJCF file.
    ``sim_dt``             MuJoCo simulation timestep.
    ``ctrl_dt``            Control timestep (= ``sim_dt × n_substeps``).
    ``num_envs``           Number of parallel environments (batch size).
    ``max_episode_steps``  Episode truncation length.
    ``target_reach_range`` Mapping of site name → ``(lo, hi)``
                           JAX arrays of shape ``(3,)``.
    ``far_th``             Distance threshold (metres) triggering done.
    ``reward_weights``     Sub-dict with ``reach``, ``bonus``, ``penalty``.
"""

from __future__ import annotations

import os
import pathlib
import tempfile

import jax
import jax.numpy as jp
import mujoco
from typing import Any
from mujoco import mjx

from myosuite.envs.myo.backends.mjx.mjx_env_base import MjxEnvAccessor, MyoMjxEnvBase
from myosuite.envs.myo.backends.mjx.mjx_spec_preprocess import preprocess_mjx_spec
from myosuite.terms.base_obs import tip_pos_obs


class MjxReachEnv(MyoMjxEnvBase):
    """Fingertip reaching task for a musculoskeletal MJX model.

    The agent must move one or more fingertip sites to randomly sampled
    3-D target positions.

    Args:
        config: Environment configuration dict (see module docstring).
        config_overrides: Optional key-value overrides applied on top of
            *config*.
    """

    _preprocess_spec = staticmethod(preprocess_mjx_spec)

    # ------------------------------------------------------------------
    # Template method implementations
    # ------------------------------------------------------------------

    def setup_model(self, config: Any) -> None:
        """Compile MJ/MJX model, stripping the scene include for speed.

        Mirrors the ``playground_reach_v0.py`` pattern: comments out any
        ``myosuite_scene.xml`` include before loading so the heavy scene
        geometry is not part of the simulation.

        Args:
            config: Frozen environment config dict.
        """
        if str(config.model_path) == "/tmp/dummy.xml":
            raise ValueError(
                "config.model_path is set to the placeholder '/tmp/dummy.xml'. "
                "Override model_path with a valid MJCF file path before instantiating "
                "this environment (e.g. via make() or a config override)."
            )
        orig_path = pathlib.Path(config.model_path)
        # Strip scene include via a temp file (matches original implementation)
        with tempfile.NamedTemporaryFile(
            "w", dir=orig_path.parent, suffix=".xml", delete=False
        ) as tmp_file:
            for line in orig_path.open():
                if "myosuite_scene.xml" in line:
                    tmp_file.write(f"<!-- {line.strip()} -->\n")
                else:
                    tmp_file.write(line)
            tmp_path = tmp_file.name

        impl = getattr(config, "mjx_impl", None) or None

        try:
            spec = mujoco.MjSpec.from_file(tmp_path)
        finally:
            os.remove(tmp_path)

        spec = self._preprocess_spec(spec, impl=impl)
        self._mj_spec = spec
        self._mj_model = spec.compile()

        self._mj_model.opt.timestep = config.sim_dt
        self._mj_model.opt.iterations = 6
        self._mj_model.opt.ls_iterations = 6
        self._mj_model.opt.ccd_iterations = 75

        self._mjx_model = mjx.put_model(self._mj_model, impl=impl)
        self._xml_path = config.model_path.as_posix()
        self._n_substeps = int(config.ctrl_dt / config.sim_dt)

        # Pre-compute site ID arrays (host-side, used for obs + reward)
        tip_sids = []
        target_sids = []
        for site_name in config.target_reach_range.keys():
            tip_sids.append(
                mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site_name
                )
            )
            target_sids.append(
                mujoco.mj_name2id(
                    self._mj_model,
                    mujoco.mjtObj.mjOBJ_SITE.value,
                    site_name + "_target",
                )
            )
        self._tip_sids = jp.array(tip_sids)
        self._target_sids = jp.array(target_sids)
        self._n_targets = len(tip_sids)
        self._near_th = float(self._n_targets) * 0.0125

    def sample_task(self, rng: jax.Array) -> dict[str, jax.Array]:
        """Sample random 3-D target positions for all tracked sites.

        Args:
            rng: JAX random key.

        Returns:
            Dict with ``"targets"`` — shape ``(n_sites, 3)`` JAX array of
            target Cartesian positions.
        """
        targets = []
        for span in self._config.target_reach_range.values():
            targets.append(
                jax.random.uniform(rng, (span[0].size,), minval=span[0], maxval=span[1])
            )
        return {"targets": jp.stack(targets)}

    def get_obs_dict(
        self,
        accessor: MjxEnvAccessor,
        task_state: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Compute ``qpos``, ``qvel``, ``act``, ``tip_pos``, and ``reach_err``.

        Args:
            accessor: MJX physics state accessor.
            task_state: Must contain ``"targets"`` (shape ``(n_sites, 3)``).

        Returns:
            Ordered obs dict.
        """
        tip_pos = tip_pos_obs(accessor, self._tip_sids)
        reach_err = (task_state["targets"] - tip_pos).ravel()
        return {
            "qpos": accessor.joint_pos(),
            "qvel": accessor.joint_vel(),
            "act": accessor.muscle_act(),
            "tip_pos": tip_pos.ravel(),
            "reach_err": reach_err,
        }

    def get_reward_dict(
        self,
        accessor: MjxEnvAccessor,
        task_state: dict[str, jax.Array],
        obs_dict: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Compute reach reward with distance penalty, bonus, and far-threshold.

        Args:
            accessor: MJX physics state accessor.
            task_state: Must contain ``"targets"``.
            obs_dict: Output of ``get_obs_dict`` (provides ``reach_err``).

        Returns:
            Dict with ``"dense"``, ``"done"``, and individual components.
        """
        rw = self._config.reward_weights
        reach_err = obs_dict["reach_err"]
        reach_dist = jp.linalg.norm(reach_err, axis=-1)

        far_th = jp.where(
            accessor.data.time > 2.0 * self._mjx_model.opt.timestep,
            float(self._config.far_th) * self._n_targets,
            jp.inf,
        )

        reach = -1.0 * reach_dist * float(rw.reach)
        bonus = (
            1.0 * (reach_dist < 2 * self._near_th) + 1.0 * (reach_dist < self._near_th)
        ) * float(rw.bonus)
        penalty = -1.0 * (reach_dist > far_th) * float(rw.penalty)

        dense = reach + bonus + penalty
        done = jp.array(reach_dist > far_th, dtype=jp.float32)
        solved = jp.array(reach_dist < self._near_th, dtype=jp.float32)

        return {
            "reach": reach,
            "bonus": bonus,
            "penalty": penalty,
            "dense": dense,
            "done": done,
            "solved": solved,
        }

    def initial_metrics(self) -> dict[str, jax.Array]:
        """Return zero metrics matching the original playground env.

        Returns:
            Dict with ``reach_reward``, ``bonus_reward``, ``penalty_reward``,
            ``solved_frac`` — all zero.
        """
        zero = jp.array(0.0)
        return {
            "reach_reward": zero,
            "bonus_reward": zero,
            "penalty_reward": zero,
            "solved_frac": zero,
        }

    def _update_metrics(
        self,
        metrics: dict[str, jax.Array],
        rwd_dict: dict[str, jax.Array],
        step_count: jax.Array,
    ) -> dict[str, jax.Array]:
        """Write reward components into the metrics dict.

        Args:
            metrics: Mutable metrics dict from ``state.metrics``.
            rwd_dict: Output of ``get_reward_dict``.
            step_count: Current step index.

        Returns:
            Updated metrics dict.
        """
        metrics.update(
            reach_reward=rwd_dict["reach"],
            bonus_reward=rwd_dict["bonus"],
            penalty_reward=rwd_dict["penalty"],
            solved_frac=rwd_dict["solved"] / self._config.max_episode_steps,
        )
        return metrics
