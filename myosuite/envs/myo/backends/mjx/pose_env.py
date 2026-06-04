# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MJX pose environment — joint-angle posing task.

Replaces ``playground_pose_v0.py``.  Delegates obs and reward computation
to the shared term functions in ``myosuite/terms/``, accessed via
``MjxEnvAccessor``.

Config keys (in addition to ``MyoMjxEnvBase`` defaults):
    ``model_path``         Path to the MJCF file.
    ``sim_dt``             MuJoCo simulation timestep.
    ``ctrl_dt``            Control timestep (= ``sim_dt × n_substeps``).
    ``num_envs``           Number of parallel environments (batch size).
    ``max_episode_steps``  Episode truncation length.
    ``target_jnt_range``   Mapping of joint name → ``(lo, hi)``
                           JAX arrays of shape ``(n_joints,)``.
    ``reward_config``      Sub-dict with:
                           ``angle_reward_weight``, ``ctrl_cost_weight``,
                           ``bonus_weight``, ``pose_thd``, ``far_th``.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco
from typing import Any
from mujoco import mjx

from myosuite.envs.myo.backends.mjx.mjx_env_base import MjxEnvAccessor, MyoMjxEnvBase
from myosuite.envs.myo.backends.mjx.mjx_spec_preprocess import preprocess_mjx_spec
from myosuite.terms.base_obs import pose_error_obs
from myosuite.terms.base_reward import pose_reward


class MjxPoseEnv(MyoMjxEnvBase):
    """Joint-angle posing task for a musculoskeletal MJX model.

    The agent must move all tracked joints to a randomly sampled target
    configuration.

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
        """Compile MJ/MJX model from *config.model_path*.

        Applies contact-margin zeroing and configures solver iterations
        matching the original ``playground_pose_v0.py`` defaults.

        Args:
            config: Frozen environment config dict.
        """
        if str(config.model_path) == "/tmp/dummy.xml":
            raise ValueError(
                "config.model_path is set to the placeholder '/tmp/dummy.xml'. "
                "Override model_path with a valid MJCF file path before instantiating "
                "this environment (e.g. via make() or a config override)."
            )
        impl = getattr(config, "mjx_impl", None) or None
        spec = mujoco.MjSpec.from_file(config.model_path.as_posix())
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

    def sample_task(self, rng: jax.Array) -> dict[str, jax.Array]:
        """Sample a random target joint configuration.

        Args:
            rng: JAX random key.

        Returns:
            Dict with ``"target_angles"`` — 1-D JAX array of target angles,
            concatenated from all joints in ``config.target_jnt_range``.
        """
        targets = []
        for span in self._config.target_jnt_range.values():
            targets.append(
                jax.random.uniform(rng, (span[0].size,), minval=span[0], maxval=span[1])
            )
        return {"target_angles": jp.hstack(targets)}

    def get_obs_dict(
        self,
        accessor: MjxEnvAccessor,
        task_state: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Compute ``qpos``, ``qvel``, ``act``, and ``pose_err``.

        Args:
            accessor: MJX physics state accessor.
            task_state: Must contain ``"target_angles"``.

        Returns:
            Ordered obs dict.
        """
        return {
            "qpos": accessor.joint_pos(),
            "qvel": accessor.joint_vel(),
            "act": accessor.muscle_act(),
            "pose_err": pose_error_obs(accessor, task_state["target_angles"]),
        }

    def get_reward_dict(
        self,
        accessor: MjxEnvAccessor,
        task_state: dict[str, jax.Array],
        obs_dict: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Compute pose reward components with config-specified weights.

        Calls ``pose_reward()`` from ``myosuite/terms/`` and applies the
        per-component weights from ``config.reward_config``.

        Args:
            accessor: MJX physics state accessor.
            task_state: Must contain ``"target_angles"``.
            obs_dict: Output of ``get_obs_dict`` (unused directly; kept for
                uniform signature).

        Returns:
            Dict with ``"dense"``, ``"done"``, and individual components.
        """
        rc = self._config.reward_config
        rwd = pose_reward(accessor, task_state, pose_thd=float(rc.pose_thd))

        act_mag = jp.linalg.norm(accessor.muscle_act(), axis=-1)
        act_reg = act_mag * -float(rc.ctrl_cost_weight)

        dense = (
            rwd["pose"] * float(rc.angle_reward_weight)
            + rwd["bonus"] * float(rc.bonus_weight)
            + rwd["penalty"]
            + act_reg
        )
        return {**rwd, "act_reg": act_reg, "dense": dense}

    def initial_metrics(self) -> dict[str, jax.Array]:
        """Return zero metrics matching the original playground env.

        Returns:
            Dict with ``pose_reward``, ``act_reg_reward``, ``bonus_reward``,
            ``penalty_reward``, ``solved_frac`` — all zero.
        """
        zero = jp.array(0.0)
        return {
            "pose_reward": zero,
            "act_reg_reward": zero,
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
            pose_reward=rwd_dict["pose"],
            act_reg_reward=rwd_dict.get("act_reg", jp.array(0.0)),
            bonus_reward=rwd_dict["bonus"],
            penalty_reward=rwd_dict["penalty"],
            solved_frac=jp.array(rwd_dict["solved"], dtype=jp.float32)
            / self._config.max_episode_steps,
        )
        return metrics
