# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MJX walk environment — locomotion task for the musculoskeletal leg model.

Ports ``WalkEnvV0`` (``myobase/walk_v0.py``) to the MJX (JAX) backend so that
the same myoLegWalk task can run on:
    - CPU (Gymnasium): ``myoLegWalk-v0`` using ``WalkEnvV0``
    - MJX XLA (JAX CPU/GPU/TPU): ``MjxLegWalk-v0`` using this class (``impl=None``)
    - MJX Warp (GPU): ``MjxLegWalk-v0`` using this class with ``impl="warp"``

The three backends share the same observation space, reward function, and
action space so that training results can be directly compared.

Config keys (in addition to ``MyoMjxEnvBase`` defaults):
    ``model_path``     Path to ``myolegs.xml`` (leg musculoskeletal model).
    ``sim_dt``         MuJoCo simulation timestep (default: 0.002 s).
    ``ctrl_dt``        Control timestep = ``sim_dt × n_substeps`` (default: 0.02 s).
    ``num_envs``       Parallel env count (used by brax PPO; 1 for SB3 SAC mode).
    ``min_height``     Minimum COM height before episode termination (default: 0.8 m).
    ``max_rot``        Maximum (R @ e_x)[0] magnitude before termination (default: 0.8).
    ``hip_period``     Desired cyclic hip period in steps (default: 100).
    ``target_x_vel``   Target x-axis COM velocity m/s (default: 0.0).
    ``target_y_vel``   Target y-axis COM velocity m/s (default: 1.2).
    ``max_episode_steps`` Episode truncation length (default: 1000).
    ``mjx_impl``       ``None`` for XLA backend, ``"warp"`` for GPU Warp backend.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from typing import Any
from mujoco import mjx

from myosuite.envs.myo.backends.mjx.mjx_env_base import MjxEnvAccessor, MyoMjxEnvBase
from myosuite.envs.myo.backends.mjx.mjx_spec_preprocess import preprocess_mjx_spec
from myosuite.physics.quat_math_jax import quat2mat


class MjxWalkEnv(MyoMjxEnvBase):
    """Locomotion task for the musculoskeletal leg model on the MJX backend.

    Mirrors the observation space and reward function of ``WalkEnvV0`` exactly
    so that CPU, MJX XLA, and MJX Warp backends can be compared numerically.

    Observation vector components (concatenated in order):
        qpos_without_xy  (nq - 2 = 33)   joint positions excluding x, y
        qvel             (nv = 34)         scaled by ctrl_dt
        com_vel          (2)               x, y COM velocity
        torso_angle      (4)               torso quaternion (xquat)
        feet_heights     (2)               talus_l, talus_r z-position
        height           (1)               mass-weighted COM z-height
        feet_rel_positions (6)             feet positions relative to pelvis
        phase_var        (1)               cyclic phase in [0, 1)
        muscle_length    (nu = 80)         actuator_length
        muscle_velocity  (nu = 80)         clipped actuator_velocity
        muscle_force     (nu = 80)         clipped / scaled actuator_force
        act              (na = 80)         muscle activations

    Total observation size: 403 dimensions.

    Reward components (same weights as ``WalkEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS``):
        vel_reward      × 5.0   — forward velocity match
        done            × -100  — termination penalty
        cyclic_hip      × -10   — hip flexion periodicity
        ref_rot         × 10.0  — torso rotation tracking
        joint_angle_rew × 5.0   — hip adduction/rotation penalty
    """

    # ------------------------------------------------------------------
    # Template method implementations
    # ------------------------------------------------------------------

    def setup_model(self, config: Any) -> None:
        """Load the myolegs musculoskeletal model and precompute body/joint IDs.

        Args:
            config: Frozen environment config dict.
        """
        impl = getattr(config, "mjx_impl", None) or None
        spec = mujoco.MjSpec.from_file(config.model_path.as_posix())
        spec = preprocess_mjx_spec(spec, impl=impl)
        self._mj_spec = spec
        self._mj_model = spec.compile()
        self._mj_model.opt.timestep = config.sim_dt
        self._mj_model.opt.iterations = 6
        self._mj_model.opt.ls_iterations = 6
        self._mj_model.opt.ccd_iterations = 75

        if impl == "warp":
            self._mjx_model = mjx.put_model(self._mj_model, impl=impl)
        else:
            self._mjx_model = mjx.put_model(self._mj_model)
        self._xml_path = config.model_path.as_posix()
        self._n_substeps = int(round(config.ctrl_dt / config.sim_dt))

        # Body IDs (host-side, used as integer indices into JAX arrays)
        self._pelvis_id: int = self._mj_model.body("pelvis").id
        self._torso_id: int = self._mj_model.body("torso").id
        self._talus_l_id: int = self._mj_model.body("talus_l").id
        self._talus_r_id: int = self._mj_model.body("talus_r").id

        # Joint qpos address table for reward computation
        def _qadr(name: str) -> int:
            jnt_id = self._mj_model.joint(name).id
            return int(self._mj_model.jnt_qposadr[jnt_id])

        self._hip_flex_l_adr: int = _qadr("hip_flexion_l")
        self._hip_flex_r_adr: int = _qadr("hip_flexion_r")
        self._hip_adduct_l_adr: int = _qadr("hip_adduction_l")
        self._hip_adduct_r_adr: int = _qadr("hip_adduction_r")
        self._hip_rot_l_adr: int = _qadr("hip_rotation_l")
        self._hip_rot_r_adr: int = _qadr("hip_rotation_r")

        # Body mass array (constant — precompute as JAX array)
        self._body_mass_jax: jax.Array = jp.array(self._mj_model.body_mass)

        # Store initial standing qpos (keyframe 2 = "init" pose used by WalkEnvV0)
        nkey = self._mj_model.nkey
        nq = self._mj_model.nq
        if nkey >= 3:
            self._init_qpos_np: np.ndarray = self._mj_model.key_qpos.reshape(nkey, nq)[
                2
            ].copy()
        else:
            self._init_qpos_np = self._mj_model.qpos0.copy()

    def sample_task(self, rng: jax.Array) -> dict[str, jax.Array]:
        """Return episode-invariant task config.

        For plain walking, the only episode-varying parameter is the target
        orientation (set once from the initial qpos so the agent learns to
        stay upright rather than track a rotating reference).

        Args:
            rng: JAX random key (unused — walk task is deterministic per run).

        Returns:
            Dict with ``"target_rot"`` — shape (4,) target torso quaternion.
        """
        target_rot = jp.array(self._init_qpos_np[3:7], dtype=jp.float32)
        return {"target_rot": target_rot}

    def initial_qpos_qvel(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the canonical standing qpos (keyframe 2) with zero velocity.

        Matches ``WalkEnvV0.reset()`` with ``reset_type="init"``.

        Args:
            rng: JAX random key (unused — deterministic init).

        Returns:
            Tuple ``(qpos, qvel)`` as float32 JAX arrays.
        """
        qpos = jp.array(self._init_qpos_np, dtype=jp.float32)
        qvel = jp.zeros(self._mjx_model.nv, dtype=jp.float32)
        return qpos, qvel

    def _normalize_action(self, action: jax.Array) -> jax.Array:
        """Clip muscle activations to [0, 1] — matches the CPU env ctrlrange clip.

        ``WalkEnvV0`` (CPU) clips actions to ``ctrlrange [0, 1]`` via MuJoCo.
        We replicate this with a simple clip so that SAR-generated activations
        in [0, 1] are passed through unchanged (instead of the default sigmoid
        which would squash them to [0.37, 0.63]).

        Args:
            action: Raw muscle activation vector in [0, 1].

        Returns:
            Clipped vector in [0, 1].
        """
        return jp.clip(action, 0.0, 1.0)

    def get_obs_dict(
        self,
        accessor: MjxEnvAccessor,
        task_state: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Compute the full WalkEnvV0-equivalent observation dict in JAX.

        All observations match the CPU env field-for-field so that numerical
        differences are attributable to physics precision (float32 vs float64)
        rather than observation mis-implementation.

        Args:
            accessor: Read-only MJX physics state accessor.
            task_state: Episode task config (contains ``"target_rot"``).

        Returns:
            Ordered dict of named JAX arrays (concatenated to obs vector by
            base class).
        """
        data = accessor.data
        ctrl_dt = accessor.dt()

        # --- joint positions (exclude x, y root translation) ---
        qpos_without_xy = data.qpos[2:]

        # --- joint velocities scaled by ctrl_dt ---
        qvel = data.qvel * ctrl_dt

        # --- COM velocity (x, y): mass-weighted mean of body cvel[3:5] ---
        # MuJoCo cvel convention: negated translational velocity in world frame.
        mass = self._body_mass_jax  # (nbody,)
        com_vel = jp.sum(mass[:, None] * (-data.cvel[:, 3:5]), axis=0) / jp.sum(
            mass
        )  # shape (2,)

        # --- torso quaternion ---
        torso_angle = data.xquat[self._torso_id]  # (4,)

        # --- feet z-heights ---
        feet_heights = jp.array(
            [data.xpos[self._talus_l_id, 2], data.xpos[self._talus_r_id, 2]]
        )  # (2,)

        # --- mass-weighted COM z-height using body CoM positions (xipos) ---
        height = jp.atleast_1d(jp.sum(mass * data.xipos[:, 2]) / jp.sum(mass))  # (1,)

        # --- feet positions relative to pelvis ---
        pelvis_pos = data.xpos[self._pelvis_id]  # (3,)
        feet_rel_positions = jp.concatenate(
            [
                data.xpos[self._talus_l_id] - pelvis_pos,
                data.xpos[self._talus_r_id] - pelvis_pos,
            ]
        )  # (6,)

        # --- cyclic phase variable: (t / (hip_period × ctrl_dt)) mod 1 ---
        hip_period_time = float(self._config.hip_period) * ctrl_dt
        phase_var = jp.atleast_1d((data.time / hip_period_time) % 1.0)  # (1,)

        # --- muscle state: length, velocity (clipped), force (clipped/scaled) ---
        muscle_length = data.actuator_length  # (nu,)
        muscle_velocity = jp.clip(data.actuator_velocity, -100.0, 100.0)  # (nu,)
        muscle_force = jp.clip(data.actuator_force / 1000.0, -100.0, 100.0)  # (nu,)

        # --- muscle activations ---
        act = data.act  # (na,)

        return {
            "qpos_without_xy": qpos_without_xy,
            "qvel": qvel,
            "com_vel": com_vel,
            "torso_angle": torso_angle,
            "feet_heights": feet_heights,
            "height": height,
            "feet_rel_positions": feet_rel_positions,
            "phase_var": phase_var,
            "muscle_length": muscle_length,
            "muscle_velocity": muscle_velocity,
            "muscle_force": muscle_force,
            "act": act,
        }

    def get_reward_dict(
        self,
        accessor: MjxEnvAccessor,
        task_state: dict[str, jax.Array],
        obs_dict: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Compute WalkEnvV0-equivalent reward components in JAX.

        Weights match ``WalkEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS``:
            vel_reward × 5, done × -100, cyclic_hip × -10,
            ref_rot × 10, joint_angle_rew × 5.

        Args:
            accessor: Read-only MJX physics state accessor.
            task_state: Must contain ``"target_rot"`` (shape (4,)).
            obs_dict: Output of ``get_obs_dict``.

        Returns:
            Dict with ``"dense"``, ``"done"``, and all individual components.
        """
        data = accessor.data
        cfg = self._config

        com_vel = obs_dict["com_vel"]  # (2,)
        height = obs_dict["height"][0]  # scalar
        phase_var = obs_dict["phase_var"][0]  # scalar

        target_y_vel = float(cfg.target_y_vel)
        target_x_vel = float(cfg.target_x_vel)
        min_height = float(cfg.min_height)
        max_rot = float(cfg.max_rot)

        # ----- vel_reward -----
        vel_reward = jp.exp(-jp.square(target_y_vel - com_vel[1])) + jp.exp(
            -jp.square(target_x_vel - com_vel[0])
        )

        # ----- done: height or rotation exceeded -----
        done_height = (height < min_height).astype(jp.float32)

        # rotation condition: abs((quat2mat(qpos[3:7]) @ e_x)[0]) > max_rot
        quat = data.qpos[3:7]
        rot_mat = quat2mat(quat)  # (3,3)
        rot_x_component = rot_mat[0, 0]  # = (R @ [1,0,0])[0]
        done_rot = (jp.abs(rot_x_component) > max_rot).astype(jp.float32)

        done = jp.maximum(done_height, done_rot)  # 1.0 if either condition triggered

        # ----- cyclic_hip -----
        des_hip = jp.array(
            [
                0.8 * jp.cos(phase_var * 2.0 * jp.pi + jp.pi),
                0.8 * jp.cos(phase_var * 2.0 * jp.pi),
            ]
        )
        actual_hip = jp.array(
            [data.qpos[self._hip_flex_l_adr], data.qpos[self._hip_flex_r_adr]]
        )
        cyclic_hip = jp.linalg.norm(des_hip - actual_hip)

        # ----- ref_rot -----
        target_rot = task_state["target_rot"]  # (4,)
        ref_rot = jp.exp(-jp.linalg.norm(5.0 * (data.qpos[3:7] - target_rot)))

        # ----- joint_angle_rew -----
        hip_angles = jp.array(
            [
                data.qpos[self._hip_adduct_l_adr],
                data.qpos[self._hip_adduct_r_adr],
                data.qpos[self._hip_rot_l_adr],
                data.qpos[self._hip_rot_r_adr],
            ]
        )
        joint_angle_rew = jp.exp(-5.0 * jp.mean(jp.abs(hip_angles)))

        # ----- dense reward (matching WalkEnvV0 weighted sum) -----
        dense = (
            5.0 * vel_reward
            + (-100.0) * done
            + (-10.0) * cyclic_hip
            + 10.0 * ref_rot
            + 5.0 * joint_angle_rew
        )

        return {
            "vel_reward": vel_reward,
            "cyclic_hip": cyclic_hip,
            "ref_rot": ref_rot,
            "joint_angle_rew": joint_angle_rew,
            "done": done,
            "sparse": vel_reward,
            "solved": (vel_reward >= 1.0).astype(jp.float32),
            "dense": dense,
        }

    def initial_metrics(self) -> dict[str, jax.Array]:
        """Return zero-valued metrics matching reward component names.

        Returns:
            Dict with all reward component keys initialized to zero.
        """
        zero = jp.array(0.0)
        return {
            "vel_reward": zero,
            "cyclic_hip": zero,
            "ref_rot": zero,
            "joint_angle_rew": zero,
            "solved_frac": zero,
            "com_vel_y": zero,
            "height": zero,
        }

    def _update_metrics(
        self,
        metrics: dict[str, jax.Array],
        rwd_dict: dict[str, jax.Array],
        step_count: jax.Array,
    ) -> dict[str, jax.Array]:
        """Update running metrics with current-step reward components.

        Args:
            metrics: Mutable metrics dict from ``state.metrics``.
            rwd_dict: Output of ``get_reward_dict``.
            step_count: Current step index (post-increment).

        Returns:
            Updated metrics dict.
        """
        metrics.update(
            vel_reward=rwd_dict["vel_reward"],
            cyclic_hip=rwd_dict["cyclic_hip"],
            ref_rot=rwd_dict["ref_rot"],
            joint_angle_rew=rwd_dict["joint_angle_rew"],
            solved_frac=rwd_dict["solved"],
        )
        return metrics
