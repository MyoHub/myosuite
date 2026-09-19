# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import collections
from typing import Any

import mujoco
import numpy as np

import gymnasium as gym
from gymnasium.utils import EzPickle

from myosuite.core.model_builder import ModelBuilder
from myosuite.core.muscle_conditions import apply_sarcopenia_to_model
from myosuite.envs.gymnasium_env import CpuEnvAccessor, MyoGymnasiumEnv
from myosuite.envs.myo.assets._resolve import warn_torso_pip_calibration_divergence
from myosuite.physics.fatigue import CumulativeFatigue
from myosuite.terms.base_action import sigmoid_muscle_activation


class TorsoEnvV0(MyoGymnasiumEnv, EzPickle):
    """Pose-tracking task for the musculoskeletal torso model.

    The environment presents a target joint configuration for the first 18
    joints and rewards the agent for minimising the angular distance.

    Args:
        model_path: Absolute path to the MuJoCo XML model.
        obsd_model_path: Unused (kept for API compatibility).
        seed: Random seed.
        viz_site_targets: Site names for target visualisation.
        target_jnt_range: Dict ``{joint_name: (min, max)}`` for random targets.
        target_jnt_value: Fixed target joint vector.
        reset_type: One of ``"none"``, ``"init"``, or ``"random"``.
        obs_keys: Observation keys.
        weighted_reward_keys: Dict ``{reward_key: weight}`` for dense reward.
        pose_thd: Threshold (rad) for task success.
        normalize_act: If ``True``, action space is ``[-1, 1]``.
        frame_skip: Number of MuJoCo substeps per :meth:`step` call.
        muscle_condition: One of ``""``, ``"sarcopenia"``, ``"fatigue"``.
        fatigue_reset_vec: Initial fatigue state vector.
        fatigue_reset_random: If ``True``, randomise fatigue state on reset.
    """

    DEFAULT_OBS_KEYS = ["qpos", "qvel", "pose_err"]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pose": 1.0,
        "bonus": 4.0,
        "act_reg": 1.0,
        "penalty": 50,
        "done": 0,
    }

    MYO_CREDIT = """\
    MyoSuite: A contact-rich simulation suite for musculoskeletal motor control
        Vittorio Caggiano, Huawei Wang, Guillaume Durandau, Massimo Sartori, Vikash Kumar
        L4DC-2019 | https://sites.google.com/view/myosuite
    """

    def __init__(
        self,
        model_path: str,
        obsd_model_path: str | None = None,
        seed: int | None = None,
        viz_site_targets: tuple | None = None,
        target_jnt_range: dict | None = None,
        target_jnt_value: list | None = None,
        reset_type: str = "init",
        obs_keys: list = DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict[str, float] = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        pose_thd: float = 0.25,
        normalize_act: bool = True,
        frame_skip: int = 10,
        muscle_condition: str = "",
        fatigue_reset_vec=None,
        fatigue_reset_random: bool = False,
        **kwargs: Any,
    ) -> None:
        MyoGymnasiumEnv.__init__(
            self, frame_skip=frame_skip, render_mode=kwargs.get("render_mode")
        )
        # EzPickle must be called AFTER MyoGymnasiumEnv.__init__ to avoid
        # the cooperative super() chain resetting _ezpickle_args to ().
        EzPickle.__init__(
            self,
            model_path,
            obsd_model_path,
            seed,
            viz_site_targets=viz_site_targets,
            target_jnt_range=target_jnt_range,
            target_jnt_value=target_jnt_value,
            reset_type=reset_type,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            pose_thd=pose_thd,
            normalize_act=normalize_act,
            frame_skip=frame_skip,
            muscle_condition=muscle_condition,
            fatigue_reset_vec=fatigue_reset_vec,
            fatigue_reset_random=fatigue_reset_random,
            **kwargs,
        )

        # ── Load model ─────────────────────────────────────────────────────
        warn_torso_pip_calibration_divergence()
        self.model, self._mj_spec = ModelBuilder.from_xml_file(model_path).build()
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * frame_skip)

        # ── Muscle condition ───────────────────────────────────────────────
        self.muscle_condition = muscle_condition
        self.fatigue_reset_vec = fatigue_reset_vec
        self.fatigue_reset_random = fatigue_reset_random
        self._muscle_act_ind = self.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self._init_muscle_condition()

        # ── Target configuration ───────────────────────────────────────────
        self.reset_type = reset_type
        self.pose_thd = pose_thd

        if target_jnt_range is not None:
            jnt_ids, jnt_ranges = [], []
            for jnt_name, jnt_range in target_jnt_range.items():
                jnt_ids.append(self.model.joint(jnt_name).id)
                jnt_ranges.append(jnt_range)
            self._target_jnt_ids: list[int] | None = jnt_ids
            self._target_jnt_range: np.ndarray | None = np.array(jnt_ranges)
            self.target_jnt_value = np.mean(self._target_jnt_range, axis=1)
        else:
            self._target_jnt_ids = None
            self._target_jnt_range = None
            self.target_jnt_value = (
                np.array(target_jnt_value) if target_jnt_value is not None else None
            )

        # Number of torso joints tracked in pose_err (first 18)
        self._n_pose_jnts = 18

        # ── Reward / obs config ────────────────────────────────────────────
        self.rwd_keys_wt = weighted_reward_keys
        self.obs_keys = list(obs_keys)
        if self.model.na > 0 and "act" not in self.obs_keys:
            self.obs_keys.append("act")

        # ── Action space ───────────────────────────────────────────────────
        self.normalize_act = normalize_act
        if normalize_act:
            act_low = -np.ones(self.model.nu, dtype=np.float32)
            act_high = np.ones(self.model.nu, dtype=np.float32)
        else:
            act_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
            act_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)

        # ── Initial state ──────────────────────────────────────────────────
        mujoco.mj_forward(self.model, self.data)
        init_qpos = self.data.qpos.ravel().copy()
        if normalize_act:
            actuated_jnt_ids = self.model.actuator_trnid[
                self.model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT, 0
            ]
            linear_jnt_mask = np.logical_or(
                self.model.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE,
                self.model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE,
            )
            linear_jnt_ids = np.where(linear_jnt_mask)[0]
            linear_act_jnt_ids = np.intersect1d(actuated_jnt_ids, linear_jnt_ids)
            qpos_ids = self.model.jnt_qposadr[linear_act_jnt_ids]
            init_qpos[qpos_ids] = np.mean(
                self.model.jnt_range[linear_act_jnt_ids], axis=1
            )
        self._init_qpos = init_qpos
        self._init_qvel = self.data.qvel.ravel().copy()

        import gymnasium

        self._input_seed = seed
        gymnasium.Env.reset(self, seed=seed)

        # ── Observation space ──────────────────────────────────────────────
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs = self._obs_dict_to_vec(self.get_obs_dict(self._accessor))
        self.observation_space = gym.spaces.Box(
            -10.0 * np.ones(obs.size, dtype=np.float32),
            10.0 * np.ones(obs.size, dtype=np.float32),
            dtype=np.float32,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _init_muscle_condition(self) -> None:
        """Apply the muscle condition to the compiled model."""
        if self.muscle_condition == "sarcopenia":
            apply_sarcopenia_to_model(self.model, force_scale=0.5)
        elif self.muscle_condition == "fatigue":
            self.muscle_fatigue = CumulativeFatigue(
                self.model, self.frame_skip, seed=None
            )

    def _apply_action(self, action: np.ndarray) -> None:
        """Map action to MuJoCo ctrl and write to data.ctrl.

        Args:
            action: Action vector in the action-space (already clipped).
        """
        ctrl = action.copy()

        if self.model.na > 0 and self.normalize_act:
            ctrl[self._muscle_act_ind] = sigmoid_muscle_activation(
                ctrl[self._muscle_act_ind], np
            )
        elif self.normalize_act:
            ctrl_range = self.model.actuator_ctrlrange
            ctrl = (
                np.mean(ctrl_range, axis=-1)
                + ctrl * (ctrl_range[:, 1] - ctrl_range[:, 0]) / 2.0
            )

        if self.muscle_condition == "fatigue":
            ctrl[self._muscle_act_ind], _, _ = self.muscle_fatigue.compute_act(
                ctrl[self._muscle_act_ind]
            )

        self.data.ctrl[:] = ctrl

    # ── MyoGymnasiumEnv interface ──────────────────────────────────────────

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        """Compute the observation dictionary.

        Args:
            accessor: CPU environment accessor.

        Returns:
            Filtered dict with obs_keys as keys.
        """
        qpos = accessor.joint_pos()
        obs: dict[str, np.ndarray] = {
            "time": np.array([accessor.time()]),
            "qpos": qpos,
            "qvel": accessor.joint_vel() * accessor.dt(),
            "act": accessor.muscle_act(),
            "pose_err": self.target_jnt_value - qpos[: self._n_pose_jnts],
        }
        return {k: obs[k] for k in self.obs_keys if k in obs}

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        """Compute the reward dictionary.

        Args:
            obs_dict: Output of :meth:`get_obs_dict`.

        Returns:
            Ordered dict with reward components.
        """
        pose_dist = float(
            np.linalg.norm(obs_dict.get("pose_err", np.zeros(1)), axis=-1)
        )
        act_mag = float(np.linalg.norm(obs_dict.get("act", np.zeros(1)), axis=-1))
        if self.model.na != 0:
            act_mag = act_mag / self.model.na
        far_th = np.pi

        rwd_dict = collections.OrderedDict(
            (
                ("pose", -1.0 * pose_dist),
                (
                    "bonus",
                    1.0 * (pose_dist < self.pose_thd)
                    + 1.0 * (pose_dist < 1.5 * self.pose_thd),
                ),
                ("penalty", -1.0 * (pose_dist > far_th)),
                ("act_reg", -1.0 * act_mag),
                ("sparse", -1.0 * pose_dist),
                ("solved", pose_dist < self.pose_thd),
                ("done", pose_dist > far_th),
            )
        )
        rwd_dict["dense"] = float(
            np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()])
        )
        return rwd_dict

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        """Target is fixed (set at init); no per-episode sampling needed.

        Args:
            np_random: Unused.

        Returns:
            Empty task state dict.
        """
        return {}

    # ── Gymnasium step / reset ─────────────────────────────────────────────

    def step(self, action: np.ndarray, **kwargs: Any):
        """Advance the simulation by one control step.

        Args:
            action: Control command in the action space.
            **kwargs: Ignored compatibility kwargs (e.g. update_exteroception).

        Returns:
            Tuple ``(obs, reward, terminated, truncated, info)``.
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data, self.frame_skip)
        mujoco.mj_kinematics(self.model, self.data)

        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs_dict = self.get_obs_dict(self._accessor)
        rwd_dict = self.get_reward_dict(obs_dict)

        obs = self._obs_dict_to_vec(obs_dict)
        reward = float(rwd_dict.get("dense", 0.0))
        terminated = bool(rwd_dict.get("done", False))
        info = {k: v for k, v in rwd_dict.items() if k not in ("dense", "done")}
        info["obs_dict"] = obs_dict
        info["rwd_dict"] = rwd_dict
        return obs, reward, terminated, False, info

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
        **kwargs: Any,
    ):
        """Reset the environment to a new episode.

        Args:
            seed: Random seed for this episode.
            options: Unused.

        Returns:
            Tuple ``(obs, info)``.
        """
        import gymnasium

        gymnasium.Env.reset(self, seed=seed)

        if self.muscle_condition == "fatigue":
            self.muscle_fatigue.reset(
                fatigue_reset_vec=self.fatigue_reset_vec,
                fatigue_reset_random=self.fatigue_reset_random,
            )

        if self.reset_type is None or self.reset_type == "none":
            # No physics reset; return current obs
            self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
            obs = self._obs_dict_to_vec(self.get_obs_dict(self._accessor))
            return obs, {}

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel

        if self.reset_type == "random":
            jnt_init = self.np_random.uniform(
                high=self.model.jnt_range[:, 1],
                low=self.model.jnt_range[:, 0],
            )
            self.data.qpos[:] = jnt_init

        mujoco.mj_forward(self.model, self.data)
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs = self._obs_dict_to_vec(self.get_obs_dict(self._accessor))
        return obs, {}

    # ── Compatibility helpers ──────────────────────────────────────────────

    @property
    def dt(self) -> float:
        """Control timestep in seconds."""
        return self._ctrl_dt

    def get_obs(self) -> np.ndarray:
        """Return the current observation vector (legacy compat)."""
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        return self._obs_dict_to_vec(self.get_obs_dict(self._accessor))
