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

from myosuite.core.muscle_conditions import apply_sarcopenia_to_model
from myosuite.envs.gymnasium_env import CpuEnvAccessor, MyoGymnasiumEnv
from myosuite.core.model_builder import ModelBuilder, build_from_recipe
from myosuite.physics.fatigue import CumulativeFatigue
from myosuite.terms.base_action import sigmoid_muscle_activation


class ReachEnvV0(MyoGymnasiumEnv, EzPickle):
    """Reaching task for musculoskeletal (or motor) MuJoCo models.

    The agent must move one or more fingertip sites to randomly sampled
    target positions in 3-D space.

    Args:
        model_path: Absolute path to the MuJoCo XML model.
        obsd_model_path: Unused (kept for API compatibility).
        seed: Random seed.
        target_reach_range: Dict ``{site_name: (low_3d, high_3d)}``
            defining the sampling bounds for each target site.
        far_th: Distance threshold beyond which a penalty is applied
            (only active after ``2 * dt`` seconds).
        obs_keys: Keys to include in the observation vector.
        weighted_reward_keys: Dict ``{reward_key: weight}`` for dense reward.
        normalize_act: If ``True``, action space is ``[-1, 1]``; sigmoid
            (muscles) or linear (motors) denormalisation applied internally.
        frame_skip: Number of MuJoCo substeps per :meth:`step` call.
        muscle_condition: One of ``""``, ``"sarcopenia"``, ``"fatigue"``,
            ``"reafferentation"``.
        fatigue_reset_vec: Initial fatigue state vector.
        fatigue_reset_random: If ``True``, randomise fatigue state on reset.
    """

    DEFAULT_OBS_KEYS = ["qpos", "qvel", "tip_pos", "reach_err"]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": 1.0,
        "bonus": 4.0,
        "penalty": 50,
    }

    MYO_CREDIT = """\
    MyoSuite: A contact-rich simulation suite for musculoskeletal motor control
        Vittorio Caggiano, Huawei Wang, Guillaume Durandau, Massimo Sartori, Vikash Kumar
        L4DC-2019 | https://sites.google.com/view/myosuite
    """

    def __init__(
        self,
        model_path: str = "",
        obsd_model_path: str | None = None,
        seed: int | None = None,
        target_reach_range: dict | None = None,
        far_th: float = 0.35,
        obs_keys: list = DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict[str, float] = DEFAULT_RWD_KEYS_AND_WEIGHTS,
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
            target_reach_range=target_reach_range,
            far_th=far_th,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            normalize_act=normalize_act,
            frame_skip=frame_skip,
            muscle_condition=muscle_condition,
            fatigue_reset_vec=fatigue_reset_vec,
            fatigue_reset_random=fatigue_reset_random,
            **kwargs,
        )

        # ── Load model ─────────────────────────────────────────────────────
        model_recipe = kwargs.pop("model_recipe", None)
        edit_fn = kwargs.pop("edit_fn", None)
        if model_recipe is not None:
            self._model_recipe = model_recipe
            self.model, self._mj_spec = build_from_recipe(model_recipe)
            if edit_fn is not None:
                edit_fn(self._mj_spec)
                self.model = self._mj_spec.compile()
        else:
            builder = ModelBuilder.from_xml_file(model_path)
            if edit_fn is not None:

                def _wrap(spec: mujoco.MjSpec, _fn=edit_fn) -> mujoco.MjSpec:
                    _fn(spec)
                    return spec

                builder = builder.apply_transform(_wrap)
            self.model, self._mj_spec = builder.build()
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * frame_skip)

        # ── Muscle condition ───────────────────────────────────────────────
        self.muscle_condition = muscle_condition
        self.fatigue_reset_vec = fatigue_reset_vec
        self.fatigue_reset_random = fatigue_reset_random
        self._muscle_act_ind = self.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self._init_muscle_condition()

        # ── Reach target config ────────────────────────────────────────────
        if target_reach_range is None:
            raise ValueError("target_reach_range is required")
        self.target_reach_range = target_reach_range
        self.far_th = far_th

        # Site IDs: tip site and corresponding <site>_target site
        self.tip_sids: list[int] = []
        self.target_sids: list[int] = []
        for site_name in target_reach_range:
            self.tip_sids.append(self.model.site(site_name).id)
            self.target_sids.append(self.model.site(site_name + "_target").id)

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

        # Store seed for backward-compat get_input_seed() / seed() API.
        self._input_seed = seed

        # Initialise np_random without triggering our overridden reset()
        import gymnasium

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
        elif self.muscle_condition == "reafferentation":
            sfx = "_r" if hasattr(self, "_model_recipe") else ""
            self.EPLpos = self.model.actuator(f"EPL{sfx}").id
            self.EIPpos = self.model.actuator(f"EIP{sfx}").id

    def _apply_action(self, action: np.ndarray) -> None:
        """Map action to MuJoCo ctrl and write to data.ctrl.

        Args:
            action: Action vector in the action-space (already clipped).
        """
        ctrl = action.copy()  # preserve float32

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
        elif self.muscle_condition == "reafferentation":
            ctrl[self.EPLpos] = ctrl[self.EIPpos].copy()
            ctrl[self.EIPpos] = 0.0

        self.data.ctrl[:] = ctrl

    # ── MyoGymnasiumEnv interface ──────────────────────────────────────────

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        """Compute the observation dictionary.

        Args:
            accessor: CPU environment accessor.

        Returns:
            Filtered dict with obs_keys as keys.
        """
        tip_pos = np.concatenate(
            [accessor.site_xpos(sid).ravel() for sid in self.tip_sids]
        )
        target_pos = np.concatenate(
            [accessor.site_xpos(sid).ravel() for sid in self.target_sids]
        )
        obs: dict[str, np.ndarray] = {
            "time": np.array([accessor.time()]),
            "qpos": accessor.joint_pos(),
            "qvel": accessor.joint_vel() * accessor.dt(),
            "act": accessor.muscle_act(),
            "tip_pos": tip_pos,
            "target_pos": target_pos,
            "reach_err": target_pos - tip_pos,
        }
        return {k: obs[k] for k in self.obs_keys if k in obs}

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        """Compute the reward dictionary.

        Args:
            obs_dict: Output of :meth:`get_obs_dict`.

        Returns:
            Ordered dict with reward components including ``"dense"`` and
            ``"done"``.
        """
        reach_err = obs_dict.get("reach_err", np.zeros(3 * len(self.tip_sids)))
        reach_dist = float(np.linalg.norm(reach_err))
        act_mag = (
            float(np.linalg.norm(obs_dict.get("act", np.zeros(1)))) / self.model.na
            if self.model.na != 0
            else 0.0
        )
        near_th = len(self.tip_sids) * 0.0125
        # Penalty only applies after the first 2 control steps
        far_th = (
            self.far_th * len(self.tip_sids)
            if self.data.time > 2.0 * self._ctrl_dt
            else float("inf")
        )

        rwd_dict = collections.OrderedDict(
            (
                ("reach", -1.0 * reach_dist),
                (
                    "bonus",
                    1.0 * (reach_dist < 2 * near_th) + 1.0 * (reach_dist < near_th),
                ),
                ("act_reg", -1.0 * act_mag),
                ("penalty", -1.0 * (reach_dist > far_th)),
                ("sparse", -1.0 * reach_dist),
                ("solved", reach_dist < near_th),
                ("done", reach_dist > far_th),
            )
        )
        rwd_dict["dense"] = float(
            np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()])
        )
        return rwd_dict

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        """Sample new target positions for all reach sites.

        Args:
            np_random: NumPy random generator from gymnasium.

        Returns:
            Empty task state dict (targets stored on model directly).
        """
        for site_name, span in self.target_reach_range.items():
            sid = self.model.site(site_name + "_target").id
            self.model.site_pos[sid] = np_random.uniform(low=span[0], high=span[1])
        mujoco.mj_forward(self.model, self.data)
        return {}

    # ── Gymnasium step / reset ─────────────────────────────────────────────

    def step(self, action: np.ndarray, **kwargs: Any):
        """Advance the simulation by one control step.

        Args:
            action: Control command in the action space.

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

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel

        # Sample target positions first (matches old generate_target_pose() ordering)
        self._task_state = self.reset_task(self.np_random)

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
