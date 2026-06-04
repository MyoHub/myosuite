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
from myosuite.physics.fatigue import CumulativeFatigue
from myosuite.terms.base_action import sigmoid_muscle_activation
from myosuite.physics.quat_math import euler2quat
from myosuite.physics.quat_math import calculate_cosine


class PenTwirlFixedEnvV0(MyoGymnasiumEnv, EzPickle):
    """Pen-twirling task with a fixed target orientation.

    The agent must align a pen object with a fixed target orientation while
    keeping it within a grasp-zone site. The hand starts open and palm-up.

    Args:
        model_path: Absolute path to the MuJoCo XML model.
        obsd_model_path: Unused (kept for API compatibility).
        seed: Random seed.
        obs_keys: Observation keys.
        weighted_reward_keys: Dict ``{reward_key: weight}`` for dense reward.
        normalize_act: If ``True``, action space is ``[-1, 1]``.
        frame_skip: Number of MuJoCo substeps per :meth:`step` call.
        muscle_condition: One of ``""``, ``"sarcopenia"``, ``"fatigue"``,
            ``"reafferentation"``.
        fatigue_reset_vec: Initial fatigue state vector.
        fatigue_reset_random: If ``True``, randomise fatigue state on reset.
    """

    DEFAULT_OBS_KEYS = [
        "hand_jnt",
        "obj_pos",
        "obj_vel",
        "obj_rot",
        "obj_des_rot",
        "obj_err_pos",
        "obj_err_rot",
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pos_align": 1.0,
        "rot_align": 1.0,
        "act_reg": 5.0,
        "drop": 5.0,
        "bonus": 10.0,
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
        self.model, self._mj_spec = ModelBuilder.from_xml_file(model_path).build()
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * frame_skip)

        # ── Muscle condition ───────────────────────────────────────────────
        self.muscle_condition = muscle_condition
        self.fatigue_reset_vec = fatigue_reset_vec
        self.fatigue_reset_random = fatigue_reset_random
        self._muscle_act_ind = self.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self._init_muscle_condition()

        # ── Body / site IDs ────────────────────────────────────────────────
        mujoco.mj_forward(self.model, self.data)
        self.target_obj_bid = self.model.body("target").id
        self.S_grasp_sid = self.model.site("S_grasp").id
        self.obj_bid = self.model.body("Object").id
        self.eps_ball_sid = self.model.site("eps_ball").id
        self.obj_t_sid = self.model.site("object_top").id
        self.obj_b_sid = self.model.site("object_bottom").id
        self.tar_t_sid = self.model.site("target_top").id
        self.tar_b_sid = self.model.site("target_bottom").id

        self.pen_length = float(
            np.linalg.norm(
                self.model.site_pos[self.obj_t_sid]
                - self.model.site_pos[self.obj_b_sid]
            )
        )
        self.tar_length = float(
            np.linalg.norm(
                self.model.site_pos[self.tar_t_sid]
                - self.model.site_pos[self.tar_b_sid]
            )
        )

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
        # Old code: init_qpos[:-6] *= 0 (open hand), init_qpos[0] = -1.5 (palm up)
        init_qpos = self.data.qpos.ravel().copy()
        init_qpos[:-6] = 0.0
        init_qpos[0] = -1.5
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
        elif self.muscle_condition == "reafferentation":
            self.EPLpos = self.model.actuator("EPL").id
            self.EIPpos = self.model.actuator("EIP").id

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
        obj_pos = accessor.data.xpos[self.obj_bid].copy()
        obj_des_pos = accessor.site_xpos(self.eps_ball_sid).ravel()
        obj_rot = (
            accessor.site_xpos(self.obj_t_sid) - accessor.site_xpos(self.obj_b_sid)
        ) / self.pen_length
        obj_des_rot = (
            accessor.site_xpos(self.tar_t_sid) - accessor.site_xpos(self.tar_b_sid)
        ) / self.tar_length

        obs: dict[str, np.ndarray] = {
            "time": np.array([accessor.time()]),
            "hand_jnt": accessor.data.qpos[:-6].copy(),
            "obj_pos": obj_pos,
            "obj_des_pos": obj_des_pos,
            "obj_vel": accessor.data.qvel[-6:].copy() * accessor.dt(),
            "obj_rot": obj_rot,
            "obj_des_rot": obj_des_rot,
            "obj_err_pos": obj_pos - obj_des_pos,
            "obj_err_rot": obj_rot - obj_des_rot,
            "act": accessor.muscle_act(),
        }
        return {k: obs[k] for k in self.obs_keys if k in obs}

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        """Compute the reward dictionary.

        Args:
            obs_dict: Output of :meth:`get_obs_dict`.

        Returns:
            Ordered dict with reward components.
        """
        obj_rot = obs_dict.get("obj_rot", np.zeros(3))
        obj_des_rot = obs_dict.get("obj_des_rot", np.zeros(3))
        pos_err = obs_dict.get("obj_err_pos", np.zeros(3))
        pos_align = float(np.linalg.norm(pos_err))
        rot_align = float(calculate_cosine(obj_rot, obj_des_rot))
        dropped = pos_align > 0.075
        act_mag = (
            float(np.linalg.norm(obs_dict.get("act", np.zeros(1)))) / self.model.na
            if self.model.na != 0
            else 0.0
        )

        rwd_dict = collections.OrderedDict(
            (
                ("pos_align", -1.0 * pos_align),
                ("rot_align", rot_align),
                ("act_reg", -1.0 * act_mag),
                ("drop", -1.0 * dropped),
                (
                    "bonus",
                    1.0 * (rot_align > 0.9) * (pos_align < 0.075)
                    + 5.0 * (rot_align > 0.95) * (pos_align < 0.075),
                ),
                ("sparse", -1.0 * pos_align + rot_align),
                ("solved", (rot_align > 0.95) * (~dropped)),
                ("done", dropped),
            )
        )
        rwd_dict["dense"] = float(
            np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()])
        )
        return rwd_dict

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        """Fixed target orientation — no per-episode sampling.

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


class PenTwirlRandomEnvV0(PenTwirlFixedEnvV0):
    """Pen-twirling task with a randomly sampled target orientation each episode.

    Extends :class:`PenTwirlFixedEnvV0` by overriding :meth:`reset_task` to
    randomise the target body quaternion via Euler angles.
    """

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        """Sample a random target orientation via Euler angles.

        Args:
            np_random: NumPy random generator from gymnasium.

        Returns:
            Empty task state dict.
        """
        desired_orien = np.zeros(3)
        desired_orien[0] = np_random.uniform(low=-1, high=1)
        desired_orien[1] = np_random.uniform(low=-1, high=1)
        self.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        return {}
