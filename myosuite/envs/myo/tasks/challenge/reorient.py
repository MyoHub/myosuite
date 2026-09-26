# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Die reorientation task on MyoGymnasiumEnv (native implementation)."""

from __future__ import annotations

import collections
from typing import Any

import gymnasium as gym
from gymnasium.utils import EzPickle
import mujoco
import numpy as np

from myosuite.core.model_builder import ModelBuilder
from myosuite.envs.gymnasium_env import CpuEnvAccessor, MyoGymnasiumEnv
from myosuite.envs.myo.tasks.challenge.challenge_common import MuscleActionMixin
from myosuite.physics.quat_math import euler2quat, mat2euler


class ReorientEnv(MuscleActionMixin, MyoGymnasiumEnv, EzPickle):
    """Die reorientation task with parity-focused behavior."""

    DEFAULT_OBS_KEYS = [
        "hand_qpos_noMD5",
        "hand_qvel",
        "obj_pos",
        "goal_pos",
        "pos_err",
        "obj_rot",
        "goal_rot",
        "rot_err",
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pos_dist": 100.0,
        "rot_dist": 1.0,
        "bonus": 0.0,
        "act_reg": 0.0,
        "penalty": 0.0,
    }

    def __init__(
        self,
        model_path: str,
        obsd_model_path: str | None = None,
        seed: int | None = None,
        obs_keys: list[str] | None = None,
        weighted_reward_keys: dict[str, float] | None = None,
        goal_pos: tuple[float, float] = (0.0, 0.0),
        goal_rot: tuple[float, float] = (0.785, 0.785),
        obj_size_change: float = 0.0,
        obj_mass_range: tuple[float, float] = (0.108, 0.108),
        obj_friction_change: tuple[float, float, float] = (0.0, 0.0, 0.0),
        pos_th: float = 0.025,
        rot_th: float = 0.262,
        drop_th: float = 0.200,
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
        EzPickle.__init__(
            self,
            model_path,
            obsd_model_path,
            seed,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            goal_pos=goal_pos,
            goal_rot=goal_rot,
            obj_size_change=obj_size_change,
            obj_mass_range=obj_mass_range,
            obj_friction_change=obj_friction_change,
            pos_th=pos_th,
            rot_th=rot_th,
            drop_th=drop_th,
            normalize_act=normalize_act,
            frame_skip=frame_skip,
            muscle_condition=muscle_condition,
            fatigue_reset_vec=fatigue_reset_vec,
            fatigue_reset_random=fatigue_reset_random,
            **kwargs,
        )

        self.model, self._mj_spec = ModelBuilder.from_xml_file(model_path).build()
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * frame_skip)

        self.object_sid = self.model.site("object_o").id
        self.goal_sid = self.model.site("target_o").id
        self.success_indicator_sid = self.model.site("target_ball").id
        self.goal_bid = self.model.body("target").id

        self.goal_pos = goal_pos
        self.goal_rot = goal_rot
        self.pos_th = pos_th
        self.rot_th = rot_th
        self.drop_th = drop_th
        self.normalize_act = normalize_act
        self.muscle_condition = muscle_condition
        self.fatigue_reset_vec = fatigue_reset_vec
        self.fatigue_reset_random = fatigue_reset_random

        self.target_gid = self.model.geom("target_dice").id
        self.target_default_size = self.model.geom_size[self.target_gid].copy()

        self.object_bid = self.model.body("Object").id
        self.object_gid0 = self.model.body_geomadr[self.object_bid]
        self.object_gidn = self.object_gid0 + self.model.body_geomnum[self.object_bid]
        self.object_default_size = self.model.geom_size[
            self.object_gid0 : self.object_gidn
        ].copy()
        self.object_default_pos = self.model.geom_pos[
            self.object_gid0 : self.object_gidn
        ].copy()
        self.obj_mass_range = {"low": obj_mass_range[0], "high": obj_mass_range[1]}
        self.obj_size_range = {"low": -obj_size_change, "high": obj_size_change}
        self.obj_friction_range = {
            "low": self.model.geom_friction[self.object_gid0 : self.object_gidn]
            - obj_friction_change,
            "high": self.model.geom_friction[self.object_gid0 : self.object_gidn]
            + obj_friction_change,
        }

        self.rwd_keys_wt = weighted_reward_keys or self.DEFAULT_RWD_KEYS_AND_WEIGHTS
        self.obs_keys = list(obs_keys or self.DEFAULT_OBS_KEYS)
        if self.model.na > 0 and "act" not in self.obs_keys:
            self.obs_keys.append("act")

        self._muscle_act_ind = self.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self.init_muscle_condition()

        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()
        self._init_qpos[:-7] *= 0.0
        self._init_qpos[0] = -1.5
        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel
        mujoco.mj_forward(self.model, self.data)
        # Match legacy reference frame used for goal/object deltas.
        self.goal_init_pos = self.data.site_xpos[self.goal_sid].copy()
        self.goal_obj_offset = (
            self.data.site_xpos[self.goal_sid] - self.data.site_xpos[self.object_sid]
        )

        gym.Env.reset(self, seed=seed)
        mujoco.mj_forward(self.model, self.data)
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs = self._obs_dict_to_vec(self._get_obs_dict(self._accessor))
        self.observation_space = gym.spaces.Box(
            -10.0 * np.ones(obs.size, dtype=np.float32),
            10.0 * np.ones(obs.size, dtype=np.float32),
            dtype=np.float32,
        )
        act_low = (
            -np.ones(self.model.nu, dtype=np.float32)
            if normalize_act
            else self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        )
        act_high = (
            np.ones(self.model.nu, dtype=np.float32)
            if normalize_act
            else self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        )
        self.action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        obs_dict: dict[str, np.ndarray] = {}
        obs_dict["time"] = np.array([accessor.time()])
        obs_dict["hand_qpos_noMD5"] = self.data.qpos[:-7].copy()
        obs_dict["hand_qpos"] = self.data.qpos[:-6].copy()
        obs_dict["hand_qvel"] = self.data.qvel[:-6].copy() * accessor.dt()
        obs_dict["obj_pos"] = self.data.site_xpos[self.object_sid]
        obs_dict["goal_pos"] = self.data.site_xpos[self.goal_sid]
        obs_dict["pos_err"] = (
            obs_dict["goal_pos"] - obs_dict["obj_pos"] - self.goal_obj_offset
        )
        obs_dict["obj_rot"] = mat2euler(
            self.data.site_xmat[self.object_sid].reshape(3, 3)
        )
        obs_dict["goal_rot"] = mat2euler(
            self.data.site_xmat[self.goal_sid].reshape(3, 3)
        )
        obs_dict["rot_err"] = obs_dict["goal_rot"] - obs_dict["obj_rot"]
        if self.model.na > 0:
            obs_dict["act"] = accessor.muscle_act()
        return obs_dict

    def _obs_dict_to_vec(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [np.atleast_1d(obs_dict[k]).ravel() for k in self.obs_keys if k in obs_dict]
        )

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        pos_dist = float(np.abs(np.linalg.norm(obs_dict["pos_err"], axis=-1)))
        rot_dist = float(np.abs(np.linalg.norm(obs_dict["rot_err"], axis=-1)))
        act_mag = (
            float(np.linalg.norm(obs_dict["act"], axis=-1)) / self.model.na
            if self.model.na != 0 and "act" in obs_dict
            else 0.0
        )
        drop = pos_dist > self.drop_th
        rwd_dict = collections.OrderedDict(
            (
                ("pos_dist", -1.0 * pos_dist),
                ("rot_dist", -1.0 * rot_dist),
                (
                    "bonus",
                    1.0 * (pos_dist < 2 * self.pos_th) + 1.0 * (pos_dist < self.pos_th),
                ),
                ("act_reg", -1.0 * act_mag),
                ("penalty", -1.0 * drop),
                ("sparse", -rot_dist - 10.0 * pos_dist),
                (
                    "solved",
                    (pos_dist < self.pos_th)
                    and (rot_dist < self.rot_th)
                    and (not drop),
                ),
                ("done", drop),
            )
        )
        rwd_dict["dense"] = float(
            np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        )
        self.model.site_rgba[self.success_indicator_sid, :2] = (
            np.array([0, 2]) if rwd_dict["solved"] else np.array([2, 0])
        )
        return rwd_dict

    def _sample_goal_and_object(self) -> None:
        self.model.body_pos[self.goal_bid] = (
            self.goal_init_pos
            + self.np_random.uniform(
                high=self.goal_pos[1], low=self.goal_pos[0], size=3
            )
        )
        self.model.body_quat[self.goal_bid] = euler2quat(
            self.np_random.uniform(high=self.goal_rot[1], low=self.goal_rot[0], size=3)
        )
        self.model.geom_friction[self.object_gid0 : self.object_gidn] = (
            self.np_random.uniform(**self.obj_friction_range)
        )
        self.model.body_mass[self.object_bid] = self.np_random.uniform(
            **self.obj_mass_range
        )
        del_size = self.np_random.uniform(**self.obj_size_range)
        self.model.geom_size[self.target_gid] = self.target_default_size + del_size
        self.model.geom_size[self.object_gid0 : self.object_gidn - 3][:, 1] = (
            self.object_default_size[:-3][:, 1] + del_size
        )
        self.model.geom_size[self.object_gidn - 3 : self.object_gidn] = (
            self.object_default_size[-3:] + del_size
        )
        object_gpos = self.model.geom_pos[self.object_gid0 : self.object_gidn]
        self.model.geom_pos[self.object_gid0 : self.object_gidn] = (
            object_gpos
            / abs(object_gpos + 1e-16)
            * (abs(self.object_default_pos) + del_size)
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
        **_kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        gym.Env.reset(self, seed=seed)
        if self.muscle_condition == "fatigue":
            self.muscle_fatigue.reset(
                fatigue_reset_vec=self.fatigue_reset_vec,
                fatigue_reset_random=self.fatigue_reset_random,
            )
        mujoco.mj_resetData(self.model, self.data)
        self._sample_goal_and_object()
        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel
        mujoco.mj_forward(self.model, self.data)
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs = self._obs_dict_to_vec(self._get_obs_dict(self._accessor))
        obs = np.clip(
            np.asarray(obs, dtype=np.float32),
            self.observation_space.low,
            self.observation_space.high,
        )
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.apply_action(action)
        mujoco.mj_step(self.model, self.data, self.frame_skip)
        mujoco.mj_kinematics(self.model, self.data)
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs_dict = self._get_obs_dict(self._accessor)
        rwd_dict = self.get_reward_dict(obs_dict)
        obs = self._obs_dict_to_vec(obs_dict)
        obs = np.clip(
            np.asarray(obs, dtype=np.float32),
            self.observation_space.low,
            self.observation_space.high,
        )
        reward = float(rwd_dict["dense"])
        terminated = bool(rwd_dict["done"])
        info = {k: v for k, v in rwd_dict.items() if k not in ("dense", "done")}
        info["obs_dict"] = obs_dict
        info["rwd_dict"] = rwd_dict
        return obs, reward, terminated, False, info
