# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Relocate task implemented on MyoGymnasiumEnv."""

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
from myosuite.physics.quat_math import euler2quat, mat2euler


class RelocateEnv(MyoGymnasiumEnv, EzPickle):
    """Relocate object to target pose; migrated from BaseV0 for parity."""

    DEFAULT_OBS_KEYS = [
        "hand_qpos",
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
    }

    def __init__(
        self,
        model_path: str,
        obsd_model_path: str | None = None,
        seed: int | None = None,
        target_xyz_range: dict | None = None,
        target_rxryrz_range: dict | None = None,
        obj_xyz_range: dict | None = None,
        obj_geom_range: dict | None = None,
        obj_mass_range: dict | None = None,
        obj_friction_range: dict | None = None,
        qpos_noise_range: float | None = None,
        obs_keys: list | None = None,
        weighted_reward_keys: dict[str, float] | None = None,
        pos_th: float = 0.025,
        rot_th: float = 0.262,
        drop_th: float = 0.50,
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
            target_xyz_range=target_xyz_range or {"low": [0], "high": [0]},
            target_rxryrz_range=target_rxryrz_range or {"low": [0], "high": [0]},
            obj_xyz_range=obj_xyz_range,
            obj_geom_range=obj_geom_range,
            obj_mass_range=obj_mass_range,
            obj_friction_range=obj_friction_range,
            qpos_noise_range=qpos_noise_range,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
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
        if target_xyz_range is None or target_rxryrz_range is None:
            raise ValueError("target_xyz_range and target_rxryrz_range required")

        self.model, self._mj_spec = ModelBuilder.from_xml_file(model_path).build()
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * frame_skip)

        self.palm_sid = self.model.site("S_grasp").id
        self.object_sid = self.model.site("object_o").id
        self.object_bid = self.model.body("Object").id
        self.goal_sid = self.model.site("target_o").id
        self.success_indicator_sid = self.model.site("target_ball").id
        self.goal_bid = self.model.body("target").id
        self.target_xyz_range = target_xyz_range
        self.target_rxryrz_range = target_rxryrz_range
        self.obj_geom_range = obj_geom_range
        self.obj_mass_range = obj_mass_range
        self.obj_friction_range = obj_friction_range
        self.obj_xyz_range = obj_xyz_range
        self.qpos_noise_range = qpos_noise_range
        self.pos_th = pos_th
        self.rot_th = rot_th
        self.drop_th = drop_th
        self.normalize_act = normalize_act
        self.muscle_condition = muscle_condition
        self.fatigue_reset_vec = fatigue_reset_vec
        self.fatigue_reset_random = fatigue_reset_random

        self.rwd_keys_wt = weighted_reward_keys or self.DEFAULT_RWD_KEYS_AND_WEIGHTS
        self.obs_keys = list(obs_keys or self.DEFAULT_OBS_KEYS)
        if self.model.na > 0 and "act" not in self.obs_keys:
            self.obs_keys.append("act")

        self._muscle_act_ind = self.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self._init_muscle_condition()

        key_frame_id = 0 if self.obj_xyz_range is None else 1
        self._init_qpos = self.model.key_qpos[key_frame_id].copy()
        self._init_qvel = self.model.key_qvel[key_frame_id].copy()

        self._input_seed = seed
        import gymnasium as _gym

        _gym.Env.reset(self, seed=seed)

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

    def _init_muscle_condition(self) -> None:
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
        ctrl = np.clip(action, self.action_space.low, self.action_space.high).astype(
            np.float64
        )
        if self.model.na > 0 and self.normalize_act:
            ctrl[self._muscle_act_ind] = sigmoid_muscle_activation(
                ctrl[self._muscle_act_ind], np
            )
        elif self.normalize_act and self.model.nu > 0:
            cr = self.model.actuator_ctrlrange
            ctrl = np.mean(cr, axis=-1) + ctrl * (cr[:, 1] - cr[:, 0]) / 2.0
        if self.muscle_condition == "fatigue":
            ctrl[self._muscle_act_ind], _, _ = self.muscle_fatigue.compute_act(
                ctrl[self._muscle_act_ind]
            )
        elif self.muscle_condition == "reafferentation":
            ctrl[self.EPLpos] = ctrl[self.EIPpos].copy()
            ctrl[self.EIPpos] = 0.0
        self.data.ctrl[:] = ctrl

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        qpos = accessor.joint_pos()
        qvel = accessor.joint_vel()
        obj_pos = accessor.site_xpos(self.object_sid).ravel()
        goal_pos = accessor.site_xpos(self.goal_sid).ravel()
        palm_pos = accessor.site_xpos(self.palm_sid).ravel()
        obj_xmat = self.data.site_xmat[self.object_sid].reshape(3, 3)
        goal_xmat = self.data.site_xmat[self.goal_sid].reshape(3, 3)
        obj_rot = mat2euler(obj_xmat)
        goal_rot = mat2euler(goal_xmat)
        obs = {
            "time": np.array([accessor.time()]),
            "hand_qpos": qpos[:-7].copy(),
            "hand_qvel": (qvel[:-6] * accessor.dt()).copy(),
            "obj_pos": obj_pos,
            "goal_pos": goal_pos,
            "palm_pos": palm_pos,
            "pos_err": goal_pos - obj_pos,
            "reach_err": palm_pos - obj_pos,
            "obj_rot": obj_rot,
            "goal_rot": goal_rot,
            "rot_err": goal_rot - obj_rot,
        }
        if self.model.na > 0:
            obs["act"] = accessor.muscle_act()
        return obs

    def _obs_dict_to_vec(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [np.atleast_1d(obs_dict[k]).ravel() for k in self.obs_keys if k in obs_dict]
        )

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        reach_dist = float(np.linalg.norm(obs_dict["reach_err"]))
        pos_dist = float(np.linalg.norm(obs_dict["pos_err"]))
        rot_dist = float(np.linalg.norm(obs_dict["rot_err"]))
        act_mag = (
            float(np.linalg.norm(obs_dict.get("act", np.zeros(1)))) / self.model.na
            if self.model.na != 0
            else 0.0
        )
        drop = reach_dist > self.drop_th
        rwd_dict = collections.OrderedDict(
            (
                ("pos_dist", -1.0 * pos_dist),
                ("rot_dist", -1.0 * rot_dist),
                ("act_reg", -1.0 * act_mag),
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
            np.sum(
                [
                    self.rwd_keys_wt[k] * rwd_dict[k]
                    for k in self.rwd_keys_wt
                    if k in rwd_dict
                ]
            )
        )
        self.model.site_rgba[self.success_indicator_sid, :2] = (
            np.array([0, 2]) if rwd_dict["solved"] else np.array([2, 0])
        )
        self.model.site_size[self.success_indicator_sid, :] = (
            np.array([0.25]) if rwd_dict["solved"] else np.array([0.1])
        )
        return rwd_dict

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        self.model.body_pos[self.goal_bid] = np_random.uniform(**self.target_xyz_range)
        self.model.body_quat[self.goal_bid] = euler2quat(
            np_random.uniform(**self.target_rxryrz_range)
        )
        if self.obj_xyz_range is not None:
            self.model.body_pos[self.object_bid] = np_random.uniform(
                **self.obj_xyz_range
            )
        if self.obj_geom_range is not None:
            bid = self.model.body("Object").id
            for i in range(self.model.body_geomnum[bid]):
                gid = self.model.body_geomadr[bid] + i
                self.model.geom_type[gid] = np_random.choice([2, 3, 4, 5, 6])
                self.model.geom_size[gid] = np_random.uniform(
                    low=self.obj_geom_range["low"],
                    high=self.obj_geom_range["high"],
                )
                self.model.geom_aabb[gid][3:] = self.obj_geom_range["high"]
                self.model.geom_rbound[gid] = 2.0 * max(self.obj_geom_range["high"])
                self.model.geom_pos[gid] = np_random.uniform(
                    low=-1.0 * self.model.geom_size[gid],
                    high=self.model.geom_size[gid],
                )
                self.model.geom_quat[gid] = euler2quat(
                    np_random.uniform(
                        low=(-np.pi / 2, -np.pi / 2, -np.pi / 2),
                        high=(np.pi / 2, np.pi / 2, np.pi / 2),
                    )
                )
                self.model.geom_rgba[gid] = np_random.uniform(
                    low=[0.2, 0.2, 0.2, 1], high=[0.9, 0.9, 0.9, 1]
                )
                if self.obj_friction_range is not None:
                    self.model.geom_friction[gid] = np_random.uniform(
                        **self.obj_friction_range
                    )
            if self.obj_mass_range is not None:
                self.model.body_mass[self.object_bid] = np_random.uniform(
                    **self.obj_mass_range
                )
            mujoco.mj_forward(self.model, self.data)
        return {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._apply_action(action)
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

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        import gymnasium as _gym

        _gym.Env.reset(self, seed=seed)
        if self.muscle_condition == "fatigue":
            self.muscle_fatigue.reset(
                fatigue_reset_vec=self.fatigue_reset_vec,
                fatigue_reset_random=self.fatigue_reset_random,
            )
        mujoco.mj_resetData(self.model, self.data)
        self._task_state = self.reset_task(self.np_random)
        if self.qpos_noise_range is not None:
            nq = len(self._init_qpos)
            self.data.qpos[:-6] = self._init_qpos[:-6] + self.np_random.uniform(
                -self.qpos_noise_range, self.qpos_noise_range, size=nq - 6
            )
            self.data.qpos[-6:] = self._init_qpos[-6:]
        else:
            self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel
        mujoco.mj_forward(self.model, self.data)
        while self.data.ncon > 0:
            self._task_state = self.reset_task(self.np_random)
            if self.qpos_noise_range is not None:
                nq = len(self._init_qpos)
                self.data.qpos[:-6] = self._init_qpos[:-6] + self.np_random.uniform(
                    -self.qpos_noise_range, self.qpos_noise_range, size=nq - 6
                )
                self.data.qpos[-6:] = self._init_qpos[-6:]
            else:
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
