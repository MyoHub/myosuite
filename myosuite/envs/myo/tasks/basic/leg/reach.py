# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Leg (torso) reach environment — migrated from BaseV0 to MyoGymnasiumEnv.

Replaces walk_v0.ReachEnvV0 for myoLegStandRandom-v0 and variants.
Preserves identical obs/reward schema and semantics for parity.
"""

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


class LegReachEnvV0(MyoGymnasiumEnv, EzPickle):
    """Torso/leg reaching task: stand still and reach target site positions.

    Migrated from walk_v0.ReachEnvV0 (BaseV0). Same obs/reward contract.
    """

    DEFAULT_OBS_KEYS = ["qpos", "qvel", "tip_pos", "reach_err"]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": 1.0,
        "bonus": 4.0,
        "penalty": 50,
        "act_reg": 1,
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
        target_reach_range: dict | None = None,
        joint_random_range: tuple[float, float] = (0.0, 0.0),
        far_th: float = 0.35,
        obs_keys: list | None = None,
        weighted_reward_keys: dict[str, float] | None = None,
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
            target_reach_range=target_reach_range,
            joint_random_range=joint_random_range,
            far_th=far_th,
            obs_keys=obs_keys or self.DEFAULT_OBS_KEYS,
            weighted_reward_keys=weighted_reward_keys
            or self.DEFAULT_RWD_KEYS_AND_WEIGHTS,
            normalize_act=normalize_act,
            frame_skip=frame_skip,
            muscle_condition=muscle_condition,
            fatigue_reset_vec=fatigue_reset_vec,
            fatigue_reset_random=fatigue_reset_random,
            **kwargs,
        )

        if target_reach_range is None:
            raise ValueError("target_reach_range is required")

        self.model, self._mj_spec = ModelBuilder.from_xml_file(model_path).build()
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * frame_skip)

        self.target_reach_range = target_reach_range
        self.joint_random_range = joint_random_range
        self.far_th = far_th
        self.normalize_act = normalize_act
        self.muscle_condition = muscle_condition
        self.fatigue_reset_vec = fatigue_reset_vec
        self.fatigue_reset_random = fatigue_reset_random

        self.tip_sids = [self.model.site(s).id for s in target_reach_range]
        self.target_sids = [
            self.model.site(s + "_target").id for s in target_reach_range
        ]

        self.rwd_keys_wt = weighted_reward_keys or self.DEFAULT_RWD_KEYS_AND_WEIGHTS
        self.obs_keys = list(obs_keys or self.DEFAULT_OBS_KEYS)
        if self.model.na > 0 and "act" not in self.obs_keys:
            self.obs_keys.append("act")

        self._muscle_act_ind = self.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self._init_muscle_condition()

        self._init_qpos = self.model.key_qpos[0].copy()
        self._init_qvel = self.model.key_qvel[0].copy()

        geom_1_indices = np.where(self.model.geom_group == 1)[0]
        self.model.geom_rgba[geom_1_indices, 3] = 0
        try:
            tid = self.model.geom("terrain").id
            self.model.geom_rgba[tid, 3] = 0.0
            self.model.geom_pos[tid] = np.array([0.0, 0.0, -10.0])
        except (KeyError, AttributeError):
            pass

        self._input_seed = seed
        import gymnasium as _gym

        _gym.Env.reset(self, seed=seed)

        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        full_obs = self._get_obs_dict(self._accessor)
        obs = self._obs_dict_to_vec(full_obs)
        self.observation_space = gym.spaces.Box(
            -10.0 * np.ones(obs.size, dtype=np.float32),
            10.0 * np.ones(obs.size, dtype=np.float32),
            dtype=np.float32,
        )
        if normalize_act:
            act_low = -np.ones(self.model.nu, dtype=np.float32)
            act_high = np.ones(self.model.nu, dtype=np.float32)
        else:
            act_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
            act_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
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

    def _generate_qpos(self) -> np.ndarray:
        qpos_rand = self.np_random.uniform(
            low=self.joint_random_range[0],
            high=self.joint_random_range[1],
            size=self._init_qpos.shape,
        )
        qpos_new = self._init_qpos.copy()
        jnt_adr = self.model.jnt_qposadr
        qpos_new[jnt_adr] += qpos_rand[jnt_adr]
        qpos_new[jnt_adr] = np.clip(
            qpos_new[jnt_adr],
            self.model.jnt_range[:, 0],
            self.model.jnt_range[:, 1],
        )
        return qpos_new

    def _generate_targets(self) -> None:
        for site_name, span in self.target_reach_range.items():
            sid = self.model.site(site_name).id
            sid_target = self.model.site(site_name + "_target").id
            self.model.site_pos[sid_target] = self.data.site_xpos[sid].copy()
            self.model.site_pos[sid_target] += self.np_random.uniform(
                low=span[0], high=span[1]
            )
        mujoco.mj_forward(self.model, self.data)

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
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
            "tip_pos": tip_pos,
            "target_pos": target_pos,
            "reach_err": target_pos - tip_pos,
        }
        if self.model.na > 0:
            obs["act"] = accessor.muscle_act()
        return obs

    def _obs_dict_to_vec(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Flatten only obs_keys from obs dict to match legacy observation vector."""
        return np.concatenate(
            [np.atleast_1d(obs_dict[k]).ravel() for k in self.obs_keys if k in obs_dict]
        )

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        reach_dist = float(np.linalg.norm(obs_dict["reach_err"]))
        vel_dist = float(np.linalg.norm(obs_dict["qvel"]))
        act_mag = (
            float(np.linalg.norm(obs_dict.get("act", np.zeros(1)))) / self.model.na
            if self.model.na != 0
            else 0.0
        )
        t = float(np.squeeze(obs_dict["time"]))
        far_th = (
            self.far_th * len(self.tip_sids)
            if t > 2.0 * self._ctrl_dt
            else float("inf")
        )
        near_th = len(self.tip_sids) * 0.050
        rwd_dict = collections.OrderedDict(
            (
                ("reach", 10.0 - 1.0 * reach_dist - 10.0 * vel_dist),
                (
                    "bonus",
                    1.0 * (reach_dist < 2 * near_th) + 1.0 * (reach_dist < near_th),
                ),
                ("act_reg", -100.0 * act_mag),
                ("penalty", -1.0 * (reach_dist > far_th)),
                ("sparse", -1.0 * reach_dist),
                ("solved", reach_dist < near_th),
                ("done", reach_dist > far_th),
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
        return rwd_dict

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        if np.ptp(self.joint_random_range) > 0:
            self.data.qpos[:] = self._generate_qpos()
        else:
            self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel
        mujoco.mj_forward(self.model, self.data)
        self._generate_targets()
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
        mujoco.mj_forward(self.model, self.data)
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs = self._obs_dict_to_vec(self._get_obs_dict(self._accessor))
        obs = np.clip(
            np.asarray(obs, dtype=np.float32),
            self.observation_space.low,
            self.observation_space.high,
        )
        return obs, {}

    @property
    def dt(self) -> float:
        return self._ctrl_dt
