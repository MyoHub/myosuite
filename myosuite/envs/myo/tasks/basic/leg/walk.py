# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Leg walk environment — migrated from BaseV0 to MyoGymnasiumEnv.

Replaces walk_v0.WalkEnvV0 for myoLegWalk-v0 and variants.
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
from myosuite.terms.base_reward import walk_env_reward
from myosuite.physics.quat_math import quat2mat


class LegWalkEnvV0(MyoGymnasiumEnv, EzPickle):
    """Locomotion task for the musculoskeletal leg model.

    Migrated from walk_v0.WalkEnvV0 (BaseV0). Same obs/reward contract.
    """

    DEFAULT_OBS_KEYS = [
        "qpos_without_xy",
        "qvel",
        "com_vel",
        "torso_angle",
        "feet_heights",
        "height",
        "feet_rel_positions",
        "phase_var",
        "muscle_length",
        "muscle_velocity",
        "muscle_force",
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "vel_reward": 5.0,
        "done": -100,
        "cyclic_hip": -10,
        "ref_rot": 10.0,
        "joint_angle_rew": 5.0,
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
        obs_keys: list | None = None,
        weighted_reward_keys: dict[str, float] | None = None,
        min_height: float = 0.8,
        max_rot: float = 0.8,
        hip_period: int = 100,
        reset_type: str = "init",
        target_x_vel: float = 0.0,
        target_y_vel: float = 1.2,
        target_rot: np.ndarray | None = None,
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
            obs_keys=obs_keys or self.DEFAULT_OBS_KEYS,
            weighted_reward_keys=weighted_reward_keys
            or self.DEFAULT_RWD_KEYS_AND_WEIGHTS,
            min_height=min_height,
            max_rot=max_rot,
            hip_period=hip_period,
            reset_type=reset_type,
            target_x_vel=target_x_vel,
            target_y_vel=target_y_vel,
            target_rot=target_rot,
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

        self.min_height = min_height
        self.max_rot = max_rot
        self.hip_period = hip_period
        self.reset_type = reset_type
        self.target_x_vel = target_x_vel
        self.target_y_vel = target_y_vel
        self.target_rot = target_rot
        self.steps = 0
        self.normalize_act = normalize_act
        self.muscle_condition = muscle_condition
        self.fatigue_reset_vec = fatigue_reset_vec
        self.fatigue_reset_random = fatigue_reset_random

        self._init_qpos = self.model.key_qpos[0].copy()
        self._init_qvel = np.zeros_like(self.model.key_qvel[0]).copy()

        self.rwd_keys_wt = weighted_reward_keys or self.DEFAULT_RWD_KEYS_AND_WEIGHTS
        self.obs_keys = list(obs_keys or self.DEFAULT_OBS_KEYS)
        if self.model.na > 0 and "act" not in self.obs_keys:
            self.obs_keys.append("act")

        self._muscle_act_ind = self.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self._init_muscle_condition()

        try:
            tid = self.model.geom("terrain").id
            self.model.geom_rgba[tid, 3] = 0.0
            self.model.geom_pos[tid] = np.array([0.0, 0.0, -10.0])
        except (KeyError, AttributeError):
            # Flat-ground models have no "terrain" geom to hide; nothing to do.
            pass

        self._input_seed = seed
        import gymnasium as _gym

        _gym.Env.reset(self, seed=seed)

        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)

        # Indices for walk_env_reward (shared term).
        def _qadr(name: str) -> int:
            return int(self.model.jnt_qposadr[self.model.joint(name).id])

        self._hip_flex_indices = (_qadr("hip_flexion_l"), _qadr("hip_flexion_r"))
        self._hip_angle_indices = (
            _qadr("hip_adduction_l"),
            _qadr("hip_adduction_r"),
            _qadr("hip_rotation_l"),
            _qadr("hip_rotation_r"),
        )
        obs = self._obs_dict_to_vec(self._get_obs_dict(self._accessor))
        self.observation_space = gym.spaces.Box(
            -10.0 * np.ones(obs.size, dtype=np.float32),
            10.0 * np.ones(obs.size, dtype=np.float32),
            dtype=np.float32,
        )
        act_low = (
            np.zeros(self.model.nu, dtype=np.float32)
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
            # Actions are already in [0, 1]; use as muscle activations (no sigmoid).
            pass
        elif self.normalize_act and self.model.nu > 0:
            cr = self.model.actuator_ctrlrange
            ctrl = cr[:, 0] + ctrl * (cr[:, 1] - cr[:, 0])
        if self.muscle_condition == "fatigue":
            ctrl[self._muscle_act_ind], _, _ = self.muscle_fatigue.compute_act(
                ctrl[self._muscle_act_ind]
            )
        elif self.muscle_condition == "reafferentation":
            ctrl[self.EPLpos] = ctrl[self.EIPpos].copy()
            ctrl[self.EIPpos] = 0.0
        self.data.ctrl[:] = ctrl

    def _get_com(self) -> np.ndarray:
        mass = np.expand_dims(self.model.body_mass, -1)
        return np.sum(mass * self.data.xipos, 0) / np.sum(mass)

    def _get_com_velocity(self) -> np.ndarray:
        mass = np.expand_dims(self.model.body_mass, -1)
        cvel = -self.data.cvel
        return (np.sum(mass * cvel, 0) / np.sum(mass))[3:5]

    def _get_height(self) -> float:
        return float(self._get_com()[2])

    def _get_torso_angle(self) -> np.ndarray:
        body_id = self.model.body("torso").id
        return self.data.xquat[body_id].copy()

    def _get_feet_heights(self) -> np.ndarray:
        foot_id_l = self.model.body("talus_l").id
        foot_id_r = self.model.body("talus_r").id
        return np.array(
            [
                float(self.data.xpos[foot_id_l][2]),
                float(self.data.xpos[foot_id_r][2]),
            ]
        )

    def _get_feet_relative_position(self) -> np.ndarray:
        foot_id_l = self.model.body("talus_l").id
        foot_id_r = self.model.body("talus_r").id
        pelvis = self.model.body("pelvis").id
        return np.array(
            [
                self.data.xpos[foot_id_l] - self.data.xpos[pelvis],
                self.data.xpos[foot_id_r] - self.data.xpos[pelvis],
            ]
        )

    def _get_angle(self, names: list[str]) -> np.ndarray:
        return np.array(
            [
                float(self.data.qpos[self.model.jnt_qposadr[self.model.joint(n).id]])
                for n in names
            ]
        )

    def _get_vel_reward(self) -> float:
        vel = self._get_com_velocity()
        return float(
            np.exp(-np.square(self.target_y_vel - vel[1]))
            + np.exp(-np.square(self.target_x_vel - vel[0]))
        )

    def _get_cyclic_rew(self) -> float:
        phase_var = (self.steps / self.hip_period) % 1
        des_angles = np.array(
            [
                0.8 * np.cos(phase_var * 2 * np.pi + np.pi),
                0.8 * np.cos(phase_var * 2 * np.pi),
            ],
            dtype=np.float32,
        )
        angles = self._get_angle(["hip_flexion_l", "hip_flexion_r"])
        return float(np.linalg.norm(des_angles - angles))

    def _get_ref_rotation_rew(self) -> float:
        target_rot = (
            self.target_rot if self.target_rot is not None else self._init_qpos[3:7]
        )
        return float(
            np.exp(-np.linalg.norm(5.0 * (self.data.qpos[3:7].copy() - target_rot)))
        )

    def _get_joint_angle_rew(self, joint_names: list[str]) -> float:
        mag = float(np.mean(np.abs(self._get_angle(joint_names))))
        return float(np.exp(-5 * mag))

    def _get_done(self) -> int:
        if self._get_height() < self.min_height:
            return 1
        if self._get_rot_condition():
            return 1
        return 0

    def _get_rot_condition(self) -> int:
        quat = self.data.qpos[3:7].copy()
        return (
            1 if np.abs((quat2mat(quat) @ np.array([1, 0, 0]))[0]) > self.max_rot else 0
        )

    def _muscle_lengths(self) -> np.ndarray:
        return self.data.actuator_length.copy()

    def _muscle_forces(self) -> np.ndarray:
        return np.clip(self.data.actuator_force / 1000, -100, 100)

    def _muscle_velocities(self) -> np.ndarray:
        return np.clip(self.data.actuator_velocity, -100, 100)

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        t = accessor.time()
        qpos = accessor.joint_pos()
        qvel = accessor.joint_vel() * accessor.dt()
        phase_var = (self.steps / self.hip_period) % 1
        obs: dict[str, np.ndarray] = {
            "t": np.array([t]),
            "time": np.array([t]),
            "qpos_without_xy": qpos[2:].copy(),
            "qvel": qvel.copy(),
            "com_vel": np.array([self._get_com_velocity().copy()]),
            "torso_angle": np.array([self._get_torso_angle().copy()]),
            "feet_heights": self._get_feet_heights().copy(),
            "height": np.array([self._get_height()]),
            "feet_rel_positions": self._get_feet_relative_position().copy(),
            "phase_var": np.array([phase_var], dtype=np.float32),
            "muscle_length": self._muscle_lengths(),
            "muscle_velocity": self._muscle_velocities(),
            "muscle_force": self._muscle_forces(),
        }
        if self.model.na > 0:
            obs["act"] = accessor.muscle_act()
        return obs

    def _obs_dict_to_vec(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [np.atleast_1d(obs_dict[k]).ravel() for k in self.obs_keys if k in obs_dict]
        )

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        target_rot = (
            self.target_rot if self.target_rot is not None else self._init_qpos[3:7]
        )
        task_state = {
            "qpos": self.data.qpos.copy(),
            "height": float(obs_dict["height"][0]),
            "com_vel": np.asarray(obs_dict["com_vel"][0]),
            "phase_var": obs_dict["phase_var"],
        }
        term_rwd = walk_env_reward(
            self._accessor,
            task_state,
            hip_flex_indices=self._hip_flex_indices,
            hip_angle_indices=self._hip_angle_indices,
            target_rot=target_rot,
            target_vel=(self.target_x_vel, self.target_y_vel),
            min_height=self.min_height,
            max_rot=self.max_rot,
        )
        vel_reward = float(term_rwd["vel_reward"])
        act_mag = (
            float(np.linalg.norm(obs_dict.get("act", np.zeros(1)))) / self.model.na
            if self.model.na != 0
            else 0.0
        )
        rwd_dict = collections.OrderedDict(
            (
                ("vel_reward", vel_reward),
                ("cyclic_hip", float(term_rwd["cyclic_hip"])),
                ("ref_rot", float(term_rwd["ref_rot"])),
                ("joint_angle_rew", float(term_rwd["joint_angle_rew"])),
                ("act_mag", act_mag),
                ("sparse", vel_reward),
                ("solved", vel_reward >= 1.0),
                ("done", self._get_done()),
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

    def _get_randomized_initial_state(self) -> tuple[np.ndarray, np.ndarray]:
        if self.np_random.uniform() < 0.5:
            qpos = self.model.key_qpos[2].copy()
            qvel = self.model.key_qvel[2].copy()
        else:
            qpos = self.model.key_qpos[3].copy()
            qvel = self.model.key_qvel[3].copy()
        rot_state = qpos[3:7].copy()
        height = qpos[2]
        qpos = qpos + self.np_random.normal(0, 0.02, size=qpos.shape)
        qpos[3:7] = rot_state
        qpos[2] = height
        return qpos, qvel

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        return {}

    def step(
        self, action: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data, self.frame_skip)
        mujoco.mj_kinematics(self.model, self.data)
        self.steps += 1
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
        self.steps = 0
        mujoco.mj_resetData(self.model, self.data)
        if self.reset_type == "random":
            qpos, qvel = self._get_randomized_initial_state()
        elif self.reset_type == "init":
            qpos = self.model.key_qpos[2].copy()
            qvel = self.model.key_qvel[2].copy()
        else:
            qpos = self._init_qpos.copy()
            qvel = self._init_qvel.copy()
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        self._task_state = self.reset_task(self.np_random)
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


class LegTerrainEnvV0(LegWalkEnvV0):
    """Terrain locomotion (rough/hilly/stairs); extends LegWalkEnvV0.

    Migrated from walk_v0.TerrainEnvV0 (BaseV0). Same obs/reward + knee done.
    """

    def __init__(
        self,
        model_path: str,
        obsd_model_path: str | None = None,
        seed: int | None = None,
        terrain: str = "rough",
        variant: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_path,
            obsd_model_path=obsd_model_path,
            seed=seed,
            **kwargs,
        )
        self.terrain = terrain
        self.variant = variant

    def _get_done(self) -> int:
        if self._get_height() < self.min_height:
            return 1
        if self._get_rot_condition():
            return 1
        if self._get_knee_condition():
            return 1
        return 0

    def _get_knee_condition(self) -> int:
        feet_heights = self._get_feet_heights()
        com_height = self._get_height()
        return 1 if com_height - float(np.mean(feet_heights)) < 0.61 else 0

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        gym.Env.reset(self, seed=seed)
        if self.muscle_condition == "fatigue":
            self.muscle_fatigue.reset(
                fatigue_reset_vec=self.fatigue_reset_vec,
                fatigue_reset_random=self.fatigue_reset_random,
            )
        self.steps = 0
        rng = self.np_random
        if self.terrain == "rough":
            rough = rng.uniform(low=-0.5, high=0.5, size=(10000,))
            normalized_data = (rough - np.min(rough)) / (np.max(rough) - np.min(rough))
            scalar, offset = 0.08, 0.02
            self.model.hfield_data[:] = normalized_data * scalar - offset
        elif self.terrain == "hilly":
            flat_length, frequency = 3000, 3
            scalar = (
                0.63
                if self.variant == "fixed"
                else float(rng.uniform(low=0.53, high=0.73))
            )
            combined_data = np.concatenate(
                (
                    -2 * np.ones(flat_length),
                    -2
                    + 0.5
                    * (
                        np.sin(
                            np.linspace(0, frequency * np.pi, int(1e4 - flat_length))
                            + np.pi / 2
                        )
                        - 1
                    ),
                )
            )
            normalized_data = (combined_data - combined_data.min()) / (
                combined_data.max() - combined_data.min()
            )
            self.model.hfield_data[:] = np.flip(
                normalized_data.reshape(100, 100) * scalar, [0, 1]
            ).reshape(10000)
        elif self.terrain == "stairs":
            num_stairs = 12
            stair_height = 0.1
            flat = 5200 - (1e4 - 5200) % num_stairs
            stairs_width = (1e4 - flat) // num_stairs
            scalar = (
                2.5
                if self.variant == "fixed"
                else float(rng.uniform(low=1.5, high=3.5))
            )
            stair_parts = [
                np.full((int(stairs_width // 100), 100), -2 + stair_height * j)
                for j in range(num_stairs)
            ]
            new_terrain_data = np.concatenate(
                [np.full((int(flat // 100), 100), -2)] + stair_parts, axis=0
            )
            normalized_data = (new_terrain_data + 2) / (2 + stair_height * num_stairs)
            self.model.hfield_data[:] = np.flip(
                normalized_data.reshape(100, 100) * scalar, [0, 1]
            ).reshape(10000)
        tid = self.model.geom("terrain").id
        self.model.geom_rgba[tid, 3] = 1.0
        self.model.geom_pos[tid] = np.array([0.0, 0.0, 0.0])
        self.model.geom_contype[tid] = 1
        self.model.geom_conaffinity[tid] = 1

        mujoco.mj_resetData(self.model, self.data)
        if self.reset_type == "random":
            qpos, qvel = self._get_randomized_initial_state()
        elif self.reset_type == "init":
            qpos = self.model.key_qpos[2].copy()
            qvel = self.model.key_qvel[2].copy()
        else:
            qpos = self._init_qpos.copy()
            qvel = self._init_qvel.copy()
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        self._task_state = self.reset_task(self.np_random)
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs = self._obs_dict_to_vec(self._get_obs_dict(self._accessor))
        obs = np.clip(
            np.asarray(obs, dtype=np.float32),
            self.observation_space.low,
            self.observation_space.high,
        )
        return obs, {}
