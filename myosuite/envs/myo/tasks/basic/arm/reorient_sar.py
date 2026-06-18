# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""SAR reorient environments."""

from __future__ import annotations

import collections
from typing import Any

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium.utils import EzPickle

from myosuite.core.model_builder import ModelBuilder, build_from_recipe
from myosuite.envs.gymnasium_env import CpuEnvAccessor, MyoGymnasiumEnv
from myosuite.envs.myo.tasks.basic.arm.reorient_sar_geometries import (
    sample_geometry_8,
    sample_geometry_100,
    sample_geometry_id,
    sample_geometry_ood,
)
from myosuite.physics.quat_math import euler2quat
from myosuite.physics.quat_math import calculate_cosine


class ReorientSAREnvV0(MyoGymnasiumEnv, EzPickle):
    """Base SAR reorient env: object pose/rotation alignment.

    Migrated from reorient_sar_v0.ProprioceptiveEnvV0. Same obs/reward; no obsd.
    Subclasses override _apply_episode_geometry() to set object/target for the variant.
    """

    DEFAULT_OBS_KEYS = [
        "hand_jnt",
        "obj_pos",
        "obj_vel",
        "obj_rot",
        "obj_des_rot",
        "obj_err_pos",
        "obj_err_rot",
        "mlen",
        "mvel",
        "mforce",
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pos_align": 1.0,
        "rot_align": 1.0,
        "act_reg": 5.0,
        "drop": 5.0,
        "bonus": 10.0,
    }

    def __init__(
        self,
        model_path: str = "",
        obsd_model_path: str | None = None,
        seed: int | None = None,
        normalize_act: bool = True,
        frame_skip: int = 5,
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
            normalize_act=normalize_act,
            frame_skip=frame_skip,
            **kwargs,
        )
        model_recipe = kwargs.pop("model_recipe", None)
        if model_recipe is not None:
            self.model, self._mj_spec = build_from_recipe(model_recipe)
            self._name_sfx = "_r"
        else:
            self.model, self._mj_spec = ModelBuilder.from_xml_file(model_path).build()
            self._name_sfx = ""
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * frame_skip)
        self.normalize_act = normalize_act

        sfx = self._name_sfx
        self.target_obj_bid = self.model.body("target").id
        self.S_grasp_sid = self.model.site(f"S_grasp{sfx}").id
        self.obj_bid = self.model.body("Object").id
        self.eps_ball_sid = self.model.site("eps_ball").id
        self.success_indicator_sid = self.model.site("success").id
        self.obj_gid = self.model.geom("obj").id
        self.tar_gid = self.model.geom("target").id
        self.obj_t_gid = self.model.geom("top").id
        self.obj_b_gid = self.model.geom("bot").id
        self.tar_t_gid = self.model.geom("t_top").id
        self.tar_b_gid = self.model.geom("t_bot").id

        self.pen_length = float(
            np.linalg.norm(
                self.model.geom_pos[self.obj_t_gid]
                - self.model.geom_pos[self.obj_b_gid]
            )
        )
        self.tar_length = float(
            np.linalg.norm(
                self.model.geom_pos[self.tar_t_gid]
                - self.model.geom_pos[self.tar_b_gid]
            )
        )
        self.model.body_mass[self.obj_bid] *= 1.25

        self.obs_keys = list(self.DEFAULT_OBS_KEYS)
        if self.model.na > 0 and "act" not in self.obs_keys:
            self.obs_keys.append("act")
        self.rwd_keys_wt = dict(self.DEFAULT_RWD_KEYS_AND_WEIGHTS)

        mujoco.mj_resetData(self.model, self.data)
        self._init_qpos = self.data.qpos.copy()
        self._init_qpos[:-6] *= 0
        self._init_qpos[0] = -1.5
        self._init_qvel = np.zeros(self.model.nv, dtype=np.float64)

        gym.Env.reset(self, seed=seed)
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs = self._obs_dict_to_vec(self._get_obs_dict(self._accessor))
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

    def _apply_episode_geometry(self, rng: np.random.Generator) -> None:
        """Set object and target geometry for this episode. Override in subclasses."""
        raise NotImplementedError

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        model, data = accessor.model, accessor.data
        obj_pos = data.xpos[self.obj_bid].copy()
        obj_des_pos = data.site_xpos[self.eps_ball_sid].ravel()
        obj_vel = data.qvel[-6:].copy() * self._ctrl_dt
        obj_rot = (
            data.geom_xpos[self.obj_t_gid] - data.geom_xpos[self.obj_b_gid]
        ) / self.pen_length
        obj_des_rot = (
            data.geom_xpos[self.tar_t_gid] - data.geom_xpos[self.tar_b_gid]
        ) / self.tar_length
        out = {
            "hand_jnt": data.qpos[:-6].copy(),
            "obj_pos": obj_pos,
            "obj_vel": obj_vel,
            "obj_rot": obj_rot,
            "obj_des_rot": obj_des_rot,
            "obj_err_pos": obj_pos - obj_des_pos,
            "obj_err_rot": obj_rot - obj_des_rot,
            "mlen": data.actuator_length.copy(),
            "mvel": data.actuator_velocity.copy(),
            "mforce": data.actuator_force.copy(),
        }
        if model.na > 0:
            out["act"] = data.act.copy()
        return out

    def get_reward_dict(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        pos_err = obs_dict["obj_err_pos"]
        pos_align = np.linalg.norm(pos_err, axis=-1)
        rot_align = calculate_cosine(
            np.asarray(obs_dict["obj_rot"]),
            np.asarray(obs_dict["obj_des_rot"]),
        )
        dropped = pos_align > 0.075
        act = obs_dict.get("act")
        act_mag = (
            float(np.linalg.norm(act)) / self.model.na
            if act is not None and self.model.na != 0
            else 0.0
        )
        rwd_dict = collections.OrderedDict(
            (
                ("pos_align", -1.0 * pos_align),
                ("rot_align", rot_align),
                ("act_reg", -1.0 * act_mag),
                ("drop", -1.0 * float(dropped)),
                (
                    "bonus",
                    1.0 * (rot_align > 0.9) * (pos_align < 0.075)
                    + 5.0 * (rot_align > 0.95) * (pos_align < 0.075),
                ),
                ("sparse", -1.0 * pos_align + rot_align),
                ("solved", (rot_align > 0.95) * (~dropped)),
                ("done", bool(dropped)),
            )
        )
        rwd_dict["dense"] = np.sum(
            [
                self.rwd_keys_wt.get(k, 0) * rwd_dict[k]
                for k in self.rwd_keys_wt
                if k in rwd_dict
            ],
            axis=0,
        )
        if list(self.model.site_rgba[self.success_indicator_sid, :2]) != [0.0, 2.0]:
            self.model.site_rgba[self.success_indicator_sid, :2] = (
                np.array([0.0, 2.0]) if rwd_dict["solved"] else np.array([2.0, 0.0])
            )
        return rwd_dict

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        return {}

    def _obs_dict_to_vec(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Flatten only the keys in obs_keys, in obs_keys order."""
        return np.concatenate(
            [np.atleast_1d(obs_dict[k]).ravel() for k in self.obs_keys if k in obs_dict]
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        gym.Env.reset(self, seed=seed)
        self._apply_episode_geometry(self.np_random)
        self.model.site_rgba[self.success_indicator_sid, :2] = np.array([2.0, 0.0])
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel
        mujoco.mj_forward(self.model, self.data)
        self._task_state = self.reset_task(self.np_random)
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs_dict = self._get_obs_dict(self._accessor)
        obs = self._obs_dict_to_vec(obs_dict)
        obs = self._ensure_obs_gymnasium_compliant(obs)
        return obs, {}


class Geometries8EnvV0Gymnasium(ReorientSAREnvV0):
    """8-object SAR reorient. Migrated from reorient_sar_v0.Geometries8EnvV0."""

    def _apply_episode_geometry(self, rng: np.random.Generator) -> None:
        geom_type, size, color, top_pos, bot_pos, desired_euler = sample_geometry_8(rng)
        self.model.geom_size[self.obj_gid] = size
        self.model.geom_type[self.obj_gid] = geom_type
        self.model.geom_rgba[self.obj_gid] = color
        self.model.geom_pos[self.obj_t_gid] = top_pos
        self.model.geom_pos[self.obj_b_gid] = bot_pos
        self.model.body_mass[self.obj_bid] = 1.2
        self.model.geom_size[self.tar_gid] = size
        self.model.geom_type[self.tar_gid] = geom_type
        self.model.geom_rgba[self.tar_gid] = color
        self.model.geom_pos[self.tar_t_gid] = top_pos
        self.model.geom_pos[self.tar_b_gid] = bot_pos
        self.model.body_quat[self.target_obj_bid] = euler2quat(desired_euler)
        self.pen_length = float(
            np.linalg.norm(
                self.model.geom_pos[self.obj_t_gid]
                - self.model.geom_pos[self.obj_b_gid]
            )
        )
        self.tar_length = float(
            np.linalg.norm(
                self.model.geom_pos[self.tar_t_gid]
                - self.model.geom_pos[self.tar_b_gid]
            )
        )


class InDistribution(ReorientSAREnvV0):
    """SAR reorient eval, in-distribution geometries. Migrated from reorient_sar_v0.InDistribution."""

    def _apply_episode_geometry(self, rng: np.random.Generator) -> None:
        geom_type, size, color, top_pos, bot_pos, desired_euler = sample_geometry_id(
            rng
        )
        self.model.geom_size[self.obj_gid] = size
        self.model.geom_type[self.obj_gid] = geom_type
        self.model.geom_rgba[self.obj_gid] = color
        self.model.geom_pos[self.obj_t_gid] = top_pos
        self.model.geom_pos[self.obj_b_gid] = bot_pos
        self.model.body_mass[self.obj_bid] = 1.2
        self.model.geom_size[self.tar_gid] = size
        self.model.geom_type[self.tar_gid] = geom_type
        self.model.geom_rgba[self.tar_gid] = color
        self.model.geom_pos[self.tar_t_gid] = top_pos
        self.model.geom_pos[self.tar_b_gid] = bot_pos
        self.model.geom_condim[self.obj_gid] = 3
        self.model.body_quat[self.target_obj_bid] = euler2quat(desired_euler)
        self.pen_length = float(
            np.linalg.norm(
                self.model.geom_pos[self.obj_t_gid]
                - self.model.geom_pos[self.obj_b_gid]
            )
        )
        self.tar_length = float(
            np.linalg.norm(
                self.model.geom_pos[self.tar_t_gid]
                - self.model.geom_pos[self.tar_b_gid]
            )
        )


class OutofDistribution(ReorientSAREnvV0):
    """SAR reorient eval, out-of-distribution geometries. Migrated from reorient_sar_v0.OutofDistribution."""

    def _apply_episode_geometry(self, rng: np.random.Generator) -> None:
        geom_type, size, color, top_pos, bot_pos, desired_euler = sample_geometry_ood(
            rng
        )
        self.model.geom_size[self.obj_gid] = size
        self.model.geom_type[self.obj_gid] = geom_type
        self.model.geom_rgba[self.obj_gid] = color
        self.model.geom_pos[self.obj_t_gid] = top_pos
        self.model.geom_pos[self.obj_b_gid] = bot_pos
        self.model.body_mass[self.obj_bid] = 1.2
        self.model.geom_size[self.tar_gid] = size
        self.model.geom_type[self.tar_gid] = geom_type
        self.model.geom_rgba[self.tar_gid] = color
        self.model.geom_pos[self.tar_t_gid] = top_pos
        self.model.geom_pos[self.tar_b_gid] = bot_pos
        self.model.geom_condim[self.obj_gid] = 3
        self.model.body_quat[self.target_obj_bid] = euler2quat(desired_euler)
        self.pen_length = float(
            np.linalg.norm(
                self.model.geom_pos[self.obj_t_gid]
                - self.model.geom_pos[self.obj_b_gid]
            )
        )
        self.tar_length = float(
            np.linalg.norm(
                self.model.geom_pos[self.tar_t_gid]
                - self.model.geom_pos[self.tar_b_gid]
            )
        )


class Geometries100EnvV0Gymnasium(ReorientSAREnvV0):
    """100-object SAR reorient. Migrated from reorient_sar_v0.Geometries100EnvV0."""

    def _apply_episode_geometry(self, rng: np.random.Generator) -> None:
        geom_type, size, color, top_pos, bot_pos, desired_euler = sample_geometry_100(
            rng
        )
        self.model.geom_size[self.obj_gid] = size
        self.model.geom_type[self.obj_gid] = geom_type
        self.model.geom_rgba[self.obj_gid] = color
        self.model.geom_pos[self.obj_t_gid] = top_pos
        self.model.geom_pos[self.obj_b_gid] = bot_pos
        self.model.body_mass[self.obj_bid] = 1.2
        self.model.geom_size[self.tar_gid] = size
        self.model.geom_type[self.tar_gid] = geom_type
        self.model.geom_rgba[self.tar_gid] = color
        self.model.geom_pos[self.tar_t_gid] = top_pos
        self.model.geom_pos[self.tar_b_gid] = bot_pos
        self.model.geom_condim[self.obj_gid] = 3
        self.model.body_quat[self.target_obj_bid] = euler2quat(desired_euler)
        self.pen_length = float(
            np.linalg.norm(
                self.model.geom_pos[self.obj_t_gid]
                - self.model.geom_pos[self.obj_b_gid]
            )
        )
        self.tar_length = float(
            np.linalg.norm(
                self.model.geom_pos[self.tar_t_gid]
                - self.model.geom_pos[self.tar_b_gid]
            )
        )
