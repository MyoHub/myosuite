# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Bimanual handoff task with Gymnasium API."""

from __future__ import annotations

import collections
import enum
from typing import Any

import gymnasium as gym
from gymnasium.utils import EzPickle
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from myosuite.core.model_builder import ModelBuilder
from myosuite.envs.gymnasium_env import CpuEnvAccessor, MyoGymnasiumEnv
from myosuite.envs.myo.tasks.challenge.challenge_common import MuscleActionMixin
from myosuite.terms.base_action import sigmoid_muscle_activation
from myosuite.utils import seed_envs
from myosuite.physics.quat_math import mat2euler

CONTACT_TRAJ_MIN_LENGTH = 100
GOAL_CONTACT = 10
MAX_TIME = 10.0


class BimanualEnv(MuscleActionMixin, MyoGymnasiumEnv, EzPickle):
    """Native Gymnasium implementation for the bimanual challenge task."""

    DEFAULT_OBS_KEYS = [
        "time",
        "myohand_qpos",
        "myohand_qvel",
        "pros_hand_qpos",
        "pros_hand_qvel",
        "object_qpos",
        "object_qvel",
        "touching_body",
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach_dist": -0.1,
        "act": 0,
        "fin_dis": -0.5,
        "pass_err": -1,
    }

    def __init__(
        self,
        model_path: str,
        obsd_model_path: str | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        frame_skip = int(kwargs.get("frame_skip", 10))
        normalize_act = bool(kwargs.get("normalize_act", True))
        MyoGymnasiumEnv.__init__(
            self, frame_skip=frame_skip, render_mode=kwargs.get("render_mode")
        )
        EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        self.model, self._mj_spec = ModelBuilder.from_xml_file(model_path).build()
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * frame_skip)
        self.input_seed = seed
        self.np_random, _ = seed_envs(seed)

        self.normalize_act = normalize_act
        self.task_choice = kwargs.get("task_choice", "fixed")
        self.proximity_th = kwargs.get("proximity_th", 0.17)
        self.start_center = np.array(kwargs.get("start_center", [-0.4, -0.25, 1.05]))
        self.goal_center = np.array(kwargs.get("goal_center", [0.4, -0.25, 1.05]))
        self.start_shifts = np.array(kwargs.get("start_shifts", [0.055, 0.055, 0]))
        self.goal_shifts = np.array(kwargs.get("goal_shifts", [0.098, 0.098, 0]))
        self.PILLAR_HEIGHT = 1.09

        self.id_info = IdInfo(self.model)
        self.start_bid = self.id_info.start_id
        self.goal_bid = self.id_info.goal_id
        self.obj_bid = self.id_info.manip_body_id
        self.obj_sid = self.model.site("touch_site").id
        self.obj_gid = self.model.body(self.obj_bid).geomadr + 1
        self.obj_mid = next(
            i for i in range(self.model.nmesh) if "box" in self.model.mesh(i).name
        )
        self.init_obj_z = self.data.site_xpos[self.obj_sid][-1]
        self.target_z = 0.2
        self.palm_sid = self.model.site("S_grasp").id
        self.init_palm_z = self.data.site_xpos[self.palm_sid][-1]
        self.fin0 = self.model.site("THtip").id
        self.fin1 = self.model.site("IFtip").id
        self.fin2 = self.model.site("MFtip").id
        self.fin3 = self.model.site("RFtip").id
        self.fin4 = self.model.site("LFtip").id
        self.rpalm1_sid = self.model.site("Lgrasp").id
        self.rpalm2_sid = self.model.site("Lgrasp").id
        self.start_pos = self.start_center
        self.goal_pos = self.goal_center
        self.model.body_pos[self.start_bid] = self.start_pos
        self.model.body_pos[self.goal_bid] = self.goal_pos

        self.over_max = False
        self.max_force = 0
        self.goal_touch = 0
        self.TARGET_GOAL_TOUCH = GOAL_CONTACT
        self.touch_history: list[set] = []

        obj_mass_change = kwargs.get("obj_mass_change", None)
        obj_scale_change = kwargs.get("obj_scale_change", None)
        obj_friction_change = kwargs.get("obj_friction_change", None)
        self.obj_mass_range = (
            {
                "low": self.model.body_mass[self.obj_bid] + obj_mass_change[0],
                "high": self.model.body_mass[self.obj_bid] + obj_mass_change[1],
            }
            if obj_mass_change
            else None
        )
        self.obj_scale_range = (
            {"low": -np.array(obj_scale_change), "high": obj_scale_change}
            if obj_scale_change
            else None
        )
        self.obj_friction_range = (
            {
                "low": self.model.geom_friction[self.obj_gid] - obj_friction_change,
                "high": self.model.geom_friction[self.obj_gid] + obj_friction_change,
            }
            if obj_friction_change
            else None
        )

        if obj_scale_change:
            self._center_box_mesh()

        self.rwd_keys_wt = kwargs.get(
            "weighted_reward_keys", self.DEFAULT_RWD_KEYS_AND_WEIGHTS
        )
        self.obs_keys = list(kwargs.get("obs_keys", self.DEFAULT_OBS_KEYS))
        if self.model.na > 0 and "act" not in self.obs_keys:
            self.obs_keys.append("act")

        self._muscle_act_ind = self.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self.muscle_condition = kwargs.get("muscle_condition", "")
        self.init_muscle_condition()

        self._init_qpos = (
            self.model.key_qpos[2].copy()
            if self.model.nkey > 2
            else self.data.qpos.copy()
        )
        self._init_qvel = self.data.qvel.copy()
        self.model.body_pos[self.start_bid] = self.start_pos
        self.model.body_pos[self.goal_bid] = self.goal_pos
        mujoco.mj_forward(self.model, self.data)
        self._default_mocap_pos = self.data.mocap_pos.copy()
        self._default_mocap_quat = self.data.mocap_quat.copy()
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs = self._obs_dict_to_vec(self._get_obs_dict(self._accessor))
        self.observation_space = gym.spaces.Box(
            -10.0 * np.ones(obs.size, dtype=np.float32),
            10.0 * np.ones(obs.size, dtype=np.float32),
            dtype=np.float32,
        )
        act_low = (
            -np.ones(self.model.nu, dtype=np.float32)
            if self.normalize_act
            else self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        )
        act_high = (
            np.ones(self.model.nu, dtype=np.float32)
            if self.normalize_act
            else self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        )
        self.action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)
        self.ignore_first_scale = bool(obj_scale_change)

    def _center_box_mesh(self) -> None:
        self.obj_size0 = self.model.geom_size[self.obj_gid].copy()
        _vertadr = int(self.model.mesh(self.obj_mid).vertadr)
        _vertnum = int(self.model.mesh(0).vertnum)
        self.obj_vert_addr = np.arange(_vertadr, _vertadr + _vertnum)
        q = self.model.geom(self.obj_gid - 1).quat
        r = R.from_quat([q[1], q[2], q[3], q[0]])
        self.model.mesh_vert[self.obj_vert_addr] = r.apply(
            self.model.mesh_vert[self.obj_vert_addr]
        )
        self.model.mesh_normal[self.obj_vert_addr] = r.apply(
            self.model.mesh_normal[self.obj_vert_addr]
        )
        self.model.geom(self.obj_gid - 1).quat = [1, 0, 0, 0]
        self.model.mesh_vert[self.obj_vert_addr] += (
            self.model.geom(self.obj_gid - 1).pos - self.model.geom(self.obj_gid).pos
        )[None, :]
        self.model.geom(self.obj_gid - 1).pos = self.model.geom(self.obj_gid).pos
        self.mesh_vert0 = self.model.mesh_vert[self.obj_vert_addr].copy()

    def _obj_label_to_obs(self, touching_body: set) -> np.ndarray:
        obs_vec = np.array([0, 0, 0, 0, 0])
        for i in touching_body:
            if i == ObjLabels.MYO:
                obs_vec[0] += 1
            elif i == ObjLabels.PROSTH:
                obs_vec[1] += 1
            elif i == ObjLabels.START:
                obs_vec[2] += 1
            elif i == ObjLabels.GOAL:
                obs_vec[3] += 1
            else:
                obs_vec[4] += 1
        return obs_vec

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        mj_model = self.model
        mj_data = self.data
        obs_dict: dict[str, np.ndarray] = {}
        obs_dict["time"] = np.array([mj_data.time])
        obs_dict["qp"] = mj_data.qpos.copy()
        obs_dict["qv"] = mj_data.qvel.copy()
        obs_dict["myohand_qpos"] = mj_data.qpos[self.id_info.myo_joint_range].copy()
        obs_dict["myohand_qvel"] = mj_data.qvel[self.id_info.myo_dof_range].copy()
        obs_dict["pros_hand_qpos"] = mj_data.qpos[
            self.id_info.prosth_joint_range
        ].copy()
        obs_dict["pros_hand_qvel"] = mj_data.qvel[self.id_info.prosth_dof_range].copy()
        obs_dict["object_qpos"] = mj_data.qpos[self.id_info.manip_joint_range].copy()
        obs_dict["object_qvel"] = mj_data.qvel[self.id_info.manip_dof_range].copy()
        obs_dict["start_pos"] = self.start_pos
        obs_dict["goal_pos"] = self.goal_pos
        obs_dict["elbow_fle"] = mj_data.joint("elbow_flexion").qpos.copy()
        touching_objects = set(get_touching_objects(mj_model, mj_data, self.id_info))
        self.touch_history.append(touching_objects)
        current_force = mj_data.sensordata[0]
        if current_force > self.max_force:
            self.max_force = current_force
        obs_dict["max_force"] = np.array([self.max_force])
        obs_dict["touching_body"] = self._obj_label_to_obs(touching_objects)
        obs_dict["palm_pos"] = mj_data.site_xpos[self.palm_sid]
        obs_dict["fin0"] = mj_data.site_xpos[self.fin0]
        obs_dict["fin1"] = mj_data.site_xpos[self.fin1]
        obs_dict["fin2"] = mj_data.site_xpos[self.fin2]
        obs_dict["fin3"] = mj_data.site_xpos[self.fin3]
        obs_dict["fin4"] = mj_data.site_xpos[self.fin4]
        obs_dict["rpalm_pos"] = (
            mj_data.site_xpos[self.rpalm1_sid] + mj_data.site_xpos[self.rpalm2_sid]
        ) / 2
        obs_dict["mpl_ori"] = mat2euler(
            mj_data.site_xmat[self.rpalm1_sid].reshape(3, 3)
        )
        obs_dict["mpl_ori_err"] = obs_dict["mpl_ori"] - np.array([np.pi, 0, np.pi])
        obs_dict["obj_pos"] = mj_data.site_xpos[self.obj_sid]
        obs_dict["reach_err"] = obs_dict["palm_pos"] - obs_dict["obj_pos"]
        obs_dict["pass_err"] = obs_dict["rpalm_pos"] - obs_dict["obj_pos"]
        if self.model.na > 0:
            obs_dict["act"] = accessor.muscle_act()
        return obs_dict

    def _obs_dict_to_vec(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [np.atleast_1d(obs_dict[k]).ravel() for k in self.obs_keys if k in obs_dict]
        )

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        reach_dist = np.abs(np.linalg.norm(obs_dict["reach_err"], axis=-1))
        pass_dist = np.abs(np.linalg.norm(obs_dict["pass_err"], axis=-1))
        obj_pos = (
            obs_dict["obj_pos"][0][0]
            if obs_dict["obj_pos"].ndim == 3
            else obs_dict["obj_pos"]
        )
        palm_pos = (
            obs_dict["palm_pos"][0][0]
            if obs_dict["palm_pos"].ndim == 3
            else obs_dict["palm_pos"]
        )
        goal_pos = (
            obs_dict["goal_pos"][0][0]
            if np.ndim(obs_dict["goal_pos"]) == 3
            else obs_dict["goal_pos"]
        )
        goal_pos = np.concatenate((goal_pos[:2], np.array([self.PILLAR_HEIGHT])))
        lift_height = np.linalg.norm(
            np.array([[[obj_pos[-1], palm_pos[-1]]]])
            - np.array([[[self.init_obj_z, self.init_palm_z]]]),
            axis=-1,
        )
        lift_height = 5 * np.exp(-10 * (lift_height - self.target_z) ** 2) - 5
        act = (
            np.linalg.norm(obs_dict["act"], axis=-1) / self.model.na
            if self.model.na != 0 and "act" in obs_dict
            else 0
        )
        fin_keys = ["fin0", "fin1", "fin2", "fin3", "fin4"]
        fin_open = sum(
            np.linalg.norm(obs_dict[fin] - obs_dict["palm_pos"], axis=-1)
            for fin in fin_keys
        )
        fin_dis = sum(
            np.linalg.norm(obs_dict[fin] - obs_dict["obj_pos"], axis=-1)
            for fin in fin_keys
        )
        elbow_err = 5 * np.exp(-10 * (obs_dict["elbow_fle"][0] - 1.0) ** 2) - 5
        goal_dis = np.array([[np.abs(np.linalg.norm(obj_pos - goal_pos, axis=-1))]])
        touching_vec = (
            obs_dict["touching_body"][0][0]
            if obs_dict["touching_body"].ndim == 3
            else obs_dict["touching_body"]
        )
        if touching_vec[3] == 1:
            self.goal_touch += 1
        rwd_dict = collections.OrderedDict(
            (
                ("reach_dist", reach_dist + np.log(reach_dist + 1e-6)),
                ("act", act),
                ("fin_open", np.exp(-5 * fin_open)),
                ("fin_dis", fin_dis + np.log(fin_dis + 1e-6)),
                ("lift_bonus", elbow_err),
                ("lift_height", lift_height),
                ("pass_err", pass_dist + np.log(pass_dist + 1e-3)),
                ("sparse", 0),
                ("goal_dist", goal_dis),
                (
                    "solved",
                    goal_dis < self.proximity_th
                    and self.goal_touch >= self.TARGET_GOAL_TOUCH,
                ),
                ("done", self._get_done(obj_pos[-1])),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return rwd_dict

    def _get_done(self, z):
        if self.obs_dict["time"] > MAX_TIME:
            return 1
        if z < 0.3:
            self.obs_dict["time"] = MAX_TIME
            return 1
        if hasattr(self, "rwd_dict") and self.rwd_dict and self.rwd_dict["solved"]:
            return 1
        return 0

    def _process_controls(self, action: np.ndarray) -> np.ndarray:
        ctrl = np.asarray(action, dtype=np.float64).copy()
        ctrl = np.clip(ctrl, self.action_space.low, self.action_space.high)
        if self.normalize_act:
            robotic_act_ind = self.model.actuator_dyntype != mujoco.mjtDyn.mjDYN_MUSCLE
            ctrl[robotic_act_ind] = (
                np.mean(self.model.actuator_ctrlrange[robotic_act_ind], axis=-1)
                + ctrl[robotic_act_ind]
                * (
                    self.model.actuator_ctrlrange[robotic_act_ind, 1]
                    - self.model.actuator_ctrlrange[robotic_act_ind, 0]
                )
                / 2.0
            )
            if self.model.na > 0:
                ctrl[self._muscle_act_ind] = sigmoid_muscle_activation(
                    ctrl[self._muscle_act_ind], np
                )
        if self.muscle_condition == "fatigue":
            ctrl[self._muscle_act_ind], _, _ = self.muscle_fatigue.compute_act(
                ctrl[self._muscle_act_ind]
            )
        elif self.muscle_condition == "reafferentation":
            ctrl[self.epl_pos] = ctrl[self.eip_pos].copy()
            ctrl[self.eip_pos] = 0.0
        return ctrl

    def reset(self, seed: int | None = None, options: dict | None = None, **_kwargs):
        if seed is not None:
            self.input_seed = seed
            self.np_random, _ = seed_envs(seed)
        self.start_pos = self.start_center + self.start_shifts * (
            2 * self.np_random.random(3) - 1
        )
        self.goal_pos = self.goal_center + self.goal_shifts * (
            2 * self.np_random.random(3) - 1
        )
        self.model.body_pos[self.start_bid] = self.start_pos
        self.model.body_pos[self.goal_bid] = self.goal_pos
        self.touch_history = []
        self.over_max = False
        self.goal_touch = 0
        if self.obj_mass_range:
            self.model.body_mass[self.obj_bid] = self.np_random.uniform(
                **self.obj_mass_range
            )
        if self.obj_friction_range:
            self.model.geom_friction[self.obj_gid] = self.np_random.uniform(
                **self.obj_friction_range
            )
        if self.obj_scale_range and not self.ignore_first_scale:
            obj_scales = self.np_random.uniform(**self.obj_scale_range) + 1
            self.model.geom(self.obj_gid).size = self.obj_size0 * obj_scales
        else:
            self.ignore_first_scale = False
        mujoco.mj_forward(self.model, self.data)
        if self.model.nkey > 2:
            self._init_qpos = self.model.key_qpos[2].copy()
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel
        if self.model.nmocap > 0:
            self.data.mocap_pos[:] = self._default_mocap_pos
            self.data.mocap_quat[:] = self._default_mocap_quat
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        self.obs_dict = self._get_obs_dict(self._accessor)
        obs = self._obs_dict_to_vec(self.obs_dict)
        object_qpos_adr = self.model.body(self.obj_bid).jntadr[0]
        self.data.qpos[object_qpos_adr : object_qpos_adr + 3] = (
            self.start_pos + np.array([0, 0, 0.1])
        )
        self.init_obj_z = self.data.site_xpos[self.obj_sid][-1]
        self.init_palm_z = self.data.site_xpos[self.palm_sid][-1]
        obs = np.clip(np.asarray(obs, dtype=np.float32), -10.0, 10.0)
        return obs, {}

    def step(self, action: np.ndarray, **kwargs: Any):
        ctrl = self._process_controls(action)
        n_frames = int(self._ctrl_dt / self.model.opt.timestep)
        self.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        self.obs_dict = self._get_obs_dict(self._accessor)
        self.rwd_dict = self.get_reward_dict(self.obs_dict)
        obs = self._obs_dict_to_vec(self.obs_dict)
        obs = np.clip(np.asarray(obs, dtype=np.float32), -10.0, 10.0)
        reward = float(np.asarray(self.rwd_dict["dense"]).ravel()[0])
        terminated = bool(np.asarray(self.rwd_dict["done"]).ravel()[0])
        info = {
            "obs_dict": self.obs_dict,
            "rwd_dict": self.rwd_dict,
            "touch_history": self.touch_history,
        }
        return obs, reward, terminated, False, info

    def close(self) -> None:
        if hasattr(self, "_legacy_delegate"):
            self._legacy_delegate.close()
            return
        super().close()


class ObjLabels(enum.Enum):
    MYO = 0
    PROSTH = 1
    START = 2
    GOAL = 3
    ENV = 4


class ContactTrajIssue(enum.Enum):
    MYO_SHORT = 0
    PROSTH_SHORT = 1
    NO_GOAL = 2
    ENV_CONTACT = 3


def _is_mpl_body(name: str) -> bool:
    """Return True if the body name belongs to the MPL prosthetic arm chain."""
    if not name:
        return False
    return name.startswith(("torso_", "roll_", "upper_torso")) or (
        len(name) > 1 and name[0] in "RL" and name[1].islower()
    )


def _is_mpl_joint(name: str) -> bool:
    """Return True if the joint name belongs to the MPL prosthetic arm chain."""
    return len(name) > 1 and name[0] in "RL" and name[1].islower()


class IdInfo:
    def __init__(self, model):
        self.manip_body_id = model.body("manip_object").id
        _skip = {"world", "start", "goal", "manip_object"}
        myo_bodies = [
            model.body(i).id
            for i in range(model.nbody)
            if not _is_mpl_body(model.body(i).name) and model.body(i).name not in _skip
        ]
        self.myo_body_range = (min(myo_bodies), max(myo_bodies))
        prosth_bodies = [
            model.body(i).id
            for i in range(model.nbody)
            if _is_mpl_body(model.body(i).name)
        ]
        self.prosth_body_range = (min(prosth_bodies), max(prosth_bodies))
        self.myo_joint_range = np.concatenate(
            [
                model.joint(i).qposadr
                for i in range(model.njnt)
                if not _is_mpl_joint(model.joint(i).name)
                and model.joint(i).name != "manip_object/freejoint"
            ]
        )
        self.myo_dof_range = np.concatenate(
            [
                model.joint(i).dofadr
                for i in range(model.njnt)
                if not _is_mpl_joint(model.joint(i).name)
                and model.joint(i).name != "manip_object/freejoint"
            ]
        )
        self.prosth_joint_range = np.concatenate(
            [
                model.joint(i).qposadr
                for i in range(model.njnt)
                if _is_mpl_joint(model.joint(i).name)
            ]
        )
        self.prosth_dof_range = np.concatenate(
            [
                model.joint(i).dofadr
                for i in range(model.njnt)
                if _is_mpl_joint(model.joint(i).name)
            ]
        )
        _qposadr = int(model.joint("manip_object/freejoint").qposadr)
        _dofadr = int(model.joint("manip_object/freejoint").dofadr)
        self.manip_joint_range = np.arange(_qposadr, _qposadr + 7)
        self.manip_dof_range = np.arange(_dofadr, _dofadr + 6)
        self.start_id = model.body("start").id
        self.goal_id = model.body("goal").id


def get_touching_objects(model, data, id_info: IdInfo):
    for con in data.contact:
        if model.geom(con.geom1).bodyid == id_info.manip_body_id:
            yield body_id_to_label(model.geom(con.geom2).bodyid, id_info)
        elif model.geom(con.geom2).bodyid == id_info.manip_body_id:
            yield body_id_to_label(model.geom(con.geom1).bodyid, id_info)


def body_id_to_label(body_id, id_info: IdInfo):
    if id_info.myo_body_range[0] <= body_id <= id_info.myo_body_range[1]:
        return ObjLabels.MYO
    if id_info.prosth_body_range[0] <= body_id <= id_info.prosth_body_range[1]:
        return ObjLabels.PROSTH
    if body_id == id_info.start_id:
        return ObjLabels.START
    if body_id == id_info.goal_id:
        return ObjLabels.GOAL
    return ObjLabels.ENV
