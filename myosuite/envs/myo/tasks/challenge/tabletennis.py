# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""TableTennis challenge environment with Gymnasium API."""

# flake8: noqa
# pylint: disable=no-member,too-many-instance-attributes,attribute-defined-outside-init

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
from myosuite.envs.myo.assets._resolve import warn_torso_pip_calibration_divergence
from myosuite.terms.base_action import sigmoid_muscle_activation
from myosuite.utils.spec_processing import (
    recursive_immobilize,
    recursive_mirror,
    recursive_remove_contacts,
)

MAX_TIME = 3.0


class TableTennisEnv(MyoGymnasiumEnv, EzPickle):
    """Native rewrite target for myoChallenge TableTennis."""

    DEFAULT_OBS_KEYS = [
        "pelvis_pos",
        "body_qpos",
        "body_qvel",
        "ball_pos",
        "ball_vel",
        "paddle_pos",
        "paddle_vel",
        "paddle_ori",
        "reach_err",
        "touching_info",
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach_dist": 1,
        "palm_dist": 1,
        "paddle_quat": 2,
        "act_reg": 0.5,
        "torso_up": 2,
        "sparse": 100,
        "solved": 1000,
        "done": -10,
    }

    def __init__(
        self,
        model_path: str = "",
        obsd_model_path: str | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        frame_skip = int(kwargs.get("frame_skip", 10))
        MyoGymnasiumEnv.__init__(
            self, frame_skip=frame_skip, render_mode=kwargs.get("render_mode")
        )
        EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        model_recipe = kwargs.pop("model_recipe", None)
        if model_recipe is not None:
            # challenge_tabletennis (myosuite/core/model_recipes.py) already
            # composes the full torso+both-arms+legs body via myo_sim-native
            # composition (immobilized legs/left-arm, contact removal, root
            # calibration baked in) -- the legacy _preprocess_spec()/
            # preprocess_tabletennis_spec() mirror-and-immobilize dance is
            # unnecessary and would error against this spec's structure
            # (no mirrored-copy step to run).
            kwargs.pop("remove_body_collisions", None)
            kwargs.pop("add_left_arm", None)
            from myosuite.core.model_recipes import build_from_recipe

            self.model, self._mj_spec = build_from_recipe(model_recipe)
        else:
            preproc_kwargs = {
                "remove_body_collisions": kwargs.pop("remove_body_collisions", True),
                "add_left_arm": kwargs.pop("add_left_arm", True),
            }
            preproc_kw = preproc_kwargs  # captured by closure below

            def _preprocess(spec: mujoco.MjSpec) -> mujoco.MjSpec:
                return self._preprocess_spec(spec, **preproc_kw)

            warn_torso_pip_calibration_divergence()
            self.model, self._mj_spec = (
                ModelBuilder.from_xml_file(model_path)
                .apply_transform(_preprocess)
                .build()
            )
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * frame_skip)

        self.normalize_act = bool(kwargs.get("normalize_act", True))
        self.ball_xyz_range = kwargs.get("ball_xyz_range", None)
        self.ball_qvel = kwargs.get("ball_qvel", None)
        self.qpos_noise_range = kwargs.get("qpos_noise_range", None)
        self.paddle_mass_range = kwargs.get("paddle_mass_range", None)
        self.ball_friction_range = kwargs.get("ball_friction_range", None)
        self.rally_count = kwargs.get("rally_count", 1)
        self.cur_rally = 0
        self.contact_trajectory: list[set[PingpongContactLabels]] = []

        self.init_paddle_quat = R.from_euler(
            "xyz", np.array([-0.3, 1.57, 0]), degrees=False
        ).as_quat()[[3, 0, 1, 2]]

        self.id_info = IdInfo(self.model)
        self.ball_dofadr = self.model.body_dofadr[self.id_info.ball_bid]
        self.ball_posadr = self.model.joint("pingpong_freejoint").qposadr[0]
        self._muscle_act_ind = self.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE

        self.obs_keys = list(kwargs.get("obs_keys", self.DEFAULT_OBS_KEYS))
        self.rwd_keys_wt = kwargs.get(
            "weighted_reward_keys", self.DEFAULT_RWD_KEYS_AND_WEIGHTS
        )
        if self.model.na > 0 and "act" not in self.obs_keys:
            self.obs_keys.append("act")

        self._init_qpos = self.model.key_qpos[0].copy()
        self._init_qvel = self.data.qvel.copy()
        self.start_vel = np.array([[5.6, 1.6, 0.1]])
        self._init_qvel[self.ball_dofadr : self.ball_dofadr + 3] = self.start_vel

        gym.Env.reset(self, seed=seed)
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

    def _obs_dict_to_vec(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [np.atleast_1d(obs_dict[k]).ravel() for k in self.obs_keys if k in obs_dict]
        )

    def get_sensor_by_name(self, name: str) -> np.ndarray:
        sensor_id = self.model.sensor(name).id
        start = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        return self.data.sensordata[start : start + dim]

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        obs_dict: dict[str, np.ndarray] = {}
        obs_dict["time"] = np.array([self.data.time])
        obs_dict["pelvis_pos"] = self.data.site_xpos[self.model.site("pelvis").id]
        obs_dict["body_qpos"] = self.data.qpos[self.id_info.myo_joint_range].copy()
        obs_dict["body_qvel"] = self.data.qvel[self.id_info.myo_dof_range].copy()
        obs_dict["ball_pos"] = self.data.site_xpos[self.id_info.ball_sid]
        obs_dict["ball_vel"] = self.get_sensor_by_name("pingpong_vel_sensor")
        obs_dict["paddle_pos"] = self.data.site_xpos[self.id_info.paddle_sid]
        obs_dict["paddle_vel"] = self.get_sensor_by_name("paddle_vel_sensor")
        obs_dict["paddle_ori"] = self.data.xquat[self.id_info.paddle_bid]
        obs_dict["padde_ori_err"] = obs_dict["paddle_ori"] - self.init_paddle_quat
        obs_dict["reach_err"] = obs_dict["paddle_pos"] - obs_dict["ball_pos"]
        obs_dict["palm_pos"] = self.data.site_xpos[self.model.site("S_grasp").id]
        obs_dict["palm_err"] = obs_dict["palm_pos"] - obs_dict["paddle_pos"]
        touching_objects = set(
            get_ball_contact_labels(self.model, self.data, self.id_info)
        )
        self.contact_trajectory.append(touching_objects)
        obs_dict["touching_info"] = self._ball_label_to_obs(touching_objects)
        if self.model.na > 0:
            obs_dict["act"] = self.data.act.copy()
        return obs_dict

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        reach_dist = float(np.abs(np.linalg.norm(obs_dict["reach_err"], axis=-1)))
        palm_dist = float(np.abs(np.linalg.norm(obs_dict["palm_err"], axis=-1)))
        act_mag = (
            float(np.linalg.norm(obs_dict["act"], axis=-1)) / self.model.na
            if self.model.na != 0 and "act" in obs_dict
            else 0.0
        )
        ball_pos = obs_dict["ball_pos"]
        solved = evaluate_pingpong_trajectory(self.contact_trajectory) is None
        paddle_quat_err = float(np.linalg.norm(obs_dict["padde_ori_err"], axis=-1))
        torso_err = abs(
            self.data.qpos[
                self.model.jnt_qposadr[self.model.joint("flex_extension").id]
            ]
        )
        paddle_touch = obs_dict["touching_info"]
        rwd_dict = collections.OrderedDict(
            (
                ("reach_dist", np.exp(-1.0 * reach_dist)),
                ("palm_dist", np.exp(-5.0 * palm_dist)),
                ("paddle_quat", np.exp(-5 * paddle_quat_err)),
                ("torso_up", np.exp(-5 * torso_err)),
                ("act_reg", -1.0 * act_mag),
                ("sparse", paddle_touch[0] == 1),
                ("solved", np.array([[solved]])),
                ("done", np.array([[self._get_done(ball_pos[-1], solved)]])),
            )
        )
        rwd_dict["dense"] = sum(
            float(wt) * float(np.array(rwd_dict[key]).squeeze())
            for key, wt in self.rwd_keys_wt.items()
        )
        if rwd_dict["solved"]:
            self.cur_rally += 1
        if rwd_dict["solved"] and self.cur_rally < self.rally_count:
            rwd_dict["done"] = False
            rwd_dict["solved"] = False
            self.data.time = 0.0
            self.contact_trajectory = []
            self.relaunch_ball()
        return rwd_dict

    def _get_done(self, z: float, solved: bool) -> int:
        if self.obs_dict["time"] > MAX_TIME:
            return 1
        if z < 0.3:
            self.obs_dict["time"] = MAX_TIME
            return 1
        if solved:
            return 1
        if evaluate_pingpong_trajectory(self.contact_trajectory) in [0, 2, 3]:
            return 1
        return 0

    def _ball_label_to_obs(
        self, touching_body: set[PingpongContactLabels]
    ) -> np.ndarray:
        obs_vec = np.array([0, 0, 0, 0, 0, 0])
        for i in touching_body:
            if i == PingpongContactLabels.PADDLE:
                obs_vec[0] += 1
            elif i == PingpongContactLabels.OWN:
                obs_vec[1] += 1
            elif i == PingpongContactLabels.OPPONENT:
                obs_vec[2] += 1
            elif i == PingpongContactLabels.NET:
                obs_vec[3] += 1
            elif i == PingpongContactLabels.GROUND:
                obs_vec[4] += 1
            else:
                obs_vec[5] += 1
        return obs_vec

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
        if self.model.na > 0 and self.normalize_act:
            ctrl[self._muscle_act_ind] = sigmoid_muscle_activation(
                ctrl[self._muscle_act_ind], np
            )
        return ctrl

    def cal_ball_qvel(self, ball_qpos: np.ndarray) -> list[list[float]]:
        table_upper = [1.35, 0.70, 0.785]
        table_lower = [0.5, -0.60, 0.785]
        gravity = 9.81
        v_z = self.np_random.uniform(*(-0.1, 0.1))
        a = -0.5 * gravity
        b = v_z
        c = ball_qpos[2] - table_upper[2]
        discriminant = b**2 - 4 * a * c
        t = (-b - discriminant**0.5) / (2 * a)
        if discriminant < 0:
            raise ValueError("No real solution for pingpong launch velocity.")
        v_upper = [(table_upper[i] - ball_qpos[i]) / t for i in range(2)]
        v_lower = [(table_lower[i] - ball_qpos[i]) / t for i in range(2)]
        return [[v_upper[0], v_upper[1], v_z], [v_lower[0], v_lower[1], v_z]]

    def relaunch_ball(self) -> None:
        ball_pos = self._init_qpos[self.ball_posadr : self.ball_dofadr + 3]
        ball_vel = self._init_qvel[self.ball_dofadr : self.ball_dofadr + 6]
        if self.ball_xyz_range is not None:
            ball_pos = self.np_random.uniform(**self.ball_xyz_range)
            self.model.body_pos[self.id_info.ball_bid] = ball_pos
            self._init_qpos[self.ball_posadr : self.ball_posadr + 3] = ball_pos
        if self.ball_qvel:
            v_bounds = self.cal_ball_qvel(ball_pos)
            v_low, v_high = v_bounds[1], v_bounds[0]
            ball_vel[:3] = self.np_random.uniform(low=v_low, high=v_high)
            self._init_qvel[self.ball_dofadr : self.ball_dofadr + 3] = ball_vel[:3]
        self.data.qpos[self.ball_posadr : self.ball_posadr + 3] = ball_pos
        self.data.qvel[self.ball_dofadr : self.ball_dofadr + 6] = ball_vel

    def reset(self, seed: int | None = None, options: dict | None = None, **_kwargs):
        gym.Env.reset(self, seed=seed)
        self.contact_trajectory = []
        self._init_qpos[:] = self.model.key_qpos[0].copy()
        if self.paddle_mass_range:
            self.model.body_mass[self.id_info.paddle_bid] = self.np_random.uniform(
                *self.paddle_mass_range
            )
        if self.ball_friction_range:
            self.model.geom_friction[self.id_info.ball_gid] = self.np_random.uniform(
                **self.ball_friction_range
            )
        ball_pos = None
        if self.ball_xyz_range is not None:
            ball_pos = self.np_random.uniform(**self.ball_xyz_range)
            self.model.body_pos[self.id_info.ball_bid] = ball_pos
            self._init_qpos[self.ball_posadr : self.ball_posadr + 3] = ball_pos
        if self.qpos_noise_range is not None:
            joint_ranges = self.model.jnt_range[:, 1] - self.model.jnt_range[:, 0]
            noise_fraction = self.np_random.uniform(
                **self.qpos_noise_range, size=joint_ranges.shape
            )
            reset_qpos_local = self._init_qpos.copy()
            for j, adr in enumerate(self.model.jnt_qposadr[:-2]):
                reset_qpos_local[adr] += noise_fraction[j] * joint_ranges[j]
                reset_qpos_local[adr] = np.clip(
                    reset_qpos_local[adr],
                    self.model.jnt_range[j, 0],
                    self.model.jnt_range[j, 1],
                )
        else:
            reset_qpos_local = self._init_qpos.copy()
        if self.ball_qvel and ball_pos is not None:
            v_bounds = self.cal_ball_qvel(ball_pos)
            v_low, v_high = v_bounds[1], v_bounds[0]
            ball_vel = self.np_random.uniform(low=v_low, high=v_high)
            self._init_qvel[self.ball_dofadr : self.ball_dofadr + 3] = ball_vel
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = reset_qpos_local
        self.data.qvel[:] = self._init_qvel
        mujoco.mj_forward(self.model, self.data)
        self.cur_rally = 0
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        self.obs_dict = self._get_obs_dict(self._accessor)
        obs = self._obs_dict_to_vec(self.obs_dict)
        return np.asarray(obs), {}

    def step(self, action: np.ndarray, **kwargs: Any):
        ctrl = self._process_controls(action)
        n_frames = int(self._ctrl_dt / self.model.opt.timestep)
        self.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        self.obs_dict = self._get_obs_dict(self._accessor)
        self.rwd_dict = self.get_reward_dict(self.obs_dict)
        obs = self._obs_dict_to_vec(self.obs_dict)
        reward = float(self.rwd_dict["dense"])
        terminated = bool(np.asarray(self.rwd_dict["done"]).ravel()[0])
        info = {
            "obs_dict": self.obs_dict,
            "rwd_dict": self.rwd_dict,
            "touch_history": self.contact_trajectory,
        }
        return np.asarray(obs), reward, terminated, False, info

    def _preprocess_spec(self, spec, remove_body_collisions=True, add_left_arm=True):
        for paddle_b in spec.bodies:
            if "paddle" in paddle_b.name and paddle_b.parent != spec.worldbody:
                break
        for s in spec.sensors:
            if "pingpong" not in s.name and "paddle" not in s.name:
                spec.delete(s)
        temp_model = spec.compile()
        removed_ids = recursive_immobilize(
            spec, temp_model, spec.body("femur_l"), remove_eqs=True
        )
        removed_ids.extend(
            recursive_immobilize(
                spec, temp_model, spec.body("femur_r"), remove_eqs=True
            )
        )
        for key in spec.keys:
            key.qpos = [j for i, j in enumerate(key.qpos) if i not in removed_ids]
        if remove_body_collisions:
            recursive_remove_contacts(
                spec.body("full_body"), return_condition=lambda b: "radius" in b.name
            )
        if add_left_arm:
            torso = spec.body("torso")
            spec_copy: mujoco.MjSpec = spec.copy()
            attachment_frame = torso.add_frame(
                quat=[0.5, 0.5, -0.5, 0.5], pos=[0.05, 0.373, -0.04]
            )
            for col in (
                spec_copy.keys,
                spec_copy.textures,
                spec_copy.materials,
                spec_copy.tendons,
                spec_copy.actuators,
                spec_copy.equalities,
                spec_copy.sensors,
                spec_copy.cameras,
            ):
                for item in list(col):
                    spec_copy.delete(item)
            recursive_immobilize(spec_copy, temp_model, spec_copy.worldbody)
            recursive_remove_contacts(spec_copy.worldbody, return_condition=None)
            meshes_to_mirror: set[str] = set()
            recursive_mirror(meshes_to_mirror, spec_copy, spec_copy.body("clavicle"))
            for mesh in list(spec_copy.meshes):
                if mesh.name in meshes_to_mirror:
                    mesh.name += "_mirrored"
                    mesh.scale[1] *= -1
                else:
                    spec_copy.delete(mesh)
            attachment_frame.attach_body(spec_copy.body("clavicle_mirrored"))
            spec.body("ulna_mirrored").quat = [0.546, 0, 0, -0.838]
            spec.body("humerus_mirrored").quat = [0.924, 0.383, 0, 0]
        return spec


class IdInfo:
    def __init__(self, model: mujoco.MjModel):
        self.paddle_sid = model.site("paddle").id
        self.paddle_bid = model.body("paddle").id
        self.ball_sid = model.site("pingpong").id
        self.ball_bid = model.body("pingpong").id
        self.ball_gid = model.geom("pingpong").id
        self.own_half_gid = model.geom("coll_own_half").id
        self.paddle_gid = model.geom("pad").id
        self.opponent_half_gid = model.geom("coll_opponent_half").id
        self.ground_gid = model.geom("ground").id
        self.net_gid = model.geom("coll_net").id
        self.myo_joint_range = np.concatenate(
            [
                model.joint(i).qposadr
                for i in range(model.njnt)
                if not model.joint(i).name.startswith("ping")
                and model.joint(i).name
                not in ("pingpong_freejoint", "paddle_freejoint")
            ]
        )
        self.myo_dof_range = np.concatenate(
            [
                model.joint(i).dofadr
                for i in range(model.njnt)
                if not model.joint(i).name.startswith("ping")
                and model.joint(i).name != "paddle_freejoint"
            ]
        )


class PingpongContactLabels(enum.Enum):
    PADDLE = 0
    OWN = 1
    OPPONENT = 2
    GROUND = 3
    NET = 4
    ENV = 5


class ContactTrajIssue(enum.Enum):
    OWN_HALF = 0
    MISS = 1
    NO_PADDLE = 2
    DOUBLE_TOUCH = 3


def get_ball_contact_labels(
    model: mujoco.MjModel, data: mujoco.MjData, id_info: IdInfo
):
    for con in data.contact:
        if model.geom(con.geom1).bodyid == id_info.ball_bid:
            yield geom_id_to_label(con.geom2, id_info)
        elif model.geom(con.geom2).bodyid == id_info.ball_bid:
            yield geom_id_to_label(con.geom1, id_info)


def geom_id_to_label(body_id: int, id_info: IdInfo):
    if body_id == id_info.paddle_gid:
        return PingpongContactLabels.PADDLE
    if body_id == id_info.own_half_gid:
        return PingpongContactLabels.OWN
    if body_id == id_info.opponent_half_gid:
        return PingpongContactLabels.OPPONENT
    if body_id == id_info.net_gid:
        return PingpongContactLabels.NET
    if body_id == id_info.ground_gid:
        return PingpongContactLabels.GROUND
    return PingpongContactLabels.ENV


def evaluate_pingpong_trajectory(contact_trajectory: list[set]):
    has_hit_paddle = False
    has_bounced_from_paddle = False
    has_bounced_from_table = False
    own_contact_count = 0
    own_contact_phase_done = False
    for s in contact_trajectory:
        if PingpongContactLabels.PADDLE not in s and has_hit_paddle:
            has_bounced_from_paddle = True
        if PingpongContactLabels.PADDLE in s and has_bounced_from_paddle:
            return ContactTrajIssue.DOUBLE_TOUCH
        if PingpongContactLabels.PADDLE in s:
            has_hit_paddle = True
        if PingpongContactLabels.OWN in s:
            if not has_bounced_from_table:
                has_bounced_from_table = True
                own_contact_count = 1
            elif not own_contact_phase_done:
                own_contact_count += 1
                if own_contact_count > 2:
                    own_contact_phase_done = True
                    return ContactTrajIssue.OWN_HALF
            else:
                return ContactTrajIssue.OWN_HALF
        elif has_bounced_from_table:
            own_contact_phase_done = True
        if PingpongContactLabels.OPPONENT in s:
            if has_hit_paddle:
                return None
            return ContactTrajIssue.NO_PADDLE
    return ContactTrajIssue.MISS
