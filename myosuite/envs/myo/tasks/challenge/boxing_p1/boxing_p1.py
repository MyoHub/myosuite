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
from ml_collections import config_dict
from mujoco import mjtObj
from scipy.spatial.transform import Rotation as R

from myosuite.core.model_builder import ModelBuilder
from myosuite.envs.gymnasium_env import CpuEnvAccessor, MyoGymnasiumEnv
from myosuite.envs.myo.tasks.challenge.boxing_specs import (
    replace_hand_visuals_with_gloves,
    add_helmet,
    replace_floor_visual_with_boxing_ring,
    build_targets_spec,
)
from myosuite.integrations.musclemimic import build_mimic_fullbody_spec
from myosuite.physics.quat_math import euler2quat

BOXING_CONFIG_P0 = config_dict.create(
    target_pos_offset=(0, 0, 0),
    reset_pos_range=((0, 0), (0, 0), (0, 0)),
    reset_rot_range=(0, 0),
    wait_time_range=(3, 3),
    qpos_noise_range=(0, 0),
    valid_targets=(0,),
    timeout=10.0,
    n_hits=3,
    guard_distance=0.4,
    sandbox_mode=False,
)


class BoxingEnv(MyoGymnasiumEnv, EzPickle):
    """Classic MyoSuite-style definition of the boxing env. Will be used for evaluation."""

    DEFAULT_OBS_KEYS = [
        "body_qpos",
        "body_qvel",
        "act",
        "target_pos",
        "target_normal",
        "box_state",
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "sparse": 100,
        "done": -10,
    }

    def __init__(
        self,
        model_path: str = None,
        obsd_model_path: str | None = None,
        seed: int | None = None,
        config: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = BOXING_CONFIG_P0 if config is None else config
        MyoGymnasiumEnv.__init__(self, **kwargs)
        EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        fullbody_spec = build_mimic_fullbody_spec(
            config=config_dict.create(disable_fingers=True, model_path=model_path)
        )[0]

        self._mj_spec = self._preprocess_spec(fullbody_spec)
        self.model = self._mj_spec.compile()
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * self.frame_skip)

        self._muscle_act_ind = self.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE

        self.obs_keys = list(kwargs.get("obs_keys", self.DEFAULT_OBS_KEYS))
        self.rwd_keys_wt = kwargs.get(
            "weighted_reward_keys", self.DEFAULT_RWD_KEYS_AND_WEIGHTS
        )
        if self.model.na > 0 and "act" not in self.obs_keys:
            self.obs_keys.append("act")

        self._init_qpos = self.model.key_qpos[1].copy()
        self._init_qvel = self.model.key_qvel[1].copy()

        self.state_info = {
            "time_left": 0.0,
            "cur_state": BoxingTaskState.GUARD,
            "n_hits_successful": 0,
            "cur_target": self.config.valid_targets[0],
            "hit_successful": False,
            "cur_impact": 0.0,
            "fail_state": BoxingFailState.NONE,
        }
        self.metrics = {"average_impact": 0, "n_hits_successful": 0}
        self.pad_ids = [self.model.body(f"pad_{i}").id for i in range(5)]
        self.pad_gids = [self.model.geom(f"pad_{i}_geom").id for i in range(5)]

        gym.Env.reset(self, seed=seed)
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs = self._obs_dict_to_vec(self._get_obs_dict(self._accessor))
        self.observation_space = gym.spaces.Box(
            -10.0 * np.ones(obs.size, dtype=np.float32),
            10.0 * np.ones(obs.size, dtype=np.float32),
            dtype=np.float32,
        )
        act_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        act_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
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
        obs_dict["time"] = np.array([accessor.data.time])
        obs_dict["pelvis_pos"] = accessor.data.site_xpos[
            accessor.model.site("pelvis").id
        ]
        obs_dict["body_qpos"] = accessor.data.qpos.copy()
        obs_dict["body_qvel"] = accessor.data.qvel.copy()
        obs_dict["act"] = accessor.data.act.copy()
        obs_dict["target_pos"] = accessor.data.xpos[
            self.pad_ids[self.state_info["cur_target"]]
        ]
        obs_dict["target_normal"] = accessor.data.xmat[
            self.pad_ids[self.state_info["cur_target"]]
        ][2::3]
        obs_dict["box_state"] = np.array([int(self.state_info["cur_state"])])
        return obs_dict

    def _get_done(self):
        return (
            self.state_info["time_left"] < 0
            and self.state_info["cur_state"] == BoxingTaskState.PUNCH
        ) or self.state_info["fail_state"] != BoxingFailState.NONE

    def _increment_task_state(self):
        self.state_info["time_left"] -= self._ctrl_dt

        punch_force = self._get_target_force()
        contact_with_target = punch_force > 0.01

        if (
            self.state_info["cur_state"] == BoxingTaskState.PUNCH
            and contact_with_target
        ):
            self.state_info["cur_state"] = BoxingTaskState.GUARD
            self.state_info["time_left"] = np.random.uniform(
                *self.config.wait_time_range
            )
            self.state_info["n_hits_successful"] += 1
            self.metrics["n_hits_successful"] += 1
            self.state_info["hit_successful"] = True
            self._highlight_target()

        if self.state_info["cur_state"] == BoxingTaskState.GUARD:
            if self.state_info["time_left"] < 0:
                self.state_info["fail_state"] = (
                    BoxingFailState.NONE
                    if self._hands_within_guard_distance()
                    else BoxingFailState.NO_GUARD
                )
                self.state_info["cur_state"] = BoxingTaskState.PUNCH
                self.state_info["cur_target"] = np.random.choice(
                    self.config.valid_targets
                )
                self.state_info["time_left"] = self.config.timeout
                self._highlight_target()
            if not contact_with_target and self.state_info["cur_impact"] > 0.0:
                self.metrics["average_impact"] = (
                    self.metrics["average_impact"]
                    + (self.state_info["cur_impact"] - self.metrics["average_impact"])
                    / self.state_info["n_hits_successful"]
                )
            if contact_with_target:
                self.state_info["cur_impact"] += punch_force * self._ctrl_dt

    def _get_target_force(self):
        return self.get_sensor_by_name(
            f"punch_sensor_pad_{self.state_info['cur_target']}"
        )[0]

    def _hands_within_guard_distance(self):
        dl = np.linalg.norm(
            self.data.body("boxing_glove_body_l").xpos - self.data.body("head").xpos
        )
        dr = np.linalg.norm(
            self.data.body("boxing_glove_body_r").xpos - self.data.body("head").xpos
        )
        return dl < self.config.guard_distance and dr < self.config.guard_distance

    def _highlight_target(self):
        passive_color = [32 / 256, 138 / 256, 160 / 256, 1.0]
        highlighted_color = [160 / 256, 138 / 256, 32 / 256, 1.0]
        if self.state_info["cur_state"] == BoxingTaskState.GUARD:
            for t in self.config.valid_targets:
                self.model.geom(self.pad_gids[t]).rgba = passive_color
            return
        valid_targets = set(self.config.valid_targets)
        valid_targets.remove(self.state_info["cur_target"])
        for t in valid_targets:
            self.model.geom(self.pad_gids[t]).rgba = passive_color
        self.model.geom(
            self.pad_gids[self.state_info["cur_target"]]
        ).rgba = highlighted_color

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        act_mag = (
            float(np.linalg.norm(obs_dict["act"], axis=-1)) / self.model.na
            if self.model.na != 0 and "act" in obs_dict
            else 0.0
        )
        sparse = 1 if self.state_info["hit_successful"] else 0
        if self.state_info["hit_successful"]:
            self.state_info["hit_successful"] = False
        solved = 0

        rwd_dict = collections.OrderedDict(
            (
                ("act_reg", -1.0 * act_mag),
                ("sparse", sparse),
                ("solved", solved),
                ("done", self._get_done()),
            )
        )
        rwd_dict["dense"] = sum(
            float(wt) * float(np.array(rwd_dict[key]).squeeze())
            for key, wt in self.rwd_keys_wt.items()
        )

        return rwd_dict

    def reset(self, seed: int | None = None, options: dict | None = None, **_kwargs):
        gym.Env.reset(self, seed=seed)
        self._init_qpos[:] = self.model.key_qpos[1].copy()
        self.data.mocap_pos[0][:] = 0
        if self.config.reset_pos_range:
            self.data.mocap_pos[0] = [
                np.random.uniform(*self.config.reset_pos_range[i]) for i in range(3)
            ]
        if self.config.reset_rot_range:
            self.data.mocap_quat[0] = euler2quat(
                [0, 0, np.random.uniform(*self.config.reset_rot_range)]
            )

        self.state_info = {
            "time_left": np.random.uniform(*self.config.wait_time_range),
            "cur_state": BoxingTaskState.GUARD,
            "n_hits_successful": 0,
            "cur_target": np.random.choice(self.config.valid_targets),
            "hit_successful": False,
            "cur_impact": 0.0,
            "fail_state": BoxingFailState.NONE,
        }
        self.metrics = {"average_impact": 0, "n_hits_successful": 0}
        if self.config.qpos_noise_range is not None:
            joint_ranges = self.model.jnt_range[:, 1] - self.model.jnt_range[:, 0]
            noise_fraction = self.np_random.uniform(
                *self.config.qpos_noise_range, size=joint_ranges.shape
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
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = reset_qpos_local
        self.data.qvel[:] = self._init_qvel
        mujoco.mj_forward(self.model, self.data)

        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        self.obs_dict = self._get_obs_dict(self._accessor)
        obs = self._obs_dict_to_vec(self.obs_dict)
        self._highlight_target()
        return np.asarray(obs), {}

    def step(self, action: np.ndarray, **kwargs: Any):
        self._increment_task_state()
        return super().step(action, **kwargs)

    def _preprocess_spec(self, spec):
        spec = replace_hand_visuals_with_gloves(spec)
        spec = replace_floor_visual_with_boxing_ring(spec)
        spec = add_helmet(spec)
        target_spec = build_targets_spec()

        frame = spec.worldbody.add_frame(pos=(0, -0.5, 0))
        frame.attach_body(target_spec.body("targets"))
        _model = spec.compile()
        _data = mujoco.MjData(_model)
        _qpos = _data.qpos
        for s in ["l", "r"]:
            _qpos[_model.joint(f"elv_angle_{s}").qposadr] = 0.9488
            _qpos[_model.joint(f"shoulder_elv_{s}").qposadr] = 0.8949
            _qpos[_model.joint(f"shoulder1_r2_{s}").qposadr] = 0.5075
            _qpos[_model.joint(f"shoulder_rot_{s}").qposadr] = -1.571
            _qpos[_model.joint(f"elbow_flex_{s}").qposadr] = 2.269
        spec.add_key(qpos=_qpos)

        spot = spec.worldbody.add_light(name="scene_spot")
        spot.pos = [-2.0, -2.0, 4.0]
        spot.dir = [0.5, 0.5, -1]
        spot.specular = [1, 1, 1]
        spot.ambient = [1, 1, 1]
        spot.castshadow = True

        if self.config.sandbox_mode:
            _data = mujoco.MjData(_model)
            mujoco.mj_forward(_model, _data)
            spec.worldbody.add_body(name="h_p", pos=_data.body("head").xpos)
            spec.add_equality(
                name="eq1",
                type=mujoco.mjtEq.mjEQ_WELD,
                name1="head",
                name2="h_p",
                objtype=mjtObj.mjOBJ_BODY,
                solimp=[0.99, 0.99, 0.00001, 0.5, 2],
                solref=[0.02, 1],
            )
            b = spec.worldbody.add_body(
                name="g_p1",
                pos=_data.body("boxing_glove_body_l").xpos,
                quat=_data.body("boxing_glove_body_l").xquat,
                mocap=True,
            )
            b.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                rgba=(1, 1, 1, 0.2),
                size=0.05,
                contype=0,
                conaffinity=0,
            )
            spec.add_equality(
                name="eq2",
                type=mujoco.mjtEq.mjEQ_CONNECT,
                name1="g_p1",
                name2="boxing_glove_body_l",
                data=[0] * 11,
                objtype=mjtObj.mjOBJ_BODY,
            )
            b = spec.worldbody.add_body(
                name="g_p2",
                pos=_data.body("boxing_glove_body_r").xpos,
                quat=_data.body("boxing_glove_body_r").xquat,
                mocap=True,
            )
            b.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                rgba=(1, 1, 1, 0.2),
                size=0.05,
                contype=0,
                conaffinity=0,
            )
            spec.add_equality(
                name="eq3",
                type=mujoco.mjtEq.mjEQ_CONNECT,
                name1="g_p2",
                name2="boxing_glove_body_r",
                data=[0] * 11,
                objtype=mjtObj.mjOBJ_BODY,
            )
        return spec


class BoxingTaskState(enum.IntEnum):
    GUARD = 0
    PUNCH = 1


class BoxingFailState(enum.IntEnum):
    NONE = 0
    NO_GUARD = 1
    WRONG_TARGET = 2


def _visu_main():
    config = BOXING_CONFIG_P0
    config.sandbox_mode = True
    env = BoxingEnv(config=config)
    print("Equality constraints added to maintain posture without policy!")
    from mujoco import viewer
    from loop_rate_limiters import rate_limiter

    action = np.zeros_like(env.action_space)
    limiter = rate_limiter.RateLimiter(frequency=1 / env._ctrl_dt)
    env.reset()
    with viewer.launch_passive(env.model, env.data) as v:
        while v.is_running():
            obs, reward, terminated, truncated, info = env.step(action)
            v.set_texts(
                [
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "\n\nAverage impact:",
                        f"\n\n{env.metrics['average_impact']}",
                    ),
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "Hits:",
                        f"{env.metrics['n_hits_successful']}",
                    ),
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "\nTask state:",
                        "\nGuard"
                        if env.state_info["cur_state"] == BoxingTaskState.GUARD
                        else "\nPunch",
                    ),
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "Hits:",
                        f"{env.metrics['n_hits_successful']}",
                    ),
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                        "\nDone:",
                        f"\n{terminated}, {truncated}",
                    ),
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                        "Time left:",
                        f"{env.state_info['time_left']:0.2f}",
                    ),
                ]
            )
            if terminated:
                env.reset()
            v.sync()
            limiter.sleep()

    pass


if __name__ == "__main__":
    _visu_main()
