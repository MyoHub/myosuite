# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""RunTrack challenge environment."""

from __future__ import annotations

import collections
import copy
import csv

from enum import Enum
from typing import Any

import mujoco
import numpy as np

import gymnasium as gym
from gymnasium.utils import EzPickle

from myosuite.core.model_builder import ModelBuilder
from myosuite.core.muscle_conditions import apply_sarcopenia_to_model
from myosuite.envs.gymnasium_env import CpuEnvAccessor, MyoGymnasiumEnv
from myosuite.envs.heightfields import TrackField
from myosuite.envs.myo.assets.leg.myoosl_control import MyoOSLController
from myosuite.physics.fatigue import CumulativeFatigue
from myosuite.physics.quat_math import intrinsic_euler2quat, quat2euler_intrinsic
from myosuite.terms.base_action import sigmoid_muscle_activation


class TrackTypes(Enum):
    FLAT = 0
    HILLY = 1
    ROUGH = 2
    STAIRS = 3
    MIXED = 4


class RunTrackEnv(MyoGymnasiumEnv, EzPickle):
    """OSL prosthetic running track challenge — native MyoGymnasiumEnv implementation.

    The agent controls 54 biological muscles; the OSL prosthetic leg joints are
    driven automatically by the ``MyoOSLController``.

    Args:
        model_path: Absolute path to the MuJoCo XML model.
        obsd_model_path: Unused (kept for API compatibility).
        seed: Random seed.
        frame_skip: MuJoCo substeps per :meth:`step` call.
        obs_keys: Observation keys to include.
        weighted_reward_keys: Dict ``{key: weight}`` for dense reward.
        normalize_act: If ``True``, action space is ``[-1, 1]``.
        muscle_condition: One of ``""``, ``"sarcopenia"``, ``"fatigue"``.
        reset_type: One of ``"init"``, ``"random"``, ``"osl_init"``.
        terrain: One of ``"flat"``, ``"hilly"``, ``"rough"``, ``"stairs"``,
            ``"random"``.
        hills_difficulties: Interleaved (difficulty, weight) pairs for hills.
        rough_difficulties: Interleaved (difficulty, weight) pairs for rough.
        stairs_difficulties: Interleaved (difficulty, weight) pairs for stairs.
        real_width: Physical width of the track (m).
        real_length: Physical length of the track (m).
        end_pos: Y-coordinate for win condition.
        start_pos: Y-coordinate for initial placement.
        init_pose_path: Optional CSV of gait-cycle initialisation data.
        osl_param_set: Number of OSL parameter sets.
        max_episode_steps: Maximum steps before time limit.
    """

    DEFAULT_OBS_KEYS = [
        "internal_qpos",
        "internal_qvel",
        "grf",
        "torso_angle",
        "model_root_pos",
        "model_root_vel",
        "muscle_length",
        "muscle_velocity",
        "muscle_force",
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS: dict[str, float] = {
        "sparse": 1,
        "solved": 10,
    }

    ACTUATOR_PARAM: dict = {}
    OSL_PARAM_LIST: list = []
    OSL_PARAM_SELECT: int = 0

    PAIN_JNT = [
        "hip_adduction_l",
        "hip_adduction_r",
        "hip_flexion_l",
        "hip_flexion_r",
        "hip_rotation_l",
        "hip_rotation_r",
        "knee_angle_l",
        "knee_angle_l_rotation2",
        "knee_angle_l_rotation3",
        "mtp_angle_l",
        "ankle_angle_l",
        "subtalar_angle_l",
    ]
    BIOLOGICAL_JNT = [
        "hip_adduction_l",
        "hip_flexion_l",
        "hip_rotation_l",
        "hip_adduction_r",
        "hip_flexion_r",
        "hip_rotation_r",
        "knee_angle_l",
        "knee_angle_l_beta_rotation1",
        "knee_angle_l_beta_translation1",
        "knee_angle_l_beta_translation2",
        "knee_angle_l_rotation2",
        "knee_angle_l_rotation3",
        "knee_angle_l_translation1",
        "knee_angle_l_translation2",
        "mtp_angle_l",
        "ankle_angle_l",
        "subtalar_angle_l",
    ]
    BIOLOGICAL_ACT = [
        "addbrev_l",
        "addbrev_r",
        "addlong_l",
        "addlong_r",
        "addmagDist_l",
        "addmagIsch_l",
        "addmagMid_l",
        "addmagProx_l",
        "bflh_l",
        "bfsh_l",
        "edl_l",
        "ehl_l",
        "fdl_l",
        "fhl_l",
        "gaslat_l",
        "gasmed_l",
        "glmax1_l",
        "glmax1_r",
        "glmax2_l",
        "glmax2_r",
        "glmax3_l",
        "glmax3_r",
        "glmed1_l",
        "glmed1_r",
        "glmed2_l",
        "glmed2_r",
        "glmed3_l",
        "glmed3_r",
        "glmin1_l",
        "glmin1_r",
        "glmin2_l",
        "glmin2_r",
        "glmin3_l",
        "glmin3_r",
        "grac_l",
        "iliacus_l",
        "iliacus_r",
        "perbrev_l",
        "perlong_l",
        "piri_l",
        "piri_r",
        "psoas_l",
        "psoas_r",
        "recfem_l",
        "sart_l",
        "semimem_l",
        "semiten_l",
        "soleus_l",
        "tfl_l",
        "tibant_l",
        "tibpost_l",
        "vasint_l",
        "vaslat_l",
        "vasmed_l",
    ]

    def __init__(
        self,
        model_path: str,
        obsd_model_path: str | None = None,
        seed: int | None = None,
        frame_skip: int = 5,
        obs_keys: list = DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict[str, float] = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        normalize_act: bool = True,
        muscle_condition: str = "",
        reset_type: str = "init",
        terrain: str = "random",
        hills_difficulties: tuple = (0, 0),
        rough_difficulties: tuple = (0, 0),
        stairs_difficulties: tuple = (0, 0),
        real_width: float = 1.0,
        real_length: float = 120.0,
        end_pos: float = -15.0,
        start_pos: float = 14.0,
        init_pose_path: str | None = None,
        osl_param_set: int = 4,
        max_episode_steps: int = 36000,
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
            frame_skip=frame_skip,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            normalize_act=normalize_act,
            muscle_condition=muscle_condition,
            reset_type=reset_type,
            terrain=terrain,
            hills_difficulties=hills_difficulties,
            rough_difficulties=rough_difficulties,
            stairs_difficulties=stairs_difficulties,
            real_width=real_width,
            real_length=real_length,
            end_pos=end_pos,
            start_pos=start_pos,
            init_pose_path=init_pose_path,
            osl_param_set=osl_param_set,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

        # ── Load model ──────────────────────────────────────────────────────
        self.model, self._mj_spec = ModelBuilder.from_xml_file(model_path).build()
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * frame_skip)
        self.dt = self._ctrl_dt

        # ── Muscle condition ────────────────────────────────────────────────
        self.muscle_condition = muscle_condition
        self._muscle_act_ind = self.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        if muscle_condition == "sarcopenia":
            apply_sarcopenia_to_model(self.model, force_scale=0.5)
        elif muscle_condition == "fatigue":
            self.muscle_fatigue = CumulativeFatigue(self.model, frame_skip, seed=None)

        # ── Task config ─────────────────────────────────────────────────────
        self.reset_type = reset_type
        self.terrain = terrain
        self.end_pos = end_pos
        self.start_pos = start_pos
        self.real_width = real_width
        self.real_length = real_length
        self.osl_param_set = osl_param_set
        self.maxTime = self.dt * max_episode_steps
        self.terrain_type = TrackTypes.FLAT.value

        # ── Gait cycle initialization data ──────────────────────────────────
        self.INIT_DATA: np.ndarray | None = None
        self.init_lookup: dict = {}
        self.gait_cycle_headers: dict = {}
        if init_pose_path is not None:
            file_path = str(init_pose_path)
            self.INIT_DATA = np.loadtxt(file_path, skiprows=1, delimiter=",")
            self.init_lookup = self.generate_init_lookup(np.arange(48), "e_swing")
            self.init_lookup = self.generate_init_lookup(
                np.arange(48, 99), "l_swing", self.init_lookup
            )
            self.init_lookup = self.generate_init_lookup(
                np.arange(99, 183), "e_stance", self.init_lookup
            )
            self.init_lookup = self.generate_init_lookup(
                np.arange(183, 247), "l_stance", self.init_lookup
            )
            with open(file_path) as csv_file:
                headers = list(dict(list(csv.DictReader(csv_file))[0]).keys())
            self.gait_cycle_headers = dict(zip(headers, range(len(headers))))

        # ── OSL controller ───────────────────────────────────────────────────
        mujoco.mj_forward(self.model, self.data)
        self.OSL_CTRL = MyoOSLController(
            np.sum(self.model.body_mass),
            init_state="e_stance",
            n_sets=self.osl_param_set,
        )
        self.OSL_CTRL.start()
        self.muscle_space = self.model.na
        self.full_ctrl_space = self.model.nu
        self._get_actuator_params()

        # ── Terrain ──────────────────────────────────────────────────────────
        self.trackfield = TrackField(
            mj_model=self.model,
            mj_data=self.data,
            rng=self.np_random,
            rough_difficulties=rough_difficulties,
            hills_difficulties=hills_difficulties,
            stairs_difficulties=stairs_difficulties,
            real_length=real_length,
            real_width=real_width,
            reset_type=terrain,
            view_distance=5,
        )

        # ── Sensor names ─────────────────────────────────────────────────────
        self.grf_sensor_names = ["l_foot", "l_toes"]

        # ── Convenience caches ───────────────────────────────────────────────
        self.actuator_names = np.array(
            [self.model.actuator(i).name for i in range(self.model.nu)]
        )
        self.joint_names = np.array(
            [self.model.joint(i).name for i in range(1, self.model.njnt)]
        )
        self.muscle_fmax = self.model.actuator_gainprm[:, 2].copy()
        self.muscle_lengthrange = self.model.actuator_lengthrange.copy()
        self.tendon_len = self.model.tendon_lengthspring.copy()
        self.musc_operating_len = self.model.actuator_gainprm[:, 0:2].copy()

        # ── Reward / obs config ──────────────────────────────────────────────
        self.rwd_keys_wt = dict(weighted_reward_keys)
        self.obs_keys = list(obs_keys)
        if self.model.na > 0 and "act" not in self.obs_keys:
            self.obs_keys.append("act")

        # ── Action space — muscles only (OSL torques appended internally) ────
        self.normalize_act = normalize_act
        if normalize_act:
            act_low = -np.ones(self.model.na, dtype=np.float32)
            act_high = np.ones(self.model.na, dtype=np.float32)
        else:
            act_low = self.model.actuator_ctrlrange[: self.model.na, 0].astype(
                np.float32
            )
            act_high = self.model.actuator_ctrlrange[: self.model.na, 1].astype(
                np.float32
            )
        self.action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)

        # ── Initial state keyframe ───────────────────────────────────────────
        self._init_qpos = self.model.keyframe("stand").qpos.copy()
        self._init_qvel = np.zeros(self.model.nv)
        self.startFlag = False
        self._input_seed = seed

        import gymnasium as _gym

        _gym.Env.reset(self, seed=seed)

        # ── Observation space ─────────────────────────────────────────────────
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs = self._obs_dict_to_vec(self._get_obs_dict(self._accessor))
        self.observation_space = gym.spaces.Box(
            -10.0 * np.ones(obs.size, dtype=np.float32),
            10.0 * np.ones(obs.size, dtype=np.float32),
            dtype=np.float32,
        )
        self.startFlag = True

    # ── Action application ────────────────────────────────────────────────────

    def _apply_action(self, mus_action: np.ndarray) -> None:
        """Apply muscle actions + OSL torques to data.ctrl."""
        full_ctrl = np.zeros(self.model.nu)

        # Sigmoid or scale muscle activations
        ctrl = mus_action.copy()
        if self.model.na > 0 and self.normalize_act:
            ctrl[self._muscle_act_ind[: self.model.na]] = sigmoid_muscle_activation(
                ctrl[self._muscle_act_ind[: self.model.na]], np
            )
        elif self.normalize_act:
            ctrl_range = self.model.actuator_ctrlrange[: self.model.na]
            ctrl = (
                np.mean(ctrl_range, axis=-1)
                + ctrl * (ctrl_range[:, 1] - ctrl_range[:, 0]) / 2.0
            )
        if self.muscle_condition == "fatigue":
            ctrl[self._muscle_act_ind[: self.model.na]], _, _ = (
                self.muscle_fatigue.compute_act(
                    ctrl[self._muscle_act_ind[: self.model.na]]
                )
            )
        full_ctrl[: self.model.na] = ctrl

        # Append OSL torques
        self.OSL_CTRL.update(self._get_osl_sens())
        osl_torque = self.OSL_CTRL.get_osl_torque()
        for jnt in ("knee", "ankle"):
            act_name = f"osl_{jnt}_torque_actuator"
            osl_id = self.model.actuator(act_name).id
            gear = self.model.actuator(act_name).gear[0]
            osl_ctrl = osl_torque[jnt] / gear
            min_ctrl = self.model.actuator(act_name).ctrlrange[0]
            max_ctrl = self.model.actuator(act_name).ctrlrange[1]
            osl_ctrl = np.clip(osl_ctrl, min_ctrl, max_ctrl)
            if self.normalize_act:
                ctrl_mean = (min_ctrl + max_ctrl) / 2.0
                ctrl_rng = (max_ctrl - min_ctrl) / 2.0
                osl_ctrl = (osl_ctrl - ctrl_mean) / ctrl_rng
            full_ctrl[osl_id] = osl_ctrl

        self.data.ctrl[:] = full_ctrl

    # ── MyoGymnasiumEnv interface ─────────────────────────────────────────────

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        """Return full observation dict; _obs_dict_to_vec filters to obs_keys."""
        data = accessor.data
        obs_dict: dict[str, np.ndarray] = {}
        obs_dict["time"] = np.array([data.time])
        obs_dict["terrain"] = np.array([self.terrain_type])
        obs_dict["internal_qpos"] = self._get_internal_qpos()
        obs_dict["internal_qvel"] = self._get_internal_qvel()
        obs_dict["grf"] = self._get_grf()
        obs_dict["socket_force"] = self.data.sensor("r_socket_load").data.copy()
        obs_dict["torso_angle"] = data.body("pelvis").xquat.copy()
        obs_dict["muscle_length"] = self._muscle_lengths()
        obs_dict["muscle_velocities"] = self._muscle_velocities()
        obs_dict["muscle_velocity"] = obs_dict["muscle_velocities"]  # alias
        obs_dict["muscle_force"] = self._muscle_forces()
        if self.model.na > 0:
            obs_dict["act"] = data.act[:].copy()
        obs_dict["model_root_pos"] = data.qpos[:2].copy()
        obs_dict["model_root_vel"] = data.qvel[:2].copy()
        if self.trackfield is not None:
            obs_dict["hfield"] = self.trackfield.get_heightmap_obs()
        return obs_dict

    def _obs_dict_to_vec(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Flatten only the obs_keys subset of obs_dict to a 1-D vector."""
        return np.concatenate(
            [np.atleast_1d(obs_dict[k]).ravel() for k in self.obs_keys if k in obs_dict]
        )

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        act_mag = (
            np.mean(np.square(obs_dict.get("act", np.zeros(1))))
            if self.model.na != 0
            else 0
        )
        pain = self._get_pain(obs_dict)
        score = self._get_score(obs_dict)
        win_cdt = self._win_condition(obs_dict)
        self._lose_condition(obs_dict)

        rwd_dict = collections.OrderedDict(
            (
                ("act_reg", float(act_mag)),
                ("pain", pain),
                ("sparse", score),
                ("solved", win_cdt),
                ("done", self._get_done(obs_dict)),
            )
        )
        rwd_dict["dense"] = float(
            np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        )
        return rwd_dict

    # ── Gymnasium step / reset ───────────────────────────────────────────────

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data, self.frame_skip)
        mujoco.mj_kinematics(self.model, self.data)
        if self.mujoco_render_frames:
            self.mj_render()

        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs_dict = self._get_obs_dict(self._accessor)
        rwd_dict = self.get_reward_dict(obs_dict)

        obs = self._obs_dict_to_vec(obs_dict)
        obs = self._ensure_obs_gymnasium_compliant(obs)
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
        OSL_params=None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        import gymnasium as _gym

        _gym.Env.reset(self, seed=seed)

        if OSL_params is not None:
            self.upload_osl_param(OSL_params)

        self._maybe_sample_terrain()
        self.terrain_type = self.trackfield.terrain_type.value
        qpos, qvel = self._get_reset_state()

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

        if self.reset_type != "init":
            new_qpos, new_qvel = self.adjust_model_height()
            self.data.qpos[:] = new_qpos
            self.data.qvel[:] = new_qvel
            mujoco.mj_forward(self.model, self.data)

        self.OSL_CTRL.start()

        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs_dict = self._get_obs_dict(self._accessor)
        obs = self._obs_dict_to_vec(obs_dict)
        obs = self._ensure_obs_gymnasium_compliant(obs)
        return obs, {}

    # ── Terminal conditions ───────────────────────────────────────────────────

    def _get_done(self, obs_dict: dict) -> int:
        if not self.startFlag:
            return 0
        if self._lose_condition(obs_dict):
            return 1
        if self._win_condition(obs_dict):
            return 1
        return 0

    def _win_condition(self, obs_dict: dict) -> int:
        y_pos = float(obs_dict["model_root_pos"].squeeze()[1])
        return 1 if y_pos < self.end_pos else 0

    def _lose_condition(self, obs_dict: dict) -> int:
        root = obs_dict["model_root_pos"].squeeze()
        x_pos, y_pos = float(root[0]), float(root[1])
        if x_pos > self.real_width or x_pos < -self.real_width:
            return 1
        if y_pos > self.start_pos + 2:
            return 1
        if self._get_fallen_condition():
            return 1
        return 0

    def _get_fallen_condition(self) -> int:
        head = self.data.site("head").xpos
        foot_l = self.data.body("talus_l").xpos
        foot_r = self.data.body("osl_foot_assembly").xpos
        mean = (foot_l + foot_r) / 2
        if head[2] - mean[2] < 0.2:
            return 1
        if head[2] < 0.2:
            return 1
        return 0

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _get_score(self, obs_dict: dict) -> float:
        if not self.startFlag:
            return -1.0
        vel = float(obs_dict["model_root_vel"].squeeze()[1])
        return -vel

    def _get_pain(self, obs_dict: dict) -> float:
        if not self.startFlag:
            return -1.0
        pain_score = 0.0
        for joint in self.PAIN_JNT:
            pain_score += (
                np.clip(np.abs(self._get_limitfrc(joint).squeeze()), -1000, 1000) / 1000
            )
        return pain_score / len(self.PAIN_JNT)

    def _get_limitfrc(self, joint_name: str) -> np.ndarray:
        non_jnt_idxs = np.where(
            self.data.efc_type != mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
        )[0]
        only_jnt_lim = self.data.efc_force.copy()
        only_jnt_lim[non_jnt_idxs] = 0.0
        joint_force = np.zeros(self.model.nv)
        mujoco.mj_mulJacTVec(self.model, self.data, joint_force, only_jnt_lim)
        return joint_force[self.model.joint(joint_name).dofadr]

    # ── Biological obs helpers ────────────────────────────────────────────────

    def _get_internal_qpos(self) -> np.ndarray:
        return np.array(
            [self.data.joint(j).qpos[0].copy() for j in self.BIOLOGICAL_JNT]
        )

    def _get_internal_qvel(self) -> np.ndarray:
        return (
            np.array([self.data.joint(j).qvel[0].copy() for j in self.BIOLOGICAL_JNT])
            * self.dt
        )

    def _get_grf(self) -> np.ndarray:
        return np.array(
            [self.data.sensor(n).data[0] for n in self.grf_sensor_names]
        ).copy()

    def _muscle_lengths(self) -> np.ndarray:
        return np.array(
            [self.data.actuator(a).length[0].copy() for a in self.BIOLOGICAL_ACT]
        )

    def _muscle_forces(self) -> np.ndarray:
        return np.clip(
            np.array(
                [self.data.actuator(a).force[0].copy() for a in self.BIOLOGICAL_ACT]
            )
            / 1000,
            -100,
            100,
        )

    def _muscle_velocities(self) -> np.ndarray:
        return np.clip(
            np.array(
                [self.data.actuator(a).velocity[0].copy() for a in self.BIOLOGICAL_ACT]
            ),
            -100,
            100,
        )

    # ── OSL controller interface ──────────────────────────────────────────────

    def _get_osl_sens(self) -> dict[str, float]:
        return {
            "knee_angle": self.data.joint("osl_knee_angle_r").qpos[0].copy(),
            "knee_vel": self.data.joint("osl_knee_angle_r").qvel[0].copy(),
            "ankle_angle": self.data.joint("osl_ankle_angle_r").qpos[0].copy(),
            "ankle_vel": self.data.joint("osl_ankle_angle_r").qvel[0].copy(),
            "load": -1.0 * self.data.sensor("r_osl_load").data[1].copy(),
        }

    def upload_osl_param(self, dict_of_dict: dict) -> None:
        """Upload a full set of OSL parameters."""
        assert len(dict_of_dict.keys()) <= 4
        for idx in dict_of_dict.keys():
            self.OSL_CTRL.set_osl_param_batch(dict_of_dict[idx], mode=idx)

    def change_osl_mode(self, mode: int = 0) -> None:
        """Activate a set of OSL state machine variables."""
        assert mode < 4
        self.OSL_CTRL.change_osl_mode(mode)

    def _get_actuator_params(self) -> None:
        for act_name in ("osl_knee_torque_actuator", "osl_ankle_torque_actuator"):
            self.ACTUATOR_PARAM[act_name] = {
                "id": self.data.actuator(act_name).id,
                "Fmax": (
                    np.max(self.model.actuator(act_name).ctrlrange)
                    * self.model.actuator(act_name).gear[0]
                ),
            }

    # ── Terrain ───────────────────────────────────────────────────────────────

    def _maybe_sample_terrain(self) -> None:
        if self.trackfield is not None:
            self.trackfield.sample(self.np_random)
            self.model.geom_rgba[self.model.geom("terrain").id][-1] = 1.0
            self.model.geom_pos[self.model.geom("terrain").id] = np.array([0, 0, 0.005])
        else:
            self.model.geom_rgba[self.model.geom("terrain").id][-1] = 0.0
            self.model.geom_pos[self.model.geom("terrain").id] = np.array([0, 0, -10])

    # ── Reset state helpers ───────────────────────────────────────────────────

    def _get_reset_state(self) -> tuple[np.ndarray, np.ndarray]:
        if self.reset_type == "random":
            qpos, qvel = self._get_randomized_initial_state()
            return self._randomize_position_orientation(qpos, qvel)
        elif self.reset_type == "osl_init":
            self.initializeFromData()
            return self._init_qpos.copy(), self._init_qvel.copy()
        else:  # "init" or default
            self.OSL_CTRL.reset("e_stance")
            return self.model.key_qpos[0].copy(), self.model.key_qvel[0].copy()

    def _get_randomized_initial_state(self) -> tuple[np.ndarray, np.ndarray]:
        rnd_int = self.np_random.integers(low=0, high=3)
        qpos = self.model.key_qpos[rnd_int].copy()
        qvel = self.model.key_qvel[rnd_int].copy()
        if rnd_int == 0 or rnd_int == 2:
            self.OSL_CTRL.reset("e_stance")
        else:
            self.OSL_CTRL.reset("e_swing")
        rot_state = qpos[3:7].copy()
        height = qpos[2]
        qpos[0] = self.np_random.uniform(-self.real_width * 0.8, self.real_width * 0.8)
        qpos[3:7] = rot_state
        qpos[2] = height
        qpos[1] = self.start_pos + 1
        return qpos, qvel

    def _randomize_position_orientation(
        self, qpos: np.ndarray, qvel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        orientation = self.np_random.uniform(np.deg2rad(-125), np.deg2rad(-60))
        euler_angle = quat2euler_intrinsic(qpos[3:7])
        euler_angle[2] = orientation
        qpos[3:7] = intrinsic_euler2quat(
            [euler_angle[0], euler_angle[1], euler_angle[2]]
        )
        qvel[:2] = np.array(
            [np.cos(orientation), np.sin(orientation)]
        ) * np.linalg.norm(qvel[:2])
        return qpos, qvel

    def adjust_model_height(self) -> tuple[np.ndarray, np.ndarray]:
        """Raise agent so lowest foot site is at ground level."""
        curr_qpos = self.data.qpos.copy()
        curr_qvel = self.data.qvel.copy()
        temp_sens_height = 100.0
        for site in ("r_heel_btm", "r_toe_btm", "l_heel_btm", "l_toe_btm"):
            h = self.data.site(site).xpos[2]
            if h < temp_sens_height:
                temp_sens_height = float(h)
        target = 0.005 if self.trackfield is not None else 0.0
        curr_qpos[2] += target - temp_sens_height
        return curr_qpos, curr_qvel

    # ── Gait-cycle initialisation ────────────────────────────────────────────

    def initializeFromData(self) -> None:
        """Set initial state from a random row in the gait-cycle CSV."""
        assert self.INIT_DATA is not None, "init_pose_path must be set to use osl_init"
        start_idx = self.np_random.integers(low=0, high=self.INIT_DATA.shape[0])
        skip = {
            "pelvis_euler_roll",
            "pelvis_euler_pitch",
            "pelvis_euler_yaw",
            "l_foot_relative_X",
            "l_foot_relative_Y",
            "l_foot_relative_Z",
            "r_foot_relative_X",
            "r_foot_relative_Y",
            "r_foot_relative_Z",
            "pelvis_vel_X",
            "pelvis_vel_Y",
            "pelvis_vel_Z",
        }
        for joint, col in self.gait_cycle_headers.items():
            if joint not in skip:
                self._init_qpos[self.model.joint(joint).qposadr[0]] = self.INIT_DATA[
                    start_idx, col
                ]
        default_quat = self._init_qpos[3:7].copy()
        self._init_qpos[3:7] = intrinsic_euler2quat(
            [
                self.INIT_DATA[start_idx, self.gait_cycle_headers["pelvis_euler_roll"]],
                self.INIT_DATA[
                    start_idx, self.gait_cycle_headers["pelvis_euler_pitch"]
                ],
                self.INIT_DATA[start_idx, self.gait_cycle_headers["pelvis_euler_yaw"]],
            ]
        )
        temp_euler = quat2euler_intrinsic(default_quat)
        world_vx, world_vy = self._rotate_frame(
            self.INIT_DATA[start_idx, self.gait_cycle_headers["pelvis_vel_X"]],
            self.INIT_DATA[start_idx, self.gait_cycle_headers["pelvis_vel_Y"]],
            temp_euler[2],
        )
        self._init_qvel[:] = 0.0
        self._init_qvel[0] = world_vx
        self._init_qvel[1] = world_vy
        self._init_qvel[2] = self.INIT_DATA[
            start_idx, self.gait_cycle_headers["pelvis_vel_Z"]
        ]
        self.OSL_CTRL.reset(self.init_lookup[start_idx])

    @staticmethod
    def _rotate_frame(x: float, y: float, theta: float) -> tuple[float, float]:
        return (
            np.cos(theta) * x - np.sin(theta) * y,
            np.sin(theta) * x + np.cos(theta) * y,
        )

    @staticmethod
    def generate_init_lookup(
        keys: np.ndarray, value: str, existing_dict: dict | None = None
    ) -> dict:
        result = copy.deepcopy(existing_dict) if existing_dict is not None else {}
        for key in keys:
            result[key] = value
        return result

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_metrics(self, paths: list) -> dict[str, float]:
        times = np.mean(
            [np.round(p["env_infos"]["obs_dict"]["time"][-1], 5) for p in paths]
        )
        score = np.mean(
            [
                np.min(p["env_infos"]["obs_dict"]["model_root_pos"][..., 1])
                for p in paths
            ]
        )
        if self.start_pos > self.end_pos:
            score = (self.start_pos - score) / (self.start_pos - self.end_pos)
        else:
            score = (score - self.end_pos) / (self.start_pos - self.end_pos)
        return {
            "score": float(np.clip(score, 0, 1)),
            "time": times,
            "effort": np.mean(
                [np.sum(p["env_infos"]["rwd_dict"]["act_reg"]) for p in paths]
            ),
            "pain": np.mean(
                [np.sum(p["env_infos"]["rwd_dict"]["pain"]) for p in paths]
            ),
        }
