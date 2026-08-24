# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Soccer challenge environment."""

from __future__ import annotations

import collections
from typing import Any

import mujoco
import numpy as np
import pink

import gymnasium as gym
from gymnasium.utils import EzPickle

from myosuite.core.model_builder import ModelBuilder
from myosuite.core.muscle_conditions import apply_sarcopenia_to_model
from myosuite.envs.gymnasium_env import CpuEnvAccessor, MyoGymnasiumEnv
from myosuite.envs.myo.assets._resolve import warn_torso_pip_calibration_divergence
from myosuite.physics.fatigue import CumulativeFatigue
from myosuite.physics.quat_math import euler2quat, quat2euler
from myosuite.terms.base_action import sigmoid_muscle_activation


class GoalKeeper:
    """Goalkeeper for the Soccer Track of the MyoChallenge 2025.

    Manages the goalkeeper's behaviour, including randomized movement speed
    and probabilities of blocking the goal.

    Args:
        mj_model: MuJoCo model object.
        mj_data: MuJoCo data object.
        rng: NumPy random generator.
        random_vel_range: Velocity range for blocking/random policies.
        probabilities: Policy probabilities (block_ball, random, stationary).
        dt: Control timestep.
    """

    FIXED_X_POS = 50
    Y_MIN_BOUND = -3.3
    Y_MAX_BOUND = 3.3
    P_GAIN = 5.0

    def __init__(
        self,
        mj_model,
        mj_data,
        rng,
        random_vel_range: tuple[float, float],
        probabilities: tuple[float, ...] = (0.1, 0.35, 0.55),
        dt: float = 0.01,
    ) -> None:
        self.dt = dt
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.goalkeeper_probabilities = probabilities
        self.random_vel_range = random_vel_range
        self.rng = rng
        self.block_velocity = self.rng.uniform(
            self.random_vel_range[0], self.random_vel_range[1]
        )
        self.reset_goalkeeper(rng=rng)

    def reset_noise_process(self) -> None:
        self.noise_process = pink.ColoredNoiseProcess(
            beta=2, size=(2, 10), scale=self.block_velocity, rng=self.rng
        )

    def get_goalkeeper_pose(self) -> np.ndarray:
        """Return goalkeeper pose as [x, y, angle]."""
        angle = quat2euler(self.mj_data.mocap_quat[0, :])[-1]
        return np.concatenate([self.mj_data.mocap_pos[0, :2], [angle]])

    def set_goalkeeper_pose(self, pose: list) -> None:
        """Teleport goalkeeper to given pose, enforcing boundary constraints."""
        pose[0] = self.FIXED_X_POS
        pose[1] = np.clip(pose[1], self.Y_MIN_BOUND, self.Y_MAX_BOUND)
        self.mj_data.mocap_pos[0, :2] = pose[:2]
        self.mj_data.mocap_quat[0, :] = euler2quat([0, 0, pose[-1]])

    def move_goalkeeper(self, vel: list) -> None:
        """Move goalkeeper by [lin_vel, rot_vel]."""
        self.goalkeeper_vel = vel
        assert len(vel) == 2
        pose = self.get_goalkeeper_pose()
        pose[1] += self.dt * vel[0]
        self.set_goalkeeper_pose(pose)

    def random_movement(self) -> np.ndarray:
        return np.clip(
            self.noise_process.sample(), -self.block_velocity, self.block_velocity
        )

    def block_ball_policy(self) -> np.ndarray:
        """Move toward ball's Y position."""
        goalkeeper_pose = self.get_goalkeeper_pose()
        ball_pos = self.mj_data.body("soccer_ball").xpos[:3].copy()
        target_y = np.clip(ball_pos[1], self.Y_MIN_BOUND, self.Y_MAX_BOUND)
        displacement = target_y - goalkeeper_pose[1]
        linear_vel_y = np.clip(displacement, -self.block_velocity, self.block_velocity)
        return np.array([linear_vel_y, 0.0])

    def sample_goalkeeper_policy(self) -> None:
        rand_num = self.rng.uniform()
        cum = 0.0
        for policy, prob in zip(
            ("block_ball", "random", "stationary"), self.goalkeeper_probabilities
        ):
            cum += prob
            if rand_num < cum:
                self.goalkeeper_policy = policy
                return

    def update_goalkeeper_state(self) -> None:
        """Execute one goalkeeper step using the active policy."""
        if self.goalkeeper_policy == "stationary":
            goalkeeper_vel = np.zeros(2)
        elif self.goalkeeper_policy == "random":
            goalkeeper_vel = self.random_movement()
        elif self.goalkeeper_policy == "block_ball":
            goalkeeper_vel = self.block_ball_policy()
        else:
            raise NotImplementedError(
                f"Unknown goalkeeper policy: {self.goalkeeper_policy}"
            )
        self.move_goalkeeper(goalkeeper_vel)

    def reset_goalkeeper(self, rng=None) -> None:
        """Reset goalkeeper position and policy."""
        if rng is not None:
            self.rng = rng
            self.reset_noise_process()
        self.goalkeeper_vel = np.zeros(2)
        self.sample_goalkeeper_policy()
        initial_y = self.rng.uniform(self.Y_MIN_BOUND, self.Y_MAX_BOUND)
        self.set_goalkeeper_pose([self.FIXED_X_POS, initial_y, 0])
        self.goalkeeper_vel[:] = 0.0
        self.block_velocity = self.rng.uniform(
            self.random_vel_range[0], self.random_vel_range[1]
        )


class SoccerEnv(MyoGymnasiumEnv, EzPickle):
    """Soccer locomotion challenge — native MyoGymnasiumEnv implementation.

    The agent must kick a soccer ball into the goal while a goalkeeper tries
    to block it.

    Args:
        model_path: Absolute path to the MuJoCo XML model.
        obsd_model_path: Unused (kept for API compatibility).
        seed: Random seed.
        frame_skip: MuJoCo substeps per :meth:`step` call.
        obs_keys: Observation keys to include.
        weighted_reward_keys: Dict ``{key: weight}`` for dense reward.
        normalize_act: If ``True``, action space is ``[-1, 1]``.
        muscle_condition: One of ``""``, ``"sarcopenia"``, ``"fatigue"``.
        reset_type: One of ``"none"``, ``"init"``, ``"random"``.
        min_agent_spawn_distance: Minimum spawn radius for agent (m).
        random_vel_range: Goalkeeper velocity range.
        rnd_pos_noise: Position noise for random reset.
        rnd_joint_noise: Joint noise for random reset.
        goalkeeper_probabilities: Per-policy probabilities.
        max_time_sec: Episode time limit (seconds).
    """

    DEFAULT_OBS_KEYS = [
        "internal_qpos",
        "internal_qvel",
        "grf",
        "torso_angle",
        "pelvis_angle",
        "ball_pos",
        "model_root_pos",
        "model_root_vel",
        "muscle_length",
        "muscle_velocity",
        "muscle_force",
        "r_toe_pos",
        "l_toe_pos",
        "goal_bounds",
        "goalkeeper_pos",
        "time",
    ]
    # Dense shaping notes:
    # - ``act_reg`` must stay near -0.1. Weight -100 collapses PPO returns
    #   (legacy Δ ≈ −thousands) before any goal signal is learnable.
    # - ``time_cost`` is a per-step constant (value 1.0), not cumulative
    #   ``obs_dict["time"]``, so the penalty does not grow unboundedly.
    # - ``alive`` keeps upright rollouts competitive with early falls.
    DEFAULT_RWD_KEYS_AND_WEIGHTS: dict[str, float] = {
        "goal_scored": 1000,
        "time_cost": -0.01,
        "act_reg": -0.1,
        "pain": -10,
        "alive": 0.5,
    }

    GOAL_X_POS = GoalKeeper.FIXED_X_POS
    GOAL_Y_MIN = GoalKeeper.Y_MIN_BOUND
    GOAL_Y_MAX = GoalKeeper.Y_MAX_BOUND
    GOAL_Z_MIN = 0.0
    GOAL_Z_MAX = 2.2

    JNT_OVEREXT = [
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

    MYO_JOINTS = [
        "Abs_r3",
        "Abs_t1",
        "Abs_t2",
        "L1_L2_AR",
        "L1_L2_FE",
        "L1_L2_LB",
        "L2_L3_AR",
        "L2_L3_FE",
        "L2_L3_LB",
        "L3_L4_AR",
        "L3_L4_FE",
        "L3_L4_LB",
        "L4_L5_AR",
        "L4_L5_FE",
        "L4_L5_LB",
        "ankle_angle_l",
        "ankle_angle_r",
        "axial_rotation",
        "flex_extension",
        "hip_adduction_l",
        "hip_adduction_r",
        "hip_flexion_l",
        "hip_flexion_r",
        "hip_rotation_l",
        "hip_rotation_r",
        "knee_angle_l",
        "knee_angle_l_beta_rotation1",
        "knee_angle_l_beta_translation1",
        "knee_angle_l_beta_translation2",
        "knee_angle_l_rotation2",
        "knee_angle_l_rotation3",
        "knee_angle_l_translation1",
        "knee_angle_l_translation2",
        "knee_angle_r",
        "knee_angle_r_beta_rotation1",
        "knee_angle_r_beta_translation1",
        "knee_angle_r_beta_translation2",
        "knee_angle_r_rotation2",
        "knee_angle_r_rotation3",
        "knee_angle_r_translation1",
        "knee_angle_r_translation2",
        "lat_bending",
        "mtp_angle_l",
        "mtp_angle_r",
        "subtalar_angle_l",
        "subtalar_angle_r",
    ]

    def __init__(
        self,
        model_path: str,
        obsd_model_path: str | None = None,
        seed: int | None = None,
        frame_skip: int = 10,
        obs_keys: list = DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict[str, float] = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        normalize_act: bool = True,
        muscle_condition: str = "",
        reset_type: str = "none",
        min_agent_spawn_distance: float = 1.0,
        random_vel_range: tuple = (1.0, 5.0),
        rnd_pos_noise: float = 1.0,
        rnd_joint_noise: float = 0.02,
        goalkeeper_probabilities: tuple = (0.1, 0.45, 0.45),
        max_time_sec: float = 10.0,
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
            min_agent_spawn_distance=min_agent_spawn_distance,
            random_vel_range=random_vel_range,
            rnd_pos_noise=rnd_pos_noise,
            rnd_joint_noise=rnd_joint_noise,
            goalkeeper_probabilities=goalkeeper_probabilities,
            max_time_sec=max_time_sec,
            **kwargs,
        )

        # ── Load model ──────────────────────────────────────────────────────
        warn_torso_pip_calibration_divergence()
        self.model, self._mj_spec = ModelBuilder.from_xml_file(model_path).build()
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * frame_skip)
        self.dt = self._ctrl_dt

        # The pinned myo_sim leg model uses a simplified hinge knee and lacks
        # the passive coupling DOFs (e.g. "knee_angle_l_beta_rotation1") some
        # legacy joint lists assume — keep only joints this model has.
        _model_joints = self._joint_names_in_model()
        self.MYO_JOINTS = [j for j in self.MYO_JOINTS if j in _model_joints]
        self.JNT_OVEREXT = [j for j in self.JNT_OVEREXT if j in _model_joints]

        # Toe geom naming flipped from "r_bofoot"/"l_bofoot" (legacy) to
        # "bofoot_r"/"bofoot_l" in the current myo_sim leg model.
        _geom_names = {self.model.geom(i).name for i in range(self.model.ngeom)}
        self._toe_geom_r = "r_bofoot" if "r_bofoot" in _geom_names else "bofoot_r"
        self._toe_geom_l = "l_bofoot" if "l_bofoot" in _geom_names else "bofoot_l"

        # ── Muscle condition ────────────────────────────────────────────────
        self.muscle_condition = muscle_condition
        self._muscle_act_ind = self.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        if muscle_condition == "sarcopenia":
            apply_sarcopenia_to_model(self.model, force_scale=0.5)
        elif muscle_condition == "fatigue":
            self.muscle_fatigue = CumulativeFatigue(self.model, frame_skip, seed=None)

        # ── Task config ─────────────────────────────────────────────────────
        self.reset_type = reset_type
        self.min_agent_spawn_distance = min_agent_spawn_distance
        self.rnd_pos_noise = rnd_pos_noise
        self.rnd_joint_noise = rnd_joint_noise
        self.max_time = max_time_sec

        # ── Model element IDs ────────────────────────────────────────────────
        mujoco.mj_forward(self.model, self.data)
        self.soccer_ball_id = self.model.body("soccer_ball").id
        self.grf_sensor_names = ["r_foot", "r_toes", "l_foot", "l_toes"]

        # ── Goalkeeper ───────────────────────────────────────────────────────
        self.goalkeeper = GoalKeeper(
            mj_model=self.model,
            mj_data=self.data,
            rng=self.np_random,
            probabilities=goalkeeper_probabilities,
            random_vel_range=random_vel_range,
        )
        self.goalkeeper.dt = self._ctrl_dt

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

        # ── Action space ─────────────────────────────────────────────────────
        self.normalize_act = normalize_act
        if normalize_act:
            act_low = -np.ones(self.model.nu, dtype=np.float32)
            act_high = np.ones(self.model.nu, dtype=np.float32)
        else:
            act_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
            act_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)

        # ── State ─────────────────────────────────────────────────────────────
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

        self._assert_settings()
        self.startFlag = True

    # ── Action application ────────────────────────────────────────────────────

    def _apply_action(self, action: np.ndarray) -> None:
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
        self.data.ctrl[:] = ctrl

    # ── MyoGymnasiumEnv interface ─────────────────────────────────────────────

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        """Return full observation dict; _obs_dict_to_vec filters to obs_keys."""
        data = accessor.data
        obs_dict: dict[str, np.ndarray] = {}
        obs_dict["time"] = np.array([data.time])
        obs_dict["internal_qpos"] = self._get_joint_qpos()
        obs_dict["internal_qvel"] = self._get_joint_qvel() * self.dt
        obs_dict["grf"] = self._get_grf()
        obs_dict["torso_angle"] = data.body("torso").xquat.copy()
        obs_dict["pelvis_angle"] = data.body("pelvis").xquat.copy()
        obs_dict["muscle_length"] = data.actuator_length.copy()
        obs_dict["muscle_velocity"] = np.clip(data.actuator_velocity, -100, 100)
        obs_dict["muscle_force"] = np.clip(data.actuator_force / 1000, -100, 100)
        obs_dict["r_toe_pos"] = data.geom(self._toe_geom_r).xpos.copy()
        obs_dict["l_toe_pos"] = data.geom(self._toe_geom_l).xpos.copy()
        if self.model.na > 0:
            obs_dict["act"] = data.act[:].copy()
        obs_dict["ball_pos"] = data.body("soccer_ball").xpos[:3].copy()
        obs_dict["goal_bounds"] = np.array(
            [
                [self.GOAL_X_POS, self.GOAL_Y_MIN, self.GOAL_Z_MIN],
                [self.GOAL_X_POS, self.GOAL_Y_MAX, self.GOAL_Z_MIN],
                [self.GOAL_X_POS, self.GOAL_Y_MIN, self.GOAL_Z_MAX],
                [self.GOAL_X_POS, self.GOAL_Y_MAX, self.GOAL_Z_MAX],
            ]
        ).flatten()
        obs_dict["model_root_pos"] = data.joint("root").qpos.copy()
        obs_dict["model_root_vel"] = data.joint("root").qvel.copy()
        obs_dict["goalkeeper_pos"] = self.goalkeeper.get_goalkeeper_pose()[:2].copy()
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
        goal_scored = self._goal_scored_condition()
        time_limit_exceeded = float(np.atleast_1d(obs_dict["time"])[0]) >= self.max_time
        fallen = self._get_fallen_condition()
        pain = self._get_jnt_limit_violation()
        done = bool(goal_scored or time_limit_exceeded or fallen)

        rwd_dict = collections.OrderedDict(
            (
                ("goal_scored", float(goal_scored)),
                ("time_cost", 1.0),
                ("act_reg", act_mag),
                ("pain", pain),
                ("alive", 0.0 if done else 1.0),
                ("sparse", float(done)),
                ("solved", float(goal_scored)),
                ("done", done),
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
        self.goalkeeper.update_goalkeeper_state()
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
    ) -> tuple[np.ndarray, dict[str, Any]]:
        import gymnasium as _gym

        _gym.Env.reset(self, seed=seed)

        qpos, qvel = self._get_reset_state()
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

        self.goalkeeper.reset_goalkeeper(rng=self.np_random)
        mujoco.mj_forward(self.model, self.data)

        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs_dict = self._get_obs_dict(self._accessor)
        obs = self._obs_dict_to_vec(obs_dict)
        obs = self._ensure_obs_gymnasium_compliant(obs)
        return obs, {}

    # ── Terminal conditions ───────────────────────────────────────────────────

    def _goal_scored_condition(self) -> bool:
        ball_pos = self.data.body(self.soccer_ball_id).xpos.copy()
        return bool(
            ball_pos[0] >= self.GOAL_X_POS
            and self.GOAL_Y_MIN <= ball_pos[1] <= self.GOAL_Y_MAX
            and self.GOAL_Z_MIN <= ball_pos[2] <= self.GOAL_Z_MAX
        )

    def _get_fallen_condition(self) -> int:
        return 1 if self.data.body("pelvis").xpos[2] < 0.2 else 0

    def _get_jnt_limit_violation(self) -> float:
        """Normalised joint-limit violation force as pain score."""
        if not self.startFlag:
            return -1.0
        limit_score = 0.0
        for joint in self.JNT_OVEREXT:
            limit_score += (
                np.abs(np.clip(self._get_limitfrc(joint).squeeze(), -1000, 1000)) / 1000
            )
        return limit_score / len(self.JNT_OVEREXT)

    def _joint_names_in_model(self) -> set[str]:
        return {self.model.joint(i).name for i in range(self.model.njnt)}

    def _get_limitfrc(self, joint_name: str) -> np.ndarray:
        non_jnt_idxs = np.where(
            self.data.efc_type != mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
        )[0]
        only_jnt_lim = self.data.efc_force.copy()
        only_jnt_lim[non_jnt_idxs] = 0.0
        joint_force = np.zeros(self.model.nv)
        mujoco.mj_mulJacTVec(self.model, self.data, joint_force, only_jnt_lim)
        return joint_force[self.model.joint(joint_name).dofadr]

    # ── Observation helpers ───────────────────────────────────────────────────

    def _get_joint_qpos(self) -> np.ndarray:
        return np.array([self.data.joint(j).qpos[0].copy() for j in self.MYO_JOINTS])

    def _get_joint_qvel(self) -> np.ndarray:
        return np.array([self.data.joint(j).qvel[0].copy() for j in self.MYO_JOINTS])

    def _get_grf(self) -> np.ndarray:
        return np.array(
            [self.data.sensor(n).data[0] for n in self.grf_sensor_names]
        ).copy()

    # ── Reset state helpers ───────────────────────────────────────────────────

    def _get_reset_state(self) -> tuple[np.ndarray, np.ndarray]:
        if self.reset_type == "random":
            return self._get_randomized_initial_state()
        return self.model.key_qpos[0].copy(), self.model.key_qvel[0].copy()

    def _get_randomized_initial_state(self) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[:] = self.model.key_qpos[0].copy()
        self.data.qvel[:] = self.model.key_qvel[0].copy()
        for jnt in self.MYO_JOINTS:
            self.data.joint(jnt).qpos[0] += self.np_random.uniform(
                -abs(self.rnd_joint_noise), abs(self.rnd_joint_noise)
            )
        self.data.joint("root").qpos[0] += self.np_random.uniform(
            -abs(self.rnd_pos_noise), 0
        )
        self.data.joint("root").qpos[1] += self.np_random.uniform(
            -abs(self.rnd_pos_noise), abs(self.rnd_pos_noise)
        )
        return self.data.qpos.copy(), self.data.qvel.copy()

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_metrics(self, paths: list) -> dict[str, float]:
        """Compute aggregate metrics over rollout paths."""
        return {
            "score": np.mean(
                [np.sum(p["env_infos"]["rwd_dict"]["solved"]) for p in paths]
            ),
            "points": np.mean(
                [np.sum(p["env_infos"]["rwd_dict"]["sparse"]) for p in paths]
            ),
            "times": np.mean(
                [np.round(p["env_infos"]["obs_dict"]["time"][-1], 2) for p in paths]
            ),
            "effort": np.mean(
                [np.sum(p["env_infos"]["rwd_dict"]["act_reg"]) for p in paths]
            ),
            "pain": np.mean(
                [np.sum(p["env_infos"]["rwd_dict"]["pain"]) for p in paths]
            ),
        }

    # ── Validation ────────────────────────────────────────────────────────────

    def _assert_settings(self) -> None:
        for prob in self.goalkeeper.goalkeeper_probabilities:
            assert 0 <= prob <= 1, f"Goalkeeper probability out of [0, 1]: {prob}"
        assert np.isclose(
            np.sum(self.goalkeeper.goalkeeper_probabilities), 1.0
        ), "Goalkeeper probabilities must sum to 1.0"
        assert (
            self.goalkeeper.random_vel_range[0] >= 0
        ), "Goalkeeper block_vel_range min must be >= 0"
        assert (
            self.goalkeeper.random_vel_range[0] <= self.goalkeeper.random_vel_range[1]
        ), "Goalkeeper block_vel_range min must be <= max"
