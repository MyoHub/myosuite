# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""ChaseTag challenge environment."""

from __future__ import annotations

import collections
from enum import Enum
from typing import Any

import mujoco
import numpy as np
import pink

import gymnasium as gym
from gymnasium.utils import EzPickle

from myosuite.core.model_builder import ModelBuilder
from myosuite.core.muscle_conditions import apply_sarcopenia_to_model
from myosuite.envs.gymnasium_env import CpuEnvAccessor, MyoGymnasiumEnv
from myosuite.envs.heightfields import ChaseTagField
from myosuite.physics.fatigue import CumulativeFatigue
from myosuite.physics.quat_math import euler2quat, quat2euler, quat2mat
from myosuite.terms.base_action import sigmoid_muscle_activation


class Task(Enum):
    CHASE = 0
    EVADE = 1


class ChallengeOpponent:
    """Training opponent for the ChaseTag locomotion challenge.

    Contains several control policies. For final evaluation an additional
    non-disclosed policy is used.

    Args:
        mj_model: MuJoCo model object.
        mj_data: MuJoCo data object.
        rng: NumPy random generator.
        probabilities: Policy probabilities (static_stationary, stationary, random).
        min_spawn_distance: Minimum spawn radius from agent.
        chase_vel_range: Velocity range for the chase policy.
        random_vel_range: Velocity range for the random policy.
        dt: Control timestep.
    """

    def __init__(
        self,
        mj_model,
        mj_data,
        rng,
        probabilities: tuple[float, ...],
        min_spawn_distance: float,
        chase_vel_range: tuple[float, float],
        random_vel_range: tuple[float, float],
        dt: float = 0.01,
    ) -> None:
        self.dt = dt
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.opponent_probabilities = probabilities
        self.min_spawn_distance = min_spawn_distance
        self.chase_vel_range = chase_vel_range
        self.random_vel_range = random_vel_range
        self.reset_opponent(rng=rng)

    def reset_noise_process(self) -> None:
        self.noise_process = pink.ColoredNoiseProcess(
            beta=2, size=(2, 2000), scale=10, rng=self.rng
        )

    def get_opponent_pose(self) -> np.ndarray:
        """Return opponent pose as [x, y, angle]."""
        angle = quat2euler(self.mj_data.mocap_quat[0, :])[-1]
        return np.concatenate([self.mj_data.mocap_pos[0, :2], [angle]])

    def set_opponent_pose(self, pose: list) -> None:
        """Teleport opponent to given pose [x, y, angle]."""
        self.mj_data.mocap_pos[0, :2] = pose[:2]
        self.mj_data.mocap_quat[0, :] = euler2quat([0, 0, pose[-1]])

    def move_opponent(self, vel: list) -> None:
        """Move opponent by velocity [lin_vel, rot_vel]."""
        self.opponent_vel = vel
        assert len(vel) == 2
        vel[0] = np.abs(vel[0])
        vel = np.clip(vel, -2, 2)
        pose = self.get_opponent_pose()
        x_vel = vel[0] * np.cos(pose[-1] + 0.5 * np.pi)
        y_vel = vel[0] * np.sin(pose[-1] + 0.5 * np.pi)
        pose[0] -= self.dt * x_vel
        pose[1] -= self.dt * y_vel
        pose[2] += self.dt * vel[1]
        pose[:2] = np.clip(pose[:2], -5.5, 5.5)
        self.set_opponent_pose(pose)

    def random_movement(self) -> np.ndarray:
        return np.clip(
            self.noise_process.sample(),
            self.random_vel_range[0],
            self.random_vel_range[1],
        )

    def sample_opponent_policy(self) -> None:
        rand_num = self.rng.uniform()
        if rand_num < self.opponent_probabilities[0]:
            self.opponent_policy = "static_stationary"
        elif rand_num < self.opponent_probabilities[0] + self.opponent_probabilities[1]:
            self.opponent_policy = "stationary"
        elif (
            rand_num
            < self.opponent_probabilities[0]
            + self.opponent_probabilities[1]
            + self.opponent_probabilities[2]
        ):
            self.opponent_policy = "random"

    def update_opponent_state(self) -> None:
        """Execute one opponent step using the active policy."""
        if self.opponent_policy in ("stationary", "static_stationary"):
            opponent_vel = np.zeros(2)
        elif self.opponent_policy == "random":
            opponent_vel = self.random_movement()
        elif self.opponent_policy == "chase_player":
            opponent_vel = self.chase_player()
        else:
            raise NotImplementedError(
                f"Unknown opponent policy: {self.opponent_policy}"
            )
        self.move_opponent(opponent_vel)

    def reset_opponent(self, player_task: str = "CHASE", rng=None) -> None:
        """Place opponent at a random position with minimum radius from agent."""
        if rng is not None:
            self.rng = rng
            self.reset_noise_process()

        self.opponent_vel = np.zeros(2)
        if player_task == "CHASE":
            self.sample_opponent_policy()
        elif player_task == "EVADE":
            self.opponent_policy = "chase_player"
        else:
            raise NotImplementedError

        dist = 0
        while dist < self.min_spawn_distance:
            pose = [
                self.rng.uniform(-5, 5),
                self.rng.uniform(-5, 5),
                self.rng.uniform(-2 * np.pi, 2 * np.pi),
            ]
            dist = np.linalg.norm(
                np.array(pose[:2]) - self.mj_data.body("root").xpos[:2]
            )
        if self.opponent_policy == "static_stationary":
            pose[:] = [0, -5, 0]
        self.set_opponent_pose(pose)
        self.opponent_vel[:] = 0.0

        self.chase_velocity = self.rng.uniform(
            self.chase_vel_range[0], self.chase_vel_range[1]
        )

    def chase_player(self) -> np.ndarray:
        pose = self.get_opponent_pose()
        vec = pose[:2]
        pel = self.mj_data.body("pelvis").xpos[:2]
        theta = pose[-1]
        new_vec = np.array([np.cos(theta), np.sin(theta)])
        new_vec2 = pel - vec
        vel = np.dot(new_vec, new_vec2)
        return np.array([self.chase_velocity, vel])


class RepellerChallengeOpponent(ChallengeOpponent):
    """Opponent that actively repels away from the agent and arena boundaries.

    Args:
        repeller_vel_range: Velocity range for the repeller policy.
        probabilities: Four probabilities (static_stationary, stationary, random, repeller).
    """

    DIST_INFLUENCE = 3.5
    ETA = 20.0
    MIN_SPAWN_DIST = 1.5
    BOUND_RESOLUTIONS = [-8.7, 8.7, 25]

    def __init__(
        self,
        mj_model,
        mj_data,
        rng,
        probabilities: tuple[float, ...],
        min_spawn_distance: float,
        chase_vel_range: tuple[float, float],
        random_vel_range: tuple[float, float],
        repeller_vel_range: tuple[float, float],
        dt: float = 0.01,
    ) -> None:
        self.dt = dt
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.rng = rng
        self.opponent_probabilities = probabilities
        self.min_spawn_distance = min_spawn_distance
        self.noise_process = pink.ColoredNoiseProcess(
            beta=2, size=(2, 2000), scale=10, rng=rng
        )
        self.chase_vel_range = chase_vel_range
        self.random_vel_range = random_vel_range
        self.repeller_vel_range = repeller_vel_range
        self.reset_opponent()

    def get_agent_pos(self) -> np.ndarray:
        return self.mj_data.body("pelvis").xpos[:2]

    def get_wall_pos(self) -> np.ndarray:
        bound_resolution = np.linspace(
            self.BOUND_RESOLUTIONS[0],
            self.BOUND_RESOLUTIONS[1],
            self.BOUND_RESOLUTIONS[2],
        )
        right_left_bounds = np.vstack(
            (
                np.array([[8.7, x] for x in bound_resolution]),
                np.array([[-8.7, x] for x in bound_resolution]),
            )
        )
        return np.vstack((right_left_bounds, right_left_bounds[:, [1, 0]]))

    def get_repellers(self) -> np.ndarray:
        return np.vstack((self.get_agent_pos(), self.get_wall_pos()))

    def repeller_stochastic(self) -> np.ndarray:
        obstacle_pos = self.get_repellers()
        opponent_pos = self.get_opponent_pose().copy()
        distance = np.array(
            [np.linalg.norm(diff) for diff in (obstacle_pos - opponent_pos[:2])]
        )
        dist_idx = np.where(distance < self.DIST_INFLUENCE)[0]
        if len(dist_idx) == 0:
            lin, rot = self.noise_process.sample()
            return np.hstack(
                (
                    np.clip(
                        lin, self.repeller_vel_range[0], self.repeller_vel_range[1]
                    ),
                    self._calc_angular_vel(opponent_pos[2], rot),
                )
            )
        repel_COM = np.mean(obstacle_pos[dist_idx, :], axis=0)
        repel_force = (
            0.5
            * self.ETA
            * (1 / np.maximum(distance[dist_idx], 0.00001) - 1 / self.DIST_INFLUENCE)
            ** 2
        )
        escape_linear = np.clip(
            np.mean(repel_force), self.repeller_vel_range[0], self.repeller_vel_range[1]
        )
        escape_xpos = opponent_pos[:2] - repel_COM
        equil_idx = np.where(np.abs(escape_xpos) <= 0.1)[0]
        for idx in equil_idx:
            escape_xpos[idx] = (
                -1 * np.sign(escape_xpos[idx]) * self.rng.uniform(low=0.3, high=0.9)
            )
        escape_direction = np.arctan2(escape_xpos[1], escape_xpos[0]) + 1.57
        return np.hstack(
            (escape_linear, self._calc_angular_vel(opponent_pos[2], escape_direction))
        )

    def _calc_angular_vel(self, current_pos: float, desired_pos: float) -> int:
        for pos in (current_pos, desired_pos):
            pass  # normalised below
        while current_pos > 2 * np.pi:
            current_pos -= 2 * np.pi
        while current_pos < 0:
            current_pos += 2 * np.pi
        while desired_pos > 2 * np.pi:
            desired_pos -= 2 * np.pi
        while desired_pos < 0:
            desired_pos += 2 * np.pi
        direction_clock = np.abs(0 - current_pos) + (2 * np.pi - desired_pos)
        direction_anticlock = (2 * np.pi - current_pos) + desired_pos
        return 1 if direction_clock < direction_anticlock else -1

    def repeller_policy(self) -> np.ndarray:
        return self.repeller_stochastic()

    def sample_opponent_policy(self) -> None:
        rand_num = self.rng.uniform()
        cum = 0.0
        for policy, prob in zip(
            ("static_stationary", "stationary", "random", "repeller"),
            self.opponent_probabilities,
        ):
            cum += prob
            if rand_num < cum:
                self.opponent_policy = policy
                return

    def update_opponent_state(self) -> None:
        if self.opponent_policy in ("stationary", "static_stationary"):
            opponent_vel = np.zeros(2)
        elif self.opponent_policy == "random":
            opponent_vel = self.random_movement()
        elif self.opponent_policy == "repeller":
            opponent_vel = self.repeller_policy()
        elif self.opponent_policy == "chase_player":
            opponent_vel = self.chase_player()
        else:
            raise NotImplementedError(
                f"Unknown opponent policy: {self.opponent_policy}"
            )
        self.move_opponent(opponent_vel)


class ChaseTagEnv(MyoGymnasiumEnv, EzPickle):
    """Chase-Tag locomotion challenge — native MyoGymnasiumEnv implementation.

    The agent must either chase and tag the opponent (CHASE) or evade it
    for the full episode duration (EVADE).

    Args:
        model_path: Absolute path to the MuJoCo XML model.
        obsd_model_path: Unused (kept for API compatibility).
        seed: Random seed.
        frame_skip: MuJoCo substeps per :meth:`step` call.
        obs_keys: Observation keys to include.
        weighted_reward_keys: Dict of ``{key: weight}`` for dense reward.
        normalize_act: If ``True``, action space is ``[-1, 1]``.
        muscle_condition: One of ``""``, ``"sarcopenia"``, ``"fatigue"``.
        reset_type: One of ``"none"``, ``"init"``, ``"random"``.
        win_distance: Tagging distance threshold (m).
        min_spawn_distance: Minimum opponent spawn radius (m).
        task_choice: ``"CHASE"``, ``"EVADE"``, or ``"random"``.
        terrain: ``"FLAT"`` or ``"random"``.
        hills_range: Height range for hill terrain.
        rough_range: Height range for rough terrain.
        relief_range: Height range for relief terrain.
        repeller_opponent: If ``True`` use the repeller opponent variant.
        chase_vel_range: Chase policy velocity range.
        random_vel_range: Random policy velocity range.
        repeller_vel_range: Repeller policy velocity range.
        opponent_probabilities: Per-policy probabilities.
    """

    DEFAULT_OBS_KEYS = [
        "internal_qpos",
        "internal_qvel",
        "grf",
        "torso_angle",
        "opponent_pose",
        "opponent_vel",
        "model_root_pos",
        "model_root_vel",
        "muscle_length",
        "muscle_velocity",
        "muscle_force",
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS: dict[str, float] = {
        "distance": -0.1,
        "lose": -1000,
    }

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
        win_distance: float = 0.5,
        min_spawn_distance: float = 2.0,
        task_choice: str = "CHASE",
        terrain: str = "FLAT",
        hills_range: tuple = (0, 0),
        rough_range: tuple = (0, 0),
        relief_range: tuple = (0, 0),
        repeller_opponent: bool = False,
        chase_vel_range: tuple = (1.0, 1.0),
        random_vel_range: tuple = (1.0, 1.0),
        repeller_vel_range: tuple = (1.0, 1.0),
        opponent_probabilities: tuple = (0.1, 0.45, 0.45),
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
            win_distance=win_distance,
            min_spawn_distance=min_spawn_distance,
            task_choice=task_choice,
            terrain=terrain,
            hills_range=hills_range,
            rough_range=rough_range,
            relief_range=relief_range,
            repeller_opponent=repeller_opponent,
            chase_vel_range=chase_vel_range,
            random_vel_range=random_vel_range,
            repeller_vel_range=repeller_vel_range,
            opponent_probabilities=opponent_probabilities,
            **kwargs,
        )

        # ── Load model ──────────────────────────────────────────────────────
        self.model, self._mj_spec = ModelBuilder.from_xml_file(model_path).build()
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * frame_skip)
        self.dt = self._ctrl_dt  # backward-compat alias

        # ── Muscle condition ────────────────────────────────────────────────
        self.muscle_condition = muscle_condition
        self._muscle_act_ind = self.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        if muscle_condition == "sarcopenia":
            apply_sarcopenia_to_model(self.model, force_scale=0.5)
        elif muscle_condition == "fatigue":
            self.muscle_fatigue = CumulativeFatigue(self.model, frame_skip, seed=None)

        # ── Task config ─────────────────────────────────────────────────────
        self.reset_type = reset_type
        self.task_choice = task_choice
        self.terrain = terrain
        self.win_distance = win_distance
        self.maxTime = 20
        self.repeller_opponent = repeller_opponent

        # ── Terrain ─────────────────────────────────────────────────────────
        mujoco.mj_forward(self.model, self.data)
        self.heightfield = (
            ChaseTagField(
                mj_model=self.model,
                mj_data=self.data,
                rng=self.np_random,
                rough_range=rough_range,
                hills_range=hills_range,
                relief_range=relief_range,
            )
            if terrain != "FLAT"
            else None
        )

        # ── Opponent ────────────────────────────────────────────────────────
        if repeller_opponent:
            self.opponent = RepellerChallengeOpponent(
                mj_model=self.model,
                mj_data=self.data,
                rng=self.np_random,
                probabilities=opponent_probabilities,
                min_spawn_distance=min_spawn_distance,
                chase_vel_range=chase_vel_range,
                random_vel_range=random_vel_range,
                repeller_vel_range=repeller_vel_range,
            )
        else:
            self.opponent = ChallengeOpponent(
                mj_model=self.model,
                mj_data=self.data,
                rng=self.np_random,
                probabilities=opponent_probabilities,
                min_spawn_distance=min_spawn_distance,
                chase_vel_range=chase_vel_range,
                random_vel_range=random_vel_range,
            )
        self.opponent.dt = self._ctrl_dt

        # ── State ────────────────────────────────────────────────────────────
        self.grf_sensor_names = ["r_foot", "r_toes", "l_foot", "l_toes"]
        self.success_indicator_sid = self.model.site("opponent_indicator").id
        self.current_task = Task.CHASE
        self.steps = 0
        # Prevents win/lose checks before first reset sets keyframe
        self.startFlag = False

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

        # ── Initial state keyframe ───────────────────────────────────────────
        self._init_qpos = self.model.key_qpos[0].copy()
        self._init_qvel = np.zeros(self.model.nv)
        self._input_seed = seed

        import gymnasium as _gym

        _gym.Env.reset(self, seed=seed)

        # ── Observation space (sized from initial forward pass) ───────────────
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
        """Map action to MuJoCo ctrl, applying sigmoid for muscle actuators."""
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
        """Return full observation dict; ``_obs_dict_to_vec`` filters to obs_keys."""
        model, data = accessor.model, accessor.data
        obs_dict: dict[str, np.ndarray] = {}
        # Always include time — needed by win/lose conditions in get_reward_dict
        obs_dict["time"] = np.array([data.time])
        obs_dict["internal_qpos"] = data.qpos[7:35].copy()
        obs_dict["internal_qvel"] = data.qvel[6:34].copy() * self.dt
        obs_dict["grf"] = self._get_grf()
        obs_dict["torso_angle"] = data.body("pelvis").xquat.copy()
        obs_dict["muscle_length"] = self._muscle_lengths()
        obs_dict["muscle_velocity"] = self._muscle_velocities()
        obs_dict["muscle_force"] = self._muscle_forces()
        if model.na > 0:
            obs_dict["act"] = data.act[:].copy()
        obs_dict["opponent_pose"] = self.opponent.get_opponent_pose().copy()
        obs_dict["opponent_vel"] = self.opponent.opponent_vel.copy()
        obs_dict["model_root_pos"] = data.qpos[:2].copy()
        obs_dict["model_root_vel"] = data.qvel[:2].copy()
        obs_dict["task"] = np.array(self.current_task.value, ndmin=2, dtype=np.int16)
        if self.heightfield is not None:
            obs_dict["hfield"] = self.heightfield.get_heightmap_obs()
        return obs_dict

    def _obs_dict_to_vec(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Flatten only the obs_keys subset of obs_dict to a 1-D vector."""
        return np.concatenate(
            [np.atleast_1d(obs_dict[k]).ravel() for k in self.obs_keys if k in obs_dict]
        )

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        act_mag = (
            np.linalg.norm(obs_dict.get("act", np.zeros(1)), axis=-1) / self.model.na
            if self.model.na != 0
            else 0
        )
        win_cdt = self._win_condition(obs_dict)
        lose_cdt = self._lose_condition(obs_dict)
        if self.current_task.name == "CHASE":
            score = (
                self._get_score(float(np.atleast_1d(obs_dict["time"])[0]))
                if win_cdt
                else 0
            )
        else:  # EVADE
            score = (
                self._get_score(float(np.atleast_1d(obs_dict["time"])[0]))
                if (win_cdt or lose_cdt)
                else 0
            )

        distance = np.linalg.norm(
            obs_dict["model_root_pos"][..., :2] - obs_dict["opponent_pose"][..., :2]
        )
        rwd_dict = collections.OrderedDict(
            (
                ("act_reg", act_mag),
                ("distance", distance),
                ("lose", lose_cdt),
                ("sparse", score),
                ("solved", win_cdt),
                ("done", self._get_done(obs_dict)),
            )
        )
        rwd_dict["dense"] = float(
            np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        )
        self.model.site_rgba[self.success_indicator_sid, :] = (
            np.array([0, 2, 0, 0.2]) if rwd_dict["solved"] else np.array([2, 0, 0, 0])
        )
        return rwd_dict

    # ── Gymnasium step / reset ───────────────────────────────────────────────

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance simulation: update opponent, apply action, step physics."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.opponent.update_opponent_state()
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data, self.frame_skip)
        mujoco.mj_kinematics(self.model, self.data)
        self.steps += 1
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
        """Reset: sample terrain, sample task, set initial state, reset opponent."""
        import gymnasium as _gym

        _gym.Env.reset(self, seed=seed)

        self._maybe_sample_terrain()
        self._sample_task()
        qpos, qvel = self._get_reset_state()

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        self._maybe_flatten_agent_patch(qpos)

        self.steps = 0
        self.opponent.reset_opponent(
            player_task=self.current_task.name, rng=self.np_random
        )
        mujoco.mj_forward(self.model, self.data)

        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs_dict = self._get_obs_dict(self._accessor)
        obs = self._obs_dict_to_vec(obs_dict)
        obs = self._ensure_obs_gymnasium_compliant(obs)
        return obs, {}

    # ── Biomechanical helpers (ported from WalkEnvV0) ────────────────────────

    def _muscle_lengths(self) -> np.ndarray:
        return self.data.actuator_length.copy()

    def _muscle_forces(self) -> np.ndarray:
        return np.clip(self.data.actuator_force / 1000, -100, 100)

    def _muscle_velocities(self) -> np.ndarray:
        return np.clip(self.data.actuator_velocity, -100, 100)

    def _get_height(self) -> float:
        return float(self._get_com()[2])

    def _get_com(self) -> np.ndarray:
        mass = np.expand_dims(self.model.body_mass, -1)
        return np.sum(mass * self.data.xipos, 0) / np.sum(mass)

    def _get_rot_condition(self) -> int:
        quat = self.data.qpos[3:7].copy()
        return 1 if np.abs((quat2mat(quat) @ [1, 0, 0])[0]) > 0.8 else 0

    def _get_grf(self) -> np.ndarray:
        return np.array(
            [self.data.sensor(n).data[0] for n in self.grf_sensor_names]
        ).copy()

    def _get_angle(self, names: list[str]) -> np.ndarray:
        return np.array(
            [
                self.data.qpos[self.model.jnt_qposadr[self.model.joint(n).id]]
                for n in names
            ]
        )

    # ── Terminal / win / lose conditions ─────────────────────────────────────

    def _get_done(self, obs_dict: dict) -> int:
        if not self.startFlag:
            return 0
        if self._lose_condition(obs_dict):
            return 1
        if self._win_condition(obs_dict):
            return 1
        return 0

    def _win_condition(self, obs_dict: dict) -> int:
        if not self.startFlag:
            return 0
        if self.current_task.name == "CHASE":
            return self._chase_win_condition(obs_dict)
        elif self.current_task.name == "EVADE":
            return self._evade_win_condition(obs_dict)
        raise NotImplementedError

    def _lose_condition(self, obs_dict: dict) -> int:
        if not self.startFlag:
            return 0
        if self._get_fallen_condition() and self.current_task.name == "CHASE":
            return 1
        if self.current_task.name == "CHASE":
            return self._chase_lose_condition(obs_dict)
        elif self.current_task.name == "EVADE":
            return self._evade_lose_condition(obs_dict)
        raise NotImplementedError

    def _chase_lose_condition(self, obs_dict: dict) -> int:
        root_pos = self.data.body("pelvis").xpos[:2]
        if float(np.atleast_1d(obs_dict["time"])[0]) >= self.maxTime:
            return 1
        if np.abs(root_pos[0]) > 6.5 or np.abs(root_pos[1]) > 6.5:
            return 1
        return 0

    def _evade_lose_condition(self, obs_dict: dict) -> int:
        root_pos = self.data.body("pelvis").xpos[:2]
        opp_pos = obs_dict["opponent_pose"][..., :2]
        if np.linalg.norm(root_pos - opp_pos) <= self.win_distance:
            return 1
        if np.abs(root_pos[0]) > 6.5 or np.abs(root_pos[1]) > 6.5:
            return 1
        return 0

    def _chase_win_condition(self, obs_dict: dict) -> int:
        root_pos = self.data.body("pelvis").xpos[:2]
        opp_pos = obs_dict["opponent_pose"][..., :2]
        if np.linalg.norm(root_pos - opp_pos) <= self.win_distance:
            return 1
        return 0

    def _evade_win_condition(self, obs_dict: dict) -> int:
        if float(np.atleast_1d(obs_dict["time"])[0]) >= self.maxTime:
            return 1
        return 0

    def _get_fallen_condition(self) -> int:
        if self.terrain == "FLAT":
            return 1 if self.data.body("pelvis").xpos[2] < 0.5 else 0
        head = self.data.site("head").xpos
        foot_l = self.data.body("talus_l").xpos
        foot_r = self.data.body("talus_r").xpos
        mean = (foot_l + foot_r) / 2
        return 1 if head[2] - mean[2] < 0.2 else 0

    # ── Task / terrain sampling ───────────────────────────────────────────────

    def _sample_task(self) -> None:
        if self.task_choice == "random":
            self.current_task = self.np_random.choice(list(Task))
        else:
            self.current_task = getattr(Task, self.task_choice)

    def _maybe_sample_terrain(self) -> None:
        if self.heightfield is not None:
            self.heightfield.sample(self.np_random)
            self.model.geom_rgba[self.model.geom("terrain").id][-1] = 1.0
            self.model.geom_pos[self.model.geom("terrain").id] = np.array([0, 0, 0])
        else:
            self.model.geom_rgba[self.model.geom("terrain").id][-1] = 0.0
            self.model.geom_pos[self.model.geom("terrain").id] = np.array([0, 0, -10])

    def _maybe_flatten_agent_patch(self, qpos: np.ndarray) -> None:
        if self.heightfield is not None:
            self.heightfield.flatten_agent_patch(qpos)
            renderer = getattr(self, "_mujoco_renderer", None)
            if renderer is not None and getattr(renderer, "_window", None) is not None:
                renderer._window.update_hfield(0)

    # ── Reset state helpers ───────────────────────────────────────────────────

    def _get_reset_state(self) -> tuple[np.ndarray, np.ndarray]:
        if self.reset_type == "random":
            qpos, qvel = self._get_randomized_initial_state()
            return self._randomize_position_orientation(qpos, qvel)
        elif self.reset_type == "init":
            return self.model.key_qpos[2].copy(), self.model.key_qvel[2].copy()
        else:
            return self.model.key_qpos[0].copy(), self.model.key_qvel[0].copy()

    def _get_randomized_initial_state(self) -> tuple[np.ndarray, np.ndarray]:
        if self.np_random.uniform() < 0.5:
            qpos = self.model.key_qpos[2].copy()
            qvel = self.model.key_qvel[2].copy()
        else:
            qpos = self.model.key_qpos[3].copy()
            qvel = self.model.key_qvel[3].copy()
        rot_state = qpos[3:7].copy()
        height = qpos[2]
        qpos[:] += self.np_random.normal(0, 0.02, size=qpos.shape)
        qpos[3:7] = rot_state
        qpos[2] = height
        return qpos, qvel

    def _randomize_position_orientation(
        self, qpos: np.ndarray, qvel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        qpos[:2] = self.np_random.uniform(-5, 5, size=(2,))
        orientation = self.np_random.uniform(0, 2 * np.pi)
        euler_angle = quat2euler(qpos[3:7])
        euler_angle[-1] = orientation
        qpos[3:7] = euler2quat(euler_angle)
        qvel[:2] = np.array(
            [np.cos(orientation), np.sin(orientation)]
        ) * np.linalg.norm(qvel[:2])
        return qpos, qvel

    # ── Score / metrics ───────────────────────────────────────────────────────

    def _get_score(self, time: float) -> float:
        time = round(time, 2)
        if self.current_task.name == "CHASE":
            return 1 - (time / self.maxTime)
        elif self.current_task.name == "EVADE":
            return time / self.maxTime
        raise NotImplementedError

    def get_metrics(self, paths: list) -> dict[str, float]:
        """Compute aggregate metrics over rollout paths."""
        score = np.mean([np.sum(p["env_infos"]["rwd_dict"]["sparse"]) for p in paths])
        points = np.mean([np.sum(p["env_infos"]["rwd_dict"]["solved"]) for p in paths])
        times = np.mean(
            [np.round(p["env_infos"]["obs_dict"]["time"][-1], 2) for p in paths]
        )
        return {"score": score, "points": points, "times": times}

    # ── Validation ────────────────────────────────────────────────────────────

    def _assert_settings(self) -> None:
        assert (
            self.opponent.chase_vel_range[0] >= 0
            and self.opponent.chase_vel_range[1] > 0
        ), f"Chase velocity range must be positive: {self.opponent.chase_vel_range}"
        assert (
            self.opponent.chase_vel_range[0] <= self.opponent.chase_vel_range[1]
        ), f"Invalid chase velocity range: {self.opponent.chase_vel_range}"
        assert (
            self.opponent.random_vel_range[0] <= self.opponent.random_vel_range[1]
        ), f"Invalid random velocity range: {self.opponent.random_vel_range}"
        if hasattr(self.opponent, "repeller_vel_range"):
            assert (
                self.opponent.repeller_vel_range[0]
                <= self.opponent.repeller_vel_range[1]
            ), f"Invalid repeller velocity range: {self.opponent.repeller_vel_range}"
        expected_n = 4 if self.repeller_opponent else 3
        assert (
            len(self.opponent.opponent_probabilities) == expected_n
        ), f"Expected {expected_n} probabilities, got {len(self.opponent.opponent_probabilities)}"
        for p in self.opponent.opponent_probabilities:
            assert 0 <= p <= 1, f"Probability out of [0, 1]: {p}"
