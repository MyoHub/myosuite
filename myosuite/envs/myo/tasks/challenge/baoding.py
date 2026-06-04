# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import collections
import enum
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


class Task(enum.Enum):
    """Task variants for baoding ball rotation."""

    HOLD = 0
    BAODING_CW = 1
    BAODING_CCW = 2


# Default task for fixed task_choice
WHICH_TASK = Task.BAODING_CCW


class BaodingEnv(MyoGymnasiumEnv, EzPickle):
    """Baoding balls manipulation task for the musculoskeletal hand.

    The agent must rotate two balls in a circular pattern while keeping
    them within the hand. Supports fixed and random task variants.

    Args:
        model_path: Absolute path to the MuJoCo XML model.
        obsd_model_path: Unused (kept for API compatibility).
        seed: Random seed.
        frame_skip: Number of MuJoCo substeps per :meth:`step` call.
        drop_th: Height threshold (m) below which a ball is considered dropped.
        proximity_th: Distance threshold (m) for task success.
        goal_time_period: ``(min, max)`` range for rotation time period.
        goal_xrange: ``(min, max)`` range for x-axis orbit radius.
        goal_yrange: ``(min, max)`` range for y-axis orbit radius.
        obj_size_range: ``(min, max)`` range for ball radius.
        obj_mass_range: ``(min, max)`` range for ball mass (kg).
        obj_friction_change: Per-axis friction change from nominal.
        task_choice: ``"fixed"`` uses ``WHICH_TASK``; ``"random"`` samples each episode.
        obs_keys: Observation keys.
        weighted_reward_keys: Dict ``{reward_key: weight}`` for dense reward.
        normalize_act: If ``True``, action space is ``[-1, 1]``.
        muscle_condition: One of ``""``, ``"sarcopenia"``, ``"fatigue"``.
        fatigue_reset_vec: Initial fatigue state vector.
        fatigue_reset_random: If ``True``, randomise fatigue state on reset.
    """

    DEFAULT_OBS_KEYS = [
        "hand_pos",
        "object1_pos",
        "object1_velp",
        "object2_pos",
        "object2_velp",
        "target1_pos",
        "target2_pos",
        "target1_err",
        "target2_err",
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pos_dist_1": 5.0,
        "pos_dist_2": 5.0,
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
        frame_skip: int = 10,
        drop_th: float = 1.25,
        proximity_th: float = 0.015,
        goal_time_period: tuple = (5, 5),
        goal_xrange: tuple = (0.025, 0.025),
        goal_yrange: tuple = (0.028, 0.028),
        obj_size_range: tuple | None = None,
        obj_mass_range: tuple | None = None,
        obj_friction_change=None,
        task_choice: str = "fixed",
        obs_keys: list = DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict[str, float] = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        normalize_act: bool = True,
        muscle_condition: str = "",
        fatigue_reset_vec=None,
        fatigue_reset_random: bool = False,
        **kwargs: Any,
    ) -> None:
        MyoGymnasiumEnv.__init__(
            self, frame_skip=frame_skip, render_mode=kwargs.get("render_mode")
        )
        # EzPickle must be called AFTER MyoGymnasiumEnv.__init__ to avoid
        # the cooperative super() chain resetting _ezpickle_args to ().
        EzPickle.__init__(
            self,
            model_path,
            obsd_model_path,
            seed,
            frame_skip=frame_skip,
            drop_th=drop_th,
            proximity_th=proximity_th,
            goal_time_period=goal_time_period,
            goal_xrange=goal_xrange,
            goal_yrange=goal_yrange,
            obj_size_range=obj_size_range,
            obj_mass_range=obj_mass_range,
            obj_friction_change=obj_friction_change,
            task_choice=task_choice,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            normalize_act=normalize_act,
            muscle_condition=muscle_condition,
            fatigue_reset_vec=fatigue_reset_vec,
            fatigue_reset_random=fatigue_reset_random,
            **kwargs,
        )

        # ── Load model ─────────────────────────────────────────────────────
        self.model, self._mj_spec = ModelBuilder.from_xml_file(model_path).build()
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * frame_skip)

        # ── Muscle condition ───────────────────────────────────────────────
        self.muscle_condition = muscle_condition
        self.fatigue_reset_vec = fatigue_reset_vec
        self.fatigue_reset_random = fatigue_reset_random
        self._muscle_act_ind = self.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self._init_muscle_condition()

        # ── Task config ────────────────────────────────────────────────────
        self.task_choice = task_choice
        self.drop_th = drop_th
        self.proximity_th = proximity_th
        self.goal_time_period = goal_time_period
        self.goal_xrange = goal_xrange
        self.goal_yrange = goal_yrange

        # Starting angles: ball1 top-right, ball2 bottom-left
        self.ball_1_starting_angle = 1.0 * np.pi / 4.0
        self.ball_2_starting_angle = self.ball_1_starting_angle - np.pi

        # Orbit centre in palm frame
        self.center_pos = [-0.0125, -0.07]

        # ── Site / body IDs ────────────────────────────────────────────────
        mujoco.mj_forward(self.model, self.data)
        self.object1_bid = self.model.body("ball1").id
        self.object2_bid = self.model.body("ball2").id
        self.object1_sid = self.model.site("ball1_site").id
        self.object2_sid = self.model.site("ball2_site").id
        self.object1_gid = self.model.geom("ball1").id
        self.object2_gid = self.model.geom("ball2").id
        self.target1_sid = self.model.site("target1_site").id
        self.target2_sid = self.model.site("target2_site").id
        self.model.site_group[self.target1_sid] = 2
        self.model.site_group[self.target2_sid] = 2

        # ── Physical property randomisation ranges ─────────────────────────
        self.obj_mass_range = (
            {"low": obj_mass_range[0], "high": obj_mass_range[1]}
            if obj_mass_range is not None
            else None
        )
        self.obj_size_range = (
            {"low": obj_size_range[0], "high": obj_size_range[1]}
            if obj_size_range is not None
            else None
        )
        if obj_friction_change is not None:
            nominal = self.model.geom_friction[self.object1_gid].copy()
            change = np.asarray(obj_friction_change)
            self.obj_friction_range: dict | None = {
                "low": nominal - change,
                "high": nominal + change,
            }
        else:
            self.obj_friction_range = None

        # ── Reward / obs config ────────────────────────────────────────────
        self.rwd_keys_wt = weighted_reward_keys
        self.obs_keys = list(obs_keys)
        if self.model.na > 0 and "act" not in self.obs_keys:
            self.obs_keys.append("act")

        # ── Action space ───────────────────────────────────────────────────
        self.normalize_act = normalize_act
        if normalize_act:
            act_low = -np.ones(self.model.nu, dtype=np.float32)
            act_high = np.ones(self.model.nu, dtype=np.float32)
        else:
            act_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
            act_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)

        # ── Initial state ──────────────────────────────────────────────────
        # Old code: init_qpos[:-14] *= 0 (open hand), init_qpos[0] = -1.57 (palm up)
        init_qpos = self.data.qpos.ravel().copy()
        init_qpos[:-14] = 0.0
        init_qpos[0] = -1.57
        self._init_qpos = init_qpos
        self._init_qvel = self.data.qvel.ravel().copy()

        # Store seed for backward-compat get_input_seed() / seed() API.
        self._input_seed = seed

        import gymnasium

        gymnasium.Env.reset(self, seed=seed)

        # ── Initial task setup (matches old _setup() np_random calls) ──────
        # This mirrors old _setup() which consumed np_random before super()._setup()
        self.which_task = (
            self.np_random.choice(list(Task)) if task_choice == "random" else WHICH_TASK
        )
        self.x_radius = self.np_random.uniform(low=goal_xrange[0], high=goal_xrange[1])
        self.y_radius = self.np_random.uniform(low=goal_yrange[0], high=goal_yrange[1])
        self.counter = 0
        self.goal = self.create_goal_trajectory(time_step=self._ctrl_dt, time_period=6)

        # ── Observation space ──────────────────────────────────────────────
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs = self._obs_dict_to_vec(self.get_obs_dict(self._accessor))
        self.observation_space = gym.spaces.Box(
            -10.0 * np.ones(obs.size, dtype=np.float32),
            10.0 * np.ones(obs.size, dtype=np.float32),
            dtype=np.float32,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _init_muscle_condition(self) -> None:
        """Apply the muscle condition to the compiled model."""
        if self.muscle_condition == "sarcopenia":
            apply_sarcopenia_to_model(self.model, force_scale=0.5)
        elif self.muscle_condition == "fatigue":
            self.muscle_fatigue = CumulativeFatigue(
                self.model, self.frame_skip, seed=None
            )

    def _apply_action(self, action: np.ndarray) -> None:
        """Map action to MuJoCo ctrl and write to data.ctrl.

        Args:
            action: Action vector in the action-space (already clipped).
        """
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

    # ── MyoGymnasiumEnv interface ──────────────────────────────────────────

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        """Compute the observation dictionary.

        Args:
            accessor: CPU environment accessor.

        Returns:
            Filtered dict with obs_keys as keys.
        """
        obs: dict[str, np.ndarray] = {
            "time": np.array([accessor.time()]),
            "hand_pos": accessor.data.qpos[:-14].copy(),
            "object1_pos": accessor.data.site_xpos[self.object1_sid].copy(),
            "object2_pos": accessor.data.site_xpos[self.object2_sid].copy(),
            "object1_velp": accessor.data.qvel[-12:-9].copy() * accessor.dt(),
            "object2_velp": accessor.data.qvel[-6:-3].copy() * accessor.dt(),
            "target1_pos": accessor.data.site_xpos[self.target1_sid].copy(),
            "target2_pos": accessor.data.site_xpos[self.target2_sid].copy(),
            "act": accessor.muscle_act(),
        }
        obs["target1_err"] = obs["target1_pos"] - obs["object1_pos"]
        obs["target2_err"] = obs["target2_pos"] - obs["object2_pos"]
        return {k: obs[k] for k in self.obs_keys if k in obs}

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        """Compute the reward dictionary.

        Args:
            obs_dict: Output of :meth:`get_obs_dict`.

        Returns:
            Ordered dict with reward components.
        """
        target1_err = obs_dict.get("target1_err", np.zeros(3))
        target2_err = obs_dict.get("target2_err", np.zeros(3))
        target1_dist = float(np.linalg.norm(target1_err))
        target2_dist = float(np.linalg.norm(target2_err))
        act_mag = (
            float(np.linalg.norm(obs_dict.get("act", np.zeros(1)))) / self.model.na
            if self.model.na != 0
            else 0.0
        )

        object1_z = float(obs_dict.get("object1_pos", np.zeros(3))[2])
        object2_z = float(obs_dict.get("object2_pos", np.zeros(3))[2])
        is_fall = (object1_z < self.drop_th) or (object2_z < self.drop_th)

        rwd_dict = collections.OrderedDict(
            (
                ("pos_dist_1", -1.0 * target1_dist),
                ("pos_dist_2", -1.0 * target2_dist),
                ("act_reg", -1.0 * act_mag),
                ("sparse", -(target1_dist + target2_dist)),
                (
                    "solved",
                    (target1_dist < self.proximity_th)
                    and (target2_dist < self.proximity_th)
                    and (not is_fall),
                ),
                ("done", is_fall),
            )
        )
        rwd_dict["dense"] = float(
            np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()])
        )

        # Visual success indicator
        self.model.geom_rgba[self.object1_gid, :2] = (
            np.array([1, 1])
            if target1_dist < self.proximity_th
            else np.array([0.5, 0.5])
        )
        self.model.geom_rgba[self.object2_gid, :2] = (
            np.array([0.9, 0.7])
            if target2_dist < self.proximity_th
            else np.array([0.5, 0.5])
        )

        return rwd_dict

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        """Resample task parameters; np_random ordering matches old reset().

        Ordering:
          1. (if random) task choice + ball_1 starting angle
          2. x_radius
          3. y_radius
          4. time_period → create_goal_trajectory
          5. obj mass (2 calls), friction (2 calls), size (2 calls)

        Args:
            np_random: NumPy random generator from gymnasium.

        Returns:
            Empty task state dict.
        """
        # ── Task and starting angle ────────────────────────────────────────
        if self.task_choice == "random":
            self.which_task = np_random.choice(list(Task))
            self.ball_1_starting_angle = np_random.uniform(low=0, high=2 * np.pi)
            self.ball_2_starting_angle = self.ball_1_starting_angle - np.pi

        self.counter = 0

        # ── Goal trajectory ────────────────────────────────────────────────
        self.x_radius = np_random.uniform(
            low=self.goal_xrange[0], high=self.goal_xrange[1]
        )
        self.y_radius = np_random.uniform(
            low=self.goal_yrange[0], high=self.goal_yrange[1]
        )
        time_period = np_random.uniform(
            low=self.goal_time_period[0], high=self.goal_time_period[1]
        )
        self.goal = self.create_goal_trajectory(
            time_step=self._ctrl_dt, time_period=time_period
        )

        # ── Physical property randomisation ────────────────────────────────
        if self.obj_mass_range is not None:
            self.model.body_mass[self.object1_bid] = np_random.uniform(
                **self.obj_mass_range
            )
            self.model.body_mass[self.object2_bid] = np_random.uniform(
                **self.obj_mass_range
            )
        if self.obj_friction_range is not None:
            self.model.geom_friction[self.object1_gid] = np_random.uniform(
                **self.obj_friction_range
            )
            self.model.geom_friction[self.object2_gid] = np_random.uniform(
                **self.obj_friction_range
            )
        if self.obj_size_range is not None:
            self.model.geom_size[self.object1_gid] = np_random.uniform(
                **self.obj_size_range
            )
            self.model.geom_size[self.object2_gid] = np_random.uniform(
                **self.obj_size_range
            )

        return {}

    # ── Gymnasium step / reset ─────────────────────────────────────────────

    def step(self, action: np.ndarray, **kwargs: Any):
        """Advance the simulation by one control step.

        Updates target site positions from the precomputed goal trajectory
        before stepping physics, then increments the counter.

        Args:
            action: Control command in the action space.

        Returns:
            Tuple ``(obs, reward, terminated, truncated, info)``.
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Update target positions for this step (before physics)
        if self.which_task in (Task.HOLD, Task.BAODING_CW, Task.BAODING_CCW):
            desired_angle = self.goal[self.counter].copy()
            desired_angle[0] += self.ball_1_starting_angle
            desired_angle[1] += self.ball_2_starting_angle
            self.model.site_pos[self.target1_sid, 0] = (
                self.x_radius * np.cos(desired_angle[0]) + self.center_pos[0]
            )
            self.model.site_pos[self.target1_sid, 1] = (
                self.y_radius * np.sin(desired_angle[0]) + self.center_pos[1]
            )
            self.model.site_pos[self.target2_sid, 0] = (
                self.x_radius * np.cos(desired_angle[1]) + self.center_pos[0]
            )
            self.model.site_pos[self.target2_sid, 1] = (
                self.y_radius * np.sin(desired_angle[1]) + self.center_pos[1]
            )

        self._apply_action(action)
        mujoco.mj_step(self.model, self.data, self.frame_skip)
        mujoco.mj_kinematics(self.model, self.data)
        self.counter += 1

        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs_dict = self.get_obs_dict(self._accessor)
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
        **kwargs: Any,
    ):
        """Reset the environment to a new episode.

        Args:
            seed: Random seed for this episode.
            options: Unused.

        Returns:
            Tuple ``(obs, info)``.
        """
        import gymnasium

        gymnasium.Env.reset(self, seed=seed)

        if self.muscle_condition == "fatigue":
            self.muscle_fatigue.reset(
                fatigue_reset_vec=self.fatigue_reset_vec,
                fatigue_reset_random=self.fatigue_reset_random,
            )

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel

        self._task_state = self.reset_task(self.np_random)

        mujoco.mj_forward(self.model, self.data)
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs = self._obs_dict_to_vec(self.get_obs_dict(self._accessor))
        obs = self._ensure_obs_gymnasium_compliant(obs)
        return obs, {}

    # ── Compatibility helpers ──────────────────────────────────────────────

    @property
    def dt(self) -> float:
        """Control timestep in seconds."""
        return self._ctrl_dt

    def get_obs(self) -> np.ndarray:
        """Return the current observation vector (legacy compat)."""
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        return self._obs_dict_to_vec(self.get_obs_dict(self._accessor))

    def create_goal_trajectory(
        self, time_step: float = 0.1, time_period: float = 6
    ) -> np.ndarray:
        """Precompute the goal angle trajectory for one or more rotations.

        Args:
            time_step: Duration of each control step (seconds).
            time_period: Duration of one full orbit revolution (seconds).

        Returns:
            Array of shape ``(1000, 2)`` with angles for ball1 and ball2.
        """
        len_of_goals = 1000  # assumes greater than env horizon

        if self.which_task == Task.HOLD:
            sign = 0
        elif self.which_task == Task.BAODING_CW:
            sign = -1
        else:  # BAODING_CCW
            sign = 1

        goal_traj = []
        for t in range(len_of_goals):
            angle = sign * 2 * np.pi * (t * time_step / time_period)
            goal_traj.append(np.array([angle, angle]))

        return np.array(goal_traj)
