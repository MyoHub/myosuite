# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import collections
from typing import Any

import mujoco
import numpy as np

import gymnasium as gym
from gymnasium.utils import EzPickle

from myosuite.core.model_builder import ModelBuilder, build_from_recipe
from myosuite.envs.gymnasium_env import CpuEnvAccessor, MyoGymnasiumEnv
from myosuite.envs.myo.tasks.basic.muscle_mixin import MuscleActionMixin


class KeyTurnEnvV0(MuscleActionMixin, MyoGymnasiumEnv, EzPickle):
    """Key-turning task for the musculoskeletal hand model.

    The agent must grasp a key with two fingers (index and thumb) and rotate
    it to a target angle.

    Args:
        model_path: Absolute path to the MuJoCo XML model.
        obsd_model_path: Unused (kept for API compatibility).
        seed: Random seed.
        goal_th: Key angle threshold (rad) for task success.
        obs_keys: Observation keys.
        weighted_reward_keys: Dict ``{reward_key: weight}`` for dense reward.
        key_init_range: ``(min, max)`` range for initial key joint angle.
        normalize_act: If ``True``, action space is ``[-1, 1]``.
        frame_skip: Number of MuJoCo substeps per :meth:`step` call.
    """

    DEFAULT_OBS_KEYS = [
        "hand_qpos",
        "hand_qvel",
        "key_qpos",
        "key_qvel",
        "IFtip_approach",
        "THtip_approach",
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "key_turn": 1.0,
        "IFtip_approach": 10.0,
        "THtip_approach": 10.0,
        "act_reg": 1.0,
        "bonus": 4.0,
        "penalty": 25.0,
    }

    MYO_CREDIT = """\
    MyoSuite: A contact-rich simulation suite for musculoskeletal motor control
        Vittorio Caggiano, Huawei Wang, Guillaume Durandau, Massimo Sartori, Vikash Kumar
        L4DC-2019 | https://sites.google.com/view/myosuite
    """

    def __init__(
        self,
        model_path: str = "",
        obsd_model_path: str | None = None,
        seed: int | None = None,
        goal_th: float = 3.14,
        obs_keys: list = DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict[str, float] = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        key_init_range: tuple = (0, 0),
        normalize_act: bool = True,
        frame_skip: int = 10,
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
            goal_th=goal_th,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            key_init_range=key_init_range,
            normalize_act=normalize_act,
            frame_skip=frame_skip,
            **kwargs,
        )

        # ── Load model ─────────────────────────────────────────────────────
        model_recipe = kwargs.pop("model_recipe", None)
        if model_recipe is not None:
            self.model, self._mj_spec = build_from_recipe(model_recipe)
        else:
            self.model, self._mj_spec = ModelBuilder.from_xml_file(model_path).build()
        self.data = mujoco.MjData(self.model)
        self._name_sfx = "" if model_recipe == "hand_keyturn" else "_r"
        self._ctrl_dt = float(self.model.opt.timestep * frame_skip)

        # ── Muscle actuators (action-pipeline stages come from wrappers)
        self._muscle_act_ind = self.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE

        # ── Task config ────────────────────────────────────────────────────
        self.goal_th = goal_th
        self.key_init_range = key_init_range

        # Site IDs
        sfx = self._name_sfx
        mujoco.mj_forward(self.model, self.data)
        self.keyhead_sid = self.model.site("keyhead").id
        self.IF_sid = self.model.site(f"IFtip{sfx}").id
        self.TH_sid = self.model.site(f"THtip{sfx}").id

        # Store keyhead initial world position for random perturbation
        self.key_init_pos = self.data.site_xpos[self.keyhead_sid].copy()

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
        # Old code sets init_qpos[:-1] *= 0 → all hand joints to 0 (open hand)
        init_qpos = self.data.qpos.ravel().copy()
        init_qpos[:-1] = 0.0
        self._init_qpos = init_qpos
        self._init_qvel = self.data.qvel.ravel().copy()

        import gymnasium

        self._input_seed = seed
        gymnasium.Env.reset(self, seed=seed)

        # ── Observation space ──────────────────────────────────────────────
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs = self._obs_dict_to_vec(self.get_obs_dict(self._accessor))
        self.observation_space = self._unbounded_obs_space(obs.size)

    # ── Private helpers ────────────────────────────────────────────────────

    # ── MyoGymnasiumEnv interface ──────────────────────────────────────────

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        """Compute the observation dictionary.

        Args:
            accessor: CPU environment accessor.

        Returns:
            Filtered dict with obs_keys as keys.
        """
        keyhead_xpos = accessor.site_xpos(self.keyhead_sid)
        obs: dict[str, np.ndarray] = {
            "time": np.array([accessor.time()]),
            "hand_qpos": accessor.data.qpos[:-1].copy(),
            "hand_qvel": accessor.data.qvel[:-1].copy() * accessor.dt(),
            "key_qpos": np.array([accessor.data.qpos[-1]]),
            "key_qvel": np.array([accessor.data.qvel[-1]]) * accessor.dt(),
            "IFtip_approach": keyhead_xpos - accessor.site_xpos(self.IF_sid),
            "THtip_approach": keyhead_xpos - accessor.site_xpos(self.TH_sid),
            "act": accessor.muscle_act(),
        }
        return self._select_obs_keys(obs)

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        """Compute the reward dictionary.

        Args:
            obs_dict: Output of :meth:`get_obs_dict`.

        Returns:
            Ordered dict with reward components.
        """
        IF_approach = obs_dict.get("IFtip_approach", np.zeros(3))
        TH_approach = obs_dict.get("THtip_approach", np.zeros(3))
        IF_approach_dist = float(abs(np.linalg.norm(IF_approach) - 0.030))
        TH_approach_dist = float(abs(np.linalg.norm(TH_approach) - 0.030))
        key_pos = float(obs_dict.get("key_qpos", np.zeros(1))[0])
        act_mag = (
            float(np.linalg.norm(obs_dict.get("act", np.zeros(1)))) / self.model.na
            if self.model.na != 0
            else 0.0
        )
        far_th = 0.1

        rwd_dict = collections.OrderedDict(
            (
                ("key_turn", key_pos),
                ("IFtip_approach", -1.0 * IF_approach_dist),
                ("THtip_approach", -1.0 * TH_approach_dist),
                ("act_reg", -1.0 * act_mag),
                ("bonus", 1.0 * (key_pos > np.pi / 2) + 1.0 * (key_pos > np.pi)),
                (
                    "penalty",
                    -1.0 * (IF_approach_dist > far_th / 2)
                    - 1.0 * (TH_approach_dist > far_th / 2),
                ),
                ("sparse", key_pos),
                ("solved", key_pos > self.goal_th),
                (
                    "done",
                    bool((IF_approach_dist > far_th) or (TH_approach_dist > far_th)),
                ),
            )
        )
        rwd_dict["dense"] = float(
            np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()])
        )
        return rwd_dict

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        """Randomise key initial angle and (optionally) position.

        Args:
            np_random: NumPy random generator from gymnasium.

        Returns:
            Empty task state dict.
        """
        # Randomise key joint angle (always)
        self.data.qpos[-1] = np_random.uniform(
            low=self.key_init_range[0], high=self.key_init_range[1]
        )
        # Randomise key body position if range is non-zero (RandomEnv)
        if self.key_init_range[0] != self.key_init_range[1]:
            self.model.body_pos[-1] = self.key_init_pos + np_random.uniform(
                low=np.array([-0.01, -0.01, -0.01]),
                high=np.array([0.01, 0.01, 0.01]),
            )
        return {}

    # ── Gymnasium step / reset ─────────────────────────────────────────────

    def step(self, action: np.ndarray, **kwargs: Any):
        """Advance the simulation by one control step.

        Args:
            action: Control command in the action space.

        Returns:
            Tuple ``(obs, reward, terminated, truncated, info)``.
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._apply_action(action)
        self._step_physics()

        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs_dict = self.get_obs_dict(self._accessor)
        rwd_dict = self.get_reward_dict(obs_dict)
        return self._finalize_step(obs_dict, rwd_dict)

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

        self._run_reset_stages()

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel

        # Randomise key (matches old reset order: key randomisation before super().reset())
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
