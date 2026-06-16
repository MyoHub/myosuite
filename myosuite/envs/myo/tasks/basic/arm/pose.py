# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import collections
from typing import Any
import warnings

import mujoco
import numpy as np

import gymnasium as gym
from gymnasium.utils import EzPickle

from myosuite.core.model_builder import ModelBuilder, build_from_recipe
from myosuite.core.muscle_conditions import apply_sarcopenia_to_model
from myosuite.envs.gymnasium_env import CpuEnvAccessor, MyoGymnasiumEnv
from myosuite.physics.fatigue import CumulativeFatigue
from myosuite.terms.base_action import sigmoid_muscle_activation
from myosuite.terms.base_obs import pose_error_obs
from myosuite.terms.base_reward import pose_reward


class PoseEnvV0(MyoGymnasiumEnv, EzPickle):
    """Pose-tracking task for musculoskeletal (or motor) MuJoCo models.

    The environment presents a target joint configuration and rewards the
    agent for minimising the angular distance to that target.  Supports
    elbow, finger, and hand models via the ``model_path`` argument.

    Muscle conditions (sarcopenia, fatigue, reafferentation) are applied
    at construction time via the ``muscle_condition`` kwarg and automatically
    registered as variant environments via explicit ``_registry.register_env`` calls.

    Args:
        model_path: Absolute path to the MuJoCo XML model.
        obsd_model_path: Unused (kept for API compatibility).
        seed: Random seed (passed to :meth:`gymnasium.Env.reset`).
        viz_site_targets: Site names used for target visualisation (tip +
            ``<name>_target`` sites must exist in the model).
        target_jnt_range: Dict ``{joint_name: (min, max)}`` defining the
            random target range.  Mutually exclusive with *target_jnt_value*.
        target_jnt_value: Fixed target joint vector.
        reset_type: One of ``"none"``, ``"init"``, or ``"random"``.
        target_type: One of ``"fixed"`` or ``"generate"``.
        obs_keys: Observation keys to include in the observation vector.
        weighted_reward_keys: Dict ``{reward_key: weight}`` for dense reward.
        pose_thd: Threshold (rad) below which the task is considered solved.
        normalize_act: If ``True``, action space is ``[-1, 1]``; linear
            denormalisation (motors) or sigmoid (muscles) is applied internally.
        frame_skip: Number of MuJoCo substeps per :meth:`step` call.
        muscle_condition: One of ``""``, ``"sarcopenia"``, ``"fatigue"``,
            ``"reafferentation"``.
        weight_bodyname: Body name for random mass perturbation.
        weight_range: ``(min_mass, max_mass)`` for random body mass.
        fatigue_reset_vec: Initial fatigue state vector (passed to
            :class:`CumulativeFatigue`).
        fatigue_reset_random: If ``True``, randomise fatigue state on reset.
    """

    DEFAULT_OBS_KEYS = ["qpos", "qvel", "pose_err"]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pose": 1.0,
        "bonus": 4.0,
        "act_reg": 1.0,
        "penalty": 50,
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
        viz_site_targets: tuple | None = None,
        target_jnt_range: dict | None = None,
        target_jnt_value: list | None = None,
        reset_type: str = "init",
        target_type: str = "generate",
        obs_keys: list = DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict[str, float] = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        pose_thd: float = 0.35,
        normalize_act: bool = True,
        frame_skip: int = 10,
        muscle_condition: str = "",
        weight_bodyname: str | None = None,
        weight_range: tuple | None = None,
        fatigue_reset_vec=None,
        fatigue_reset_random: bool = False,
        **kwargs: Any,
    ) -> None:
        MyoGymnasiumEnv.__init__(
            self, frame_skip=frame_skip, render_mode=kwargs.get("render_mode")
        )
        # EzPickle must be called AFTER MyoGymnasiumEnv.__init__ to avoid
        # the cooperative super() chain resetting _ezpickle_args to () when
        # gymnasium.Env.__init__ → Generic.__init__ → EzPickle.__init__() is
        # called with no arguments.  Calling it last ensures the correct args
        # take precedence.
        EzPickle.__init__(
            self,
            model_path,
            obsd_model_path,
            seed,
            viz_site_targets=viz_site_targets,
            target_jnt_range=target_jnt_range,
            target_jnt_value=target_jnt_value,
            reset_type=reset_type,
            target_type=target_type,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            pose_thd=pose_thd,
            normalize_act=normalize_act,
            frame_skip=frame_skip,
            muscle_condition=muscle_condition,
            weight_bodyname=weight_bodyname,
            weight_range=weight_range,
            fatigue_reset_vec=fatigue_reset_vec,
            fatigue_reset_random=fatigue_reset_random,
            **kwargs,
        )

        # ── Load model ────────────────────────────────────────────────────
        model_recipe = kwargs.pop("model_recipe", None)
        if model_recipe is not None:
            self.model, self._mj_spec = build_from_recipe(model_recipe)
            self._name_sfx = "_r"
        else:
            self.model, self._mj_spec = ModelBuilder.from_xml_file(model_path).build()
            self._name_sfx = ""
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * frame_skip)

        # ── Muscle condition ──────────────────────────────────────────────
        self.muscle_condition = muscle_condition
        self.fatigue_reset_vec = fatigue_reset_vec
        self.fatigue_reset_random = fatigue_reset_random
        self._muscle_act_ind = self.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        self._init_muscle_condition()

        # ── Target configuration ──────────────────────────────────────────
        self.reset_type = reset_type
        self.target_type = target_type
        self.pose_thd = pose_thd

        if target_jnt_range is not None:
            jnt_ids, jnt_ranges = [], []
            for jnt_name, jnt_range in target_jnt_range.items():
                jnt_ids.append(self.model.joint(jnt_name).id)
                jnt_ranges.append(jnt_range)
            self._target_jnt_ids: list[int] | None = jnt_ids
            self._target_jnt_range: np.ndarray | None = np.array(jnt_ranges)
            self.target_jnt_value = np.mean(self._target_jnt_range, axis=1)
        else:
            self._target_jnt_ids = None
            self._target_jnt_range = None
            self.target_jnt_value = (
                np.array(target_jnt_value) if target_jnt_value is not None else None
            )

        # ── Visualisation sites ───────────────────────────────────────────
        self.tip_sids: list[int] = []
        self.target_sids: list[int] = []
        if viz_site_targets:
            for site in viz_site_targets:
                self.tip_sids.append(self.model.site(site).id)
                self.target_sids.append(self.model.site(site + "_target").id)

        # ── Reward / obs config ───────────────────────────────────────────
        self.rwd_keys_wt = weighted_reward_keys
        self.obs_keys = list(obs_keys)
        if self.model.na > 0 and "act" not in self.obs_keys:
            self.obs_keys.append("act")

        # ── Action space ──────────────────────────────────────────────────
        self.normalize_act = normalize_act
        if normalize_act:
            act_low = -np.ones(self.model.nu, dtype=np.float32)
            act_high = np.ones(self.model.nu, dtype=np.float32)
        else:
            act_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
            act_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)

        # ── Weight perturbation ───────────────────────────────────────────
        self.weight_bodyname = weight_bodyname
        self.weight_range = weight_range

        # ── Initial state ─────────────────────────────────────────────────
        # Replicate MujocoEnv._setup() init_qpos logic: for normalize_act=True,
        # set actuated linear joints to mean(jnt_range) as starting position.
        mujoco.mj_forward(self.model, self.data)
        init_qpos = self.data.qpos.ravel().copy()
        if normalize_act:
            actuated_jnt_ids = self.model.actuator_trnid[
                self.model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT, 0
            ]
            linear_jnt_mask = np.logical_or(
                self.model.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE,
                self.model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE,
            )
            linear_jnt_ids = np.where(linear_jnt_mask)[0]
            linear_act_jnt_ids = np.intersect1d(actuated_jnt_ids, linear_jnt_ids)
            qpos_ids = self.model.jnt_qposadr[linear_act_jnt_ids]
            init_qpos[qpos_ids] = np.mean(
                self.model.jnt_range[linear_act_jnt_ids], axis=1
            )
        self._init_qpos = init_qpos
        self._init_qvel = self.data.qvel.ravel().copy()

        # Store seed for backward-compat get_input_seed() / seed() API.
        self._input_seed = seed

        # Initialise np_random without calling our overridden reset()
        # by calling the gymnasium base directly.
        import gymnasium

        gymnasium.Env.reset(self, seed=seed)

        # Pre-populate task state so get_reward_dict is safe before first reset().
        self._task_state: dict[str, Any] = {"target_angles": self.target_jnt_value}

        # ── Observation space (inferred from initial obs) ─────────────────
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
        elif self.muscle_condition == "reafferentation":
            sfx = self._name_sfx
            self.EPLpos = self.model.actuator(f"EPL{sfx}").id
            self.EIPpos = self.model.actuator(f"EIP{sfx}").id

    def _apply_action(self, action: np.ndarray) -> None:
        """Map action from action-space to MuJoCo ctrl and write to data.ctrl.

        Args:
            action: Action vector in the action-space (already clipped).

        Note:
            For muscle envs, ctrl is intentionally kept as float32 during the
            sigmoid so that the result is truncated back to float32 precision,
            matching the original BaseV0.step() behaviour and ensuring parity.
        """
        ctrl = action.copy()  # preserve float32 dtype from action_space

        if self.model.na > 0 and self.normalize_act:
            # Sigmoid mapping for muscle actuators: [-1, 1] → [0, 1].
            # RHS is float64 but assigned into a float32 slice, so the value
            # is truncated to float32 — identical to the original BaseV0.step().
            ctrl[self._muscle_act_ind] = sigmoid_muscle_activation(
                ctrl[self._muscle_act_ind], np
            )
        elif self.normalize_act:
            # Linear denormalisation for motor actuators: [-1, 1] → ctrl_range.
            # ctrlrange is float64, so the result naturally promotes to float64.
            ctrl_range = self.model.actuator_ctrlrange
            ctrl = (
                np.mean(ctrl_range, axis=-1)
                + ctrl * (ctrl_range[:, 1] - ctrl_range[:, 0]) / 2.0
            )

        if self.muscle_condition == "fatigue":
            ctrl[self._muscle_act_ind], _, _ = self.muscle_fatigue.compute_act(
                ctrl[self._muscle_act_ind]
            )
        elif self.muscle_condition == "reafferentation":
            ctrl[self.EPLpos] = ctrl[self.EIPpos].copy()
            ctrl[self.EIPpos] = 0.0

        self.data.ctrl[:] = ctrl

    def _get_target_pose(self) -> np.ndarray:
        """Sample a target pose from the joint range.

        Returns:
            Target joint angles, shape ``(n_target_joints,)``.
        """
        if self.target_type == "fixed":
            return self.target_jnt_value.copy()
        if self.target_type == "generate":
            return self.np_random.uniform(
                low=self._target_jnt_range[:, 0],
                high=self._target_jnt_range[:, 1],
            )
        raise TypeError(f"Unknown target_type: {self.target_type!r}")

    def _update_target(self, restore_sim: bool = False) -> None:
        """Resample the target pose and update visualisation sites.

        Args:
            restore_sim: If ``True``, save and restore qpos/qvel so the
                forward kinematics call does not disturb the current state.
        """
        if restore_sim:
            qpos_saved = self.data.qpos[:].copy()
            qvel_saved = self.data.qvel[:].copy()

        self.target_jnt_value = self._get_target_pose()

        # Drive the model to the target to read site positions
        self.data.qpos[:] = self.target_jnt_value.copy()
        mujoco.mj_forward(self.model, self.data)
        for tip_sid, tgt_sid in zip(self.tip_sids, self.target_sids):
            self.model.site_pos[tgt_sid] = self.data.site_xpos[tip_sid].copy()

        if restore_sim:
            self.data.qpos[:] = qpos_saved
            self.data.qvel[:] = qvel_saved
            mujoco.mj_forward(self.model, self.data)

    def update_target(self, restore_sim: bool = True) -> None:
        """Backward-compatible public alias for ``_update_target()``.

        .. deprecated::
            Call ``env.unwrapped._update_target(restore_sim=True)`` directly,
            or set ``env.unwrapped.target_jnt_value`` and call ``env.reset()``
            to start a new episode with the updated target.

        Args:
            restore_sim: If ``True`` (default), saves and restores qpos/qvel
                so the forward-kinematics call does not disturb the simulation.
        """
        warnings.warn(
            "update_target() is deprecated; call _update_target(restore_sim=True) "
            "directly, or update target_jnt_value and call env.reset().",
            DeprecationWarning,
            stacklevel=2,
        )
        self._update_target(restore_sim=restore_sim)

    @property
    def init_qpos(self) -> np.ndarray:
        """Backward-compatible public access to the initial joint positions.

        .. deprecated::
            Use ``env.unwrapped._init_qpos`` directly, or pass
            ``reset_qpos=`` to ``env.reset()`` to control episode start state.

        Returns the internal ``_init_qpos`` array by reference so that
        in-place mutations (``env.env.init_qpos[:] = ...``) take effect.
        """
        warnings.warn(
            "init_qpos is deprecated; use env._init_qpos directly, or pass "
            "reset_qpos= to env.reset() to set the initial joint positions.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._init_qpos

    @property
    def last_ctrl(self) -> np.ndarray:
        """Backward-compatible access to the last applied control signal.

        .. deprecated::
            Use ``env.unwrapped.data.ctrl`` to read the current control
            signal directly from the MuJoCo data struct.

        Returns a copy of the control vector as it was after action
        normalisation and before the physics step.

        Raises:
            AttributeError: If accessed before the first ``step()`` call.
        """
        warnings.warn(
            "last_ctrl is deprecated; use env.data.ctrl to read the current "
            "control signal directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not hasattr(self, "_last_ctrl"):
            raise AttributeError("last_ctrl is not available before the first step().")
        return self._last_ctrl

    # ── MyoGymnasiumEnv interface ──────────────────────────────────────────

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        """Compute the observation dictionary from the current physics state.

        Args:
            accessor: CPU environment accessor.

        Returns:
            Dict with keys ``"time"``, ``"qpos"``, ``"qvel"``, ``"act"``
            (when muscles present), and ``"pose_err"``.
        """
        obs: dict[str, np.ndarray] = {
            "time": np.array([accessor.time()]),
            "qpos": accessor.joint_pos(),
            "qvel": accessor.joint_vel() * accessor.dt(),
            "act": accessor.muscle_act(),
            "pose_err": pose_error_obs(accessor, self.target_jnt_value),
        }
        return {k: obs[k] for k in self.obs_keys if k in obs}

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        """Compute the reward dictionary from the observation dict.

        Uses :func:`~myosuite.terms.base_reward.pose_reward` for pose
        components.  Act-regularisation is kept inline because its formula
        differs from the generic ``act_reg`` term.

        Args:
            obs_dict: Output of :meth:`get_obs_dict`.

        Returns:
            Ordered dict with reward components including ``"dense"`` and
            ``"done"``.
        """
        # Safety net: ensure task_state has target_angles if called before reset().
        self._task_state.setdefault("target_angles", self.target_jnt_value)

        act_mag = np.linalg.norm(obs_dict.get("act", np.zeros(1)), axis=-1)
        if self.model.na != 0:
            act_mag = act_mag / self.model.na

        pose_components = pose_reward(
            self._accessor, self._task_state, pose_thd=self.pose_thd
        )

        rwd_dict = collections.OrderedDict(
            (
                ("pose", pose_components["pose"]),
                ("bonus", pose_components["bonus"]),
                ("penalty", pose_components["penalty"]),
                ("act_reg", -1.0 * act_mag),
                ("sparse", pose_components["pose"]),
                ("solved", pose_components["solved"]),
                ("done", pose_components["done"]),
            )
        )
        rwd_dict["dense"] = float(
            np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()])
        )
        return rwd_dict

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        """Sample a new target pose for the episode.

        Args:
            np_random: NumPy random generator from gymnasium.

        Returns:
            Task state dict containing ``"target_angles"`` for use by term
            functions such as :func:`~myosuite.terms.base_reward.pose_reward`.
        """
        if self.target_type == "generate":
            self._update_target(restore_sim=True)
        elif self.target_type == "fixed":
            self._update_target(restore_sim=True)
        return {"target_angles": self.target_jnt_value.copy()}

    # ── Gymnasium step / reset overrides ──────────────────────────────────

    def step(self, action: np.ndarray, **kwargs: Any):
        """Advance the simulation by one control step.

        Handles muscle-action normalisation before forwarding to MuJoCo.

        Args:
            action: Control command in the action space.
            **kwargs: Ignored compatibility kwargs (e.g. update_exteroception).

        Returns:
            Tuple ``(obs, reward, terminated, truncated, info)``.
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._apply_action(action)
        # Store for backward-compat last_ctrl property (do not warn per step)
        self._last_ctrl = self.data.ctrl.copy()
        mujoco.mj_step(self.model, self.data, self.frame_skip)
        mujoco.mj_kinematics(self.model, self.data)

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
        reset_qpos: np.ndarray | None = None,
        reset_qvel: np.ndarray | None = None,
        **kwargs: Any,
    ):
        """Reset the environment to a new episode.

        Args:
            seed: Random seed for this episode.
            options: Unused.
            reset_qpos: Override initial joint positions.
            reset_qvel: Override initial joint velocities.

        Returns:
            Tuple ``(obs, info)``.
        """
        import gymnasium

        gymnasium.Env.reset(self, seed=seed)  # sets self.np_random

        # Apply weight perturbation
        if self.weight_bodyname is not None:
            bid = self.model.body(self.weight_bodyname).id
            gid = self.model.body_geomadr[bid]
            weight = self.np_random.uniform(
                low=self.weight_range[0], high=self.weight_range[1]
            )
            self.model.body_mass[bid] = weight
            self.model.geom_size[gid][0] = 0.01 + 2.5 * weight / 100

        # Reset fatigue model
        if self.muscle_condition == "fatigue":
            self.muscle_fatigue.reset(
                fatigue_reset_vec=self.fatigue_reset_vec,
                fatigue_reset_random=self.fatigue_reset_random,
            )

        # Reset physics state
        mujoco.mj_resetData(self.model, self.data)
        if reset_qpos is not None:
            self.data.qpos[:] = reset_qpos
        elif hasattr(self, "_init_qpos"):
            self.data.qpos[:] = self._init_qpos
        if reset_qvel is not None:
            self.data.qvel[:] = reset_qvel
        elif hasattr(self, "_init_qvel"):
            self.data.qvel[:] = self._init_qvel

        # Resample task target FIRST — must match the old BaseV0.reset() np_random
        # consumption order: target is always sampled before the random init position.
        self._task_state = self.reset_task(self.np_random)

        # Then sample the random initial joint configuration (after target is drawn).
        if self.reset_type == "random" and hasattr(self, "model"):
            jnt_init = self.np_random.uniform(
                high=self.model.jnt_range[:, 1],
                low=self.model.jnt_range[:, 0],
            )
            self.data.qpos[:] = jnt_init

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

    def set_fatigue_reset_random(self, fatigue_reset_random: bool) -> None:
        """Update the fatigue randomisation flag.

        Args:
            fatigue_reset_random: If ``True``, randomise fatigue on reset.
        """
        if self.muscle_condition != "fatigue":
            import logging

            logging.warning("This has no effect — no fatigue model is active.")
        self.fatigue_reset_random = fatigue_reset_random
