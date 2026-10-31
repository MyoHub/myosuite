# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""CPU Gymnasium environments for MuscleMimic bimanual/full-body tasks."""

from __future__ import annotations

from typing import Any

# pylint: disable=no-member
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.utils import EzPickle

from myosuite.envs.gymnasium_env import CpuEnvAccessor, MyoGymnasiumEnv
from myosuite.integrations.musclemimic.bimanual_model import (
    BODY2SITES_FOR_MIMIC,
    compile_mimic_bimanual_mjmodel,
    default_mimic_config,
)
from myosuite.integrations.musclemimic.fullbody_model import (
    FULLBODY_BODY2SITES_FOR_MIMIC,
    compile_mimic_fullbody_mjmodel,
    default_mimic_fullbody_config,
)
from myosuite.terms.base_action import sigmoid_muscle_activation
from myosuite.terms.base_obs import joint_pos_obs, joint_vel_obs, muscle_act_obs
from myosuite.terms.mimic_obs import (
    resolve_mimic_site_ids,
    sample_mimic_target_sites,
)
from myosuite.terms.mimic_reward import (
    MimicTrackingConfig,
    compute_mimic_tracking_error,
    mimic_root_vel_reward,
    mimic_site_tracking_reward,
)


class _MuscleMimicCpuBase(MyoGymnasiumEnv, EzPickle):
    """Shared CPU implementation for MuscleMimic task variants."""

    def __init__(
        self,
        frame_skip: int,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        low = kwargs.get("mimic_target_low")
        high = kwargs.get("mimic_target_high")
        if low is None or high is None:
            raise ValueError(
                "mimic_target_low and mimic_target_high must be supplied "
                "(see MjxMuscleMimicBase.sample_task box sampling)."
            )
        MyoGymnasiumEnv.__init__(
            self, frame_skip=frame_skip, render_mode=kwargs.get("render_mode")
        )
        # Keyword args: subclass signatures order (seed, frame_skip, ...).
        EzPickle.__init__(self, frame_skip=frame_skip, seed=seed, **kwargs)
        self._site_ids: np.ndarray | None = None
        self._target_site_pos: np.ndarray | None = None
        self._target_lo = np.asarray(low, dtype=np.float64)
        self._target_hi = np.asarray(high, dtype=np.float64)

    def _resolve_mimic_sites(self, names: tuple[str, ...]) -> np.ndarray:
        return resolve_mimic_site_ids(self.model, names)


class _MuscleMimicTrackingEnv(_MuscleMimicCpuBase):
    """Random-target site tracking: the CPU half of the mjlab ``myoMimic*-v0`` tasks.

    Matches ``mimic_mjlab_env._make_mimic_env_cfg`` without a clip:

    - obs ``[qpos, qvel * ctrl_dt, act, site_pos, site_target, site_target - site_pos]``
      (mjlab terms ``qpos``, ``qvel``, ``act`` and ``mimic_site_*``);
    - reward ``exp(-tracking_reward_scale * mean site error)`` (variant config);
    - muscle ctrl ``sigmoid(5 (a - 0.5))`` (mjlab ``sigmoid`` action mode) of the
      action (unclipped, as on mjlab).

    Args:
        cfg: Variant config (``default_mimic_config`` / ``default_mimic_fullbody_config``).
        compile_fn: Model builder returning ``(model, spec, xml_path)`` for *cfg*.
        site_names: Tracked site names, in target order.
        seed: Optional RNG seed.
        frame_skip: Simulation sub-steps per control step.
        **kwargs: ``mimic_target_low`` / ``mimic_target_high`` box overrides.
    """

    def __init__(
        self,
        cfg: Any,
        compile_fn: Any,
        site_names: tuple[str, ...],
        seed: int | None,
        frame_skip: int,
        **kwargs: Any,
    ) -> None:
        mt_low = kwargs.pop(
            "mimic_target_low",
            tuple(float(x) for x in cfg.target_site_range.low),
        )
        mt_high = kwargs.pop(
            "mimic_target_high",
            tuple(float(x) for x in cfg.target_site_range.high),
        )
        super().__init__(
            frame_skip=frame_skip,
            seed=seed,
            mimic_target_low=mt_low,
            mimic_target_high=mt_high,
            **kwargs,
        )
        self.model, self._mj_spec, self._xml_path = compile_fn(cfg)
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * self.frame_skip)
        self._site_ids = self._resolve_mimic_sites(site_names)
        self._target_site_pos = np.zeros((len(self._site_ids), 3), dtype=np.float32)
        self._tracking_cfg = MimicTrackingConfig(
            reward_scale=float(cfg.tracking_reward_scale),
            success_threshold=float(cfg.tracking_success_threshold),
        )
        self._setup_spaces()

    def _setup_spaces(self) -> None:
        ctrl = self.model.actuator_ctrlrange.astype(np.float32)
        self.action_space = spaces.Box(
            low=ctrl[:, 0],
            high=ctrl[:, 1],
            dtype=np.float32,
        )
        n_site_values = 3 * 3 * len(self._site_ids)  # pos, target, error
        obs_size = int(self.model.nq + self.model.nv + self.model.na + n_site_values)
        self.observation_space = self._unbounded_obs_space(obs_size)

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        assert self._site_ids is not None
        assert self._target_site_pos is not None
        site_pos = accessor.site_xpos(self._site_ids)
        target = self._target_site_pos
        return {
            "qpos": joint_pos_obs(accessor),
            "qvel": joint_vel_obs(accessor),
            "act": muscle_act_obs(accessor),
            "mimic_site_pos": site_pos.ravel(),
            "mimic_site_target": target.ravel(),
            "mimic_site_err": (target - site_pos).ravel(),
        }

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        site_pos = obs_dict["mimic_site_pos"].reshape(-1, 3)
        target = obs_dict["mimic_site_target"].reshape(-1, 3)
        track_err = compute_mimic_tracking_error(site_pos, target)
        dense = mimic_site_tracking_reward(
            np, site_pos, target, scale=self._tracking_cfg.reward_scale
        )
        return {
            "track_err": track_err,
            "dense": float(dense),
            "sparse": -track_err,
            "solved": bool(track_err < self._tracking_cfg.success_threshold),
            "done": False,
        }

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        assert self._site_ids is not None
        self._target_site_pos = sample_mimic_target_sites(
            np_random,
            self._target_lo,
            self._target_hi,
            int(self._site_ids.shape[0]),
        )
        return {}

    def step(
        self, action: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one control step with sigmoid-mapped muscle controls.

        Args:
            action: Policy action; the sigmoid saturates it, there is no clip.
            **kwargs: Ignored compatibility kwargs.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        self.data.ctrl[:] = sigmoid_muscle_activation(action, np)
        self._step_physics()
        self._accessor = CpuEnvAccessor(self.model, self.data, self._ctrl_dt)
        obs_dict = self._get_obs_dict(self._accessor)
        return self._finalize_step(obs_dict, self.get_reward_dict(obs_dict))


class MuscleMimicBimanualEnv(_MuscleMimicTrackingEnv):
    """CPU Gymnasium MuscleMimic bimanual task."""

    def __init__(self, seed: int | None = None, frame_skip: int = 5, **kwargs: Any):
        super().__init__(
            default_mimic_config(),
            compile_mimic_bimanual_mjmodel,
            tuple(BODY2SITES_FOR_MIMIC.values()),
            seed,
            frame_skip,
            **kwargs,
        )


class MuscleMimicFullbodyEnv(_MuscleMimicTrackingEnv):
    """CPU Gymnasium MuscleMimic full-body task."""

    def __init__(self, seed: int | None = None, frame_skip: int = 5, **kwargs: Any):
        super().__init__(
            default_mimic_fullbody_config(),
            compile_mimic_fullbody_mjmodel,
            tuple(FULLBODY_BODY2SITES_FOR_MIMIC.values()),
            seed,
            frame_skip,
            **kwargs,
        )


class MuscleMimicFullbodyDirectionalEnv(_MuscleMimicCpuBase):
    """Full-body directional walk: heading command, no motion clip.

    Reward is heading-aligned root velocity (upright-gated), survival, and
    light action regularisation. There is no joint-style / DeepMimic term and
    no Hugging Face clip. Observation stays the 528-d locomotion contract
    (local q/v, act, body-frame root vel, heading, orientation).

    Registered as ``myoFullBodyDirectional-v0``.
    """

    def __init__(
        self,
        seed: int | None = None,
        frame_skip: int = 5,
        target_speed: float = 0.7,
        w_heading: float = 1.0,
        w_act_reg: float = 0.1,
        w_act_smooth: float = 0.0,
        w_survival: float = 0.2,
        upright_pelvis_z: float = 0.85,
        fall_pelvis_z: float = 0.6,
        angle_override: float | None = None,
        **kwargs: Any,
    ) -> None:
        # Older call sites passed gait-clip / style weights. Ignore them.
        kwargs.pop("w_jpos", None)
        kwargs.pop("w_jvel", None)
        kwargs.pop("gait_clip_repo", None)
        kwargs.pop("gait_clip_filename", None)
        super().__init__(
            frame_skip=frame_skip,
            seed=seed,
            mimic_target_low=(0.0,),
            mimic_target_high=(1.0,),
            **kwargs,
        )
        EzPickle.__init__(
            self,
            seed=seed,
            frame_skip=frame_skip,
            target_speed=target_speed,
            w_heading=w_heading,
            w_act_reg=w_act_reg,
            w_act_smooth=w_act_smooth,
            w_survival=w_survival,
            upright_pelvis_z=upright_pelvis_z,
            fall_pelvis_z=fall_pelvis_z,
            angle_override=angle_override,
            **kwargs,
        )
        cfg = default_mimic_fullbody_config()
        self.model, self._mj_spec, self._xml_path = compile_mimic_fullbody_mjmodel(cfg)
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * self.frame_skip)

        self._target_speed = target_speed
        self._w_heading = w_heading
        self._w_act_reg = w_act_reg
        self._w_act_smooth = w_act_smooth
        self._w_survival = w_survival
        self._prev_act: np.ndarray | None = None
        self._upright_pelvis_z = upright_pelvis_z
        self._fall_pelvis_z = fall_pelvis_z
        self._heading_dir = np.array([0.0, 1.0], dtype=np.float32)
        self._angle_override: float | None = angle_override
        self._pelvis_site_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pelvis_mimic")
        )
        self._setup_directional_spaces()

    def _setup_directional_spaces(self) -> None:
        ctrl = self.model.actuator_ctrlrange.astype(np.float32)
        self.action_space = spaces.Box(
            low=ctrl[:, 0], high=ctrl[:, 1], dtype=np.float32
        )
        nq_local = self.model.nq - 7
        nv_local = self.model.nv - 6
        na = self.model.na
        obs_size = (
            nq_local + nv_local + na + 2 + 2 + 6
        )  # local q/v, act, root_vel_xy, heading, orientation(roll,pitch,wx,wy,wz,vz)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    @staticmethod
    def _pelvis_yaw(qpos: np.ndarray) -> float:
        """Extract yaw from root quaternion qpos[3:7] = (w, x, y, z)."""
        w, x, y, z = float(qpos[3]), float(qpos[4]), float(qpos[5]), float(qpos[6])
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        qpos_local = self.data.qpos[7:].astype(np.float32)
        qvel_local = self.data.qvel[6:].astype(np.float32)
        act = (
            self.data.act.astype(np.float32)
            if self.model.na > 0
            else np.zeros(0, np.float32)
        )
        # Body-frame root velocity: rotate world-frame vel by inverse pelvis yaw.
        # This makes obs direction-invariant — "walk forward" always gives vel≈(spd,0).
        yaw = self._pelvis_yaw(self.data.qpos)
        vx, vy = float(self.data.qvel[0]), float(self.data.qvel[1])
        c, s = np.cos(-yaw), np.sin(-yaw)
        root_vel_body = np.array([c * vx - s * vy, s * vx + c * vy], dtype=np.float32)
        # Orientation features: roll/pitch from root quaternion + body-frame angular velocity.
        # Critical for tilt detection — without these the policy cannot sense it is falling.
        w, x, y, z = (
            float(self.data.qpos[3]),
            float(self.data.qpos[4]),
            float(self.data.qpos[5]),
            float(self.data.qpos[6]),
        )
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        wx_w, wy_w, wz_w = (
            float(self.data.qvel[3]),
            float(self.data.qvel[4]),
            float(self.data.qvel[5]),
        )
        wx_b = c * wx_w - s * wy_w
        wy_b = s * wx_w + c * wy_w
        vz = float(self.data.qvel[2])
        orientation = np.array([roll, pitch, wx_b, wy_b, wz_w, vz], dtype=np.float32)
        return {
            "qpos_local": qpos_local,
            "qvel_local": qvel_local,
            "act": act,
            "root_vel_xy": root_vel_body,
            "heading_cmd": self._heading_dir.astype(np.float32),
            "orientation": orientation,
        }

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        cur_vel3 = np.array(
            [float(self.data.qvel[0]), float(self.data.qvel[1]), 0.0],
            dtype=np.float64,
        )
        target_vel3 = np.array(
            [
                self._target_speed * self._heading_dir[0],
                self._target_speed * self._heading_dir[1],
                0.0,
            ],
            dtype=np.float64,
        )
        heading_r = float(mimic_root_vel_reward(np, cur_vel3, target_vel3, scale=1.0))
        cur_speed = float(np.linalg.norm(cur_vel3[:2]))
        excess = max(0.0, cur_speed - self._target_speed * 1.05)
        overspeed_penalty = excess * excess

        act = obs_dict["act"]
        act_reg = -float(np.square(act).mean()) if act.size > 0 else 0.0
        if self._w_act_smooth > 0.0 and self._prev_act is not None and act.size > 0:
            act_smooth = -float(np.square(act - self._prev_act).mean())
        else:
            act_smooth = 0.0
        self._prev_act = act.copy() if act.size > 0 else None

        pelvis_z = float(self.data.site_xpos[self._pelvis_site_id, 2])
        fell = pelvis_z < self._fall_pelvis_z
        upright = pelvis_z >= self._upright_pelvis_z
        heading_r_gated = heading_r if upright else 0.0
        survival_r = 1.0 if upright else 0.0

        dense = (
            self._w_heading * heading_r_gated
            + self._w_act_reg * act_reg
            + self._w_act_smooth * act_smooth
            + self._w_survival * survival_r
            - 5.0 * overspeed_penalty
        )
        return {
            "heading": heading_r_gated,
            "survival": survival_r,
            "dense": dense,
            "sparse": heading_r_gated,
            "solved": heading_r_gated > 0.8,
            "done": fell,
        }

    def reset_task(self, np_random: np.random.Generator) -> dict[str, Any]:
        self._prev_act = None
        if self._angle_override is not None:
            angle = float(self._angle_override)
        else:
            angle = float(np_random.uniform(0.0, 2.0 * np.pi))
        self._heading_dir = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qvel[:] = 0.0

        # Body forward = pelvis X. At rest (yaw=0) that is world -Y, so East
        # (angle=0) needs target_yaw = +π/2.
        cur_yaw = self._pelvis_yaw(self.data.qpos)
        target_yaw = angle + np.pi / 2.0
        half = (target_yaw - cur_yaw) / 2.0
        rw, rx, ry, rz = float(np.cos(half)), 0.0, 0.0, float(np.sin(half))
        ow, ox, oy, oz = (
            float(self.data.qpos[3]),
            float(self.data.qpos[4]),
            float(self.data.qpos[5]),
            float(self.data.qpos[6]),
        )
        self.data.qpos[3] = rw * ow - rx * ox - ry * oy - rz * oz
        self.data.qpos[4] = rw * ox + rx * ow + ry * oz - rz * oy
        self.data.qpos[5] = rw * oy - rx * oz + ry * ow + rz * ox
        self.data.qpos[6] = rw * oz + rx * oy - ry * ox + rz * ow

        if self.model.na > 0:
            self.data.act[:] = 0.05
        return {"heading_dir": self._heading_dir}
