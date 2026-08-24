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
from myosuite.terms.mimic_obs import (
    resolve_mimic_site_ids,
    sample_mimic_target_sites,
)
from myosuite.terms.mimic_reward import (
    MimicTrackingConfig,
    compute_mimic_reward,
    compute_mimic_tracking_error,
    mimic_joint_pos_reward,
    mimic_joint_vel_reward,
    mimic_root_vel_reward,
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
        EzPickle.__init__(self, frame_skip, seed, **kwargs)
        self._site_ids: np.ndarray | None = None
        self._target_site_pos: np.ndarray | None = None
        self._target_lo = np.asarray(low, dtype=np.float64)
        self._target_hi = np.asarray(high, dtype=np.float64)
        # Random-target CPU mimic starts with ~0.8 m site error. reward_scale=20
        # saturates exp(-20*err)≈0 and hides the learning signal from PPO; use
        # the denser scale already employed for saber clip tracking on mjlab.
        self._tracking_cfg = MimicTrackingConfig(reward_scale=2.0)

    def _setup_spaces(self) -> None:
        ctrl = self.model.actuator_ctrlrange.astype(np.float32)
        self.action_space = spaces.Box(
            low=ctrl[:, 0],
            high=ctrl[:, 1],
            dtype=np.float32,
        )
        obs_size = int(self.model.nq + self.model.nv + self.model.na + 1)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

    def _resolve_mimic_sites(self, names: tuple[str, ...]) -> np.ndarray:
        return resolve_mimic_site_ids(self.model, names)

    def _get_obs_dict(self, accessor: CpuEnvAccessor) -> dict[str, np.ndarray]:
        qpos = accessor.joint_pos().astype(np.float32)
        qvel = accessor.joint_vel().astype(np.float32)
        act = accessor.muscle_act().astype(np.float32)
        assert self._site_ids is not None
        assert self._target_site_pos is not None
        track_err = compute_mimic_tracking_error(
            self.data.site_xpos[self._site_ids],
            self._target_site_pos,
        )
        return {
            "qpos": qpos,
            "qvel": qvel,
            "act": act,
            "track_err": np.asarray([track_err], dtype=np.float32),
        }

    def get_reward_dict(self, obs_dict: dict[str, np.ndarray]) -> dict[str, Any]:
        track_err = float(obs_dict["track_err"][0])
        solved = bool(track_err < self._tracking_cfg.success_threshold)
        dense = compute_mimic_reward(track_err, self._tracking_cfg)
        return {
            "track_err": track_err,
            "dense": dense,
            "sparse": -track_err,
            "solved": solved,
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


class MuscleMimicBimanualEnv(_MuscleMimicCpuBase):
    """CPU Gymnasium MuscleMimic bimanual task."""

    def __init__(self, seed: int | None = None, frame_skip: int = 5, **kwargs: Any):
        cfg = default_mimic_config()
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
        self.model, self._mj_spec, self._xml_path = compile_mimic_bimanual_mjmodel(cfg)
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * self.frame_skip)
        self._site_ids = self._resolve_mimic_sites(tuple(BODY2SITES_FOR_MIMIC.values()))
        self._target_site_pos = np.zeros(
            (int(self._site_ids.shape[0]), 3), dtype=np.float32
        )
        self._setup_spaces()


class MuscleMimicFullbodyEnv(_MuscleMimicCpuBase):
    """CPU Gymnasium MuscleMimic full-body task."""

    def __init__(self, seed: int | None = None, frame_skip: int = 5, **kwargs: Any):
        cfg = default_mimic_fullbody_config()
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
        self.model, self._mj_spec, self._xml_path = compile_mimic_fullbody_mjmodel(cfg)
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * self.frame_skip)
        names = tuple(FULLBODY_BODY2SITES_FOR_MIMIC.values())
        self._site_ids = self._resolve_mimic_sites(names)
        self._target_site_pos = np.zeros(
            (int(self._site_ids.shape[0]), 3), dtype=np.float32
        )
        self._setup_spaces()


class MuscleMimicFullbodyDirectionalEnv(_MuscleMimicCpuBase):
    """Single-agent MuscleMimic full-body env with gait-style + heading reward.

    Drops keypoint/site tracking entirely. Instead:
    - Imitates local joint angles/velocities from a reference gait clip (style).
    - Rewards root velocity aligned with a commanded heading direction (steering).

    The commanded heading is sampled uniformly from the unit circle at each
    episode reset, then held fixed for the episode. The agent observes it as a
    2-D unit vector appended to the standard kinematic obs.

    Registered as ``myoFullBodyDirectional-v0``.
    """

    DEFAULT_GAIT_REPO = "amathislab/musclemimic-retargeted"
    DEFAULT_GAIT_FILE = "MyoFullBody/gmr/KIT/167/walking_medium06_poses.npz"

    def __init__(
        self,
        seed: int | None = None,
        frame_skip: int = 5,
        target_speed: float = 0.7,
        w_jpos: float = 0.15,
        w_jvel: float = 0.05,
        w_heading: float = 0.6,
        w_act_reg: float = 0.005,
        w_act_smooth: float = 0.0,
        w_survival: float = 0.2,
        upright_pelvis_z: float = 0.85,
        fall_pelvis_z: float = 0.6,
        gait_clip_repo: str = DEFAULT_GAIT_REPO,
        gait_clip_filename: str = DEFAULT_GAIT_FILE,
        angle_override: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            frame_skip=frame_skip,
            seed=seed,
            mimic_target_low=(0.0,),
            mimic_target_high=(1.0,),
            **kwargs,
        )
        cfg = default_mimic_fullbody_config()
        self.model, self._mj_spec, self._xml_path = compile_mimic_fullbody_mjmodel(cfg)
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * self.frame_skip)

        self._target_speed = target_speed
        self._w_jpos = w_jpos
        self._w_jvel = w_jvel
        self._w_heading = w_heading
        self._w_act_reg = w_act_reg
        self._w_act_smooth = w_act_smooth
        self._w_survival = w_survival
        self._prev_act: np.ndarray | None = None
        self._upright_pelvis_z = upright_pelvis_z
        self._fall_pelvis_z = fall_pelvis_z

        self._gait_qpos_local: np.ndarray | None = None
        self._gait_qvel_local: np.ndarray | None = None
        self._gait_T: int = 0
        self._gait_repo = gait_clip_repo
        self._gait_file = gait_clip_filename

        # Circular clip state for phase-matched initialization
        self._circ_clips: list[Any] | None = None
        self._circ_best_frames: dict[
            int, tuple[int, int]
        ] = {}  # angle_sector -> (clip_idx, frame)

        self._heading_dir = np.array([0.0, 1.0], dtype=np.float32)
        self._angle_override: float | None = angle_override
        self._pelvis_site_id: int = -1

        self._load_gait_clip()
        self._load_circular_clips()
        self._pelvis_site_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pelvis_mimic")
        )
        self._setup_directional_spaces()

    def _load_gait_clip(self) -> None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=self._gait_repo,
            filename=self._gait_file,
            repo_type="dataset",
        )
        npz = np.load(path, allow_pickle=True)
        qpos = np.asarray(npz["qpos"], dtype=np.float64)
        qvel = np.asarray(npz["qvel"], dtype=np.float64)
        self._gait_qpos_local = qpos[:, 7:]
        self._gait_qvel_local = qvel[:, 6:]
        # Full qpos/qvel for initialization (skip first 30 frames = standing/transition)
        self._gait_qpos_full = qpos[30:]
        self._gait_qvel_full = qvel[30:]
        self._gait_T = int(qpos.shape[0])

    def _load_circular_clips(self) -> None:
        """Load CW + CCW circular walking clips and precompute phase-matched frames."""
        from huggingface_hub import hf_hub_download
        from myosuite.core.trajectory_io import load_motion_clip
        from pathlib import Path as _Path

        _CIRC_FILES = [
            "MyoFullBody/gmr/KIT/4/WalkInClockwiseCircle01_poses.npz",
            # CCW08 covers E (err=0.2°, spd=1.01) vs CCW04 (err=0.5°, spd=0.60).
            # Switching to CCW08 also lets argmin pick CW01 for SW/S (spd>0.94)
            # instead of CCW04 (spd=0.60), fixing the two weakest eval sectors.
            "MyoFullBody/gmr/KIT/4/WalkInCounterClockwiseCircle08_poses.npz",
        ]
        clips = []
        for fname in _CIRC_FILES:
            p = hf_hub_download(
                repo_id=self._gait_repo, filename=fname, repo_type="dataset"
            )
            clips.append(
                load_motion_clip(
                    _Path(p), expected_nq=self.model.nq, expected_nv=self.model.nv
                )
            )
        self._circ_clips = clips

        def _ang_dist(a: float, b: float) -> float:
            return abs((a - b + 180) % 360 - 180)

        for angle_deg in range(0, 360, 5):
            best_fi, best_ci, best_err = 0, 0, 999.0
            for ci, clip in enumerate(clips):
                spd = np.linalg.norm(clip.qvel[:, :2], axis=1)
                angles = np.rad2deg(np.arctan2(clip.qvel[:, 1], clip.qvel[:, 0]))
                dists = np.array([_ang_dist(float(a), angle_deg) for a in angles])
                dists[spd < 0.35] = 999.0
                fi = int(np.argmin(dists))
                if dists[fi] < best_err:
                    best_err = float(dists[fi])
                    best_fi = fi
                    best_ci = ci
            self._circ_best_frames[angle_deg] = (best_ci, best_fi)

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
        assert self._gait_qpos_local is not None
        frame = int(round(self.data.time / self._ctrl_dt)) % self._gait_T

        jpos_r = float(
            mimic_joint_pos_reward(
                np,
                obs_dict["qpos_local"].astype(np.float64),
                self._gait_qpos_local[frame],
            )
        )
        jvel_r = float(
            mimic_joint_vel_reward(
                np,
                obs_dict["qvel_local"].astype(np.float64),
                self._gait_qvel_local[frame],
            )
        )

        cur_vel3 = np.array(
            [obs_dict["root_vel_xy"][0], obs_dict["root_vel_xy"][1], 0.0],
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
        # Penalise speed that exceeds target to prevent runaway acceleration.
        cur_speed = float(np.linalg.norm(cur_vel3))
        # Strong quadratic penalty from 1.05× target_speed — prevents runaway gait.
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

        # Gate heading reward on being upright — prevents learning to fall forward
        heading_r_gated = heading_r if upright else 0.0
        survival_r = 1.0 if upright else 0.0

        dense = (
            self._w_jpos * jpos_r
            + self._w_jvel * jvel_r
            + self._w_heading * heading_r_gated
            + self._w_act_reg * act_reg
            + self._w_act_smooth * act_smooth
            + self._w_survival * survival_r
            - 5.0 * overspeed_penalty
        )
        return {
            "gait_jpos": jpos_r,
            "gait_jvel": jvel_r,
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

        if self._circ_clips is not None:
            # Phase-matched init: pick the clip frame closest to commanded heading,
            # ±5-frame jitter. Uses circular clips which naturally cover all directions.
            angle_deg = int(round(np.rad2deg(angle))) % 360
            sector = (angle_deg // 5) * 5
            ci, fi = self._circ_best_frames.get(sector, (0, 0))
            jitter = int(np_random.integers(-5, 6))
            fi = (fi + jitter) % self._circ_clips[ci].qpos.shape[0]
            self.data.qpos[:] = self._circ_clips[ci].qpos[fi]
            self.data.qvel[:] = self._circ_clips[ci].qvel[fi]

            # Rotate body to face commanded heading.
            # Body forward = pelvis X axis. At rest (yaw=0), pelvis X ≈ -Y (South).
            # For angle=0 (East), pelvis X must point East → target_yaw = +π/2.
            cur_yaw = self._pelvis_yaw(self.data.qpos)
            target_yaw = angle + np.pi / 2.0
            delta = target_yaw - cur_yaw
            half = delta / 2.0
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
            # Rotate root velocity to match new body orientation
            vx, vy = float(self.data.qvel[0]), float(self.data.qvel[1])
            c, s = float(np.cos(delta)), float(np.sin(delta))
            self.data.qvel[0] = c * vx - s * vy
            self.data.qvel[1] = s * vx + c * vy
        else:
            # Fallback: start from gait clip frame-0 with zero root velocity
            if self._gait_qpos_local is not None:
                self.data.qpos[7:] = self._gait_qpos_local[0]
            if self._gait_qvel_local is not None:
                self.data.qvel[6:] = self._gait_qvel_local[0]
            # Pelvis X = forward = -Y at rest → target_yaw = angle + π/2 for East
            half = (angle + np.pi / 2.0) / 2.0
            orig = self.data.qpos[3:7].copy()
            rw, rx, ry, rz = float(np.cos(half)), 0.0, 0.0, float(np.sin(half))
            ow, ox, oy, oz = (
                float(orig[0]),
                float(orig[1]),
                float(orig[2]),
                float(orig[3]),
            )
            self.data.qpos[3] = rw * ow - rx * ox - ry * oy - rz * oz
            self.data.qpos[4] = rw * ox + rx * ow + ry * oz - rz * oy
            self.data.qpos[5] = rw * oy - rx * oz + ry * ow + rz * ox
            self.data.qpos[6] = rw * oz + rx * oy - ry * ox + rz * ow

        return {"heading_dir": self._heading_dir}
