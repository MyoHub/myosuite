# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Experimental batched task environment and ``rsl_rl`` ``VecEnv`` wrapper on Newton.

.. warning::
   **Experimental validation probe only.** CPU ``MyoGymnasiumEnv`` and GPU
   ``mjlab`` are the only two supported MyoSuite backends. Like the experimental
   MJX route, this module exists solely for solver compatibility diagnostics and
   should not be used as a foundation for new MyoSuite tasks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import torch

from myosuite.envs.myo.tasks.basic.arm.key_turn import KeyTurnEnvV0
from myosuite.envs.myo.tasks.basic.arm.pose import PoseEnvV0
from myosuite.envs.myo.tasks.basic.arm.reach import ReachEnvV0
from myosuite.envs.myo.tasks.basic.hand.obj_hold import ObjHoldFixedEnvV0
from myosuite.envs.myo.tasks.basic.hand.pen import PenTwirlFixedEnvV0
from myosuite.envs.myo.tasks.basic.leg.reach import LegReachEnvV0
from myosuite.envs.myo.tasks.basic.leg.walk import LegWalkEnvV0
from myosuite.envs.myo.tasks.basic.torso.pose import TorsoEnvV0
from myosuite.integrations.newton.accessor import _entry_point_defaults
from myosuite.integrations.newton.batch_sim import NewtonBatchSim
from myosuite.integrations.newton.torch_pipeline import NewtonActionCfg
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings
from myosuite.physics.quat_math import euler2quat
from myosuite.terms.base_event import terrain_heights
from myosuite.terms.base_obs import (
    com_linear_velocity,
    joint_pos_obs,
    joint_vel_obs,
    muscle_act_obs,
    pose_error_obs,
    tip_pos_obs,
)
from myosuite.terms.base_reward import (
    leg_reach_reward,
    locomotion_solved,
    locomotion_streak,
    locomotion_sustained,
    multi_site_reach_reward,
    pose_reward,
    walk_env_reward,
)
from myosuite.utils.reach_workspace import reachable_target_points

try:
    from rsl_rl.env import VecEnv as _RslRlVecEnvBase
except ImportError:  # pragma: no cover - optional RL dependency
    _RslRlVecEnvBase = object  # type: ignore[assignment,misc]


def _compute_init_qpos_qvel(
    model: mujoco.MjModel, *, normalize_act: bool
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    init_qpos = data.qpos.ravel().copy()
    if normalize_act:
        actuated_jnt_ids = model.actuator_trnid[
            model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT, 0
        ]
        linear_mask = np.logical_or(
            model.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE,
            model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE,
        )
        linear_act_ids = np.intersect1d(actuated_jnt_ids, np.where(linear_mask)[0])
        qpos_ids = model.jnt_qposadr[linear_act_ids]
        init_qpos[qpos_ids] = np.mean(model.jnt_range[linear_act_ids], axis=1)
    return init_qpos.astype(np.float32), data.qvel.ravel().copy().astype(np.float32)


class NewtonTaskEnv:
    """Experimental vectorized ``PoseEnvV0`` / ``ReachEnvV0`` runner backed by ``NewtonBatchSim``."""

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        *,
        device: str = "cuda:0",
        seed: int | None = None,
        auto_reset: bool = True,
        nconmax: int = 2048,
        njmax: int = 8192,
    ) -> None:
        import myosuite  # noqa: F401

        self.env_id = str(env_id)
        spec = gym.spec(self.env_id)
        entry_str = str(spec.entry_point)
        is_torso = "TorsoEnvV0" in entry_str
        if "PoseEnvV0" in entry_str or is_torso:
            self.task_kind = "pose"
        elif "LegReachEnvV0" in entry_str:
            self.task_kind = "leg_reach"
        elif "ReachEnvV0" in entry_str:
            self.task_kind = "reach"
        elif "ObjHold" in entry_str:
            self.task_kind = "obj_hold"
        elif "PenTwirl" in entry_str:
            self.task_kind = "pen_twirl"
        elif "KeyTurnEnvV0" in entry_str:
            self.task_kind = "key_turn"
        elif "LegWalkEnvV0" in entry_str or "LegTerrainEnvV0" in entry_str:
            self.task_kind = "walk"
        else:
            raise ValueError(
                "NewtonTaskEnv supports PoseEnvV0, TorsoEnvV0, ReachEnvV0, "
                "LegReachEnvV0, LegWalkEnvV0, LegTerrainEnvV0, ObjHold*EnvV0, "
                f"PenTwirl*EnvV0, and KeyTurnEnvV0 tasks, got {entry_str!r}."
            )

        kwargs = dict(spec.kwargs)
        defaults = _entry_point_defaults(spec.entry_point)
        action_cfg = NewtonActionCfg.from_env_id(self.env_id)
        self.sim = NewtonBatchSim(
            self.env_id,
            num_envs=num_envs,
            device=device,
            action_cfg=action_cfg,
            nconmax=nconmax,
            njmax=njmax,
            seed=seed,
        )
        self.num_envs = self.sim.num_envs
        self.auto_reset = bool(auto_reset)
        self.max_episode_steps = self.sim.max_episode_steps
        self.torch_device = self.sim.torch_accessor.device
        ref_model = self.sim.reference_model

        self._init_qpos, self._init_qvel = _compute_init_qpos_qvel(
            ref_model, normalize_act=action_cfg.normalize_act
        )
        self._couplings = JointCouplings.from_model(ref_model)
        self._rngs: list[np.random.Generator] = [
            gym.utils.seeding.np_random(
                None if seed is None else (seed if w == 0 else seed + w)
            )[0]
            for w in range(self.num_envs)
        ]

        if self.task_kind == "pose":
            pose_cls = TorsoEnvV0 if is_torso else PoseEnvV0
            obs_keys = list(kwargs.get("obs_keys", pose_cls.DEFAULT_OBS_KEYS))
            self.rwd_keys_wt = dict(
                kwargs.get(
                    "weighted_reward_keys", pose_cls.DEFAULT_RWD_KEYS_AND_WEIGHTS
                )
            )
            self.reset_type = str(
                kwargs.get("reset_type", defaults.get("reset_type", "init"))
            )
            self.target_type = (
                "fixed"
                if is_torso
                else str(
                    kwargs.get("target_type", defaults.get("target_type", "generate"))
                )
            )
            self.pose_thd = float(
                kwargs.get(
                    "pose_thd", defaults.get("pose_thd", 0.25 if is_torso else 0.35)
                )
            )
            self.pose_far_thd = math.pi if is_torso else 2.0 * math.pi
            target_jnt_range = kwargs.get("target_jnt_range")
            if target_jnt_range is not None:
                ranges = [target_jnt_range[jnt_name] for jnt_name in target_jnt_range]
                self._target_jnt_range: np.ndarray | None = np.asarray(
                    ranges, dtype=np.float64
                )
                fixed = np.mean(self._target_jnt_range, axis=1)
            else:
                self._target_jnt_range = None
                fixed = np.asarray(kwargs["target_jnt_value"], dtype=np.float64)
            self._fixed_target = fixed.astype(np.float32)
            self.target_angles = torch.zeros(
                (self.num_envs, len(self._fixed_target)),
                dtype=torch.float32,
                device=self.torch_device,
            )
            self.tip_sids: list[int] = []
            self.target_sids: list[int] = []
        elif self.task_kind in ("reach", "leg_reach"):
            reach_cls = LegReachEnvV0 if self.task_kind == "leg_reach" else ReachEnvV0
            obs_keys = list(kwargs.get("obs_keys", reach_cls.DEFAULT_OBS_KEYS))
            self.rwd_keys_wt = dict(
                kwargs.get(
                    "weighted_reward_keys", reach_cls.DEFAULT_RWD_KEYS_AND_WEIGHTS
                )
            )
            self.target_reach_range = dict(kwargs["target_reach_range"])
            self.target_sampling = str(
                kwargs.get("target_sampling", defaults.get("target_sampling", "box"))
            )
            self.far_th = float(kwargs.get("far_th", defaults.get("far_th", 0.35)))
            self.joint_random_range = tuple(
                kwargs.get(
                    "joint_random_range", defaults.get("joint_random_range", (0.0, 0.0))
                )
            )
            if self.task_kind == "leg_reach":
                self._init_qpos = ref_model.key_qpos[0].astype(np.float32).copy()
                self._init_qvel = ref_model.key_qvel[0].astype(np.float32).copy()
            self.tip_sids = [ref_model.site(s).id for s in self.target_reach_range]
            self.target_sids = [
                ref_model.site(f"{s}_target").id for s in self.target_reach_range
            ]
            self._workspace_points: np.ndarray | None = None
            if self.task_kind == "reach" and self.target_sampling == "workspace":
                spans = list(self.target_reach_range.values())
                self._workspace_points = reachable_target_points(
                    ref_model,
                    self.tip_sids,
                    np.array([sp[0] for sp in spans]),
                    np.array([sp[1] for sp in spans]),
                )
            self.target_pos = torch.zeros(
                (self.num_envs, 3 * len(self.tip_sids)),
                dtype=torch.float32,
                device=self.torch_device,
            )
        elif self.task_kind == "obj_hold":
            obs_keys = list(kwargs.get("obs_keys", ObjHoldFixedEnvV0.DEFAULT_OBS_KEYS))
            self.rwd_keys_wt = dict(
                kwargs.get(
                    "weighted_reward_keys",
                    ObjHoldFixedEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS,
                )
            )
            self._is_random_task = "ObjHoldRandomEnvV0" in entry_str
            ref_data = mujoco.MjData(ref_model)
            mujoco.mj_forward(ref_model, ref_data)
            self._object_sid = int(ref_model.site("object").id)
            self._goal_sid = int(ref_model.site("goal").id)
            self._object_init_pos = ref_data.site_xpos[self._object_sid].copy()
            init_qp = ref_data.qpos.ravel().astype(np.float32).copy()
            init_qp[:-7] = 0.0
            init_qp[0] = -1.5
            self._init_qpos = init_qp
            self._init_qvel = ref_data.qvel.ravel().astype(np.float32).copy()
            self.tip_sids = []
            self.target_sids = []
        elif self.task_kind == "pen_twirl":
            obs_keys = list(kwargs.get("obs_keys", PenTwirlFixedEnvV0.DEFAULT_OBS_KEYS))
            self.rwd_keys_wt = dict(
                kwargs.get(
                    "weighted_reward_keys",
                    PenTwirlFixedEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS,
                )
            )
            self._is_random_task = "PenTwirlRandomEnvV0" in entry_str
            ref_data = mujoco.MjData(ref_model)
            mujoco.mj_forward(ref_model, ref_data)
            self._target_obj_bid = int(ref_model.body("target").id)
            self._obj_bid = int(ref_model.body("Object").id)
            self._eps_ball_sid = int(ref_model.site("eps_ball").id)
            self._obj_t_sid = int(ref_model.site("object_top").id)
            self._obj_b_sid = int(ref_model.site("object_bottom").id)
            self._tar_t_sid = int(ref_model.site("target_top").id)
            self._tar_b_sid = int(ref_model.site("target_bottom").id)
            self.pen_length = float(
                np.linalg.norm(
                    ref_model.site_pos[self._obj_t_sid]
                    - ref_model.site_pos[self._obj_b_sid]
                )
            )
            self.tar_length = float(
                np.linalg.norm(
                    ref_model.site_pos[self._tar_t_sid]
                    - ref_model.site_pos[self._tar_b_sid]
                )
            )
            init_qp = ref_data.qpos.ravel().astype(np.float32).copy()
            init_qp[:-6] = 0.0
            init_qp[0] = -1.5
            self._init_qpos = init_qp
            self._init_qvel = ref_data.qvel.ravel().astype(np.float32).copy()
            self.tip_sids = []
            self.target_sids = []
        elif self.task_kind == "key_turn":
            obs_keys = list(kwargs.get("obs_keys", KeyTurnEnvV0.DEFAULT_OBS_KEYS))
            self.rwd_keys_wt = dict(
                kwargs.get(
                    "weighted_reward_keys", KeyTurnEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS
                )
            )
            self.goal_th = float(kwargs.get("goal_th", defaults.get("goal_th", 3.14)))
            self.key_init_range = tuple(
                kwargs.get("key_init_range", defaults.get("key_init_range", (0.0, 0.0)))
            )
            sfx = "" if kwargs.get("model_recipe") == "hand_keyturn" else "_r"
            ref_data = mujoco.MjData(ref_model)
            mujoco.mj_forward(ref_model, ref_data)
            self._keyhead_sid = int(ref_model.site("keyhead").id)
            self._if_sid = int(ref_model.site(f"IFtip{sfx}").id)
            self._th_sid = int(ref_model.site(f"THtip{sfx}").id)
            self._key_init_pos = ref_data.site_xpos[self._keyhead_sid].copy()
            init_qp = ref_data.qpos.ravel().astype(np.float32).copy()
            init_qp[:-1] = 0.0
            self._init_qpos = init_qp
            self._init_qvel = ref_data.qvel.ravel().astype(np.float32).copy()
            self.tip_sids = []
            self.target_sids = []
        else:
            obs_keys = list(kwargs.get("obs_keys", LegWalkEnvV0.DEFAULT_OBS_KEYS))
            self.rwd_keys_wt = dict(
                kwargs.get(
                    "weighted_reward_keys", LegWalkEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS
                )
            )
            self.terrain: str | None = kwargs.get("terrain")
            self.terrain_variant: str | None = kwargs.get("variant")
            if self.terrain is not None:
                self.sim.set_geom_pos("terrain", [0.0, 0.0, 0.0])
            self.min_height = float(
                kwargs.get("min_height", defaults.get("min_height", 0.8))
            )
            self.max_rot = float(kwargs.get("max_rot", defaults.get("max_rot", 0.8)))
            self.hip_period = int(
                kwargs.get("hip_period", defaults.get("hip_period", 100))
            )
            self.reset_type = str(
                kwargs.get("reset_type", defaults.get("reset_type", "init"))
            )
            self.target_x_vel = float(
                kwargs.get("target_x_vel", defaults.get("target_x_vel", 0.0))
            )
            self.target_y_vel = float(
                kwargs.get("target_y_vel", defaults.get("target_y_vel", 1.2))
            )
            self._init_qpos = ref_model.key_qpos[0].astype(np.float32).copy()
            self._init_qvel = np.zeros_like(ref_model.key_qvel[0], dtype=np.float32)
            raw_target_rot = kwargs.get("target_rot", defaults.get("target_rot"))
            target_rot_np = np.asarray(
                raw_target_rot if raw_target_rot is not None else self._init_qpos[3:7],
                dtype=np.float32,
            )
            self._target_rot_t = torch.as_tensor(
                target_rot_np, dtype=torch.float32, device=self.torch_device
            )
            self._target_vel_t = torch.tensor(
                [self.target_x_vel, self.target_y_vel],
                dtype=torch.float32,
                device=self.torch_device,
            )
            self._body_mass = torch.as_tensor(
                ref_model.body_mass, dtype=torch.float32, device=self.torch_device
            )
            self._total_mass = self._body_mass.sum()
            self._body_rootid = torch.as_tensor(
                ref_model.body_rootid, dtype=torch.long, device=self.torch_device
            )
            self._torso_bid = int(ref_model.body("torso").id)
            self._pelvis_bid = int(ref_model.body("pelvis").id)
            self._talus_l_bid = int(ref_model.body("talus_l").id)
            self._talus_r_bid = int(ref_model.body("talus_r").id)
            self._feet_bids = torch.tensor(
                [self._talus_l_bid, self._talus_r_bid],
                dtype=torch.long,
                device=self.torch_device,
            )

            def _qadr(name: str) -> int:
                return int(ref_model.jnt_qposadr[ref_model.joint(name).id])

            self._hip_flex_indices = (_qadr("hip_flexion_l"), _qadr("hip_flexion_r"))
            self._hip_angle_indices = (
                _qadr("hip_adduction_l"),
                _qadr("hip_adduction_r"),
                _qadr("hip_rotation_l"),
                _qadr("hip_rotation_r"),
            )
            self._on_target_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.torch_device
            )
            self.tip_sids = []
            self.target_sids = []

        if ref_model.na > 0 and "act" not in obs_keys:
            obs_keys.append("act")
        self.obs_keys = tuple(obs_keys)
        self.num_actions = self.sim.nu
        obs0, _ = self.reset_torch(seed=seed)
        self.num_obs = int(obs0.shape[-1])

    def _walk_kinematics_torch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_acc = self.sim.torch_accessor
        xipos = t_acc.body_xipos()
        cvel = t_acc.body_cvel()
        sub_com = t_acc.subtree_com()
        com = (self._body_mass[None, :, None] * xipos).sum(dim=1) / self._total_mass
        com_vel = -com_linear_velocity(
            torch, self._body_mass, cvel, xipos, sub_com[:, self._body_rootid]
        )[:, :2]
        steps_t = torch.as_tensor(
            self.sim.step_count, dtype=torch.float32, device=self.torch_device
        )
        phase_var = ((steps_t / float(self.hip_period)) % 1.0).unsqueeze(-1)
        return com, com_vel, phase_var

    def _compute_obs_torch(self) -> torch.Tensor:
        t_acc = self.sim.torch_accessor
        t_col = t_acc.time().reshape(self.num_envs, 1)
        qpos = joint_pos_obs(t_acc).reshape(self.num_envs, -1)
        obs_map: dict[str, torch.Tensor] = {
            "t": t_col,
            "time": t_col,
            "qpos": qpos,
            "qvel": joint_vel_obs(t_acc).reshape(self.num_envs, -1),
            "act": muscle_act_obs(t_acc).reshape(self.num_envs, -1),
        }
        if self.task_kind == "pose":
            obs_map["pose_err"] = pose_error_obs(
                t_acc, target=self.target_angles
            ).reshape(self.num_envs, -1)
        elif self.task_kind in ("reach", "leg_reach"):
            tip = tip_pos_obs(t_acc, self.tip_sids).reshape(self.num_envs, -1)
            obs_map["tip_pos"] = tip
            obs_map["target_pos"] = self.target_pos
            obs_map["reach_err"] = self.target_pos - tip
        elif self.task_kind == "obj_hold":
            qvel_dt = obs_map["qvel"]
            obj_pos = t_acc.site_xpos(self._object_sid)
            goal_pos = t_acc.site_xpos(self._goal_sid)
            obs_map["hand_qpos"] = qpos[:, :-7]
            obs_map["hand_qvel"] = qvel_dt[:, :-6]
            obs_map["obj_pos"] = obj_pos
            obs_map["obj_err"] = goal_pos - obj_pos
        elif self.task_kind == "pen_twirl":
            qvel_dt = obs_map["qvel"]
            obj_pos = t_acc.body_xpos()[:, self._obj_bid]
            obj_des_pos = t_acc.site_xpos(self._eps_ball_sid)
            obj_rot = (
                t_acc.site_xpos(self._obj_t_sid) - t_acc.site_xpos(self._obj_b_sid)
            ) / self.pen_length
            obj_des_rot = (
                t_acc.site_xpos(self._tar_t_sid) - t_acc.site_xpos(self._tar_b_sid)
            ) / self.tar_length
            obs_map["hand_jnt"] = qpos[:, :-6]
            obs_map["obj_pos"] = obj_pos
            obs_map["obj_des_pos"] = obj_des_pos
            obs_map["obj_vel"] = qvel_dt[:, -6:]
            obs_map["obj_rot"] = obj_rot
            obs_map["obj_des_rot"] = obj_des_rot
            obs_map["obj_err_pos"] = obj_pos - obj_des_pos
            obs_map["obj_err_rot"] = obj_rot - obj_des_rot
        elif self.task_kind == "key_turn":
            qvel_dt = obs_map["qvel"]
            keyhead_xpos = t_acc.site_xpos(self._keyhead_sid)
            obs_map["hand_qpos"] = qpos[:, :-1]
            obs_map["hand_qvel"] = qvel_dt[:, :-1]
            obs_map["key_qpos"] = qpos[:, -1:]
            obs_map["key_qvel"] = qvel_dt[:, -1:]
            obs_map["IFtip_approach"] = keyhead_xpos - t_acc.site_xpos(self._if_sid)
            obs_map["THtip_approach"] = keyhead_xpos - t_acc.site_xpos(self._th_sid)
        else:
            xpos = t_acc.body_xpos()
            xquat = t_acc.body_xquat()
            com, com_vel, phase_var = self._walk_kinematics_torch()
            obs_map["qpos_without_xy"] = qpos[:, 2:]
            obs_map["com_vel"] = com_vel
            obs_map["torso_angle"] = xquat[:, self._torso_bid]
            obs_map["feet_heights"] = xpos[:, self._feet_bids, 2]
            obs_map["height"] = com[:, 2:3]
            obs_map["feet_rel_positions"] = torch.cat(
                [
                    xpos[:, self._talus_l_bid] - xpos[:, self._pelvis_bid],
                    xpos[:, self._talus_r_bid] - xpos[:, self._pelvis_bid],
                ],
                dim=-1,
            )
            obs_map["phase_var"] = phase_var
            obs_map["muscle_length"] = t_acc.muscle_length()
            obs_map["muscle_velocity"] = torch.clamp(
                t_acc.muscle_velocity(), -100.0, 100.0
            )
            obs_map["muscle_force"] = torch.clamp(
                t_acc.muscle_force() / 1000.0, -100.0, 100.0
            )
        return torch.cat([obs_map[k] for k in self.obs_keys], dim=-1).to(
            dtype=torch.float32
        )

    def _compute_reward_dict_torch(self) -> dict[str, torch.Tensor]:
        t_acc = self.sim.torch_accessor
        act = t_acc.muscle_act()
        act_mag = (
            torch.linalg.norm(act, dim=-1) / act.shape[-1]
            if act.shape[-1] > 0
            else torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.torch_device
            )
        )
        if self.task_kind == "pose":
            comp = pose_reward(
                t_acc,
                {"target_angles": self.target_angles},
                pose_thd=self.pose_thd,
                far_thd=self.pose_far_thd,
            )
            rwd: dict[str, torch.Tensor] = {
                "pose": comp["pose"],
                "bonus": comp["bonus"],
                "penalty": comp["penalty"],
                "act_reg": -act_mag,
                "sparse": comp["pose"],
                "solved": comp["solved"],
                "done": comp["done"],
            }
        elif self.task_kind in ("reach", "leg_reach"):
            reward_fn = (
                leg_reach_reward
                if self.task_kind == "leg_reach"
                else multi_site_reach_reward
            )
            rwd = reward_fn(
                t_acc,
                {
                    "tip_site_ids": self.tip_sids,
                    "target_pos": self.target_pos,
                    "penalty_active": t_acc.time() > 2.0 * self.sim.ctrl_dt,
                },
                far_th=self.far_th,
            )
        elif self.task_kind == "obj_hold":
            obj_err = t_acc.site_xpos(self._goal_sid) - t_acc.site_xpos(
                self._object_sid
            )
            goal_dist = torch.linalg.norm(obj_err, dim=-1)
            goal_th = 0.010
            drop = goal_dist > 0.300
            rwd = {
                "goal_dist": -goal_dist,
                "bonus": 1.0 * (goal_dist < 2.0 * goal_th)
                + 1.0 * (goal_dist < goal_th),
                "act_reg": -act_mag,
                "penalty": -1.0 * drop,
                "sparse": -goal_dist,
                "solved": goal_dist < goal_th,
                "done": drop,
            }
        elif self.task_kind == "pen_twirl":
            obj_pos = t_acc.body_xpos()[:, self._obj_bid]
            obj_des_pos = t_acc.site_xpos(self._eps_ball_sid)
            obj_rot = (
                t_acc.site_xpos(self._obj_t_sid) - t_acc.site_xpos(self._obj_b_sid)
            ) / self.pen_length
            obj_des_rot = (
                t_acc.site_xpos(self._tar_t_sid) - t_acc.site_xpos(self._tar_b_sid)
            ) / self.tar_length
            pos_align = torch.linalg.norm(obj_pos - obj_des_pos, dim=-1)
            norm_prod = torch.linalg.norm(obj_rot, dim=-1) * torch.linalg.norm(
                obj_des_rot, dim=-1
            )
            rot_align = (obj_rot * obj_des_rot).sum(dim=-1) / torch.where(
                norm_prod == 0.0, torch.ones_like(norm_prod), norm_prod
            )
            dropped = pos_align > 0.075
            in_zone = pos_align < 0.075
            rwd = {
                "pos_align": -pos_align,
                "rot_align": rot_align,
                "act_reg": -act_mag,
                "drop": -1.0 * dropped,
                "bonus": 1.0 * ((rot_align > 0.9) & in_zone)
                + 5.0 * ((rot_align > 0.95) & in_zone),
                "sparse": -pos_align + rot_align,
                "solved": (rot_align > 0.95) & ~dropped,
                "done": dropped,
            }
        elif self.task_kind == "key_turn":
            key_pos = t_acc.joint_pos()[:, -1]
            keyhead_xpos = t_acc.site_xpos(self._keyhead_sid)
            if_dist = torch.abs(
                torch.linalg.norm(keyhead_xpos - t_acc.site_xpos(self._if_sid), dim=-1)
                - 0.030
            )
            th_dist = torch.abs(
                torch.linalg.norm(keyhead_xpos - t_acc.site_xpos(self._th_sid), dim=-1)
                - 0.030
            )
            far_th = 0.1
            rwd = {
                "key_turn": key_pos,
                "IFtip_approach": -if_dist,
                "THtip_approach": -th_dist,
                "act_reg": -act_mag,
                "bonus": 1.0 * (key_pos > math.pi / 2.0) + 1.0 * (key_pos > math.pi),
                "penalty": -1.0 * (if_dist > far_th / 2.0)
                - 1.0 * (th_dist > far_th / 2.0),
                "sparse": key_pos,
                "solved": key_pos > self.goal_th,
                "done": (if_dist > far_th) | (th_dist > far_th),
            }
        else:
            qpos = t_acc.joint_pos()
            com, com_vel, phase_var = self._walk_kinematics_torch()
            comp = walk_env_reward(
                t_acc,
                {
                    "qpos": qpos,
                    "height": com[:, 2],
                    "com_vel": com_vel,
                    "phase_var": phase_var,
                },
                hip_flex_indices=self._hip_flex_indices,
                hip_angle_indices=self._hip_angle_indices,
                target_rot=self._target_rot_t,
                target_vel=(self.target_x_vel, self.target_y_vel),
                min_height=self.min_height,
                max_rot=self.max_rot,
            )
            done_bool = comp["done"] > 0.5
            if self.terrain is not None:
                feet_z = t_acc.body_xpos()[:, self._feet_bids, 2]
                knee_fail = (com[:, 2] - feet_z.mean(dim=-1)) < 0.61
                done_bool = done_bool | knee_fail
                comp["done"] = done_bool.to(dtype=comp["vel_reward"].dtype)
            vel_error = torch.linalg.norm(self._target_vel_t - com_vel, dim=-1)
            on_target = locomotion_solved(torch, vel_error, done_bool)
            streak = locomotion_streak(self._on_target_steps, on_target)
            rwd = {
                "vel_reward": comp["vel_reward"],
                "cyclic_hip": comp["cyclic_hip"],
                "ref_rot": comp["ref_rot"],
                "joint_angle_rew": comp["joint_angle_rew"],
                "act_mag": act_mag,
                "sparse": comp["vel_reward"],
                "solved": locomotion_sustained(streak, self.sim.ctrl_dt),
                "on_target": on_target,
                "done": comp["done"],
                "_streak": streak,
            }
        dense = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.torch_device
        )
        for key, wt in self.rwd_keys_wt.items():
            dense = dense + float(wt) * rwd[key].to(dtype=torch.float32)
        rwd["dense"] = dense
        return rwd

    def reset_torch(
        self,
        *,
        seed: int | None = None,
        env_ids: Sequence[int] | np.ndarray | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset all or selected worlds matching CPU ``PoseEnvV0`` / ``ReachEnvV0`` RNG order."""
        selected = (
            np.arange(self.num_envs, dtype=np.intp)
            if env_ids is None
            else np.asarray(env_ids, dtype=np.intp).reshape(-1)
        )
        if len(selected) == 0:
            return self._compute_obs_torch(), {}

        if seed is not None:
            for w in selected:
                self._rngs[int(w)], _ = gym.utils.seeding.np_random(
                    seed if int(w) == 0 else seed + int(w)
                )

        ref_model = self.sim.reference_model
        qpos_batch = np.broadcast_to(
            self._init_qpos.astype(np.float64), (len(selected), self.sim.nq)
        ).copy()
        qvel_batch = np.broadcast_to(
            self._init_qvel.astype(np.float64), (len(selected), self.sim.nv)
        ).copy()

        if self.task_kind == "pose":
            targets = np.empty(
                (len(selected), self.target_angles.shape[-1]), dtype=np.float32
            )
            for row, w in enumerate(selected):
                rng = self._rngs[int(w)]
                if self.target_type == "fixed":
                    targets[row] = self._fixed_target
                else:
                    assert self._target_jnt_range is not None
                    targets[row] = rng.uniform(
                        low=self._target_jnt_range[:, 0],
                        high=self._target_jnt_range[:, 1],
                    ).astype(np.float32)
                if self.reset_type == "random":
                    qpos_batch[row] = rng.uniform(
                        low=ref_model.jnt_range[:, 0],
                        high=ref_model.jnt_range[:, 1],
                    )
                elif self.reset_type == "none":
                    qpos_batch[row] = ref_model.qpos0
            idx_t = torch.as_tensor(
                selected, dtype=torch.long, device=self.torch_device
            )
            self.target_angles[idx_t] = torch.as_tensor(
                targets, dtype=torch.float32, device=self.torch_device
            )
        elif self.task_kind == "reach":
            n_sites = len(self.tip_sids)
            targets = np.empty((len(selected), 3 * n_sites), dtype=np.float32)
            spans = list(self.target_reach_range.values())
            for row, w in enumerate(selected):
                rng = self._rngs[int(w)]
                if self._workspace_points is not None:
                    choice = self._workspace_points[
                        rng.integers(len(self._workspace_points))
                    ]
                    targets[row] = np.asarray(choice, dtype=np.float32).reshape(-1)
                else:
                    pts = [rng.uniform(low=sp[0], high=sp[1]) for sp in spans]
                    targets[row] = np.concatenate(pts).astype(np.float32)
            idx_t = torch.as_tensor(
                selected, dtype=torch.long, device=self.torch_device
            )
            self.target_pos[idx_t] = torch.as_tensor(
                targets, dtype=torch.float32, device=self.torch_device
            )
        elif self.task_kind == "leg_reach":
            jnt_adr = ref_model.jnt_qposadr
            if np.ptp(self.joint_random_range) > 0:
                for row, w in enumerate(selected):
                    rng = self._rngs[int(w)]
                    qpos_rand = rng.uniform(
                        low=self.joint_random_range[0],
                        high=self.joint_random_range[1],
                        size=self._init_qpos.shape,
                    )
                    qp = self._init_qpos.astype(np.float64).copy()
                    qp[jnt_adr] += qpos_rand[jnt_adr]
                    qp[jnt_adr] = np.clip(
                        qp[jnt_adr],
                        ref_model.jnt_range[:, 0],
                        ref_model.jnt_range[:, 1],
                    )
                    qpos_batch[row] = qp
        elif self.task_kind == "obj_hold":
            if self._is_random_task:
                goals = np.empty((len(selected), 3), dtype=np.float32)
                sizes = np.empty((len(selected), 3), dtype=np.float32)
                for row, w in enumerate(selected):
                    rng = self._rngs[int(w)]
                    goals[row] = self._object_init_pos + rng.uniform(
                        high=np.array([0.030, 0.030, 0.030]),
                        low=np.array([-0.030, -0.030, -0.030]),
                    )
                    sizes[row] = rng.uniform(
                        high=np.array([0.030, 0.030, 0.030]),
                        low=np.array([0.020, 0.020, 0.020]),
                    )
                self.sim.set_site_pos(self._goal_sid, goals, env_ids=selected)
                self.sim.set_geom_size(-1, sizes, env_ids=selected)
        elif self.task_kind == "pen_twirl":
            pen_quats: np.ndarray | None = None
            if self._is_random_task:
                pen_quats = np.empty((len(selected), 4), dtype=np.float32)
                for row, w in enumerate(selected):
                    rng = self._rngs[int(w)]
                    desired_orien = np.zeros(3)
                    desired_orien[0] = rng.uniform(low=-1.0, high=1.0)
                    desired_orien[1] = rng.uniform(low=-1.0, high=1.0)
                    pen_quats[row] = euler2quat(desired_orien)
        elif self.task_kind == "key_turn":
            randomize_pos = self.key_init_range[0] != self.key_init_range[1]
            key_pos_batch: np.ndarray | None = (
                np.empty((len(selected), 3), dtype=np.float32)
                if randomize_pos
                else None
            )
            for row, w in enumerate(selected):
                rng = self._rngs[int(w)]
                qpos_batch[row, -1] = rng.uniform(
                    low=self.key_init_range[0], high=self.key_init_range[1]
                )
                if key_pos_batch is not None:
                    key_pos_batch[row] = self._key_init_pos + rng.uniform(
                        low=np.array([-0.01, -0.01, -0.01]),
                        high=np.array([0.01, 0.01, 0.01]),
                    )
        else:
            for row, w in enumerate(selected):
                rng = self._rngs[int(w)]
                if self.terrain is not None:
                    heights = terrain_heights(self.terrain, self.terrain_variant, rng)
                    if int(w) == 0:
                        self.sim.set_hfield_data(heights)
                if self.reset_type == "random":
                    k_idx = 2 if rng.uniform() < 0.5 else 3
                    qp = ref_model.key_qpos[k_idx].copy()
                    qv = ref_model.key_qvel[k_idx].copy()
                    rot_state = qp[3:7].copy()
                    height = qp[2]
                    qp = qp + rng.normal(0.0, 0.02, size=qp.shape)
                    qp[3:7] = rot_state
                    qp[2] = height
                    qpos_batch[row] = qp
                    qvel_batch[row] = qv
                elif self.reset_type == "init":
                    qpos_batch[row] = ref_model.key_qpos[2]
                    qvel_batch[row] = ref_model.key_qvel[2]
            idx_t = torch.as_tensor(
                selected, dtype=torch.long, device=self.torch_device
            )
            self._on_target_steps[idx_t] = 0

        project_joint_couplings(self._couplings, qpos_batch, qvel_batch)

        _ = self.sim.action_pipeline
        self.sim.reset(
            seed=seed if env_ids is None else None,
            env_ids=None if env_ids is None else selected,
            qpos=qpos_batch,
            qvel=qvel_batch,
        )
        if self.task_kind == "pen_twirl" and pen_quats is not None:
            self.sim.set_body_pose(
                self._target_obj_bid, quat=pen_quats, env_ids=selected
            )
        elif self.task_kind == "key_turn" and key_pos_batch is not None:
            self.sim.set_body_pose(-1, pos=key_pos_batch, env_ids=selected)
        if self.task_kind == "leg_reach":
            n_sites = len(self.tip_sids)
            targets = np.empty((len(selected), 3 * n_sites), dtype=np.float32)
            tip_xpos = self.sim.accessor.site_xpos(self.tip_sids)
            spans = list(self.target_reach_range.values())
            for row, w in enumerate(selected):
                rng = self._rngs[int(w)]
                w_int = int(w)
                pts = [
                    tip_xpos[w_int, s_i] + rng.uniform(low=sp[0], high=sp[1])
                    for s_i, sp in enumerate(spans)
                ]
                targets[row] = np.concatenate(pts).astype(np.float32)
            for s_i, target_sid in enumerate(self.target_sids):
                self.sim.set_site_pos(
                    target_sid, targets[:, 3 * s_i : 3 * (s_i + 1)], env_ids=selected
                )
            idx_t = torch.as_tensor(
                selected, dtype=torch.long, device=self.torch_device
            )
            self.target_pos[idx_t] = torch.as_tensor(
                targets, dtype=torch.float32, device=self.torch_device
            )
        return self._compute_obs_torch(), {"time": self.sim.torch_accessor.time()}

    def step_torch(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step ``num_envs`` worlds and return ``(obs, reward, terminated, truncated, info)``."""
        pipeline = self.sim.action_pipeline
        pipeline.write_to_control(
            actions, self.sim.control, self.sim._conv_to_src_ctrl_torch
        )
        for _ in range(self.sim.frame_skip):
            self.sim.solver.step(
                self.sim.state_in,
                self.sim.state_out,
                self.sim.control,
                None,
                self.sim.sim_dt,
            )
            self.sim.state_in, self.sim.state_out = (
                self.sim.state_out,
                self.sim.state_in,
            )
        self.sim._refresh_derived()
        self.sim.step_count += 1

        final_obs = self._compute_obs_torch()
        rwd_dict = self._compute_reward_dict_torch()
        if self.task_kind == "walk":
            self._on_target_steps = rwd_dict.pop("_streak")
        reward = rwd_dict["dense"]
        terminated = rwd_dict["done"].to(dtype=torch.bool)
        truncated = (
            torch.as_tensor(
                self.sim.step_count >= int(self.max_episode_steps),
                dtype=torch.bool,
                device=self.torch_device,
            )
            if self.max_episode_steps is not None
            else torch.zeros(self.num_envs, dtype=torch.bool, device=self.torch_device)
        )
        dones = terminated | truncated
        info: dict[str, Any] = {
            "rwd_dict": rwd_dict,
            "solved": rwd_dict["solved"].to(dtype=torch.bool),
            "time_outs": truncated & ~terminated,
            "final_obs": final_obs,
        }
        if self.auto_reset and torch.any(dones):
            done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1).cpu().numpy()
            obs, _ = self.reset_torch(env_ids=done_ids)
        else:
            obs = final_obs
        return obs, reward, terminated, truncated, info


class NewtonRslRlVecEnv(_RslRlVecEnvBase):
    """Experimental ``rsl_rl.env.VecEnv`` adapter wrapping :class:`NewtonTaskEnv`."""

    def __init__(
        self,
        task_env: NewtonTaskEnv | str,
        num_envs: int = 64,
        *,
        device: str = "cuda:0",
        seed: int | None = 0,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        from tensordict import TensorDict

        self._td_cls = TensorDict
        self.task_env = (
            task_env
            if isinstance(task_env, NewtonTaskEnv)
            else NewtonTaskEnv(
                task_env, num_envs=num_envs, device=device, seed=seed, auto_reset=True
            )
        )
        self.num_envs = self.task_env.num_envs
        self.num_obs = self.task_env.num_obs
        self.num_actions = self.task_env.num_actions
        self.max_episode_length = int(self.task_env.max_episode_steps or 100)
        self.device = self.task_env.torch_device
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.cfg = {"env_id": self.task_env.env_id} if cfg is None else cfg
        self._obs, _ = self.task_env.reset_torch(seed=seed)

    def _pack_obs(self, obs: torch.Tensor) -> Any:
        return self._td_cls(
            {"policy": obs.clone()},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def get_observations(self) -> Any:
        return self._pack_obs(self._obs)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.task_env.step_torch(actions)
        dones = terminated | truncated
        self.episode_length_buf += 1
        self.episode_length_buf[dones] = 0
        self._obs = obs
        extras: dict[str, Any] = {
            "time_outs": info["time_outs"],
            "final_obs": self._pack_obs(info["final_obs"]),
            "log": {"/solved": info["solved"].to(dtype=torch.float32).mean()},
        }
        return self._pack_obs(obs), reward, dones, extras
