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

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import torch

from myosuite.envs.myo.tasks.basic.arm.pose import PoseEnvV0
from myosuite.envs.myo.tasks.basic.arm.reach import ReachEnvV0
from myosuite.integrations.newton.accessor import _entry_point_defaults
from myosuite.integrations.newton.batch_sim import NewtonBatchSim
from myosuite.integrations.newton.torch_pipeline import NewtonActionCfg
from myosuite.terms.base_obs import (
    joint_pos_obs,
    joint_vel_obs,
    muscle_act_obs,
    pose_error_obs,
    tip_pos_obs,
)
from myosuite.terms.base_reward import multi_site_reach_reward, pose_reward
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
        import myosuite  # noqa: F401, PLC0415

        self.env_id = str(env_id)
        spec = gym.spec(self.env_id)
        entry_str = str(spec.entry_point)
        if "PoseEnvV0" in entry_str:
            self.task_kind = "pose"
        elif "ReachEnvV0" in entry_str:
            self.task_kind = "reach"
        else:
            raise ValueError(
                f"NewtonTaskEnv supports PoseEnvV0 and ReachEnvV0 tasks, got {entry_str!r}."
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
        self._rngs: list[np.random.Generator] = [
            gym.utils.seeding.np_random(
                None if seed is None else (seed if w == 0 else seed + w)
            )[0]
            for w in range(self.num_envs)
        ]

        if self.task_kind == "pose":
            obs_keys = list(kwargs.get("obs_keys", PoseEnvV0.DEFAULT_OBS_KEYS))
            self.rwd_keys_wt = dict(
                kwargs.get(
                    "weighted_reward_keys", PoseEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS
                )
            )
            self.reset_type = str(
                kwargs.get("reset_type", defaults.get("reset_type", "init"))
            )
            self.target_type = str(
                kwargs.get("target_type", defaults.get("target_type", "generate"))
            )
            self.pose_thd = float(
                kwargs.get("pose_thd", defaults.get("pose_thd", 0.35))
            )
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
        else:
            obs_keys = list(kwargs.get("obs_keys", ReachEnvV0.DEFAULT_OBS_KEYS))
            self.rwd_keys_wt = dict(
                kwargs.get(
                    "weighted_reward_keys", ReachEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS
                )
            )
            self.target_reach_range = dict(kwargs["target_reach_range"])
            self.target_sampling = str(
                kwargs.get("target_sampling", defaults.get("target_sampling", "box"))
            )
            self.far_th = float(kwargs.get("far_th", defaults.get("far_th", 0.35)))
            self.tip_sids = [ref_model.site(s).id for s in self.target_reach_range]
            self.target_sids = [
                ref_model.site(f"{s}_target").id for s in self.target_reach_range
            ]
            self._workspace_points: np.ndarray | None = None
            if self.target_sampling == "workspace":
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

        if ref_model.na > 0 and "act" not in obs_keys:
            obs_keys.append("act")
        self.obs_keys = tuple(obs_keys)
        self.num_actions = self.sim.nu
        obs0, _ = self.reset_torch(seed=seed)
        self.num_obs = int(obs0.shape[-1])

    def _compute_obs_torch(self) -> torch.Tensor:
        t_acc = self.sim.torch_accessor
        obs_map: dict[str, torch.Tensor] = {
            "time": t_acc.time().reshape(self.num_envs, 1),
            "qpos": joint_pos_obs(t_acc).reshape(self.num_envs, -1),
            "qvel": joint_vel_obs(t_acc).reshape(self.num_envs, -1),
            "act": muscle_act_obs(t_acc).reshape(self.num_envs, -1),
        }
        if self.task_kind == "pose":
            obs_map["pose_err"] = pose_error_obs(
                t_acc, target=self.target_angles
            ).reshape(self.num_envs, -1)
        else:
            tip = tip_pos_obs(t_acc, self.tip_sids).reshape(self.num_envs, -1)
            obs_map["tip_pos"] = tip
            obs_map["target_pos"] = self.target_pos
            obs_map["reach_err"] = self.target_pos - tip
        return torch.cat([obs_map[k] for k in self.obs_keys], dim=-1).to(
            dtype=torch.float32
        )

    def _compute_reward_dict_torch(self) -> dict[str, torch.Tensor]:
        t_acc = self.sim.torch_accessor
        if self.task_kind == "pose":
            comp = pose_reward(
                t_acc, {"target_angles": self.target_angles}, pose_thd=self.pose_thd
            )
            act = t_acc.muscle_act()
            act_mag = (
                torch.linalg.norm(act, dim=-1) / act.shape[-1]
                if act.shape[-1] > 0
                else torch.zeros_like(comp["pose"])
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
        else:
            rwd = multi_site_reach_reward(
                t_acc,
                {
                    "tip_site_ids": self.tip_sids,
                    "target_pos": self.target_pos,
                    "penalty_active": t_acc.time() > 2.0 * self.sim.ctrl_dt,
                },
                far_th=self.far_th,
            )
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
            self._init_qpos, (len(selected), self.sim.nq)
        ).copy()
        qvel_batch = np.broadcast_to(
            self._init_qvel, (len(selected), self.sim.nv)
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
                    ).astype(np.float32)
                elif self.reset_type == "none":
                    qpos_batch[row] = ref_model.qpos0.astype(np.float32)
            idx_t = torch.as_tensor(
                selected, dtype=torch.long, device=self.torch_device
            )
            self.target_angles[idx_t] = torch.as_tensor(
                targets, dtype=torch.float32, device=self.torch_device
            )
        else:
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

        _ = self.sim.action_pipeline
        self.sim.reset(
            seed=seed if env_ids is None else None,
            env_ids=None if env_ids is None else selected,
            qpos=qpos_batch,
            qvel=qvel_batch,
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
        from tensordict import TensorDict  # noqa: PLC0415

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
