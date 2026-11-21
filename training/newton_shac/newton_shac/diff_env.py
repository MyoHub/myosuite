# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Differentiable batched MuJoCo Warp musculoskeletal environment for PoseEnvV0 tasks."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from myosuite import make_env
from myosuite.terms.base_action import sigmoid_muscle_activation

try:
    from mujoco_warp._src import adjoint as _mjw_adjoint

    _HAS_MJW_ADJOINT = hasattr(mjw, "enable_grad") and hasattr(
        _mjw_adjoint, "step_backward"
    )
except ImportError:  # pragma: no cover
    _mjw_adjoint = None  # type: ignore[assignment]
    _HAS_MJW_ADJOINT = False

_GRAD_FIELDS = ("qpos", "qvel", "act", "ctrl", "qacc", "act_dot")


class _MuscleStepFn(torch.autograd.Function):
    """Differentiable ``frame_skip``-substep musculoskeletal transition."""

    @staticmethod
    def forward(
        ctx: Any,
        env: DifferentiablePoseEnv,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        act: torch.Tensor,
        ctrl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.env = env
        ctx.save_for_backward(
            qpos.detach().clone(),
            qvel.detach().clone(),
            act.detach().clone(),
            ctrl.detach().clone(),
        )
        q_next, v_next, a_next = env._forward_substeps(qpos, qvel, act, ctrl)
        return q_next, v_next, a_next

    @staticmethod
    def backward(
        ctx: Any,
        grad_qpos: torch.Tensor,
        grad_qvel: torch.Tensor,
        grad_act: torch.Tensor,
    ) -> tuple[None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        env: DifferentiablePoseEnv = ctx.env
        qpos, qvel, act, ctrl = ctx.saved_tensors
        gq, gv, ga, gu = env._backward_substeps(
            qpos, qvel, act, ctrl, grad_qpos, grad_qvel, grad_act
        )
        return None, gq, gv, ga, gu


class DifferentiablePoseEnv:
    """Batched GPU musculoskeletal ``PoseEnvV0`` with analytic ``mujoco_warp`` VJPs."""

    def __init__(
        self,
        env_id: str,
        num_envs: int = 128,
        *,
        device: str = "cuda:0",
        seed: int = 0,
        naconmax: int = 128,
        njmax: int = 256,
        bonus_smooth_temp: float = 0.05,
    ) -> None:
        cpu_env = make_env(env_id)
        try:
            unwrapped = cpu_env.unwrapped
            self.env_id = str(env_id)
            self.mj_model: mujoco.MjModel = unwrapped.model
            self.frame_skip = int(unwrapped.frame_skip)
            self.sim_dt = float(self.mj_model.opt.timestep)
            self.ctrl_dt = float(self.sim_dt * self.frame_skip)
            self.max_episode_steps = int(cpu_env.spec.max_episode_steps or 100)
            self.normalize_act = bool(unwrapped.normalize_act)
            self.pose_thd = float(unwrapped.pose_thd)
            self.rwd_keys_wt = dict(unwrapped.rwd_keys_wt)
            self.obs_keys = tuple(unwrapped.obs_keys)
            self.reset_type = str(unwrapped.reset_type)
            self.target_type = str(unwrapped.target_type)
            self._init_qpos = unwrapped._init_qpos.astype(np.float32).copy()
            self._init_qvel = unwrapped._init_qvel.astype(np.float32).copy()
            self._target_jnt_range = (
                None
                if unwrapped._target_jnt_range is None
                else unwrapped._target_jnt_range.astype(np.float64).copy()
            )
            self._fixed_target = np.asarray(
                unwrapped.target_jnt_value, dtype=np.float32
            ).copy()
        finally:
            cpu_env.close()

        self.num_envs = int(num_envs)
        self.bonus_smooth_temp = float(bonus_smooth_temp)
        self.wp_device = device if wp.is_cuda_available() else "cpu"
        self.device = torch.device(
            self.wp_device
            if (self.wp_device.startswith("cuda") and torch.cuda.is_available())
            else "cpu"
        )
        self._dlpack = str(self.device) == str(self.wp_device)

        self.nq = int(self.mj_model.nq)
        self.nv = int(self.mj_model.nv)
        self.nu = int(self.mj_model.nu)
        self.na = int(self.mj_model.na)
        self.num_actions = self.nu

        muscle_ids = [
            i
            for i in range(self.nu)
            if int(self.mj_model.actuator_dyntype[i]) == int(mujoco.mjtDyn.mjDYN_MUSCLE)
        ]
        self.muscle_cols = torch.tensor(
            muscle_ids, dtype=torch.long, device=self.device
        )
        ctrl_range = torch.as_tensor(
            self.mj_model.actuator_ctrlrange, dtype=torch.float32, device=self.device
        )
        self._ctrl_lo = ctrl_range[:, 0]
        self._ctrl_hi = ctrl_range[:, 1]
        self._ctrl_limited = torch.as_tensor(
            self.mj_model.actuator_ctrllimited.astype(bool),
            dtype=torch.bool,
            device=self.device,
        )

        self.has_analytic_grad = bool(_HAS_MJW_ADJOINT)
        if self.has_analytic_grad:
            mjw.enable_grad()

        with wp.ScopedDevice(self.wp_device):
            self.mjw_model = mjw.put_model(self.mj_model)
            self._d_in = [
                mjw.make_data(
                    self.mj_model,
                    nworld=self.num_envs,
                    naconmax=naconmax,
                    njmax=njmax,
                )
                for _ in range(self.frame_skip)
            ]
            self._d_out = [
                mjw.make_data(
                    self.mj_model,
                    nworld=self.num_envs,
                    naconmax=naconmax,
                    njmax=njmax,
                )
                for _ in range(self.frame_skip)
            ]
            if self.has_analytic_grad:
                for buf in self._d_in + self._d_out:
                    for fname in _GRAD_FIELDS:
                        arr = getattr(buf, fname)
                        if arr.size > 0:
                            arr.requires_grad = True
                reasons, _ = _mjw_adjoint._model_supported(self.mjw_model)
                if reasons:
                    self.has_analytic_grad = False
                else:
                    self._bc = _mjw_adjoint.create_backward_context(
                        self.mjw_model, self._d_in[0]
                    )

        self._rngs: list[np.random.Generator] = [
            gym.utils.seeding.np_random(seed if w == 0 else seed + w)[0]
            for w in range(self.num_envs)
        ]
        self.qpos = torch.zeros(
            (self.num_envs, self.nq), dtype=torch.float32, device=self.device
        )
        self.qvel = torch.zeros(
            (self.num_envs, self.nv), dtype=torch.float32, device=self.device
        )
        self.act = torch.zeros(
            (self.num_envs, self.na), dtype=torch.float32, device=self.device
        )
        self.target_angles = torch.zeros(
            (self.num_envs, len(self._fixed_target)),
            dtype=torch.float32,
            device=self.device,
        )
        self.step_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        obs0 = self.reset(seed=seed)
        self.num_obs = int(obs0.shape[-1])

    def _copy_to_wp(self, wp_arr: Any, t: torch.Tensor) -> None:
        if wp_arr.size == 0:
            return
        if self._dlpack:
            wp.to_torch(wp_arr).copy_(t.detach())
        else:
            wp_arr.assign(t.detach().cpu().numpy().astype(np.float32))

    def _read_wp(self, wp_arr: Any) -> torch.Tensor:
        if wp_arr.size == 0:
            return torch.zeros(
                (self.num_envs, 0), dtype=torch.float32, device=self.device
            )
        if self._dlpack:
            return wp.to_torch(wp_arr).clone()
        return torch.as_tensor(wp_arr.numpy(), dtype=torch.float32, device=self.device)

    def _forward_substeps(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        act: torch.Tensor,
        ctrl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with wp.ScopedDevice(self.wp_device):
            self._copy_to_wp(self._d_in[0].qpos, qpos)
            self._copy_to_wp(self._d_in[0].qvel, qvel)
            if self.na > 0:
                self._copy_to_wp(self._d_in[0].act, act)
            for k in range(self.frame_skip):
                if k > 0:
                    wp.copy(self._d_in[k].qpos, self._d_out[k - 1].qpos)
                    wp.copy(self._d_in[k].qvel, self._d_out[k - 1].qvel)
                    if self.na > 0:
                        wp.copy(self._d_in[k].act, self._d_out[k - 1].act)
                self._copy_to_wp(self._d_in[k].ctrl, ctrl)
                mjw.step(self.mjw_model, self._d_in[k], self._d_out[k])
            last = self._d_out[-1]
            return (
                self._read_wp(last.qpos),
                self._read_wp(last.qvel),
                self._read_wp(last.act),
            )

    def _backward_substeps(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        act: torch.Tensor,
        ctrl: torch.Tensor,
        grad_qpos: torch.Tensor,
        grad_qvel: torch.Tensor,
        grad_act: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.has_analytic_grad:
            self._forward_substeps(qpos, qvel, act, ctrl)
            with wp.ScopedDevice(self.wp_device):
                for buf in self._d_in + self._d_out:
                    for fname in _GRAD_FIELDS:
                        arr = getattr(buf, fname)
                        if arr.grad is not None:
                            arr.grad.zero_()
                last_out = self._d_out[-1]
                self._copy_to_wp(last_out.qpos.grad, grad_qpos.contiguous())
                self._copy_to_wp(last_out.qvel.grad, grad_qvel.contiguous())
                if self.na > 0:
                    self._copy_to_wp(last_out.act.grad, grad_act.contiguous())

                grad_ctrl = torch.zeros_like(ctrl)
                for k in range(self.frame_skip - 1, -1, -1):
                    _mjw_adjoint.step_backward(
                        self.mjw_model, self._d_in[k], self._d_out[k], self._bc
                    )
                    grad_ctrl = grad_ctrl + self._read_wp(self._d_in[k].ctrl.grad)
                    if k > 0:
                        wp.copy(self._d_out[k - 1].qpos.grad, self._d_in[k].qpos.grad)
                        wp.copy(self._d_out[k - 1].qvel.grad, self._d_in[k].qvel.grad)
                        if self.na > 0:
                            wp.copy(self._d_out[k - 1].act.grad, self._d_in[k].act.grad)
                gq = torch.nan_to_num(self._read_wp(self._d_in[0].qpos.grad), nan=0.0)
                gv = torch.nan_to_num(self._read_wp(self._d_in[0].qvel.grad), nan=0.0)
                ga = torch.nan_to_num(self._read_wp(self._d_in[0].act.grad), nan=0.0)
                gu = torch.nan_to_num(grad_ctrl, nan=0.0)
                return gq, gv, ga, gu

        return self._backward_substeps_fd(
            qpos, qvel, act, ctrl, grad_qpos, grad_qvel, grad_act
        )

    def _backward_substeps_fd(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        act: torch.Tensor,
        ctrl: torch.Tensor,
        grad_qpos: torch.Tensor,
        grad_qvel: torch.Tensor,
        grad_act: torch.Tensor,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = [
            qpos.detach().cpu().numpy().astype(np.float64),
            qvel.detach().cpu().numpy().astype(np.float64),
            act.detach().cpu().numpy().astype(np.float64),
            ctrl.detach().cpu().numpy().astype(np.float64),
        ]
        cot_q = grad_qpos.detach().cpu().numpy().astype(np.float64)
        cot_v = grad_qvel.detach().cpu().numpy().astype(np.float64)
        cot_a = grad_act.detach().cpu().numpy().astype(np.float64)
        grads = [np.zeros_like(arr) for arr in inputs]
        d_cpu = mujoco.MjData(self.mj_model)

        def _eval_world(
            w: int, q_w: np.ndarray, v_w: np.ndarray, a_w: np.ndarray, u_w: np.ndarray
        ) -> float:
            d_cpu.qpos[:] = q_w
            d_cpu.qvel[:] = v_w
            if self.na > 0:
                d_cpu.act[:] = a_w
            for _ in range(self.frame_skip):
                d_cpu.ctrl[:] = u_w
                mujoco.mj_step(self.mj_model, d_cpu)
            val = float(np.sum(d_cpu.qpos * cot_q[w]) + np.sum(d_cpu.qvel * cot_v[w]))
            if self.na > 0:
                val += float(np.sum(d_cpu.act * cot_a[w]))
            return val

        for w in range(self.num_envs):
            w_arrs = [arr[w].copy() for arr in inputs]
            for slot, arr_w in enumerate(w_arrs):
                for j in range(arr_w.size):
                    orig = arr_w[j]
                    arr_w[j] = orig + eps
                    vp = _eval_world(w, *w_arrs)
                    arr_w[j] = orig - eps
                    vm = _eval_world(w, *w_arrs)
                    arr_w[j] = orig
                    grads[slot][w, j] = (vp - vm) / (2.0 * eps)

        return tuple(
            torch.as_tensor(g, dtype=torch.float32, device=self.device) for g in grads
        )  # type: ignore[return-value]

    def actions_to_ctrl(self, actions: torch.Tensor) -> torch.Tensor:
        clamped = actions + (torch.clamp(actions, -1.0, 1.0) - actions).detach()
        if self.normalize_act:
            if self.na > 0 and self.muscle_cols.numel() > 0:
                ctrl = clamped.clone()
                ctrl[:, self.muscle_cols] = sigmoid_muscle_activation(
                    clamped[:, self.muscle_cols], torch
                )
            else:
                ctrl = 0.5 * (self._ctrl_lo + self._ctrl_hi) + 0.5 * clamped * (
                    self._ctrl_hi - self._ctrl_lo
                )
        else:
            ctrl = actions
        if torch.any(self._ctrl_limited):
            lo = self._ctrl_lo[self._ctrl_limited]
            hi = self._ctrl_hi[self._ctrl_limited]
            sub = ctrl[:, self._ctrl_limited]
            ctrl = ctrl.clone()
            ctrl[:, self._ctrl_limited] = (
                sub + (torch.clamp(sub, lo, hi) - sub).detach()
            )
        return ctrl

    def build_obs(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        act: torch.Tensor,
        target_angles: torch.Tensor,
        step_count: torch.Tensor,
    ) -> torch.Tensor:
        tdim = target_angles.shape[-1]
        parts_map = {
            "time": (step_count.to(dtype=torch.float32) * self.ctrl_dt).unsqueeze(-1),
            "qpos": qpos,
            "qvel": qvel * self.ctrl_dt,
            "act": act,
            "pose_err": target_angles - qpos[:, :tdim],
        }
        return torch.cat([parts_map[k] for k in self.obs_keys], dim=-1)

    def compute_rewards(
        self,
        qpos: torch.Tensor,
        act: torch.Tensor,
        target_angles: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(rwd_true, rwd_smooth, solved, terminated)``."""
        tdim = target_angles.shape[-1]
        err = target_angles - qpos[:, :tdim]
        dist = torch.sqrt(torch.sum(err * err, dim=-1) + 1e-8)
        act_mag = (
            torch.sqrt(torch.sum(act * act, dim=-1) + 1e-8) / self.na
            if self.na > 0
            else torch.zeros_like(dist)
        )
        far_thd = 2.0 * math.pi
        wt_pose = float(self.rwd_keys_wt.get("pose", 1.0))
        wt_bonus = float(self.rwd_keys_wt.get("bonus", 4.0))
        wt_act = float(self.rwd_keys_wt.get("act_reg", 1.0))
        wt_pen = float(self.rwd_keys_wt.get("penalty", 50.0))

        solved = dist < self.pose_thd
        terminated = dist > far_thd
        bonus_true = solved.to(dtype=torch.float32) + (dist < 1.5 * self.pose_thd).to(
            dtype=torch.float32
        )
        pen_true = -terminated.to(dtype=torch.float32)
        rwd_true = (
            wt_pose * (-dist)
            + wt_bonus * bonus_true
            + wt_act * (-act_mag)
            + wt_pen * pen_true
        )

        tau = self.bonus_smooth_temp
        bonus_smooth = torch.sigmoid((self.pose_thd - dist) / tau) + torch.sigmoid(
            (1.5 * self.pose_thd - dist) / tau
        )
        rwd_smooth = (
            wt_pose * (-dist)
            + wt_bonus * bonus_smooth
            + wt_act * (-act_mag)
            + wt_pen * pen_true
        )
        return rwd_true, rwd_smooth, solved, terminated

    def _sample_reset_arrays(
        self, env_ids: np.ndarray, *, seed: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if seed is not None:
            for w in env_ids:
                self._rngs[int(w)], _ = gym.utils.seeding.np_random(
                    seed if int(w) == 0 else seed + int(w)
                )
        n = len(env_ids)
        qpos_b = np.broadcast_to(self._init_qpos, (n, self.nq)).copy()
        qvel_b = np.broadcast_to(self._init_qvel, (n, self.nv)).copy()
        act_b = np.zeros((n, self.na), dtype=np.float32)
        targets_b = np.empty((n, len(self._fixed_target)), dtype=np.float32)

        for row, w in enumerate(env_ids):
            rng = self._rngs[int(w)]
            if self.target_type == "fixed":
                targets_b[row] = self._fixed_target
            else:
                assert self._target_jnt_range is not None
                targets_b[row] = rng.uniform(
                    low=self._target_jnt_range[:, 0],
                    high=self._target_jnt_range[:, 1],
                ).astype(np.float32)
            if self.reset_type == "random":
                qpos_b[row] = rng.uniform(
                    low=self.mj_model.jnt_range[:, 0],
                    high=self.mj_model.jnt_range[:, 1],
                ).astype(np.float32)
            elif self.reset_type == "none":
                qpos_b[row] = self.mj_model.qpos0.astype(np.float32)

        return (
            torch.as_tensor(qpos_b, dtype=torch.float32, device=self.device),
            torch.as_tensor(qvel_b, dtype=torch.float32, device=self.device),
            torch.as_tensor(act_b, dtype=torch.float32, device=self.device),
            torch.as_tensor(targets_b, dtype=torch.float32, device=self.device),
        )

    def detach_state(self) -> None:
        """Detach simulation state tensors at truncated-BPTT horizon boundaries."""
        self.qpos = self.qpos.detach()
        self.qvel = self.qvel.detach()
        self.act = self.act.detach()

    def reset(self, *, seed: int | None = None) -> torch.Tensor:
        all_ids = np.arange(self.num_envs, dtype=np.intp)
        q, v, a, tgt = self._sample_reset_arrays(all_ids, seed=seed)
        self.qpos = q
        self.qvel = v
        self.act = a
        self.target_angles = tgt
        self.step_count.zero_()
        return self.build_obs(
            self.qpos, self.qvel, self.act, self.target_angles, self.step_count
        )

    def step_diff(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Differentiable step returning ``(obs, rwd_smooth, dones, info)``."""
        ctrl = self.actions_to_ctrl(actions)
        q_next, v_next, a_next = _MuscleStepFn.apply(
            self, self.qpos, self.qvel, self.act, ctrl
        )
        next_steps = self.step_count + 1
        final_obs = self.build_obs(
            q_next, v_next, a_next, self.target_angles, next_steps
        )
        rwd_true, rwd_smooth, solved, terminated = self.compute_rewards(
            q_next, a_next, self.target_angles
        )
        truncated = next_steps >= self.max_episode_steps
        dones = terminated | truncated

        if torch.any(dones):
            done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1).cpu().numpy()
            rq, rv, ra, rtgt = self._sample_reset_arrays(done_ids)
            q_reset = q_next.detach().clone()
            v_reset = v_next.detach().clone()
            a_reset = a_next.detach().clone()
            idx_t = torch.as_tensor(done_ids, dtype=torch.long, device=self.device)
            q_reset[idx_t] = rq
            v_reset[idx_t] = rv
            a_reset[idx_t] = ra
            self.target_angles = self.target_angles.clone()
            self.target_angles[idx_t] = rtgt
            self.qpos = torch.where(dones.unsqueeze(-1), q_reset, q_next)
            self.qvel = torch.where(dones.unsqueeze(-1), v_reset, v_next)
            self.act = torch.where(dones.unsqueeze(-1), a_reset, a_next)
            self.step_count = torch.where(
                dones, torch.zeros_like(next_steps), next_steps
            )
            obs = self.build_obs(
                self.qpos, self.qvel, self.act, self.target_angles, self.step_count
            )
        else:
            self.qpos = q_next
            self.qvel = v_next
            self.act = a_next
            self.step_count = next_steps
            obs = final_obs

        info = {
            "rwd_true": rwd_true.detach(),
            "solved": solved.detach(),
            "terminated": terminated,
            "truncated": truncated,
            "final_obs": final_obs,
        }
        return obs, rwd_smooth, dones, info
