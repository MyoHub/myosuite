# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Experimental batched N-world Newton ``SolverMuJoCo`` rollout probe.

.. warning::
   **Experimental validation probe only.** CPU ``MyoGymnasiumEnv`` and GPU
   ``mjlab`` are the only two supported MyoSuite backends. Like the experimental
   MJX route, this module exists solely for solver compatibility diagnostics and
   should not be used as a foundation for new MyoSuite tasks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite.core.protocols import EnvAccessor
from myosuite.envs.muscle_stages import ExcitationStage, muscle_mask
from myosuite.integrations.newton.accessor import (
    IndexMapping,
    NewtonAccessor,
    _integrator_name,
    _require_newton,
    resolve_env_xml,
)
from myosuite.terms.base_action import sigmoid_muscle_activation
from myosuite.terms.base_obs import joint_pos_obs, joint_vel_obs, muscle_act_obs


class NewtonBatchSim:
    """Experimental batched N-world Newton ``SolverMuJoCo`` wrapper for MyoSuite models.

    Args:
        env_id_or_xml: Registered MyoSuite env ID or MJCF file path.
        num_envs: Number of parallel worlds to replicate.
        device: Warp device string (e.g. ``"cuda:0"`` or ``"cpu"``).
        frame_skip: Physics substeps per ``step()`` call; defaults to env metadata or 1.
        normalize_act: Apply MyoSuite sigmoid mapping on muscle actions; defaults to env metadata.
        max_episode_steps: Optional truncation horizon per world.
        excitation_stages: Optional portable :class:`ExcitationStage` instances.
        obs_terms: Sequence of ``fn(accessor)`` callables concatenated to build ``obs``.
        reward_fn: Optional ``fn(accessor) -> (num_envs,)`` reward callable.
        terminated_fn: Optional ``fn(accessor) -> (num_envs,)`` bool callable.
        nconmax: Per-world contact capacity for ``SolverMuJoCo``.
        njmax: Per-world constraint row capacity for ``SolverMuJoCo``.
        seed: Optional seed for ``self.np_random``.
    """

    def __init__(
        self,
        env_id_or_xml: str | Path,
        num_envs: int = 1,
        *,
        device: str = "cuda:0",
        frame_skip: int | None = None,
        normalize_act: int | bool | None = None,
        max_episode_steps: int | None = None,
        excitation_stages: Sequence[ExcitationStage] = (),
        action_cfg: Any | None = None,
        obs_terms: Sequence[Callable[[EnvAccessor], Any]] | None = None,
        reward_fn: Callable[[EnvAccessor], Any] | None = None,
        terminated_fn: Callable[[EnvAccessor], Any] | None = None,
        nconmax: int = 2048,
        njmax: int = 8192,
        seed: int | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}.")
        newton, wp = _require_newton()
        self._newton = newton
        self._wp = wp

        xml_path, meta = resolve_env_xml(env_id_or_xml)
        self.xml_path = xml_path
        self.num_envs = int(num_envs)
        self.device = device
        self.reference_model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.frame_skip = int(
            meta.get("frame_skip", 1) if frame_skip is None else frame_skip
        )
        self.normalize_act = bool(
            meta.get("normalize_act", True) if normalize_act is None else normalize_act
        )
        self.max_episode_steps = (
            meta.get("max_episode_steps")
            if max_episode_steps is None
            else max_episode_steps
        )
        self.sim_dt = float(self.reference_model.opt.timestep)
        self.ctrl_dt = self.sim_dt * self.frame_skip
        self.np_random = np.random.default_rng(seed)

        template = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(template)
        template.default_shape_cfg.gap = 0.0
        template.add_mjcf(str(xml_path), ctrl_direct=True)
        if self.num_envs == 1:
            builder = template
        else:
            builder = newton.ModelBuilder()
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
            builder.replicate(template, copy_count=self.num_envs)
        self.model = builder.finalize(device=device)

        enable_multiccd = not bool(
            self.reference_model.opt.disableflags
            & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        )
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=False,
            use_mujoco_contacts=True,
            nconmax=nconmax,
            njmax=njmax,
            strict_capacity=True,
            integrator=_integrator_name(self.reference_model),
            iterations=int(self.reference_model.opt.iterations),
            ls_iterations=int(self.reference_model.opt.ls_iterations),
            enable_multiccd=enable_multiccd,
            kinematic_switchable_roots=False,
        )
        self.mapping = IndexMapping.build(self.reference_model, self.model, self.solver)
        self.state_in = self.model.state()
        self.state_out = self.model.state()
        self.control = self.model.control()
        self._muscle_ind = muscle_mask(self.reference_model)
        self._stages = list(excitation_stages)
        self._action_cfg_init = action_cfg
        self._seed_init = seed
        self.accessor = NewtonAccessor(self)
        self._torch_accessor: Any | None = None
        self._action_pipeline: Any | None = None
        self._conv_to_src_ctrl_torch: Any | None = None

        if obs_terms is None:
            default_terms: list[Callable[[EnvAccessor], Any]] = [
                joint_pos_obs,
                joint_vel_obs,
            ]
            if self.reference_model.na > 0:
                default_terms.append(muscle_act_obs)
            self._obs_terms = tuple(default_terms)
        else:
            self._obs_terms = tuple(obs_terms)
        self._reward_fn = reward_fn
        self._terminated_fn = terminated_fn
        self.step_count = np.zeros(self.num_envs, dtype=np.int32)
        self._refresh_derived()

    @property
    def nu(self) -> int:
        return int(self.reference_model.nu)

    @property
    def nq(self) -> int:
        return int(self.reference_model.nq)

    @property
    def nv(self) -> int:
        return int(self.reference_model.nv)

    @property
    def na(self) -> int:
        return int(self.reference_model.na)

    def _refresh_derived(self) -> None:
        mjw = self.solver._mujoco_warp
        with (
            self._wp.ScopedDevice(self.model.device),
            self.solver._scoped_mujoco_warp_execution(),
        ):
            mjw.fwd_position(
                self.solver.mjw_model, self.solver.mjw_data, factorize=False
            )
            mjw.fwd_velocity(self.solver.mjw_model, self.solver.mjw_data)
            mjw.fwd_actuation(self.solver.mjw_model, self.solver.mjw_data)

    def _compute_obs(self) -> np.ndarray:
        parts = []
        for term in self._obs_terms:
            val = np.asarray(term(self.accessor), dtype=np.float32)
            parts.append(val.reshape(self.num_envs, -1))
        return (
            np.concatenate(parts, axis=-1)
            if parts
            else np.zeros((self.num_envs, 0), dtype=np.float32)
        )

    def _action_to_ctrl(self, actions: np.ndarray) -> np.ndarray:
        ctrl = np.asarray(actions, dtype=np.float64)
        if ctrl.shape == (self.nu,) and self.num_envs == 1:
            ctrl = ctrl[None, :]
        if ctrl.shape != (self.num_envs, self.nu):
            raise ValueError(
                f"Expected actions of shape ({self.num_envs}, {self.nu}), got {ctrl.shape}."
            )
        ctrl = ctrl.copy()
        if self.normalize_act and np.any(self._muscle_ind):
            ctrl[:, self._muscle_ind] = sigmoid_muscle_activation(
                ctrl[:, self._muscle_ind], np
            )
        if self._stages and np.any(self._muscle_ind):
            u = ctrl[:, self._muscle_ind]
            for stage in self._stages:
                u = stage(u, np)
            ctrl[:, self._muscle_ind] = u
        lo = self.reference_model.actuator_ctrlrange[:, 0]
        hi = self.reference_model.actuator_ctrlrange[:, 1]
        limited = self.reference_model.actuator_ctrllimited.astype(bool)
        if np.any(limited):
            ctrl[:, limited] = np.clip(ctrl[:, limited], lo[limited], hi[limited])
        return ctrl[:, self.mapping.conv_to_src_ctrl].astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        env_ids: Sequence[int] | np.ndarray | None = None,
        qpos: np.ndarray | None = None,
        qvel: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset all or selected worlds and return ``(obs, info)``."""
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        if env_ids is None:
            mask_np = np.ones(self.num_envs, dtype=bool)
            world_mask = None
        else:
            arr = np.asarray(env_ids)
            if arr.dtype == bool:
                mask_np = arr.reshape(self.num_envs)
            else:
                mask_np = np.zeros(self.num_envs, dtype=bool)
                mask_np[arr] = True
            padded = np.zeros(self.num_envs + 1, dtype=bool)
            padded[: self.num_envs] = mask_np
            world_mask = self._wp.array(
                padded, dtype=self._wp.bool, device=self.model.device
            )

        self.solver.reset(self.state_in, world_mask=world_mask)
        self.solver._update_mjc_data(
            self.solver.mjw_data, self.model, self.state_in, world_mask=world_mask
        )

        selected = np.flatnonzero(mask_np)
        if qpos is not None or qvel is not None:
            mjw_qpos = self.solver.mjw_data.qpos.numpy()
            mjw_qvel = self.solver.mjw_data.qvel.numpy()
            if qpos is not None:
                qp = np.asarray(qpos, dtype=np.float32)
                if qp.ndim == 1:
                    qp = np.broadcast_to(qp, (len(selected), self.nq))
                mjw_qpos[selected] = qp[:, self.mapping.conv_to_src_qpos]
                self.solver.mjw_data.qpos.assign(mjw_qpos)
            if qvel is not None:
                qv = np.asarray(qvel, dtype=np.float32)
                if qv.ndim == 1:
                    qv = np.broadcast_to(qv, (len(selected), self.nv))
                mjw_qvel[selected] = qv[:, self.mapping.conv_to_src_qvel]
                self.solver.mjw_data.qvel.assign(mjw_qvel)
            self._refresh_derived()
            self.solver._update_newton_state(
                self.model,
                self.state_in,
                self.solver.mjw_data,
                state_prev=self.state_in,
            )

        times = self.solver.mjw_data.time.numpy()
        times[selected] = 0.0
        self.solver.mjw_data.time.assign(times)
        self.step_count[selected] = 0
        for stage in self._stages:
            stage.reset(None if env_ids is None else selected)
        if self._action_pipeline is not None:
            if seed is not None:
                self._action_pipeline.seed(seed)
            self._action_pipeline.reset(None if env_ids is None else selected)

        self._refresh_derived()
        return self._compute_obs(), {
            "time": self.accessor.time(),
            "step_count": self.step_count.copy(),
        }

    @property
    def torch_accessor(self) -> Any:
        """Lazily constructed zero-copy PyTorch :class:`TorchNewtonAccessor`."""
        if self._torch_accessor is None:
            from myosuite.integrations.newton.torch_pipeline import (  # noqa: PLC0415
                TorchNewtonAccessor,
            )

            self._torch_accessor = TorchNewtonAccessor(self)
        return self._torch_accessor

    @property
    def action_pipeline(self) -> Any:
        """Lazily constructed PyTorch :class:`NewtonActionPipeline`."""
        if self._action_pipeline is None:
            import torch  # noqa: PLC0415

            from myosuite.integrations.newton.torch_pipeline import (  # noqa: PLC0415
                NewtonActionCfg,
                NewtonActionPipeline,
            )

            cfg = self._action_cfg_init
            if cfg is None:
                cfg = NewtonActionCfg(
                    normalize_act=self.normalize_act,
                    excitation_stages=tuple((lambda s=st: s) for st in self._stages),
                )
            t_acc = self.torch_accessor
            self._action_pipeline = NewtonActionPipeline(
                self.reference_model,
                cfg,
                num_envs=self.num_envs,
                ctrl_dt=self.ctrl_dt,
                device=t_acc.device,
                seed=self._seed_init,
            )
            self._conv_to_src_ctrl_torch = torch.as_tensor(
                self.mapping.conv_to_src_ctrl,
                dtype=torch.long,
                device=t_acc.device,
            )
        return self._action_pipeline

    def set_site_pos(
        self,
        site_id_or_name: int | str,
        pos: Any,
        env_ids: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        """Update authored site local position across all or selected worlds."""
        sid = (
            self.accessor.site_id(site_id_or_name)
            if isinstance(site_id_or_name, str)
            else int(site_id_or_name)
        )
        conv_sid = int(self.mapping.src_to_conv_site[sid])
        site_pos_np = self.solver.mjw_model.site_pos.numpy().copy()
        pos_arr = np.asarray(pos, dtype=np.float32)
        if site_pos_np.shape[0] == 1 and self.num_envs > 1:
            if pos_arr.ndim == 2:
                pos_arr = pos_arr[0]
            site_pos_np[0, conv_sid] = pos_arr
        elif env_ids is None:
            site_pos_np[:, conv_sid] = np.broadcast_to(pos_arr, (self.num_envs, 3))
        else:
            sel = np.asarray(env_ids)
            site_pos_np[sel, conv_sid] = pos_arr
        self.solver.mjw_model.site_pos.assign(site_pos_np)
        self._refresh_derived()

    def set_mocap_pose(
        self,
        mocap_id: int,
        pos: Any | None = None,
        quat: Any | None = None,
        env_ids: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        """Update mocap body world pose (``wxyz`` quaternion) across worlds."""
        if int(self.solver.mj_model.nmocap) <= int(mocap_id):
            raise ValueError(
                f"mocap_id={mocap_id} out of range for nmocap={self.solver.mj_model.nmocap}."
            )
        sel = slice(None) if env_ids is None else np.asarray(env_ids)
        if pos is not None:
            mpos = self.solver.mjw_data.mocap_pos.numpy().copy()
            mpos[sel, int(mocap_id)] = np.asarray(pos, dtype=np.float32)
            self.solver.mjw_data.mocap_pos.assign(mpos)
        if quat is not None:
            mquat = self.solver.mjw_data.mocap_quat.numpy().copy()
            mquat[sel, int(mocap_id)] = np.asarray(quat, dtype=np.float32)
            self.solver.mjw_data.mocap_quat.assign(mquat)
        self._refresh_derived()

    def _compute_obs_torch(self) -> Any:
        import torch  # noqa: PLC0415

        t_acc = self.torch_accessor
        parts = [term(t_acc).reshape(self.num_envs, -1) for term in self._obs_terms]
        if not parts:
            return torch.zeros(
                (self.num_envs, 0), dtype=torch.float32, device=t_acc.device
            )
        return torch.cat(parts, dim=-1).to(dtype=torch.float32)

    def step_torch(self, actions: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """Advance ``frame_skip`` substeps using zero-copy PyTorch tensors on device."""
        import torch  # noqa: PLC0415

        pipeline = self.action_pipeline
        t_acc = self.torch_accessor
        pipeline.write_to_control(actions, self.control, self._conv_to_src_ctrl_torch)
        for _ in range(self.frame_skip):
            self.solver.step(
                self.state_in, self.state_out, self.control, None, self.sim_dt
            )
            self.state_in, self.state_out = self.state_out, self.state_in
        self._refresh_derived()
        self.step_count += 1

        obs = self._compute_obs_torch()
        reward = (
            self._reward_fn(t_acc).reshape(self.num_envs).to(dtype=torch.float32)
            if self._reward_fn is not None
            else torch.zeros(self.num_envs, dtype=torch.float32, device=t_acc.device)
        )
        terminated = (
            self._terminated_fn(t_acc).reshape(self.num_envs).to(dtype=torch.bool)
            if self._terminated_fn is not None
            else torch.zeros(self.num_envs, dtype=torch.bool, device=t_acc.device)
        )
        truncated = torch.as_tensor(
            self.step_count >= int(self.max_episode_steps)
            if self.max_episode_steps is not None
            else np.zeros(self.num_envs, dtype=bool),
            dtype=torch.bool,
            device=t_acc.device,
        )
        info = {"time": t_acc.time(), "step_count": self.step_count.copy()}
        return obs, reward, terminated, truncated, info

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Advance ``frame_skip`` substeps and return ``(obs, reward, terminated, truncated, info)``."""
        conv_ctrl = self._action_to_ctrl(actions)
        self.control.mujoco.ctrl.assign(conv_ctrl.reshape(-1))
        for _ in range(self.frame_skip):
            self.solver.step(
                self.state_in, self.state_out, self.control, None, self.sim_dt
            )
            self.state_in, self.state_out = self.state_out, self.state_in
        self._refresh_derived()
        self.step_count += 1

        obs = self._compute_obs()
        reward = (
            np.asarray(self._reward_fn(self.accessor), dtype=np.float32).reshape(
                self.num_envs
            )
            if self._reward_fn is not None
            else np.zeros(self.num_envs, dtype=np.float32)
        )
        terminated = (
            np.asarray(self._terminated_fn(self.accessor), dtype=bool).reshape(
                self.num_envs
            )
            if self._terminated_fn is not None
            else np.zeros(self.num_envs, dtype=bool)
        )
        truncated = (
            self.step_count >= int(self.max_episode_steps)
            if self.max_episode_steps is not None
            else np.zeros(self.num_envs, dtype=bool)
        )
        info = {"time": self.accessor.time(), "step_count": self.step_count.copy()}
        return obs, reward, terminated, truncated, info
