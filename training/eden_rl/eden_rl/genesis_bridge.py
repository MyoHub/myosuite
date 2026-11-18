# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""User-space musculoskeletal actuation bridge for ``genesis-world`` and ``Eden``.

``genesis-world`` (``<=1.4.6``) rejects MJCF files with spatial ``<tendon>`` paths or
more than one actuator per joint, and lacks ``mjDYN_MUSCLE`` / ``mjGAIN_MUSCLE`` /
``mjBIAS_MUSCLE`` kernels. :class:`GenesisMuscleBridge` enables running any MyoSuite
skeleton (``hand``, ``arm``, ``leg``, ``myofullbody``) inside a Genesis rigid/soft/fluid
scene **today** without patching Genesis C++/Quadrants:

1. :meth:`GenesisMuscleBridge.export_genesis_mjcf` strips spatial tendons and Hill
   muscles via :func:`myosuite.physics.hill_muscle.strip_muscles_for_joint_torque_sim`
   while preserving all bodies, geoms, sites, and ``mjEQ_JOINT`` couplings.
2. Each physics substep, :meth:`GenesisMuscleBridge.step_actuation` maps policy actions
   through MyoSuite's canonical sigmoid, integrates Millard ``mjDYN_MUSCLE`` activation
   dynamics in PyTorch, evaluates muscle lengths/velocities/moment arms, computes exact
   ``mjGAIN_MUSCLE`` + ``mjBIAS_MUSCLE`` generalized torques ``tau = J^T @ f``, and
   applies them via ``entity.control_dofs_force(tau, ...)``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch

from myosuite.physics.hill_muscle import (
    HillMuscleModel,
    MuscleParams,
    dense_actuator_moment,
    step_muscle_activation,
    strip_muscles_for_joint_torque_sim,
)
from myosuite.terms.base_action import sigmoid_muscle_activation

if TYPE_CHECKING:
    import mujoco

GeometryCallback = Callable[
    [torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


class RigidEntityLike(Protocol):
    """Minimal ``genesis.engine.entities.RigidEntity`` protocol used by the bridge."""

    def get_dofs_position(self, dofs_idx_local: Any = None) -> torch.Tensor: ...
    def get_dofs_velocity(self, dofs_idx_local: Any = None) -> torch.Tensor: ...
    def control_dofs_force(
        self, force: torch.Tensor, dofs_idx_local: Any = None
    ) -> None: ...


class GenesisMuscleBridge:
    """Batched PyTorch Hill-type muscle controller driving a ``genesis-world`` ``RigidEntity``."""

    def __init__(
        self,
        reference_model: mujoco.MjModel | None = None,
        num_envs: int = 1,
        *,
        params: MuscleParams | None = None,
        nv: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        normalize_act: bool = True,
        dofs_idx_local: Any = None,
        geometry_fn: GeometryCallback | None = None,
    ) -> None:
        self.mj_model = reference_model
        self.mj_data: Any = None
        if reference_model is not None:
            import mujoco as _mujoco

            self.mj_data = _mujoco.MjData(reference_model)
            self.hill = HillMuscleModel(reference_model)
        elif params is not None and nv is not None:
            if geometry_fn is None:
                raise ValueError(
                    "geometry_fn is required when constructing GenesisMuscleBridge from params without an MjModel."
                )
            self.hill = HillMuscleModel(params=params, nv=nv)
        else:
            raise ValueError(
                "Pass either reference_model=<MjModel> or (params=<MuscleParams>, nv=<int>, geometry_fn=...)."
            )

        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.normalize_act = bool(normalize_act)
        self.dofs_idx_local = dofs_idx_local
        self.geometry_fn = geometry_fn

        self.nu = self.hill.nu
        self.nv = self.hill.nv
        self.act = torch.zeros(
            (self.num_envs, self.nu), device=self.device, dtype=self.dtype
        )

    @staticmethod
    def export_genesis_mjcf(
        spec: mujoco.MjSpec,
        output_xml: str | Path,
        *,
        add_joint_motors: bool = True,
    ) -> tuple[Path, mujoco.MjModel]:
        """Compile reference ``MjModel`` (with muscles), strip muscles, and write Genesis MJCF."""
        reference_model = spec.compile()
        stripped = strip_muscles_for_joint_torque_sim(
            spec, add_joint_motors=add_joint_motors
        )
        out_path = Path(output_xml)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(stripped.to_xml(), encoding="utf-8")
        return out_path, reference_model

    def reset(self, env_ids: torch.Tensor | None = None, init_act: float = 0.0) -> None:
        """Reset muscle activation state ``act`` for selected environments (or all)."""
        if env_ids is None:
            self.act.fill_(float(init_act))
        else:
            self.act[env_ids] = float(init_act)

    def _cpu_mujoco_geometry(
        self, qpos: torch.Tensor, qvel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate exact ``(actuator_length, actuator_velocity, actuator_moment)`` via MuJoCo FK."""
        import mujoco as _mujoco

        assert self.mj_model is not None and self.mj_data is not None
        q_np = qpos.detach().cpu().numpy().astype(np.float64, copy=False)
        qd_np = qvel.detach().cpu().numpy().astype(np.float64, copy=False)
        b = q_np.shape[0]
        lengths = np.empty((b, self.nu), dtype=np.float32)
        vels = np.empty((b, self.nu), dtype=np.float32)
        moments = np.empty((b, self.nu, self.nv), dtype=np.float32)

        for i in range(b):
            self.mj_data.qpos[:] = q_np[i]
            self.mj_data.qvel[:] = qd_np[i]
            _mujoco.mj_kinematics(self.mj_model, self.mj_data)
            _mujoco.mj_comPos(self.mj_model, self.mj_data)
            _mujoco.mj_tendon(self.mj_model, self.mj_data)
            _mujoco.mj_transmission(self.mj_model, self.mj_data)
            lengths[i] = self.mj_data.actuator_length
            vels[i] = self.mj_data.actuator_velocity
            moments[i] = dense_actuator_moment(self.mj_model, self.mj_data)

        return (
            torch.as_tensor(lengths, device=self.device, dtype=self.dtype),
            torch.as_tensor(vels, device=self.device, dtype=self.dtype),
            torch.as_tensor(moments, device=self.device, dtype=self.dtype),
        )

    def compute_torques(
        self,
        action: torch.Tensor,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        dt: float,
        *,
        substeps: int = 1,
    ) -> torch.Tensor:
        """Advance muscle activation ODE by ``dt`` and return ``(B, nv)`` generalized torques."""
        act_t = action.to(device=self.device, dtype=self.dtype)
        ctrl = sigmoid_muscle_activation(act_t, torch) if self.normalize_act else act_t
        next_act, _ = step_muscle_activation(
            self.act, ctrl, self.hill.params, float(dt), substeps=substeps
        )
        self.act = next_act

        geom = self.geometry_fn or self._cpu_mujoco_geometry
        length, velocity, moment = geom(qpos, qvel)
        return self.hill.generalized_forces(length, velocity, self.act, moment)

    def step_actuation(
        self,
        entity: RigidEntityLike,
        action: torch.Tensor,
        dt: float,
        *,
        qpos: torch.Tensor | None = None,
        qvel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Query DOFs from ``entity``, compute Hill muscle torques, and call ``control_dofs_force``."""
        q = qpos if qpos is not None else entity.get_dofs_position(self.dofs_idx_local)
        qd = qvel if qvel is not None else entity.get_dofs_velocity(self.dofs_idx_local)
        if q.ndim == 1:
            q = q.unsqueeze(0)
            qd = qd.unsqueeze(0)
        tau = self.compute_torques(action, q, qd, dt)
        entity.control_dofs_force(tau, self.dofs_idx_local)
        return tau
