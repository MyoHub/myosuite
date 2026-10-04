# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MjlabEnvAccessor — EnvAccessor implementation for the mjlab/MuJoCo Warp backend.

mjlab (MuJoCo Warp) uses ``torch.Tensor`` for all physics state arrays and runs
in parallel across ``N`` environments on the GPU.  This accessor wraps the mjlab
data object and exposes the ``EnvAccessor`` protocol so that all shared term
functions in ``myosuite/terms/`` can run on this backend without modification.

Expected mjlab data interface (analogous to ``mjx.Data``)::

    data.qpos       # torch.Tensor (N, nq)
    data.qvel       # torch.Tensor (N, nv)
    data.act        # torch.Tensor (N, na)  — muscle activation state
    data.site_xpos  # torch.Tensor (N, nsite, 3)
    data.time       # torch.Tensor (N,)

Expected mjlab model interface::

    model.actuator_ctrlrange  # array-like (nu, 2)
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from myosuite.core.protocols import EnvAccessor, PhysicsPath

if TYPE_CHECKING:
    import torch


def _to_torch(value: np.ndarray, device: Any) -> torch.Tensor:
    """Static model array as a tensor: integers as long (indices), floats as float32."""
    import torch

    dtype = torch.long if np.issubdtype(value.dtype, np.integer) else torch.float32
    return torch.as_tensor(value, dtype=dtype, device=device)


class MjlabEnvAccessor(EnvAccessor):
    """EnvAccessor wrapping mjlab/MuJoCo Warp physics data as ``torch.Tensor``.

    This class satisfies the :class:`~myosuite.core.protocols.EnvAccessor`
    protocol so that all shared term functions in ``myosuite/terms/`` can run
    unmodified on the mjlab (MuJoCo Warp) backend.

    Args:
        model: mjlab model object exposing ``actuator_ctrlrange``.
        data: mjlab data object with ``qpos``, ``qvel``, ``act``,
            ``site_xpos``, and ``time`` as batched ``torch.Tensor`` fields.
        ctrl_dt: Control timestep in seconds (= ``sim_dt × n_substeps``).

    Example:
        >>> accessor = MjlabEnvAccessor(model, data, ctrl_dt=0.02)
        >>> dist = accessor.joint_pos() - target
    """

    def __init__(self, model: Any, data: Any, ctrl_dt: float) -> None:
        self._model = model
        self._data = data
        self._ctrl_dt = ctrl_dt

    # ------------------------------------------------------------------
    # EnvAccessor protocol
    # ------------------------------------------------------------------

    @property
    def physics_path(self) -> PhysicsPath:
        """Returns ``PhysicsPath.MJLAB``."""
        return PhysicsPath.MJLAB

    def joint_pos(self) -> torch.Tensor:
        """Joint positions, shape ``(N, nq)``."""
        return self._data.qpos

    def joint_vel(self) -> torch.Tensor:
        """Joint velocities, shape ``(N, nv)``."""
        return self._data.qvel

    def muscle_act(self) -> torch.Tensor:
        """Muscle activation state, shape ``(N, na)``."""
        return self._data.act

    def site_xpos(self, site_ids: Any) -> torch.Tensor:
        """Cartesian positions of requested sites, shape ``(N, len, 3)``."""
        return self._data.site_xpos[:, site_ids, :]

    def site_id(self, name: str) -> int:
        """Return the integer id for a site name."""
        if hasattr(self._model, "site"):
            site = self._model.site(name)
            if hasattr(site, "id"):
                return int(site.id)
        raise ValueError(f"Site {name!r} not found in model")

    def time(self) -> torch.Tensor:
        """Simulation time per environment, shape ``(N,)``."""
        return self._data.time

    def ctrl_range(self) -> torch.Tensor:
        """Actuator control range, shape ``(nu, 2)``."""
        import torch

        return torch.as_tensor(self._model.actuator_ctrlrange, dtype=torch.float32)

    def joint_range(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(qpos_ids, ranges)`` of the limited hinge/slide joints."""
        from myosuite.physics.joint_limits import joint_range_from_model

        qpos_ids, ranges = joint_range_from_model(self._model)
        return _to_torch(qpos_ids, self._data.qpos.device), _to_torch(
            ranges, self._data.qpos.device
        )

    def qfrc_actuator(self) -> torch.Tensor:
        """Actuator forces in joint space, shape ``(N, nv)``."""
        return self._data.qfrc_actuator

    def _muscle_ids(self) -> torch.Tensor:
        from myosuite.physics.muscle import muscle_columns

        return _to_torch(muscle_columns(self._model), self._data.qpos.device)

    def muscle_force(self) -> torch.Tensor:
        """Muscle forces (N, tension < 0), shape ``(N, n_muscles)``."""
        return self._data.actuator_force[:, self._muscle_ids()]

    def muscle_length(self) -> torch.Tensor:
        """Muscle-tendon unit lengths (m), shape ``(N, n_muscles)``."""
        return self._data.actuator_length[:, self._muscle_ids()]

    def muscle_velocity(self) -> torch.Tensor:
        """Muscle-tendon unit velocities (m/s, + = lengthening), ``(N, n_muscles)``."""
        return self._data.actuator_velocity[:, self._muscle_ids()]

    def muscle_params(self) -> Any:
        """Static :class:`~myosuite.physics.muscle.MuscleParams` as torch tensors."""
        from myosuite.physics.muscle import muscle_params_from_model

        device = self._data.qpos.device
        return muscle_params_from_model(self._model).map(
            lambda _, v: _to_torch(v, device)
        )

    def dt(self) -> float:
        """Control timestep in seconds."""
        return self._ctrl_dt

    def array_module(self) -> Any:
        """Returns the ``torch`` module for use in term functions.

        Term functions call ``xp = accessor.array_module()`` then use
        ``xp`` for all array operations, ensuring backend-agnostic code.

        Returns:
            The ``torch`` module.
        """
        import torch

        return torch


class MjlabEntityAccessor(EnvAccessor):
    """EnvAccessor over one mjlab scene entity, laid out like the CPU ``MjData``.

    ``joint_pos`` / ``joint_vel`` follow CPU ``qpos`` / ``qvel`` layout: a free
    root contributes ``[pos (relative to the env origin), quat]`` and
    ``[lin vel (world), ang vel (body)]`` ahead of the remaining joints, so
    shared term functions see the same vectors on both backends.

    Args:
        env: The mjlab environment.
        entity_name: Scene entity key.
    """

    def __init__(self, env: Any, entity_name: str) -> None:
        self._env = env
        self._entity = env.scene[entity_name]

    @property
    def physics_path(self) -> PhysicsPath:
        """Returns ``PhysicsPath.MJLAB``."""
        return PhysicsPath.MJLAB

    def joint_pos(self) -> torch.Tensor:
        """CPU-layout ``qpos``, shape ``(N, nq)``."""
        import torch

        data = self._entity.data
        if self._entity.is_fixed_base:
            return data.joint_pos
        root_pos = data.root_link_pos_w - self._env.scene.env_origins
        return torch.cat([root_pos, data.root_link_quat_w, data.joint_pos], dim=-1)

    def joint_vel(self) -> torch.Tensor:
        """CPU-layout ``qvel``, shape ``(N, nv)``."""
        import torch

        if self._entity.is_fixed_base:
            return self._entity.data.joint_vel
        # ``qvel`` itself (like the CPU env): the root link velocities of the entity data
        # derive from ``cvel``, which is only refreshed by ``forward()``.
        qvel, index = self._env.sim.data.qvel, self._entity.indexing
        return torch.cat(
            [
                qvel[:, index.free_joint_v_adr.long()],
                qvel[:, index.joint_v_adr.long()],
            ],
            dim=-1,
        )

    def muscle_act(self) -> torch.Tensor:
        """Muscle activation state, shape ``(N, na)``."""
        # accepted: no entity.data API for muscle activation — entity.data.data.act
        return self._entity.data.data.act

    def site_xpos(self, site_ids: Any) -> torch.Tensor:
        """Entity-local site positions in the CPU world frame, ``(N, k, 3)``.

        Floating-base entities are spawned at their env origin, so the origin
        is removed; fixed-base entities are never offset.
        """
        pos = self._entity.data.site_pos_w[:, site_ids, :]
        if self._entity.is_fixed_base:
            return pos
        return pos - self._env.scene.env_origins[:, None, :]

    def site_id(self, name: str) -> int:
        """Entity-local id of site *name*."""
        ids, _ = self._entity.find_sites((f"^{name}$",), preserve_order=True)
        return int(ids[0])

    def time(self) -> torch.Tensor:
        """Time since episode start, shape ``(N,)``."""
        return self._env.episode_length_buf * self._env.step_dt

    def ctrl_range(self) -> torch.Tensor:
        """Control range of the entity actuators, shape ``(nu, 2)``."""
        import torch

        ids = self._entity.indexing.ctrl_ids.cpu().numpy()
        return torch.as_tensor(
            self._env.sim.mj_model.actuator_ctrlrange[ids],
            dtype=torch.float32,
            device=self._env.device,
        )

    def joint_range(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(qpos_ids, ranges)`` of the limited hinge/slide joints.

        ``qpos_ids`` index the CPU-layout :meth:`joint_pos` (after the free
        root, if any); ranges come from the nominal model.
        """
        from myosuite.physics.joint_limits import joint_range_from_model

        # Host-side ids from the entity spec (no device sync).
        joint_ids = [j.id for j in self._entity.indexing.joints]
        qpos_ids, ranges = joint_range_from_model(
            self._env.sim.mj_model,
            np.asarray(joint_ids, dtype=np.int64),
            qpos_offset=0 if self._entity.is_fixed_base else 7,
        )
        return _to_torch(qpos_ids, self._env.device), _to_torch(
            ranges, self._env.device
        )

    def qfrc_actuator(self) -> torch.Tensor:
        """CPU-layout actuator forces in joint space, shape ``(N, nv)``."""
        import torch

        if self._entity.is_fixed_base:
            return self._entity.data.qfrc_actuator
        # The free root's dofs, as in joint_vel(): entity.data covers the joints only.
        qfrc, index = self._env.sim.data.qfrc_actuator, self._entity.indexing
        return torch.cat(
            [qfrc[:, index.free_joint_v_adr.long()], self._entity.data.qfrc_actuator],
            dim=-1,
        )

    @cached_property
    def _actuator_ids(self) -> np.ndarray:
        """Global ids of the entity actuators, host-side (no device sync)."""
        actuators = self._entity.indexing.actuators or ()
        return np.asarray([a.id for a in actuators], dtype=np.int64)

    @cached_property
    def _muscle_cols(self) -> torch.Tensor:
        """Entity actuator columns of the muscles."""
        from myosuite.physics.muscle import muscle_columns

        cols = muscle_columns(self._env.sim.mj_model, self._actuator_ids)
        return _to_torch(cols, self._env.device)

    @cached_property
    def _muscle_actuator_ids(self) -> torch.Tensor:
        """Global actuator ids of the muscles."""
        from myosuite.physics.muscle import muscle_columns

        ids = self._actuator_ids
        return _to_torch(
            ids[muscle_columns(self._env.sim.mj_model, ids)], self._env.device
        )

    def muscle_force(self) -> torch.Tensor:
        """Muscle forces (N, tension < 0), shape ``(N, n_muscles)``."""
        return self._entity.data.actuator_force[:, self._muscle_cols]

    def muscle_length(self) -> torch.Tensor:
        """Muscle-tendon unit lengths (m), shape ``(N, n_muscles)``."""
        # accepted: no entity.data API for actuator length — entity.data.data.actuator_length
        return self._entity.data.data.actuator_length[:, self._muscle_actuator_ids]

    def muscle_velocity(self) -> torch.Tensor:
        """Muscle-tendon unit velocities (m/s, + = lengthening), ``(N, n_muscles)``."""
        # accepted: no entity.data API for actuator velocity — entity.data.data.actuator_velocity
        return self._entity.data.data.actuator_velocity[:, self._muscle_actuator_ids]

    @cached_property
    def _muscle_params(self) -> Any:
        from myosuite.physics.muscle import muscle_params_from_model

        params = muscle_params_from_model(self._env.sim.mj_model, self._actuator_ids)
        return params.map(lambda _, v: _to_torch(v, self._env.device))

    def muscle_params(self) -> Any:
        """Static :class:`~myosuite.physics.muscle.MuscleParams` (nominal model).

        Cached on the accessor: hold one accessor (e.g. in a ``ManagerTermBase``)
        to build the tensors once.
        """
        return self._muscle_params

    def dt(self) -> float:
        """Control timestep in seconds."""
        return self._env.step_dt

    def array_module(self) -> Any:
        """Returns the ``torch`` module."""
        import torch

        return torch


try:
    from mjlab.envs.mdp.events import resolve_env_ids as normalize_mjlab_env_ids
except ImportError:
    import torch as _torch

    def normalize_mjlab_env_ids(env: Any, env_ids: Any) -> _torch.Tensor:  # type: ignore[misc]
        """Normalise env_ids to a 1-D long tensor (mjlab < 1.4 fallback)."""
        if env_ids is None:
            return _torch.arange(env.num_envs, device=env.device, dtype=_torch.long)
        if isinstance(env_ids, slice):
            return _torch.arange(env.num_envs, device=env.device, dtype=_torch.long)[
                env_ids
            ]
        return _torch.as_tensor(env_ids, device=env.device, dtype=_torch.long).reshape(
            -1
        )
