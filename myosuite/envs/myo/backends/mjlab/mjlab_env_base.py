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

from typing import TYPE_CHECKING, Any

from myosuite.core.protocols import EnvAccessor, PhysicsPath

if TYPE_CHECKING:
    import torch


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

        device = getattr(self._data.qpos, "device", None)
        return torch.as_tensor(
            self._model.actuator_ctrlrange, dtype=torch.float32, device=device
        )

    def joint_limits(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-qpos scalar joint limits ``(lo, hi, limited)``, each of shape ``(nq,)``."""
        import numpy as np
        import torch

        device = getattr(self._data.qpos, "device", None)
        nq = int(self._data.qpos.shape[-1])
        if not hasattr(self._model, "jnt_range") or not hasattr(
            self._model, "jnt_type"
        ):
            zeros = torch.zeros(nq, dtype=torch.float32, device=device)
            return zeros, zeros, torch.zeros(nq, dtype=torch.bool, device=device)

        import mujoco

        lo = np.zeros(nq, dtype=np.float32)
        hi = np.zeros(nq, dtype=np.float32)
        limited = np.zeros(nq, dtype=bool)
        scalar_types = (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        )
        for j in range(int(self._model.njnt)):
            if int(self._model.jnt_type[j]) in scalar_types and bool(
                self._model.jnt_limited[j]
            ):
                qadr = int(self._model.jnt_qposadr[j])
                if qadr < nq:
                    lo[qadr] = float(self._model.jnt_range[j, 0])
                    hi[qadr] = float(self._model.jnt_range[j, 1])
                    limited[qadr] = True
        return (
            torch.as_tensor(lo, dtype=torch.float32, device=device),
            torch.as_tensor(hi, dtype=torch.float32, device=device),
            torch.as_tensor(limited, dtype=torch.bool, device=device),
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
        self._joint_limits_cache: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None

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

    def joint_limits(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-qpos scalar joint limits ``(lo, hi, limited)`` matching :meth:`joint_pos`."""
        if self._joint_limits_cache is None:
            import mujoco
            import numpy as np
            import torch

            mj_model = self._env.sim.mj_model
            prefix = 0 if self._entity.is_fixed_base else 7
            joint_ids = self._entity.indexing.joint_ids.cpu().numpy()
            nq = prefix + int(self._entity.data.joint_pos.shape[-1])
            lo = np.zeros(nq, dtype=np.float32)
            hi = np.zeros(nq, dtype=np.float32)
            limited = np.zeros(nq, dtype=bool)
            scalar_types = (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            )
            offset = prefix
            for jid in joint_ids:
                j = int(jid)
                jtype = int(mj_model.jnt_type[j])
                width = (
                    7
                    if jtype == int(mujoco.mjtJoint.mjJNT_FREE)
                    else 4
                    if jtype == int(mujoco.mjtJoint.mjJNT_BALL)
                    else 1
                )
                if (
                    jtype in scalar_types
                    and bool(mj_model.jnt_limited[j])
                    and offset < nq
                ):
                    lo[offset] = float(mj_model.jnt_range[j, 0])
                    hi[offset] = float(mj_model.jnt_range[j, 1])
                    limited[offset] = True
                offset += width
            device = self._env.device
            self._joint_limits_cache = (
                torch.as_tensor(lo, dtype=torch.float32, device=device),
                torch.as_tensor(hi, dtype=torch.float32, device=device),
                torch.as_tensor(limited, dtype=torch.bool, device=device),
            )
        return self._joint_limits_cache

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

    def normalize_mjlab_env_ids(env: Any, env_ids: Any) -> torch.Tensor:  # type: ignore[misc]
        """Normalise env_ids to a 1-D long tensor (mjlab < 1.4 fallback)."""
        import torch as _torch

        if env_ids is None:
            return _torch.arange(env.num_envs, device=env.device, dtype=_torch.long)
        if isinstance(env_ids, slice):
            return _torch.arange(env.num_envs, device=env.device, dtype=_torch.long)[
                env_ids
            ]
        return _torch.as_tensor(env_ids, device=env.device, dtype=_torch.long).reshape(
            -1
        )
