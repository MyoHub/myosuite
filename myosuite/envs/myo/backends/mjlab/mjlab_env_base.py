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
        self._ctrl_range: torch.Tensor | None = None

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

        device = getattr(getattr(self._data, "qpos", None), "device", None)
        if self._ctrl_range is None or (
            device is not None and self._ctrl_range.device != device
        ):
            self._ctrl_range = torch.as_tensor(
                self._model.actuator_ctrlrange,
                dtype=torch.float32,
                device=device,
            )
        return self._ctrl_range

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
        self._ctrl_range: torch.Tensor | None = None
        self._act_ids: torch.Tensor | None = None
        self._act_ids_resolved: bool = False

    def warm_cache(self) -> None:
        """Resolve per-entity activation indices and ``ctrl_range`` during setup."""
        data_holder = getattr(self._entity, "data", None)
        raw_data = getattr(data_holder, "data", None)
        act = getattr(raw_data, "act", None)
        device = act.device if act is not None else getattr(self._env, "device", None)
        if device is not None:
            self._resolve_act_ids(device)
        mj_model = getattr(getattr(self._env, "sim", None), "mj_model", None)
        ctrl_ids = getattr(getattr(self._entity, "indexing", None), "ctrl_ids", None)
        if mj_model is not None and ctrl_ids is not None and device is not None:
            self._resolve_ctrl_range(device)

    def _resolve_act_ids(self, device: Any) -> torch.Tensor | None:
        cache: dict[Any, torch.Tensor | None] | None = getattr(
            self._entity, "_myosuite_act_ids_by_device", None
        )
        if cache is not None and device in cache:
            return cache[device]
        if self._act_ids_resolved:
            return self._act_ids

        act_ids: torch.Tensor | None = None
        mj_model = getattr(getattr(self._env, "sim", None), "mj_model", None)
        ctrl_ids = getattr(getattr(self._entity, "indexing", None), "ctrl_ids", None)
        if mj_model is not None and ctrl_ids is not None and int(mj_model.na) > 0:
            import torch

            ids_np = ctrl_ids.detach().cpu().numpy()
            adrs = mj_model.actuator_actadr[ids_np]
            nums = mj_model.actuator_actnum[ids_np]
            cols: list[int] = []
            for adr, num in zip(adrs.tolist(), nums.tolist(), strict=False):
                if adr >= 0 and num > 0:
                    cols.extend(range(int(adr), int(adr) + int(num)))
            if cols != list(range(int(mj_model.na))):
                act_ids = torch.tensor(cols, dtype=torch.long, device=device)
            elif len(cols) == 0:
                act_ids = torch.empty(0, dtype=torch.long, device=device)

        self._act_ids_resolved = True
        self._act_ids = act_ids
        if cache is None:
            cache = {}
            try:
                setattr(self._entity, "_myosuite_act_ids_by_device", cache)
            except AttributeError:
                return act_ids
        cache[device] = act_ids
        return act_ids

    def _resolve_ctrl_range(self, device: Any) -> torch.Tensor:
        import torch

        cache: dict[Any, torch.Tensor] | None = getattr(
            self._entity, "_myosuite_ctrl_range_by_device", None
        )
        if cache is not None and device in cache:
            return cache[device]
        if self._ctrl_range is not None and self._ctrl_range.device == device:
            return self._ctrl_range

        ids = self._entity.indexing.ctrl_ids.detach().cpu().numpy()
        ctrl_range = torch.as_tensor(
            self._env.sim.mj_model.actuator_ctrlrange[ids],
            dtype=torch.float32,
            device=device,
        )
        self._ctrl_range = ctrl_range
        if cache is None:
            cache = {}
            try:
                setattr(self._entity, "_myosuite_ctrl_range_by_device", cache)
            except AttributeError:
                return ctrl_range
        cache[device] = ctrl_range
        return ctrl_range

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
        act = self._entity.data.data.act
        cache: dict[Any, torch.Tensor | None] | None = getattr(
            self._entity, "_myosuite_act_ids_by_device", None
        )
        if cache is not None and act.device in cache:
            act_ids = cache[act.device]
        else:
            act_ids = self._resolve_act_ids(act.device)
        if act_ids is not None:
            return act[:, act_ids]
        return act

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
        cache: dict[Any, torch.Tensor] | None = getattr(
            self._entity, "_myosuite_ctrl_range_by_device", None
        )
        if cache is not None and self._env.device in cache:
            return cache[self._env.device]
        return self._resolve_ctrl_range(self._env.device)

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
