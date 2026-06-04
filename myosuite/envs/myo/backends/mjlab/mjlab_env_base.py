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

        return torch.as_tensor(self._model.actuator_ctrlrange, dtype=torch.float32)

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
