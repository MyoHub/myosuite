# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""
EnvAccessor protocol and PhysicsPath enum shared across all three backend paths
(CPU/Gymnasium, MJX+Brax, MuJoCo Warp/mjlab).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable


class PhysicsPath(Enum):
    """Identifies which physics backend an environment runs on."""

    CPU = "cpu"
    MJX = "mjx"
    MJLAB = "mjlab"


@runtime_checkable
class EnvAccessor(Protocol):
    """Minimal read interface exposed to term functions.

    Returns the native array type for the current path:
    - CPU: numpy.ndarray
    - MJX: jax.Array
    - mjlab: torch.Tensor

    Term functions receive an EnvAccessor and call accessor.array_module
    for any array operations so the same code runs on all three paths.
    """

    @property
    def physics_path(self) -> PhysicsPath:
        """The backend physics path this accessor wraps."""
        ...

    def joint_pos(self) -> Any:
        """Joint positions. Shape: (nq,) or (N, nq) for batched envs."""
        ...

    def joint_vel(self) -> Any:
        """Joint velocities. Shape: (nv,) or (N, nv) for batched envs."""
        ...

    def muscle_act(self) -> Any:
        """Muscle activations. Shape: (na,) or (N, na) for batched envs."""
        ...

    def site_xpos(self, site_ids: Any) -> Any:
        """Cartesian positions of sites in world frame. Shape: (len(site_ids), 3)."""
        ...

    def time(self) -> Any:
        """Simulation time in seconds."""
        ...

    def ctrl_range(self) -> Any:
        """Actuator control range. Shape: (nu, 2) where [:,0] is min, [:,1] is max."""
        ...

    def dt(self) -> float:
        """Control timestep (ctrl_dt) in seconds."""
        ...

    def array_module(self) -> Any:
        """The array library for this backend: numpy, jax.numpy, or torch."""
        ...

    def site_id(self, name: str) -> int:
        """Return the integer index of a named site in the model.

        Args:
            name: Site name as defined in the MJCF XML.

        Returns:
            Integer site index usable with :meth:`site_xpos`.

        Raises:
            ValueError: If the site name is not found in the model.
        """
        ...
