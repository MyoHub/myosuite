# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""
Termination condition functions for MyoSuite environments.

Each function returns a boolean scalar (or array for batched envs)
indicating whether the episode should terminate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from myosuite.terms._array import xp_asarray as _xp_asarray

if TYPE_CHECKING:
    from myosuite.core.protocols import EnvAccessor


def _maybe_item(value: Any) -> Any:
    if isinstance(value, np.ndarray | np.generic) and np.ndim(value) == 0:
        return value.item()
    return value


def pelvis_fall_termination(
    pelvis_pos: np.ndarray,
    fall_pelvis_z_threshold: float,
) -> bool:
    """Return whether pelvis height is below the fall threshold.

    Args:
        pelvis_pos: Pelvis world position, shape ``(3,)``.
        fall_pelvis_z_threshold: Minimum allowed pelvis Z position.

    Returns:
        ``True`` when the agent is considered fallen.
    """
    return float(pelvis_pos[2]) < float(fall_pelvis_z_threshold)


def joint_limit_violation(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    margin: float = 0.0,
    **kwargs: Any,
) -> Any:
    """Terminate when any limited 1-DoF joint exceeds its range of motion.

    Args:
        accessor: Environment state accessor.
        task_state: Unused; present for uniform call signature.
        margin: Additional margin (radians) beyond the nominal range that
            triggers termination. Default is 0 (hard limit).
        **kwargs: Unused extra keyword arguments.

    Returns:
        Boolean scalar (unbatched) or shape-``(N,)`` boolean array (batched).
    """
    xp = accessor.array_module()
    qpos = accessor.joint_pos()
    lo, hi, limited = accessor.joint_limits()
    violated = xp.logical_and(
        limited,
        xp.logical_or(qpos < lo - margin, qpos > hi + margin),
    )
    return _maybe_item(xp.any(violated, axis=-1))


def fall_termination(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    height_site: str = "root",
    min_height: float = 0.5,
    **kwargs: Any,
) -> Any:
    """Terminate when the agent falls below a minimum height.

    Args:
        accessor: Environment state accessor.
        task_state: Must contain ``"height_site_id"`` if *height_site* is
            not supplied via kwargs.
        height_site: Name of the site used to measure height (Z axis).
        min_height: Minimum acceptable height in metres.
        **kwargs: Unused extra keyword arguments.

    Returns:
        Boolean scalar — True if the agent has fallen below *min_height*.
    """
    site_id = task_state.get("height_site_id")
    if site_id is None:
        return False
    pos = accessor.site_xpos(site_id)
    return pos[..., 2] < min_height


def time_limit(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    max_time: float = 10.0,
    **kwargs: Any,
) -> Any:
    """Terminate when the simulation time exceeds a maximum.

    Args:
        accessor: Environment state accessor.
        task_state: Unused; present for uniform call signature.
        max_time: Maximum simulation time in seconds before truncation.
        **kwargs: Unused extra keyword arguments.

    Returns:
        Boolean scalar — True if simulation time has exceeded *max_time*.
    """
    return accessor.time() >= max_time


def upright_posture_failure(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    upright_posture_threshold: float = 0.9,
    **kwargs: Any,
) -> Any:
    """Terminate when the shared task-state upright score falls below threshold."""

    del kwargs
    xp = accessor.array_module()
    posture = _xp_asarray(
        xp,
        task_state.get("upright_posture", 0.0),
        dtype=getattr(xp, "float32", None),
        device=getattr(accessor.joint_pos(), "device", None),
    )
    return _maybe_item(posture < float(upright_posture_threshold))


def enclosure_violation(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    *,
    contact_threshold: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Terminate when any tracked site exits the enclosure or exceeds *contact_threshold*.

    Args:
        accessor: Environment state accessor.
        task_state: Dict with ``"enclosure_site_ids"`` (or ``"tip_site_ids"``),
            ``"bounds_low"``, ``"bounds_high"``, and optional ``"plane_normals"``,
            ``"plane_offsets"``, and ``"enclosure_contact_force"``.
        contact_threshold: Allowed contact force magnitude [N] before violation.
        **kwargs: Unused extra keyword arguments.

    Returns:
        Boolean scalar (unbatched CPU) or shape-``(N,)`` boolean array (batched).
    """
    from myosuite.physics.spatial_enclosure import (  # noqa: PLC0415
        excess_contact_force,
        min_enclosure_clearance,
    )

    del kwargs
    xp = accessor.array_module()
    batched = accessor.joint_pos().ndim > 1
    raw_ids = (
        task_state["enclosure_site_ids"]
        if "enclosure_site_ids" in task_state
        else task_state["tip_site_ids"]
    )
    ids = [int(raw_ids)] if isinstance(raw_ids, int | np.integer) else raw_ids
    pos = accessor.site_xpos(ids)
    min_clearance = min_enclosure_clearance(
        xp,
        pos,
        task_state["bounds_low"],
        task_state["bounds_high"],
        plane_normals=task_state.get("plane_normals"),
        plane_offsets=task_state.get("plane_offsets"),
    )
    excess_force = excess_contact_force(
        xp,
        task_state.get("enclosure_contact_force"),
        threshold=contact_threshold,
        reference=min_clearance,
        batched=batched,
    )
    return _maybe_item((min_clearance < 0.0) | (excess_force > 0.0))
