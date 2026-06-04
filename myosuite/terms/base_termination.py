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

if TYPE_CHECKING:
    from myosuite.core.protocols import EnvAccessor


def _xp_asarray(xp: Any, value: Any, *, dtype: Any | None = None) -> Any:
    if getattr(xp, "__name__", "") == "torch":
        tensor = xp.as_tensor(value)
        return tensor.to(dtype=dtype) if dtype is not None else tensor
    return xp.asarray(value, dtype=dtype) if dtype is not None else xp.asarray(value)


def _maybe_item(value: Any) -> Any:
    ndim = getattr(value, "ndim", None)
    if ndim == 0 and hasattr(value, "item"):
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
    """Terminate when any joint exceeds its control range.

    Args:
        accessor: Environment state accessor.
        task_state: Unused; present for uniform call signature.
        margin: Additional margin (radians) beyond the nominal range that
            triggers termination. Default is 0 (hard limit).
        **kwargs: Unused extra keyword arguments.

    Returns:
        Boolean scalar — True if any joint is outside its limits.
    """
    xp = accessor.array_module()
    ctrl_range = accessor.ctrl_range()
    qpos = accessor.joint_pos()
    lo = ctrl_range[:, 0] - margin
    hi = ctrl_range[:, 1] + margin
    # Use bitwise | instead of Python `or` so this expression is traceable by JAX JIT.
    return xp.any(qpos < lo) | xp.any(qpos > hi)


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
    )
    return _maybe_item(posture < float(upright_posture_threshold))


def saber_pool_done(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    max_target_misses: int | None = None,
    use_health_status: bool = True,
    **kwargs: Any,
) -> Any:
    """Terminate when the saber pool reaches its miss or health limit."""

    del kwargs
    xp = accessor.array_module()
    bool_dtype = getattr(xp, "bool", bool)
    misses = _xp_asarray(
        xp,
        task_state.get("target_pool_misses", 0),
        dtype=getattr(xp, "float32", None),
    )
    miss_limit = (
        misses >= float(max_target_misses)
        if max_target_misses is not None
        else xp.zeros_like(misses, dtype=bool_dtype)
    )
    if use_health_status:
        health_status = _xp_asarray(
            xp,
            task_state.get("health_status", 0.5),
            dtype=getattr(xp, "float32", None),
        )
        health_done = health_status <= 0.0
    else:
        health_done = xp.zeros_like(miss_limit, dtype=bool_dtype)
    return _maybe_item(miss_limit | health_done)
