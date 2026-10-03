# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""
Event / reset term functions for MyoSuite environments.

These functions are called at episode reset or at startup to randomise
the initial state, apply physiological conditions, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from myosuite.core.protocols import EnvAccessor


def reset_random_pose(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    target_jnt_range: np.ndarray,
    rng: np.random.Generator | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Sample a random target joint configuration within the given range.

    Args:
        accessor: Environment state accessor (unused here; present for
            uniform call signature).
        task_state: Current task state dict (mutated in-place).
        target_jnt_range: Array of shape (nj, 2) with [lo, hi] per joint.
        rng: NumPy random generator. Uses ``np.random`` if None.
        **kwargs: Unused extra keyword arguments.

    Returns:
        Updated task_state with ``"target_angles"`` set to a random sample.
    """
    _rng: np.random.Generator | np.random.RandomState = (
        rng if rng is not None else cast(np.random.RandomState, np.random)
    )
    lo, hi = target_jnt_range[:, 0], target_jnt_range[:, 1]
    task_state["target_angles"] = _rng.uniform(lo, hi)
    return task_state


def reset_fixed_pose(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    target_angles: np.ndarray,
    **kwargs: Any,
) -> dict[str, Any]:
    """Set a fixed target joint configuration.

    Args:
        accessor: Environment state accessor (unused here).
        task_state: Current task state dict (mutated in-place).
        target_angles: Target joint angles, shape (nj,).
        **kwargs: Unused extra keyword arguments.

    Returns:
        Updated task_state with ``"target_angles"`` set.
    """
    task_state["target_angles"] = np.array(target_angles, dtype=np.float64)
    return task_state


def apply_sarcopenia(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    force_scale: float = 0.5,
    **kwargs: Any,
) -> dict[str, Any]:
    """Apply sarcopenic muscle weakening to the compiled model.

    Scales actuator force ranges on the model accessible via the accessor.
    This is intended as a startup event (called once per env instantiation).

    Args:
        accessor: Environment state accessor; must expose ``.model`` attribute.
        task_state: Current task state dict (returned unchanged).
        force_scale: Fraction of original peak force to retain (0–1).
        **kwargs: Unused extra keyword arguments.

    Returns:
        Unchanged task_state.
    """
    if hasattr(accessor, "model"):
        from myosuite.core.muscle_conditions import apply_sarcopenia_to_model

        apply_sarcopenia_to_model(accessor.model, force_scale=force_scale)
    return task_state
