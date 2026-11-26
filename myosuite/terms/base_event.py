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

from typing import TYPE_CHECKING, Any

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
        rng: Seeded NumPy random generator (e.g. ``env.np_random``).
        **kwargs: Unused extra keyword arguments.

    Returns:
        Updated task_state with ``"target_angles"`` set to a random sample.

    Raises:
        ValueError: If *rng* is ``None``.
    """
    if rng is None:
        raise ValueError(
            "reset_random_pose requires an explicit seeded rng (e.g. env.np_random)."
        )
    lo, hi = target_jnt_range[:, 0], target_jnt_range[:, 1]
    task_state["target_angles"] = rng.uniform(lo, hi)
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
        force_scale: Fraction of original peak force to retain (0â€“1).
        **kwargs: Unused extra keyword arguments.

    Returns:
        Unchanged task_state.
    """
    if hasattr(accessor, "model"):
        from myosuite.core.muscle_conditions import apply_sarcopenia_to_model

        apply_sarcopenia_to_model(accessor.model, force_scale=force_scale)
    return task_state


_TERRAIN_SAMPLES = 10_000


def terrain_heights(
    terrain: str, variant: str | None, rng: np.random.Generator
) -> np.ndarray:
    """Sample ``(10000,)`` heightfield elevations matching ``LegTerrainEnvV0.reset``."""
    if terrain == "rough":
        rough = rng.uniform(low=-0.5, high=0.5, size=(_TERRAIN_SAMPLES,))
        normalized = (rough - rough.min()) / (rough.max() - rough.min())
        return normalized * 0.08 - 0.02
    if terrain == "hilly":
        flat_length, frequency = 3000, 3
        scalar = 0.63 if variant == "fixed" else float(rng.uniform(0.53, 0.73))
        combined = np.concatenate(
            (
                -2 * np.ones(flat_length),
                -2
                + 0.5
                * (
                    np.sin(
                        np.linspace(
                            0, frequency * np.pi, _TERRAIN_SAMPLES - flat_length
                        )
                        + np.pi / 2
                    )
                    - 1
                ),
            )
        )
        normalized = (combined - combined.min()) / (combined.max() - combined.min())
        return np.flip(normalized.reshape(100, 100) * scalar, (0, 1)).reshape(
            _TERRAIN_SAMPLES
        )
    if terrain == "stairs":
        num_stairs, stair_height = 12, 0.1
        flat = 5200 - (_TERRAIN_SAMPLES - 5200) % num_stairs
        stairs_width = (_TERRAIN_SAMPLES - flat) // num_stairs
        scalar = 2.5 if variant == "fixed" else float(rng.uniform(1.5, 3.5))
        parts = [
            np.full((int(stairs_width // 100), 100), -2 + stair_height * j)
            for j in range(num_stairs)
        ]
        data = np.concatenate([np.full((int(flat // 100), 100), -2)] + parts, axis=0)
        normalized = (data + 2) / (2 + stair_height * num_stairs)
        return np.flip(normalized.reshape(100, 100) * scalar, (0, 1)).reshape(
            _TERRAIN_SAMPLES
        )
    raise ValueError(f"Unknown terrain {terrain!r}")
