# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Deterministic Gymnasium RNG seeding helper."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


def seed_envs(seed: int | None) -> tuple[np.random.Generator, Any]:
    """Create a seeded NumPy random generator via gymnasium.

    Args:
        seed: Integer random seed, or ``None`` for fresh entropy.

    Returns:
        Tuple of ``(np_random, seed)`` where ``np_random`` is a ``numpy.random.Generator``.
    """
    np_random, out_seed = gym.utils.seeding.np_random(seed)
    return np_random, out_seed


__all__ = ["seed_envs"]
