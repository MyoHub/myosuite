# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
from typing import Any

import gymnasium as gym

__all__ = ["gym", "seed_envs"]

_COMPAT_SUBMODULES = frozenset(
    {
        "asset_path_resolver",
        "colored_noise",
        "mujoco_geom_utils",
        "mujoco_joint_utils",
        "spec_processing",
    }
)


def seed_envs(seed):
    """Create a seeded NumPy random generator via gymnasium.

    Args:
        seed: Integer random seed.

    Returns:
        Tuple of (np_random, seed) where np_random is a numpy.random.Generator.
    """
    np_random, seed = gym.utils.seeding.np_random(seed)
    return np_random, seed


def __getattr__(name: str) -> Any:
    if name in _COMPAT_SUBMODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
