# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

import gymnasium as gym

__all__ = ["gym", "seed_envs"]


def seed_envs(seed):
    """Create a seeded NumPy random generator via gymnasium.

    Args:
        seed: Integer random seed.

    Returns:
        Tuple of (np_random, seed) where np_random is a numpy.random.Generator.
    """
    np_random, seed = gym.utils.seeding.np_random(seed)
    return np_random, seed
