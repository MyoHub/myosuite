""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import gym

# Register MyoSuite Envs
_current_gym_envs = gym.envs.registration.registry.env_specs.keys()
_current_gym_envs = set(_current_gym_envs)

# Register Myo Suite
import myosuite.envs.myo.myobase # noqa
import myosuite.envs.myo.myochallenge # noqa
myo_suite_envs = set(gym.envs.registration.registry.env_specs.keys())-_current_gym_envs
myo_suite_envs = sorted(myo_suite_envs)

from typing import List

from .version import __version_tuple__

__version__ = ".".join([str(x) for x in __version_tuple__])
__all__: List[str] = []
