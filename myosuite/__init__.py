""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import gym

# Register MyoSuite Envs
_current_gym_envs = gym.envs.registration.registry.env_specs.keys()
_current_gym_envs = set(_current_gym_envs)
myosuite_env_suite = set()


# Register MyoBase Suite
import myosuite.envs.myo.myobase # noqa
myosuite_myobase_suite = set(gym.envs.registration.registry.env_specs.keys())-myosuite_env_suite-_current_gym_envs
myosuite_env_suite  = myosuite_env_suite | myosuite_myobase_suite
myosuite_myobase_suite = sorted(myosuite_myobase_suite)

# Register MyoChal Suite
import myosuite.envs.myo.myochallenge # noqa
myosuite_myochal_suite = set(gym.envs.registration.registry.env_specs.keys())-myosuite_env_suite-_current_gym_envs
myosuite_env_suite  = myosuite_env_suite | myosuite_myochal_suite
myosuite_myochal_suite = sorted(myosuite_myochal_suite)

# Register MyoDex Suite
import myosuite.envs.myo # noqa
import myosuite.envs.myo.myodex # noqa
myosuite_myodex_suite = set(gym.envs.registration.registry.env_specs.keys())-myosuite_env_suite-_current_gym_envs
myosuite_env_suite  = myosuite_env_suite | myosuite_myodex_suite
myosuite_myodex_suite = sorted(myosuite_myodex_suite)

# All myosuite Envs
myosuite_env_suite = sorted(myosuite_env_suite)

from typing import List

from .version import __version_tuple__

__version__ = ".".join([str(x) for x in __version_tuple__])
__all__: List[str] = []
