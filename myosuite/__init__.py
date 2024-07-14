""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

# import gym
from myosuite.utils import gym
from myosuite.utils.implement_for import implement_for

#TODO: check versions
@implement_for("gym", None, "0.24")
def gym_registry_specs():
    return gym.envs.registry.env_specs

@implement_for("gym", "0.24", None)
def gym_registry_specs():
    return gym.envs.registry

@implement_for("gymnasium")
def gym_registry_specs():
    return gym.envs.registry

# Register MyoSuite Envs
_current_gym_envs = gym_registry_specs().keys()
_current_gym_envs = set(_current_gym_envs)
myosuite_env_suite = set()

# Register MyoBase Suite
import myosuite.envs.myo.myobase # noqa
myosuite_myobase_suite = set(gym_registry_specs().keys())-myosuite_env_suite-_current_gym_envs
myosuite_env_suite  = myosuite_env_suite | myosuite_myobase_suite
myosuite_myobase_suite = sorted(myosuite_myobase_suite)

# Register MyoChal Suite
import myosuite.envs.myo.myochallenge # noqa
myosuite_myochal_suite = set(gym_registry_specs().keys())-myosuite_env_suite-_current_gym_envs
myosuite_env_suite  = myosuite_env_suite | myosuite_myochal_suite
myosuite_myochal_suite = sorted(myosuite_myochal_suite)

# Register MyoDM Suite
import myosuite.envs.myo # noqa
import myosuite.envs.myo.myodm # noqa
myosuite_myodm_suite = set(gym_registry_specs().keys())-myosuite_env_suite-_current_gym_envs
myosuite_env_suite  = myosuite_env_suite | myosuite_myodm_suite
myosuite_myodm_suite = sorted(myosuite_myodm_suite)

# All myosuite Envs
myosuite_env_suite = sorted(myosuite_env_suite)

from typing import List

from .version import __version_tuple__

__version__ = ".".join([str(x) for x in __version_tuple__])
__all__: List[str] = []
