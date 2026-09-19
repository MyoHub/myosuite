# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""MyoSuite package entrypoint.

Importing this module registers all MyoSuite environments with Gymnasium,
so ``gym.make("myoEnvName-v0")`` works immediately after ``import myosuite``.
"""

from __future__ import annotations

from typing import Any

from myosuite.utils import gym

from .version import __version_tuple__

__version__ = ".".join([str(x) for x in __version_tuple__])

# Populated by register_all_envs().
myosuite_env_suite: list[str] = []
myosuite_myobase_suite: list[str] = []
myosuite_myochal_suite: list[str] = []
myosuite_myomimic_suite: list[str] = []
myosuite_myoedits_suite: list[str] = []


def gym_registry_specs() -> Any:
    """Return the Gymnasium environment registry."""

    return gym.envs.registry


def register_all_envs() -> dict[str, list[str]]:
    """Register all MyoSuite environment suites with Gymnasium.

    Returns:
        Mapping of suite name to registered environment ids.
    """

    global myosuite_env_suite
    global myosuite_myobase_suite
    global myosuite_myochal_suite
    global myosuite_myomimic_suite
    global myosuite_myoedits_suite

    current_envs = set(gym_registry_specs().keys())
    suite_envs: set[str] = set()

    # Register MyoBase suite
    import myosuite.envs.myo.tasks.basic as _myobase  # noqa: F401

    del _myobase

    myosuite_myobase_suite = sorted(
        set(gym_registry_specs().keys()) - suite_envs - current_envs
    )
    suite_envs |= set(myosuite_myobase_suite)

    # Register MyoChallenge suite
    import myosuite.envs.myo.tasks.challenge as _myochallenge  # noqa: F401

    del _myochallenge

    myosuite_myochal_suite = sorted(
        set(gym_registry_specs().keys()) - suite_envs - current_envs
    )
    suite_envs |= set(myosuite_myochal_suite)

    # Register MyoMimic suite
    import myosuite.envs.myo.tasks.mimic as _myomimic  # noqa: F401

    del _myomimic

    myosuite_myomimic_suite = sorted(
        set(gym_registry_specs().keys()) - suite_envs - current_envs
    )
    suite_envs |= set(myosuite_myomimic_suite)

    # Register MyoEdits suite
    import myosuite.envs.myo.myoedits as _myoedits  # noqa: F401

    del _myoedits

    myosuite_myoedits_suite = sorted(
        set(gym_registry_specs().keys()) - suite_envs - current_envs
    )
    suite_envs |= set(myosuite_myoedits_suite)

    myosuite_env_suite = sorted(suite_envs)

    return {
        "myobase": myosuite_myobase_suite,
        "myochallenge": myosuite_myochal_suite,
        "myomimic": myosuite_myomimic_suite,
        "myoedits": myosuite_myoedits_suite,
        "all": myosuite_env_suite,
    }


register_all_envs()

__all__: list[str] = [
    "__version__",
    "gym_registry_specs",
    "register_all_envs",
    "myosuite_env_suite",
    "myosuite_myobase_suite",
    "myosuite_myochal_suite",
    "myosuite_myomimic_suite",
    "myosuite_myoedits_suite",
]
