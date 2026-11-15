# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Re-registering an env id: identical specs are a no-op, and only the module that
registered an id may change its spec."""

import copy
import importlib
from collections.abc import Iterator

import gymnasium as gym
import numpy as np
import pytest

import myosuite
from myosuite import make_env
from myosuite.core.registry import (
    _ENV_REGISTRY,
    _REGISTERED_BY,
    _deep_equal,
    register_env,
)
from myosuite.envs.myo.myoedits import edit_fn_arm_reaching

pytestmark = pytest.mark.tier2

# The myoArmReach ids and their muscle-condition variants (all from myoedits).
_ARM_REACH_IDS = tuple(
    f"myo{condition}ArmReach{target}-v0"
    for condition in ("", "Sarc", "Fati")
    for target in ("Fixed", "Random")
)
_REACH_EP = "myosuite.envs.myo.tasks.basic.arm.reach:ReachEnvV0"


@pytest.fixture
def restore_registry() -> Iterator[None]:
    """Restore the env registries after the test, so a reload cannot leak into later tests."""
    live = (gym.registry, _ENV_REGISTRY, _REGISTERED_BY)
    saved = tuple(dict(registry) for registry in live)
    yield
    for registry, snapshot in zip(live, saved):
        registry.clear()
        registry.update(snapshot)


def _model_size(env_id: str) -> tuple[int, int]:
    """(nu, nv) of the model ``make_env(env_id)`` builds."""
    env = make_env(env_id)
    try:
        return env.unwrapped.model.nu, env.unwrapped.model.nv
    finally:
        env.close()


def _register_from(module: str, env_id: str, far_th: float) -> None:
    """Call ``register_env`` from code that belongs to *module*."""
    # exec with its own globals gives the calling frame another module name
    exec(
        "register_env(env_id, ep, max_episode_steps=10, kwargs={'far_th': far_th})",
        {
            "__name__": module,
            "register_env": register_env,
            "env_id": env_id,
            "ep": _REACH_EP,
            "far_th": far_th,
        },
    )


def test_deep_equal_handles_arrays_and_nesting() -> None:
    a = {"t": np.arange(3.0), "n": {"x": [np.ones(2), 1]}}
    assert _deep_equal(a, copy.deepcopy(a))
    assert not _deep_equal(a, {"t": np.arange(3.0) + 1, "n": a["n"]})
    assert not _deep_equal(a, {"t": a["t"]})


@pytest.mark.usefixtures("restore_registry")
def test_reloading_the_basic_registrations_with_array_kwargs_does_not_raise() -> None:
    import myosuite.envs.myo.tasks.basic as basic

    before = gym.spec("myoHandPoseFixed-v0").kwargs["target_jnt_value"]
    importlib.reload(basic)
    after = gym.spec("myoHandPoseFixed-v0").kwargs["target_jnt_value"]
    assert np.array_equal(before, after)


@pytest.mark.usefixtures("restore_registry")
@pytest.mark.parametrize("env_id", _ARM_REACH_IDS)
def test_reloading_the_basic_registrations_keeps_the_arm_reach_model(
    env_id: str,
) -> None:
    """The basic suite used to register these ids too, with a 63-muscle arm."""
    import myosuite.envs.myo.tasks.basic as basic

    before = _model_size(env_id)
    importlib.reload(basic)
    assert gym.spec(env_id).kwargs["edit_fn"] is edit_fn_arm_reaching
    assert _model_size(env_id) == before


def test_the_arm_reach_ids_are_in_the_myoedits_suite_only() -> None:
    assert set(_ARM_REACH_IDS) <= set(myosuite.myosuite_myoedits_suite)
    assert not set(_ARM_REACH_IDS) & set(myosuite.myosuite_myobase_suite)
    assert {_REGISTERED_BY[i] for i in _ARM_REACH_IDS} == {"myosuite.envs.myo.myoedits"}


@pytest.mark.usefixtures("restore_registry")
def test_only_the_registering_module_can_change_a_spec() -> None:
    env_id = "TestReregistrationOwner-v0"
    _register_from("owner_module", env_id, far_th=1.0)
    _register_from("owner_module", env_id, far_th=2.0)  # e.g. importlib.reload
    assert gym.spec(env_id).kwargs["far_th"] == 2.0

    _register_from("other_module", env_id, far_th=2.0)  # identical: no-op
    with pytest.raises(ValueError, match="already registered by owner_module"):
        _register_from("other_module", env_id, far_th=3.0)
    assert gym.spec(env_id).kwargs["far_th"] == 2.0
    assert _REGISTERED_BY[env_id] == "owner_module"
