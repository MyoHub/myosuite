# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Re-registering an identical spec is a no-op, also with array kwargs."""

import copy
import importlib

import gymnasium as gym
import numpy as np

import myosuite  # noqa: F401  (registers the envs)
from myosuite.core.registry import _deep_equal


def test_deep_equal_handles_arrays_and_nesting() -> None:
    a = {"t": np.arange(3.0), "n": {"x": [np.ones(2), 1]}}
    assert _deep_equal(a, copy.deepcopy(a))
    assert not _deep_equal(a, {"t": np.arange(3.0) + 1, "n": a["n"]})
    assert not _deep_equal(a, {"t": a["t"]})


def test_reloading_the_basic_registrations_with_array_kwargs_does_not_raise() -> None:
    import myosuite.envs.myo.tasks.basic as basic

    before = gym.spec("myoHandPoseFixed-v0").kwargs["target_jnt_value"]
    importlib.reload(basic)
    after = gym.spec("myoHandPoseFixed-v0").kwargs["target_jnt_value"]
    assert np.array_equal(before, after)
