# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The joint couplings and the fatigue model live in myo_sim; MyoSuite's old import paths re-export them."""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.tier1

_SHIMS = {
    "myosuite.physics.joint_coupling": (
        "myo_sim.physics.joint_coupling",
        ["JointCouplings", "project_joint_couplings", "projected_joint_couplings"],
    ),
    "myosuite.physics.fatigue": (
        "myo_sim.physics.fatigue",
        [
            "CumulativeFatigue",
            "TorchFatigueState",
            "MUSCLE_FATIGUE_PARAMS",
            "three_cc_r_step",
        ],
    ),
    "myosuite.core.muscle_conditions": (
        "myo_sim.physics.fatigue",
        [
            "CumulativeFatigue",
            "TorchFatigueState",
            "MUSCLE_FATIGUE_PARAMS",
            "FATIGUE_REST_THRESHOLD",
            "three_cc_r_step",
            "torch_three_cc_r_step",
            "_UNIFORM_PARAMS",
            "_muscle_group",
        ],
    ),
    "myosuite.core.muscle_groups": (
        "myo_sim.physics.muscle_groups",
        ["FUNCTIONAL_MUSCLE_GROUPS", "MUSCLE_FMG"],
    ),
}


@pytest.mark.parametrize("shim", sorted(_SHIMS))
def test_shim_reexports_myo_sim(shim: str) -> None:
    target, names = _SHIMS[shim]
    old, new = importlib.import_module(shim), importlib.import_module(target)
    for name in names:
        assert getattr(old, name) is getattr(new, name), name


def test_jax_shim_reexports_myo_sim() -> None:
    pytest.importorskip("jax")
    from myo_sim.physics import fatigue_jax

    from myosuite.physics import fatigue_jax as shim

    assert shim.cumulative_fatigue_step is fatigue_jax.cumulative_fatigue_step
    assert shim.CumulativeFatigue is fatigue_jax.CumulativeFatigue
