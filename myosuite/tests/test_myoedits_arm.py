# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Geometry of the arm-reaching model edit (rigid, joint-free digits)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite.core.model_builder import build_from_recipe
from myosuite.envs.myo.myoedits import edit_fn_arm_reaching

_METACARPALS = ("secondmc_r", "thirdmc_r", "fourthmc_r", "fifthmc_r")


def _ancestors(model: mujoco.MjModel, body_id: int):
    while body_id > 0:
        yield body_id
        body_id = model.body_parentid[body_id]


def _digit_reach(model: mujoco.MjModel, metacarpal: str) -> tuple[int, float]:
    """Number of bodies below *metacarpal* and the distance to the farthest one."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    root = model.body(metacarpal).id
    sub = [i for i in range(model.nbody) if i != root and root in _ancestors(model, i)]
    far = max(np.linalg.norm(data.xpos[i] - data.xpos[root]) for i in sub)
    return len(sub), float(far)


@pytest.mark.parametrize("metacarpal", _METACARPALS)
def test_edited_digits_keep_their_metacarpal_and_length(metacarpal: str) -> None:
    """Each rebuilt digit hangs off its own metacarpal with the original reach.

    Regression: MjSpec name lookups after deleting/adding bodies returned
    index-shifted bodies, chaining digits 3-5 onto the index finger.
    """
    original, _ = build_from_recipe("full_arm")
    _, spec = build_from_recipe("full_arm")
    edit_fn_arm_reaching(spec)
    edited = spec.compile()

    n_orig, reach_orig = _digit_reach(original, metacarpal)
    n_edit, reach_edit = _digit_reach(edited, metacarpal)
    assert n_edit == n_orig
    assert reach_edit == pytest.approx(reach_orig, abs=1e-6)


def test_index_tip_site_is_on_the_index_chain() -> None:
    _, spec = build_from_recipe("full_arm")
    edit_fn_arm_reaching(spec)
    model = spec.compile()
    tip_body = model.site_bodyid[model.site("IFtip").id]
    chain = [model.body(i).name for i in _ancestors(model, tip_body)]
    assert "secondmc_r" in chain
    assert not any("3proxph" in n or "thirdmc" in n for n in chain)


def test_hand_is_rigid_no_thumb_muscles_or_hand_joints() -> None:
    """Only shoulder/elbow/forearm/wrist joints and their muscles remain."""
    _, spec = build_from_recipe("full_arm")
    edit_fn_arm_reaching(spec, min_moment=1e-2, remove_wrist=False)
    model = spec.compile()
    joints = {model.joint(i).name for i in range(model.njnt)}
    {model.actuator(i).name for i in range(model.nu)}
    {model.tendon(i).name for i in range(model.ntendon)}

    assert {"pro_sup_r", "deviation_r", "flexion_r"} <= joints  # wrist kept
    assert not joints & {"cmc_flexion", "cmc_abduction"}
    assert not any(n.startswith(("mcp", "pm", "md", "ip_", "mp_")) for n in joints)
    assert model.nu == 45
