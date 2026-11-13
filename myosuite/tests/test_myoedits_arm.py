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
from myosuite.utils.spec_processing import delete_joint

pytestmark = pytest.mark.tier2

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


_NODE_XML = """
<mujoco>
  <worldbody>
    <body name="a">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2"/>
      <site name="s1" pos="0.05 0 0"/>
      <body name="node" pos="0 0 -0.1">
        <joint name="slide" type="slide" axis="0 0 1"/>
        <geom type="sphere" size="0.01"/>
        <site name="s2" pos="0.05 0 0"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="slide_t"><joint joint="slide" coef="1"/></fixed>
    <spatial name="path_t"><site site="s1"/><site site="s2"/></spatial>
  </tendon>
  <equality>
    <tendon tendon1="slide_t" tendon2="path_t"/>
    <joint joint1="slide" joint2="hinge"/>
  </equality>
  <actuator>
    <motor name="m_slide" joint="slide"/>
    <motor name="m_t" tendon="slide_t"/>
    <motor name="m_path" tendon="path_t"/>
  </actuator>
  <sensor>
    <jointpos joint="slide"/>
    <tendonpos tendon="slide_t"/>
  </sensor>
</mujoco>
"""


def test_delete_joint_removes_its_dependents() -> None:
    """A tendon-tied node joint is deleted with its tendon, equalities, actuators and sensors."""
    spec = mujoco.MjSpec.from_string(_NODE_XML)
    delete_joint(spec, spec.joint("slide"))
    model = spec.compile()
    assert model.njnt == 1 and model.neq == 0 and model.nsensor == 0
    assert [model.tendon(i).name for i in range(model.ntendon)] == ["path_t"]
    assert [model.actuator(i).name for i in range(model.nu)] == ["m_path"]


def test_edit_builds_with_lumbrical_nodes() -> None:
    """The edit compiles on hands with tendon-tied lumbrical nodes (myo_sim#147).

    Skipped when the installed myo_sim hand has no such nodes.
    """
    original, spec = build_from_recipe("full_arm")
    names = {original.joint(i).name for i in range(original.njnt)}
    if not any(n.startswith("node_LUM") for n in names):
        pytest.skip("installed myo_sim hand has no lumbrical junction nodes")
    edit_fn_arm_reaching(spec)
    model = spec.compile()
    hand = set()
    for i in range(model.njnt):
        if any(
            model.body(b).name.endswith("mc_r")
            for b in _ancestors(model, model.jnt_bodyid[i])
        ):
            hand.add(model.joint(i).name)
    assert not hand
    assert {"IFtip", "IFtip_target"} <= {model.site(i).name for i in range(model.nsite)}
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, nstep=10)
    assert np.isfinite(data.qpos).all()
