# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for temporal contact-wrench continuity regularization in StaticOptimizer."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite.biomechanics.static_optimization import (
    StaticOptimizationConfig,
    StaticOptimizer,
    static_optimization,
)

pytestmark = pytest.mark.tier1


def test_contact_smooth_weight_regularizes_double_support_handoff() -> None:
    xml = """
    <mujoco>
      <option gravity="0 0 -9.81"/>
      <worldbody>
        <geom name="floor" type="plane" size="2 2 0.1"/>
        <body name="pelvis" pos="0 0 0.5">
          <freejoint name="root"/>
          <inertial pos="0 0 0" mass="40.0" diaginertia="1.0 1.0 1.0"/>
          <body name="heel" pos="-0.08 0 -0.45">
            <joint name="heel_z" type="slide" axis="0 0 1" range="-0.1 0.1"/>
            <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
            <geom name="heel_sole" type="sphere" size="0.05"/>
          </body>
          <body name="midfoot" pos="0 0 -0.45">
            <joint name="mid_z" type="slide" axis="0 0 1" range="-0.1 0.1"/>
            <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
            <geom name="mid_sole" type="sphere" size="0.05"/>
          </body>
          <body name="toe" pos="0.08 0 -0.45">
            <joint name="toe_z" type="slide" axis="0 0 1" range="-0.1 0.1"/>
            <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
            <geom name="toe_sole" type="sphere" size="0.05"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor name="act_h" joint="heel_z" gear="400" ctrllimited="true" ctrlrange="-1 1"/>
        <motor name="act_m" joint="mid_z" gear="400" ctrllimited="true" ctrlrange="-1 1"/>
        <motor name="act_t" joint="toe_z" gear="400" ctrllimited="true" ctrlrange="-1 1"/>
      </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    t_steps = 16
    qpos = np.tile(model.qpos0, (t_steps, 1))
    jitter = np.array([0.0045 if k % 2 == 0 else -0.0045 for k in range(t_steps)])
    qpos[:, 7] = jitter
    qpos[:, 8] = -jitter
    qpos[:, 9] = jitter
    qvel = np.zeros((t_steps, model.nv))
    qacc = np.zeros((t_steps, model.nv))

    res_default = static_optimization(
        model,
        qpos,
        qvel,
        qacc,
        config=StaticOptimizationConfig(contact_forces=True, contact_distance=0.01),
    )
    res_raw = static_optimization(
        model,
        qpos,
        qvel,
        qacc,
        config=StaticOptimizationConfig(
            contact_forces=True, contact_distance=0.01, contact_smooth_weight=0.0
        ),
    )
    # Default contact_smooth_weight=0.0 preserves bit-identical outputs.
    np.testing.assert_array_equal(res_default.activation, res_raw.activation)
    np.testing.assert_array_equal(res_default.contact_force, res_raw.contact_force)

    res_smooth = static_optimization(
        model,
        qpos,
        qvel,
        qacc,
        config=StaticOptimizationConfig(
            contact_forces=True, contact_distance=0.01, contact_smooth_weight=50.0
        ),
    )

    mid_id = model.body("midfoot").id
    jump_raw = float(np.max(np.abs(np.diff(res_raw.contact_force[:, mid_id, 2]))))
    jump_smooth = float(np.max(np.abs(np.diff(res_smooth.contact_force[:, mid_id, 2]))))
    assert jump_raw > 50.0
    assert jump_smooth < 0.25 * jump_raw

    opt = StaticOptimizer(
        model,
        StaticOptimizationConfig(
            contact_forces=True, contact_distance=0.01, contact_smooth_weight=25.0
        ),
    )
    f0 = opt.solve(qpos[0], qvel[0], qacc[0])
    _ = opt.solve(qpos[1], qvel[1], qacc[1])
    opt.reset_contact_history()
    f0_again = opt.solve(qpos[0], qvel[0], qacc[0])
    np.testing.assert_allclose(f0.contact_force, f0_again.contact_force, atol=1e-6)
