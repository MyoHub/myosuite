# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Measured wrench and held-mass mapping regression tests."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite.biomechanics.external_loads import (
    apply_body_wrenches_to_qfrc,
    force_at_body,
    force_at_site,
    held_mass_qfrc,
)

pytestmark = pytest.mark.tier1


def _arm() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><worldbody><body name="arm">'
        '<joint name="elbow" axis="0 1 0"/><geom size=".1" mass="3"/>'
        '<site name="hand" pos="1 0 0"/></body></worldbody></mujoco>'
    )


def test_wrench_mapping_matches_lever_arm_and_rotation() -> None:
    model = _arm()
    for theta in (0.0, 0.7, 1.5):
        q = np.array([theta])
        expected = 2 * 9.81 * np.cos(theta)
        assert held_mass_qfrc(model, q, 2, site="hand")[0] == pytest.approx(expected)
        from_site = force_at_site(model, q, "hand", np.array([0, 0, -10.0]))
        from_body = force_at_body(
            model,
            q,
            "arm",
            np.array([0, 0, -10.0]),
            point=np.array([np.cos(theta), 0, -np.sin(theta)]),
        )
        np.testing.assert_allclose(from_site, from_body, atol=1e-10)
        assert force_at_site(model, q, "hand", np.zeros(3), torque=np.array([0, 5, 0]))[
            0
        ] == pytest.approx(5)
        assert force_at_site(model, q, "hand", np.array([0, 0, -10]), frame="site")[
            0
        ] == pytest.approx(10)
        np.testing.assert_allclose(
            force_at_body(
                model,
                q,
                "arm",
                np.array([0, 0, -10]),
                point=np.array([1, 0, 0]),
                frame="body",
            ),
            force_at_site(model, q, "hand", np.array([0, 0, -10]), frame="site"),
        )
    with pytest.raises(ValueError, match="choose body"):
        held_mass_qfrc(model, np.zeros(1), 2, body="arm", site="hand")


def test_apply_body_wrenches_to_qfrc_matches_force_at_body_and_validates_shapes() -> (
    None
):
    model = _arm()
    qpos = np.array([[0.0], [0.4], [-0.3]])
    point = np.array([0.6, 0.0, -0.2])
    forces = np.array([[0.0, 0.0, -12.0], [3.0, 0.0, -8.0], [-2.0, 0.0, 5.0]])
    tau_free = np.array([[0.0, 1.5, 0.0], [0.0, -0.5, 0.0], [0.0, 2.0, 0.0]])
    tau_origin = tau_free + np.cross(point[None, :], forces)
    bid = model.body("arm").id

    got = apply_body_wrenches_to_qfrc(model, qpos, [(bid, forces, tau_origin)])
    expected = np.stack(
        [
            force_at_body(
                model, qpos[t], "arm", forces[t], point=point, torque=tau_free[t]
            )
            for t in range(len(qpos))
        ],
        axis=0,
    )
    np.testing.assert_allclose(got, expected, atol=1e-12)

    with pytest.raises(ValueError, match="force array must be finite"):
        apply_body_wrenches_to_qfrc(model, qpos, [(bid, forces[:2], tau_origin)])
    with pytest.raises(ValueError, match="torque array must be finite"):
        apply_body_wrenches_to_qfrc(model, qpos, [(bid, forces, tau_origin[:2])])
