# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Physical landmark geometry, coupling derivatives and nuisance ambiguity."""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest

from myosuite.biomechanics.camera import PinholeCamera
from myosuite.biomechanics.observability import (
    camera_landmark_observability,
    independent_dofs,
    landmark_observability,
)
from myosuite.biomechanics.retarget import Landmark

pytestmark = pytest.mark.tier1


def _arm() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><worldbody>'
        '<body name="upper"><joint name="swing" axis="0 0 1"/>'
        '<geom size=".1" mass="1"/><body name="fore">'
        '<joint name="twist" axis="1 0 0"/><geom size=".1" mass="1"/>'
        '<site name="axial" pos="1 0 0"/><site name="radial" pos="1 .2 0"/>'
        '<body><joint name="finger"/><geom size=".02" mass=".1"/>'
        "</body></body></body></worldbody></mujoco>"
    )


def _sites(*names: str) -> list[Landmark]:
    return [Landmark(name, "", site=name) for name in names]


def test_axial_landmarks_do_not_observe_twist_and_off_axis_points_do() -> None:
    model = _arm()
    q = np.array([0.3, 0.4, 0.2])
    axial = landmark_observability(model, q, _sites("axial"), target_dofs=[0, 1])
    assert axial.rank == 1 and axial.ambiguous_dofs == ("twist",)
    assert axial.unconstrained_dofs == ("twist",)
    assert axial.record()["condition_number"] is None
    observed = landmark_observability(
        model, q, _sites("axial", "radial"), target_dofs=[0, 1]
    )
    assert observed.rank == 2 and np.isfinite(observed.condition_number)
    assert observed.ambiguous_groups == ()
    assert observed.nuisance_dofs[0].name == "finger"
    # An untracked finger is diagnostic when all DOFs are requested, but cannot
    # invalidate the requested arm scope when it has no effect on these points.
    assert landmark_observability(model, q, _sites("axial", "radial")).rank == 2
    json.dumps(axial.record(include_matrices=True), allow_nan=False)


def test_short_off_axis_lever_reduces_conditioning_before_rank_is_lost() -> None:
    model = _arm()
    q = np.zeros(3)
    strong = landmark_observability(
        model, q, _sites("axial", "radial"), target_dofs=[0, 1]
    )
    model.site("radial").pos[1] *= 0.01
    weak = landmark_observability(
        model, q, _sites("axial", "radial"), target_dofs=[0, 1]
    )
    assert strong.rank == weak.rank == 2
    assert weak.condition_number > 90 * strong.condition_number
    thresholded = landmark_observability(
        model, q, _sites("axial", "radial"), target_dofs=[0, 1], rtol=0.01
    )
    assert thresholded.rank == 1 and thresholded.ambiguous_dofs == ("twist",)


def test_omitted_joint_is_free_to_mimic_target_until_explicitly_fixed() -> None:
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint name="a" axis="0 0 1"/>'
        '<geom size=".1" mass="1"/><body><joint name="b" axis="0 0 1"/>'
        '<geom size=".1" mass="1"/><site name="tip" pos="1 0 0"/>'
        "</body></body></worldbody></mujoco>"
    )
    q = np.array([0.2, 0.3])
    full = landmark_observability(model, q, _sites("tip"))
    assert full.rank == 1 and full.ambiguous_groups == (("a", "b"),)
    target = landmark_observability(model, q, _sites("tip"), target_dofs=[0], atol=0)
    assert np.linalg.norm(target.jacobian) > 0.9
    assert target.rank == 0 and target.nuisance_rank == 1
    assert target.ambiguous_dofs == ("a",)
    fixed = landmark_observability(
        model, q, _sites("tip"), target_dofs=[0], fixed_dofs=[1]
    )
    assert fixed.rank == 1 and fixed.condition_number == pytest.approx(1)
    assert fixed.record()["fixed_count"] == 1


def test_coupled_position_derivative_uses_existing_velocity_projection() -> None:
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><worldbody><body>'
        '<joint name="master" axis="0 0 1"/><geom size=".1" mass="1"/>'
        '<body pos=".5 0 0"><joint name="dependent" axis="0 0 1"/>'
        '<geom size=".1" mass="1"/><site name="tip" pos="1 0 0"/>'
        '</body></body></worldbody><equality><joint joint1="dependent" '
        'joint2="master" polycoef="0 2 .5 0 0"/></equality></mujoco>'
    )
    q = np.array([0.4, 99.0])
    result = landmark_observability(model, q, _sites("tip"))
    m, dep, slope = 0.4, 2 * 0.4 + 0.5 * 0.4**2, 2 + 0.4
    expected = [
        -0.5 * np.sin(m) - (1 + slope) * np.sin(m + dep),
        0.5 * np.cos(m) + (1 + slope) * np.cos(m + dep),
        0,
    ]
    np.testing.assert_allclose(result.jacobian[:, 0], expected, atol=1e-12)
    assert result.target_dofs[0].name == "master" and len(result.target_dofs) == 1
    np.testing.assert_array_equal(q, [0.4, 99.0])
    with pytest.raises(ValueError, match="independent"):
        landmark_observability(model, q, _sites("tip"), target_dofs=[1])


def test_free_root_units_scaling_and_native_tangent_derivatives() -> None:
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint name="root"/>'
        '<geom size=".1" mass="1"/><site name="p0"/>'
        '<site name="p1" pos="1 0 0"/><site name="p2" pos="0 1 0"/>'
        "</body></worldbody></mujoco>"
    )
    q = np.array([0.2, -0.3, 0.4, np.cos(0.3), 0, np.sin(0.3), 0])
    result = landmark_observability(
        model,
        q,
        _sites("p0", "p1", "p2"),
        translation_scale_m=0.2,
        rotation_scale_rad=0.3,
        landmark_scale_m=0.02,
        dof_scales={3: 0.5},
    )
    assert result.rank == 6
    assert [d.unit for d in result.target_dofs] == ["m"] * 3 + ["rad"] * 3
    assert result.target_dofs[3].name == "root/rx"
    np.testing.assert_allclose(result.target_scales, [0.2, 0.2, 0.2, 0.5, 0.3, 0.3])
    np.testing.assert_allclose(
        result.effective_jacobian, result.jacobian * result.target_scales / 0.02
    )
    for index in range(6):
        velocity = np.eye(6)[index]
        positions = []
        for step in (-1e-6, 1e-6):
            perturbed = q.copy()
            mujoco.mj_integratePos(model, perturbed, velocity, step)
            data = mujoco.MjData(model)
            data.qpos[:] = perturbed
            mujoco.mj_kinematics(model, data)
            positions.append(data.site_xpos.copy().ravel())
        np.testing.assert_allclose(
            (positions[1] - positions[0]) / 2e-6, result.jacobian[:, index], atol=1e-9
        )
    with pytest.raises(ValueError, match="quaternions"):
        landmark_observability(model, np.zeros(7), _sites("p0"))


def test_empty_landmarks_and_no_independent_dofs_are_explicit() -> None:
    missing = landmark_observability(_arm(), np.zeros(3), [])
    assert missing.rank == 0 and missing.nullspace.shape == (3, 3)
    for xml, q in [
        ('<mujoco><worldbody><site name="p"/></worldbody></mujoco>', np.empty(0)),
        (
            '<mujoco><worldbody><body><joint name="locked"/><geom size=".1"/>'
            '<site name="p"/></body></worldbody><equality><joint joint1="locked" '
            'polycoef=".2 0 0 0 0"/></equality></mujoco>',
            np.array([0.7]),
        ),
    ]:
        model = mujoco.MjModel.from_xml_string(xml)
        assert independent_dofs(model) == ()
        result = landmark_observability(model, q, _sites("p"))
        assert result.rank == 0 and result.nullspace.shape == (0, 0)
        assert result.record()["status"] == "empty_scope"
        json.dumps(result.record(include_matrices=True), allow_nan=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_dofs": [True]},
        {"target_dofs": [0, 0]},
        {"target_dofs": [0], "fixed_dofs": [0]},
        {"dof_scales": {0: 0}},
        {"rotation_scale_rad": float("nan")},
        {"atol": -1},
    ],
)
def test_invalid_scopes_and_scales_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        landmark_observability(_arm(), np.zeros(3), _sites("axial"), **kwargs)


def test_malformed_landmark_mapping_does_not_silently_fall_back() -> None:
    model = _arm()
    with pytest.raises(KeyError):
        landmark_observability(model, np.zeros(3), _sites("missing_site"))
    with pytest.raises(ValueError, match="unique names"):
        landmark_observability(model, np.zeros(3), _sites("axial", "axial"))


def test_camera_observability_monocular_depth_nullspace_stereo_and_finite_diff() -> (
    None
):
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body>"
        '<joint name="tx" type="slide" axis="1 0 0"/>'
        '<joint name="ty" type="slide" axis="0 1 0"/>'
        '<joint name="tz" type="slide" axis="0 0 1"/>'
        '<geom size="0.05" mass="1"/><site name="pt" pos="0 0 0"/>'
        "</body></worldbody></mujoco>"
    )
    q = np.array([0.1, -0.2, 3.0])
    k = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    cam0 = PinholeCamera(k, np.eye(3), np.zeros(3), "cam0")
    cam1 = PinholeCamera(k, np.eye(3), np.array([-0.8, 0.0, 0.0]), "cam1")

    # One camera sees 2 pixel equations -> blind along the optical ray (rank 2 < 3).
    mono = camera_landmark_observability(model, q, _sites("pt"), [cam0])
    assert mono.observability.rank == 2
    assert "tz" in mono.observability.ambiguous_dofs
    assert mono.views_per_landmark.tolist() == [1]

    # Stereo parallax requires min_views=2: cam0 alone gives rank 0; cam0 + cam1 gives rank 3.
    stereo_blocked = camera_landmark_observability(
        model, q, _sites("pt"), [cam0], min_views=2
    )
    assert (
        stereo_blocked.observability.rank == 0 and stereo_blocked.active_landmarks == ()
    )

    stereo = camera_landmark_observability(
        model, q, _sites("pt"), [cam0, cam1], min_views=2
    )
    assert stereo.observability.rank == 3
    assert stereo.observability.ambiguous_dofs == ()

    # Verify analytic pixel Jacobian matches central finite differences of project().
    eps = 1e-6
    for col in range(3):
        dq = np.zeros(3)
        dq[col] = eps
        uv_vals = []
        for sign in (-1.0, 1.0):
            d = mujoco.MjData(model)
            d.qpos[:] = q + sign * dq
            mujoco.mj_kinematics(model, d)
            uv_vals.append(
                np.concatenate([c.project(d.site_xpos[0]) for c in (cam0, cam1)])
            )
        fd = (uv_vals[1] - uv_vals[0]) / (2.0 * eps)
        np.testing.assert_allclose(stereo.observability.jacobian[:, col], fd, atol=1e-7)
    json.dumps(stereo.record(include_matrices=True), allow_nan=False)


def test_camera_observability_detects_mj_ray_occlusion_and_frustum_bounds() -> None:
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        '<geom name="pillar" type="sphere" pos="0 0 1.5" size="0.35"/>'
        '<body name="seg" pos="0 0 3.0">'
        '<joint name="slide_x" type="slide" axis="1 0 0"/>'
        '<geom type="sphere" size="0.06" mass="1"/>'
        '<site name="marker" pos="0 0 -0.06"/>'
        "</body></worldbody></mujoco>"
    )
    k = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    blocked_cam = PinholeCamera(k, np.eye(3), np.zeros(3), "blocked")
    side_cam = PinholeCamera(k, np.eye(3), np.array([-1.2, 0.0, 0.0]), "clear")

    res = camera_landmark_observability(
        model, np.zeros(1), _sites("marker"), [blocked_cam, side_cam]
    )
    assert res.visibility.tolist() == [[False], [True]]
    assert res.observability.rank == 1

    # Ignoring occlusion restores visibility in blocked_cam, unless cropped by image_sizes.
    no_occ = camera_landmark_observability(
        model,
        np.zeros(1),
        _sites("marker"),
        [blocked_cam, side_cam],
        check_occlusion=False,
        image_sizes=[(1280, 720), (300, 300)],
    )
    assert no_occ.visibility.tolist() == [[True], [False]]
