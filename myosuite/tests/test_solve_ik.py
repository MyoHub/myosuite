# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""solve_ik: position and pose targets, joint ranges, couplings and reachability."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite.physics import inverse_kinematics as ik
from myosuite.physics.inverse_kinematics import IKError, solve_ik

pytestmark = pytest.mark.tier1

# Planar arm: upper 0.3 m, lower 0.25 m, hinges about z.
PLANAR = """<mujoco><compiler angle="radian"/><worldbody>
<body name="upper"><joint name="shoulder" axis="0 0 1"/>
  <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/>
  <body name="lower" pos=".3 0 0"><joint name="elbow" axis="0 0 1" range="0 {elbow_max}"/>
    <geom type="capsule" fromto="0 0 0 .25 0 0" size=".02"/>
    <site name="hand" pos=".25 0 0"/>
  </body>
</body></worldbody></mujoco>"""

# Spatial arm: ball shoulder, hinge elbow and a limited slide along the forearm.
SPATIAL = """<mujoco><compiler angle="radian"/><worldbody>
<body name="upper"><joint name="shoulder" type="ball"/>
  <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/>
  <body name="lower" pos=".3 0 0"><joint name="elbow" axis="0 0 1" range="0 2.5"/>
    <joint name="telescope" type="slide" axis="1 0 0" range="-.02 .02"/>
    <geom type="capsule" fromto="0 0 0 .25 0 0" size=".02"/>
    <site name="hand" pos=".25 0 0"/>
  </body>
</body></worldbody></mujoco>"""

# A follower joint tied to the elbow: follower = 0.1 + 0.5 elbow + 0.2 elbow^2.
COUPLED = """<mujoco><compiler angle="radian"/><worldbody>
<body name="upper"><joint name="shoulder" axis="0 0 1"/>
  <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/>
  <body name="lower" pos=".3 0 0"><joint name="elbow" axis="0 0 1" range="0 2.5"/>
    <geom type="capsule" fromto="0 0 0 .25 0 0" size=".02"/>
    <body name="hand" pos=".25 0 0"><joint name="follower" axis="0 0 1" range="-1 2"/>
      <geom type="box" size=".05 .01 .01" pos=".05 0 0"/>
      <site name="tip" pos=".1 0 0"/>
    </body>
  </body>
</body></worldbody>
<equality><joint name="tie" joint1="follower" joint2="elbow" polycoef=".1 .5 .2 0 0"/></equality>
</mujoco>"""

# Every joint type, two sites and a coupling, for the Jacobian check.
MIXED = """<mujoco><compiler angle="radian"/><worldbody>
<body name="base" pos="0 0 1"><freejoint name="root"/><geom type="box" size=".1 .1 .1"/>
  <body name="upper" pos="0 0 -.1"><joint name="shoulder" type="ball" range="0 1.2"/>
    <geom type="capsule" fromto="0 0 0 0 0 -.3" size=".03"/>
    <body name="lower" pos="0 0 -.3"><joint name="elbow" axis="0 1 0" range="-2 .1"/>
      <joint name="slide" type="slide" axis="0 0 1" range="-.05 .05"/>
      <joint name="follower" axis="1 0 0" range="-1 1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -.3" size=".03"/>
      <site name="tip" pos="0 0 -.3"/><site name="mid" pos=".02 0 -.15"/>
    </body>
  </body>
</body></worldbody>
<equality><joint joint1="follower" joint2="elbow" polycoef=".1 .5 .2 0 0"/></equality>
</mujoco>"""


def _planar(elbow_max: float = 2.5) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(PLANAR.format(elbow_max=elbow_max))


def _site_pose(
    model: mujoco.MjModel, qpos: np.ndarray, site: str
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.site(site).xmat)
    return data.site(site).xpos.copy(), quat


def _angle(qa: np.ndarray, qb: np.ndarray) -> float:
    return 2 * np.arccos(min(1.0, abs(float(qa @ qb))))


def test_position_target_recovers_the_planar_angles() -> None:
    model = _planar()
    hand, _ = _site_pose(model, np.array([0.3, 1.0]), "hand")
    qpos = solve_ik(model, {"hand": hand})
    np.testing.assert_allclose(qpos, [0.3, 1.0], atol=1e-8)
    np.testing.assert_allclose(_site_pose(model, qpos, "hand")[0], hand, atol=1e-10)


def test_pose_target_moves_a_ball_joint_on_its_manifold() -> None:
    model = mujoco.MjModel.from_xml_string(SPATIAL)
    quat = np.array([0.8, 0.2, -0.4, 0.4]) / np.linalg.norm([0.8, 0.2, -0.4, 0.4])
    truth = np.concatenate([quat, [1.1, 0.01]])
    pos, orientation = _site_pose(model, truth, "hand")
    qpos = solve_ik(model, {"hand": (pos, orientation)}, tol=1e-9)
    got_pos, got_quat = _site_pose(model, qpos, "hand")
    np.testing.assert_allclose(got_pos, pos, atol=1e-9)
    assert _angle(got_quat, orientation) < 1e-8
    assert np.isclose(np.linalg.norm(qpos[:4]), 1.0)
    # Orientation alone: the hand's axis turns, its position is free.
    only = solve_ik(model, {"hand": (None, orientation)}, tol=1e-9)
    assert _angle(_site_pose(model, only, "hand")[1], orientation) < 1e-8


def test_joint_ranges_bound_the_solution() -> None:
    model = _planar(elbow_max=1.0)
    hand, _ = _site_pose(model, np.array([0.2, 1.5]), "hand")  # needs elbow 1.5
    with pytest.raises(IKError):
        solve_ik(model, {"hand": hand})
    qpos = solve_ik(
        model, {"hand": hand}, qpos=[0.0, 3.0], strict=False
    )  # seed is clipped
    assert qpos[1] == pytest.approx(1.0)
    # The closest reachable point: elbow at its limit, shoulder aimed at the target.
    spatial = mujoco.MjModel.from_xml_string(SPATIAL)
    far = solve_ik(spatial, {"hand": [0.0, 0.0, 0.9]}, strict=False)
    assert 0.0 <= far[4] <= 2.5 and -0.02 <= far[5] <= 0.02
    assert far[5] == pytest.approx(0.02)  # the telescope stretches to its limit
    # Large residuals converge linearly: micrometres from the exact closest point.
    np.testing.assert_allclose(
        _site_pose(spatial, far, "hand")[0], [0, 0, 0.57], atol=1e-5
    )


def test_coupled_joint_stays_tied_to_its_leader() -> None:
    model = mujoco.MjModel.from_xml_string(COUPLED)
    elbow = 0.9
    truth = np.array([0.4, elbow, 0.1 + 0.5 * elbow + 0.2 * elbow**2])
    tip, _ = _site_pose(model, truth, "tip")

    def tie(qpos: np.ndarray) -> float:
        return abs(qpos[2] - (0.1 + 0.5 * qpos[1] + 0.2 * qpos[1] ** 2))

    qpos = solve_ik(model, {"tip": tip}, tol=1e-9)
    assert tie(qpos) < 1e-9
    np.testing.assert_allclose(qpos, truth, atol=1e-7)
    # Selecting only the elbow still moves its follower; the shoulder holds.
    seed = np.array([0.4, 0.2, 0.1 + 0.5 * 0.2 + 0.2 * 0.2**2])
    moved = solve_ik(model, {"tip": tip}, joints=["elbow"], qpos=seed, strict=False)
    assert moved[0] == seed[0] and tie(moved) < 1e-6 and moved[2] != seed[2]
    # Out of reach, the closest pose keeps the coupling nearly exact.
    far = solve_ik(model, {"tip": [2.0, 0.0, 0.0]}, strict=False)
    assert tie(far) < 1e-4


def test_unreachable_target_raises_or_returns_the_closest_pose() -> None:
    model = _planar()
    target = np.array([0.6, 0.8, 0.0])  # 1 m away; the arm reaches 0.55 m
    with pytest.raises(IKError) as raised:
        solve_ik(model, {"hand": target})
    assert raised.value.error == pytest.approx(0.45, abs=1e-6)
    closest = solve_ik(model, {"hand": target}, strict=False)
    np.testing.assert_allclose(closest, raised.value.qpos)
    np.testing.assert_allclose(
        _site_pose(model, closest, "hand")[0], 0.55 * target, atol=1e-6
    )


def test_a_stalled_seed_is_retried_from_mid_range() -> None:
    model = _planar()
    target = [0.4, 0.0, 0.0]  # straight back along the stretched (singular) arm
    with pytest.raises(IKError):
        solve_ik(model, {"hand": target}, retry=False)
    qpos = solve_ik(model, {"hand": target})
    np.testing.assert_allclose(_site_pose(model, qpos, "hand")[0], target, atol=1e-9)


def test_unselected_joints_keep_their_seed() -> None:
    model = mujoco.MjModel.from_xml_string(SPATIAL)
    seed = np.array([1.0, 0.0, 0.0, 0.0, 0.5, 0.0])
    target = _site_pose(model, np.array([1.0, 0, 0, 0, 1.2, 0.0]), "hand")[0]
    qpos = solve_ik(model, {"hand": target}, joints=["elbow"], qpos=seed)
    np.testing.assert_array_equal(qpos[[0, 1, 2, 3, 5]], seed[[0, 1, 2, 3, 5]])
    assert qpos[4] == pytest.approx(1.2, abs=1e-8)


@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
def test_analytic_jacobian_matches_finite_differences(jacobian: str) -> None:
    model = mujoco.MjModel.from_xml_string(MIXED)
    model.opt.jacobian = getattr(mujoco.mjtJacobian, f"mjJAC_{jacobian.upper()}")
    rng = np.random.default_rng(0)
    targets = ik._parse_targets(
        model,
        {"tip": (rng.normal(size=3), rng.normal(size=4)), "mid": rng.normal(size=3)},
    )
    problem = ik._IKProblem(
        model, list(range(model.njnt)), rot_weight=0.7, equality_weight=3.0
    )
    lower, _ = problem.prepare(targets, ik._seed(model, rng.normal(size=model.nq)))
    assert len(problem._eq_rows) == 1  # the coupling row is part of the residual
    step = 1e-6
    for _ in range(5):
        x = rng.normal(scale=0.8, size=len(lower))  # large rotation-vector steps
        numeric = np.array(
            [
                (problem.residual(x + step * e) - problem.residual(x - step * e))
                / (2 * step)
                for e in np.eye(len(x))
            ]
        ).T
        np.testing.assert_allclose(problem.jacobian(x), numeric, atol=1e-7)


def test_inputs_are_validated_and_the_model_is_untouched() -> None:
    model = _planar()
    qpos0 = model.qpos0.copy()
    with pytest.raises(KeyError):
        solve_ik(model, {"nope": [0, 0, 0]})
    with pytest.raises(KeyError):
        solve_ik(model, {"hand": [0.3, 0, 0]}, joints=["nope"])
    for bad in ([0, 0], [np.nan, 0, 0], ([0, 0, 0], [0, 0, 0, 0]), (None, None)):
        with pytest.raises(ValueError):
            solve_ik(model, {"hand": bad})
    with pytest.raises(ValueError):
        solve_ik(model, {})
    with pytest.raises(ValueError):
        solve_ik(model, {"hand": [0.3, 0, 0]}, qpos=[0.0])
    with pytest.raises(ValueError):
        solve_ik(model, {"hand": [0.3, 0, 0]}, tol=0)
    np.testing.assert_array_equal(model.qpos0, qpos0)


class _PhysicsModel:
    def __init__(self, model: mujoco.MjModel) -> None:
        self.ptr = model

    def site_name2id(self, name: str) -> int:
        return self.ptr.site(name).id


class _Physics:
    """The RoboHive physics interface that qpos_from_site_pose expects."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = _PhysicsModel(model)
        self.data = mujoco.MjData(model)

    def get_handle(self, obj: object) -> object:
        return obj.ptr if isinstance(obj, _PhysicsModel) else obj

    def get_mjlib(self) -> object:
        return mujoco

    def get_state(self) -> dict[str, np.ndarray]:
        return {"qpos": self.data.qpos.copy()}

    def set_state(self, qpos: np.ndarray) -> None:
        self.data.qpos[:] = qpos

    def forward(self) -> None:
        mujoco.mj_forward(self.model.ptr, self.data)


def test_qpos_from_site_pose_is_deprecated_but_still_solves() -> None:
    model = _planar()
    hand, _ = _site_pose(model, np.array([0.3, 1.0]), "hand")
    physics = _Physics(model)
    physics.data.qpos[:] = [0.1, 0.5]
    with pytest.deprecated_call(match="solve_ik"):
        result = ik.qpos_from_site_pose(physics, "hand", target_pos=hand, tol=1e-10)
    assert result.success
    np.testing.assert_allclose(
        _site_pose(model, result.qpos, "hand")[0], hand, atol=1e-9
    )
    np.testing.assert_array_equal(physics.data.qpos, [0.1, 0.5])  # inplace=False
    jac = np.array([[1.0, 0.0], [0.0, 2.0]])
    np.testing.assert_allclose(
        ik.nullspace_method(jac, np.array([1.0, 1.0])), [1.0, 0.5]
    )
