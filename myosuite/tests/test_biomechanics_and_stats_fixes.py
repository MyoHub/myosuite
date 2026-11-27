# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for legacy IK native MuJoCo port, retargeting, static opt, and running stats."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from myosuite.biomechanics.inverse_kinematics import qpos_from_site_pose
from myosuite.biomechanics.retarget import (
    ClipTargets,
    Landmark,
    LandmarkIK,
    LandmarkMap,
    save_refit,
    stamp_fingerprint,
    validate_refit,
)
from myosuite.biomechanics.retarget_smplh import _rotation_between
from myosuite.biomechanics.static_optimization import (
    StaticOptimizationConfig,
    StaticOptimizationError,
    StaticOptimizationWarning,
    StaticOptimizer,
)
from myosuite.physics.running_stats import RunningMeanStd, update

pytestmark = pytest.mark.tier1


_HYBRID_MJCF = """
<mujoco model="hybrid_test">
  <compiler angle="radian"/>
  <worldbody>
    <body name="root" pos="0 0 1">
      <freejoint name="root_free"/>
      <geom type="sphere" size="0.05" mass="1.0"/>
      <site name="pelvis_s" pos="0 0 0"/>
      <body name="link1" pos="0 0 0">
        <joint name="j_hinge" type="hinge" axis="0 1 0" limited="true" range="-1.0 1.0"/>
        <geom type="capsule" size="0.04 0.2" mass="1.0"/>
        <site name="tip" pos="0 0 -0.2"/>
        <body name="link2" pos="0 0 -0.4">
          <joint name="j_unlimited" type="hinge" axis="0 1 0" limited="false"/>
          <geom type="capsule" size="0.03 0.15" mass="0.5"/>
          <site name="ee" pos="0 0 -0.15"/>
        </body>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial name="t1">
      <site site="pelvis_s"/>
      <site site="tip"/>
    </spatial>
    <spatial name="t2">
      <site site="tip"/>
      <site site="ee"/>
    </spatial>
  </tendon>
  <actuator>
    <motor name="m_root" joint="j_unlimited" ctrlrange="-10 10"/>
    <muscle name="musc1" tendon="t1" lengthrange="0.1 0.3"/>
    <muscle name="musc2" tendon="t2" lengthrange="0.1 0.3"/>
  </actuator>
</mujoco>
"""


def test_native_mujoco_qpos_from_site_pose_without_mutating_caller_data() -> None:
    from myosuite.physics.inverse_kinematics import qpos_from_site_pose as legacy_ik

    assert legacy_ik is qpos_from_site_pose
    model = mujoco.MjModel.from_xml_string(_HYBRID_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    qpos_before = data.qpos.copy()

    probe = mujoco.MjData(model)
    probe.qpos[:] = data.qpos
    probe.qpos[7] = 0.45
    probe.qpos[8] = -0.35
    mujoco.mj_fwdPosition(model, probe)
    ee_id = model.site("ee").id
    target_pos = probe.site_xpos[ee_id].copy()
    target_quat = np.empty(4)
    mujoco.mju_mat2Quat(target_quat, probe.site_xmat[ee_id])

    with pytest.warns(DeprecationWarning):
        res = qpos_from_site_pose(
            (model, data),
            "ee",
            target_pos=target_pos,
            target_quat=target_quat,
            joint_names=["j_hinge", "j_unlimited"],
            tol=1e-9,
            inplace=False,
        )
    assert res.success
    assert res.err_norm < 1e-9
    np.testing.assert_array_equal(data.qpos, qpos_before)

    with pytest.warns(DeprecationWarning):
        res_at_target = qpos_from_site_pose(
            model,
            "ee",
            target_pos=target_pos,
            data=probe,
            inplace=True,
        )
    assert res_at_target.success
    assert res_at_target.steps == 0


def test_static_optimizer_zero_gravity_and_infeasible_qp_failure_policy() -> None:
    model = mujoco.MjModel.from_xml_string(_HYBRID_MJCF)
    model.opt.gravity[:] = 0.0
    opt = StaticOptimizer(model)
    qpos = model.qpos0.copy()
    qvel = np.zeros(model.nv)
    qacc = np.zeros(model.nv)
    res_ok = opt.solve(qpos, qvel, qacc)
    assert np.all(np.isfinite(res_ok.activation))

    def stub_qp(A, lower, upper, weight):
        return np.array([None] * A.shape[1], dtype=object), SimpleNamespace(
            status="primal infeasible", iter=7, prim_res=np.inf, dual_res=np.inf
        )

    opt_raise = StaticOptimizer(model, StaticOptimizationConfig(on_failure="raise"))
    opt_raise._solve_qp = stub_qp  # type: ignore[method-assign]
    with pytest.raises(StaticOptimizationError, match="primal infeasible"):
        opt_raise.solve(qpos, qvel, qacc)

    opt._solve_qp = stub_qp  # type: ignore[method-assign]
    with pytest.warns(StaticOptimizationWarning, match="primal infeasible"):
        res_infeas = opt.solve(qpos, qvel, qacc)
    assert str(res_infeas.status) == "primal infeasible"
    assert res_infeas.iterations == 7
    assert res_infeas.activation.shape == (model.nu,)
    assert np.all(np.isfinite(res_infeas.activation))


def test_rotation_between_handles_antiparallel_and_zero_vectors() -> None:
    a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    b = np.array(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
    )
    rot = _rotation_between(a, b)
    assert rot.shape == (4, 3, 3)
    assert np.all(np.isfinite(rot))
    for i in range(3):
        np.testing.assert_allclose(rot[i] @ a[i], b[i], atol=1e-12)
        np.testing.assert_allclose(rot[i] @ rot[i].T, np.eye(3), atol=1e-12)
        assert np.linalg.det(rot[i]) == pytest.approx(1.0)
    np.testing.assert_allclose(rot[3], np.eye(3), atol=1e-12)


def test_retarget_validate_uncoupled_and_atomic_save_stamp(tmp_path: Path) -> None:
    model = mujoco.MjModel.from_xml_string(_HYBRID_MJCF)
    lmap = LandmarkMap(
        name="hybrid",
        version="1",
        landmarks=(Landmark(source="tip", body="link1", site="tip"),),
        orientations=(),
        feet=((), ()),
    )
    ik = LandmarkIK(model, lmap)
    qpos = np.tile(model.qpos0, (4, 1))
    pos = np.stack([ik.landmarks(q) for q in qpos])
    targets = ClipTargets(
        frequency=50.0,
        pos=pos,
        quat=np.zeros((4, 0, 4)),
        seed=model.qpos0.copy(),
        contact=np.zeros((4, 2), dtype=bool),
    )
    metrics = validate_refit(ik, targets, qpos)
    assert metrics["dependent_limit_violations"] == 0
    assert metrics["flags"] == []

    out = tmp_path / "clip.npz"
    save_refit(out, model, qpos, 50.0, {"model": "hybrid_test", "model_version": "1.0"})
    assert stamp_fingerprint(out, "hybrid_test", "1.0", "abcd1234abcd1234") is True
    assert list(tmp_path.glob("*.tmp.npz")) == []


def test_running_stats_empty_batch_is_noop() -> None:
    stats = RunningMeanStd.zeros(4)
    same = update(stats, np.empty((0, 4)))
    assert same.count == 0
    assert np.all(np.isfinite(same.mean))
    assert np.all(np.isfinite(same.var))
