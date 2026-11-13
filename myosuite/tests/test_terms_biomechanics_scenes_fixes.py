# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for terms, biomechanics, running stats, and scene builder fixes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
from myosuite.biomechanics.static_optimization import StaticOptimizer
from myosuite.core.protocols import PhysicsPath
from myosuite.envs.gymnasium_env import CpuEnvAccessor
from myosuite.physics.running_stats import RunningMeanStd, update
from myosuite.scenes.library import SCENES, ObjectSpec, SceneSpec
from myosuite.scenes.scene_builder import (
    get_scene,
    scene_to_mjcf_xml,
    scene_to_mjlab_cfg,
)
from myosuite.terms.base_action import MuscleActionTerm, MuscleActionTermCfg
from myosuite.terms.base_event import reset_random_pose
from myosuite.terms.base_obs import heading_cmd_obs, health_status_obs
from myosuite.terms.base_reward import (
    heading_reward,
    joint_penalty,
    leg_reach_reward,
    multi_site_reach_reward,
    upright_posture_reward,
)
from myosuite.terms.base_termination import (
    joint_limit_violation,
    upright_posture_failure,
)
from myosuite.terms.mimic_reward import (
    mimic_composite_reward,
    mimic_root_orient_reward,
    mimic_root_pos_reward,
    mimic_site_tracking_reward,
)
from myosuite.terms.mimic_track import quat_to_rotvec
from myosuite.terms.multiplayer.chase_tag_vs_obs import own_kinematics_obs
from myosuite.terms.multiplayer.chase_tag_vs_reward import compute_damage, get_score

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


class _BatchedNumpyAccessor:
    def __init__(self, model: mujoco.MjModel, batch_size: int = 3) -> None:
        cpu = CpuEnvAccessor(model, mujoco.MjData(model), ctrl_dt=0.02)
        self._lo, self._hi, self._limited = cpu.joint_limits()
        self.qpos = np.zeros((batch_size, model.nq), dtype=np.float32)
        self.qpos[:, 3] = 1.0
        self.qpos[:, 2] = np.array([1.0, 0.5, 1.0], dtype=np.float32)[:batch_size]
        self.qvel = np.zeros((batch_size, model.nv), dtype=np.float32)
        self.act = np.zeros((batch_size, model.na), dtype=np.float32)
        self.sites = np.zeros((batch_size, model.nsite, 3), dtype=np.float32)
        self._ctrl_range = model.actuator_ctrlrange.copy()

    @property
    def physics_path(self) -> PhysicsPath:
        return PhysicsPath.CPU

    def joint_pos(self) -> np.ndarray:
        return self.qpos

    def joint_vel(self) -> np.ndarray:
        return self.qvel

    def muscle_act(self) -> np.ndarray:
        return self.act

    def site_xpos(self, site_ids: Any) -> np.ndarray:
        return self.sites[:, site_ids, :]

    def time(self) -> np.ndarray:
        return np.zeros(len(self.qpos), dtype=np.float32)

    def ctrl_range(self) -> np.ndarray:
        return self._ctrl_range

    def joint_limits(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._lo, self._hi, self._limited

    def dt(self) -> float:
        return 0.02

    def array_module(self) -> Any:
        return np


def test_joint_limits_terms_handle_free_unlimited_and_muscle_actuators() -> None:
    model = mujoco.MjModel.from_xml_string(_HYBRID_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    acc = CpuEnvAccessor(model, data, ctrl_dt=0.02)

    assert model.nq != model.nu
    lo, hi, limited = acc.joint_limits()
    assert lo.shape == (model.nq,)
    assert limited.tolist() == [False] * 7 + [True, False]

    data.qpos[0] = 50.0
    data.qpos[8] = 25.0
    data.qpos[7] = -0.4
    assert joint_penalty(acc, {})["joint_penalty"] == pytest.approx(0.0)
    assert joint_limit_violation(acc, {}) is False

    data.qpos[7] = 1.25
    assert joint_penalty(acc, {})["joint_penalty"] < 0.0
    assert joint_limit_violation(acc, {}) is True


def test_batched_joint_limit_violation_and_obs_reward_terms() -> None:
    model = mujoco.MjModel.from_xml_string(_HYBRID_MJCF)
    acc = _BatchedNumpyAccessor(model, batch_size=3)
    acc.qpos[1, 7] = 1.5

    viol = joint_limit_violation(acc, {})
    assert isinstance(viol, np.ndarray) and viol.shape == (3,)
    assert viol.tolist() == [False, True, False]

    h_obs = health_status_obs(acc, health_status=1.0)
    assert h_obs.shape == (3, 1)
    np.testing.assert_allclose(h_obs, np.ones((3, 1), dtype=np.float32))

    cmd_obs = heading_cmd_obs(acc, heading_dir=(1.0, 0.0))
    assert cmd_obs.shape == (3, 2)

    h_rew = heading_reward(acc, {}, heading_dir=(1.0, 0.0), target_speed=1.0)
    assert np.shape(h_rew["dense"]) == (3,)
    assert h_rew["done"].tolist() == [False, True, False]

    up_rew = upright_posture_reward(acc, {"upright_posture": [0.95, 0.80, 0.92]})
    assert np.shape(up_rew["dense"]) == (3,)
    up_fail = upright_posture_failure(acc, {"upright_posture": [0.95, 0.80, 0.92]})
    assert up_fail.tolist() == [False, True, False]

    reach_state = {
        "tip_site_ids": [1],
        "target_pos": np.zeros((3, 3), dtype=np.float32),
        "penalty_active": [True, False, True],
    }
    assert multi_site_reach_reward(acc, reach_state)["reach"].shape == (3,)
    assert leg_reach_reward(acc, reach_state)["reach"].shape == (3,)

    comp = mimic_composite_reward(
        np,
        np.zeros((3, 2, 3), dtype=np.float32),
        np.zeros((3, 2, 3), dtype=np.float32),
        None,
        None,
        None,
        None,
        np.zeros((3, 3), dtype=np.float32),
        np.zeros((3, 3), dtype=np.float32),
        None,
        None,
        np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (3, 1)),
        np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (3, 1)),
    )
    assert np.shape(comp["mean_site_dist"]) == (3,)
    np.testing.assert_allclose(comp["root_orient"], np.ones(3, dtype=np.float32))


def test_reset_random_pose_requires_seeded_rng() -> None:
    model = mujoco.MjModel.from_xml_string(_HYBRID_MJCF)
    data = mujoco.MjData(model)
    acc = CpuEnvAccessor(model, data, ctrl_dt=0.02)
    jnt_range = np.array([[-0.5, 0.5]])
    with pytest.raises(ValueError, match="rng"):
        reset_random_pose(acc, {}, target_jnt_range=jnt_range, rng=None)


def test_muscle_action_term_uses_write_ctrl_to_sim_on_numpy_stub() -> None:
    recorded: list[np.ndarray] = []

    class _EntityStub:
        num_actuators = 2

        def write_ctrl_to_sim(self, ctrl: np.ndarray) -> None:
            recorded.append(ctrl)

    env = SimpleNamespace(scene={"robot": _EntityStub()})
    cfg = MuscleActionTermCfg(
        entity_name="robot", normalize=False, muscle_fatigue=False
    )
    term = MuscleActionTerm(cfg, env)
    term.process_actions(np.array([[-0.2, 0.7], [0.4, 1.5]], dtype=np.float32))
    term.apply_actions()
    assert len(recorded) == 1
    np.testing.assert_allclose(recorded[0], [[0.0, 0.7], [0.4, 1.0]])


def test_quaternion_and_distance_terms_at_identity_and_antipodal() -> None:
    q_id = np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
    rv = quat_to_rotvec(np, q_id)
    np.testing.assert_allclose(rv, np.zeros((2, 3)), atol=1e-12)

    # Small perturbation around identity matches first-order 2 * [qx, qy, qz]
    eps = 1e-7
    q_eps = np.array([1.0, eps, -2.0 * eps, 0.5 * eps])
    q_eps /= np.linalg.norm(q_eps)
    np.testing.assert_allclose(
        quat_to_rotvec(np, q_eps),
        2.0 * np.array([eps, -2.0 * eps, 0.5 * eps]),
        rtol=1e-5,
    )

    # Slightly unnormalized float32 drift (> 1.0 dot) must never emit NaN
    q_drift = np.array([[1.0001, 0.0, 0.0, 0.0]], dtype=np.float32)
    r_ori = mimic_root_orient_reward(np, q_drift, q_drift)
    assert np.all(np.isfinite(r_ori))
    np.testing.assert_allclose(r_ori, [1.0], atol=1e-6)

    # 90-degree Z rotation check
    s = np.sqrt(0.5)
    q_90 = np.array([[s, 0.0, 0.0, s]])
    r_90 = mimic_root_orient_reward(
        np, q_90, np.array([[1.0, 0.0, 0.0, 0.0]]), scale=1.0
    )
    np.testing.assert_allclose(r_90, [np.exp(-0.5 * np.pi)], atol=1e-6)

    assert mimic_site_tracking_reward(
        np, np.zeros((2, 3)), np.zeros((2, 3))
    ) == pytest.approx(1.0)
    assert mimic_root_pos_reward(np, np.zeros(3), np.zeros(3)) == pytest.approx(1.0)


def test_chase_tag_evade_damage_score_clamp_and_hybrid_act_indexing() -> None:
    from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_config import (
        ChaseTagVsConfig,
    )

    cfg = ChaseTagVsConfig()
    model = mujoco.MjModel.from_xml_string(_HYBRID_MJCF)
    data = mujoco.MjData(model)
    data.act[:] = [0.25, 0.75]
    data.ctrl[0] = 0.5
    mujoco.mj_forward(model, data)

    meta = SimpleNamespace(
        site_ids={
            "agent_0": {"pelvis_site": model.site("pelvis_s").id},
            "agent_1": {"pelvis_site": model.site("pelvis_s").id},
        },
        jnt_ids={"agent_0": np.array([1], dtype=np.intp)},
        act_indices={"agent_0": np.array([0, 1, 2], dtype=np.intp)},
    )
    dmg = compute_damage(data, meta, cfg, task="EVADE")
    assert dmg["agent_1"] == pytest.approx(cfg.ko_health_threshold)
    assert dmg["agent_0"] == 0.0

    assert get_score(cfg.maxTime + 5.0, "CHASE", cfg) == 0.0
    assert get_score(cfg.maxTime + 5.0, "EVADE", cfg) == 1.0

    obs = own_kinematics_obs(model, data, meta, "agent_0", cfg)
    np.testing.assert_allclose(obs[-3:], np.array([0.5, 0.25, 0.75], dtype=np.float32))


def test_native_mujoco_inverse_kinematics_solves_without_mutating_caller_data() -> None:
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

    res_at_target = qpos_from_site_pose(
        model,
        "ee",
        target_pos=target_pos,
        data=probe,
        inplace=True,
    )
    assert res_at_target.success
    assert res_at_target.steps == 0


def test_static_optimizer_zero_gravity_and_infeasible_qp_graceful_return() -> None:
    model = mujoco.MjModel.from_xml_string(_HYBRID_MJCF)
    model.opt.gravity[:] = 0.0
    opt = StaticOptimizer(model)
    qpos = model.qpos0.copy()
    qvel = np.zeros(model.nv)
    qacc = np.zeros(model.nv)
    res_ok = opt.solve(qpos, qvel, qacc)
    assert np.all(np.isfinite(res_ok.activation))

    opt._solve_qp = lambda A, lower, upper, weight: (  # type: ignore[method-assign]
        np.array([None] * A.shape[1], dtype=object),
        "primal infeasible",
    )
    res_infeas = opt.solve(qpos, qvel, qacc)
    assert str(res_infeas.status) == "primal infeasible"
    assert res_infeas.activation.shape == (model.nu,)
    assert np.all(np.isfinite(res_infeas.activation))


def test_rotation_between_handles_antiparallel_and_zero_vectors() -> None:
    a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    b = np.array(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
    )
    R = _rotation_between(a, b)
    assert R.shape == (4, 3, 3)
    assert np.all(np.isfinite(R))
    for i in range(3):
        np.testing.assert_allclose(R[i] @ a[i], b[i], atol=1e-12)
        np.testing.assert_allclose(R[i] @ R[i].T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R[i]) == pytest.approx(1.0)
    np.testing.assert_allclose(R[3], np.eye(3), atol=1e-12)


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


def test_all_registered_scenes_compile_and_get_scene_returns_deepcopy() -> None:
    s1 = get_scene("cube_reach")
    s1.objects.clear()
    assert len(get_scene("cube_reach").objects) == 1

    for name in SCENES:
        xml = scene_to_mjcf_xml(get_scene(name))
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

    multi = SceneSpec(
        objects=[ObjectSpec(name="cap", shape="capsule", size=0.04, count=3)]
    )
    model_multi = mujoco.MjModel.from_xml_string(scene_to_mjcf_xml(multi))
    assert model_multi.nbody == 4
    assert isinstance(scene_to_mjlab_cfg(multi), dict)
