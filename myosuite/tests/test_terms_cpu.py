# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for term functions on CPU/numpy path (Phase 1)."""

from __future__ import annotations


import numpy as np
import pytest


pytestmark = pytest.mark.tier1


class _FakeAccessor:
    """Minimal EnvAccessor stub returning fixed numpy arrays."""

    def __init__(self, nq: int = 4, na: int = 6, nu: int = 4) -> None:
        self._nq = nq
        self._na = na
        self._nu = nu
        self._qpos = np.zeros(nq)
        self._qvel = np.zeros(nq)
        self._act = np.ones(na) * 0.3
        self._ctrl_range = np.column_stack([np.full(nu, -1.0), np.full(nu, 1.0)])

    @property
    def physics_path(self):
        from myosuite.core.protocols import PhysicsPath

        return PhysicsPath.CPU

    def joint_pos(self):
        return self._qpos.copy()

    def joint_vel(self):
        return self._qvel.copy()

    def muscle_act(self):
        return self._act.copy()

    def site_xpos(self, site_ids):
        # Mirror real MuJoCo behaviour: scalar index → (3,); sequence → (n, 3).
        if np.isscalar(site_ids):
            return np.zeros(3)
        return np.zeros((len(site_ids), 3))

    def time(self):
        return 1.0

    def ctrl_range(self):
        return self._ctrl_range.copy()

    def dt(self):
        return 0.01

    def array_module(self):
        return np


# ---------------------------------------------------------------------------
# Observation terms
# ---------------------------------------------------------------------------


def test_joint_pos_obs_shape():
    from myosuite.terms.base_obs import joint_pos_obs

    acc = _FakeAccessor(nq=4)
    obs = joint_pos_obs(acc)
    assert obs.shape == (4,)
    assert obs.dtype in (np.float32, np.float64)


def test_joint_vel_obs_shape():
    from myosuite.terms.base_obs import joint_vel_obs

    acc = _FakeAccessor(nq=4)
    obs = joint_vel_obs(acc)
    assert obs.shape == (4,)


def test_muscle_act_obs_shape():
    from myosuite.terms.base_obs import muscle_act_obs

    acc = _FakeAccessor(na=6)
    obs = muscle_act_obs(acc)
    assert obs.shape == (6,)


def test_pose_error_obs():
    from myosuite.terms.base_obs import pose_error_obs

    acc = _FakeAccessor(nq=4)
    target = np.array([0.1, 0.2, 0.3, 0.4])
    err = pose_error_obs(acc, target=target)
    np.testing.assert_allclose(err, target)  # qpos is zero


def test_tip_pos_obs_shape():
    from myosuite.terms.base_obs import tip_pos_obs

    acc = _FakeAccessor()
    pos = tip_pos_obs(acc, site_ids=[0, 1])
    assert pos.shape == (2, 3)


# ---------------------------------------------------------------------------
# Reward terms
# ---------------------------------------------------------------------------


def test_pose_reward_keys():
    from myosuite.terms.base_reward import pose_reward

    acc = _FakeAccessor(nq=4)
    task_state = {"target_angles": np.zeros(4)}
    result = pose_reward(acc, task_state)
    for key in ("pose", "bonus", "penalty", "dense", "solved", "done"):
        assert key in result, f"Missing key: {key}"


def test_pose_reward_solved_when_at_target():
    from myosuite.terms.base_reward import pose_reward

    acc = _FakeAccessor(nq=4)
    task_state = {"target_angles": np.zeros(4)}
    result = pose_reward(acc, task_state, pose_thd=0.35)
    assert result["solved"], "Should be solved when at zero distance"


def test_pose_reward_not_solved_when_far():
    from myosuite.terms.base_reward import pose_reward

    acc = _FakeAccessor(nq=4)
    acc._qpos = np.ones(4) * 2.0
    task_state = {"target_angles": np.zeros(4)}
    result = pose_reward(acc, task_state, pose_thd=0.35)
    assert not result["solved"]


def test_act_reg_returns_nonpositive():
    from myosuite.terms.base_reward import act_reg

    acc = _FakeAccessor(na=6)
    result = act_reg(acc, {})
    assert result["act_reg"] <= 0.0, "Activation regularisation should be non-positive"


def test_joint_penalty_zero_at_centre():
    from myosuite.terms.base_reward import joint_penalty

    acc = _FakeAccessor(nu=4)
    # qpos is 0, ctrl_range is [-1, 1], well within limits
    result = joint_penalty(acc, {})
    assert result["joint_penalty"] <= 0.0


# ---------------------------------------------------------------------------
# Termination terms
# ---------------------------------------------------------------------------


def test_joint_limit_no_violation():
    from myosuite.terms.base_termination import joint_limit_violation

    acc = _FakeAccessor(nq=4, nu=4)
    # qpos=0, range=[-1,1] — no violation
    result = joint_limit_violation(acc, {})
    assert not result


def test_joint_limit_violation_detected():
    from myosuite.terms.base_termination import joint_limit_violation

    acc = _FakeAccessor(nq=4, nu=4)
    acc._qpos = np.array([2.0, 0.0, 0.0, 0.0])  # outside range
    result = joint_limit_violation(acc, {})
    assert result


# ---------------------------------------------------------------------------
# Event terms
# ---------------------------------------------------------------------------


def test_reset_random_pose_sets_target():
    from myosuite.terms.base_event import reset_random_pose

    acc = _FakeAccessor()
    rng = np.random.default_rng(0)
    target_range = np.column_stack([np.full(4, -0.5), np.full(4, 0.5)])
    task_state = {}
    result = reset_random_pose(acc, task_state, target_jnt_range=target_range, rng=rng)
    assert "target_angles" in result
    assert result["target_angles"].shape == (4,)
    assert np.all(result["target_angles"] >= -0.5)
    assert np.all(result["target_angles"] <= 0.5)


def test_reset_fixed_pose_sets_target():
    from myosuite.terms.base_event import reset_fixed_pose

    acc = _FakeAccessor()
    target = np.array([0.1, 0.2, 0.3, 0.4])
    task_state = {}
    result = reset_fixed_pose(acc, task_state, target_angles=target)
    np.testing.assert_allclose(result["target_angles"], target)


# ---------------------------------------------------------------------------
# Finite-value checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "term_fn,kwargs",
    [
        ("joint_pos_obs", {}),
        ("joint_vel_obs", {}),
        ("muscle_act_obs", {}),
    ],
)
def test_obs_terms_finite(term_fn, kwargs):
    import myosuite.terms.base_obs as obs_mod

    fn = getattr(obs_mod, term_fn)
    acc = _FakeAccessor()
    result = fn(acc, **kwargs)
    assert np.all(np.isfinite(result)), f"{term_fn} produced non-finite values"


# ---------------------------------------------------------------------------
# Reward terms — reach
# ---------------------------------------------------------------------------


def test_reach_reward_keys():
    from myosuite.terms.base_reward import reach_reward

    acc = _FakeAccessor()
    task_state = {"target_pos": np.zeros(3), "tip_site_ids": [0]}
    result = reach_reward(acc, task_state)
    for key in ("reach", "bonus", "dense", "solved", "done"):
        assert key in result, f"Missing key: {key}"


def test_reach_reward_solved_when_at_target():
    from myosuite.terms.base_reward import reach_reward

    acc = _FakeAccessor()
    # site_xpos returns zeros, target_pos is zero → distance 0 → solved
    task_state = {"target_pos": np.zeros(3), "tip_site_ids": [0]}
    result = reach_reward(acc, task_state, reach_thd=0.05)
    assert result["solved"], "Should be solved when tip is at target"


def test_reach_reward_not_solved_when_far():
    from myosuite.terms.base_reward import reach_reward

    acc = _FakeAccessor()
    task_state = {"target_pos": np.array([1.0, 0.0, 0.0]), "tip_site_ids": [0]}
    result = reach_reward(acc, task_state, reach_thd=0.05)
    assert not result["solved"], "Should not be solved when tip is far from target"


def test_reach_reward_multi_site_uses_mean():
    from myosuite.terms.base_reward import reach_reward

    acc = _FakeAccessor()
    # Two sites both at (0,0,0) → mean tip at (0,0,0)
    task_state = {"target_pos": np.zeros(3), "tip_site_ids": [0, 1]}
    result = reach_reward(acc, task_state, reach_thd=0.05)
    assert result["solved"]


# ---------------------------------------------------------------------------
# Reward terms — act_reg dense output
# ---------------------------------------------------------------------------


def test_act_reg_dense_equals_act_reg():
    from myosuite.terms.base_reward import act_reg

    acc = _FakeAccessor(na=6)
    result = act_reg(acc, {})
    # dense must equal act_reg component
    assert float(result["dense"]) == float(result["act_reg"])


# ---------------------------------------------------------------------------
# Termination terms — fall and time limit
# ---------------------------------------------------------------------------


def test_fall_termination_no_site_returns_false():
    from myosuite.terms.base_termination import fall_termination

    acc = _FakeAccessor()
    result = fall_termination(acc, task_state={})
    assert not result, "Should return False when height_site_id is absent"


def test_fall_termination_above_min_height():
    from myosuite.terms.base_termination import fall_termination

    acc = _FakeAccessor()
    # site_xpos returns zeros → height = 0.0, which is < 0.5 → should terminate
    result = fall_termination(acc, task_state={"height_site_id": 0}, min_height=0.5)
    assert result, "Should terminate when site height (0.0) is below min_height (0.5)"


def test_fall_termination_above_threshold():
    from myosuite.terms.base_termination import fall_termination

    class _HighAccessor(_FakeAccessor):
        def site_xpos(self, site_ids):
            pos = np.zeros((1, 3))
            pos[0, 2] = 1.0  # height = 1.0 m
            return pos

    acc = _HighAccessor()
    result = fall_termination(acc, task_state={"height_site_id": 0}, min_height=0.5)
    assert not result, "Should not terminate when site height (1.0) >= min_height (0.5)"


def test_time_limit_not_exceeded():
    from myosuite.terms.base_termination import time_limit

    acc = _FakeAccessor()  # time() returns 1.0
    result = time_limit(acc, task_state={}, max_time=10.0)
    assert not result, "Should not terminate before max_time"


def test_time_limit_exceeded():
    from myosuite.terms.base_termination import time_limit

    class _LateAccessor(_FakeAccessor):
        def time(self):
            return 10.0

    acc = _LateAccessor()
    result = time_limit(acc, task_state={}, max_time=10.0)
    assert result, "Should terminate when simulation time >= max_time"


# ---------------------------------------------------------------------------
# Observation terms — time_obs
# ---------------------------------------------------------------------------


def test_time_obs_returns_scalar():
    from myosuite.terms.base_obs import time_obs

    acc = _FakeAccessor()
    t = time_obs(acc)
    assert float(t) == 1.0, "time_obs should return accessor.time()"


# ---------------------------------------------------------------------------
# joint_limit_violation — margin parameter
# ---------------------------------------------------------------------------


def test_joint_limit_violation_with_margin():
    from myosuite.terms.base_termination import joint_limit_violation

    acc = _FakeAccessor(nq=4, nu=4)
    # qpos=0, range=[-1, 1], margin=1.1 → effective range is [-2.1, 2.1]
    # qpos=0 is inside → no violation
    result = joint_limit_violation(acc, {}, margin=1.1)
    assert not result, "qpos=0 should not violate limits with large margin"


def test_joint_limit_violation_tight_margin():
    from myosuite.terms.base_termination import joint_limit_violation

    acc = _FakeAccessor(nq=4, nu=4)
    # qpos=0, range=[-1,1], margin=-0.1 → effective range is [-0.9, 0.9]
    # qpos=0 is still inside → no violation
    result = joint_limit_violation(acc, {}, margin=-0.1)
    assert not result
