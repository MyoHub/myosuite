# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""myoHandPose* targets are matched to the loaded hand by joint name.

Runs on the released hand (myo_sim 0.2.x) and on the revised hand of
myo_sim >= 0.3; the revised-hand checks skip on the released one.
"""

from __future__ import annotations

import functools
import json

import gymnasium as gym
import mujoco
import numpy as np
import pytest

import myosuite  # noqa: F401
from myosuite import make_env
from myosuite.core.model_recipes import build_from_recipe
from myosuite.envs.myo.tasks.basic.arm import hand_pose_targets as hpt
from myosuite.terms.base_event import fit_pose_target_to_couplings
from myosuite.utils.target_ranges import TargetCoupling

pytestmark = pytest.mark.tier1

BASE_IDS = (
    "myoHandPoseFixed-v0",
    *(f"myoHandPose{k}Fixed-v0" for k in range(10)),
    "myoHandPoseRandom-v0",
)
# Joint limits of the released hand are 1.5708, some targets 1.571.
_LIMIT_TOL = 1e-3


@functools.cache
def _model() -> mujoco.MjModel:
    return build_from_recipe("hand_pose")[0]


def _revised() -> bool:
    names = {_model().joint(i).name for i in range(_model().njnt)}
    return hpt.revised_hand_marker_joint() in names


revised_only = pytest.mark.skipif("not _revised()", reason="needs myo_sim >= 0.3")


def _qposadr(name: str) -> int:
    return int(_model().jnt_qposadr[_model().joint(name).id])


@pytest.mark.parametrize("env_id", BASE_IDS)
def test_targets_match_the_hand_joints_by_name(env_id: str) -> None:
    env = make_env(env_id)
    u = env.unwrapped
    env.reset(seed=0)
    assert len(u.target_jnt_value) == len(hpt.HAND_POSE_JOINTS)
    expected = [_qposadr(n) for n in hpt.HAND_POSE_JOINTS]
    np.testing.assert_array_equal(u.target_qpos_ids, expected)
    if "Fixed" in env_id:
        registered = np.asarray(gym.spec(env_id).kwargs["target_jnt_value"], float)
        np.testing.assert_array_equal(u.target_jnt_value, registered)
    _, _, _, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
    np.testing.assert_allclose(
        info["obs_dict"]["pose_err"], u.target_jnt_value - u.data.qpos[expected]
    )
    env.close()


@pytest.mark.parametrize("env_id", BASE_IDS)
def test_scored_joints_exclude_node_and_cmc45(env_id: str) -> None:
    env = make_env(env_id)
    adr_to_name = {_qposadr(_model().joint(i).name): _model().joint(i).name
                   for i in range(_model().njnt)}
    scored = {adr_to_name[int(a)] for a in env.unwrapped.target_qpos_ids}
    assert scored == set(hpt.HAND_POSE_JOINTS)
    assert not {n for n in scored if n.startswith("node_") or "cmc4" in n or "cmc5" in n}
    env.close()


@pytest.mark.skipif("_revised()", reason="released-hand layout")
@pytest.mark.parametrize("env_id", BASE_IDS)
def test_released_hand_keeps_the_leading_qpos_layout(env_id: str) -> None:
    """On myo_sim 0.2.x the target covers qpos[:23] and nothing is coupled."""
    u = make_env(env_id).unwrapped
    assert u._target_qpos_ids is None and u._target_couplings == ()
    assert u.model.nq == len(hpt.HAND_POSE_JOINTS)


@pytest.mark.parametrize("env_id", BASE_IDS[:-1])
def test_fixed_targets_are_nearly_reachable(env_id: str) -> None:
    """Mapped by name, the fixed targets are within reach of the loaded hand.

    The distance from a target to the hand's joint limits and coupled limits
    stays well below the success threshold (0.7 rad). On the revised hand a few
    ASL targets need more MCP abduction than its flexion-dependent limit
    allows; on the released hand myoHandPoseFixed-v0 asks for 2.6 deg of index
    PIP hyperextension.
    """
    m = _model()
    u = make_env(env_id).unwrapped
    gap = np.zeros(len(u.target_jnt_value))
    for i, (name, value) in enumerate(zip(hpt.HAND_POSE_JOINTS, u.target_jnt_value)):
        lo, hi = m.jnt_range[m.joint(name).id]
        gap[i] = max(lo - value, value - hi - _LIMIT_TOL, 0.0)
    for c in u._target_couplings:
        t = u.target_jnt_value
        value = c.coef * t[c.index] + c.other_coef * t[c.other]
        excess = max(c.lo - value, value - c.hi, 0.0)
        gap[c.index] = max(gap[c.index], excess / abs(c.coef))
    assert np.linalg.norm(gap) < 0.25 < u.pose_thd


def test_active_range_data_is_sourced_and_ordered() -> None:
    data = json.loads(hpt._DATA.read_text())
    assert set(data["ranges"]) == set(hpt.HAND_POSE_JOINTS)
    for name, row in data["ranges"].items():
        lo, hi = row["range"]
        assert lo <= hi, name
        assert row["source"] == "task" or row["source"] in data["sources"], name


@revised_only
def test_active_range_inside_the_revised_joint_limits() -> None:
    m = _model()
    for name, (lo, hi) in hpt.revised_hand_active_range().items():
        jlo, jhi = m.jnt_range[m.joint(name).id]
        assert jlo - 1e-6 <= lo <= hi <= jhi + 1e-6, name  # XML rounds to 1e-6


@revised_only
@pytest.mark.parametrize("env_id", ["myoHandPoseRandom-v0", "myoFatiHandPoseRandom-v0"])
def test_revised_random_targets_are_active_and_reachable(env_id: str) -> None:
    """Random targets: inside the active range, the joint limits and the
    flexion-dependent MCP abduction limits (tendon limits) of the revised hand."""
    env = make_env(env_id)
    u = env.unwrapped
    rom = np.array(list(hpt.revised_hand_active_range().values()))
    np.testing.assert_array_equal(u._target_jnt_range, rom)
    assert len(u._target_couplings) == 8  # two abduction limits per finger
    m, d = u.model, mujoco.MjData(u.model)
    limited = np.flatnonzero(m.tendon_limited)
    env.reset(seed=0)
    at_limit = 0
    for _ in range(500):
        t = u._get_target_pose()
        assert np.all(t >= rom[:, 0] - 1e-12) and np.all(t <= rom[:, 1] + 1e-12)
        d.qpos[:] = m.qpos0
        d.qpos[u.target_qpos_ids] = t
        mujoco.mj_fwdPosition(m, d)
        length = d.ten_length[limited]
        lo, hi = m.tendon_range[limited].T
        assert np.all(length >= lo - 1e-9) and np.all(length <= hi + 1e-9)
        at_limit += int(np.any(np.isclose(length, hi, atol=1e-6)))
    assert at_limit < 25  # remapped, not clipped onto the limit
    env.close()


@revised_only
def test_revised_random_reset_is_deterministic() -> None:
    a, b = make_env("myoHandPoseRandom-v0"), make_env("myoHandPoseRandom-v0")
    oa, _ = a.reset(seed=3)
    ob, _ = b.reset(seed=3)
    np.testing.assert_array_equal(oa, ob)
    np.testing.assert_array_equal(
        a.unwrapped.target_jnt_value, b.unwrapped.target_jnt_value
    )


class _Np:
    @staticmethod
    def array_module():
        return np


def test_fit_to_couplings_remaps_into_the_allowed_interval() -> None:
    # t0 + 0.5 t1 <= 0.5 and -t0 + 0.5 t1 <= 0.5 on the box t0 in [-1, 1].
    couplings = (
        TargetCoupling(0, 1, 1.0, 0.5, -10.0, 0.5),
        TargetCoupling(0, 1, -1.0, 0.5, -10.0, 0.5),
    )
    low, high = (-1.0, 0.0), (1.0, 2.0)
    for t0, t1, expected in [(-1.0, 1.0, 0.0), (1.0, 1.0, 0.0), (0.5, 0.0, 0.25),
                             (-1.0, 0.0, -0.5), (1.0, -1.0, 1.0)]:
        out = fit_pose_target_to_couplings(
            _Np(), np.array([t0, t1]), low, high, couplings
        )
        np.testing.assert_allclose(out, [expected, t1])
