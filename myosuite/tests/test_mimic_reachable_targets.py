# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Random-target ``myoMimic*-v0``: each episode's targets are the sites of one posture.

The targets were drawn independently per site in one box, so they broke the
skeleton's segment lengths and, on the fixed-base bimanual model, asked the torso
to move: the best attainable mean site error was 0.18 m (full body) and 0.42 m
(bimanual), above the 0.04 m ``solved`` threshold.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.tests.support.joint_couplings import joint_equality_residuals
from myosuite.utils import gym

pytestmark = pytest.mark.tier1

_IDS = ("myoMimicBimanual-v0", "myoMimicFullbody-v0")
_SEEDS = range(12)
# Tracked sites on one rigid chain link (a hip joint centre in the pelvis, an
# elbow centre in the humerus, ...): their distance is the same in every posture.
_RIGID_PAIRS = {
    "myoMimicBimanual-v0": (
        ("left_shoulder_mimic", "left_elbow_mimic"),
        ("right_shoulder_mimic", "right_elbow_mimic"),
    ),
    "myoMimicFullbody-v0": (
        ("pelvis_mimic", "left_hip_mimic"),
        ("pelvis_mimic", "right_hip_mimic"),
        ("left_hip_mimic", "right_hip_mimic"),
        ("upper_body_mimic", "head_mimic"),
        ("left_shoulder_mimic", "left_elbow_mimic"),
        ("right_shoulder_mimic", "right_elbow_mimic"),
        ("left_knee_mimic", "left_ankle_mimic"),
        ("right_knee_mimic", "right_ankle_mimic"),
    ),
}


@pytest.fixture(scope="module", params=_IDS)
def mimic(request: pytest.FixtureRequest) -> gym.Env:
    env = make_env(request.param)
    yield env
    env.close()


def _names(u: gym.Env) -> list[str]:
    return [u.model.site(int(i)).name for i in u._site_ids]


def test_targets_keep_the_skeleton(mimic: gym.Env) -> None:
    """Rigidly linked sites keep their distance; an immovable site keeps its place."""
    u = mimic.unwrapped
    names = _names(u)
    mimic.reset(seed=0)
    rest = u.data.site_xpos[u._site_ids].copy()
    moves = np.zeros(len(names), dtype=bool)
    for seed in _SEEDS:
        mimic.reset(seed=seed)
        target = u._target_site_pos.astype(np.float64)
        for a, b in _RIGID_PAIRS[mimic.spec.id]:
            ia, ib = names.index(a), names.index(b)
            np.testing.assert_allclose(
                np.linalg.norm(target[ia] - target[ib]),
                np.linalg.norm(rest[ia] - rest[ib]),
                atol=1e-6,
                err_msg=f"seed {seed}: {a} - {b}",
            )
        moves |= np.abs(target - rest).max(axis=1) > 1e-3
    if u.model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        assert names[int(np.flatnonzero(~moves)[0])] == "upper_body_mimic"
        assert moves.sum() == len(names) - 1


def test_a_posture_on_the_couplings_matches_every_target(mimic: gym.Env) -> None:
    """The posture the targets came from is within the joint ranges, on the joint
    couplings, and puts every tracked site on its target: ``solved`` is reachable."""
    u = mimic.unwrapped
    m = u.model
    limited = np.flatnonzero((m.jnt_limited != 0) & np.isin(m.jnt_type, (2, 3)))
    adr = m.jnt_qposadr[limited]
    for seed in _SEEDS:
        mimic.reset(seed=seed)
        posture = u._target_qpos[u._target_index]
        assert (posture[adr] >= m.jnt_range[limited, 0] - 1e-9).all()
        assert (posture[adr] <= m.jnt_range[limited, 1] + 1e-9).all()
        assert np.abs(joint_equality_residuals(m, posture)[0]).max() < 1e-9
        u.data.qpos[:], u.data.qvel[:] = posture, 0.0
        rwd = u.forward()[4]["rwd_dict"]
        assert rwd["track_err"] < 1e-6, (seed, rwd["track_err"])
        assert rwd["solved"]


def test_full_body_targets_stand_on_the_floor() -> None:
    """The lowest target is as high as the lowest tracked site of the standing pose."""
    env = make_env("myoMimicFullbody-v0")
    u = env.unwrapped
    try:
        env.reset(seed=0)
        standing = u.data.site_xpos[u._site_ids, 2].min()
        for seed in _SEEDS:
            env.reset(seed=seed)
            assert u._target_site_pos[:, 2].min() == pytest.approx(standing, abs=1e-6)
    finally:
        env.close()


def test_reset_seed_determines_the_targets(mimic: gym.Env) -> None:
    """``reset(seed)`` reproduces the targets, also on a new instance."""
    mimic.reset(seed=5)
    first = mimic.unwrapped._target_site_pos.copy()
    mimic.reset(seed=6)
    assert not np.allclose(mimic.unwrapped._target_site_pos, first)
    mimic.reset(seed=5)
    np.testing.assert_array_equal(mimic.unwrapped._target_site_pos, first)
    other = make_env(mimic.spec.id)
    try:
        other.reset(seed=5)
        np.testing.assert_array_equal(other.unwrapped._target_site_pos, first)
    finally:
        other.close()


def test_target_box_kwargs_are_rejected() -> None:
    with pytest.raises(TypeError, match="mimic_target_low"):
        make_env("myoMimicBimanual-v0", mimic_target_low=(0.0, 0.0, 0.0))
