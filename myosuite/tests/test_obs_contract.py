# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Step contract of the CPU envs: ``MyoGymnasiumEnv`` and its ``step()`` overrides.

The registry-wide dtype / bounds / obs-freshness checks live in
``test_registry.py``.
"""

from __future__ import annotations

import random

import mujoco
import numpy as np
import pytest

from myosuite.envs.gymnasium_env import MyoGymnasiumEnv
from myosuite import make_env

pytestmark = pytest.mark.tier1

# One env id per MyoGymnasiumEnv class with its own step(), plus the base step().
STEP_OVERRIDE_IDS = (
    "myoElbowPose1D6MRandom-v0",  # arm.pose.PoseEnvV0
    "myoHandReachFixed-v0",  # arm.reach.ReachEnvV0
    "myoHandKeyTurnFixed-v0",  # arm.key_turn.KeyTurnEnvV0
    "myoHandObjHoldFixed-v0",  # hand.obj_hold
    "myoHandPenTwirlFixed-v0",  # hand.pen
    "myoTorsoPoseFixed-v0",  # torso.pose.TorsoEnvV0
    "myoLegStandRandom-v0",  # leg.reach.LegReachEnvV0
    "myoLegWalk-v0",  # leg.walk.LegWalkEnvV0
    "myoChallengeBaodingP1-v1",
    "myoChallengeBimanual-v0",
    "myoChallengeChaseTagP1-v0",
    "myoChallengeDieReorientP1-v0",
    "myoChallengeOslRunFixed-v0",
    "myoChallengeRelocateP1-v0",
    "myoChallengeSoccerP1-v0",
    "myoChallengeTableTennisP0-v0",
    "myoElbowPoseTaskFixed-v0",  # ModularTaskEnv
    "myoHandReorient8-v0",  # base MyoGymnasiumEnv.step
)


@pytest.mark.parametrize(
    "env_id", [e for e in STEP_OVERRIDE_IDS if e != "myoElbowPoseTaskFixed-v0"]
)
def test_unknown_obs_key_raises(env_id: str) -> None:
    """An obs key the env does not compute fails loudly instead of being dropped."""
    with pytest.raises(KeyError, match="no_such_obs_key"):
        make_env(env_id, obs_keys=["no_such_obs_key"])


def test_relocate_hand_obs_cover_every_hand_joint() -> None:
    """hand_qpos/hand_qvel hold every joint except the object's (md5_flexion_r last)."""
    env = make_env("myoChallengeRelocateP1-v0").unwrapped
    try:
        env.reset(seed=0)
        m = env.model
        hand = [j for j in range(m.njnt) if m.jnt_bodyid[j] != env.object_bid]
        assert m.joint(hand[-1]).name == "md5_flexion_r"
        obs = env._get_obs_dict(env._accessor)
        np.testing.assert_array_equal(
            obs["hand_qpos"], env.data.qpos[m.jnt_qposadr[hand]]
        )
        np.testing.assert_array_equal(
            obs["hand_qvel"], env.data.qvel[m.jnt_dofadr[hand]] * env._ctrl_dt
        )
        np.testing.assert_array_equal(obs["hand_qpos_corrected"], obs["hand_qpos"])
    finally:
        env.close()


@pytest.mark.parametrize("env_id", STEP_OVERRIDE_IDS)
def test_step_keeps_base_contract(env_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every step() renders on request and rejects a reward dict without "done"."""
    env = make_env(env_id).unwrapped
    try:
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)

        frames: list[int] = []
        monkeypatch.setattr(env, "mj_render", lambda: frames.append(1))
        env.mujoco_render_frames = True
        env.step(action)
        assert frames == [1]
        env.mujoco_render_frames = False

        reward_dict = env.get_reward_dict
        monkeypatch.setattr(
            env,
            "get_reward_dict",
            lambda obs_dict: {
                k: v for k, v in reward_dict(obs_dict).items() if k != "done"
            },
        )
        with pytest.raises(KeyError, match="done"):
            env.step(action)
    finally:
        env.close()


# MjData fields that mj_step leaves one substep stale and mj_forward recomputes.
DERIVED_FIELDS = (
    "xpos",
    "xquat",
    "site_xpos",
    "cvel",
    "subtree_com",
    "ten_length",
    "actuator_length",
    "actuator_velocity",
    "actuator_force",
    "qfrc_actuator",
    "sensordata",
)


@pytest.mark.parametrize("env_id", STEP_OVERRIDE_IDS)
def test_step_leaves_derived_quantities_fresh(env_id: str) -> None:
    """After step(), an extra mj_forward changes no derived quantity or contact."""
    env = make_env(env_id).unwrapped
    try:
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(3):
            env.step(rng.uniform(env.action_space.low, env.action_space.high))
            ref = mujoco.MjData(env.model)
            mujoco.mj_copyData(ref, env.model, env.data)
            mujoco.mj_forward(env.model, ref)
            for field in DERIVED_FIELDS:
                np.testing.assert_array_equal(
                    getattr(env.data, field), getattr(ref, field), err_msg=field
                )
            assert env.data.ncon == ref.ncon
            np.testing.assert_array_equal(
                env.data.contact.pos[: ref.ncon], ref.contact.pos[: ref.ncon]
            )
    finally:
        env.close()


# Pose, reach, walk (flat and height field), torso, reorient and challenge envs
# with contacts, covering every step() path (n substeps in one call included).
# OslRun is left out on purpose: its OSL controller reads the prosthesis load
# sensor, so a fresh reading legitimately changes the prosthesis torques.
DYNAMICS_IDS = (
    "myoElbowPose1D6MRandom-v0",
    "myoTorsoPoseFixed-v0",
    "myoHandReachRandom-v0",
    "myoLegWalk-v0",
    "myoLegRoughTerrainWalk-v0",
    "myoLegDirectionalForward-v0",  # ModularTaskEnv
    "myoHandReorient8-v0",
    "myoChallengeBaodingP1-v1",
    "myoChallengeBimanual-v0",
    "myoChallengeChaseTagP1-v0",
    "myoChallengeRelocateP1-v0",
    "myoChallengeSoccerP1-v0",
    "myoChallengeTableTennisP0-v0",
)


def _kinematics_only_step_physics(
    self: MyoGymnasiumEnv, nstep: int | None = None
) -> None:
    """The former refresh: positions only, so other derived quantities stay stale."""
    mujoco.mj_step(self.model, self.data, self.frame_skip if nstep is None else nstep)
    mujoco.mj_kinematics(self.model, self.data)


def _state_trajectory(env_id: str, n_steps: int = 40) -> np.ndarray:
    """time, qpos, qvel, act and mocap poses after each of ``n_steps`` fixed actions."""
    np.random.seed(0)
    random.seed(0)
    env = make_env(env_id).unwrapped
    try:
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        states = []
        for _ in range(n_steps):  # termination ignored: keep simulating
            env.step(rng.uniform(env.action_space.low, env.action_space.high))
            d = env.data
            states.append(
                np.concatenate([[d.time], d.qpos, d.qvel, d.act, d.mocap_pos.ravel()])
            )
        return np.stack(states)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", DYNAMICS_IDS)
def test_fresh_derived_quantities_leave_dynamics_unchanged(
    env_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The state trajectory is bit-identical with mj_forward or mj_kinematics after mj_step.

    Only derived quantities (observations, rewards) may change.
    """
    fresh = _state_trajectory(env_id)
    monkeypatch.setattr(MyoGymnasiumEnv, "_step_physics", _kinematics_only_step_physics)
    stale = _state_trajectory(env_id)
    np.testing.assert_array_equal(fresh, stale)
