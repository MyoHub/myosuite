# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Smoke tests for the directional myoLeg and 1v1 chase-tag environments."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from myosuite.tests.support.optional_deps import require_musclemimic_models

pytestmark = pytest.mark.tier1


def test_chase_tag_vs_config_fields_are_subset_of_task_config() -> None:
    """``ChaseTagVsConfig``'s fields must stay a subset of ``ChaseTagVsTaskConfig``'s.

    ``ChaseTagVsTaskConfig._low_level_config()`` projects itself onto
    ``ChaseTagVsConfig`` generically (by matching field names), rather than
    listing every field by hand. This test is the guard that keeps that
    projection valid: if someone adds a field to ``ChaseTagVsConfig`` without
    a same-named field on ``ChaseTagVsTaskConfig``, the projection would
    raise a ``TypeError`` at construction time -- this test catches it at
    collection time instead.
    """
    import dataclasses

    from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_config import (
        ChaseTagVsConfig,
    )
    from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_task_config import (
        ChaseTagVsTaskConfig,
    )

    low_level_fields = {f.name for f in dataclasses.fields(ChaseTagVsConfig)}
    task_config_fields = {f.name for f in dataclasses.fields(ChaseTagVsTaskConfig)}
    missing = low_level_fields - task_config_fields
    assert not missing, (
        f"ChaseTagVsConfig fields {missing} have no matching field on "
        "ChaseTagVsTaskConfig; _low_level_config()'s generic projection "
        "would fail."
    )


class TestLegDirectionalRegistry:
    """Smoke tests for the single-agent directional myoLeg locomotion tasks."""

    @pytest.mark.parametrize(
        "env_id", ["myoLegDirectionalForward-v0", "myoLegDirectionalBackward-v0"]
    )
    def test_registered_and_runs(self, env_id: str) -> None:
        import myosuite

        myosuite.register_all_envs()
        assert gym.spec(env_id) is not None

        env = gym.make(env_id)
        try:
            obs, info = env.reset(seed=0)
            assert env.observation_space.contains(obs)
            for _ in range(5):
                action = env.action_space.sample()
                obs, rwd, terminated, truncated, info = env.step(action)
                assert np.isfinite(rwd)
                assert env.observation_space.contains(obs)
        finally:
            env.close()


class TestFullBodyChaseTagRegistry:
    """Smoke tests for myoChallengeChaseTagFBP2-v0 (full-body vs. scripted opponent)."""

    ENV_ID = "myoChallengeChaseTagFBP2-v0"

    def test_registered_and_runs(self) -> None:
        import myosuite
        from myosuite.envs.myo.tasks.challenge.chase_tag_fb_model import (
            CHASETAG_FB_ARENA_BYTES,
        )

        myosuite.register_all_envs()
        assert gym.spec(self.ENV_ID) is not None

        env = gym.make(self.ENV_ID)
        try:
            model = env.unwrapped.model
            assert model.na == 354, f"expected 354 full-body muscles, got {model.na}"
            assert model.nu == 354
            assert model.narena == CHASETAG_FB_ARENA_BYTES
            assert model.body("opponent").id is not None
            assert env.unwrapped._pelvis_body_name == "pelvis"
            env.unwrapped.model.body(env.unwrapped._pelvis_body_name)  # resolves ok

            obs, info = env.reset(seed=0)
            assert env.observation_space.contains(obs)
            for _ in range(30):
                action = env.action_space.sample()
                obs, rwd, terminated, truncated, info = env.step(action)
                assert np.isfinite(rwd)
                assert np.isfinite(obs).all()
        finally:
            env.close()

    def test_arena_fits_every_contact_and_limit_active(self) -> None:
        """The explicit arena replaces the 1.3 GB legacy one with >= 10x headroom.

        The worst case puts the body inside a 0.6-1 m tall hfield block (the
        terrain at its non-FLAT position) with 5 m contact margins and 10 rad
        joint margins, so every body geom touches the floor and the terrain and
        both sides of every joint limit are active (~5 MiB used).
        """
        import mujoco

        from myosuite.envs.myo.tasks.challenge.chase_tag_fb_model import (
            CHASETAG_FB_ARENA_BYTES,
            build_fullbody_chasetag_spec,
        )

        spec = build_fullbody_chasetag_spec()
        model = spec.compile()
        assert model.narena == CHASETAG_FB_ARENA_BYTES
        # Legacy sizes stay on the model for the MJX Warp path.
        assert (model.nconmax, model.njmax) == (spec.nconmax, spec.njmax)

        for geom in spec.geoms:
            if geom.contype or geom.conaffinity:
                geom.margin = 5.0
        for pair in spec.pairs:
            pair.margin = 5.0
        for joint in spec.joints:
            if joint.type != mujoco.mjtJoint.mjJNT_FREE:
                joint.margin = 10.0
        worst = spec.compile()
        assert worst.narena == CHASETAG_FB_ARENA_BYTES
        worst.geom_pos[worst.geom("terrain").id] = 0.0
        rng = np.random.default_rng(0)
        worst.hfield_data[:] = rng.uniform(0.6, 1.0, worst.hfield_data.size)
        colliders = (worst.geom_contype | worst.geom_conaffinity) > 0
        world = np.flatnonzero(colliders & (worst.geom_bodyid == 0)).tolist()
        body = np.flatnonzero(colliders & (worst.geom_bodyid > 0)).tolist()
        hinges = worst.jnt_type != mujoco.mjtJoint.mjJNT_FREE
        n_limits = int(worst.jnt_limited[hinges].sum())
        data = mujoco.MjData(worst)
        for _ in range(3):
            mujoco.mj_resetData(worst, data)
            quat = rng.normal(size=4)
            data.qpos[3:7] = quat / np.linalg.norm(quat)
            data.qpos[2] = 0.45
            # mj_forward: mj_step would auto-reset (and zero maxuse) on a bad qacc.
            mujoco.mj_forward(worst, data)
            touching = {
                tuple(sorted(g)) for g in data.contact.geom[: data.ncon].tolist()
            }
            assert all((w, b) in touching for w in world for b in body)
            efc_type = data.efc_type[: data.nefc]
            limit_rows = efc_type == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
            assert int(limit_rows.sum()) == 2 * n_limits
            for warning in (
                mujoco.mjtWarning.mjWARN_CONTACTFULL,
                mujoco.mjtWarning.mjWARN_CNSTRFULL,
            ):
                assert data.warning[warning].number == 0
            assert 10 * data.maxuse_arena <= worst.narena


class TestChaseTagVsRegistry:
    """Smoke tests for myoChallengeChaseTagFBVs-v0 (fullbody steered variant)."""

    ENV_ID = "myoChallengeChaseTagFBVs-v0"

    def test_registered_and_runs(self) -> None:
        require_musclemimic_models()
        import myosuite

        myosuite.register_all_envs()
        assert gym.spec(self.ENV_ID) is not None

        try:
            env = gym.make(self.ENV_ID)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Gait reference clip unavailable (no network?): {exc}")
            return
        try:
            obs, info = env.reset(seed=0)
            assert set(obs.keys()) == {"agent_0", "agent_1"}
            for _ in range(5):
                actions = {a: env.unwrapped.action_space[a].sample() for a in obs}
                obs, rewards, terminated, truncated, info = env.step(actions)
                for agent_id in ("agent_0", "agent_1"):
                    assert np.isfinite(rewards[agent_id])
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Gait reference clip unavailable (no network?): {exc}")
        finally:
            env.close()

    def test_chaser_catching_runner_increases_runner_health(self) -> None:
        """Driving the chaser onto the runner's pelvis should accrue tag pressure."""
        import myosuite
        from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_task_config import (
            ChaseTagVsTaskConfig,
        )
        from myosuite.envs.multi_agent_modular_env import ModularMultiAgentTaskEnv

        myosuite.register_all_envs()
        env = ModularMultiAgentTaskEnv(ChaseTagVsTaskConfig(agent_separation_m=0.5))
        obs, info = env.reset(seed=0)
        zero_actions = {
            a: np.zeros(env.action_space[a].shape, dtype=np.float32) for a in obs
        }
        for _ in range(20):
            obs, rewards, terminated, truncated, info = env.step(zero_actions)
        # Info should contain MyoChallenge-compatible fields.
        assert "task" in info
        assert info["task"] in ("CHASE", "EVADE")
        assert "elapsed_s" in info
        assert "score" in info
        env.close()
