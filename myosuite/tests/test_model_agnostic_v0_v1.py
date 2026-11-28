# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Verify model-agnostic task envs across myo_sim v0 (0.2.5) and v1 (>=0.3.0)."""

from __future__ import annotations

from pathlib import Path
import tempfile

import mujoco
import numpy as np
import pytest

from myosuite.core.model_builder import build_from_recipe
from myosuite.core.training_recipes import (
    build_anatomical_synergy_map,
    resolve_training_recipe,
)
from myosuite.envs.synergy_wrapper import (
    SynergyResidualActionWrapper,
    TaskOnlyRewardWrapper,
)
from myosuite.envs.tasks.fullbody.waypoint.env import WaypointEnv, WaypointTaskCfg
from myosuite.envs.tasks.leg.walk.env import LegWalkEnvV0
from myosuite.envs.tasks.mimic.tracking.env import MimicEnv
from myosuite.envs.tasks.mimic.tracking.models import (
    MODELS,
    compiled_scene,
)
from myosuite.envs.tasks.mimic.tracking.motion import write_synthetic_clip
from myosuite.physics.compliant_plant import (
    detect_model_generation,
    has_compliant_tendons,
    has_dgf_actuators,
)

pytestmark = pytest.mark.tier2


def _has_v1_myo_sim() -> bool:
    try:
        from myo_sim import compliant as _compliant  # noqa: F401
    except ImportError:
        return False
    return True


def test_leg_walk_default_and_task_only_modes() -> None:
    """LegWalkEnvV0 runs cleanly in both default and task_only modes and with synergy actions."""
    env = LegWalkEnvV0(reward_mode="default", seed=0)
    try:
        obs, _ = env.reset(seed=7)
        assert obs.dtype == np.float32
        assert np.isfinite(obs).all()
        recipe = resolve_training_recipe("walk", env.model)
        assert env.rwd_keys_wt == recipe.reward_weights

        wrapped = TaskOnlyRewardWrapper(SynergyResidualActionWrapper(env))
        act = np.zeros(wrapped.action_space.shape, dtype=np.float32)
        obs2, rwd, term, trunc, info = wrapped.step(act)
        assert obs2.dtype == np.float32
        assert np.isfinite(obs2).all()
        assert "shaped_reward" in info
        assert rwd in (-1.0, 0.0, 1.0)
        assert isinstance(term, bool) and isinstance(trunc, bool)
    finally:
        env.close()


def test_synergy_map_scales_across_models() -> None:
    """build_anatomical_synergy_map groups muscles cleanly for any MjModel."""
    m = compiled_scene("myofullbody", self_collisions=False)
    smap = build_anatomical_synergy_map(m)
    assert smap.expand_matrix.shape == (len(smap.group_names), m.nu)
    assert len(smap.group_names) < m.nu
    np.testing.assert_allclose(smap.expand_matrix.sum(axis=0), np.ones(m.nu))


def test_mimic_env_runs_with_synthetic_and_cross_model_clips() -> None:
    """MimicEnv compiles, resets, equilibrates plant, and steps on native and cross-model clips."""
    candidates = ["myofullbody"]
    if _has_v1_myo_sim():
        candidates.extend(["myofullbody_simple_locked_st", "myofullbody_simple"])

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        subset_source = (
            "myofullbody_simple_locked_st" if _has_v1_myo_sim() else "myofullbody"
        )
        m_subset = compiled_scene(subset_source, False)
        shared_clip = tmp_path / "shared_walk.npz"
        write_synthetic_clip(
            shared_clip,
            m_subset,
            seconds=0.5,
            speed=1.0,
            amp=0.1,
            seed=0,
            model_name=subset_source,
        )

        for model_name in candidates:
            assert model_name in MODELS
            native_clip = tmp_path / f"native_{model_name}.npz"
            write_synthetic_clip(
                native_clip,
                compiled_scene(model_name, False),
                seconds=0.5,
                speed=1.0,
                amp=0.1,
                seed=1,
                model_name=model_name,
            )

            for clip_path, strict_meta in (
                (native_clip, True),
                (shared_clip, False),
            ):
                env = MimicEnv(
                    model=model_name,
                    overrides={
                        "motion.clips": str(clip_path),
                        "motion.strict_refit_meta": strict_meta,
                        "reset.mode": "start",
                    },
                    seed=0,
                )
                try:
                    obs, _ = env.reset(seed=0)
                    assert obs.shape == env.observation_space.shape
                    assert np.isfinite(obs).all()
                    if _has_v1_myo_sim() and not model_name.endswith("_rigid"):
                        assert has_dgf_actuators(env.model)
                        assert has_compliant_tendons(env.model)
                        assert detect_model_generation(env.model) == "v1_compliant"
                        assert env.data.userdata.size >= 3
                        assert np.any(env.data.userdata > 0.0)
                    act = np.full(env.action_space.shape, 0.15, dtype=np.float32)
                    for _ in range(3):
                        obs, rwd, term, trunc, _ = env.step(act)
                        assert np.isfinite(obs).all()
                        assert np.isfinite(rwd)
                        if term or trunc:
                            break
                finally:
                    env.close()


def test_ppo_smoke_walk_and_mimic_with_synergy_wrapper() -> None:
    """Short SB3 PPO smoke run executes cleanly on Walk and Mimic with SynergyResidualActionWrapper."""
    sb3 = pytest.importorskip("stable_baselines3")

    walk_recipe = "myolegs_v1" if _has_v1_myo_sim() else None
    walk_env = SynergyResidualActionWrapper(LegWalkEnvV0(model_recipe=walk_recipe, seed=0))
    try:
        ppo = sb3.PPO(
            "MlpPolicy",
            walk_env,
            n_steps=32,
            batch_size=16,
            n_epochs=2,
            seed=0,
            device="cpu",
            verbose=0,
        )
        ppo.learn(total_timesteps=64)
        obs, _ = walk_env.reset(seed=0)
        action, _ = ppo.predict(obs, deterministic=True)
        assert action.shape == walk_env.action_space.shape
        assert np.isfinite(action).all()
    finally:
        walk_env.close()

    mimic_model = "myofullbody_simple_locked_st" if _has_v1_myo_sim() else "myofullbody"
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "smoke.npz"
        write_synthetic_clip(
            clip,
            compiled_scene(mimic_model, False),
            seconds=1.0,
            speed=1.0,
            amp=0.05,
            seed=0,
            model_name=mimic_model,
        )
        mimic_env = SynergyResidualActionWrapper(
            MimicEnv(
                model=mimic_model,
                overrides={
                    "motion.clips": str(clip),
                    "reset.mode": "start",
                },
                seed=0,
            )
        )
        try:
            ppo_m = sb3.PPO(
                "MlpPolicy",
                mimic_env,
                n_steps=32,
                batch_size=16,
                n_epochs=2,
                seed=0,
                device="cpu",
                verbose=0,
            )
            ppo_m.learn(total_timesteps=64)
            obs, _ = mimic_env.reset(seed=0)
            action, _ = ppo_m.predict(obs, deterministic=True)
            assert action.shape == mimic_env.action_space.shape
            assert np.isfinite(action).all()
        finally:
            mimic_env.close()


@pytest.mark.skipif(not _has_v1_myo_sim(), reason="Requires myo_sim >= 0.3.0")
def test_v1_walk_and_waypoint_across_simple_and_fullbody() -> None:
    """LegWalkEnvV0 and WaypointEnv run with exact compliant forces on v1 models."""
    from myo_sim import compliant

    for recipe_name in ("myolegs_v1", "myofullbody_simple_locked_st", "myofullbody_simple"):
        env = LegWalkEnvV0(model_recipe=recipe_name, seed=3)
        try:
            obs, _ = env.reset(seed=3)
            assert np.isfinite(obs).all()
            assert detect_model_generation(env.model) == "v1_compliant"

            m_ref, _ = build_from_recipe(recipe_name)
            d_ref = mujoco.MjData(m_ref)
            d_ref.qpos[:] = env.data.qpos
            d_ref.qvel[:] = env.data.qvel
            d_ref.act[:] = env.data.act
            p_ref = compliant.CompliantTendons(m_ref, install=True)
            compliant.forward(m_ref, d_ref, p_ref)
            np.testing.assert_allclose(
                env.data.actuator_force, d_ref.actuator_force, atol=1e-4
            )

            syn_env = SynergyResidualActionWrapper(env)
            obs2, rwd, _, _, _ = syn_env.step(
                np.zeros(syn_env.action_space.shape, dtype=np.float32)
            )
            assert np.isfinite(obs2).all() and np.isfinite(rwd)
        finally:
            env.close()

    wp_env = WaypointEnv(
        model_recipe="myofullbody_simple_locked_st",
        task=WaypointTaskCfg(waypoints=((0.0, -1.5),)),
        seed=5,
    )
    try:
        obs, _ = wp_env.reset(seed=5)
        assert np.isfinite(obs).all()
        obs2, rwd, _, _, info = wp_env.step(
            np.full(wp_env.action_space.shape, 0.1, dtype=np.float32)
        )
        assert np.isfinite(obs2).all() and np.isfinite(rwd)
        assert "distance" in info
    finally:
        wp_env.close()
