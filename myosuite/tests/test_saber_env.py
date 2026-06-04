# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for myoChallengeSaberP0-v0 CPU prototype task."""

from __future__ import annotations

import numpy as np
import pytest

from myosuite.tests.support.optional_deps import require_musclemimic_models


def test_saber_env_spec_points_to_stable_env_id():
    from myosuite.envs.myo.tasks.challenge.saber_task_spec import (
        SABER_P0_ENV_ID,
        get_saber_p0_env_spec,
    )

    env_spec = get_saber_p0_env_spec()
    assert env_spec.env_id == SABER_P0_ENV_ID
    assert env_spec.env_id == "myoChallengeSaberP0-v0"


@pytest.fixture(scope="module")
def saber_env():
    """Registered saber environment fixture."""
    require_musclemimic_models()
    from myosuite import register_all_envs

    register_all_envs()
    import gymnasium as gym

    env = gym.make("myoChallengeSaberP0-v0")
    yield env
    env.close()


class TestSaberEnvBasic:
    """Basic reset/step and info signal checks."""

    def test_reset_returns_obs_info(self, saber_env):
        obs, info = saber_env.reset(seed=0)
        assert obs.ndim == 1
        assert obs.shape[0] > 0
        assert isinstance(info, dict)

    def test_reset_publishes_target_pool_preference_metadata(self, saber_env):
        saber_env.reset(seed=0)
        task_state = saber_env.unwrapped._task_state  # pylint: disable=protected-access
        assert "target_pool_hit_area" in task_state
        assert "target_pool_preferred_hand" in task_state
        assert "health_status" in task_state
        assert "upright_posture" in task_state
        assert "target_pool_hit_y_at_hit" in task_state
        assert "target_pool_hit_y_deviation_at_hit" in task_state
        assert task_state["target_pool_hit_area"].shape == (100,)
        assert task_state["target_pool_preferred_hand"].shape == (100,)
        assert task_state["health_status"] == pytest.approx(0.5)
        assert task_state["upright_posture"] == pytest.approx(1.0, abs=1e-5)
        assert task_state["target_pool_hit_y_at_hit"] == 0.0
        assert task_state["target_pool_hit_y_deviation_at_hit"] == 0.0

    def test_step_returns_5_tuple(self, saber_env):
        saber_env.reset(seed=0)
        action = saber_env.action_space.sample()
        obs, rew, term, trunc, info = saber_env.step(action)
        assert obs.ndim == 1
        assert isinstance(rew, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

    def test_info_has_hit_keys(self, saber_env):
        saber_env.reset(seed=1)
        _, _, _, _, info = saber_env.step(saber_env.action_space.sample())
        assert "saber_target_pool/hit_this_step" in info
        assert "saber_target_pool/hits_total" in info
        assert "saber_target_pool/total_spawned" in info
        assert "saber_target_pool/wrong_hand_this_step" in info
        assert "saber_target_pool/wrong_face_this_step" in info

    def test_saber_effective_step_timing_matches_ctrl_dt(self, saber_env):
        unwrapped = saber_env.unwrapped
        effective_dt = float(unwrapped.frame_skip) * float(unwrapped.model.opt.timestep)
        expected_dt = float(unwrapped._task_config.backend.ctrl_dt)  # pylint: disable=protected-access
        assert effective_dt == pytest.approx(expected_dt, abs=1e-10)


class TestSaberModel:
    """Model-level checks for MyoTorso bimanual + saber pool augmentation."""

    def test_uses_myotorso_bimanual_bodies(self, saber_env):
        import mujoco

        model = saber_env.unwrapped.model
        mjt = mujoco.mjtObj
        assert mujoco.mj_name2id(model, mjt.mjOBJ_BODY, "lunate_l") != -1
        assert mujoco.mj_name2id(model, mjt.mjOBJ_BODY, "lunate_r") != -1
        assert mujoco.mj_name2id(model, mjt.mjOBJ_BODY, "thorax") != -1
        assert mujoco.mj_name2id(model, mjt.mjOBJ_BODY, "femur_r") != -1
        # Fingers on + torso/back + leg muscles (rigid leg DOFs; no pelvis slides).
        assert int(model.nu) == 336

    def test_has_saber_geoms_and_sites(self, saber_env):
        import mujoco

        model = saber_env.unwrapped.model
        for geom_name in ("left_saber_geom", "right_saber_geom"):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            assert gid != -1, f"Missing saber geom: {geom_name}"
        for geom_name in ("left_blade_aura", "right_blade_aura"):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            assert gid != -1, f"Missing saber aura geom: {geom_name}"
        for i in (0, 50, 99):
            gid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"saber_target_geom_{i}"
            )
            assert gid != -1, f"Missing pool sphere geom at index {i}"
        for site_name in ("left_saber_tip", "right_saber_tip", "saber_target_0_site"):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            assert sid != -1, f"Missing required site: {site_name}"

    def test_saber_target_contacts_are_detectable_but_solver_inactive(self, saber_env):
        import mujoco

        unwrapped = saber_env.unwrapped
        model = unwrapped.model
        data = unwrapped.data
        saber_env.reset(seed=0)

        target_gid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "saber_target_geom_0"
        )
        saber_gid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "left_saber_geom"
        )
        target_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "saber_target_0"
        )
        target_mid = int(model.body_mocapid[target_bid])

        data.mocap_pos[target_mid] = data.geom_xpos[saber_gid].copy()
        mujoco.mj_forward(model, data)

        target_saber_contacts = []
        for k in range(int(data.ncon)):
            contact = data.contact[k]
            pair = {int(contact.geom1), int(contact.geom2)}
            if target_gid in pair and saber_gid in pair:
                target_saber_contacts.append(contact)

        assert target_saber_contacts, "expected overlapping target/saber contact"
        assert float(model.geom_margin[target_gid]) == pytest.approx(0.0)
        assert float(model.geom_gap[target_gid]) > 0.0
        assert all(int(contact.efc_address) == -1 for contact in target_saber_contacts)

    def test_saber_tip_linvel_sensors_exist(self, saber_env):
        import mujoco

        model = saber_env.unwrapped.model
        for name in ("left_saber_tip_linvel", "right_saber_tip_linvel"):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            assert sid != -1, f"Missing sensor: {name}"

    def test_shadow_mapping_disabled(self, saber_env):
        assert int(saber_env.unwrapped.model.vis.quality.shadowsize) == 0

    def test_offscreen_mj_renderer_tunes_scene_flags(self, saber_env):
        import mujoco

        from myosuite.viz.mj_renderer import MJRenderer

        r = MJRenderer(saber_env.unwrapped.model, saber_env.unwrapped.data)
        r.render_offscreen(width=64, height=64)
        scn = r._renderer.scene
        assert int(scn.flags[int(mujoco.mjtRndFlag.mjRND_SHADOW)]) == 0
        assert int(scn.flags[int(mujoco.mjtRndFlag.mjRND_REFLECTION)]) == 0
        assert int(scn.flags[int(mujoco.mjtRndFlag.mjRND_SKYBOX)]) == 0

    def test_matskin_reflectance_zero(self, saber_env):
        import mujoco

        model = saber_env.unwrapped.model
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "MatSkin")
        assert mid >= 0
        assert float(model.mat_reflectance[mid]) == 0.0

    def test_runtime_gl_hotfix_zeros_reflectance_and_haze(self, saber_env):
        model = saber_env.unwrapped.model
        # Cross-platform MuJoCo builds can retain non-zero reflectance on some
        # imported materials; assert the explicit GL hotfix knobs instead.
        for mat_name in (
            "MatSkin",
            "saber_blue_core",
            "saber_blue_aura",
            "saber_red_core",
            "saber_red_aura",
        ):
            import mujoco

            mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_name)
            if mid >= 0:
                assert float(model.mat_reflectance[mid]) == 0.0
        assert float(model.vis.map.haze) == 0.0
        assert float(model.vis.map.alpha) == 0.0
        # MuJoCo can clamp/normalize glow to 1.0 depending on build/runtime.
        # Keep this as a sanity check instead of requiring an exact zero.
        assert 0.0 <= float(model.vis.global_.glow) <= 1.0

    def test_scene_fill_light_is_directional_without_shadow(self, saber_env):
        import mujoco

        model = saber_env.unwrapped.model
        lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "saber_scene_fill")
        if lid >= 0:
            assert int(model.light_type[lid]) == int(
                mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
            )
            assert int(model.light_castshadow[lid]) == 0
        else:
            # Some builds may rename/strip this helper light; require an equivalent
            # directional non-shadow light to keep the same rendering behavior.
            directional_no_shadow = False
            for i in range(int(model.nlight)):
                if (
                    int(model.light_type[i])
                    == int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
                    and int(model.light_castshadow[i]) == 0
                ):
                    directional_no_shadow = True
                    break
            assert directional_no_shadow

        # Light count can differ by backend/build (e.g., stripped helper lights).
        # Require the named lights below instead of a global count threshold.
        assert int(model.nlight) >= 3

        top_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "saber_fill_top")
        if top_id >= 0:
            assert float(model.light_cutoff[top_id]) == 180.0
            assert int(model.light_castshadow[top_id]) == 0

        for hand in ("left_saber_light", "right_saber_light"):
            hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, hand)
            assert hid >= 0
            assert float(model.light_cutoff[hid]) == 180.0

        # Headlight defaults vary by MuJoCo build (zeros, [0.4,…], or [0.5,…]).
        ambient = np.asarray(model.vis.headlight.ambient, dtype=np.float32)
        diffuse = np.asarray(model.vis.headlight.diffuse, dtype=np.float32)
        assert np.all((ambient >= 0.0) & (ambient <= 0.5))
        assert np.all((diffuse >= 0.0) & (diffuse <= 0.5))

    def test_saber_targets_are_mocap(self, saber_env):
        """Pool spheres are mocap bodies (kinematic poses from the env)."""
        import mujoco

        model = saber_env.unwrapped.model
        for body_name in ("saber_target_0", "saber_target_99"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            assert bid != -1
            assert int(model.body_mocapid[bid]) >= 0

    def test_target_site_world_positions_change_over_steps(self, saber_env):
        """After the first spawn interval, an active pool target moves each step."""
        import mujoco

        saber_env.reset(seed=0)
        model = saber_env.unwrapped.model
        data = saber_env.unwrapped.data
        # First spawn at control step 10 (see spawn_interval_steps).
        for _ in range(10):
            saber_env.step(np.zeros(saber_env.action_space.shape, dtype=np.float32))
        spawned_id = None
        for i in range(100):
            z = float(
                data.site_xpos[
                    mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_SITE, f"saber_target_{i}_site"
                    )
                ][2]
            )
            if z > -5.0:
                spawned_id = i
                break
        assert (
            spawned_id is not None
        ), "expected at least one spawned target above graveyard"
        sid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"saber_target_{spawned_id}_site"
        )
        mujoco.mj_forward(model, data)
        p0 = data.site_xpos[sid].copy()
        for _ in range(15):
            saber_env.step(np.zeros(saber_env.action_space.shape, dtype=np.float32))
        p1 = data.site_xpos[sid].copy()
        assert float(np.linalg.norm(p1 - p0)) > 1e-4


class TestSaberRewardTerm:
    """Unit tests for saber_targets_reward."""

    def _mock_accessor(self, left_pos, right_pos, target_a, target_b, time_sec=0.0):
        class MockAccessor:
            def __init__(self):
                self._sites = {
                    "left_saber_tip": np.asarray(left_pos),
                    "right_saber_tip": np.asarray(right_pos),
                    "saber_target_a_site": np.asarray(target_a),
                    "saber_target_b_site": np.asarray(target_b),
                    "upper_body_mimic": np.zeros(3, dtype=np.float64),
                }
                self._site_ids = {name: i for i, name in enumerate(self._sites)}
                self._time = float(time_sec)

            def site_xpos(self, sid):
                name = list(self._sites.keys())[sid]
                return self._sites[name]

            def site_id(self, name):
                return self._site_ids[name]

            def time(self):
                return self._time

            def array_module(self):
                return np

        return MockAccessor()

    def test_no_hits_when_far(self):
        from myosuite.terms.base_reward import saber_targets_reward

        acc = self._mock_accessor(
            left_pos=[0.0, 0.0, 0.0],
            right_pos=[0.0, 0.0, 0.0],
            target_a=[1.0, 0.0, 0.0],
            target_b=[1.0, 1.0, 0.0],
        )
        result = saber_targets_reward(acc, {}, motion_amplitude=0.0, hit_threshold=0.1)
        assert result["left_saber_hits"] == 0.0
        assert result["right_saber_hits"] == 0.0
        assert result["total_hits"] == 0.0

    def test_hits_when_close(self):
        from myosuite.terms.base_reward import saber_targets_reward

        acc = self._mock_accessor(
            left_pos=[0.0, 0.0, 0.0],
            right_pos=[0.0, 0.0, 0.0],
            target_a=[0.05, 0.0, 0.0],
            target_b=[0.05, 0.0, 0.0],
        )
        result = saber_targets_reward(acc, {}, motion_amplitude=0.0, hit_threshold=0.1)
        assert result["left_saber_hits"] == 1.0
        assert result["right_saber_hits"] == 1.0
        assert result["total_hits"] == pytest.approx(2.0)
        assert result["dense"] > 0.0

    def test_pool_reward_exposes_wrong_hand_face_terms(self):
        from myosuite.terms.base_reward import saber_target_pool_reward

        class MockAccessor:
            def array_module(self):
                return np

        task_state = {
            "target_pool_hit_this_step": False,
            "target_pool_correct_hand_this_step": False,
            "target_pool_correct_face_this_step": False,
            "target_pool_wrong_hand_this_step": True,
            "target_pool_wrong_face_this_step": True,
            "target_pool_correct_hand_hits": 0,
            "target_pool_correct_face_hits": 0,
            "target_pool_wrong_hand_hits": 1,
            "target_pool_wrong_face_hits": 2,
            "last_action": np.zeros(4, dtype=np.float32),
        }
        out = saber_target_pool_reward(MockAccessor(), task_state, w_correct_hand=0.5)
        assert out["correct_hand_this_step"] == 0.0
        assert out["correct_face_this_step"] == 0.0
        assert out["wrong_hand_this_step"] == 1.0
        assert out["wrong_face_this_step"] == 1.0
        assert out["correct_hand_total"] == 0.0
        assert out["correct_face_total"] == 0.0
        assert out["wrong_hand_total"] == 1.0
        assert out["wrong_face_total"] == 2.0
        assert out["success_score"] == 0.0

    def test_pool_reward_combines_success_slicing_and_timing_terms(self):
        from myosuite.terms.base_reward import saber_target_pool_reward

        class MockAccessor:
            def array_module(self):
                return np

        task_state = {
            "target_pool_hit_this_step": True,
            "target_pool_hit_y_at_hit": 0.8,
            "target_pool_hit_y_deviation_at_hit": 0.25,
            "target_pool_face_surface_distance_at_hit": 0.02,
            "target_pool_slicing_accuracy_this_step": 0.75,
            "target_pool_timing_accuracy_this_step": -0.25,
            "target_pool_miss_zone": False,
            "target_pool_correct_hand_this_step": True,
            "target_pool_correct_face_this_step": True,
        }
        out = saber_target_pool_reward(
            MockAccessor(),
            task_state,
            w_hit=2.0,
            w_correct_hand=4.0,
            w_correct_face=3.0,
        )
        assert out["hit_y_at_hit"] == pytest.approx(0.8)
        assert out["hit_y_deviation_at_hit"] == pytest.approx(0.25)
        assert out["face_surface_distance_at_hit"] == pytest.approx(0.02)
        assert out["slicing_accuracy"] == pytest.approx(0.75)
        assert out["timing_accuracy"] == pytest.approx(-0.25)
        assert out["success_score"] == pytest.approx(9.0)
        assert out["dense"] == pytest.approx(9.5)

    def test_pool_reward_terminates_at_zero_health(self):
        from myosuite.terms.base_reward import saber_target_pool_reward

        class MockAccessor:
            def array_module(self):
                return np

        task_state = {
            "target_pool_hit_this_step": False,
            "target_pool_miss_zone": False,
            "health_status": 0.0,
        }
        out = saber_target_pool_reward(
            MockAccessor(),
            task_state,
        )
        assert out["health_status"] == 0.0
        assert out["health_depleted"] is True
        assert out["done"] is True

    def test_pool_reward_ignores_zero_health_when_disabled(self):
        from myosuite.terms.base_reward import saber_target_pool_reward

        class MockAccessor:
            def array_module(self):
                return np

        task_state = {
            "target_pool_hit_this_step": False,
            "target_pool_miss_zone": False,
            "health_status": 0.0,
        }
        out = saber_target_pool_reward(
            MockAccessor(),
            task_state,
            use_health_status=False,
        )
        assert out["health_status"] == 0.0
        assert out["health_depleted"] is False
        assert out["done"] is False

    def test_pool_reward_terminates_at_miss_limit(self):
        from myosuite.terms.base_reward import saber_target_pool_reward

        class MockAccessor:
            def array_module(self):
                return np

        task_state = {
            "target_pool_hit_this_step": False,
            "target_pool_miss_zone": False,
            "target_pool_misses": 3,
            "target_pool_correct_hand_this_step": True,
            "target_pool_wrong_face_this_step": False,
        }
        out = saber_target_pool_reward(
            MockAccessor(),
            task_state,
            max_target_misses=3,
        )
        assert out["misses_total"] == 3.0
        assert out["miss_limit_reached"] == 1.0
        assert out["done"] is True

    def test_saber_keyframe_pose_reward_prefers_small_pose_error(self):
        from myosuite.terms.base_reward import saber_keyframe_pose_reward

        class MockAccessor:
            def array_module(self):
                return np

        near_pose = saber_keyframe_pose_reward(
            MockAccessor(),
            {"saber_keyframe_pose_error": 0.0},
            saber_keyframe_pose_error_scale=40.0,
        )
        drifted_pose = saber_keyframe_pose_reward(
            MockAccessor(),
            {"saber_keyframe_pose_error": 0.05},
            saber_keyframe_pose_error_scale=40.0,
        )

        assert near_pose["saber_keyframe_pose"] == pytest.approx(1.0)
        assert drifted_pose["saber_keyframe_pose"] < near_pose["saber_keyframe_pose"]
        assert drifted_pose["saber_keyframe_pose_error"] == pytest.approx(0.05)


def test_movement_efficiency_reward_matches_one_minus_mean_squared_activation():
    from myosuite.terms.base_reward import movement_efficiency_reward

    class MockAccessor:
        def array_module(self):
            return np

        def muscle_act(self):
            return np.array([0.0, 0.5, 1.0], dtype=np.float32)

    out = movement_efficiency_reward(MockAccessor(), {})
    assert out["mean_squared_activation"] == pytest.approx((0.0 + 0.25 + 1.0) / 3.0)
    assert out["movement_efficiency"] == pytest.approx(1.0 - (0.0 + 0.25 + 1.0) / 3.0)
    assert out["dense"] == pytest.approx(out["movement_efficiency"])


def test_health_status_update_matches_requested_dynamics():
    from myosuite.envs.myo.tasks.challenge.saber.saber_target_pool import (
        SaberTargetPoolRuntime,
        apply_health_status_update,
    )

    runtime = SaberTargetPoolRuntime()
    assert runtime.health_status == pytest.approx(0.5)

    apply_health_status_update(runtime, hit=True)
    assert runtime.health_status == pytest.approx(0.51)

    apply_health_status_update(runtime, miss=True)
    assert runtime.health_status == pytest.approx(0.36)

    apply_health_status_update(runtime, obstacle_contact=True)
    assert runtime.health_status == pytest.approx(0.21)

    apply_health_status_update(runtime, wrong_face=True)
    assert runtime.health_status == pytest.approx(0.11)

    apply_health_status_update(runtime, wrong_hand=True)
    assert runtime.health_status == pytest.approx(0.01)

    apply_health_status_update(runtime, miss=True)
    assert runtime.health_status == pytest.approx(0.0)
    assert runtime.health_depleted_this_step is True


def test_saber_task_default_miss_limit():
    from myosuite.envs.myo.tasks.challenge.saber_task_config import SaberP0Task
    from myosuite.envs.myo.tasks.challenge.saber.saber_target_pool import (
        SABER_POOL_MAX_TARGET_MISSES,
    )

    task = SaberP0Task()
    assert task.saber_cfg.max_target_misses == SABER_POOL_MAX_TARGET_MISSES


def test_saber_task_health_toggle_keeps_obs_and_updates_reward_wiring():
    from myosuite.envs.myo.tasks.challenge.saber_task_config import SaberP0Task

    enabled_task = SaberP0Task(use_health_status=True)
    assert "health_status" in enabled_task.obs.keys
    assert enabled_task.saber_cfg.use_health_status is True

    disabled_task = SaberP0Task(use_health_status=False)
    assert "health_status" in disabled_task.obs.keys
    assert disabled_task.saber_cfg.use_health_status is False


def test_saber_cfg_use_health_status_is_authoritative():
    """saber_cfg set directly must not be overwritten by the top-level default."""
    from myosuite.envs.myo.tasks.challenge.saber_task_config import (
        SaberP0Cfg,
        SaberP0Task,
    )

    task = SaberP0Task(saber_cfg=SaberP0Cfg(use_health_status=True))
    assert task.saber_cfg.use_health_status is True


def test_saber_target_pool_obs_uses_hit_area_and_hand_metadata():
    from myosuite.terms.base_obs import saber_target_pool_obs

    class MockAccessor:
        def __init__(self):
            self._sites = {
                "left_saber_tip": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                "right_saber_tip": np.array([2.0, 0.0, 0.0], dtype=np.float32),
                "saber_target_0_site": np.array([0.1, 0.0, 0.0], dtype=np.float32),
                "saber_target_1_site": np.array([5.0, 0.0, 0.0], dtype=np.float32),
                "saber_target_2_site": np.array([1.9, 0.0, 0.0], dtype=np.float32),
            }
            self._site_ids = {name: i for i, name in enumerate(self._sites)}

        def site_xpos(self, sid):
            name = list(self._sites.keys())[sid]
            return self._sites[name]

        def site_id(self, name):
            return self._site_ids[name]

        def array_module(self):
            return np

    obs = saber_target_pool_obs(
        MockAccessor(),
        pool_size=3,
        num_visible=2,
        target_pool_active=np.array([True, False, True]),
        target_pool_vel=np.array(
            [[0.1, 0.2, 0.3], [9.0, 9.0, 9.0], [0.4, 0.5, 0.6]], dtype=np.float32
        ),
        target_pool_hit_area=np.array([2, 1, 4], dtype=np.int8),
        target_pool_preferred_hand=np.array([0, 1, 1], dtype=np.int8),
    )

    expected = np.array(
        [
            0.1,
            0.0,
            0.0,
            0.1,
            0.2,
            0.3,
            2.0,
            0.0,
            -0.1,
            0.0,
            0.0,
            0.4,
            0.5,
            0.6,
            4.0,
            1.0,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(obs, expected, atol=1e-7)


def test_health_status_obs_exposes_scalar():
    from myosuite.terms.base_obs import health_status_obs

    class MockAccessor:
        def array_module(self):
            return np

    obs = health_status_obs(MockAccessor(), health_status=0.37)
    np.testing.assert_allclose(obs, np.array([0.37], dtype=np.float32), atol=1e-7)


def test_plan_target_spawn_keeps_cpu_and_mjlab_spawn_rules_aligned():
    from myosuite.envs.myo.tasks.challenge.saber.saber_target_pool import (
        SaberTargetPoolConfig,
        plan_target_spawn,
    )

    cfg = SaberTargetPoolConfig(
        spawn_portal_x=(-0.25, 0.25),
        spawn_portal_y=(-1.5, -1.5),
        spawn_portal_z=(1.15, 1.15),
        speed_mps_range=(0.75, 0.75),
        obstacle_fraction=0.0,
    )

    plan = plan_target_spawn(
        cfg,
        available_slot_ids=[3],
        active_axis_vals=[],
        np_random=np.random.default_rng(7),
        spawn_lane_cursor=0,
        spawn_hand_cursor=0,
        spawn_hit_area_cursor=0,
    )

    assert plan is not None
    assert plan.slot_index == 3
    assert plan.next_spawn_lane_cursor == 1
    assert plan.next_spawn_hand_cursor == 1
    assert plan.next_spawn_hit_area_cursor == 1
    assert plan.spawn_pos[0] in cfg.spawn_portal_x
    np.testing.assert_allclose(
        plan.spawn_pos[1:],
        np.array([-1.5, 1.15], dtype=np.float64),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        plan.velocity,
        np.array([0.0, 0.75, 0.0], dtype=np.float64),
        atol=1e-12,
    )
    assert not plan.is_obstacle
    assert plan.preferred_hand == (
        1 if plan.spawn_pos[0] == cfg.spawn_portal_x[0] else 0
    )
    assert plan.preferred_hit_area == 0
