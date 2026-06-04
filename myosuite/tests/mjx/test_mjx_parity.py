# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Parity tests for the myosuite4 MJX implementation vs MyoHub/myosuite mjx branch.

These tests verify the four functional gaps fixed in the myosuite4 refactor:

1. ``ccd_iterations = 75`` is set on every env's ``mj_model``.
2. Cylinder and ellipsoid contacts are disabled for the JAX/XLA stack (``impl``
   is ``None`` or ``"jax"``), matching ``mjx.put_model`` defaults; they stay
   enabled for Warp (``impl == "warp"``).
3. The ``norm_actions`` config flag bypasses the sigmoid normalisation.
4. ``CumulativeFatigue`` (``mjx/fatigue_jax.py``) computes correct state
   updates and ``FatigueWrapper`` stores fatigue in ``data.userdata``.

The tests in classes 1–3 instantiate real environments (requires myoelbow
model XML) and are skipped if MJX stack or model files are not available.
Class 4 tests the fatigue model on a lightweight finger model and only
needs JAX + MuJoCo.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Optional-import guard — skip the whole module if MJX is unavailable
# ---------------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jp
    import mujoco
    import numpy as np
    from mujoco import mjx  # noqa: F401  # importability check for MJX stack

    jax.config.update("jax_platform_name", "cpu")
except (ImportError, AttributeError) as _err:
    pytest.skip(
        f"JAX/MJX not available ({_err}); install with `uv sync --extra mjx`",
        allow_module_level=True,
    )

pytestmark = pytest.mark.tier2

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FINGER_MODEL = "myosuite/simhive/myo_sim/finger/myofinger_v0.xml"
ELBOW_MODEL = "myosuite/envs/myo/assets/elbow/myoelbow_1dof6muscles.xml"
LEG_MODEL = "myosuite/simhive/myo_sim/leg/myolegs.xml"


def _make_pose_env(impl: str = "jax"):
    """Instantiate ``MjxPoseEnv`` (elbow) with the given backend *impl*.

    Uses ``impl="jax"`` by default because the elbow model has cylinder-mesh
    pairs that the JAX XLA backend cannot handle without contact disabling.
    Passing ``impl="jax"`` triggers ``_preprocess_spec`` to disable those
    contacts, matching the upstream ``MjxMyoBase`` behaviour.
    """
    import copy
    import jax.numpy as jp
    from ml_collections import config_dict
    from myosuite.envs.myo.backends.mjx import _elbow_pose_config
    from myosuite.envs.myo.backends.mjx.pose_env import MjxPoseEnv

    cfg = copy.deepcopy(_elbow_pose_config)
    cfg["mjx_impl"] = impl
    cfg["target_jnt_range"] = config_dict.create(r_elbow_flex=jp.array((0.0, 2.27)))
    return MjxPoseEnv(config=cfg)


def _make_reach_env(impl: str = "jax"):
    """Instantiate ``MjxReachEnv`` (hand) with the given backend *impl*."""
    import copy
    import jax.numpy as jp
    from ml_collections import config_dict
    from myosuite.envs.myo.backends.mjx import _hand_reach_config
    from myosuite.envs.myo.backends.mjx.reach_env import MjxReachEnv

    cfg = copy.deepcopy(_hand_reach_config)
    cfg["mjx_impl"] = impl
    cfg["far_th"] = 0.044
    cfg["target_reach_range"] = config_dict.create(
        THtip=jp.array(((-0.165, -0.537, 1.495), (-0.165, -0.537, 1.495))),
    )
    return MjxReachEnv(config=cfg)


# ---------------------------------------------------------------------------
# 1. Solver params — ccd_iterations and friends
# ---------------------------------------------------------------------------


class TestSolverParams:
    """ccd_iterations = 75 must be present on every MJX env."""

    def test_pose_env_ccd_iterations(self):
        env = _make_pose_env()
        assert (
            env._mj_model.opt.ccd_iterations == 75
        ), "MjxPoseEnv is missing opt.ccd_iterations = 75 (Gap 1)"

    def test_pose_env_iterations(self):
        env = _make_pose_env()
        assert env._mj_model.opt.iterations == 6
        assert env._mj_model.opt.ls_iterations == 6

    def test_reach_env_ccd_iterations(self):
        env = _make_reach_env()
        assert (
            env._mj_model.opt.ccd_iterations == 75
        ), "MjxReachEnv is missing opt.ccd_iterations = 75 (Gap 1)"

    def test_reach_env_iterations(self):
        env = _make_reach_env()
        assert env._mj_model.opt.iterations == 6
        assert env._mj_model.opt.ls_iterations == 6


# ---------------------------------------------------------------------------
# 2. Cylinder contact impl guard
# ---------------------------------------------------------------------------


class TestCylinderContactGuard:
    """Cylinder contacts are cleared for JAX/XLA (``impl`` ``None`` or ``jax``).

    ``mjx.put_model(..., impl=None)`` uses the JAX backend, so preprocessing must
    match ``impl="jax"``. Warp keeps cylinder collisions enabled.
    """

    def test_preprocess_spec_impl_none_disables_cylinder_contacts(self):
        """_preprocess_spec(impl=None) must disable cylinders like impl='jax'."""
        from myosuite.envs.myo.backends.mjx.pose_env import MjxPoseEnv

        spec = mujoco.MjSpec.from_file(ELBOW_MODEL)
        n_cylinders = sum(
            1 for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_CYLINDER
        )
        assert n_cylinders > 0, "Elbow model has no cylinder geoms — test is vacuous"

        result = MjxPoseEnv._preprocess_spec(spec, impl=None)

        disabled = [
            g
            for g in result.geoms
            if g.type == mujoco.mjtGeom.mjGEOM_CYLINDER
            and g.contype == 0
            and g.conaffinity == 0
        ]
        assert len(disabled) == n_cylinders, (
            f"impl=None should disable all {n_cylinders} cylinder contacts, "
            f"only {len(disabled)} were disabled"
        )

    def test_preprocess_spec_impl_warp_leaves_cylinders_unchanged(self):
        """_preprocess_spec with impl='warp' should leave cylinder contacts unchanged."""
        from myosuite.envs.myo.backends.mjx.pose_env import MjxPoseEnv

        spec = mujoco.MjSpec.from_file(ELBOW_MODEL)
        orig = {
            g.name: (g.contype, g.conaffinity)
            for g in spec.geoms
            if g.type == mujoco.mjtGeom.mjGEOM_CYLINDER
        }

        result = MjxPoseEnv._preprocess_spec(spec, impl="warp")

        for g in result.geoms:
            if g.type == mujoco.mjtGeom.mjGEOM_CYLINDER and g.name in orig:
                assert (g.contype, g.conaffinity) == orig[
                    g.name
                ], f"Geom {g.name!r}: impl='warp' should not change cylinder contacts"

    def test_preprocess_spec_impl_jax_disables_cylinder_contacts(self):
        """_preprocess_spec with impl='jax' should disable all cylinder contacts."""
        from myosuite.envs.myo.backends.mjx.pose_env import MjxPoseEnv

        spec = mujoco.MjSpec.from_file(ELBOW_MODEL)
        n_cylinders = sum(
            1 for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_CYLINDER
        )
        assert n_cylinders > 0, "Elbow model has no cylinder geoms — test is vacuous"

        result = MjxPoseEnv._preprocess_spec(spec, impl="jax")

        disabled = [
            g
            for g in result.geoms
            if g.type == mujoco.mjtGeom.mjGEOM_CYLINDER
            and g.contype == 0
            and g.conaffinity == 0
        ]
        assert len(disabled) == n_cylinders, (
            f"impl='jax' should disable all {n_cylinders} cylinder contacts, "
            f"only {len(disabled)} were disabled (Gap 2)"
        )

    def test_reach_preprocess_spec_impl_none_disables_cylinders(self):
        """MjxReachEnv._preprocess_spec(impl=None) matches JAX cylinder policy."""
        from myosuite.envs.myo.backends.mjx.reach_env import MjxReachEnv

        spec = mujoco.MjSpec.from_file(ELBOW_MODEL)
        n_cylinders = sum(
            1 for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_CYLINDER
        )
        assert n_cylinders > 0, "Elbow model has no cylinder geoms — test is vacuous"

        result_none = MjxReachEnv._preprocess_spec(spec, impl=None)
        disabled_none = [
            g
            for g in result_none.geoms
            if g.type == mujoco.mjtGeom.mjGEOM_CYLINDER
            and g.contype == 0
            and g.conaffinity == 0
        ]
        assert (
            len(disabled_none) == n_cylinders
        ), "MjxReachEnv: impl=None must disable all cylinder contacts for JAX default"

    def test_impl_jax_env_has_no_cylinder_contacts(self):
        """Full env with impl='jax' should have all cylinder contacts disabled."""
        env = _make_pose_env(impl="jax")
        disabled = sum(
            1
            for i in range(env._mj_model.ngeom)
            if env._mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER
            and env._mj_model.geom_contype[i] == 0
            and env._mj_model.geom_conaffinity[i] == 0
        )
        n_cylinders = sum(
            1
            for i in range(env._mj_model.ngeom)
            if env._mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER
        )
        assert (
            disabled == n_cylinders
        ), f"impl='jax' env should have all {n_cylinders} cylinder contacts disabled"


# ---------------------------------------------------------------------------
# 2b. Ellipsoid contact guard (leg / walk-style models)
# ---------------------------------------------------------------------------


class TestEllipsoidContactGuard:
    """Ellipsoid contacts follow the same JAX vs Warp policy as cylinders."""

    def test_preprocess_disables_ellipsoids_for_impl_none(self):
        """Shared preprocessor clears ellipsoid contacts for JAX/XLA."""
        from myosuite.envs.myo.backends.mjx.mjx_spec_preprocess import (
            preprocess_mjx_spec,
        )

        spec = mujoco.MjSpec.from_file(LEG_MODEL)
        n_ellipsoid = sum(
            1 for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_ELLIPSOID
        )
        assert n_ellipsoid > 0, "Leg model has no ellipsoid geoms — test is vacuous"

        preprocess_mjx_spec(spec, impl=None)
        cleared = sum(
            1
            for g in spec.geoms
            if g.type == mujoco.mjtGeom.mjGEOM_ELLIPSOID
            and g.contype == 0
            and g.conaffinity == 0
        )
        assert cleared == n_ellipsoid, (
            f"impl=None should disable all {n_ellipsoid} ellipsoid contacts, "
            f"only {cleared} disabled"
        )

    def test_preprocess_warp_leaves_ellipsoids_unchanged(self):
        """Warp backend keeps ellipsoid contact masks."""
        from myosuite.envs.myo.backends.mjx.mjx_spec_preprocess import (
            preprocess_mjx_spec,
        )

        spec = mujoco.MjSpec.from_file(LEG_MODEL)
        before = [
            (g.contype, g.conaffinity)
            for g in spec.geoms
            if g.type == mujoco.mjtGeom.mjGEOM_ELLIPSOID
        ]
        assert before, "Leg model should list at least one ellipsoid geom"

        preprocess_mjx_spec(spec, impl="warp")

        after = [
            (g.contype, g.conaffinity)
            for g in spec.geoms
            if g.type == mujoco.mjtGeom.mjGEOM_ELLIPSOID
        ]
        assert after == before

    def test_mjx_leg_walk_compiles_and_clears_ellipsoid_contacts(self):
        """MjxLegWalk loads myolegs.xml and clears ellipsoid contacts for JAX/XLA."""
        import copy

        from myosuite.envs.myo.backends.mjx import _leg_walk_config
        from myosuite.envs.myo.backends.mjx.walk_env import MjxWalkEnv

        cfg = copy.deepcopy(_leg_walk_config)
        env = MjxWalkEnv(cfg)
        mj = env._mj_model
        disabled = sum(
            1
            for i in range(mj.ngeom)
            if mj.geom_type[i] == mujoco.mjtGeom.mjGEOM_ELLIPSOID
            and mj.geom_contype[i] == 0
            and mj.geom_conaffinity[i] == 0
        )
        n_ellipsoid = sum(
            1
            for i in range(mj.ngeom)
            if mj.geom_type[i] == mujoco.mjtGeom.mjGEOM_ELLIPSOID
        )
        assert (
            disabled == n_ellipsoid
        ), "MjxLegWalk default impl should disable all ellipsoid collision masks"


# ---------------------------------------------------------------------------
# 3. norm_actions config flag
# ---------------------------------------------------------------------------


class TestNormActionsFlag:
    """norm_actions=True applies sigmoid; norm_actions=False passes action through."""

    def _get_base_env(self, norm_actions: bool):
        env = _make_pose_env()
        env._config.norm_actions = norm_actions
        return env

    def test_norm_actions_true_applies_sigmoid(self):
        """When norm_actions=True (default), _normalize_action applies sigmoid."""
        env = _make_pose_env()
        assert env._config.norm_actions is True

        action = jp.zeros(env._mj_model.nu)
        result = env._normalize_action(action)
        expected = 1.0 / (1.0 + jp.exp(-5.0 * (action - 0.5)))
        np.testing.assert_allclose(np.array(result), np.array(expected), atol=1e-6)

    def test_norm_actions_false_passes_through(self):
        """When norm_actions=False, _normalize_action is identity."""
        env = self._get_base_env(norm_actions=False)

        action = jp.array([0.1, 0.9, 0.5, 0.3, 0.7, 0.2], dtype=jp.float32)
        result = env._normalize_action(action)
        np.testing.assert_allclose(np.array(result), np.array(action), atol=1e-7)

    def test_norm_actions_sigmoid_maps_zero_to_half(self):
        """Sigmoid at 0.5 (midpoint) should give 0.5."""
        env = _make_pose_env()
        action = jp.full((env._mj_model.nu,), 0.5)
        result = env._normalize_action(action)
        np.testing.assert_allclose(
            np.array(result), np.full(env._mj_model.nu, 0.5), atol=1e-6
        )

    def test_default_configs_have_norm_actions_true(self):
        """All default configs should include norm_actions=True."""
        from myosuite.envs.myo.backends.mjx import (
            _pose_env_config,
            _reach_env_config,
            _leg_walk_config,
        )

        assert _pose_env_config.norm_actions is True
        assert _reach_env_config.norm_actions is True
        assert _leg_walk_config.norm_actions is True

    def test_impl_and_n_substeps_properties(self):
        """MyoMjxEnvBase should expose impl and n_substeps properties."""
        env = _make_pose_env()
        assert env.impl == "jax"  # Test helper uses impl="jax"
        assert env.n_substeps == int(env._config.ctrl_dt / env._config.sim_dt)


# ---------------------------------------------------------------------------
# 4. CumulativeFatigue (mjx/fatigue_jax.py) — parity with upstream
# ---------------------------------------------------------------------------


class TestMjxCumulativeFatigue:
    """Tests for the new MJX-specific CumulativeFatigue in mjx/fatigue_jax.py."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from myosuite.envs.myo.backends.mjx.fatigue_jax import CumulativeFatigue

        self.model = mujoco.MjModel.from_xml_path(FINGER_MODEL)
        self.frame_skip = 5
        self.fatigue = CumulativeFatigue(self.model, frame_skip=self.frame_skip)
        self.test_act = np.array([0.5] * 5, dtype=np.float32)

    def test_reset_default_zero_fatigue(self):
        """Default reset: MA=0, MR=1, MF=0."""
        rng = jax.random.PRNGKey(0)
        state = self.fatigue.reset(rng)
        np.testing.assert_allclose(np.array(state["MA"]), np.zeros(self.fatigue.na))
        np.testing.assert_allclose(np.array(state["MR"]), np.ones(self.fatigue.na))
        np.testing.assert_allclose(np.array(state["MF"]), np.zeros(self.fatigue.na))

    def test_reset_random_sums_to_one(self):
        """Random reset: MA + MR + MF == 1 for every muscle."""
        rng = jax.random.PRNGKey(42)
        state = self.fatigue.reset(rng, fatigue_reset_random=True)
        total = np.array(state["MA"]) + np.array(state["MR"]) + np.array(state["MF"])
        np.testing.assert_allclose(total, np.ones(self.fatigue.na), atol=1e-6)

    def test_reset_random_reproducible(self):
        """Same seed → same random fatigue state."""
        rng = jax.random.PRNGKey(7)
        s1 = self.fatigue.reset(rng, fatigue_reset_random=True)
        s2 = self.fatigue.reset(rng, fatigue_reset_random=True)
        np.testing.assert_allclose(np.array(s1["MA"]), np.array(s2["MA"]))
        np.testing.assert_allclose(np.array(s1["MF"]), np.array(s2["MF"]))

    def test_reset_from_vec(self):
        """Reset from a fixed MF vector: MF=vec, MR=1-vec, MA=0."""
        rng = jax.random.PRNGKey(0)
        vec = [0.1, 0.2, 0.3, 0.4, 0.5]
        state = self.fatigue.reset(rng, fatigue_reset_vec=vec)
        np.testing.assert_allclose(np.array(state["MF"]), vec, atol=1e-6)
        np.testing.assert_allclose(
            np.array(state["MR"]), [0.9, 0.8, 0.7, 0.6, 0.5], atol=1e-6
        )
        np.testing.assert_allclose(np.array(state["MA"]), np.zeros(5), atol=1e-6)

    def test_compute_act_preserves_mass(self):
        """MA + MR + MF should remain close to 1 after one step."""
        rng = jax.random.PRNGKey(0)
        state = self.fatigue.reset(rng)
        act = jp.array(self.test_act)
        state2 = self.fatigue.compute_act(act, fatigue_state=state)
        total = np.array(state2["MA"]) + np.array(state2["MR"]) + np.array(state2["MF"])
        # Allow small numerical drift
        np.testing.assert_allclose(total, np.ones(self.fatigue.na), atol=1e-4)

    def test_compute_act_increases_ma_from_zero(self):
        """Starting from MA=0, compute_act with positive TL should increase MA."""
        rng = jax.random.PRNGKey(0)
        state = self.fatigue.reset(rng)  # MA=0
        act = jp.ones(self.fatigue.na) * 0.8
        state2 = self.fatigue.compute_act(act, fatigue_state=state)
        assert (
            float(jp.mean(state2["MA"])) > 0.0
        ), "MA should increase from zero with positive TL"

    def test_compute_act_deterministic(self):
        """Same inputs → same outputs (pure function, no side effects)."""
        rng = jax.random.PRNGKey(1)
        state = self.fatigue.reset(rng)
        act = jp.array(self.test_act)
        out1 = self.fatigue.compute_act(act, fatigue_state=state)
        out2 = self.fatigue.compute_act(act, fatigue_state=state)
        np.testing.assert_allclose(np.array(out1["MA"]), np.array(out2["MA"]))

    def test_compute_act_jittable(self):
        """compute_act must be JIT-compilable."""
        rng = jax.random.PRNGKey(0)
        state = self.fatigue.reset(rng)
        fatigue = self.fatigue  # closure

        @jax.jit
        def step(act, s):
            return fatigue.compute_act(act, fatigue_state=s)

        act = jp.array(self.test_act)
        result = step(act, state)
        assert result["MA"].shape == (self.fatigue.na,)

    def test_get_effort(self):
        """get_effort returns a non-negative scalar."""
        rng = jax.random.PRNGKey(0)
        state = self.fatigue.reset(rng)
        act = jp.array(self.test_act)
        effort = self.fatigue.get_effort(act, fatigue_state=state)
        assert float(effort) >= 0.0

    def test_set_coefficients(self):
        """Setters update the stored JAX scalars."""
        from myosuite.envs.myo.backends.mjx.fatigue_jax import CumulativeFatigue

        f = CumulativeFatigue(self.model, frame_skip=1)
        f.set_FatigueCoefficient(0.05)
        np.testing.assert_allclose(float(f.F), 0.05, atol=1e-7)
        f.set_RecoveryCoefficient(0.002)
        np.testing.assert_allclose(float(f.R), 0.002, atol=1e-7)
        f.set_RecoveryMultiplier(200.0)
        np.testing.assert_allclose(float(f.r), 200.0, atol=1e-7)


# ---------------------------------------------------------------------------
# 5. FatigueWrapper (mjx/fatigue_jax.py)
# ---------------------------------------------------------------------------


class TestFatigueWrapper:
    """FatigueWrapper stores and updates fatigue state in data.userdata."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from myosuite.envs.myo.backends.mjx.fatigue_jax import (
            FatigueWrapper,
            ALLOWED_FATIGUE_OBS_KEYS,
        )

        self.FatigueWrapper = FatigueWrapper
        self.ALLOWED_FATIGUE_OBS_KEYS = ALLOWED_FATIGUE_OBS_KEYS

    def test_skim_config_extracts_fatigue_keys(self):
        """skim_config separates fatigue_config from main config."""
        from ml_collections import config_dict
        from myosuite.envs.myo.backends.mjx.fatigue_jax import FatigueWrapper

        cfg = config_dict.create(
            norm_actions=True,
            fatigue_config=config_dict.create(
                fatigue_reset_vec=None,
                fatigue_reset_random=False,
                fatigue_obs_keys=["MA"],
            ),
        )
        env_cfg, fat_cfg = FatigueWrapper.skim_config(cfg)
        assert "fatigue_config" not in env_cfg
        assert list(fat_cfg.fatigue_obs_keys) == ["MA"]

    def test_skim_config_applies_overrides(self):
        """skim_config applies fatigue-key config_overrides to fatigue_config."""
        from ml_collections import config_dict
        from myosuite.envs.myo.backends.mjx.fatigue_jax import FatigueWrapper

        cfg = config_dict.create(norm_actions=True)
        overrides = {"fatigue_reset_random": True}
        env_cfg, fat_cfg = FatigueWrapper.skim_config(cfg, config_overrides=overrides)
        assert fat_cfg.fatigue_reset_random is True
        assert "fatigue_reset_random" not in env_cfg

    def test_wrapper_disables_inner_norm_actions(self):
        """FatigueWrapper must set env._config.norm_actions = False."""
        env = _make_pose_env()
        assert env._config.norm_actions is True  # sanity: default is True
        wrapped = self.FatigueWrapper(env)
        assert wrapped.env._config.norm_actions is False

    def test_wrapper_expands_nuserdata(self):
        """FatigueWrapper expands nuserdata by 3×nu."""
        env = _make_pose_env()
        nu = env.mj_model.nu
        original_nuserdata = env.mj_model.nuserdata
        wrapped = self.FatigueWrapper(env)
        assert wrapped.env.mj_model.nuserdata == original_nuserdata + 3 * nu

    def test_wrapper_reset_writes_userdata(self):
        """After reset, fatigue userdata slots should be non-trivially filled."""
        env = _make_pose_env()
        wrapped = self.FatigueWrapper(env)

        rng = jax.random.PRNGKey(0)
        state = wrapped.reset(rng)

        # MR should be 1.0 (default non-fatigued reset)
        mr_vals = np.array(state.data.userdata[wrapped.fatigue_index_MR])
        np.testing.assert_allclose(mr_vals, np.ones_like(mr_vals), atol=1e-6)

        # MA and MF should be zero by default
        ma_vals = np.array(state.data.userdata[wrapped.fatigue_index_MA])
        np.testing.assert_allclose(ma_vals, np.zeros_like(ma_vals), atol=1e-6)

    def test_wrapper_step_updates_userdata(self):
        """After step, MA userdata should change (fatigue evolves)."""
        env = _make_pose_env()
        wrapped = self.FatigueWrapper(env)

        rng = jax.random.PRNGKey(0)
        state = wrapped.reset(rng)

        action = jp.ones(wrapped.env.mj_model.nu) * 0.8
        next_state = wrapped.step(state, action)

        ma_after = np.array(next_state.data.userdata[wrapped.fatigue_index_MA])
        # MA should have increased from 0 given a high action
        assert (
            np.mean(ma_after) > 0.0
        ), "MA should increase after step with high activation"

    def test_wrapper_obs_keys_appended(self):
        """When fatigue_obs_keys=['MA'], obs state vector grows by nu."""
        from ml_collections import config_dict

        env = _make_pose_env()
        nu = env.mj_model.nu
        fat_cfg = config_dict.create(
            fatigue_reset_vec=None,
            fatigue_reset_random=False,
            fatigue_obs_keys=["MA"],
        )
        wrapped = self.FatigueWrapper(env, fatigue_config=fat_cfg)

        rng = jax.random.PRNGKey(0)

        # Get baseline obs size without fatigue appended
        env_plain = _make_pose_env()
        state_plain = env_plain.reset(rng)
        plain_size = state_plain.obs["state"].shape[0]

        state = wrapped.reset(rng)
        wrapped_size = state.obs["state"].shape[0]
        assert (
            wrapped_size == plain_size + nu
        ), f"Expected obs size {plain_size + nu} with MA appended, got {wrapped_size}"

    def test_wrapper_invalid_obs_key_raises(self):
        """FatigueWrapper should raise on invalid fatigue_obs_keys."""
        from ml_collections import config_dict

        env = _make_pose_env()
        fat_cfg = config_dict.create(
            fatigue_reset_vec=None,
            fatigue_reset_random=False,
            fatigue_obs_keys=["INVALID_KEY"],
        )
        with pytest.raises(AssertionError, match="Invalid fatigue_obs_keys"):
            self.FatigueWrapper(env, fatigue_config=fat_cfg)

    def test_cumulative_fatigue_matches_mjx_import(self):
        """CumulativeFatigue from mjx/fatigue_jax must be the new dict-based version."""
        from myosuite.envs.myo.backends.mjx.fatigue_jax import CumulativeFatigue

        model = mujoco.MjModel.from_xml_path(FINGER_MODEL)
        f = CumulativeFatigue(model, frame_skip=1)
        rng = jax.random.PRNGKey(0)
        state = f.reset(rng)
        # New version returns plain dict, not a pytree object
        assert isinstance(state, dict)
        assert "MA" in state and "MR" in state and "MF" in state
