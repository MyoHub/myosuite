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

They also guard the silent-corruption fixes: pose/reach targets resolved by
name and sampled independently per coordinate, 3CC-r compartments conserved,
and ``FatigueWrapper`` keeping the model options and the env's action mapping.
The finger reach env checks far targets from the same control step as CPU
``ReachEnvV0`` and samples its reachable-fingertip table.

The tests in classes 1–3 instantiate real environments (requires myoelbow
model XML) and are skipped if MJX stack or model files are not available.
Class 4 tests the fatigue model on a lightweight finger model and only
needs JAX + MuJoCo.
"""

from __future__ import annotations

import pytest
from myosuite.envs.myo.assets._resolve import (
    resolve_elbow_xml as _resolve_elbow_xml,
    resolve_finger_xml as _resolve_finger_xml,
    resolve_leg_xml as _resolve_leg_xml,
)
from myosuite import make_env

# ---------------------------------------------------------------------------
# Optional-import guard — skip the whole module if MJX is unavailable
# ---------------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jp
    import mujoco
    from myosuite.core.model_builder import ModelBuilder
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


FINGER_MODEL = str(_resolve_finger_xml("myofinger_v0.xml"))
ELBOW_MODEL = str(_resolve_elbow_xml("myoelbow_1dof6muscles.xml"))
LEG_MODEL = str(_resolve_leg_xml("myolegs_with_torso.xml"))


def _make_pose_env(impl: str = "jax"):
    """Instantiate ``MjxPoseEnv`` (elbow) with the given backend *impl*.

    Uses ``impl="jax"`` by default because the elbow model has cylinder-mesh
    pairs that the JAX XLA backend cannot handle without contact disabling.
    Passing ``impl="jax"`` triggers ``_preprocess_spec`` to disable those
    contacts, matching the upstream ``MjxMyoBase`` behaviour.
    """
    import jax.numpy as jp
    from ml_collections import config_dict
    from myosuite.envs.myo.backends.mjx import _cfg_elbow_random, _to_config_dict
    from myosuite.envs.myo.backends.mjx.pose_env import MjxPoseEnv

    cfg = _to_config_dict(_cfg_elbow_random())
    cfg["mjx_impl"] = impl
    cfg["target_jnt_range"] = config_dict.create(r_elbow_flex=jp.array((0.0, 2.27)))
    return MjxPoseEnv(config=cfg)


def _make_reach_env(impl: str = "jax"):
    """Instantiate ``MjxReachEnv`` (hand) with the given backend *impl*."""
    import jax.numpy as jp
    from ml_collections import config_dict
    from myosuite.envs.myo.backends.mjx import _cfg_hand_reach_fixed, _to_config_dict
    from myosuite.envs.myo.backends.mjx.reach_env import MjxReachEnv

    cfg = _to_config_dict(_cfg_hand_reach_fixed())
    cfg["mjx_impl"] = impl
    cfg["far_th"] = 0.044
    cfg["target_reach_range"] = config_dict.create(
        THtip_r=jp.array(((-0.165, -0.537, 1.495), (-0.165, -0.537, 1.495))),
    )
    return MjxReachEnv(config=cfg)


def _uniform_fractions(targets, lo, hi) -> np.ndarray:
    """Map sampled targets to [0, 1] per coordinate (only coordinates with lo < hi)."""
    t = np.asarray(targets).reshape(len(targets), -1)
    lo, hi = np.asarray(lo).ravel(), np.asarray(hi).ravel()
    var = hi > lo
    return (t[:, var] - lo[var]) / (hi[var] - lo[var])


# ---------------------------------------------------------------------------
# 0. Targets: resolved by name, independent per coordinate
# ---------------------------------------------------------------------------


class TestTargetResolution:
    """Pose/reach targets match the CPU twins and are sampled independently."""

    def test_finger_pose_fixed_targets_the_right_joints(self):
        """CPU myoFingerPoseFixed-v0: IFadb, IFmcp = 0 and IFpip, IFdip = 0.75."""
        from myosuite.envs.myo.backends.mjx import make

        env = make("MjxFingerPoseFixed-v0")
        target = env.sample_task(jax.random.PRNGKey(0))["target_angles"]
        np.testing.assert_allclose(np.asarray(target), [0.0, 0.0, 0.75, 0.75])

    @pytest.mark.parametrize(
        "env_name", ["MjxFingerPoseRandom-v0", "MjxHandPoseRandom-v0"]
    )
    def test_pose_targets_independent_per_joint(self, env_name):
        """One key used to drive every joint (correlation 1.0)."""
        from myosuite.envs.myo.backends.mjx import make

        env = make(env_name)
        keys = jax.random.split(jax.random.PRNGKey(0), 2000)
        targets = jax.vmap(env.sample_task)(keys)["target_angles"]
        u = _uniform_fractions(targets, env._target_lo, env._target_hi)
        corr = np.corrcoef(u.T)
        assert np.abs(corr[~np.eye(len(corr), dtype=bool)]).max() < 0.15

    def test_hand_reach_tracks_five_distinct_tips_in_cpu_order(self):
        """Unsuffixed names used to resolve to id -1, i.e. LFtip_r five times."""
        from myosuite.envs.myo.backends.mjx import make

        env = make("MjxHandReachRandom-v0")
        names = [env.mj_model.site(int(i)).name for i in np.asarray(env._tip_sids)]
        assert names == ["THtip_r", "IFtip_r", "MFtip_r", "RFtip_r", "LFtip_r"]
        keys = jax.random.split(jax.random.PRNGKey(0), 2000)
        targets = jax.vmap(env.sample_task)(keys)["targets"]
        u = _uniform_fractions(targets, env._target_lo, env._target_hi)
        corr = np.corrcoef(u.T)
        assert np.abs(corr[~np.eye(len(corr), dtype=bool)]).max() < 0.15

    def test_unknown_site_name_raises(self):
        from ml_collections import config_dict
        from myosuite.envs.myo.backends.mjx import (
            _cfg_hand_reach_fixed,
            _to_config_dict,
        )
        from myosuite.envs.myo.backends.mjx.reach_env import MjxReachEnv

        cfg = _to_config_dict(_cfg_hand_reach_fixed())
        cfg["target_reach_range"] = config_dict.create(
            THtip=jp.array(((-0.165, -0.537, 1.495), (-0.165, -0.537, 1.495))),
        )
        with pytest.raises(KeyError):
            MjxReachEnv(config=cfg)

    def test_env_creation_warns_experimental(self):
        from myosuite.envs.myo.backends.mjx import mjx_env_base

        mjx_env_base._warn_experimental_once.cache_clear()
        with pytest.warns(UserWarning, match="experimental"):
            _make_pose_env()


# ---------------------------------------------------------------------------
# 0b. Reach: far check timing and finger targets match CPU ReachEnvV0
# ---------------------------------------------------------------------------


class TestReachMatchesCpu:
    """``MjxFingerReachRandom-v0`` against CPU ``myoFingerReachRandom-v0``."""

    def test_far_check_first_active_at_cpu_control_step(self):
        """The check used 2 physics steps (4 ms), so it ended episodes at step 1."""
        from myosuite.envs.myo.backends.mjx import make

        env = make("MjxFingerReachRandom-v0")
        state = jax.jit(env.reset)(jax.random.PRNGKey(0))
        info = dict(state.info)
        info["targets"] = info["targets"] + jp.array([1.0, 0.0, 0.0])
        state = state.replace(info=info)
        step = jax.jit(env.step)
        mjx_done = []
        for _ in range(2):
            state = step(state, jp.zeros(env.action_size))
            mjx_done.append(bool(state.done))

        cpu = make_env("myoFingerReachRandom-v0")
        u = cpu.unwrapped
        cpu.reset(seed=0)
        u.model.site_pos[u.target_sids[0]] += np.array([1.0, 0.0, 0.0])
        action = np.zeros(cpu.action_space.shape, dtype=np.float32)
        cpu_done = [bool(cpu.step(action)[2]) for _ in range(2)]
        cpu.close()

        assert cpu_done == [False, True]
        assert mjx_done == cpu_done

    def test_finger_reach_reads_cpu_registration(self):
        """far_th was hard-coded to 0.10 (CPU: ReachEnvV0 default 0.35)."""
        from myosuite.envs.myo.backends.mjx import get_default_config

        cfg = get_default_config("MjxFingerReachRandom-v0")
        cpu = make_env("myoFingerReachRandom-v0").unwrapped
        assert cfg.far_th == cpu.far_th == 0.35
        assert cfg.target_sampling == cpu.target_sampling == "workspace"

    def test_finger_reach_samples_the_cpu_workspace_table(self):
        """Targets were drawn uniformly in the box, partly out of reach."""
        from myosuite.envs.myo.backends.mjx import make

        env = make("MjxFingerReachRandom-v0")
        table = make_env("myoFingerReachRandom-v0").unwrapped._workspace_points
        np.testing.assert_allclose(np.asarray(env._workspace_points), table, atol=1e-6)
        keys = jax.random.split(jax.random.PRNGKey(0), 200)
        targets = np.asarray(jax.vmap(env.sample_task)(keys)["targets"])
        points = table.reshape(-1, 3)
        for target in targets.reshape(-1, 3):
            assert np.abs(points - target).sum(axis=1).min() < 1e-6
        assert len(np.unique(targets.reshape(len(keys), -1), axis=0)) > 150

    def test_unknown_target_sampling_raises(self):
        from myosuite.envs.myo.backends.mjx import make

        with pytest.raises(ValueError, match="target_sampling"):
            make("MjxFingerReachRandom-v0", {"target_sampling": "grid"})


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

        spec = ModelBuilder.from_xml_file(LEG_MODEL).build()[1]
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

        spec = ModelBuilder.from_xml_file(LEG_MODEL).build()[1]
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
        from myosuite.envs.myo.backends.mjx import _cfg_leg_walk, _to_config_dict
        from myosuite.envs.myo.backends.mjx.walk_env import MjxWalkEnv

        cfg = _to_config_dict(_cfg_leg_walk())
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
            _cfg_elbow_random,
            _cfg_hand_reach_fixed,
            _cfg_leg_walk,
        )

        assert _cfg_elbow_random().norm_actions is True
        assert _cfg_hand_reach_fixed().norm_actions is True
        assert _cfg_leg_walk().norm_actions is True

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
        """MA + MR + MF stays 1 over a long load/rest sequence.

        MF used to be integrated with the already-updated MA (drift ~2e-4).
        """
        state = self.fatigue.reset(jax.random.PRNGKey(0))
        step = jax.jit(lambda tl, s: self.fatigue.compute_act(tl, fatigue_state=s))
        rng = np.random.default_rng(0)
        worst = 0.0
        for k in range(3000):
            on = (k // 300) % 2 == 0
            tl = (
                rng.uniform(0.0, 1.0, self.fatigue.na)
                if on
                else np.zeros(self.fatigue.na)
            )
            state = step(jp.asarray(tl, dtype=jp.float32), state)
            total = np.asarray(state["MA"] + state["MR"] + state["MF"])
            worst = max(worst, float(np.max(np.abs(total - 1.0))))
        assert worst < 1e-5, worst

    def test_compute_act_uses_shared_step(self):
        """The MJX model and physics.fatigue_jax share one 3CC-r update."""
        from myosuite.physics.fatigue_jax import cumulative_fatigue_step

        state = {"MA": jp.full(5, 0.3), "MR": jp.full(5, 0.5), "MF": jp.full(5, 0.2)}
        act = jp.array(self.test_act)
        out = self.fatigue.compute_act(act, fatigue_state=state)
        ref = cumulative_fatigue_step(
            state["MA"],
            state["MR"],
            state["MF"],
            act,
            F=self.fatigue.F,
            R=self.fatigue.R,
            r=self.fatigue.r,
            dt=self.fatigue.dt,
            tauact=self.fatigue.tauact,
            taudeact=self.fatigue.taudeact,
        )
        for key, value in zip(("MA", "MR", "MF"), ref):
            np.testing.assert_array_equal(np.asarray(out[key]), np.asarray(value))

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

    def test_wrapper_keeps_inner_action_mapping(self):
        """The wrapper reuses the env's own mapping instead of disabling it."""
        env = _make_pose_env()
        wrapped = self.FatigueWrapper(env)
        assert wrapped.env._config.norm_actions is True

    def test_wrapper_keeps_model_options(self):
        """Recompiling for userdata must not revert timestep/solver options."""
        env = _make_pose_env()
        names = ("timestep", "iterations", "ls_iterations", "ccd_iterations")
        before = {n: getattr(env.mj_model.opt, n) for n in names}
        wrapped = self.FatigueWrapper(env)
        assert {n: getattr(wrapped.env.mj_model.opt, n) for n in names} == before
        assert wrapped.muscle_fatigue.dt == pytest.approx(env._config.ctrl_dt)

    def test_wrapper_step_applies_env_mapping_once(self):
        """Muscles get MA computed from the env mapping of the action."""
        env = _make_pose_env()
        wrapped = self.FatigueWrapper(env)
        state = wrapped.reset(jax.random.PRNGKey(0))
        action = jp.linspace(-1.0, 1.0, env.mj_model.nu)
        mask = wrapped.muscle_act_ind
        prev = {
            "MA": state.data.userdata[wrapped.fatigue_index_MA],
            "MR": state.data.userdata[wrapped.fatigue_index_MR],
            "MF": state.data.userdata[wrapped.fatigue_index_MF],
        }
        expected = wrapped.muscle_fatigue.compute_act(
            env._normalize_action(action)[mask], fatigue_state=prev
        )["MA"]
        next_state = wrapped.step(state, action)
        np.testing.assert_allclose(
            np.asarray(next_state.data.ctrl)[mask], np.asarray(expected), atol=1e-6
        )

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
        plain = _make_pose_env().reset(rng)

        # The fatigue arrays go into their own "fatigue_state" obs key.
        state = wrapped.reset(rng)
        assert state.obs["fatigue_state"].shape[0] == nu
        assert state.obs["state"].shape == plain.obs["state"].shape

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
