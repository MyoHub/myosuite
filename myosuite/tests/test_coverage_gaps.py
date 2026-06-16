# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Coverage-gap tests for muscle_conditions, event_terms, registry, trajectory_io, and reward_terms."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.tier1

# Import only available at module level so ClassVar is recognised by @dataclass.
pytest.importorskip("mujoco", reason="mujoco required for registry tests")

from myosuite.core.config import (  # noqa: E402
    ActuatorGroupSpec,
    BackendConfig,
    GoalSpec,
    ObsSpec,
    RewardSpec,
    TaskConfig,
    VariantSpec,
)


@dataclass
class _MyoVariantTask(TaskConfig):
    """Task with myo-prefixed id and non-empty config_delta for registry variant tests."""

    model: str = "elbow_standard"
    obs: ObsSpec = field(default_factory=lambda: ObsSpec(keys=["joint_pos"]))
    goal: GoalSpec = field(
        default_factory=lambda: GoalSpec(
            target_type="joint_angles",
            randomize=False,
            range={"r_elbow_flex": (1.0, 1.0)},
        )
    )
    reward: RewardSpec = field(default_factory=lambda: RewardSpec(terms=["pose"]))
    actuators: list[ActuatorGroupSpec] = field(
        default_factory=lambda: [ActuatorGroupSpec()]
    )
    max_episode_steps: int = 10
    backend: BackendConfig = field(
        default_factory=lambda: BackendConfig(n_substeps=2, ctrl_dt=0.002)
    )
    some_flag: bool = False
    variants: ClassVar[list[VariantSpec]] = [
        VariantSpec(suffix="Sarc", config_delta={"some_flag": True}),
    ]


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------


class _FakeAccessor:
    """Minimal EnvAccessor stub for event-term tests."""

    def __init__(self, n_act: int = 4) -> None:
        self._n_act = n_act
        self._act = np.zeros(n_act)

    def muscle_act(self) -> np.ndarray:
        return self._act.copy()

    def array_module(self) -> Any:
        return np


class _FakeAccessorWithModel(_FakeAccessor):
    """Accessor that also exposes a mock MjModel (for sarcopenia tests)."""

    def __init__(self, n_act: int = 4, n_actuators: int = 4) -> None:
        super().__init__(n_act)
        self.model = MagicMock()
        self.model.actuator_gainprm = np.ones((n_actuators, 10))


# ---------------------------------------------------------------------------
# muscle_conditions.CumulativeFatigue
# ---------------------------------------------------------------------------


def _make_fatigue_model(n: int) -> CumulativeFatigue:  # noqa: F821
    """Build a CumulativeFatigue with n uniform muscle actuators (no real MjModel needed)."""
    import mujoco
    from myosuite.core.muscle_conditions import CumulativeFatigue

    mock = MagicMock()
    mock.opt.timestep = 0.002
    mock.nu = n
    mock.actuator_dyntype = np.full(n, mujoco.mjtDyn.mjDYN_MUSCLE)
    mock.actuator_dynprm = np.tile([0.01, 0.04] + [0.0] * 8, (n, 1))
    return CumulativeFatigue(mock, use_uniform_params=True)


class TestCumulativeFatigue:
    def test_init_compartments(self) -> None:
        f = _make_fatigue_model(6)
        assert f.MA.shape == (6,)
        assert f.MF.shape == (6,)
        assert f.MR.shape == (6,)
        assert np.all(f.MA == 0.0)
        assert np.all(f.MF == 0.0)
        assert np.all(f.MR == 1.0)

    def test_step_output_shape(self) -> None:
        f = _make_fatigue_model(3)
        out = f.step(np.ones(3) * 0.5, dt=0.01)
        assert out.shape == (3,)

    def test_step_output_within_bounds(self) -> None:
        f = _make_fatigue_model(4)
        for _ in range(50):
            out = f.step(np.random.rand(4), dt=0.01)
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_step_monotone_accumulation(self) -> None:
        f = _make_fatigue_model(2)
        # MA should increase from zero when excitation > MA
        f.step(np.ones(2), dt=1.0)
        assert np.all(f.MA > 0.0), "Active compartment must grow"

    def test_reset_restores_initial_state(self) -> None:
        f = _make_fatigue_model(3)
        for _ in range(20):
            f.step(np.ones(3) * 0.8, dt=0.01)
        f.reset()
        assert np.all(f.MA == 0.0)
        assert np.all(f.MF == 0.0)
        assert np.all(f.MR == 1.0)

    def test_custom_rates(self) -> None:
        f = _make_fatigue_model(2)
        f.set_FatigueCoefficient(0.1)
        f.set_RecoveryCoefficient(0.05)
        assert f.F == pytest.approx(0.1)
        assert f.R == pytest.approx(0.05)


class TestApplySarcopenia:
    def test_apply_sarcopenia_to_model_scales_gainprm(self) -> None:
        from myosuite.core.muscle_conditions import apply_sarcopenia_to_model

        model = MagicMock()
        model.actuator_gainprm = np.ones((5, 10))
        apply_sarcopenia_to_model(model, force_scale=0.5)
        assert np.all(model.actuator_gainprm[:, 2] == pytest.approx(0.5))

    def test_apply_sarcopenia_to_model_default_scale(self) -> None:
        from myosuite.core.muscle_conditions import apply_sarcopenia_to_model

        model = MagicMock()
        model.actuator_gainprm = np.full((3, 10), 2.0)
        apply_sarcopenia_to_model(model)
        assert np.all(model.actuator_gainprm[:, 2] == pytest.approx(1.0))

    def test_apply_sarcopenia_to_spec(self) -> None:
        from myosuite.core.muscle_conditions import apply_sarcopenia_to_spec

        act1 = MagicMock()
        act1.forcerange = [100.0, 500.0]
        act2 = MagicMock()
        act2.forcerange = [200.0, 800.0]
        spec = MagicMock()
        spec.actuators = [act1, act2]
        returned = apply_sarcopenia_to_spec(spec, force_scale=0.5)
        assert returned is spec
        assert act1.forcerange[0] == pytest.approx(50.0)
        assert act1.forcerange[1] == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# myo_event_terms: apply_sarcopenia
# ---------------------------------------------------------------------------


class TestApplySarcopeniaEventTerm:
    def test_apply_sarcopenia_scales_model(self) -> None:
        from myosuite.terms.base_event import apply_sarcopenia

        acc = _FakeAccessorWithModel(n_actuators=4)
        state: dict = {}
        apply_sarcopenia(acc, state, force_scale=0.5)
        assert np.all(acc.model.actuator_gainprm[:, 2] == pytest.approx(0.5))

    def test_apply_sarcopenia_no_model_attr_is_noop(self) -> None:
        from myosuite.terms.base_event import apply_sarcopenia

        acc = _FakeAccessor()
        state: dict = {}
        result = apply_sarcopenia(acc, state, force_scale=0.5)
        assert result is state  # unchanged

    def test_apply_sarcopenia_returns_state(self) -> None:
        from myosuite.terms.base_event import apply_sarcopenia

        acc = _FakeAccessorWithModel()
        state = {"key": 42}
        result = apply_sarcopenia(acc, state)
        assert result is state
        assert result["key"] == 42


# ---------------------------------------------------------------------------
# core/trajectory_io: resolve_motion_path additional branches
# ---------------------------------------------------------------------------


def _write_npz(path: Path, nq: int, nv: int) -> None:
    np.savez(
        path,
        qpos=np.zeros((4, nq), dtype=np.float64),
        qvel=np.zeros((4, nv), dtype=np.float64),
        site_xpos=np.zeros((4, 3), dtype=np.float64),
        frequency=np.array(120.0, dtype=np.float64),
    )


class TestResolveMotionPath:
    def test_empty_string_raises(self) -> None:
        from myosuite.core.trajectory_io import resolve_motion_path

        with pytest.raises(ValueError, match="empty"):
            resolve_motion_path("")

    def test_absolute_path_resolved(self, tmp_path: Path) -> None:
        from myosuite.core.trajectory_io import resolve_motion_path

        p = tmp_path / "clip.npz"
        _write_npz(p, nq=3, nv=3)
        got = resolve_motion_path(str(p))
        assert got == p.resolve()

    def test_cwd_relative_path_resolved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from myosuite.core.trajectory_io import resolve_motion_path

        p = tmp_path / "relclip.npz"
        _write_npz(p, nq=2, nv=2)
        monkeypatch.chdir(tmp_path)
        got = resolve_motion_path("relclip.npz")
        assert got == p.resolve()

    def test_cache_root_with_npz_extension(self, tmp_path: Path) -> None:
        from myosuite.core.trajectory_io import resolve_motion_path

        env_name = "TestEnv"
        cache_file = tmp_path / env_name / "gmr" / "clip.npz"
        cache_file.parent.mkdir(parents=True)
        _write_npz(cache_file, nq=2, nv=2)
        got = resolve_motion_path("clip.npz", env_name=env_name, cache_root=tmp_path)
        assert got == cache_file.resolve()

    def test_cache_root_auto_appends_npz(self, tmp_path: Path) -> None:
        from myosuite.core.trajectory_io import resolve_motion_path

        env_name = "TestEnv2"
        cache_file = tmp_path / env_name / "gmr" / "clip.npz"
        cache_file.parent.mkdir(parents=True)
        _write_npz(cache_file, nq=2, nv=2)
        # Pass without .npz extension → should auto-append
        got = resolve_motion_path("clip", env_name=env_name, cache_root=tmp_path)
        assert got == cache_file.resolve()

    def test_not_found_raises(self, tmp_path: Path) -> None:
        from myosuite.core.trajectory_io import resolve_motion_path

        with pytest.raises(FileNotFoundError):
            resolve_motion_path("nonexistent.npz", cache_root=tmp_path)


class TestLoadMotionClip:
    def test_full_clip_loaded(self, tmp_path: Path) -> None:
        from myosuite.core.trajectory_io import load_motion_clip

        p = tmp_path / "full.npz"
        _write_npz(p, nq=5, nv=4)
        clip = load_motion_clip(p, expected_nq=5, expected_nv=4)
        assert clip.qpos.shape == (4, 5)
        assert clip.qvel is not None and clip.qvel.shape == (4, 4)
        assert clip.site_xpos is not None
        assert clip.frequency_hz == pytest.approx(120.0)

    def test_missing_qpos_raises(self, tmp_path: Path) -> None:
        from myosuite.core.trajectory_io import load_motion_clip

        p = tmp_path / "no_qpos.npz"
        np.savez(p, qvel=np.zeros((4, 3)))
        with pytest.raises(KeyError, match="qpos"):
            load_motion_clip(p, expected_nq=3, expected_nv=3)

    def test_1d_qpos_raises(self, tmp_path: Path) -> None:
        from myosuite.core.trajectory_io import load_motion_clip

        p = tmp_path / "rank1.npz"
        np.savez(p, qpos=np.zeros(4))
        with pytest.raises(ValueError, match="rank-2"):
            load_motion_clip(p, expected_nq=4, expected_nv=3)

    def test_qvel_wrong_width_raises(self, tmp_path: Path) -> None:
        from myosuite.core.trajectory_io import load_motion_clip

        p = tmp_path / "bad_qvel.npz"
        np.savez(p, qpos=np.zeros((4, 3)), qvel=np.zeros((4, 5)))
        with pytest.raises(ValueError, match="qvel width mismatch"):
            load_motion_clip(p, expected_nq=3, expected_nv=3)

    def test_no_optional_keys(self, tmp_path: Path) -> None:
        from myosuite.core.trajectory_io import load_motion_clip

        p = tmp_path / "minimal.npz"
        np.savez(p, qpos=np.zeros((4, 3)))
        clip = load_motion_clip(p, expected_nq=3, expected_nv=3)
        assert clip.qvel is None
        assert clip.site_xpos is None
        assert clip.frequency_hz is None

    def test_bad_frequency_handled(self, tmp_path: Path) -> None:
        from myosuite.core.trajectory_io import load_motion_clip

        p = tmp_path / "bad_freq.npz"
        # frequency as a multi-element array → reshape(()) fails → None
        np.savez(p, qpos=np.zeros((4, 3)), frequency=np.array([1.0, 2.0]))
        clip = load_motion_clip(p, expected_nq=3, expected_nv=3)
        assert clip.frequency_hz is None


# ---------------------------------------------------------------------------
# registry: make_env MJX/mjlab import-error paths
# ---------------------------------------------------------------------------


class TestMakeEnvBackends:
    def test_make_env_mjx_import_error(self) -> None:
        from myosuite.core.registry import make_env

        with patch.dict("sys.modules", {"mujoco_playground": None}):
            with pytest.raises(ImportError, match="mujoco_playground"):
                make_env("NonExistent-v0", backend="mjx")

    def test_make_env_mjlab_import_error(self) -> None:
        from myosuite.core.registry import make_env

        with patch.dict("sys.modules", {"mjlab": None, "mjlab.envs": None}):
            with pytest.raises(ImportError, match="mjlab"):
                make_env("NonExistent-v0", backend="mjlab")


# ---------------------------------------------------------------------------
# registry: variant expansion with "myo" prefix and non-empty config_delta
# ---------------------------------------------------------------------------


class TestRegistryVariantPaths:
    def test_myo_prefix_variant_id_format(self) -> None:
        """A 'myo'-prefixed base_env_id generates 'myoSarc...-v0' variant id."""
        import gymnasium as gym
        from myosuite.core.registry import register_task

        base_id = "myoTestElbowVariant-v0"
        register_task(_MyoVariantTask(), env_id=base_id)
        assert base_id in gym.envs.registry
        variant_id = "myoSarcTestElbowVariant-v0"
        assert variant_id in gym.envs.registry

    def test_nonempty_config_delta_registers_variant(self) -> None:
        """VariantSpec with non-empty config_delta must register the variant env_id."""
        import gymnasium as gym
        from myosuite.core.registry import _ENV_REGISTRY, register_task

        base_id = "myoTestElbowDelta-v0"
        register_task(_MyoVariantTask(), env_id=base_id)
        variant_id = "myoSarcTestElbowDelta-v0"
        assert variant_id in _ENV_REGISTRY
        assert variant_id in gym.envs.registry


# ---------------------------------------------------------------------------
# myo_reward_terms: walk_env_reward
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# myo_action_terms: MuscleActionTerm (mjlab-style)
# ---------------------------------------------------------------------------


class TestMuscleActionTerm:
    def _make_term(self, normalize: bool = True) -> Any:
        from myosuite.terms.base_action import (
            MuscleActionTerm,
            MuscleActionTermCfg,
        )

        mock_entity = MagicMock()
        mock_entity.num_actuators = 4
        mock_env = MagicMock()
        mock_env.scene = {"robot": mock_entity}
        cfg = MuscleActionTermCfg(entity_name="robot", normalize=normalize)
        return MuscleActionTerm(cfg, mock_env), mock_entity

    def test_action_dim(self) -> None:
        term, entity = self._make_term()
        assert term.action_dim == 4

    def test_process_actions_normalize_true(self) -> None:
        term, _ = self._make_term(normalize=True)
        actions = np.array([[-1.0, 0.0, 1.0, 2.0]])
        term.process_actions(actions)
        expected = 1.0 / (1.0 + np.exp(-5.0 * (actions - 0.5)))
        np.testing.assert_allclose(term._processed, expected)

    def test_process_actions_normalize_false(self) -> None:
        term, _ = self._make_term(normalize=False)
        actions = np.array([[-1.0, 0.5, 1.0, 2.0]])
        term.process_actions(actions)
        expected = np.clip(actions, 0, 1)
        np.testing.assert_allclose(term._processed, expected)

    def test_apply_actions_calls_set_ctrl(self) -> None:
        term, entity = self._make_term()
        actions = np.array([[0.2, 0.4, 0.6, 0.8]])
        term.process_actions(actions)
        term.apply_actions()
        entity.set_ctrl.assert_called_once()

    def test_apply_actions_no_op_when_not_processed(self) -> None:
        term, entity = self._make_term()
        term.apply_actions()  # _processed is None
        entity.set_ctrl.assert_not_called()


class TestWalkEnvReward:
    def _make_task_state(self) -> dict:
        qpos = np.zeros(30)
        qpos[3] = 1.0  # quaternion w=1
        return {
            "qpos": qpos,
            "height": 1.0,
            "com_vel": np.array([0.0, 1.0]),
        }

    def _make_accessor(self) -> _FakeAccessor:
        return _FakeAccessor()

    def test_returns_required_keys(self) -> None:
        from myosuite.terms.base_reward import walk_env_reward

        acc = self._make_accessor()
        state = self._make_task_state()
        result = walk_env_reward(
            acc,
            state,
            hip_flex_indices=(10, 15),
            hip_angle_indices=(11, 16, 12, 17),
            target_rot=np.array([1.0, 0.0, 0.0, 0.0]),
            target_vel=(0.0, 1.2),
            min_height=0.8,
            max_rot=0.8,
        )
        for key in (
            "vel_reward",
            "done",
            "solved",
            "cyclic_hip",
            "ref_rot",
            "joint_angle_rew",
            "dense",
        ):
            assert key in result, f"Missing key: {key}"

    def test_terminates_on_low_height(self) -> None:
        from myosuite.terms.base_reward import walk_env_reward

        acc = self._make_accessor()
        state = self._make_task_state()
        state["height"] = 0.1  # below min_height=0.8
        result = walk_env_reward(
            acc,
            state,
            hip_flex_indices=(10, 15),
            hip_angle_indices=(11, 16, 12, 17),
            target_rot=np.array([1.0, 0.0, 0.0, 0.0]),
            min_height=0.8,
            max_rot=0.8,
        )
        assert float(result["done"]) != 0.0

    def test_com_vel_indices_branch(self) -> None:
        from myosuite.terms.base_reward import walk_env_reward

        acc = self._make_accessor()
        state = self._make_task_state()
        state["com_vel"] = np.array([0.0, 0.5, 1.0])  # 3-element vector
        result = walk_env_reward(
            acc,
            state,
            hip_flex_indices=(10, 15),
            hip_angle_indices=(11, 16, 12, 17),
            target_rot=np.array([1.0, 0.0, 0.0, 0.0]),
            com_vel_indices=(0, 2),  # vx=com_vel[0], vy=com_vel[2]
        )
        assert "vel_reward" in result

    def test_com_height_index_branch(self) -> None:
        from myosuite.terms.base_reward import walk_env_reward

        acc = self._make_accessor()
        state = self._make_task_state()
        state["height_like"] = np.array([0.5, 1.2, 0.9])
        result = walk_env_reward(
            acc,
            state,
            hip_flex_indices=(10, 15),
            hip_angle_indices=(11, 16, 12, 17),
            target_rot=np.array([1.0, 0.0, 0.0, 0.0]),
            com_height_index=1,  # use height_like[1]=1.2
        )
        assert "dense" in result
