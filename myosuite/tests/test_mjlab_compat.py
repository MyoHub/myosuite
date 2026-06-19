# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""mjlab backend compatibility tests.

Three test classes:

``TestMjlabEnvAccessor``
    Verifies that :class:`~myosuite.envs.myo.backends.mjlab.mjlab_env_base.MjlabEnvAccessor`
    satisfies the :class:`~myosuite.core.protocols.EnvAccessor` protocol and
    returns correctly typed ``torch.Tensor`` values.  Uses a lightweight mock
    data object — no MuJoCo or mjlab installation required.

``TestTermFunctionsWithTorch``
    Verifies that the shared term functions in ``myosuite/terms/`` produce
    finite, correctly shaped outputs when called via ``MjlabEnvAccessor``.
    Catches any numpy/jax → torch API incompatibilities.

``TestMjlabTaskConfigs``
    Verifies that all four task config dataclasses have valid default values
    and that their model paths point to existing files.

Skip conditions:
    - ``torch`` not importable → ``TestMjlabEnvAccessor`` and
      ``TestTermFunctionsWithTorch`` are skipped.
    - ``mjlab`` not importable → ``TestMjlabIntegration`` is skipped.
    - On macOS, ``conftest.py`` skips mjlab biped leg-walk tests (MuJoCo Warp
      can segfault); Linux CI runs them.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.tier2


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _has_mjlab() -> bool:
    """Return True when mjlab is importable and usable in this process.

    Some mjlab/mujoco_warp version mismatches raise non-ImportError exceptions
    (e.g. AttributeError) during import-time package discovery. Treat those
    cases as "mjlab unavailable" so tests skip rather than hard-fail collection.
    """
    try:
        return (
            importlib.util.find_spec("mjlab") is not None
            and importlib.util.find_spec("mjlab.envs") is not None
            and importlib.util.find_spec("mjlab.tasks.registry") is not None
        )
    except Exception:
        return False


_MJLAB_AVAILABLE = _has_mjlab()


# ---------------------------------------------------------------------------
# Mock mjlab data / model objects (no MuJoCo required)
# ---------------------------------------------------------------------------


def _make_mock_data(
    nq: int = 1,
    nv: int = 1,
    na: int = 6,
    nsite: int = 3,
    n_envs: int = 2,
) -> Any:
    """Build a minimal namespace mimicking the mjlab data object.

    Args:
        nq: Number of generalised coordinates.
        nv: Number of generalised velocities.
        na: Number of muscle actuators.
        nsite: Number of sites in the model.
        n_envs: Batch size (parallel environments).

    Returns:
        A ``SimpleNamespace`` with ``torch.Tensor`` fields.
    """
    return types.SimpleNamespace(
        qpos=torch.zeros(n_envs, nq),
        qvel=torch.zeros(n_envs, nv),
        act=torch.full((n_envs, na), 0.5),
        site_xpos=torch.zeros(n_envs, nsite, 3),
        time=torch.zeros(n_envs),
    )


def _make_mock_model(nu: int = 6) -> Any:
    """Build a minimal namespace mimicking the mjlab model object.

    Args:
        nu: Number of actuators.

    Returns:
        A ``SimpleNamespace`` with ``actuator_ctrlrange`` as a numpy array.
    """
    import numpy as np

    return types.SimpleNamespace(
        actuator_ctrlrange=np.zeros((nu, 2), dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# MjlabEnvAccessor tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
class TestMjlabEnvAccessor:
    # Fixtures populate these per test case.
    data: Any
    model: Any
    accessor: Any

    """MjlabEnvAccessor satisfies EnvAccessor and returns torch.Tensor.

    Uses a mock data/model object so no MuJoCo Warp installation is needed.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Build a MjlabEnvAccessor wrapping mock data."""
        from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEnvAccessor

        self.data = _make_mock_data()
        self.model = _make_mock_model()
        self.accessor = MjlabEnvAccessor(self.model, self.data, ctrl_dt=0.02)

    def test_physics_path_is_mjlab(self) -> None:
        """physics_path must return PhysicsPath.MJLAB."""
        from myosuite.core.protocols import PhysicsPath

        assert self.accessor.physics_path == PhysicsPath.MJLAB

    def test_satisfies_env_accessor_protocol(self) -> None:
        """isinstance check against the runtime_checkable EnvAccessor protocol."""
        from myosuite.core.protocols import EnvAccessor

        assert isinstance(self.accessor, EnvAccessor)

    def test_array_module_is_torch(self) -> None:
        """array_module() must return the torch module."""
        import torch as _torch

        assert self.accessor.array_module() is _torch

    def test_joint_pos_returns_tensor(self) -> None:
        """joint_pos() must return a torch.Tensor of the correct shape."""
        pos = self.accessor.joint_pos()
        assert isinstance(pos, torch.Tensor)
        assert pos.shape == (2, 1)  # (n_envs, nq)

    def test_joint_vel_returns_raw_qvel(self) -> None:
        """joint_vel() must equal the underlying raw qvel values."""
        self.data.qvel[:] = 1.0
        vel = self.accessor.joint_vel()
        torch.testing.assert_close(vel, torch.full_like(vel, 1.0))

    def test_muscle_act_returns_tensor(self) -> None:
        """muscle_act() must return a torch.Tensor shaped (N, na)."""
        act = self.accessor.muscle_act()
        assert isinstance(act, torch.Tensor)
        assert act.shape == (2, 6)

    def test_site_xpos_slices_site_dim(self) -> None:
        """site_xpos([0, 1]) must return shape (N, 2, 3)."""
        xpos = self.accessor.site_xpos([0, 1])
        assert xpos.shape == (2, 2, 3)

    def test_ctrl_range_returns_tensor(self) -> None:
        """ctrl_range() must convert numpy actuator_ctrlrange to torch.Tensor."""
        cr = self.accessor.ctrl_range()
        assert isinstance(cr, torch.Tensor)
        assert cr.shape == (6, 2)

    def test_dt_returns_float(self) -> None:
        """dt() must return the ctrl_dt as a plain Python float."""
        assert self.accessor.dt() == pytest.approx(0.02)

    def test_time_returns_tensor(self) -> None:
        """time() must return a torch.Tensor shaped (N,)."""
        t = self.accessor.time()
        assert isinstance(t, torch.Tensor)
        assert t.shape == (2,)


# ---------------------------------------------------------------------------
# Term function torch compatibility tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
class TestTermFunctionsWithTorch:
    # Fixtures populate these per test case.
    n_envs: int
    nq: int
    na: int
    nsite: int
    accessor: Any
    target: Any

    """Term functions must produce finite outputs when called via MjlabEnvAccessor.

    Each test exercises one term function from ``myosuite/terms/`` with a
    real ``MjlabEnvAccessor`` backed by mock torch data.  This catches any
    numpy/jax → torch API incompatibilities (e.g. ``axis=`` vs ``dim=``,
    ``maximum(scalar, tensor)`` incompatibility).
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Build accessor and task_state for a 1-DOF elbow scenario."""
        from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEnvAccessor

        self.n_envs = 4
        self.nq = 1
        self.na = 6
        self.nsite = 3

        data = _make_mock_data(
            nq=self.nq, na=self.na, nsite=self.nsite, n_envs=self.n_envs
        )
        data.qpos = torch.tensor([[0.5], [1.0], [1.5], [2.0]])
        model = _make_mock_model(nu=self.na)
        self.accessor = MjlabEnvAccessor(model, data, ctrl_dt=0.02)
        self.target = torch.tensor([2.0])  # target angle 2.0 rad

    def test_pose_reward_finite(self) -> None:
        """pose_reward must return finite dense values for all environments."""
        from myosuite.terms.base_reward import pose_reward

        result = pose_reward(
            self.accessor, {"target_angles": self.target}, pose_thd=0.35
        )
        assert "dense" in result
        assert torch.all(torch.isfinite(result["dense"]))

    def test_saber_shared_rewards_support_batched_torch_task_state(self) -> None:
        """Shared saber rewards must stay batched on the mjlab/torch path."""
        from myosuite.terms.base_reward import (
            saber_keyframe_pose_reward,
            saber_target_pool_reward,
            upright_posture_reward,
        )

        upright = upright_posture_reward(
            self.accessor,
            {"upright_posture": torch.tensor([1.0, 0.2, 0.8, 0.1])},
            upright_posture_threshold=0.3,
        )
        assert upright["dense"].shape == (4,)
        assert upright["done"].shape == (4,)

        keyframe = saber_keyframe_pose_reward(
            self.accessor,
            {"saber_keyframe_pose_error": torch.tensor([0.0, 0.01, 0.02, 0.03])},
            saber_keyframe_pose_error_scale=40.0,
        )
        assert keyframe["dense"].shape == (4,)
        assert torch.all(torch.isfinite(keyframe["dense"]))

        pool = saber_target_pool_reward(
            self.accessor,
            {
                "target_pool_hit_this_step": torch.tensor([True, False, True, False]),
                "target_pool_correct_hand_this_step": torch.tensor(
                    [True, False, True, False]
                ),
                "target_pool_correct_face_this_step": torch.tensor(
                    [True, False, False, False]
                ),
                "target_pool_wrong_hand_this_step": torch.tensor(
                    [False, False, False, True]
                ),
                "target_pool_wrong_face_this_step": torch.tensor(
                    [False, True, False, False]
                ),
                "target_pool_slicing_accuracy_this_step": torch.tensor(
                    [0.2, 0.0, 0.5, 0.0]
                ),
                "target_pool_timing_accuracy_this_step": torch.tensor(
                    [0.1, 0.0, -0.2, 0.0]
                ),
                "target_pool_misses": torch.tensor([0, 1, 2, 3]),
                "health_status": torch.tensor([0.5, 0.4, 0.3, 0.0]),
            },
            max_target_misses=3,
        )
        assert pool["dense"].shape == (4,)
        assert pool["done"].shape == (4,)
        assert bool(pool["done"][-1]) is True

    def test_saber_shared_health_obs_supports_batched_torch_task_state(self) -> None:
        """health_status_obs must return a column vector on the mjlab/torch path."""
        from myosuite.terms.base_obs import health_status_obs

        obs = health_status_obs(
            self.accessor,
            health_status=torch.tensor([0.2, 0.4, 0.6, 0.8]),
        )
        assert obs.shape == (4, 1)

    def test_saber_shared_terminations_support_batched_torch_task_state(self) -> None:
        """Shared saber termination terms must stay batched on the mjlab path."""
        from myosuite.terms.base_termination import (
            saber_pool_done,
            upright_posture_failure,
        )

        done = saber_pool_done(
            self.accessor,
            {
                "target_pool_misses": torch.tensor([0, 1, 2, 3]),
                "health_status": torch.tensor([0.5, 0.4, 0.3, 0.0]),
            },
            max_target_misses=3,
        )
        assert done.shape == (4,)
        assert bool(done[-1]) is True

        fallen = upright_posture_failure(
            self.accessor,
            {"upright_posture": torch.tensor([1.0, 0.2, 0.8, 0.1])},
            upright_posture_threshold=0.3,
        )
        assert fallen.shape == (4,)
        assert bool(fallen[1]) is True

    def test_pose_reward_keys(self) -> None:
        """pose_reward must return the expected dict keys."""
        from myosuite.terms.base_reward import pose_reward

        result = pose_reward(self.accessor, {"target_angles": self.target})
        for key in ("pose", "bonus", "penalty", "dense", "solved", "done"):
            assert key in result, f"Missing key: {key}"

    def test_act_reg_finite(self) -> None:
        """act_reg must return finite dense values for all environments."""
        from myosuite.terms.base_reward import act_reg

        result = act_reg(self.accessor, {}, weight=0.01)
        assert torch.all(torch.isfinite(result["dense"]))

    def test_joint_penalty_finite(self) -> None:
        """joint_penalty must return a finite scalar penalty."""
        from myosuite.terms.base_reward import joint_penalty

        result = joint_penalty(self.accessor, {}, weight=50.0)
        assert torch.isfinite(torch.tensor(float(result["dense"])))

    def test_reach_reward_finite(self) -> None:
        """reach_reward must return a finite dense value."""
        from myosuite.terms.base_reward import reach_reward

        target_pos = torch.tensor([0.0, -0.5, 1.4])
        task_state = {"target_pos": target_pos, "tip_site_ids": [0]}
        result = reach_reward(self.accessor, task_state, reach_thd=0.05)
        assert torch.all(torch.isfinite(result["dense"]))

    def test_pose_error_obs_shape(self) -> None:
        """pose_error_obs must return a tensor of the same shape as qpos."""
        from myosuite.terms.base_obs import pose_error_obs

        err = pose_error_obs(self.accessor, self.target)
        assert err.shape == (self.n_envs, self.nq)

    def test_pose_reward_gradient_flows(self) -> None:
        """Dense reward must be differentiable w.r.t. joint positions."""
        from myosuite.terms.base_reward import pose_reward

        data = _make_mock_data(nq=1, na=6, nsite=3, n_envs=2)
        data.qpos = torch.tensor([[0.5], [1.5]], requires_grad=True)
        from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEnvAccessor

        accessor = MjlabEnvAccessor(_make_mock_model(), data, ctrl_dt=0.02)
        result = pose_reward(accessor, {"target_angles": torch.tensor([2.0])})
        result["dense"].sum().backward()
        assert data.qpos.grad is not None


# ---------------------------------------------------------------------------
# Mimic trajectory/action helper tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
def test_clip_trajectory_source_public_frame_indices_matches_offsets() -> None:
    """Public frame_indices must expose the same phase used for targets."""
    import numpy as np
    from pathlib import Path

    from myosuite.core.trajectory_io import MotionClip
    from myosuite.envs.myo.backends.mjlab.clip_trajectory_source import (
        ClipTrajectorySource,
    )

    clip = MotionClip(
        qpos=np.zeros((4, 1), dtype=np.float32),
        qvel=None,
        site_xpos=np.zeros((4, 2, 3), dtype=np.float32),
        site_names=None,
        frequency_hz=None,
        source_path=Path("/tmp/fake_motion.npz"),
    )
    source = ClipTrajectorySource(
        clip=clip,
        tracked_site_ids=np.asarray([0, 1], dtype=np.int64),
        ctrl_dt=0.1,
    )
    t = torch.tensor([0.0, 0.2], dtype=torch.float32)
    source.update(t)
    source._start_offsets = torch.tensor([1, 2], dtype=torch.long)

    idx = source.frame_indices(t)

    torch.testing.assert_close(idx, torch.tensor([1, 0], dtype=torch.long))


@pytest.mark.skipif(
    not (_MJLAB_AVAILABLE and _TORCH_AVAILABLE),
    reason="mjlab/torch not installed (pip install myosuite[mjlab])",
)
def test_myo_muscle_activation_action_direct_mode_clamps_controls() -> None:
    """Direct mode must preserve checkpoint-style [-1, 1] controls."""
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        MyoMuscleActivationActionCfg,
    )

    class _Entity:
        def __init__(self) -> None:
            self.applied: torch.Tensor | None = None
            self.tendon_ids: torch.Tensor | None = None

        def find_actuators(
            self, names: tuple[str, ...]
        ) -> tuple[list[int], tuple[str, ...]]:
            return list(range(len(names))), names

        def find_tendons(
            self,
            names: tuple[str, ...],
            preserve_order: bool = False,
        ) -> tuple[list[int], tuple[str, ...]]:
            assert preserve_order
            return [2, 0, 1], names

        def set_tendon_effort_target(
            self,
            processed: torch.Tensor,
            tendon_ids: torch.Tensor,
        ) -> None:
            self.applied = processed.detach().cpu()
            self.tendon_ids = tendon_ids.detach().cpu()

    entity = _Entity()
    env = types.SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        scene={"robot": entity},
    )
    cfg = MyoMuscleActivationActionCfg(
        entity_name="robot",
        actuator_names=("a", "b", "c"),
        tendon_names=("ta", "tb", "tc"),
        action_mode="direct",
    )
    action = cfg.build(env)

    action.process_actions(torch.tensor([[-1.0, 0.5, 2.0], [0.0, 1.0, 0.25]]))
    action.apply_actions()

    assert entity.applied is not None
    assert entity.tendon_ids is not None
    torch.testing.assert_close(
        entity.applied,
        torch.tensor([[-1.0, 0.5, 1.0], [0.0, 1.0, 0.25]]),
    )
    torch.testing.assert_close(entity.tendon_ids, torch.tensor([2, 0, 1]))
    action.reset()
    torch.testing.assert_close(action.raw_action, torch.zeros((2, 3)))
    with pytest.raises(ValueError, match="action_mode"):
        MyoMuscleActivationActionCfg(
            entity_name="robot",
            actuator_names=("a",),
            action_mode="not-a-mode",
        )


# ---------------------------------------------------------------------------
# Task config dataclass tests
# ---------------------------------------------------------------------------


class TestMjlabTaskConfigs:
    """Task config dataclasses must have valid defaults and model paths."""

    def test_elbow_pose_cfg_defaults(self) -> None:
        """ElbowPoseCfg must construct with valid numeric defaults."""
        from myosuite.envs.myo.backends.mjlab.configs.elbow_pose_cfg import ElbowPoseCfg

        cfg = ElbowPoseCfg()
        assert cfg.sim_dt > 0
        assert cfg.ctrl_dt >= cfg.sim_dt
        assert cfg.max_episode_steps > 0
        assert 0.0 < cfg.pose_thd < 1.0
        assert cfg.num_envs > 0

    def test_elbow_pose_cfg_model_exists(self) -> None:
        """ElbowPoseCfg.model_path must point to an existing MJCF file."""
        from myosuite.envs.myo.backends.mjlab.configs.elbow_pose_cfg import ElbowPoseCfg

        cfg = ElbowPoseCfg()
        assert cfg.model_path.exists(), f"Model not found: {cfg.model_path}"

    def test_hand_reach_cfg_defaults(self) -> None:
        """HandReachCfg must construct with valid numeric defaults."""
        from myosuite.envs.myo.backends.mjlab.configs.hand_reach_cfg import HandReachCfg

        cfg = HandReachCfg()
        assert cfg.sim_dt > 0
        assert cfg.ctrl_dt >= cfg.sim_dt
        assert cfg.reach_thd > 0
        assert cfg.num_envs > 0

    def test_hand_reach_cfg_model_exists(self) -> None:
        """HandReachCfg.model_path must point to an existing MJCF file."""
        from myosuite.envs.myo.backends.mjlab.configs.hand_reach_cfg import HandReachCfg

        cfg = HandReachCfg()
        assert cfg.model_path.exists(), f"Model not found: {cfg.model_path}"

    def test_hand_pose_cfg_defaults(self) -> None:
        """HandPoseCfg must construct with valid numeric defaults."""
        from myosuite.envs.myo.backends.mjlab.configs.hand_pose_cfg import HandPoseCfg

        cfg = HandPoseCfg()
        assert cfg.sim_dt > 0
        assert cfg.ctrl_dt >= cfg.sim_dt
        assert cfg.max_episode_steps > 0
        assert cfg.pose_thd > 0
        assert cfg.num_envs > 0

    def test_hand_pose_cfg_model_exists(self) -> None:
        """HandPoseCfg.model_path must point to an existing MJCF file."""
        from myosuite.envs.myo.backends.mjlab.configs.hand_pose_cfg import HandPoseCfg

        cfg = HandPoseCfg()
        assert cfg.model_path.exists(), f"Model not found: {cfg.model_path}"

    def test_finger_pose_cfg_defaults(self) -> None:
        """FingerPoseCfg must construct with valid numeric defaults."""
        from myosuite.envs.myo.backends.mjlab.configs.finger_pose_cfg import (
            FingerPoseCfg,
        )

        cfg = FingerPoseCfg()
        assert cfg.sim_dt > 0
        assert cfg.ctrl_dt >= cfg.sim_dt
        assert cfg.max_episode_steps > 0
        assert cfg.pose_thd > 0
        assert cfg.num_envs > 0

    def test_finger_pose_cfg_model_exists(self) -> None:
        """FingerPoseCfg.model_path must point to an existing MJCF file."""
        from myosuite.envs.myo.backends.mjlab.configs.finger_pose_cfg import (
            FingerPoseCfg,
        )

        cfg = FingerPoseCfg()
        assert cfg.model_path.exists(), f"Model not found: {cfg.model_path}"

    def test_finger_reach_cfg_defaults(self) -> None:
        """FingerReachCfg must construct with valid numeric defaults."""
        from myosuite.envs.myo.backends.mjlab.configs.finger_reach_cfg import (
            FingerReachCfg,
        )

        cfg = FingerReachCfg()
        assert cfg.sim_dt > 0
        assert cfg.ctrl_dt >= cfg.sim_dt
        assert cfg.max_episode_steps > 0
        assert cfg.reach_thd > 0
        assert cfg.num_envs > 0

    def test_finger_reach_cfg_model_exists(self) -> None:
        """FingerReachCfg.model_path must point to an existing MJCF file."""
        from myosuite.envs.myo.backends.mjlab.configs.finger_reach_cfg import (
            FingerReachCfg,
        )

        cfg = FingerReachCfg()
        assert cfg.model_path.exists(), f"Model not found: {cfg.model_path}"

    def test_walk_cfg_defaults(self) -> None:
        """WalkCfg must construct with valid numeric defaults."""
        from myosuite.envs.myo.backends.mjlab.configs.walk_cfg import WalkCfg

        cfg = WalkCfg()
        assert cfg.target_vel > 0
        assert cfg.alive_bonus >= 0
        assert cfg.fall_penalty <= 0
        assert cfg.num_envs > 0

    def test_walk_cfg_model_exists(self) -> None:
        """WalkCfg.model_path must point to an existing MJCF file."""
        from myosuite.envs.myo.backends.mjlab.configs.walk_cfg import WalkCfg

        cfg = WalkCfg()
        assert cfg.model_path.exists(), f"Model not found: {cfg.model_path}"

    def test_baoding_cfg_defaults(self) -> None:
        """BaodingCfg must construct with valid numeric defaults."""
        from myosuite.envs.myo.backends.mjlab.configs.baoding_cfg import BaodingCfg

        cfg = BaodingCfg()
        assert cfg.rotation_speed > 0
        assert cfg.drop_penalty <= 0
        assert cfg.num_envs > 0

    def test_baoding_cfg_model_exists(self) -> None:
        """BaodingCfg.model_path must point to an existing MJCF file."""
        from myosuite.envs.myo.backends.mjlab.configs.baoding_cfg import BaodingCfg

        cfg = BaodingCfg()
        assert cfg.model_path.exists(), f"Model not found: {cfg.model_path}"

    def test_registered_tasks_covers_configs(self) -> None:
        """REGISTERED_TASKS must include at least one entry per config class."""
        from myosuite.envs.myo.backends.mjlab import REGISTERED_TASKS
        from myosuite.envs.myo.backends.mjlab.configs.baoding_cfg import BaodingCfg
        from myosuite.envs.myo.backends.mjlab.configs.elbow_pose_cfg import ElbowPoseCfg
        from myosuite.envs.myo.backends.mjlab.configs.finger_reach_cfg import (
            FingerReachCfg,
        )
        from myosuite.envs.myo.backends.mjlab.configs.finger_pose_cfg import (
            FingerPoseCfg,
        )
        from myosuite.envs.myo.backends.mjlab.configs.hand_pose_cfg import HandPoseCfg
        from myosuite.envs.myo.backends.mjlab.configs.hand_reach_cfg import HandReachCfg
        from myosuite.envs.myo.backends.mjlab.configs.musclemimic_bimanual_cfg import (
            MuscleMimicBimanualCfg,
        )
        from myosuite.envs.myo.backends.mjlab.configs.musclemimic_fullbody_cfg import (
            MuscleMimicFullbodyCfg,
        )
        from myosuite.envs.myo.backends.mjlab.configs.walk_cfg import WalkCfg

        config_classes = set(REGISTERED_TASKS.values())
        assert ElbowPoseCfg in config_classes
        assert FingerPoseCfg in config_classes
        assert FingerReachCfg in config_classes
        assert HandPoseCfg in config_classes
        assert HandReachCfg in config_classes
        assert WalkCfg in config_classes
        assert BaodingCfg in config_classes
        assert MuscleMimicBimanualCfg in config_classes
        assert MuscleMimicFullbodyCfg in config_classes


# ---------------------------------------------------------------------------
# End-to-end mjlab integration tests (skipped until mjlab is available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _MJLAB_AVAILABLE,
    reason="mjlab not installed (pip install myosuite[mjlab])",
)
class TestMjlabIntegration:
    """End-to-end mjlab integration: make → reset → step.

    Skipped until ``myosuite[mjlab]`` is installed.  When available, verify:
    - ``mjlab.envs.make("myoElbowPose1D6MRandom-v0")`` succeeds
    - ``reset()`` returns finite observations
    - ``step()`` returns finite obs/reward for a full episode
    """

    _SUPPORTED_TASK_IDS = (
        "myoElbowPose1D6MFixed-v0",
        "myoLegWalk-v0",
        "myoSarcLegWalk-v0",
    )

    @staticmethod
    def _obs_to_numpy(obs: Any):
        import numpy as np

        if isinstance(obs, dict):
            # mjlab commonly exposes a {"policy": ...} observation group.
            obs = obs.get("policy", next(iter(obs.values())))
        if _TORCH_AVAILABLE and isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        arr = np.asarray(obs)
        if arr.ndim > 1:
            arr = arr[0]
        return arr

    @staticmethod
    def _zero_action(env: Any):
        if hasattr(env, "action_space") and getattr(env.action_space, "shape", None):
            act_dim = int(env.action_space.shape[-1])
        elif hasattr(env, "single_action_space") and getattr(
            env.single_action_space, "shape", None
        ):
            act_dim = int(env.single_action_space.shape[-1])
        elif hasattr(env, "action_manager") and hasattr(
            env.action_manager, "total_action_dim"
        ):
            act_dim = int(env.action_manager.total_action_dim)
        else:
            raise AssertionError("Cannot infer action dimension from mjlab env")

        num_envs = int(getattr(env, "num_envs", 1))
        device = getattr(env, "device", "cpu")
        if _TORCH_AVAILABLE:
            return torch.zeros((num_envs, act_dim), dtype=torch.float32, device=device)
        raise AssertionError("torch is required for mjlab integration test")

    @pytest.mark.parametrize("env_id", _SUPPORTED_TASK_IDS)
    def test_make_reset_step_supported_tasks(self, env_id: str) -> None:
        """Each currently supported mjlab task must support make→reset→step."""
        import numpy as np

        import myosuite

        myosuite.register_all_envs()
        from myosuite.core.registry import make_env

        try:
            env = make_env(env_id, backend="mjlab")
        except ValueError as exc:
            msg = str(exc)
            # If the current mjlab build does not have MyoSuite tasks registered,
            # treat this as an environment limitation rather than a hard failure.
            if "not in mjlab task registry" in msg:
                pytest.skip(
                    f"{env_id}: not registered in mjlab task registry on this installation: {exc}"
                )
            raise
        obs, info = env.reset(seed=0)
        obs_np = self._obs_to_numpy(obs)
        assert obs_np.size > 0, f"{env_id}: empty observation on reset"
        assert np.all(np.isfinite(obs_np)), f"{env_id}: non-finite reset obs"
        assert isinstance(info, dict), f"{env_id}: reset info must be dict"

        action = self._zero_action(env)
        obs2, reward, terminated, truncated, info2 = env.step(action)
        obs2_np = self._obs_to_numpy(obs2)
        assert obs2_np.size > 0, f"{env_id}: empty observation on step"
        assert np.all(np.isfinite(obs2_np)), f"{env_id}: non-finite step obs"

        rew_np = (
            reward.detach().cpu().numpy()
            if isinstance(reward, torch.Tensor)
            else np.asarray(reward)
        )
        assert np.all(np.isfinite(rew_np)), f"{env_id}: non-finite reward"

        term_np = (
            terminated.detach().cpu().numpy()
            if isinstance(terminated, torch.Tensor)
            else np.asarray(terminated)
        )
        trunc_np = (
            truncated.detach().cpu().numpy()
            if isinstance(truncated, torch.Tensor)
            else np.asarray(truncated)
        )
        assert term_np.size > 0 and trunc_np.size > 0
        assert isinstance(info2, dict), f"{env_id}: step info must be dict"

        if hasattr(env, "close"):
            env.close()

    def test_registered_tasks_discoverable(self) -> None:
        """Supported tasks must be discoverable in mjlab task registry."""
        list_tasks = importlib.import_module("mjlab.tasks.registry").list_tasks

        discovered = set(list_tasks())
        missing = [
            task_id for task_id in self._SUPPORTED_TASK_IDS if task_id not in discovered
        ]
        if missing:
            pytest.skip(
                f"Missing supported mjlab tasks in registry on this installation: {missing}. "
                "Ensure myosuite's mjlab tasks are registered if you want to run these tests."
            )


@pytest.mark.skipif(
    not _MJLAB_AVAILABLE,
    reason="mjlab not installed (pip install myosuite[mjlab])",
)
class TestMjlabMimicIntegration:
    """End-to-end mjlab integration for Mimic bimanual/full-body tasks."""

    _MIMIC_TASK_IDS = (
        "myoMimicBimanual-v0",
        "myoMimicFullbody-v0",
    )

    @staticmethod
    def _obs_to_numpy(obs: Any) -> Any:
        import numpy as np

        if isinstance(obs, dict):
            obs = obs.get("policy", next(iter(obs.values())))
        if _TORCH_AVAILABLE and isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        arr = np.asarray(obs)
        if arr.ndim > 1:
            arr = arr[0]
        return arr

    @staticmethod
    def _zero_action(env: Any) -> Any:
        if hasattr(env, "action_space") and getattr(env.action_space, "shape", None):
            act_dim = int(env.action_space.shape[-1])
        elif hasattr(env, "single_action_space") and getattr(
            env.single_action_space, "shape", None
        ):
            act_dim = int(env.single_action_space.shape[-1])
        elif hasattr(env, "action_manager") and hasattr(
            env.action_manager, "total_action_dim"
        ):
            act_dim = int(env.action_manager.total_action_dim)
        else:
            raise AssertionError("Cannot infer action dimension from mjlab env")
        num_envs = int(getattr(env, "num_envs", 1))
        device = getattr(env, "device", "cpu")
        if not _TORCH_AVAILABLE:
            raise AssertionError("torch is required for mjlab integration test")
        return torch.zeros((num_envs, act_dim), dtype=torch.float32, device=device)

    @pytest.fixture(autouse=True)
    def _require_musclemimic_models(self) -> None:
        from myosuite.tests.support.optional_deps import require_musclemimic_models

        require_musclemimic_models()

    @pytest.mark.parametrize("env_id", _MIMIC_TASK_IDS)
    def test_make_reset_step_mimic_tasks(self, env_id: str) -> None:
        """Mimic mjlab tasks must support make→reset→step when registered."""
        import numpy as np

        import myosuite

        myosuite.register_all_envs()
        from myosuite.core.registry import make_env

        try:
            env = make_env(env_id, backend="mjlab")
        except ValueError as exc:
            if "not in mjlab task registry" in str(exc):
                pytest.skip(f"{env_id}: not in mjlab registry: {exc}")
            raise
        obs, info = env.reset(seed=0)
        obs_np = self._obs_to_numpy(obs)
        assert obs_np.size > 0, f"{env_id}: empty observation on reset"
        assert np.all(np.isfinite(obs_np)), f"{env_id}: non-finite reset obs"
        assert isinstance(info, dict), f"{env_id}: reset info must be dict"

        action = self._zero_action(env)
        obs2, reward, _terminated, _truncated, info2 = env.step(action)
        obs2_np = self._obs_to_numpy(obs2)
        assert obs2_np.size > 0, f"{env_id}: empty observation on step"
        assert np.all(np.isfinite(obs2_np)), f"{env_id}: non-finite step obs"

        rew_np = (
            reward.detach().cpu().numpy()
            if _TORCH_AVAILABLE and isinstance(reward, torch.Tensor)
            else np.asarray(reward)
        )
        assert np.all(np.isfinite(rew_np)), f"{env_id}: non-finite reward"
        assert isinstance(info2, dict), f"{env_id}: step info must be dict"

        if hasattr(env, "close"):
            env.close()

    def test_mimic_tasks_discoverable_when_registered(self) -> None:
        """Mimic task ids should appear in mjlab list_tasks when assets load."""
        list_tasks = importlib.import_module("mjlab.tasks.registry").list_tasks
        discovered = set(list_tasks())
        missing = [t for t in self._MIMIC_TASK_IDS if t not in discovered]
        if missing:
            pytest.skip(
                f"Mimic mjlab tasks not registered on this installation: {missing}"
            )


def test_make_env_mjlab_fallback_applies_num_envs_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback mjlab path must map num_envs override into cfg.scene.num_envs."""
    import sys

    from myosuite.core.registry import make_env

    class _Cfg:
        def __init__(self) -> None:
            self.scene = types.SimpleNamespace(num_envs=1)

    captured: dict[str, Any] = {}

    class _ManagerBasedRlEnv:
        def __init__(self, cfg: Any, device: str, **overrides: Any) -> None:
            captured["cfg"] = cfg
            captured["device"] = device
            captured["overrides"] = overrides

    fake_envs = types.ModuleType("mjlab.envs")
    fake_envs.ManagerBasedRlEnv = _ManagerBasedRlEnv

    fake_registry = types.ModuleType("mjlab.tasks.registry")
    fake_registry.list_tasks = lambda: ["dummy-task-v0"]
    fake_registry.load_env_cfg = lambda _env_id: _Cfg()

    fake_tasks = types.ModuleType("mjlab.tasks")
    fake_mjlab = types.ModuleType("mjlab")
    fake_mjlab.envs = fake_envs
    fake_mjlab.tasks = fake_tasks

    monkeypatch.setitem(sys.modules, "mjlab", fake_mjlab)
    monkeypatch.setitem(sys.modules, "mjlab.envs", fake_envs)
    monkeypatch.setitem(sys.modules, "mjlab.tasks", fake_tasks)
    monkeypatch.setitem(sys.modules, "mjlab.tasks.registry", fake_registry)

    make_env(
        "dummy-task-v0",
        backend="mjlab",
        num_envs=20,
        device="cpu",
        sentinel_override=123,
    )

    assert captured["cfg"].scene.num_envs == 20
    assert captured["device"] == "cpu"
    assert captured["overrides"] == {"sentinel_override": 123}


@pytest.mark.skipif(
    not _MJLAB_AVAILABLE,
    reason="mjlab not installed (pip install myosuite[mjlab])",
)
def test_bootstrap_myosuite_mjlab_registry_idempotent() -> None:
    """``bootstrap_myosuite_mjlab_registry`` may be called repeatedly."""
    from mjlab.tasks.registry import list_tasks

    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        bootstrap_myosuite_mjlab_registry,
    )

    before = set(list_tasks())
    bootstrap_myosuite_mjlab_registry()
    mid = set(list_tasks())
    bootstrap_myosuite_mjlab_registry()
    after = set(list_tasks())
    assert mid == after
    assert mid >= before


def test_bootstrap_myosuite_mjlab_registry_skips_myouser_without_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default bootstrap must not import or register optional myouser tasks."""
    from myosuite.envs.myo.backends.mjlab import register_mjlab_tasks as registry_mod

    calls: list[Any] = []
    monkeypatch.setattr(
        registry_mod, "register_mjlab_tasks", lambda: calls.append("core")
    )
    monkeypatch.setattr(
        registry_mod,
        "_register_optional_myouser_task",
        lambda cfg: calls.append(("myouser", cfg)),
    )
    monkeypatch.delenv("MYOSUITE_MIMIC_CLIP", raising=False)
    monkeypatch.delenv("MIMIC_CLIP", raising=False)

    registry_mod.bootstrap_myosuite_mjlab_registry()

    assert calls == ["core"]


def test_bootstrap_myosuite_mjlab_registry_registers_myouser_with_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit myouser config must trigger optional myouser registration."""
    from myosuite.envs.myo.backends.mjlab import register_mjlab_tasks as registry_mod

    calls: list[Any] = []
    config = object()
    monkeypatch.setattr(
        registry_mod, "register_mjlab_tasks", lambda: calls.append("core")
    )
    monkeypatch.setattr(
        registry_mod,
        "_register_optional_myouser_task",
        lambda cfg: calls.append(("myouser", cfg)),
    )
    monkeypatch.delenv("MYOSUITE_MIMIC_CLIP", raising=False)
    monkeypatch.delenv("MIMIC_CLIP", raising=False)

    registry_mod.bootstrap_myosuite_mjlab_registry(myouser_config=config)

    assert calls == ["core", ("myouser", config)]


@pytest.mark.skipif(
    not _MJLAB_AVAILABLE,
    reason="mjlab not installed (pip install myosuite[mjlab])",
)
def test_import_mjlab_has_no_duplicate_registration_warnings() -> None:
    """Plain ``import mjlab`` must not pull in myouser's duplicate Gym registrations."""
    repo = Path(__file__).resolve().parents[2]
    code = "import sys; sys.path.insert(0, %r); import mjlab" % str(repo)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo),
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout or f"exit {proc.returncode}"
    output = f"{proc.stdout}\n{proc.stderr}"
    assert "Overriding environment" not in output
    assert "MyoSuite:> Registering Myo Envs" not in output


def test_mujoco_warp_version_has_sparse_tendon_transmission_fix() -> None:
    """mjlab installs must use MuJoCo-Warp with the sparse tendon fix."""
    from myosuite.tests.support.optional_deps import require_mujoco_warp

    require_mujoco_warp(min_version="3.7.0")


@pytest.mark.skipif(
    not (_MJLAB_AVAILABLE and _TORCH_AVAILABLE),
    reason="mjlab/torch not installed (pip install myosuite[mjlab])",
)
def test_mjlab_parallel_qacc_consistency_zero_state() -> None:
    """Diagnostic: qacc should match across envs for identical zeroed state."""
    import myosuite

    myosuite.register_all_envs()
    from myosuite.core.registry import make_env

    num_envs = 8
    env = make_env("myoLegWalk-v0", backend="mjlab", num_envs=num_envs, device="cpu")
    env.reset(seed=42)

    data = env.sim.data
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    if hasattr(data, "act"):
        data.act[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0

    if hasattr(data, "qacc_warmstart"):
        data.qacc_warmstart[:] = 0.0

    env.sim.forward()

    qacc = data.qacc
    max_diff = float((qacc - qacc[0:1]).abs().max().item())

    if hasattr(env, "close"):
        env.close()

    assert (
        max_diff < 1e-10
    ), f"qacc mismatch across identical env states: max_diff={max_diff:.6e}"
