# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for ModularTaskEnv and register_task (Phase 5 — Modular Task System)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="mujoco required for ModularTaskEnv tests")

import mujoco  # noqa: E402

pytestmark = pytest.mark.tier1

from myosuite.core.config import (  # noqa: E402
    ActuatorGroupSpec,
    BackendConfig,
    GoalSpec,
    ObsSpec,
    RewardSpec,
    TaskConfig,
)
from myosuite.core.config import VariantSpec  # noqa: E402
from myosuite.core.registry import register_env, register_task  # noqa: E402
from myosuite.envs.modular_env import (  # noqa: E402
    ModularTaskEnv,
    _load_obs_fn,
    _load_reward_fn,
    _sample_goal,
)


_THREE_HINGE_XML = """
<mujoco>
  <worldbody>
    <body name="b0">
      <joint name="j0" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.05"/>
      <body name="b1" pos="0.1 0 0">
        <joint name="j1" type="hinge" axis="0 0 1"/>
        <geom type="sphere" size="0.05"/>
        <body name="b2" pos="0.1 0 0">
          <joint name="j2" type="hinge" axis="0 0 1"/>
          <geom type="sphere" size="0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_ONE_HINGE_XML = """
<mujoco>
  <worldbody>
    <body>
      <joint name="j0" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


def _three_hinge_model_qpos() -> tuple[mujoco.MjModel, np.ndarray]:
    model = mujoco.MjModel.from_xml_string(_THREE_HINGE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    return model, data.qpos.copy()


def _one_hinge_model_qpos() -> tuple[mujoco.MjModel, np.ndarray]:
    model = mujoco.MjModel.from_xml_string(_ONE_HINGE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    return model, data.qpos.copy()


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@dataclass
class _ElbowTask(TaskConfig):
    """Minimal elbow pose task used across tests."""

    model: str = "elbow_standard"
    obs: ObsSpec = field(
        default_factory=lambda: ObsSpec(keys=["joint_pos", "joint_vel"])
    )
    goal: GoalSpec = field(
        default_factory=lambda: GoalSpec(
            target_type="joint_angles",
            randomize=True,
            range={"r_elbow_flex": (0.0, 2.27)},
        )
    )
    reward: RewardSpec = field(default_factory=lambda: RewardSpec(terms=["pose"]))
    actuators: list[ActuatorGroupSpec] = field(
        default_factory=lambda: [ActuatorGroupSpec()]
    )
    max_episode_steps: int = 50
    backend: BackendConfig = field(
        default_factory=lambda: BackendConfig(n_substeps=2, ctrl_dt=0.002)
    )


@dataclass
class _VariantTask(TaskConfig):
    """Task with VariantSpec used in variant expansion tests."""

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
    variants: ClassVar[list[VariantSpec]] = [
        VariantSpec(suffix="TestVar", config_delta={}),
    ]


@dataclass
class _NoDoubleTask(TaskConfig):
    """Task with VariantSpec used to test no-double-expansion."""

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
    variants: ClassVar[list[VariantSpec]] = [
        VariantSpec(suffix="X", config_delta={}),
    ]


@pytest.fixture(scope="module")
def elbow_env() -> ModularTaskEnv:
    """Shared ModularTaskEnv instance for the elbow task."""
    return ModularTaskEnv(_ElbowTask())


# ---------------------------------------------------------------------------
# Unit tests — loader helpers
# ---------------------------------------------------------------------------


def test_load_obs_fn_joint_pos() -> None:
    fn = _load_obs_fn("joint_pos")
    assert callable(fn)


def test_load_obs_fn_unknown_raises() -> None:
    with pytest.raises(AttributeError, match="not found"):
        _load_obs_fn("nonexistent_term_xyz")


def test_load_reward_fn_pose() -> None:
    fn = _load_reward_fn("pose")
    assert callable(fn)


def test_load_reward_fn_act_reg_no_suffix() -> None:
    # "act_reg" does not follow the _reward suffix; fallback must find it.
    fn = _load_reward_fn("act_reg")
    assert callable(fn)


def test_load_reward_fn_unknown_raises() -> None:
    with pytest.raises(AttributeError, match="not found"):
        _load_reward_fn("totally_unknown_term")


# ---------------------------------------------------------------------------
# Unit tests — goal sampling
# ---------------------------------------------------------------------------


def test_sample_goal_joint_angles_shape() -> None:
    spec = GoalSpec(
        target_type="joint_angles",
        randomize=True,
        range={"j0": (0.0, 1.0), "j1": (-0.5, 0.5)},
    )
    model, qpos = _three_hinge_model_qpos()
    rng = np.random.default_rng(42)
    goal = _sample_goal(spec, model, qpos, rng)
    assert "target_angles" in goal
    assert goal["target_angles"].shape == (model.nq,)
    # j2 not in range → should match baseline qpos (zeros after reset)
    assert goal["target_angles"][2] == 0.0


def test_sample_goal_joint_angles_bounds() -> None:
    spec = GoalSpec(
        target_type="joint_angles",
        randomize=True,
        range={"j0": (0.5, 1.5)},
    )
    model, qpos = _one_hinge_model_qpos()
    rng = np.random.default_rng(0)
    for _ in range(20):
        goal = _sample_goal(spec, model, qpos, rng)
        val = goal["target_angles"][0]
        assert 0.5 <= val <= 1.5, f"Out of bounds: {val}"


def test_sample_goal_fixed_no_randomize() -> None:
    spec = GoalSpec(
        target_type="joint_angles",
        randomize=False,
        range={"j0": (1.0, 2.0)},
    )
    model, qpos = _one_hinge_model_qpos()
    rng = np.random.default_rng(99)
    goal = _sample_goal(spec, model, qpos, rng)
    # Without randomize, lo is used as the fixed value
    assert goal["target_angles"][0] == pytest.approx(1.0)


def test_sample_goal_unsupported_type_raises() -> None:
    spec = GoalSpec(target_type="trajectory")
    model, qpos = _one_hinge_model_qpos()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Unsupported"):
        _sample_goal(spec, model, qpos, rng)


# ---------------------------------------------------------------------------
# Integration tests — ModularTaskEnv lifecycle
# ---------------------------------------------------------------------------


def test_env_spaces_shapes(elbow_env: ModularTaskEnv) -> None:
    obs_space = elbow_env.observation_space
    act_space = elbow_env.action_space
    assert len(obs_space.shape) == 1
    assert obs_space.shape[0] > 0
    assert len(act_space.shape) == 1
    assert act_space.shape[0] > 0


def test_reset_returns_valid_obs(elbow_env: ModularTaskEnv) -> None:
    obs, info = elbow_env.reset(seed=0)
    assert obs.shape == elbow_env.observation_space.shape
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_step_returns_5_tuple(elbow_env: ModularTaskEnv) -> None:
    elbow_env.reset(seed=1)
    action = elbow_env.action_space.sample()
    result = elbow_env.step(action)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == elbow_env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "rwd_dict" in info


def test_task_state_contains_target(elbow_env: ModularTaskEnv) -> None:
    elbow_env.reset(seed=7)
    assert "target_angles" in elbow_env._task_state


def test_reward_dict_has_dense_and_done(elbow_env: ModularTaskEnv) -> None:
    elbow_env.reset(seed=3)
    action = elbow_env.action_space.sample()
    _, _, _, _, info = elbow_env.step(action)
    rwd = info["rwd_dict"]
    assert "dense" in rwd
    assert "done" in rwd
    assert isinstance(rwd["dense"], float)


# ---------------------------------------------------------------------------
# Integration tests — register_task
# ---------------------------------------------------------------------------


def test_register_task_returns_env_id() -> None:
    env_id = register_task(_ElbowTask(), env_id="TestElbow-v0")
    assert env_id == "TestElbow-v0"


def test_register_task_auto_name() -> None:
    env_id = register_task(_ElbowTask())
    assert env_id == "_ElbowTask-v0"


def test_register_task_idempotent() -> None:
    """Registering the same env_id twice must not raise."""
    register_task(_ElbowTask(), env_id="TestElbowIdempotent-v0")
    register_task(_ElbowTask(), env_id="TestElbowIdempotent-v0")


def test_register_task_gymnasium_make() -> None:
    """gym.make() must work after register_task()."""
    import gymnasium as gym

    env_id = register_task(_ElbowTask(), env_id="TestElbowGymMake-v0")
    env = gym.make(env_id)
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] > 0
    env.close()


def test_register_task_applies_instability_wrapper_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gymnasium as gym

    env_id = register_task(_ElbowTask(), env_id="TestElbowInstabilityWrapped-v0")
    env = gym.make(env_id)
    base_env = env.unwrapped
    env.reset(seed=0)
    monkeypatch.setattr(
        base_env,
        "get_reward_dict",
        lambda obs_dict: {"dense": 0.0, "done": False},
    )
    monkeypatch.setattr(base_env, "_check_mj_instability_termination", lambda: True)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, _, terminated, truncated, _ = env.step(action)
    assert terminated is True
    assert truncated is False
    env.close()


def test_register_env_can_disable_instability_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gymnasium as gym

    env_id = "TestElbowInstabilityWrapperOff-v0"
    register_env(
        env_id=env_id,
        entry_point="myosuite.envs.modular_env:ModularTaskEnv",
        max_episode_steps=_ElbowTask().max_episode_steps,
        kwargs={"task_config": _ElbowTask()},
        wrap_mj_instability_termination=False,
    )
    env = gym.make(env_id)
    base_env = env.unwrapped
    env.reset(seed=0)
    monkeypatch.setattr(
        base_env,
        "get_reward_dict",
        lambda obs_dict: {"dense": 0.0, "done": False},
    )
    monkeypatch.setattr(base_env, "_check_mj_instability_termination", lambda: True)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, _, terminated, truncated, _ = env.step(action)
    assert terminated is False
    assert truncated is False
    env.close()


# ---------------------------------------------------------------------------
# TaskConfig dataclass contracts
# ---------------------------------------------------------------------------


def test_task_config_to_env_config() -> None:
    task = _ElbowTask()
    env_cfg = task.to_env_config("myTest-v0")
    assert env_cfg.env_id == "myTest-v0"
    assert env_cfg.model == "elbow_standard"
    assert env_cfg.max_episode_steps == 50


def test_reward_spec_weight_for_default() -> None:
    spec = RewardSpec(terms=["pose"], weights={"pose": 2.5})
    assert spec.weight_for("pose") == pytest.approx(2.5)
    assert spec.weight_for("unlisted") == pytest.approx(1.0)


def test_obs_spec_default_keys() -> None:
    spec = ObsSpec()
    assert "joint_pos" in spec.keys
    assert "joint_vel" in spec.keys
    assert "muscle_act" in spec.keys


# ---------------------------------------------------------------------------
# Callable obs/reward terms
# ---------------------------------------------------------------------------


def test_callable_obs_term() -> None:
    """Callable in ObsSpec.keys should work like a named term."""
    from myosuite.terms.base_obs import joint_pos_obs

    task = _ElbowTask()
    task.obs = ObsSpec(keys=[joint_pos_obs, "joint_vel"])
    env = ModularTaskEnv(task)
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] > 0


def test_callable_reward_term() -> None:
    """Callable in RewardSpec.terms should work like a named term."""
    from myosuite.terms.base_reward import pose_reward

    task = _ElbowTask()
    task.reward = RewardSpec(terms=[pose_reward])
    env = ModularTaskEnv(task)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(env.action_space.sample())
    assert isinstance(reward, float)


# ---------------------------------------------------------------------------
# Scene list randomization
# ---------------------------------------------------------------------------


def test_scene_list_env_resets() -> None:
    """list[str] scene — reset must not crash and obs shape is stable."""
    task = _ElbowTask()
    task.scene = [
        "flat_floor",
        "flat_floor",
    ]  # two variants (same model, different scene labels)
    env = ModularTaskEnv(task)
    obs1, _ = env.reset(seed=0)
    obs2, _ = env.reset(seed=1)
    assert obs1.shape == obs2.shape


# ---------------------------------------------------------------------------
# Action noise
# ---------------------------------------------------------------------------


def test_action_noise_applied() -> None:
    """ActuatorGroupSpec.noise > 0 should perturb actions."""
    task = _ElbowTask()
    task.actuators = [ActuatorGroupSpec(noise=1.0)]
    env = ModularTaskEnv(task)
    env.reset(seed=42)
    # Deterministic zero action — if noise applied, ctrl will differ from zeros
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, _, _, _, info = env.step(action)
    # Just verify it doesn't crash; noise makes result non-deterministic
    assert "rwd_dict" in info


# ---------------------------------------------------------------------------
# MJX modular env smoke (skip if JAX unavailable)
# ---------------------------------------------------------------------------

try:
    import jax as _jax
    from mujoco import mjx as _mjx  # noqa: F401

    _MJX_OK = True
except (ImportError, AttributeError):
    _MJX_OK = False


@pytest.mark.tier2
@pytest.mark.skipif(not _MJX_OK, reason="JAX/MJX not available")
def test_mjx_modular_env_reset() -> None:
    from myosuite.envs.myo.backends.mjx.mjx_modular_env import MjxModularTaskEnv

    env = MjxModularTaskEnv(_ElbowTask())
    state = env.reset(_jax.random.PRNGKey(0))
    assert "state" in state.obs
    assert state.obs["state"].shape[0] > 0


@pytest.mark.tier2
@pytest.mark.skipif(not _MJX_OK, reason="JAX/MJX not available")
def test_mjx_modular_env_step() -> None:
    import jax.numpy as jnp

    from myosuite.envs.myo.backends.mjx.mjx_modular_env import MjxModularTaskEnv

    env = MjxModularTaskEnv(_ElbowTask())
    state = env.reset(_jax.random.PRNGKey(0))
    action = jnp.zeros(env.action_size)
    next_state = env.step(state, action)
    assert next_state.obs["state"].shape == state.obs["state"].shape


# ---------------------------------------------------------------------------
# mjlab modular cfg
# ---------------------------------------------------------------------------


def test_make_modular_mjlab_cfg_basic() -> None:
    from myosuite.envs.myo.backends.mjlab.configs.modular_cfg import (
        make_modular_mjlab_cfg,
    )

    cfg = make_modular_mjlab_cfg(_ElbowTask())
    assert cfg.model_recipe == "elbow_standard"
    assert cfg.max_episode_steps == 50
    assert cfg.goal_type == "joint_angles"
    assert len(cfg.action_terms) == 1
    assert cfg.action_terms[0].name == "muscles"


def test_make_modular_mjlab_cfg_sarcopenia() -> None:
    from myosuite.envs.myo.backends.mjlab.configs.modular_cfg import (
        make_modular_mjlab_cfg,
    )

    task = _ElbowTask()
    task.actuators = [ActuatorGroupSpec(name="elbow", condition="sarcopenia")]
    cfg = make_modular_mjlab_cfg(task)
    assert cfg.action_terms[0].condition == "sarcopenia"


def test_make_modular_mjlab_cfg_callable_obs() -> None:
    from myosuite.terms.base_obs import joint_pos_obs

    from myosuite.envs.myo.backends.mjlab.configs.modular_cfg import (
        make_modular_mjlab_cfg,
    )

    task = _ElbowTask()
    task.obs = ObsSpec(keys=[joint_pos_obs])
    cfg = make_modular_mjlab_cfg(task)
    assert cfg.obs_keys == ["joint_pos_obs"]


# ---------------------------------------------------------------------------
# VariantSpec expansion via register_task
# ---------------------------------------------------------------------------


def test_register_task_variant_expansion() -> None:
    """VariantSpec entries on the class must register variant env_ids."""
    import gymnasium as gym

    base_id = "TestVariantBase-v0"
    register_task(_VariantTask(), env_id=base_id)
    assert base_id in gym.envs.registry
    # For non-"myo" ids: suffix appended before version tag
    variant_id_actual = "TestVariantBaseTestVar-v0"
    assert variant_id_actual in gym.envs.registry


def test_register_task_no_double_variant_expansion() -> None:
    """Calling register_task on a variant must not cause recursion."""
    env_id = register_task(_NoDoubleTask(), env_id="TestNoDouble-v0")
    assert env_id == "TestNoDouble-v0"


# ---------------------------------------------------------------------------
# muscle_normalize_action term
# ---------------------------------------------------------------------------


def test_muscle_normalize_action_midpoint() -> None:
    """Action=0.5 should map to ~0.5 (symmetric sigmoid midpoint)."""
    import numpy as np
    from myosuite.envs.gymnasium_env import CpuEnvAccessor
    from myosuite.terms.base_action import muscle_normalize_action

    model = mujoco.MjModel.from_xml_string(_ONE_HINGE_XML)
    data = mujoco.MjData(model)
    accessor = CpuEnvAccessor(model, data, ctrl_dt=0.002)
    action = np.array([0.5], dtype=np.float64)
    result = muscle_normalize_action(accessor, action)
    assert float(result[0]) == pytest.approx(0.5, abs=1e-6)


def test_muscle_normalize_action_bounds() -> None:
    """muscle_normalize_action output stays in (0, 1)."""
    import numpy as np
    from myosuite.envs.gymnasium_env import CpuEnvAccessor
    from myosuite.terms.base_action import muscle_normalize_action

    model = mujoco.MjModel.from_xml_string(_ONE_HINGE_XML)
    data = mujoco.MjData(model)
    accessor = CpuEnvAccessor(model, data, ctrl_dt=0.002)
    for val in [-5.0, -1.0, 0.0, 0.5, 1.0, 5.0]:
        action = np.array([val], dtype=np.float64)
        result = muscle_normalize_action(accessor, action)
        assert 0.0 < float(result[0]) < 1.0, f"Out of bounds for action={val}"


def test_muscle_normalize_action_monotone() -> None:
    """Higher action → higher output (monotonically increasing)."""
    import numpy as np
    from myosuite.envs.gymnasium_env import CpuEnvAccessor
    from myosuite.terms.base_action import muscle_normalize_action

    model = mujoco.MjModel.from_xml_string(_ONE_HINGE_XML)
    data = mujoco.MjData(model)
    accessor = CpuEnvAccessor(model, data, ctrl_dt=0.002)
    actions = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]
    results = [
        float(muscle_normalize_action(accessor, np.array([a]))[0]) for a in actions
    ]
    assert results == sorted(results), "Output should be monotonically increasing"


# ---------------------------------------------------------------------------
# make_env basic smoke
# ---------------------------------------------------------------------------


def test_make_env_cpu_returns_env() -> None:
    from myosuite.core.registry import make_env

    register_task(_ElbowTask(), env_id="TestMakeEnv-v0")
    env = make_env("TestMakeEnv-v0", backend="cpu")
    assert env is not None
    env.close()


def test_make_env_unknown_backend_raises() -> None:
    from myosuite.core.registry import make_env

    with pytest.raises(ValueError, match="Unknown backend"):
        make_env("TestMakeEnv-v0", backend="foobar")
