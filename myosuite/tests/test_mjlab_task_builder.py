# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Parity tests: ``mjlab_env_cfg_from_task_config``'s factory-built config must match a
hand-built reference one, term for term.

The reference/factory pair reuses the elbow spec/entity/actuators (already exercised by
``_make_elbow_env_cfg``) with two dummy reward terms -- the point of the comparison is the
generic factory's structural output (decimation, episode length, sim settings, term
ordering, reward weights), not any particular task's semantics.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mjlab")

from mjlab.actuator.actuator import TransmissionType  # noqa: E402
from mjlab.envs import ManagerBasedRlEnvCfg  # noqa: E402
from mjlab.envs.mdp import terminations as mdp_terminations  # noqa: E402
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg  # noqa: E402
from mjlab.managers.reward_manager import RewardTermCfg  # noqa: E402
from mjlab.managers.termination_manager import TerminationTermCfg  # noqa: E402
from mjlab.sim import MujocoCfg, SimulationCfg  # noqa: E402

try:
    from mjlab.actuator import XmlActuatorCfg as _XmlWrappedActuatorCfg  # noqa: E402
except ImportError:
    from mjlab.actuator import XmlMuscleActuatorCfg as _XmlWrappedActuatorCfg  # noqa: E402

from myosuite.core.config import TaskConfig  # noqa: E402
from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import (  # noqa: E402
    mjlab_env_cfg_from_task_config,
)
from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (  # noqa: E402
    MyoMuscleActivationActionCfg,
    _ELBOW_ENTITY_NAME,
    _elbow_obs_act,
    _elbow_obs_qpos,
    _elbow_obs_qvel,
    _elbow_spec_fn,
    _elbow_tendon_names,
    _make_elbow_env_cfg,
)


def _dummy_reward_a(env) -> object:
    """Arbitrary reward term: only its presence/weight is checked, not its value."""
    return _elbow_obs_qpos(env)[:, 0]


def _dummy_reward_b(env) -> object:
    """A second arbitrary reward term, so key ordering has something to compare."""
    return _elbow_obs_qvel(env)[:, 0]


def _reference_cfg_kwargs() -> dict:
    """Shared pieces of the reference/factory pair below, built once."""
    tendon_names = _elbow_tendon_names()
    muscle_names = tuple(n.replace("_tendon", "") for n in tendon_names)
    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "qpos": ObservationTermCfg(func=_elbow_obs_qpos),
                "qvel": ObservationTermCfg(func=_elbow_obs_qvel),
                "act": ObservationTermCfg(func=_elbow_obs_act),
            },
        ),
    }
    actions = {
        "muscles": MyoMuscleActivationActionCfg(
            entity_name=_ELBOW_ENTITY_NAME,
            actuator_names=muscle_names,
        ),
    }
    rewards = {
        "a": RewardTermCfg(func=_dummy_reward_a, weight=5.0),
        "b": RewardTermCfg(func=_dummy_reward_b, weight=-2.0),
    }
    return {
        "tendon_names": tendon_names,
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
    }


def _build_reference_cfg() -> ManagerBasedRlEnvCfg:
    """Inline reference config, built by hand instead of through the factory."""
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.scene import SceneCfg

    kw = _reference_cfg_kwargs()
    articulation = EntityArticulationInfoCfg(
        actuators=(
            _XmlWrappedActuatorCfg(
                target_names_expr=kw["tendon_names"],
                transmission_type=TransmissionType.TENDON,
            ),
        )
    )
    entity_cfg = EntityCfg(spec_fn=_elbow_spec_fn, articulation=articulation)
    scene_cfg = SceneCfg(num_envs=1, entities={_ELBOW_ENTITY_NAME: entity_cfg})
    terminations = {
        "time_out": TerminationTermCfg(func=mdp_terminations.time_out, time_out=True),
    }
    return ManagerBasedRlEnvCfg(
        scene=scene_cfg,
        decimation=10,
        episode_length_s=20.0,
        observations=kw["observations"],
        actions=kw["actions"],
        terminations=terminations,
        rewards=kw["rewards"],
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.002, ccd_iterations=500)),
    )


def _build_factory_cfg() -> ManagerBasedRlEnvCfg:
    """The same config, built through ``mjlab_env_cfg_from_task_config``."""
    kw = _reference_cfg_kwargs()
    return mjlab_env_cfg_from_task_config(
        cfg=TaskConfig(max_episode_steps=1000),
        spec_fn=_elbow_spec_fn,
        entity_name=_ELBOW_ENTITY_NAME,
        actuators=(
            _XmlWrappedActuatorCfg(
                target_names_expr=kw["tendon_names"],
                transmission_type=TransmissionType.TENDON,
            ),
        ),
        observations=kw["observations"],
        actions=kw["actions"],
        rewards=kw["rewards"],
        num_envs=1,
        decimation=10,
        sim_cfg=SimulationCfg(mujoco=MujocoCfg(timestep=0.002, ccd_iterations=500)),
        episode_length_s=20.0,
    )


@pytest.fixture(scope="module")
def configs() -> tuple[ManagerBasedRlEnvCfg, ManagerBasedRlEnvCfg]:
    return _build_reference_cfg(), _build_factory_cfg()


def test_decimation(configs: tuple) -> None:
    old_cfg, new_cfg = configs
    assert new_cfg.decimation == old_cfg.decimation


def test_episode_length_s(configs: tuple) -> None:
    old_cfg, new_cfg = configs
    assert new_cfg.episode_length_s == old_cfg.episode_length_s


def test_sim_timestep(configs: tuple) -> None:
    old_cfg, new_cfg = configs
    assert new_cfg.sim.mujoco.timestep == old_cfg.sim.mujoco.timestep


def test_sim_ccd_iterations(configs: tuple) -> None:
    old_cfg, new_cfg = configs
    assert new_cfg.sim.mujoco.ccd_iterations == old_cfg.sim.mujoco.ccd_iterations


def test_observation_keys(configs: tuple) -> None:
    old_cfg, new_cfg = configs
    assert list(new_cfg.observations["policy"].terms.keys()) == list(
        old_cfg.observations["policy"].terms.keys()
    )


def test_action_keys(configs: tuple) -> None:
    old_cfg, new_cfg = configs
    assert list(new_cfg.actions.keys()) == list(old_cfg.actions.keys())


def test_reward_keys(configs: tuple) -> None:
    old_cfg, new_cfg = configs
    assert list(new_cfg.rewards.keys()) == list(old_cfg.rewards.keys())


def test_reward_weights(configs: tuple) -> None:
    old_cfg, new_cfg = configs
    for k in old_cfg.rewards:
        assert new_cfg.rewards[k].weight == old_cfg.rewards[k].weight, (
            f"reward '{k}' weight mismatch: {new_cfg.rewards[k].weight} != {old_cfg.rewards[k].weight}"
        )


def test_elbow_obs_keys_match_cpu() -> None:
    """mjlab elbow obs terms must match the CPU env obs_keys order and count."""
    cfg = _make_elbow_env_cfg()
    mjlab_keys = list(cfg.observations["policy"].terms.keys())
    # CPU myoElbowPose1D6MFixed-v0 obs_keys (PoseEnvV0 with na>0):
    cpu_keys = ["qpos", "qvel", "pose_err", "act"]
    assert mjlab_keys == cpu_keys


def test_elbow_obs_dim_matches_cpu() -> None:
    """mjlab elbow obs group must contain 4 terms summing to 9 dims (CPU shape=(9,))."""
    cfg = _make_elbow_env_cfg()
    assert len(cfg.observations["policy"].terms) == 4


def test_elbow_action_key_is_muscles() -> None:
    """Elbow mjlab action must use MyoMuscleActivationActionCfg (sigmoid), not TendonLengthActionCfg."""
    cfg = _make_elbow_env_cfg()
    assert "muscles" in cfg.actions
    assert isinstance(cfg.actions["muscles"], MyoMuscleActivationActionCfg)


def test_elbow_action_dim_matches_cpu() -> None:
    """Elbow mjlab action dim must equal CPU action_space.shape[0] = 6."""
    cfg = _make_elbow_env_cfg()
    cpu_act_dim = 6  # myoElbowPose1D6MFixed-v0 action_space.shape=(6,)
    assert len(cfg.actions["muscles"].actuator_names) == cpu_act_dim


def test_elbow_ctrl_dt_is_50hz() -> None:
    """ctrl_dt must equal 0.02 s so mjswan's hardcoded 50 Hz matches training."""
    cfg = _make_elbow_env_cfg()
    ctrl_dt = cfg.sim.mujoco.timestep * cfg.decimation
    assert ctrl_dt == pytest.approx(0.02)


def test_episode_length_uses_sim_timestep() -> None:
    """episode_length_s must derive from sim_cfg.mujoco.timestep, not a hardcoded constant."""
    cfg = TaskConfig(max_episode_steps=500)
    non_default_dt = 0.001
    result = mjlab_env_cfg_from_task_config(
        cfg=cfg,
        spec_fn=_elbow_spec_fn,
        entity_name="e",
        actuators=(),
        observations={"policy": ObservationGroupCfg(terms={})},
        actions={},
        decimation=5,
        sim_cfg=SimulationCfg(mujoco=MujocoCfg(timestep=non_default_dt)),
    )
    assert result.episode_length_s == pytest.approx(500 * 5 * non_default_dt)
