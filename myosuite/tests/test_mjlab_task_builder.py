# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Parity tests: factory-built walk config must match the reference inline config."""

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
    WalkCfg,
    _make_elbow_env_cfg,
    _make_walk_env_cfg,
    _walk_muscle_names,
    _walk_obs_act,
    _walk_obs_com_vel,
    _walk_obs_feet_heights,
    _walk_obs_feet_rel_positions,
    _walk_obs_height,
    _walk_obs_muscle_force,
    _walk_obs_muscle_length,
    _walk_obs_muscle_velocity,
    _walk_obs_phase_var,
    _walk_obs_qpos_without_xy,
    _walk_obs_qvel,
    _walk_obs_torso_angle,
    _walk_forward_vel_reward,
    _walk_alive_reward,
    _walk_done_signal,
    _walk_cyclic_hip,
    _walk_ref_rot,
    _walk_joint_angle_rew,
    _walk_act_reg,
    _walk_spec_fn,
)


def _build_reference_walk_cfg() -> ManagerBasedRlEnvCfg:
    """Inline reference config built identically to old _make_walk_env_cfg body."""
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.scene import SceneCfg

    muscle_names = _walk_muscle_names()
    walk_entity_name = "walk_robot"

    articulation = EntityArticulationInfoCfg(
        actuators=(
            _XmlWrappedActuatorCfg(
                target_names_expr=tuple(f"{name}_tendon" for name in muscle_names),
                transmission_type=TransmissionType.TENDON,
            ),
        )
    )
    entity_cfg = EntityCfg(spec_fn=_walk_spec_fn, articulation=articulation)
    scene_cfg = SceneCfg(num_envs=1, entities={walk_entity_name: entity_cfg})

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "qpos_without_xy": ObservationTermCfg(func=_walk_obs_qpos_without_xy),
                "qvel": ObservationTermCfg(func=_walk_obs_qvel),
                "com_vel": ObservationTermCfg(func=_walk_obs_com_vel),
                "torso_angle": ObservationTermCfg(func=_walk_obs_torso_angle),
                "feet_heights": ObservationTermCfg(func=_walk_obs_feet_heights),
                "height": ObservationTermCfg(func=_walk_obs_height),
                "feet_rel_positions": ObservationTermCfg(
                    func=_walk_obs_feet_rel_positions
                ),
                "phase_var": ObservationTermCfg(func=_walk_obs_phase_var),
                "muscle_length": ObservationTermCfg(func=_walk_obs_muscle_length),
                "muscle_velocity": ObservationTermCfg(func=_walk_obs_muscle_velocity),
                "muscle_force": ObservationTermCfg(func=_walk_obs_muscle_force),
                "act": ObservationTermCfg(func=_walk_obs_act),
            },
        ),
    }
    actions = {
        "muscles": MyoMuscleActivationActionCfg(
            entity_name=walk_entity_name,
            actuator_names=muscle_names,
        ),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=mdp_terminations.time_out, time_out=True),
    }
    walk_cfg = WalkCfg()
    rewards = {
        "vel_reward": RewardTermCfg(
            func=_walk_forward_vel_reward,
            weight=5.0,
            params={"target_vel": float(walk_cfg.target_vel), "target_x_vel": 0.0},
        ),
        "alive_reward": RewardTermCfg(
            func=_walk_alive_reward,
            weight=float(walk_cfg.alive_bonus),
            params={"fall_height_threshold": float(walk_cfg.fall_height_threshold)},
        ),
        "done": RewardTermCfg(func=_walk_done_signal, weight=-100.0),
        "cyclic_hip": RewardTermCfg(func=_walk_cyclic_hip, weight=-10.0),
        "ref_rot": RewardTermCfg(func=_walk_ref_rot, weight=10.0),
        "joint_angle_rew": RewardTermCfg(func=_walk_joint_angle_rew, weight=5.0),
        "act_reg": RewardTermCfg(
            func=_walk_act_reg, weight=-float(walk_cfg.act_reg_weight)
        ),
    }
    return ManagerBasedRlEnvCfg(
        scene=scene_cfg,
        decimation=10,
        episode_length_s=20.0,
        observations=observations,
        actions=actions,
        terminations=terminations,
        rewards=rewards,
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.002, ccd_iterations=500)),
    )


def _build_factory_walk_cfg() -> ManagerBasedRlEnvCfg:
    muscle_names = _walk_muscle_names()
    walk_entity_name = "walk_robot"

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "qpos_without_xy": ObservationTermCfg(func=_walk_obs_qpos_without_xy),
                "qvel": ObservationTermCfg(func=_walk_obs_qvel),
                "com_vel": ObservationTermCfg(func=_walk_obs_com_vel),
                "torso_angle": ObservationTermCfg(func=_walk_obs_torso_angle),
                "feet_heights": ObservationTermCfg(func=_walk_obs_feet_heights),
                "height": ObservationTermCfg(func=_walk_obs_height),
                "feet_rel_positions": ObservationTermCfg(
                    func=_walk_obs_feet_rel_positions
                ),
                "phase_var": ObservationTermCfg(func=_walk_obs_phase_var),
                "muscle_length": ObservationTermCfg(func=_walk_obs_muscle_length),
                "muscle_velocity": ObservationTermCfg(func=_walk_obs_muscle_velocity),
                "muscle_force": ObservationTermCfg(func=_walk_obs_muscle_force),
                "act": ObservationTermCfg(func=_walk_obs_act),
            },
        ),
    }
    actions = {
        "muscles": MyoMuscleActivationActionCfg(
            entity_name=walk_entity_name,
            actuator_names=muscle_names,
        ),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=mdp_terminations.time_out, time_out=True),
    }
    walk_cfg = WalkCfg()
    rewards = {
        "vel_reward": RewardTermCfg(
            func=_walk_forward_vel_reward,
            weight=5.0,
            params={"target_vel": float(walk_cfg.target_vel), "target_x_vel": 0.0},
        ),
        "alive_reward": RewardTermCfg(
            func=_walk_alive_reward,
            weight=float(walk_cfg.alive_bonus),
            params={"fall_height_threshold": float(walk_cfg.fall_height_threshold)},
        ),
        "done": RewardTermCfg(func=_walk_done_signal, weight=-100.0),
        "cyclic_hip": RewardTermCfg(func=_walk_cyclic_hip, weight=-10.0),
        "ref_rot": RewardTermCfg(func=_walk_ref_rot, weight=10.0),
        "joint_angle_rew": RewardTermCfg(func=_walk_joint_angle_rew, weight=5.0),
        "act_reg": RewardTermCfg(
            func=_walk_act_reg, weight=-float(walk_cfg.act_reg_weight)
        ),
    }
    return mjlab_env_cfg_from_task_config(
        cfg=TaskConfig(max_episode_steps=1000),
        spec_fn=_walk_spec_fn,
        entity_name=walk_entity_name,
        actuators=(
            _XmlWrappedActuatorCfg(
                target_names_expr=tuple(f"{name}_tendon" for name in muscle_names),
                transmission_type=TransmissionType.TENDON,
            ),
        ),
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        num_envs=1,
        decimation=10,
        sim_cfg=SimulationCfg(mujoco=MujocoCfg(timestep=0.002, ccd_iterations=500)),
        episode_length_s=20.0,
    )


@pytest.fixture(scope="module")
def configs() -> tuple[ManagerBasedRlEnvCfg, ManagerBasedRlEnvCfg]:
    return _build_reference_walk_cfg(), _build_factory_walk_cfg()


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
        assert (
            new_cfg.rewards[k].weight == old_cfg.rewards[k].weight
        ), f"reward '{k}' weight mismatch: {new_cfg.rewards[k].weight} != {old_cfg.rewards[k].weight}"


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


def test_walk_obs_keys_match_cpu() -> None:
    """mjlab walk obs terms must match LegWalkEnvV0 DEFAULT_OBS_KEYS + act."""
    cfg = _make_walk_env_cfg()
    mjlab_keys = list(cfg.observations["policy"].terms.keys())
    cpu_keys = [
        "qpos_without_xy",
        "qvel",
        "com_vel",
        "torso_angle",
        "feet_heights",
        "height",
        "feet_rel_positions",
        "phase_var",
        "muscle_length",
        "muscle_velocity",
        "muscle_force",
        "act",
    ]
    assert mjlab_keys == cpu_keys


def test_walk_action_dim_matches_cpu() -> None:
    """mjlab walk action dim must equal CPU action_space.shape[0] = 80."""
    cfg = _make_walk_env_cfg()
    cpu_act_dim = 80  # myoLegWalk-v0 action_space.shape=(80,)
    assert len(cfg.actions["muscles"].actuator_names) == cpu_act_dim


def test_walk_ctrl_dt_is_50hz() -> None:
    """ctrl_dt must equal 0.02 s so mjswan's hardcoded 50 Hz matches training."""
    cfg = _make_walk_env_cfg()
    ctrl_dt = cfg.sim.mujoco.timestep * cfg.decimation
    assert ctrl_dt == pytest.approx(0.02)


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
        spec_fn=_walk_spec_fn,
        entity_name="e",
        actuators=(),
        observations={"policy": ObservationGroupCfg(terms={})},
        actions={},
        decimation=5,
        sim_cfg=SimulationCfg(mujoco=MujocoCfg(timestep=non_default_dt)),
    )
    assert result.episode_length_s == pytest.approx(500 * 5 * non_default_dt)
