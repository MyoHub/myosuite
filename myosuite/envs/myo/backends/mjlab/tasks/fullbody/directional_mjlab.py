# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""mjlab twin of CPU ``myoFullBodyDirectional-v0`` (heading only, no clip)."""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

import torch
from mjlab.actuator import XmlActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import terminations as mdp_terminations
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg

from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import (
    MyoMuscleActivationActionCfg,
)
from myosuite.envs.myo.backends.mjlab.tasks.mdp.commands import HeadingCommandCfg
from myosuite.envs.myo.backends.mjlab.tasks.mdp.terminations import (
    SYNC_TERM,
    sync_forward,
)

ENV_ID = "myoFullBodyDirectional-v0"
ENTITY = "mimic_fullbody_robot"
COMMAND = "heading"
TARGET_SPEED = 0.7
FALL_Z = 0.6
UPRIGHT_Z = 0.85
MAX_EPISODE_STEPS = 500


def _qpos(env: Any) -> torch.Tensor:
    return env.scene[ENTITY].data.data.qpos


def _qvel(env: Any) -> torch.Tensor:
    return env.scene[ENTITY].data.data.qvel


def _yaw(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rotate_neg_yaw(xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    x, y = xy.unbind(dim=1)
    return torch.stack([c * x - s * y, s * x + c * y], dim=1)


def obs_qpos_local(env: Any) -> torch.Tensor:
    return env.scene[ENTITY].data.joint_pos


def obs_qvel_local(env: Any) -> torch.Tensor:
    return env.scene[ENTITY].data.joint_vel


def obs_act(env: Any) -> torch.Tensor:
    return env.scene[ENTITY].data.data.act


def obs_root_vel_body(env: Any) -> torch.Tensor:
    return _rotate_neg_yaw(_qvel(env)[:, :2], _yaw(_qpos(env)[:, 3:7]))


def obs_heading_cmd(env: Any) -> torch.Tensor:
    return env.command_manager.get_command(COMMAND)


def obs_orientation(env: Any) -> torch.Tensor:
    qpos, qvel = _qpos(env), _qvel(env)
    w, x, y, z = qpos[:, 3:7].unbind(dim=1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    ang_b = _rotate_neg_yaw(qvel[:, 3:5], _yaw(qpos[:, 3:7]))
    return torch.cat(
        [roll[:, None], pitch[:, None], ang_b, qvel[:, 5:6], qvel[:, 2:3]], dim=1
    )


def heading_reward(env: Any) -> torch.Tensor:
    vel = _qvel(env)[:, :2]
    cmd = env.command_manager.get_command(COMMAND) * TARGET_SPEED
    diff = torch.cat(
        [vel - cmd, torch.zeros(vel.shape[0], 1, device=vel.device)], dim=1
    )
    heading = torch.exp(-diff.square().mean(dim=-1))
    z = _qpos(env)[:, 2]
    return torch.where(z >= UPRIGHT_Z, heading, torch.zeros_like(heading))


def survival_reward(env: Any) -> torch.Tensor:
    return (_qpos(env)[:, 2] >= UPRIGHT_Z).to(dtype=torch.float32)


def act_reg_reward(env: Any) -> torch.Tensor:
    act = env.scene[ENTITY].data.data.act
    return -torch.mean(torch.square(act), dim=1)


def fallen(env: Any) -> torch.Tensor:
    return _qpos(env)[:, 2] < FALL_Z


def make_fullbody_directional_env_cfg(*, num_envs: int = 1) -> ManagerBasedRlEnvCfg:
    from ml_collections import config_dict

    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
        _init_state_from_model,
        _muscle_actuator_names,
        _muscle_tendon_names,
        _policy_actor_critic_groups,
        mimic_viewer_cfg,
    )
    from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import (
        musclemimic_mujoco_cfg,
    )
    from myosuite.integrations.musclemimic.fullbody_model import (
        build_mimic_fullbody_spec,
        default_mimic_fullbody_config,
    )
    from mjlab.actuator.actuator import TransmissionType
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    cfg = default_mimic_fullbody_config()
    spec, _ = build_mimic_fullbody_spec(config_dict.create(**dict(cfg)))
    mj_model = spec.compile()
    muscles = _muscle_actuator_names(mj_model)
    tendons = _muscle_tendon_names(mj_model)

    def _spec_fn() -> Any:
        built, _ = build_mimic_fullbody_spec(config_dict.create(**dict(cfg)))
        return built

    ctrl_dt = float(cfg.ctrl_dt)
    sim_dt = float(cfg.sim_dt)
    decimation = max(1, int(round(ctrl_dt / sim_dt)))
    entity_cfg = EntityCfg(
        spec_fn=_spec_fn,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                XmlActuatorCfg(
                    target_names_expr=tendons,
                    transmission_type=TransmissionType.TENDON,
                ),
            )
        ),
        init_state=_init_state_from_model(mj_model),
    )
    obs_terms = {
        "qpos_local": ObservationTermCfg(func=obs_qpos_local),
        "qvel_local": ObservationTermCfg(func=obs_qvel_local),
        "act": ObservationTermCfg(func=obs_act),
        "root_vel_xy": ObservationTermCfg(func=obs_root_vel_body),
        "heading_cmd": ObservationTermCfg(func=obs_heading_cmd),
        "orientation": ObservationTermCfg(func=obs_orientation),
    }
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(num_envs=num_envs, entities={ENTITY: entity_cfg}),
        decimation=decimation,
        episode_length_s=float(MAX_EPISODE_STEPS) * ctrl_dt,
        observations=_policy_actor_critic_groups(obs_terms),
        actions={
            "muscles": MyoMuscleActivationActionCfg(
                entity_name=ENTITY,
                actuator_names=muscles,
                tendon_names=tendons,
                action_mode="direct",
            )
        },
        commands={
            COMMAND: HeadingCommandCfg(
                heading_dir=(0.0, 1.0),
                randomize=True,
            )
        },
        rewards={
            "heading": RewardTermCfg(func=heading_reward, weight=1.0),
            "survival": RewardTermCfg(func=survival_reward, weight=0.2),
            "act_reg": RewardTermCfg(func=act_reg_reward, weight=0.1),
        },
        terminations={
            SYNC_TERM: TerminationTermCfg(func=sync_forward),
            "time_out": TerminationTermCfg(
                func=mdp_terminations.time_out, time_out=True
            ),
            "fallen": TerminationTermCfg(func=fallen),
        },
        sim=SimulationCfg(
            mujoco=musclemimic_mujoco_cfg("fullbody", timestep=sim_dt),
            njmax=512,
            nconmax=256,
        ),
        viewer=mimic_viewer_cfg(ENTITY),
        scale_rewards_by_dt=False,
    )


def register_fullbody_directional_mjlab(
    register_mjlab_task: Callable[..., None],
    rl_cfg_fn: Callable[[], Any],
) -> None:
    """Register ``myoFullBodyDirectional-v0`` on the mjlab task list."""
    env_cfg = make_fullbody_directional_env_cfg(num_envs=1)
    try:
        register_mjlab_task(
            task_id=ENV_ID,
            env_cfg=env_cfg,
            play_env_cfg=env_cfg,
            rl_cfg=rl_cfg_fn(),
            runner_cls=None,
        )
    except ValueError:
        pass
