# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Register ``myoHalloween-v0`` for MJLab GPU training.

MJLab requires each Entity to contain at most one free joint. The Halloween
CPU model remains full body + free pumpkin, while MJLab represents the
character and projectile as separate entities and composes them into one scene.
"""

from __future__ import annotations

import logging
from typing import Any

import mujoco
import numpy as np
import torch
from mjlab.actuator import XmlActuatorCfg as _XmlWrappedActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import terminations as mdp_terminations
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.viewer.viewer_config import ViewerConfig

from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
    _init_state_from_model as _mimic_init_state_from_model,
)
from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import MyoMuscleActivationActionCfg
from myosuite.envs.myo.backends.mjlab.tasks.mdp import SYNC_TERM, sync_forward
from myosuite.envs.myo.tasks.halloween.halloween_model import (
    compile_halloween_model,
    park_pumpkin,
    throw_pumpkin,
)
from myosuite.integrations.musclemimic.fullbody_model import (
    build_mimic_fullbody_spec,
    default_mimic_fullbody_config,
)

logger = logging.getLogger(__name__)
_TASK_ID = "myoHalloween-v0"
_ROBOT = "halloween_fullbody"
_PUMPKIN = "halloween_pumpkin"
_SIM_DT = 0.002
_CTRL_DT = 0.01
_MAX_EPISODE_STEPS = 500
_PARK_POS = (6.5, 6.5, 0.15)


def _build_robot_spec() -> mujoco.MjSpec:
    cfg = default_mimic_fullbody_config()
    spec, _ = build_mimic_fullbody_spec(cfg)
    spec.option.timestep = _SIM_DT
    return spec


def _build_pumpkin_spec() -> mujoco.MjSpec:
    """Minimal projectile entity for training; browser/CPU model owns visuals."""
    spec = mujoco.MjSpec()
    spec.option.timestep = _SIM_DT
    body = spec.worldbody.add_body(name="pumpkin", pos=list(_PARK_POS))
    body.add_freejoint(name="pumpkin_free")
    geom = body.add_geom(
        name="pumpkin_collision",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.09, 0.0, 0.0],
        rgba=[1.0, 0.45, 0.05, 1.0],
        mass=0.35,
    )
    geom.contype = 1
    geom.conaffinity = 1
    geom.condim = 3
    geom.friction = [0.9, 0.1, 0.01]
    geom.solref = [0.02, 1.0]
    geom.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]
    return spec


def _muscle_and_tendon_names(model: mujoco.MjModel) -> tuple[tuple[str, ...], tuple[str, ...]]:
    muscles: list[str] = []
    tendons: list[str] = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act_{i}"
        muscles.append(name)
        tid = int(model.actuator_trnid[i, 0])
        tname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, tid) if tid >= 0 else None
        tendons.append(tname or name)
    return tuple(muscles), tuple(tendons)


def _pumpkin_init_state(model: mujoco.MjModel) -> EntityCfg.InitialStateCfg:
    q = np.asarray(model.qpos0, dtype=np.float64)
    return EntityCfg.InitialStateCfg(
        pos=(float(q[0]), float(q[1]), float(q[2])),
        rot=(float(q[3]), float(q[4]), float(q[5]), float(q[6])),
    )


_PUMPKIN_RADIUS = 0.09
_HIT_MARGIN = 0.11      # body link origins sit inside the body: a hit is the pumpkin this close to one
_FALLEN_Z = 0.6         # pelvis height below which the character has fallen
_THROW_EVERY_S = (1.5, 3.0)
_EPISODE_S = 6.0


def _obs_root(entity: str):
    """Pelvis height, orientation and body-frame velocities."""
    def _fn(env: Any) -> torch.Tensor:
        d = env.scene[entity].data
        return torch.cat([d.root_link_pos_w[:, 2:3], d.root_link_quat_w, d.root_link_lin_vel_b, d.root_link_ang_vel_b], dim=-1)
    return _fn


def _obs_joint_pos(entity: str):
    def _fn(env: Any) -> torch.Tensor:
        return env.scene[entity].data.joint_pos
    return _fn


def _obs_joint_vel(entity: str):
    def _fn(env: Any) -> torch.Tensor:
        return env.scene[entity].data.joint_vel
    return _fn


def _obs_act(entity: str):
    def _fn(env: Any) -> torch.Tensor:
        return env.scene[entity].data.data.act   # no stable API equivalent (accepted exception)
    return _fn


def _obs_pumpkin_rel(robot: str, pumpkin: str):
    """Pumpkin position relative to the pelvis and its velocity (world frame)."""
    def _fn(env: Any) -> torch.Tensor:
        root = env.scene[robot].data.root_link_pos_w
        p = env.scene[pumpkin].data
        return torch.cat([p.root_link_pos_w - root, p.root_link_lin_vel_w], dim=-1)
    return _fn


def _pumpkin_clearance(env: Any) -> torch.Tensor:
    """Distance from the pumpkin's centre to the nearest body link origin of the character."""
    links = env.scene[_ROBOT].data.body_link_pos_w                      # (envs, bodies, 3)
    pumpkin = env.scene[_PUMPKIN].data.root_link_pos_w[:, None, :]
    return (links - pumpkin).norm(dim=-1).min(dim=-1).values


def _reward_lateral_dodge(robot: str, pumpkin: str):
    def _fn(env: Any) -> torch.Tensor:
        root_y = env.scene[robot].data.root_link_pos_w[:, 1]
        pumpkin_y = env.scene[pumpkin].data.root_link_pos_w[:, 1]
        return (root_y - pumpkin_y).abs().clamp(max=1.5)
    return _fn


def _reward_upright(entity: str):
    def _fn(env: Any) -> torch.Tensor:
        z = env.scene[entity].data.root_link_pos_w[:, 2]
        return (z - 0.5).clamp(min=0.0, max=0.6)
    return _fn


def _reward_alive(_entity: str):
    def _fn(env: Any) -> torch.Tensor:
        return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)
    return _fn


def _term_fallen(entity: str):
    def _fn(env: Any) -> torch.Tensor:
        return env.scene[entity].data.root_link_pos_w[:, 2] < _FALLEN_Z
    return _fn


def _term_hit(env: Any) -> torch.Tensor:
    """The pumpkin reached the character's body."""
    return _pumpkin_clearance(env) < _PUMPKIN_RADIUS + _HIT_MARGIN


def _event_park_pumpkin(env: Any, env_ids: Any) -> None:
    """Reset: the pumpkin waits on the ground far away until the first throw."""
    pumpkin = env.scene[_PUMPKIN]
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else torch.as_tensor(env_ids, device=env.device)
    state = torch.zeros(len(ids), 13, device=env.device)
    state[:, :3] = torch.tensor(_PARK_POS, device=env.device)
    state[:, 3] = 1.0
    pumpkin.write_root_state_to_sim(state, env_ids=ids)


def _event_throw(throw_speed: float = 1.0):
    """Interval event: throw the pumpkin at the character from its left or right (world y)."""
    def _fn(env: Any, env_ids: Any) -> None:
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else torch.as_tensor(env_ids, device=env.device)
        if ids.numel() == 0:
            return
        root = env.scene[_ROBOT].data.root_link_pos_w[ids]
        sides = torch.where(torch.rand(len(ids), device=env.device) < 0.5, -1.0, 1.0)
        spd = float(throw_speed)
        state = torch.zeros(len(ids), 13, device=env.device)
        state[:, 0] = root[:, 0] + 0.35
        state[:, 1] = root[:, 1] + sides * 1.55
        state[:, 2] = root[:, 2].clamp(min=0.85)
        state[:, 3] = 1.0
        state[:, 7] = 0.15 * spd
        state[:, 8] = -sides * 3.6 * spd
        state[:, 9] = 0.35 * spd
        state[:, 11] = sides * 2.5 * spd
        env.scene[_PUMPKIN].write_root_state_to_sim(state, env_ids=ids)
    return _fn


def make_halloween_env_cfg(*, num_envs: int = 1, throw_speed: float = 1.0) -> ManagerBasedRlEnvCfg:
    """Build the MJLab config with robot and projectile as separate entities."""
    robot_model = _build_robot_spec().compile()
    pumpkin_model = _build_pumpkin_spec().compile()
    muscles, tendons = _muscle_and_tendon_names(robot_model)

    articulation = EntityArticulationInfoCfg(
        actuators=(
            _XmlWrappedActuatorCfg(
                target_names_expr=tendons,
                transmission_type=TransmissionType.TENDON,
            ),
        )
    )
    robot_cfg = EntityCfg(
        spec_fn=_build_robot_spec,
        articulation=articulation,
        init_state=_mimic_init_state_from_model(robot_model),
    )
    pumpkin_cfg = EntityCfg(
        spec_fn=_build_pumpkin_spec,
        init_state=_pumpkin_init_state(pumpkin_model),
    )
    scene = SceneCfg(num_envs=num_envs, entities={_ROBOT: robot_cfg, _PUMPKIN: pumpkin_cfg})

    obs_terms = {
        "root": ObservationTermCfg(func=_obs_root(_ROBOT)),
        "joint_pos": ObservationTermCfg(func=_obs_joint_pos(_ROBOT)),
        "joint_vel": ObservationTermCfg(func=_obs_joint_vel(_ROBOT)),
        "act": ObservationTermCfg(func=_obs_act(_ROBOT)),
        "pumpkin_rel": ObservationTermCfg(func=_obs_pumpkin_rel(_ROBOT, _PUMPKIN)),
    }
    observations = {
        "policy": ObservationGroupCfg(terms=obs_terms, concatenate_terms=True),
        "critic": ObservationGroupCfg(terms=obs_terms, concatenate_terms=True),
    }
    actions = {
        "muscles": MyoMuscleActivationActionCfg(
            entity_name=_ROBOT,
            actuator_names=muscles,
            tendon_names=tendons,
            action_mode="direct",
        )
    }
    rewards = {
        "alive": RewardTermCfg(func=_reward_alive(_ROBOT), weight=1.0),
        "upright": RewardTermCfg(func=_reward_upright(_ROBOT), weight=2.0),
        "dodge": RewardTermCfg(func=_reward_lateral_dodge(_ROBOT, _PUMPKIN), weight=3.0),
    }
    terminations = {
        SYNC_TERM: TerminationTermCfg(func=sync_forward),
        "time_out": TerminationTermCfg(func=mdp_terminations.time_out, time_out=True),
        "fallen": TerminationTermCfg(func=_term_fallen(_ROBOT), time_out=False),
        "hit": TerminationTermCfg(func=_term_hit, time_out=False),
    }
    # The pumpkin waits after a reset, then is thrown again and again: stand first, then dodge each throw.
    events = {
        "park_pumpkin": EventTermCfg(func=_event_park_pumpkin, mode="reset"),
        "throw_pumpkin": EventTermCfg(func=_event_throw(throw_speed), mode="interval", interval_range_s=_THROW_EVERY_S),
    }
    return ManagerBasedRlEnvCfg(
        scene=scene,
        decimation=max(1, int(round(_CTRL_DT / _SIM_DT))),
        episode_length_s=_EPISODE_S,
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        events=events,
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=_SIM_DT), njmax=512, nconmax=256),
        viewer=ViewerConfig(
            lookat=(0.0, 0.0, 0.9),
            distance=3.2,
            elevation=-10.0,
            azimuth=50.0,
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name=_ROBOT,
            body_name="pelvis",
        ),
    )


def _halloween_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        num_steps_per_env=24,
        max_iterations=2000,
        obs_groups={"actor": ("policy",), "critic": ("critic",)},
    )


def register_halloween_mjlab_task() -> None:
    try:
        register_mjlab_task(
            task_id=_TASK_ID,
            env_cfg=make_halloween_env_cfg(num_envs=1),
            play_env_cfg=make_halloween_env_cfg(num_envs=1),
            rl_cfg=_halloween_ppo_runner_cfg(),
            runner_cls=None,
        )
    except ValueError:
        pass
    except Exception:
        logger.warning("mjlab: failed to register %s", _TASK_ID, exc_info=True)


__all__ = [
    "make_halloween_env_cfg",
    "register_halloween_mjlab_task",
    "compile_halloween_model",
    "throw_pumpkin",
    "park_pumpkin",
]
