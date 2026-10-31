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
from mjlab.envs.mdp import events as mdp_events
from mjlab.envs.mdp import terminations as mdp_terminations
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.viewer.viewer_config import ViewerConfig

from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
    _init_state_from_model as _mimic_init_state_from_model,
)
from myosuite.envs.myo.backends.mjlab.mjlab_env_base import normalize_mjlab_env_ids
from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import (
    MyoMuscleActivationActionCfg,
)
from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import mujoco_cfg_from_model
from myosuite.envs.myo.backends.mjlab.tasks.mdp import SYNC_TERM, sync_forward
from myosuite.envs.myo.tasks.halloween.config import HALLOWEEN
from myosuite.envs.myo.tasks.halloween.halloween_model import (
    add_pumpkin_contact_pairs,
    compile_halloween_model,
    park_pumpkin,
    throw_pumpkin,
)
from myosuite.integrations.musclemimic.fullbody_model import (
    build_mimic_fullbody_spec,
    default_mimic_fullbody_config,
)
from myosuite.terms.halloween import halloween_scores

logger = logging.getLogger(__name__)
_TASK_ID = "myoHalloween-v0"
_ROBOT = "halloween_fullbody"
_PUMPKIN = "halloween_pumpkin"
_SIM_DT = HALLOWEEN.sim_dt
_CTRL_DT = HALLOWEEN.ctrl_dt
_PARK_POS = HALLOWEEN.park_pos


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
        size=[HALLOWEEN.pumpkin_radius, 0.0, 0.0],
        rgba=[1.0, 0.45, 0.05, 1.0],
        mass=HALLOWEEN.pumpkin_mass,
    )
    geom.contype = 1
    geom.conaffinity = 1
    geom.condim = 3
    geom.friction = [0.9, 0.1, 0.01]
    geom.solref = [0.02, 1.0]
    geom.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]
    return spec


def _muscle_and_tendon_names(
    model: mujoco.MjModel,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    muscles: list[str] = []
    tendons: list[str] = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act_{i}"
        muscles.append(name)
        tid = int(model.actuator_trnid[i, 0])
        tname = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, tid)
            if tid >= 0
            else None
        )
        tendons.append(tname or name)
    return tuple(muscles), tuple(tendons)


def _pumpkin_init_state(model: mujoco.MjModel) -> EntityCfg.InitialStateCfg:
    q = np.asarray(model.qpos0, dtype=np.float64)
    return EntityCfg.InitialStateCfg(
        pos=(float(q[0]), float(q[1]), float(q[2])),
        rot=(float(q[3]), float(q[4]), float(q[5]), float(q[6])),
    )


_PUMPKIN_RADIUS = HALLOWEEN.pumpkin_radius
_HIT_MARGIN = 0.11  # Distance shaping only; physical contacts determine hits.
_FALLEN_Z = HALLOWEEN.fall_pelvis_z
_FALLEN_HEAD_Z = HALLOWEEN.fall_head_z
_THROW_EVERY_S = HALLOWEEN.throw_interval_s
_THROW_EASY = HALLOWEEN.easy_start
_THROW_RAMP_STEPS = HALLOWEEN.ramp_steps
_EPISODE_S = HALLOWEEN.episode_s


def _obs_root(entity: str):
    """Pelvis height, orientation and body-frame velocities."""

    def _fn(env: Any) -> torch.Tensor:
        d = env.scene[entity].data
        return torch.cat(
            [
                d.root_link_pos_w[:, 2:3],
                d.root_link_quat_w,
                d.root_link_lin_vel_b,
                d.root_link_ang_vel_b,
            ],
            dim=-1,
        )

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
        return env.scene[
            entity
        ].data.data.act  # no stable API equivalent (accepted exception)

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
    links = env.scene[_ROBOT].data.body_link_pos_w  # (envs, bodies, 3)
    pumpkin = env.scene[_PUMPKIN].data.root_link_pos_w[:, None, :]
    return (links - pumpkin).norm(dim=-1).min(dim=-1).values


def _reward_dodge(robot: str, pumpkin: str):
    """Clearance from the body while the pumpkin is close (1 when it is far or at the hit distance + 0.3 m)."""

    def _fn(env: Any) -> torch.Tensor:
        p = env.scene[pumpkin].data.root_link_pos_w
        near = (p[:, :2] - env.scene[robot].data.root_link_pos_w[:, :2]).norm(
            dim=-1
        ) < 1.0
        score = ((_pumpkin_clearance(env) - _PUMPKIN_RADIUS - _HIT_MARGIN) / 0.3).clamp(
            min=0.0, max=1.0
        )
        return torch.where(near, score, torch.ones_like(score))

    return _fn


def _head_pos(env: Any, entity: str) -> torch.Tensor:
    ent = env.scene[entity]
    return ent.data.body_link_pos_w[:, ent.find_bodies("^head$")[0][0]]


def _reward_upright(entity: str):
    """Head height (0 at 1.0 m, 1 from 1.5 m up): a high pelvis alone does not count."""

    def _fn(env: Any) -> torch.Tensor:
        return halloween_scores(
            torch,
            env.scene[entity].data.root_link_pos_w,
            _head_pos(env, entity),
            _pumpkin_clearance(env),
        )["upright"]

    return _fn


def _reward_posture(entity: str):
    """Head above the pelvis: cosine between the pelvis-to-head direction and world up."""

    def _fn(env: Any) -> torch.Tensor:
        return halloween_scores(
            torch,
            env.scene[entity].data.root_link_pos_w,
            _head_pos(env, entity),
            _pumpkin_clearance(env),
        )["posture"]

    return _fn


def _reward_alive(_entity: str):
    def _fn(env: Any) -> torch.Tensor:
        return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)

    return _fn


def _term_fallen(entity: str):
    def _fn(env: Any) -> torch.Tensor:
        return (env.scene[entity].data.root_link_pos_w[:, 2] < _FALLEN_Z) | (
            _head_pos(env, entity)[:, 2] < _FALLEN_HEAD_Z
        )

    return _fn


def _term_hit(env: Any) -> torch.Tensor:
    """Retain physical pumpkin/body collisions across all five substeps."""
    contacts = env.scene["pumpkin_body_contact"].data
    return contacts.force_history.norm(dim=-1).amax(dim=(-1, -2)) > 0


def _event_park_pumpkin(env: Any, env_ids: Any) -> None:
    """Reset: the pumpkin waits on the ground far away until the first throw."""
    pumpkin = env.scene[_PUMPKIN]
    ids = normalize_mjlab_env_ids(env, env_ids)
    state = torch.zeros(len(ids), 13, device=env.device)
    state[:, :3] = env.scene.env_origins[ids] + torch.tensor(
        _PARK_POS, device=env.device
    )
    state[:, 3] = 1.0
    pumpkin.write_root_state_to_sim(state, env_ids=ids)


def _event_throw(throw_speed: float = 1.0, easy_start: float = _THROW_EASY):
    """Interval event: throw the pumpkin from the character's left or right (world y) so that it arrives at the
    upper body (chest to head height, slightly off-centre) after 0.55-0.75 s / throw_speed: standing still is a hit."""

    def _fn(env: Any, env_ids: Any) -> None:
        ids = normalize_mjlab_env_ids(env, env_ids)
        if ids.numel() == 0:
            return
        n = len(ids)

        def rnd(lo, hi):
            return lo + (hi - lo) * torch.rand(n, device=env.device)

        root = env.scene[_ROBOT].data.root_link_pos_w[ids]
        sides = torch.where(torch.rand(n, device=env.device) < 0.5, -1.0, 1.0)
        target = root.clone()
        target[:, 0] += rnd(-0.08, 0.08)
        target[:, 2] += rnd(0.25, 0.55)
        start = target.clone()
        start[:, 1] += sides * 1.6
        easy = 1.0 + (easy_start - 1.0) * max(
            0.0, 1.0 - env.common_step_counter / _THROW_RAMP_STEPS
        )
        flight = rnd(0.55, 0.75) * easy / float(throw_speed)
        state = torch.zeros(n, 13, device=env.device)
        state[:, :3] = start
        state[:, 3] = 1.0
        state[:, 7:10] = (target - start) / flight[:, None]
        state[:, 9] += (
            9.81 * flight / 2
        )  # ballistic arc back to the start height at the target
        state[:, 11] = sides * 2.5
        env.scene[_PUMPKIN].write_root_state_to_sim(state, env_ids=ids)

    return _fn


def _scene_contacts(spec: mujoco.MjSpec) -> None:
    add_pumpkin_contact_pairs(
        spec, pumpkin_geom=f"{_PUMPKIN}/pumpkin_collision", robot_prefix=f"{_ROBOT}/"
    )


def make_halloween_env_cfg(
    *, num_envs: int = 1, throw_speed: float = 1.0, curriculum: bool = False
) -> ManagerBasedRlEnvCfg:
    """Build the MJLab config with robot and projectile as separate entities."""
    if not np.isfinite(throw_speed) or not 0.5 <= throw_speed <= 2.0:
        raise ValueError("throw_speed must be finite and between 0.5 and 2.0")
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
    scene = SceneCfg(
        num_envs=num_envs,
        entities={_ROBOT: robot_cfg, _PUMPKIN: pumpkin_cfg},
        spec_fn=_scene_contacts,
        sensors=(
            ContactSensorCfg(
                name="pumpkin_body_contact",
                primary=ContactMatch(
                    mode="geom", pattern="^pumpkin_collision$", entity=_PUMPKIN
                ),
                secondary=ContactMatch(
                    mode="subtree",
                    pattern=robot_model.body(int(robot_model.jnt_bodyid[0])).name,
                    entity=_ROBOT,
                ),
                fields=("found", "force"),
                reduce="maxforce",
                history_length=5,
            ),
        ),
    )

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
            action_mode="excitation",
        )
    }
    rewards = {
        "alive": RewardTermCfg(func=_reward_alive(_ROBOT), weight=1.0),
        "upright": RewardTermCfg(func=_reward_upright(_ROBOT), weight=2.0),
        "posture": RewardTermCfg(func=_reward_posture(_ROBOT), weight=1.0),
        "dodge": RewardTermCfg(func=_reward_dodge(_ROBOT, _PUMPKIN), weight=1.0),
    }
    terminations = {
        SYNC_TERM: TerminationTermCfg(func=sync_forward),
        "time_out": TerminationTermCfg(func=mdp_terminations.time_out, time_out=True),
        "fallen": TerminationTermCfg(func=_term_fallen(_ROBOT), time_out=False),
        "hit": TerminationTermCfg(func=_term_hit, time_out=False),
    }
    # The pumpkin waits after a reset, then is thrown again and again: stand first, then dodge each throw.
    events = {
        "reset_scene_to_default": EventTermCfg(
            func=mdp_events.reset_scene_to_default, mode="reset"
        ),
        "park_pumpkin": EventTermCfg(func=_event_park_pumpkin, mode="reset"),
        "throw_pumpkin": EventTermCfg(
            func=_event_throw(throw_speed, _THROW_EASY if curriculum else 1.0),
            mode="interval",
            interval_range_s=_THROW_EVERY_S,
        ),
    }
    return ManagerBasedRlEnvCfg(
        scene=scene,
        decimation=max(1, round(_CTRL_DT / _SIM_DT)),
        episode_length_s=_EPISODE_S,
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        events=events,
        sim=SimulationCfg(
            mujoco=mujoco_cfg_from_model(robot_model), njmax=512, nconmax=256
        ),
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
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            # capped noise: with clipped activations a large std acts as extra activation and the mean action fails alone
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.2,
                "std_type": "scalar",
                "std_range": (0.01, 0.2),
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.0,  # the entropy bonus drove the std to its cap
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.03,  # 0.01 held muscle-driven tasks at the learning-rate floor
            max_grad_norm=1.0,
        ),
        num_steps_per_env=24,
        max_iterations=8000,
        obs_groups={"actor": ("policy",), "critic": ("critic",)},
    )


def register_halloween_mjlab_task() -> None:
    try:
        register_mjlab_task(
            task_id=_TASK_ID,
            env_cfg=make_halloween_env_cfg(num_envs=1),
            play_env_cfg=make_halloween_env_cfg(
                num_envs=1, curriculum=False
            ),  # playback: full-speed throws
            rl_cfg=_halloween_ppo_runner_cfg(),
            runner_cls=None,
        )
    except ValueError:
        from mjlab.tasks.registry import list_tasks

        if _TASK_ID not in list_tasks():
            raise
    except Exception:
        logger.warning("mjlab: failed to register %s", _TASK_ID, exc_info=True)


__all__ = [
    "compile_halloween_model",
    "make_halloween_env_cfg",
    "park_pumpkin",
    "register_halloween_mjlab_task",
    "throw_pumpkin",
]
