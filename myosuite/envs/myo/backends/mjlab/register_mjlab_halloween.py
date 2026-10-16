# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Register ``myoHalloween-v0``: MyoFullBody dodge of a free-joint pumpkin.

Trainable mjlab task. Pumpkin visual mesh is Kenney Food Kit (CC0). The policy
actuates the same 354 muscles as MuscleMimic full-body; rewards encourage
lateral dodge away from the thrown pumpkin while staying upright.
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
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.viewer.viewer_config import ViewerConfig

from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import (
    MyoMuscleActivationActionCfg,
)
from myosuite.envs.myo.backends.mjlab.tasks.mdp import SYNC_TERM, sync_forward
from myosuite.envs.myo.tasks.halloween.halloween_model import (
    PUMPKIN_BODY,
    PUMPKIN_JOINT,
    build_default_halloween_spec,
    compile_halloween_model,
    park_pumpkin,
    throw_pumpkin,
)

logger = logging.getLogger(__name__)

_TASK_ID = "myoHalloween-v0"
_ENTITY = "halloween_fullbody"
_SIM_DT = 0.002
_CTRL_DT = 0.01
_MAX_EPISODE_STEPS = 500


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


def _init_state_from_model(model: mujoco.MjModel) -> EntityCfg.InitialStateCfg:
    if model.nkey:
        q = np.asarray(model.key_qpos[0], dtype=np.float64)
    else:
        q = np.asarray(model.qpos0, dtype=np.float64)
    # Robot free joint only (pumpkin parks via event).
    return EntityCfg.InitialStateCfg(
        pos=(float(q[0]), float(q[1]), float(q[2])),
        rot=(float(q[3]), float(q[4]), float(q[5]), float(q[6])),
    )


def _pumpkin_adr(model: mujoco.MjModel) -> tuple[int, int, int]:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PUMPKIN_JOINT)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PUMPKIN_BODY)
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]), int(bid)


def _obs_qpos(entity: str, pumpkin_qposadr: int):
    def _fn(env: Any) -> torch.Tensor:
        # Robot qpos only (exclude free pumpkin 7-DoF tail).
        q = env.scene[entity].data.qpos
        return q[:, :pumpkin_qposadr]

    return _fn


def _obs_qvel(entity: str, pumpkin_dofadr: int):
    def _fn(env: Any) -> torch.Tensor:
        v = env.scene[entity].data.qvel
        return v[:, :pumpkin_dofadr]

    return _fn


def _obs_act(entity: str):
    def _fn(env: Any) -> torch.Tensor:
        return env.scene[entity].data.act

    return _fn


def _obs_pumpkin_rel(entity: str, pumpkin_qposadr: int, pumpkin_dofadr: int):
    """Relative pumpkin pose/vel in root XY and height (trainable threat features)."""

    def _fn(env: Any) -> torch.Tensor:
        data = env.scene[entity].data
        root = data.qpos[:, :3]
        pumpkin = data.qpos[:, pumpkin_qposadr : pumpkin_qposadr + 3]
        pvel = data.qvel[:, pumpkin_dofadr : pumpkin_dofadr + 3]
        rel = pumpkin - root
        return torch.cat([rel, pvel], dim=-1)

    return _fn


def _reward_lateral_dodge(entity: str, pumpkin_qposadr: int):
    """Reward moving root Y away from the pumpkin's Y (correct dodge side)."""

    def _fn(env: Any) -> torch.Tensor:
        data = env.scene[entity].data
        root_y = data.qpos[:, 1]
        pumpkin_y = data.qpos[:, pumpkin_qposadr + 1]
        # Signed separation: positive when character is on the safe side of approach.
        sep = (root_y - pumpkin_y).abs()
        return sep.clamp(max=1.5)

    return _fn


def _reward_upright(entity: str):
    def _fn(env: Any) -> torch.Tensor:
        z = env.scene[entity].data.qpos[:, 2]
        return (z - 0.5).clamp(min=0.0, max=0.6)

    return _fn


def _reward_alive(entity: str):
    def _fn(env: Any) -> torch.Tensor:
        return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)

    return _fn


def _term_fallen(entity: str):
    def _fn(env: Any) -> torch.Tensor:
        return env.scene[entity].data.qpos[:, 2] < 0.35

    return _fn


def _event_reset_and_throw(model: mujoco.MjModel, throw_speed: float = 1.0):
    """Park then throw from a random side each episode reset (train curriculum)."""

    qposadr = int(
        model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PUMPKIN_JOINT)
        ]
    )
    dofadr = int(
        model.jnt_dofadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PUMPKIN_JOINT)
        ]
    )

    def _fn(env: Any, env_ids: Any) -> None:
        # Write pumpkin state through the shared sim buffers for selected envs.
        data = env.scene[_ENTITY].data
        if env_ids is None:
            idx = torch.arange(env.num_envs, device=env.device)
        else:
            idx = env_ids
        # Random side ∈ {-1, +1}
        sides = torch.where(
            torch.rand(idx.shape, device=env.device) < 0.5,
            -torch.ones_like(idx, dtype=torch.float32),
            torch.ones_like(idx, dtype=torch.float32),
        )
        root = data.qpos[idx, :3]
        spd = float(throw_speed)
        # Position
        data.qpos[idx, qposadr] = root[:, 0] + 0.35
        data.qpos[idx, qposadr + 1] = root[:, 1] + sides * 1.55
        data.qpos[idx, qposadr + 2] = root[:, 2].clamp(min=0.85)
        data.qpos[idx, qposadr + 3] = 1.0
        data.qpos[idx, qposadr + 4 : qposadr + 7] = 0.0
        # Velocity toward the body
        data.qvel[idx, dofadr] = 0.15 * spd
        data.qvel[idx, dofadr + 1] = -sides * 3.6 * spd
        data.qvel[idx, dofadr + 2] = 0.35 * spd
        data.qvel[idx, dofadr + 3 : dofadr + 6] = 0.0

    return _fn


def make_halloween_env_cfg(
    *,
    num_envs: int = 1,
    throw_speed: float = 1.0,
) -> ManagerBasedRlEnvCfg:
    """Build the mjlab env cfg for MyoHalloween."""
    model, _spec, _label = compile_halloween_model()
    muscles, tendons = _muscle_and_tendon_names(model)
    qposadr, dofadr, _bid = _pumpkin_adr(model)

    articulation = EntityArticulationInfoCfg(
        actuators=(
            _XmlWrappedActuatorCfg(
                target_names_expr=tendons,
                transmission_type=TransmissionType.TENDON,
            ),
        )
    )
    entity_cfg = EntityCfg(
        spec_fn=build_default_halloween_spec,
        articulation=articulation,
        init_state=_init_state_from_model(model),
    )
    scene = SceneCfg(num_envs=num_envs, entities={_ENTITY: entity_cfg})

    obs_terms = {
        "qpos": ObservationTermCfg(func=_obs_qpos(_ENTITY, qposadr)),
        "qvel": ObservationTermCfg(func=_obs_qvel(_ENTITY, dofadr)),
        "act": ObservationTermCfg(func=_obs_act(_ENTITY)),
        "pumpkin_rel": ObservationTermCfg(
            func=_obs_pumpkin_rel(_ENTITY, qposadr, dofadr)
        ),
    }
    observations = {
        "policy": ObservationGroupCfg(terms=obs_terms, concatenate_terms=True),
        "critic": ObservationGroupCfg(terms=obs_terms, concatenate_terms=True),
    }
    actions = {
        "muscles": MyoMuscleActivationActionCfg(
            entity_name=_ENTITY,
            actuator_names=muscles,
            tendon_names=tendons,
            action_mode="direct",
        ),
    }
    rewards = {
        "alive": RewardTermCfg(func=_reward_alive(_ENTITY), weight=1.0),
        "upright": RewardTermCfg(func=_reward_upright(_ENTITY), weight=2.0),
        "dodge": RewardTermCfg(
            func=_reward_lateral_dodge(_ENTITY, qposadr), weight=3.0
        ),
    }
    terminations = {
        SYNC_TERM: TerminationTermCfg(func=sync_forward),
        "time_out": TerminationTermCfg(func=mdp_terminations.time_out, time_out=True),
        "fallen": TerminationTermCfg(func=_term_fallen(_ENTITY), time_out=False),
    }
    events = {
        "throw_pumpkin": EventTermCfg(
            func=_event_reset_and_throw(model, throw_speed=throw_speed),
            mode="reset",
        ),
    }
    decimation = max(1, int(round(_CTRL_DT / _SIM_DT)))
    return ManagerBasedRlEnvCfg(
        scene=scene,
        decimation=decimation,
        episode_length_s=float(_MAX_EPISODE_STEPS) * _CTRL_DT,
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        events=events,
        sim=SimulationCfg(
            mujoco=MujocoCfg(timestep=_SIM_DT),
            njmax=512,
            nconmax=256,
        ),
        viewer=ViewerConfig(
            lookat=(0.0, 0.0, 0.9),
            distance=3.2,
            elevation=-10.0,
            azimuth=50.0,
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name=_ENTITY,
            body_name="pelvis",
        ),
    )


def _halloween_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
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
    """Register ``myoHalloween-v0`` with mjlab (idempotent)."""
    try:
        env_cfg = make_halloween_env_cfg(num_envs=1)
        play_cfg = make_halloween_env_cfg(num_envs=1)
        register_mjlab_task(
            task_id=_TASK_ID,
            env_cfg=env_cfg,
            play_env_cfg=play_cfg,
            rl_cfg=_halloween_ppo_runner_cfg(),
            runner_cls=None,
        )
    except ValueError:
        pass  # already registered
    except Exception:
        logger.warning("mjlab: failed to register %s", _TASK_ID, exc_info=True)


# CPU helpers re-exported for unit tests / notebooks
__all__ = [
    "make_halloween_env_cfg",
    "register_halloween_mjlab_task",
    "compile_halloween_model",
    "throw_pumpkin",
    "park_pumpkin",
]
