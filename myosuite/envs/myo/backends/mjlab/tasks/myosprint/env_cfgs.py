# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Task config for myoSprint-v0: Bolt's full-body sprinter model on the mjlab/MuJoCo Warp backend, driven
by myo_sim.warp's Bolt-parity muscle/contact physics (not myosuite's own muscle model -- see
../actuators/bolt_muscle_action.py for why).

This is a first, minimal working env config (Phase 4 of the myo_bolt_vv porting plan is still in progress):
observations are raw qpos/qvel rather than the full 336-D Bolt-convention observation, and the reward is a
simplified version of the Bolt sprinting paper's r = r_speed - w*r_pain + r_alive (pain term not yet
wired). Sufficient to construct a working ManagerBasedRlEnv end to end and gate BoltMuscleAction's physics
against myo_sim.warp directly; the full Bolt-parity observation/reward/transfer validation are tracked
separately (Phases 4-6 of the plan).
"""
from __future__ import annotations

import torch
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim.sim import SimulationCfg, MujocoCfg
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.envs.mdp.terminations import time_out

from myosuite.envs.myo.backends.mjlab.actuators.bolt_muscle_action import BoltMuscleActionCfg

FALL_HEIGHT = 0.55  # pelvis y (this model's up axis) below which an episode ends -- matches the fixed-body
# fall threshold used throughout myo_bolt_vv's own sprint trainers (scripts/msk_envs_repro/warp_sprint_env.py).
SPRINT_ENTITY = "sprinter"


def _sprint_spec_fn():
    import myo_sim

    return myo_sim.load_spec("myosprinter")


def _obs_qpos(env) -> torch.Tensor:
    return torch.as_tensor(env.sim.wp_data.qpos.numpy(), device=env.device)


def _obs_qvel(env) -> torch.Tensor:
    return torch.as_tensor(env.sim.wp_data.qvel.numpy(), device=env.device)


def _pelvis_y(env) -> torch.Tensor:
    # qpos[:, 1] is the pelvis y coordinate (this model's up axis) -- freejoint pelvis is qpos[0:3].
    return torch.as_tensor(env.sim.wp_data.qpos.numpy()[:, 1], device=env.device)


def _forward_vel_reward(env) -> torch.Tensor:
    # qvel[:, 0] is the pelvis x (forward) linear velocity.
    return torch.as_tensor(env.sim.wp_data.qvel.numpy()[:, 0], device=env.device)


def _alive_reward(env) -> torch.Tensor:
    y = _pelvis_y(env)
    return torch.clamp(y / 1.0, 0.0, 1.0)


def _fall_termination(env) -> torch.Tensor:
    return _pelvis_y(env) < FALL_HEIGHT


def make_sprint_env_cfg(num_envs: int = 64) -> ManagerBasedRlEnvCfg:
    """A minimal but real ManagerBasedRlEnvCfg for the Bolt-parity sprinter, driven by BoltMuscleAction."""
    # mjlab's own InitialStateCfg default (pos=(0,0,0)) overrides the compiled spec's own qpos0 on every
    # reset -- without this, the pelvis starts co-located with the ground contact geoms and immediately
    # blows up (NaN within the first step). 1.05 m matches the standing pelvis-ty height used throughout
    # myo_bolt_vv's own sprint scripts (scripts/msk_envs_repro/bolt_sprint_parity.py and friends).
    init_state = EntityCfg.InitialStateCfg(pos=(0.0, 1.05, 0.0))
    entity_cfg = EntityCfg(spec_fn=_sprint_spec_fn, init_state=init_state)
    scene_cfg = SceneCfg(num_envs=num_envs, env_spacing=3.0, entities={SPRINT_ENTITY: entity_cfg})

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "qpos": ObservationTermCfg(func=_obs_qpos),
                "qvel": ObservationTermCfg(func=_obs_qvel),
            },
        ),
    }
    actions = {
        "bolt_muscles": BoltMuscleActionCfg(entity_name=SPRINT_ENTITY),
    }
    rewards = {
        "forward_vel": RewardTermCfg(func=_forward_vel_reward, weight=1.0),
        "alive": RewardTermCfg(func=_alive_reward, weight=0.2),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
        "fall": TerminationTermCfg(func=_fall_termination, time_out=False),
    }
    events = {
        "reset_scene_to_default": EventTermCfg(func=reset_scene_to_default, mode="reset"),
    }

    return ManagerBasedRlEnvCfg(
        decimation=10,
        scene=scene_cfg,
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        events=events,
        episode_length_s=10.0,
        # This model is Y-up (gravity along -y, matching Bolt's own convention), not mjlab's Z-up default --
        # SimulationCfg.mujoco.apply() unconditionally overwrites model.opt.gravity, so it must be set
        # explicitly here or the model falls sideways relative to its own up axis.
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=1e-4, gravity=(0.0, -9.80665, 0.0))),
    )
