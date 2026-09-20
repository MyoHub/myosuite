# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Task config for myoSprint-v0: Bolt's full-body sprinter model on the mjlab/MuJoCo Warp backend, driven
by myo_sim.warp's Bolt-parity muscle/contact physics (not myosuite's own muscle model -- see
../actuators/bolt_muscle_action.py for why).

This is a first, minimal working env config (Phase 4 of the myo_bolt_vv porting plan is still in progress):
observations are raw qpos/qvel rather than the full 336-D Bolt-convention observation. Sufficient to
construct a working ManagerBasedRlEnv end to end and gate BoltMuscleAction's physics against myo_sim.warp
directly; the full Bolt-parity observation/transfer validation are tracked separately (Phases 4-6).

Reward lambdas cross-checked against a sibling exploration (myosuite branch
feature/myofullbody-simple-sprint, athletic_contract.py's AthleticSpec -- itself sourced from Bolt/
msk_envs_repro's own SprintConfig/LaneConfig): lambda_vel=1e-2, lambda_limit=-3e-4. Two things worth noting
from that cross-check: (1) lambda_limit=-3e-4 matches this session's own earlier finding that a GPU reward
bug had it set to +1 instead; (2) that branch's comment states "Bolt has no explicit alive term" -- taken at
face value here (alive_bonus=0), overriding this file's earlier alive-bonus placeholder.
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

# Bolt / msk_envs SprintConfig lambdas (cross-checked against athletic_contract.py, see module docstring).
LAMBDA_VEL = 1e-2
LAMBDA_LIMIT = -3e-4
FALL_HEIGHT = 0.55  # pelvis y (this model's up axis) below which an episode ends -- matches the fixed-body
# fall threshold used throughout myo_bolt_vv's own sprint trainers (scripts/msk_envs_repro/warp_sprint_env.py).
SPRINT_ENTITY = "sprinter"


def _sprint_spec_fn():
    import myo_sim

    return myo_sim.load_spec("myosprinter")


def _init_state() -> EntityCfg.InitialStateCfg:
    """Standing pose, not mjlab's zero default.

    mjlab's own InitialStateCfg default (pos=(0,0,0)) silently overrides the compiled spec's own qpos0 on
    every reset -- without this, the pelvis starts co-located with the ground contact geoms and immediately
    blows up (NaN within the first step). Tried reading the pose from the model's own qpos0/keyframe first
    (mirroring the feature/myofullbody-simple-sprint cross-check's ``_init_state()``, which works for its
    model), but full_free_noplugin.xml carries no keyframe and a trivial all-zero qpos0 -- myo_bolt_vv's own
    scripts always set the 1.05 m standing pelvis-ty height programmatically at runtime, never baked into
    the XML (scripts/msk_envs_repro/bolt_sprint_parity.py and friends), so that pattern doesn't transfer
    here. Hardcoding the same runtime value instead.
    """
    return EntityCfg.InitialStateCfg(pos=(0.0, 1.05, 0.0))


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


def _limit_pain(env) -> torch.Tensor:
    """Soft hinge-limit overshoot sum (positive when a joint is past its range); weighted by LAMBDA_LIMIT<0,
    i.e. it's a penalty. Bolt's own r_pain term -- see the paper's r = r_speed - w*r_pain + r_alive."""
    import mujoco

    sim = env.sim
    m = sim.mj_model
    qpos = sim.wp_data.qpos.numpy()
    pain = torch.zeros(qpos.shape[0], device=env.device)
    for j in range(int(m.njnt)):
        jtype = int(m.jnt_type[j])
        if jtype not in (int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE)):
            continue
        lo, hi = float(m.jnt_range[j, 0]), float(m.jnt_range[j, 1])
        if not (hi > lo) or abs(hi - lo) > 1e6:
            continue
        adr = int(m.jnt_qposadr[j])
        q = torch.as_tensor(qpos[:, adr], device=env.device)
        pain = pain + torch.clamp(torch.tensor(lo, device=env.device) - q, min=0.0)
        pain = pain + torch.clamp(q - torch.tensor(hi, device=env.device), min=0.0)
    return pain


def _fall_termination(env) -> torch.Tensor:
    return _pelvis_y(env) < FALL_HEIGHT


def make_sprint_env_cfg(num_envs: int = 64) -> ManagerBasedRlEnvCfg:
    """A minimal but real ManagerBasedRlEnvCfg for the Bolt-parity sprinter, driven by BoltMuscleAction."""
    entity_cfg = EntityCfg(spec_fn=_sprint_spec_fn, init_state=_init_state())
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
        "forward_vel": RewardTermCfg(func=_forward_vel_reward, weight=LAMBDA_VEL),
        "limit_pain": RewardTermCfg(func=_limit_pain, weight=LAMBDA_LIMIT),
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
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=1e-4,
                # This model is Y-up (gravity along -y, matching Bolt's own convention), not mjlab's Z-up
                # default -- SimulationCfg.mujoco.apply() unconditionally overwrites model.opt.gravity, so
                # it must be set explicitly here or the model falls sideways relative to its own up axis.
                # (The feature/myofullbody-simple-sprint cross-check doesn't need this override -- myo_sim's
                # own composed models are Z-up; this Y-up quirk is specific to Bolt's own OpenSim asset.)
                gravity=(0.0, -9.80665, 0.0),
                ccd_iterations=500,
            ),
            # Generous headroom for this contact-rich model (27 geoms, 297 pairs) -- avoids
            # "solver iterations limit reached" warnings seen during Phase 0 testing at the defaults.
            njmax=768,
            nconmax=384,
        ),
    )
