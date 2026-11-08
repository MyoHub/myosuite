# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# this package for the notices of the code it includes.
"""Manager terms of the tracking task: the muscle excitation action and MuscleMimic's observation, reward and
termination terms, each a thin wrapper of its analytic implementation in :mod:`.heads` (attached with ``analytic``
so the runner evaluates the same term on imagined states).

* action: one excitation in [0, 1] per muscle, in actuator (= activation) order, written as the effort target of the
  muscle's tendon (muscles and tendons are declared in different orders, so each actuator's tendon is resolved by
  name);
* observations: the decoder sees the full state (root height and orientation, every joint position and velocity,
  muscle length / velocity / force / excitation / activation, foot touch) and the state-dependent part of
  GoalTrajMimic (root-relative site positions, rotations and velocities); the encoder the reference-only part;
* rewards: MuscleMimic's MimicReward, one term per component, comparing the post-step state with frame f + 1;
* terminations: MuscleMimic's deviation checks (hard here, sigmoid-softened in imagination) and the clip's last
  frame (a time-out).

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at commit 8a6244b,
``src/muscle_supermimic/tasks/tracking/mdp/{actions,observations,rewards,terminations}.py``, merged into one module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp import time_out  # noqa: F401
from mjlab.envs.mdp.actions import TendonEffortActionCfg
from mjlab.envs.mdp.actions.actions import TendonEffortAction

from . import heads as H
from .heads import StepCtx, TrackingHeads, analytic

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from .commands import MotionCommand


def _obs_ctx(
    env: ManagerBasedRlEnv, command_name: str
) -> tuple[TrackingHeads, StepCtx]:
    cmd: MotionCommand = env.command_manager.get_term(command_name)
    return cmd.heads, cmd.obs_ctx()


def _ctx(env: ManagerBasedRlEnv, command_name: str) -> tuple[TrackingHeads, StepCtx]:
    cmd: MotionCommand = env.command_manager.get_term(command_name)
    return cmd.heads, cmd.transition_ctx()


# ---------------- action


class MuscleExcitationAction(TendonEffortAction):
    def _find_targets(
        self, cfg: MuscleExcitationActionCfg
    ) -> tuple[list[int], list[str]]:
        tendon_names = list(self._entity.tendon_names)
        names = [
            act.target.split("/")[-1] for act in self._entity.spec.actuators
        ]  # as Entity.tendon_names
        return [tendon_names.index(n) for n in names], names


@dataclass(kw_only=True)
class MuscleExcitationActionCfg(TendonEffortActionCfg):
    actuator_names: tuple[str, ...] = (
        ".*",
    )  # resolved from the actuators, see MuscleExcitationAction
    clip: dict[str, tuple] | None = field(default_factory=lambda: {".*": (0.0, 1.0)})

    def build(self, env: ManagerBasedRlEnv) -> MuscleExcitationAction:
        return MuscleExcitationAction(self, env)


# ---------------- observations


@analytic(TrackingHeads.root_pos)
def root_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.root_pos(*_obs_ctx(env, command_name))


@analytic(TrackingHeads.joint_pos)
def joint_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.joint_pos(*_obs_ctx(env, command_name))


@analytic(TrackingHeads.root_vel)
def root_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.root_vel(*_obs_ctx(env, command_name))


@analytic(TrackingHeads.joint_vel)
def joint_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.joint_vel(*_obs_ctx(env, command_name))


@analytic(TrackingHeads.muscle_length)
def muscle_length(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.muscle_length(*_obs_ctx(env, command_name))


@analytic(TrackingHeads.muscle_velocity)
def muscle_velocity(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.muscle_velocity(*_obs_ctx(env, command_name))


@analytic(TrackingHeads.muscle_force)
def muscle_force(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.muscle_force(*_obs_ctx(env, command_name))


@analytic(TrackingHeads.muscle_excitation)
def muscle_excitation(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.muscle_excitation(*_obs_ctx(env, command_name))


@analytic(TrackingHeads.muscle_activation)
def muscle_activation(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.muscle_activation(*_obs_ctx(env, command_name))


@analytic(TrackingHeads.touch)
def touch(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.touch(*_obs_ctx(env, command_name))


@analytic(TrackingHeads.site_rpos)
def site_rpos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.site_rpos(*_obs_ctx(env, command_name))


@analytic(TrackingHeads.site_rangles)
def site_rangles(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.site_rangles(*_obs_ctx(env, command_name))


@analytic(TrackingHeads.site_rvel)
def site_rvel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.site_rvel(*_obs_ctx(env, command_name))


@analytic(TrackingHeads.goal)
def motion_goal(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.goal(*_obs_ctx(env, command_name))


# ---------------- rewards


@analytic(H.mimic_qpos)
def mimic_qpos(env: ManagerBasedRlEnv, command_name: str, w_exp: float) -> torch.Tensor:
    return H.mimic_qpos(*_ctx(env, command_name), w_exp=w_exp)


@analytic(H.mimic_qvel)
def mimic_qvel(env: ManagerBasedRlEnv, command_name: str, w_exp: float) -> torch.Tensor:
    return H.mimic_qvel(*_ctx(env, command_name), w_exp=w_exp)


@analytic(H.mimic_root_pos)
def mimic_root_pos(
    env: ManagerBasedRlEnv, command_name: str, w_exp: float
) -> torch.Tensor:
    return H.mimic_root_pos(*_ctx(env, command_name), w_exp=w_exp)


@analytic(H.mimic_root_vel)
def mimic_root_vel(
    env: ManagerBasedRlEnv, command_name: str, w_exp: float
) -> torch.Tensor:
    return H.mimic_root_vel(*_ctx(env, command_name), w_exp=w_exp)


@analytic(H.mimic_rpos)
def mimic_rpos(env: ManagerBasedRlEnv, command_name: str, w_exp: float) -> torch.Tensor:
    return H.mimic_rpos(*_ctx(env, command_name), w_exp=w_exp)


@analytic(H.mimic_rquat)
def mimic_rquat(
    env: ManagerBasedRlEnv, command_name: str, w_exp: float
) -> torch.Tensor:
    return H.mimic_rquat(*_ctx(env, command_name), w_exp=w_exp)


@analytic(H.mimic_rvel_rot)
def mimic_rvel_rot(
    env: ManagerBasedRlEnv, command_name: str, w_exp: float
) -> torch.Tensor:
    return H.mimic_rvel_rot(*_ctx(env, command_name), w_exp=w_exp)


@analytic(H.mimic_rvel_lin)
def mimic_rvel_lin(
    env: ManagerBasedRlEnv, command_name: str, w_exp: float
) -> torch.Tensor:
    return H.mimic_rvel_lin(*_ctx(env, command_name), w_exp=w_exp)


# ---------------- terminations


@analytic(H.bad_body_pos_dev)
def bad_body_pos_dev(
    env: ManagerBasedRlEnv, command_name: str, threshold: float, soft_width: float
) -> torch.Tensor:
    del soft_width  # imagination only
    return H.bad_body_pos_dev(*_ctx(env, command_name), threshold=threshold)


@analytic(H.bad_root_pos_dev)
def bad_root_pos_dev(
    env: ManagerBasedRlEnv, command_name: str, threshold: float, soft_width: float
) -> torch.Tensor:
    del soft_width  # imagination only
    return H.bad_root_pos_dev(*_ctx(env, command_name), threshold=threshold)


@analytic(H.bad_root_ori_dev)
def bad_root_ori_dev(
    env: ManagerBasedRlEnv, command_name: str, threshold: float, soft_width: float
) -> torch.Tensor:
    del soft_width  # imagination only
    return H.bad_root_ori_dev(*_ctx(env, command_name), threshold=threshold)


def motion_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """The step reached the clip's last frame (a time-out, not a failure)."""
    cmd: MotionCommand = env.command_manager.get_term(command_name)
    m = cmd.motion
    return cmd.f + 1 >= m.clip_end[m.clip_id[cmd.f]]
