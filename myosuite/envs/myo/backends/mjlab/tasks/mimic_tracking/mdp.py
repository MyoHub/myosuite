# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Manager terms of the ``myoMimic-v0`` twin.

Each term reads the shared tracking state of the motion command and calls a pure term
of :mod:`myosuite.terms.mimic_track`. The pure function is attached as ``func.pure``
(called with ``(xp, state, features, ref, **params)`` minus ``command_name``), so a
learning method can score states it did not simulate with the env cfg's own terms
and params.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from myosuite.terms import mimic_track as mt

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from .commands import MotionCommand


def _cmd(env: ManagerBasedRlEnv, command_name: str) -> MotionCommand:
    return env.command_manager.get_term(command_name)


def _pure(fn: Any) -> Any:
    def deco(term: Any) -> Any:
        term.pure = fn
        return term

    return deco


@_pure(mt.observation)
def track_obs(env: ManagerBasedRlEnv, command_name: str, name: str) -> torch.Tensor:
    """A body-state observation term of the current state."""
    s, f = _cmd(env, command_name).obs_state()
    return mt.observation(torch, s, f, name)


@_pure(mt.goal_observation)
def goal(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """The reference-only goal (lookahead and phase)."""
    return _cmd(env, command_name).goal()


@_pure(mt.reward)
def track_reward(
    env: ManagerBasedRlEnv, command_name: str, name: str, kernel: float
) -> torch.Tensor:
    """A MimicReward component of the post-step state against frame f + 1."""
    s, f, r = _cmd(env, command_name).transition()
    return mt.reward(torch, s, f, r, name, kernel)


@_pure(mt.exceeds)
def track_failure(
    env: ManagerBasedRlEnv, command_name: str, name: str, threshold: float
) -> torch.Tensor:
    """One deviation of the failure rule above its threshold."""
    s, f, r = _cmd(env, command_name).transition()
    return mt.exceeds(torch, s, f, r, name, threshold)


def motion_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """The step reached the clip's last frame (a time-out, not a failure)."""
    cmd = _cmd(env, command_name)
    m = cmd.motion
    return cmd.f + 1 >= m.clip_end[m.clip_id[cmd.f]]
