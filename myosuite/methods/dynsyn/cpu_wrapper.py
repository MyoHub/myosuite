# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""DynSyn on a CPU env: an action wrapper with the mapping of the mjlab action term."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.utils import RecordConstructorArgs

from myosuite.methods.dynsyn.groups import (
    DynSynGroups,
    expand_actions,
    weight_amplitude,
)


class DynSynWrapper(RecordConstructorArgs, gym.ActionWrapper):
    """DynSyn actions on a CPU env (He et al., ICML 2024; see :mod:`myosuite.methods.dynsyn`).

    The policy commands one action per muscle group, then a bounded correction per
    other muscle; the bound grows from ``start_steps`` over ``ramp_steps`` env steps.
    The mjlab equivalent is ``scripts/train_mjlab.py --dynsyn-groups``.

    Args:
        env: Env whose actions are its model's actuators, in model order.
        groups: JSON file from ``scripts/dynsyn_groups.py``, or loaded groups.
        start_steps: Env steps (summed over ``num_envs`` copies) before corrections start.
        ramp_steps: Env steps over which the bound grows to ``max_amplitude``.
        max_amplitude: Largest relative correction of the action (before the env's
            sigmoid, see :func:`~myosuite.methods.dynsyn.groups.expand_actions`).
        num_envs: Number of parallel copies of this env, so the schedule counts all steps.
        step_offset: Added to this env's steps in the schedule (continuing a training).
        frozen: Ignore this env's steps: the bound stays at ``step_offset``'s value.

    Raises:
        ValueError: If the groups were made for other actuators.
    """

    def __init__(
        self,
        env: gym.Env,
        groups: str | DynSynGroups,
        start_steps: float,
        ramp_steps: float,
        max_amplitude: float = 0.1,
        num_envs: int = 1,
        step_offset: float = 0.0,
        frozen: bool = False,
    ) -> None:
        RecordConstructorArgs.__init__(
            self,
            groups=groups,
            start_steps=start_steps,
            ramp_steps=ramp_steps,
            max_amplitude=max_amplitude,
            num_envs=num_envs,
            step_offset=step_offset,
            frozen=frozen,
        )
        gym.ActionWrapper.__init__(self, env)
        self.groups = (
            groups if isinstance(groups, DynSynGroups) else DynSynGroups.load(groups)
        )
        model = env.unwrapped.model
        names = [model.actuator(i).name.split("/")[-1] for i in range(model.nu)]
        if names != list(self.groups.muscle_names):
            raise ValueError(
                "The DynSyn groups were made for other actuators than this env's "
                f"({len(self.groups.muscle_names)} vs {len(names)})."
            )
        self.start_steps, self.ramp_steps, self.max_amplitude = (
            start_steps,
            ramp_steps,
            max_amplitude,
        )
        self.num_envs, self.env_steps = num_envs, 0
        self.step_offset, self.frozen = step_offset, frozen
        if max_amplitude == 0:  # groups only: one action per group
            box = env.action_space
            self.action_space = gym.spaces.Box(
                low=float(np.min(box.low)),
                high=float(np.max(box.high)),
                shape=(len(self.groups.groups),),
                dtype=box.dtype,
            )

    @property
    def amplitude(self) -> float:
        """Current correction bound."""
        steps = 0 if self.frozen else self.env_steps * self.num_envs
        return weight_amplitude(
            self.step_offset + steps,
            self.start_steps,
            self.ramp_steps,
            self.max_amplitude,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        return expand_actions(
            np.asarray(action, dtype=np.float64),
            self.groups.groups,
            self.amplitude,
            np,
            self.max_amplitude,
        ).astype(np.asarray(action).dtype)

    def step(self, action: np.ndarray) -> tuple:
        self.env_steps += 1
        return super().step(action)
