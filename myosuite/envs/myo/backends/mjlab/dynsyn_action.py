# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""DynSyn on mjlab: a muscle action term that takes group actions plus corrections.

:func:`apply_dynsyn` wraps every :class:`MyoMuscleActivationActionCfg` of an env
config, so any mjlab task with muscle actions trains with DynSyn (PPO or any other
algorithm; the policy only sees a different action layout of the same size).
"""

from __future__ import annotations

from typing import Any

from myosuite.core.dynsyn import DynSynGroups, expand_actions, weight_amplitude
from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import (
    MyoMuscleActivationActionCfg,
)
from myosuite.envs.myo.backends.mjlab.tasks.mdp.actions import MyoActionCfg

# The muscle action terms DynSyn can wrap: the twins' ``MyoActionCfg`` and the special
# tasks' ``MyoMuscleActivationActionCfg``.
MUSCLE_ACTION_CFGS = (MyoActionCfg, MyoMuscleActivationActionCfg)


def action_actuator_names(term: Any, env: Any) -> list[str]:
    """Actuator names of a built muscle action term, one per action column."""
    cfg = term.cfg
    if isinstance(cfg, MyoMuscleActivationActionCfg):
        return [n.split("/")[-1] for n in cfg.actuator_names]
    model = env.sim.mj_model
    ids = env.scene[cfg.entity_name].indexing.ctrl_ids.cpu().tolist()
    return [model.actuator(i).name.split("/")[-1] for i in ids]


class DynSynActionCfg:
    """Config of :class:`DynSynAction`.

    Args:
        base: The muscle action term it feeds.
        groups: Groups over ``base``'s muscles (:class:`DynSynGroups`).
        start_steps: Environment steps (all envs together) before corrections start.
        ramp_steps: Environment steps from zero to the full correction bound.
        max_amplitude: Full correction bound (0.1 in the paper).
    """

    def __init__(
        self,
        *,
        base: Any,
        groups: DynSynGroups,
        start_steps: float,
        ramp_steps: float,
        max_amplitude: float = 0.1,
    ) -> None:
        self.base, self.groups = base, groups
        self.start_steps, self.ramp_steps, self.max_amplitude = (
            start_steps,
            ramp_steps,
            max_amplitude,
        )

    def build(self, env: Any) -> DynSynAction:
        return DynSynAction(self, env)

    def __getattr__(self, name: str) -> Any:
        # Everything else (entity_name, ...) is the base term's.
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)


class DynSynAction:
    """Expand DynSyn actions to per-muscle actions and pass them to the base term."""

    def __init__(self, cfg: DynSynActionCfg, env: Any) -> None:
        import torch  # noqa: PLC0415

        self.cfg, self._env = cfg, env
        self._base = cfg.base.build(env)
        names = action_actuator_names(self._base, env)
        if names != list(cfg.groups.muscle_names):
            raise ValueError(
                "The DynSyn groups were made for other actuators than this action term's "
                f"({len(cfg.groups.muscle_names)} vs {len(names)}); run scripts/dynsyn_groups.py for this task."
            )
        # Without corrections (max_amplitude 0) the policy has one action per group only: action
        # dimensions that do nothing would still drift and inflate PPO's KL estimate.
        self._dim = (
            len(cfg.groups.groups) if cfg.max_amplitude == 0 else self._base.action_dim
        )
        self._raw = torch.zeros((env.num_envs, self._dim), device=env.device)

    @property
    def action_dim(self) -> int:
        return self._dim

    @property
    def raw_action(self) -> Any:
        return self._raw

    @property
    def amplitude(self) -> float:
        """Current correction bound, from the environment steps taken so far."""
        steps = float(self._env.common_step_counter) * self._env.num_envs
        return weight_amplitude(
            steps, self.cfg.start_steps, self.cfg.ramp_steps, self.cfg.max_amplitude
        )

    def process_actions(self, actions: Any) -> None:
        import torch  # noqa: PLC0415

        self._raw[:] = actions.to(self._raw.device)
        self._base.process_actions(
            expand_actions(self._raw, self.cfg.groups.groups, self.amplitude, torch)
        )

    def apply_actions(self) -> None:
        self._base.apply_actions()

    def __getattr__(self, name: str) -> Any:
        if name == "_base":
            raise AttributeError(name)
        return getattr(self._base, name)

    def reset(self, env_ids: Any = None) -> None:
        if env_ids is None:
            self._raw.zero_()
        else:
            self._raw[env_ids] = 0.0
        self._base.reset(env_ids)


def muscle_action_terms(env_cfg: Any) -> dict[str, Any]:
    """The env config's muscle action terms, by name."""
    return {
        k: v for k, v in env_cfg.actions.items() if isinstance(v, MUSCLE_ACTION_CFGS)
    }


def apply_dynsyn(
    env_cfg: Any,
    groups: DynSynGroups,
    start_steps: float,
    ramp_steps: float,
    max_amplitude: float = 0.1,
) -> None:
    """Replace the env config's muscle action term by its DynSyn version (in place).

    Raises:
        ValueError: The config has no muscle action term, or more than one.
    """
    terms = muscle_action_terms(env_cfg)
    if len(terms) != 1:
        raise ValueError(
            f"DynSyn needs exactly one muscle action term, found {sorted(terms)}."
        )
    ((name, base),) = terms.items()
    env_cfg.actions[name] = DynSynActionCfg(
        base=base,
        groups=groups,
        start_steps=start_steps,
        ramp_steps=ramp_steps,
        max_amplitude=max_amplitude,
    )
