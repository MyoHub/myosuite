# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""mjlab action term that applies an :class:`~myosuite.utils.action_map.ActionMap`.

:class:`ActionMapAction` sits in front of a task's muscle action term: the policy
commands ``z``, the map turns it into one action per muscle, and the muscle term
(sigmoid, muscle-command stages) runs as before. :func:`apply_trained_action_map`
puts a checkpoint's map on an env config, for playback or to continue a training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import normalize_mjlab_env_ids
from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import (
    MyoMuscleActivationActionCfg,
)
from myosuite.envs.myo.backends.mjlab.tasks.mdp.actions import MyoActionCfg
from myosuite.utils.action_map import (
    ActionMap,
    TrainedActionMap,
    action_map_from_manifest,
    effective_hold_steps,
    forward_torch,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def muscle_action_terms(env_cfg: Any) -> dict[str, Any]:
    """The mjlab env config's muscle action terms, by name."""
    cfgs = (MyoActionCfg, MyoMuscleActivationActionCfg)
    return {k: v for k, v in env_cfg.actions.items() if isinstance(v, cfgs)}


def action_actuator_names(term: Any, env: ManagerBasedRlEnv) -> list[str]:
    """Actuator names of a built mjlab muscle action term, one per action column."""
    cfg = term.cfg
    if isinstance(cfg, ActionMapActionCfg):
        cfg = cfg.base
    if isinstance(cfg, MyoMuscleActivationActionCfg):
        return [n.split("/")[-1] for n in cfg.actuator_names]
    model = env.sim.mj_model
    ids = env.scene[cfg.entity_name].indexing.ctrl_ids.cpu().tolist()
    return [model.actuator(i).name.split("/")[-1] for i in ids]


@dataclass(kw_only=True)
class ActionMapActionCfg(ActionTermCfg):
    """Config of :class:`ActionMapAction`.

    Attributes:
        base: The muscle action term it feeds.
        action_map: Manifest of the map (:meth:`ActionMap.to_manifest`).
        step_offset: Added to this env's steps in ``env_steps``.
        frozen: Ignore this env's steps: ``env_steps`` stays at ``step_offset``.
    """

    base: ActionTermCfg
    action_map: dict[str, Any]
    step_offset: float = 0.0
    frozen: bool = False

    def build(self, env: ManagerBasedRlEnv) -> ActionMapAction:
        return ActionMapAction(self, env)


class ActionMapAction(ActionTerm):
    """Map policy actions to per-muscle actions and pass them to the base term."""

    cfg: ActionMapActionCfg

    def __init__(self, cfg: ActionMapActionCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self.action_map = action_map_from_manifest(cfg.action_map)
        self._base = cfg.base.build(env)
        names = action_actuator_names(self._base, env)
        if names != list(self.action_map.muscle_names):
            raise ValueError(
                f"The {self.action_map.kind} action map was made for other actuators than "
                f"this action term's ({len(self.action_map.muscle_names)} vs {len(names)})."
            )
        self._raw = torch.zeros(
            (env.num_envs, self.action_map.action_dim), device=env.device
        )
        self._latched = torch.zeros_like(self._raw)
        self._hold_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        if isinstance(cfg.base, MyoMuscleActivationActionCfg):
            mj_model = env.sim.mj_model
            act_ids = [mj_model.actuator(n).id for n in cfg.base.actuator_names]
            self._ctrl_ids = torch.tensor(act_ids, dtype=torch.long, device=env.device)
        else:
            self._ctrl_ids = env.scene[cfg.entity_name].indexing.ctrl_ids.to(
                device=env.device, dtype=torch.long
            )

    @property
    def action_dim(self) -> int:
        return self.action_map.action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    @property
    def env_steps(self) -> float:
        """Training progress the map sees (environment steps, all envs)."""
        if self.cfg.frozen:
            return self.cfg.step_offset
        steps = float(self._env.common_step_counter) * self._env.num_envs
        return self.cfg.step_offset + steps

    def trained(self) -> TrainedActionMap:
        """The map at the current step, as a checkpoint stores it."""
        return TrainedActionMap(
            self.action_map,
            env_steps=self.env_steps,
            saved_counter=int(self._env.common_step_counter),
        )

    def set_hold_steps(self, hold_steps: int) -> None:
        """Override ``action_map.hold_steps`` at runtime."""
        if hold_steps < 1:
            raise ValueError(f"hold_steps must be >= 1, got {hold_steps}")
        if hasattr(self.action_map, "hold_steps"):
            self.action_map.hold_steps = int(hold_steps)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions.to(self._raw.device)
        cur_steps = self.env_steps
        hold = effective_hold_steps(self.action_map, cur_steps)
        if hold > 1:
            mask = (self._hold_step % hold) == 0
            self._latched = torch.where(mask[:, None], self._raw, self._latched)
            z = self._latched
            self._hold_step += 1
        else:
            z = self._raw
        data = self._env.scene[self.cfg.entity_name].data.data
        state = {
            "env_steps": cur_steps,
            "actuator_length": data.actuator_length[:, self._ctrl_ids],
            "actuator_velocity": data.actuator_velocity[:, self._ctrl_ids],
        }
        self._base.process_actions(forward_torch(self.action_map, z, state))

    def apply_actions(self) -> None:
        self._base.apply_actions()

    def reset(self, env_ids: Any = None) -> None:
        ids = normalize_mjlab_env_ids(self._env, env_ids)
        self._raw[ids] = 0.0
        self._latched[ids] = 0.0
        self._hold_step[ids] = 0
        self._base.reset(env_ids)

    def __getattr__(self, name: str) -> Any:
        # Everything else (processed_action, muscle-stage state, ...) is the base term's.
        if name.startswith("__") or name == "_base":
            raise AttributeError(name)
        return getattr(self._base, name)


def apply_action_map(
    env_cfg: Any,
    action_map: ActionMap,
    step_offset: float = 0.0,
    frozen: bool = False,
) -> None:
    """Put an action map in front of the mjlab env config's muscle action term (in place).

    Args:
        env_cfg: Env config to change.
        action_map: Map over the term's muscles.
        step_offset: Added to the env's steps in ``env_steps``.
        frozen: Keep ``env_steps`` at ``step_offset`` (evaluation).

    Raises:
        ValueError: The config has no muscle action term, or more than one.
    """
    terms = muscle_action_terms(env_cfg)
    if len(terms) != 1:
        raise ValueError(
            f"An action map needs exactly one muscle action term, found {sorted(terms)}."
        )
    ((name, base),) = terms.items()
    env_cfg.actions[name] = ActionMapActionCfg(
        entity_name=base.entity_name,
        base=base,
        action_map=action_map.to_manifest(),
        step_offset=step_offset,
        frozen=frozen,
    )


def apply_trained_action_map(
    env_cfg: Any, trained: TrainedActionMap, frozen: bool = False
) -> None:
    """Apply a training run's action map, at its checkpoint's training progress.

    Args:
        env_cfg: Env config to change.
        trained: From :func:`myosuite.utils.action_map.trained_action_map`.
        frozen: Stay there (evaluation). Otherwise training goes on from there, also
            with another ``num_envs``: the runner restores the step counter when it
            loads the checkpoint, and the offset makes up the difference.
    """
    restored = 0 if frozen else trained.saved_counter * env_cfg.scene.num_envs
    apply_action_map(
        env_cfg,
        trained.action_map,
        step_offset=trained.env_steps - restored,
        frozen=frozen,
    )


def action_map_term(env: ManagerBasedRlEnv) -> ActionMapAction | None:
    """The mjlab env's :class:`ActionMapAction`, if it has one."""
    for name, term in env.cfg.actions.items():
        if isinstance(term, ActionMapActionCfg):
            return env.action_manager.get_term(name)
    return None
