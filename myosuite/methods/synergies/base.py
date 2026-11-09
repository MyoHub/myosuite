# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Action maps ``a_muscles = M(z, state)`` and the two hosts that apply them.

An :class:`ActionMap` turns a policy action ``z`` (``action_dim``) into one action per
muscle (``n_muscles``, in the order of the env's muscle action term). The same map runs
on the CPU (:class:`ActionMapWrapper`) and on mjlab (:class:`ActionMapActionCfg`).

``state`` holds what a map may depend on besides ``z``: ``"env_steps"``, the training
progress in environment steps summed over all envs (for scheduled maps such as DynSyn).

Torch and mjlab are imported only by the mjlab term, when it is built.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

import gymnasium as gym
import numpy as np
from gymnasium.utils import RecordConstructorArgs

from myosuite.utils.checkpoint_manifest import ACTION_MAP_FIELD

_ACTION_MAPS: dict[str, type] = {}


@runtime_checkable
class ActionMap(Protocol):
    """A map from policy actions to per-muscle actions.

    Attributes:
        kind: Name under which the map is stored in manifests.
    """

    kind: ClassVar[str]

    @property
    def action_dim(self) -> int:
        """Size of the policy action ``z``."""
        ...

    @property
    def muscle_names(self) -> list[str]:
        """Actuator names of the muscles, in output order."""
        ...

    @property
    def n_muscles(self) -> int:
        """Number of per-muscle outputs."""
        return len(self.muscle_names)

    def forward(self, z: np.ndarray, state: Mapping[str, Any]) -> np.ndarray:
        """``(..., action_dim)`` -> ``(..., n_muscles)`` with numpy."""
        ...

    def forward_torch(self, z: Any, state: Mapping[str, Any]) -> Any:
        """Torch version of :meth:`forward` (differentiable where the map is)."""
        raise NotImplementedError

    def to_manifest(self) -> dict[str, Any]:
        """Plain types, with ``"kind"``, that :meth:`from_manifest` reads back."""
        ...

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> ActionMap:
        """Rebuild the map from :meth:`to_manifest`."""
        ...


def register_action_map(cls: type) -> type:
    """Class decorator: make ``cls`` loadable from manifests of its ``kind``."""
    _ACTION_MAPS[cls.kind] = cls
    return cls


def action_map_from_manifest(manifest: Mapping[str, Any]) -> ActionMap:
    """Rebuild any registered action map from its manifest.

    Raises:
        ValueError: No action map of the manifest's kind is registered.
    """
    kind = manifest["kind"]
    if kind not in _ACTION_MAPS:
        raise ValueError(f"Unknown action map {kind!r}; known: {sorted(_ACTION_MAPS)}.")
    return _ACTION_MAPS[kind].from_manifest(manifest)


def _forward_torch(action_map: ActionMap, z: Any, state: Mapping[str, Any]) -> Any:
    """Torch forward, falling back to the numpy one (not differentiable)."""
    try:
        return action_map.forward_torch(z, state)
    except NotImplementedError:
        import torch  # noqa: PLC0415

        out = action_map.forward(z.detach().cpu().numpy(), state)
        return torch.as_tensor(out, dtype=z.dtype, device=z.device)


@dataclass
class TrainedActionMap:
    """Action map of a training run at one of its checkpoints.

    Attributes:
        action_map: The run's action map.
        env_steps: Training progress (environment steps) at the checkpoint.
        saved_counter: The env's step counter stored in the checkpoint.
    """

    action_map: ActionMap
    env_steps: float
    saved_counter: int = 0

    def to_infos(self) -> dict[str, Any]:
        """Plain types for a checkpoint's ``infos[ACTION_MAP_FIELD]``."""
        return {"manifest": self.action_map.to_manifest(), "env_steps": self.env_steps}

    @staticmethod
    def from_infos(
        infos: Mapping[str, Any], saved_counter: int = 0
    ) -> TrainedActionMap:
        """Read back what :meth:`to_infos` wrote."""
        return TrainedActionMap(
            action_map_from_manifest(infos["manifest"]),
            env_steps=infos["env_steps"],
            saved_counter=saved_counter,
        )


class ActionMapWrapper(RecordConstructorArgs, gym.ActionWrapper):
    """Apply an action map on a CPU env (its actions: the model's actuators, in order).

    Args:
        env: Env to wrap.
        action_map: The map; its muscles must be the env's actuators.
        num_envs: Parallel copies of this env, so ``env_steps`` counts all their steps.
        step_offset: Added to this env's steps (continuing a training).
        frozen: Ignore this env's steps: ``env_steps`` stays at ``step_offset``.

    Raises:
        ValueError: If the map was made for other actuators.
    """

    def __init__(
        self,
        env: gym.Env,
        action_map: ActionMap,
        num_envs: int = 1,
        step_offset: float = 0.0,
        frozen: bool = False,
    ) -> None:
        RecordConstructorArgs.__init__(
            self,
            action_map=action_map,
            num_envs=num_envs,
            step_offset=step_offset,
            frozen=frozen,
        )
        gym.ActionWrapper.__init__(self, env)
        model = env.unwrapped.model
        names = [model.actuator(i).name.split("/")[-1] for i in range(model.nu)]
        if names != list(action_map.muscle_names):
            raise ValueError(
                f"The {action_map.kind} action map was made for other actuators than "
                f"this env's ({len(action_map.muscle_names)} vs {len(names)})."
            )
        self.action_map = action_map
        self.num_envs, self.step_offset, self.frozen = num_envs, step_offset, frozen
        self._steps = 0
        box = env.action_space
        if action_map.action_dim != box.shape[0]:
            self.action_space = gym.spaces.Box(
                low=float(np.min(box.low)),
                high=float(np.max(box.high)),
                shape=(action_map.action_dim,),
                dtype=box.dtype,
            )

    @property
    def env_steps(self) -> float:
        """Training progress the map sees."""
        steps = 0 if self.frozen else self._steps * self.num_envs
        return self.step_offset + steps

    def action(self, action: np.ndarray) -> np.ndarray:
        return self.action_map.forward(action, {"env_steps": self.env_steps})

    def step(self, action: np.ndarray) -> tuple:
        self._steps += 1
        return super().step(action)


def muscle_action_terms(env_cfg: Any) -> dict[str, Any]:
    """The mjlab env config's muscle action terms, by name."""
    from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import (  # noqa: PLC0415
        MyoMuscleActivationActionCfg,
    )
    from myosuite.envs.myo.backends.mjlab.tasks.mdp.actions import (  # noqa: PLC0415
        MyoActionCfg,
    )

    cfgs = (MyoActionCfg, MyoMuscleActivationActionCfg)
    return {k: v for k, v in env_cfg.actions.items() if isinstance(v, cfgs)}


def action_actuator_names(term: Any, env: Any) -> list[str]:
    """Actuator names of a built mjlab muscle action term, one per action column."""
    from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import (  # noqa: PLC0415
        MyoMuscleActivationActionCfg,
    )

    cfg = term.cfg
    if isinstance(cfg, MyoMuscleActivationActionCfg):
        return [n.split("/")[-1] for n in cfg.actuator_names]
    model = env.sim.mj_model
    ids = env.scene[cfg.entity_name].indexing.ctrl_ids.cpu().tolist()
    return [model.actuator(i).name.split("/")[-1] for i in ids]


class ActionMapActionCfg:
    """Config of :class:`ActionMapAction`, an mjlab action term in front of a muscle term.

    Args:
        base: The muscle action term it feeds.
        action_map: Manifest of the map (:meth:`ActionMap.to_manifest`).
        step_offset: Added to this env's steps in ``env_steps``.
        frozen: Ignore this env's steps: ``env_steps`` stays at ``step_offset``.
    """

    def __init__(
        self,
        *,
        base: Any,
        action_map: dict[str, Any],
        step_offset: float = 0.0,
        frozen: bool = False,
    ) -> None:
        self.base, self.action_map = base, action_map
        self.step_offset, self.frozen = step_offset, frozen

    def build(self, env: Any) -> ActionMapAction:
        return ActionMapAction(self, env)

    def __getattr__(self, name: str) -> Any:
        # Everything else (entity_name, ...) is the base term's.
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)


class ActionMapAction:
    """Map policy actions to per-muscle actions and pass them to the base term."""

    def __init__(self, cfg: ActionMapActionCfg, env: Any) -> None:
        import torch  # noqa: PLC0415

        self.cfg, self._env = cfg, env
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
        # Another term whose env_steps this one follows (the success-check env).
        self.schedule_source: ActionMapAction | None = None

    @property
    def action_dim(self) -> int:
        return self.action_map.action_dim

    @property
    def raw_action(self) -> Any:
        return self._raw

    @property
    def env_steps(self) -> float:
        """Training progress the map sees."""
        if self.schedule_source is not None:
            return self.schedule_source.env_steps
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

    def process_actions(self, actions: Any) -> None:
        self._raw[:] = actions.to(self._raw.device)
        self._base.process_actions(
            _forward_torch(self.action_map, self._raw, {"env_steps": self.env_steps})
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
        trained: From :func:`myosuite.methods.synergies.checkpoint.trained_action_map`.
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


def action_map_term(env: Any) -> ActionMapAction | None:
    """The mjlab env's :class:`ActionMapAction`, if it has one."""
    for name, term in env.cfg.actions.items():
        if isinstance(term, ActionMapActionCfg):
            return env.action_manager.get_term(name)
    return None


def track_action_map_in_checkpoints(runner: Any) -> None:
    """Keep a runner's checkpoints and its env's action map together (in place).

    Every checkpoint the runner saves stores the map in ``infos[ACTION_MAP_FIELD]``,
    so a checkpoint without its run directory (a W&B download) can be resumed and
    evaluated as trained. Loading a checkpoint with an action map into an env
    without one raises instead of reading its actions as per-muscle actions.

    Args:
        runner: An RSL-RL runner whose ``save`` takes ``infos`` (mjlab's runners).
    """
    from myosuite.methods.synergies.checkpoint import (  # noqa: PLC0415
        trained_action_map,
    )

    term = action_map_term(runner.env.unwrapped)
    save, load = runner.save, runner.load

    def save_with_map(path: str, infos: dict | None = None) -> None:
        if term is not None:
            infos = {**(infos or {}), ACTION_MAP_FIELD: term.trained().to_infos()}
        save(path, infos)

    def load_checked(path: str, *args: Any, **kwargs: Any) -> Any:
        if term is None and (trained := trained_action_map(path)) is not None:
            raise ValueError(
                f"{path} was trained through the {trained.action_map.kind!r} action map, "
                "but this env has per-muscle actions. Resume it with "
                "scripts/train_mjlab.py, which applies the run's map."
            )
        return load(path, *args, **kwargs)

    runner.save, runner.load = save_with_map, load_checked
