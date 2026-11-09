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

from myosuite.methods.dynsyn.groups import (
    CHECKPOINT_KEY,
    DynSynGroups,
    TrainedDynSyn,
    expand_actions,
    weight_amplitude,
)

try:
    from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import (
        MyoMuscleActivationActionCfg,
    )
    from myosuite.envs.myo.backends.mjlab.tasks.mdp.actions import MyoActionCfg
except ImportError as e:
    raise ImportError(
        "The DynSyn mjlab action term needs mjlab: pip install 'myosuite[mjlab]'"
    ) from e

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
        max_amplitude: Full correction bound.
        step_offset: Added to this env's steps in the schedule.
        frozen: Ignore this env's steps: the bound stays at ``step_offset``'s value.
    """

    def __init__(
        self,
        *,
        base: Any,
        groups: DynSynGroups,
        start_steps: float,
        ramp_steps: float,
        max_amplitude: float = 0.1,
        step_offset: float = 0.0,
        frozen: bool = False,
    ) -> None:
        self.base, self.groups = base, groups
        self.start_steps, self.ramp_steps, self.max_amplitude = (
            start_steps,
            ramp_steps,
            max_amplitude,
        )
        self.step_offset, self.frozen = step_offset, frozen

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
        self.schedule_source: DynSynAction | None = None

    @property
    def action_dim(self) -> int:
        return self._dim

    @property
    def raw_action(self) -> Any:
        return self._raw

    @property
    def env_steps(self) -> float:
        """Environment steps of the correction schedule."""
        if self.schedule_source is not None:
            return self.schedule_source.env_steps
        if self.cfg.frozen:
            return self.cfg.step_offset
        steps = float(self._env.common_step_counter) * self._env.num_envs
        return self.cfg.step_offset + steps

    @property
    def amplitude(self) -> float:
        """Current correction bound."""
        return weight_amplitude(
            self.env_steps,
            self.cfg.start_steps,
            self.cfg.ramp_steps,
            self.cfg.max_amplitude,
        )

    def trained(self) -> TrainedDynSyn:
        """The setup at the current step, as a checkpoint stores it."""
        return TrainedDynSyn(
            groups=self.cfg.groups,
            start_steps=self.cfg.start_steps,
            ramp_steps=self.cfg.ramp_steps,
            max_amplitude=self.cfg.max_amplitude,
            env_steps=self.env_steps,
            saved_counter=int(self._env.common_step_counter),
        )

    def process_actions(self, actions: Any) -> None:
        import torch  # noqa: PLC0415

        self._raw[:] = actions.to(self._raw.device)
        self._base.process_actions(
            expand_actions(
                self._raw,
                self.cfg.groups.groups,
                self.amplitude,
                torch,
                self.cfg.max_amplitude,
            )
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
    step_offset: float = 0.0,
    frozen: bool = False,
) -> None:
    """Replace the env config's muscle action term by its DynSyn version (in place).

    Args:
        env_cfg: Env config to change.
        groups: Groups over the term's muscles.
        start_steps: Environment steps before corrections start.
        ramp_steps: Environment steps from zero to the full correction bound.
        max_amplitude: Full correction bound.
        step_offset: Added to the env's steps in the schedule.
        frozen: Keep the bound at ``step_offset``'s value (evaluation).

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
        step_offset=step_offset,
        frozen=frozen,
    )


def apply_trained_dynsyn(
    env_cfg: Any, trained: TrainedDynSyn, frozen: bool = False
) -> None:
    """Apply a training run's DynSyn setup, with the bound at its checkpoint's value.

    Args:
        env_cfg: Env config to change.
        trained: From :func:`myosuite.methods.dynsyn.checkpoint.trained_dynsyn`.
        frozen: Keep the bound there (evaluation). Otherwise training goes on from
            there, also with another ``num_envs``: the runner restores the step counter
            when it loads the checkpoint, and the offset makes up the difference.
    """
    restored = 0 if frozen else trained.saved_counter * env_cfg.scene.num_envs
    apply_dynsyn(
        env_cfg,
        trained.groups,
        trained.start_steps,
        trained.ramp_steps,
        trained.max_amplitude,
        step_offset=trained.env_steps - restored,
        frozen=frozen,
    )


def dynsyn_term(env: Any) -> DynSynAction | None:
    """The env's :class:`DynSynAction`, if it has one."""
    for name, term in env.cfg.actions.items():
        if isinstance(term, DynSynActionCfg):
            return env.action_manager.get_term(name)
    return None


def track_dynsyn_in_checkpoints(runner: Any) -> None:
    """Keep a runner's checkpoints and its env's DynSyn setup together (in place).

    Every checkpoint the runner saves stores the env's setup in its ``infos``
    (:meth:`TrainedDynSyn.to_infos`), so a checkpoint without its run directory (a
    W&B download) can be resumed and evaluated as trained. Loading a DynSyn
    checkpoint into an env without DynSyn raises instead of reading its actions as
    per-muscle actions.

    Args:
        runner: An RSL-RL runner whose ``save`` takes ``infos`` (mjlab's runners).
    """
    from myosuite.methods.dynsyn.checkpoint import trained_dynsyn  # noqa: PLC0415

    term = dynsyn_term(runner.env.unwrapped)
    save, load = runner.save, runner.load

    def save_with_dynsyn(path: str, infos: dict | None = None) -> None:
        if term is not None:
            infos = {**(infos or {}), CHECKPOINT_KEY: term.trained().to_infos()}
        save(path, infos)

    def load_checked(path: str, *args: Any, **kwargs: Any) -> Any:
        if term is None and trained_dynsyn(path) is not None:
            raise ValueError(
                f"{path} was trained with DynSyn, but this env has per-muscle actions. "
                "Resume it with scripts/train_mjlab.py, which applies the run's setup."
            )
        return load(path, *args, **kwargs)

    runner.save, runner.load = save_with_dynsyn, load_checked
