# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Factory and shared action classes for mjlab task configuration.

Exports:
- ``mjlab_env_cfg_from_task_config``: build a ManagerBasedRlEnvCfg from a TaskConfig.
- ``MyoMuscleActivationActionCfg`` / ``MyoMuscleActivationAction``: shared muscle
  action term used by all mjlab tasks. Import from here, not from registration files.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import terminations as mdp_terminations
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg

from myosuite.core.config import TaskConfig
from myosuite.terms.base_action import sigmoid_muscle_activation

if TYPE_CHECKING:  # pragma: no cover
    import torch

_DEFAULT_SIM_DT = 0.002


class MyoMuscleActivationActionCfg:
    """Action term config that writes directly to muscle actuator ctrl slots.

    Standalone (does not inherit BaseActionCfg) so mjlab's BaseAction._find_targets
    is never invoked — avoiding the tendon-name mismatch that occurs when the base
    class tries to resolve muscle actuator names as tendon names.

    Args:
        entity_name: mjlab scene entity key.
        actuator_names: Tuple of muscle actuator names (without ``_tendon`` suffix).
        tendon_names: Optional tuple of tendon target names in the same order as
            ``actuator_names``.  Use this for checkpoint playback so the action
            vector preserves MuJoCo actuator order even when local actuator and
            tendon indices differ.
        action_mode: How raw policy actions are mapped to muscle excitations.
            ``"sigmoid"`` applies the canonical MyoSuite sigmoid for training.
            ``"direct"`` clamps to ``[-1, 1]`` for checkpoint playback.
            ``"excitation"`` clamps to ``[0, 1]`` when the policy already
            outputs muscle excitations directly.
        pre_physics_fn: Optional zero-argument callable ``fn(env)`` invoked at
            the end of :meth:`process_actions`, before the physics step.  Use
            this to run task-specific pre-physics logic (e.g. advancing a
            target pool) without subclassing the action term.
    """

    def __init__(
        self,
        *,
        entity_name: str,
        actuator_names: tuple[str, ...],
        tendon_names: tuple[str, ...] | None = None,
        action_mode: str = "sigmoid",
        muscle_fatigue: bool = False,
        persist_muscle_fatigue: bool = False,
        ctrl_dt: float = 0.01,
        pre_physics_fn: Callable[..., None] | None = None,
    ) -> None:
        if action_mode not in ("sigmoid", "direct", "excitation"):
            raise ValueError(
                "MyoMuscleActivationActionCfg.action_mode must be "
                f"'sigmoid', 'direct', or 'excitation', got {action_mode!r}."
            )
        self.entity_name = entity_name
        self.actuator_names = actuator_names
        self.tendon_names = tendon_names
        self.action_mode = action_mode
        self.muscle_fatigue = muscle_fatigue
        self.persist_muscle_fatigue = persist_muscle_fatigue
        self.ctrl_dt = ctrl_dt
        self.pre_physics_fn = pre_physics_fn

    def build(self, env: ManagerBasedRlEnv) -> MyoMuscleActivationAction:
        return MyoMuscleActivationAction(self, env)


class MyoMuscleActivationAction:
    """Map policy actions to muscle control targets.

    The default ``sigmoid`` mode uses the same normalisation as WalkEnvV0 on
    CPU: ``ctrl = sigmoid(5 * (a - 0.5))``.  ``direct`` mode clamps actions to
    ``[-1, 1]`` for checkpoint inference paths that already emit MuJoCo controls.
    Both modes write directly to tendon effort targets after resolving the
    actuator-to-tendon mapping from the entity.
    """

    def __init__(self, cfg: MyoMuscleActivationActionCfg, env: ManagerBasedRlEnv):
        import torch  # noqa: PLC0415

        self.cfg = cfg
        self._env = env
        self.num_envs = env.num_envs
        self.device = env.device

        self._entity = env.scene[cfg.entity_name]
        if cfg.tendon_names is not None:
            if len(cfg.tendon_names) != len(cfg.actuator_names):
                raise ValueError(
                    "MyoMuscleActivationActionCfg.tendon_names must have the "
                    "same length as actuator_names."
                )
            target_ids, resolved_names = self._entity.find_tendons(
                cfg.tendon_names,
                preserve_order=True,
            )
            if tuple(resolved_names) != tuple(cfg.tendon_names):
                raise ValueError(
                    "Resolved tendon order does not match requested tendon order."
                )
        else:
            target_ids, resolved_names = self._entity.find_actuators(
                cfg.actuator_names,
                preserve_order=True,
            )
            if tuple(resolved_names) != tuple(cfg.actuator_names):
                raise ValueError(
                    f"Resolved actuator order does not match requested action order: {resolved_names} vs {cfg.actuator_names}."
                )
        self._target_ids = torch.tensor(
            target_ids, device=self.device, dtype=torch.long
        )
        self._num_targets = len(target_ids)

        self._raw_actions = torch.zeros(
            (self.num_envs, self._num_targets), device=self.device
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._fatigue: Any = None
        if cfg.muscle_fatigue:
            from myosuite.core.muscle_conditions import TorchFatigueState  # noqa: PLC0415

            mj_model = getattr(getattr(env, "sim", None), "mj_model", None)
            if mj_model is not None:
                self._fatigue = TorchFatigueState.from_mj_model(
                    mj_model, num_envs=self.num_envs, device=str(self.device)
                )
            else:
                self._fatigue = TorchFatigueState(
                    num_envs=self.num_envs,
                    n_muscles=self._num_targets,
                    device=str(self.device),
                )

    @property
    def action_dim(self) -> int:
        return self._num_targets

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        import torch  # noqa: PLC0415

        self._raw_actions[:] = actions.to(self.device)
        if self.cfg.action_mode == "direct":
            self._processed_actions[:] = torch.clamp(self._raw_actions, -1.0, 1.0)
        elif self.cfg.action_mode == "excitation":
            self._processed_actions[:] = torch.clamp(self._raw_actions, 0.0, 1.0)
        else:
            self._processed_actions[:] = sigmoid_muscle_activation(
                self._raw_actions, torch
            )
        if self._fatigue is not None:
            self._processed_actions[:] = self._fatigue.step(
                self._processed_actions, self.cfg.ctrl_dt
            )
        if self.cfg.pre_physics_fn is not None:
            self.cfg.pre_physics_fn(self._env)

    def apply_actions(self) -> None:
        self._entity.set_tendon_effort_target(
            self._processed_actions, tendon_ids=self._target_ids
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
        if self._fatigue is not None and not self.cfg.persist_muscle_fatigue:
            self._fatigue.reset(env_ids)


def mjlab_env_cfg_from_task_config(
    cfg: TaskConfig,
    spec_fn: Callable[[], Any],
    entity_name: str,
    actuators: tuple,
    observations: dict,
    actions: dict,
    rewards: dict | None = None,
    terminations: dict | None = None,
    events: dict | None = None,
    num_envs: int = 1,
    decimation: int = 10,
    sim_cfg: SimulationCfg | None = None,
    episode_length_s: float | None = None,
) -> ManagerBasedRlEnvCfg:
    """Build a ``ManagerBasedRlEnvCfg`` from a myosuite ``TaskConfig``.

    Args:
        cfg: The myosuite task config (provides ``max_episode_steps``).
        spec_fn: Zero-argument callable returning a ``mujoco.MjSpec`` for the model.
        entity_name: Key used for the entity inside ``SceneCfg.entities``.
        actuators: Tuple of actuator cfg instances (e.g. ``XmlActuatorCfg``).
        observations: Pre-built observations dict mapping group name to ``ObservationGroupCfg``.
        actions: Pre-built actions dict.
        rewards: Optional rewards dict; omitted from cfg if None.
        terminations: Optional terminations dict; defaults to ``{"time_out": ...}`` if None.
        events: Optional events dict; omitted from cfg if None.
        num_envs: Number of parallel environments.
        decimation: Physics steps per control step.
        sim_cfg: Optional ``SimulationCfg``; defaults to ``MujocoCfg(timestep=0.002)``.
        episode_length_s: Override computed episode length. If None, computed as
            ``cfg.max_episode_steps * decimation * _DEFAULT_SIM_DT``.

    Returns:
        A fully populated ``ManagerBasedRlEnvCfg``.
    """
    if sim_cfg is None:
        sim_cfg = SimulationCfg(mujoco=MujocoCfg(timestep=_DEFAULT_SIM_DT))

    if terminations is None:
        terminations = {
            "time_out": TerminationTermCfg(
                func=mdp_terminations.time_out, time_out=True
            ),
        }

    if episode_length_s is None:
        episode_length_s = cfg.max_episode_steps * decimation * sim_cfg.mujoco.timestep

    articulation = EntityArticulationInfoCfg(actuators=actuators)
    entity_cfg = EntityCfg(spec_fn=spec_fn, articulation=articulation)
    scene_cfg = SceneCfg(num_envs=num_envs, entities={entity_name: entity_cfg})

    kwargs: dict[str, Any] = dict(
        scene=scene_cfg,
        decimation=decimation,
        episode_length_s=episode_length_s,
        observations=observations,
        actions=actions,
        terminations=terminations,
        sim=sim_cfg,
    )
    if rewards is not None:
        kwargs["rewards"] = rewards
    if events is not None:
        kwargs["events"] = events

    return ManagerBasedRlEnvCfg(**kwargs)
