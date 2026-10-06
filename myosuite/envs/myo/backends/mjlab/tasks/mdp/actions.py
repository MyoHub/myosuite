# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Action term reproducing the CPU MyoSuite action pipeline.

CPU envs (``PoseEnvV0``, ``ReachEnvV0``, ...) take one action per actuator in
MuJoCo actuator order and map it to ``ctrl`` as follows:

1. clip to the action space (``[-1, 1]``, or ``[0, 1]`` for the leg-walk
   envs, when ``normalize_act``);
2. muscles (model ``na > 0``): ``sigmoid`` on muscle actuators (walk envs:
   used as-is), other actuators keep the clipped action; motors-only models:
   linear map from the action range to ``ctrlrange``;
3. the stages, run by priority (``muscle_stages.STAGE_ORDER``). **No stage is on
   by default**: noise needs a level above zero in ``motor_noise``, fatigue
   ``muscle_fatigue=True``, reafferentation a ``reroute`` pair and a custom stage
   its factory in ``excitation_stages``; ``cpu_reference.action_cfg`` sets them from
   the registration's wrappers. ``motor_noise`` (20): signal-dependent + constant
   noise on muscle excitations (``torch.randn``, independent per env and muscle);
   ``fatigue`` (30): muscle ctrl
   replaced by the 3CC-r active compartment, whose state is reset like the CPU env's
   (fresh, ``fatigue_reset_vec`` or ``fatigue_reset_random``); ``reafferentation``
   (40): one actuator's command is rerouted to another and the source is silenced.
   Portable custom stages (:class:`~myosuite.envs.muscle_stages.ExcitationStage`)
   run after them, in the order of ``excitation_stages`` (an explicit ``order`` inserts
   one earlier).

mjlab's :class:`~mjlab.envs.mdp.actions.BaseAction` only supports affine maps on
a single transmission type, so this term writes the processed ctrl of every
actuator to its own joint/tendon effort target (wrapped by ``XmlActuatorCfg``).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from myosuite.core.muscle_conditions import TorchFatigueState
from myosuite.envs.muscle_stages import (
    LATE_ORDER,
    STAGE_ORDER,
    ExcitationStage,
    check_custom_order,
    warn_order_clash,
)
from myosuite.terms.base_action import (
    MotorNoiseCfg,
    sample_motor_noise,
    sigmoid_muscle_activation,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class _TermStage:
    """A stage of :class:`MyoAction`, on the ctrl of every env, ``(num_envs, n_actuators)``.

    Noise, fatigue, reafferentation and the portable custom stages are all of this
    kind: the term sorts them by ``order``, runs them in turn and resets them with the
    envs. (mjlab has no gym wrappers, so the term owns each stage's per-env state.)
    """

    name: str = ""
    order: float = 0.0

    def active(self) -> bool:
        """Whether the stage does anything now (checked when the order is validated)."""
        return True

    def __call__(self, ctrl: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def reset(self, env_ids: torch.Tensor | slice) -> None:
        """Clear the per-env state of the envs *env_ids* (none by default)."""


class _NoiseStage(_TermStage):
    name, order = "noise", STAGE_ORDER["noise"]

    def __init__(self, term: MyoAction) -> None:
        self._term = term

    def active(self) -> bool:
        # the level can change at run time (``cfg.motor_noise``); off: no draw at all
        return self._term.cfg.motor_noise.enabled

    def __call__(self, ctrl: torch.Tensor) -> torch.Tensor:
        if not self.active():  # off: no gather/scatter on the step
            return ctrl
        cols, term = self._term._muscle_cols, self._term
        ctrl[:, cols] = sample_motor_noise(
            ctrl[:, cols], term.cfg.motor_noise, term._randn, torch
        )
        return ctrl


class _FatigueStage(_TermStage):
    name, order = "fatigue", STAGE_ORDER["fatigue"]

    def __init__(self, term: MyoAction, state: TorchFatigueState) -> None:
        self._term, self.state = term, state

    def __call__(self, ctrl: torch.Tensor) -> torch.Tensor:
        cols, term = self._term._muscle_cols, self._term
        ctrl[:, cols] = self.state.step(ctrl[:, cols], term._env.step_dt)
        return ctrl

    def reset(self, env_ids: torch.Tensor | slice) -> None:
        cfg = self._term.cfg
        self.state.reset(
            env_ids,
            fatigue_reset_vec=cfg.fatigue_reset_vec,
            fatigue_reset_random=cfg.fatigue_reset_random,
        )


class _RerouteStage(_TermStage):
    name, order = "reroute", STAGE_ORDER["reroute"]

    def __init__(self, src: int, dst: int) -> None:
        self._src, self._dst = src, dst

    def __call__(self, ctrl: torch.Tensor) -> torch.Tensor:
        ctrl[:, self._dst] = ctrl[:, self._src]
        ctrl[:, self._src] = 0.0
        return ctrl


class _CustomStage(_TermStage):
    """A portable :class:`~myosuite.envs.muscle_stages.ExcitationStage` on the muscle columns."""

    def __init__(self, term: MyoAction, stage: ExcitationStage) -> None:
        self._term, self.stage = term, stage
        self.name = stage.name
        self.order = LATE_ORDER if stage.order is None else stage.order
        self.explicit_order = stage.order

    def __call__(self, ctrl: torch.Tensor) -> torch.Tensor:
        cols = self._term._muscle_cols
        ctrl[:, cols] = self.stage(ctrl[:, cols], torch)
        return ctrl

    def reset(self, env_ids: torch.Tensor | slice) -> None:
        self.stage.reset(env_ids)


@dataclass(kw_only=True)
class MyoActionCfg(ActionTermCfg):
    """Config for :class:`MyoAction`.

    Attributes:
        normalize_act: CPU ``normalize_act`` (normalized action space).
        action_range: Normalized action space bounds.
        muscle_sigmoid: Map muscle actions through the MyoSuite sigmoid.
        muscle_fatigue: Apply the 3CC-r fatigue model to muscle ctrl.
        fatigue_reset_vec: CPU ``fatigue_reset_vec``: fatigued fraction ``MF``
            of each muscle at every reset (``None``: start fresh).
        fatigue_reset_random: CPU ``fatigue_reset_random``: draw a random
            fatigue state per env at every reset.
        reroute: ``(source, destination)`` actuator names for reafferentation:
            ``ctrl[dst] = ctrl[src]; ctrl[src] = 0``.
        motor_noise: Noise on muscle excitations, applied before fatigue.
        excitation_stages: Factories of portable custom stages
            (:class:`~myosuite.envs.muscle_stages.ExcitationStage`) on the muscle
            excitations; each env scene builds its own instances. They run after the
            built-in stages in list order, or at their explicit ``order``.
    """

    normalize_act: bool = True
    action_range: tuple[float, float] = (-1.0, 1.0)
    muscle_sigmoid: bool = True
    muscle_fatigue: bool = False
    fatigue_reset_vec: tuple[float, ...] | None = None
    fatigue_reset_random: bool = False
    reroute: tuple[str, str] | None = None
    motor_noise: MotorNoiseCfg = field(default_factory=MotorNoiseCfg)
    excitation_stages: tuple[Callable[[], ExcitationStage], ...] = ()

    def build(self, env: ManagerBasedRlEnv) -> MyoAction:
        return MyoAction(self, env)


class MyoAction(ActionTerm):
    """One action per entity actuator, processed like the CPU envs."""

    cfg: MyoActionCfg

    def __init__(self, cfg: MyoActionCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg=cfg, env=env)
        model = env.sim.mj_model
        entity = self._entity
        ctrl_ids = entity.indexing.ctrl_ids.cpu().tolist()
        joint_ids = entity.indexing.joint_ids.cpu().tolist()
        tendon_ids = entity.indexing.tendon_ids.cpu().tolist()

        joint_cols, joint_targets, tendon_cols, tendon_targets = [], [], [], []
        for col, act_id in enumerate(ctrl_ids):
            trn_type = int(model.actuator_trntype[act_id])
            trn_id = int(model.actuator_trnid[act_id, 0])
            if trn_type == mujoco.mjtTrn.mjTRN_JOINT:
                joint_cols.append(col)
                joint_targets.append(joint_ids.index(trn_id))
            elif trn_type == mujoco.mjtTrn.mjTRN_TENDON:
                tendon_cols.append(col)
                tendon_targets.append(tendon_ids.index(trn_id))
            else:
                raise ValueError(
                    f"Actuator {model.actuator(act_id).name!r}: only joint and "
                    "tendon transmissions are supported."
                )
        for kind, targets in (("joint", joint_targets), ("tendon", tendon_targets)):
            if len(set(targets)) != len(targets):
                raise ValueError(f"Several actuators drive the same {kind}.")

        def _long(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.long, device=self.device)

        self._joint_cols, self._joint_targets = _long(joint_cols), _long(joint_targets)
        self._tendon_cols = _long(tendon_cols)
        self._tendon_targets = _long(tendon_targets)

        self._action_dim = len(ctrl_ids)
        dyn_type = model.actuator_dyntype[ctrl_ids]
        self._muscle_cols = _long(
            [i for i, d in enumerate(dyn_type) if d == mujoco.mjtDyn.mjDYN_MUSCLE]
        )
        self._has_activation = int(model.actuator_actnum[ctrl_ids].sum()) > 0
        ctrl_range = torch.as_tensor(
            model.actuator_ctrlrange[ctrl_ids], dtype=torch.float32, device=self.device
        )
        self._ctrl_lo, self._ctrl_hi = ctrl_range[:, 0], ctrl_range[:, 1]

        self._reroute: tuple[int, int] | None = None
        if cfg.reroute is not None:
            names = list(entity.actuator_names)
            self._reroute = (names.index(cfg.reroute[0]), names.index(cfg.reroute[1]))

        # Global torch RNG on the sim device, seeded by mjlab (``seed_rng``).
        self._randn = functools.partial(torch.randn, device=self.device)

        self._stages = self._build_stages(model)

        self._raw_actions = torch.zeros(
            self.num_envs, self._action_dim, device=self.device
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_action(self) -> torch.Tensor:
        """The ctrl written to the actuators (after all CPU processing)."""
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        a_lo, a_hi = self.cfg.action_range
        if self.cfg.normalize_act:
            ctrl = torch.clamp(actions, a_lo, a_hi)
        else:
            ctrl = torch.clamp(actions, self._ctrl_lo, self._ctrl_hi)

        if self.cfg.normalize_act and self._has_activation:
            if self.cfg.muscle_sigmoid:
                ctrl[:, self._muscle_cols] = sigmoid_muscle_activation(
                    ctrl[:, self._muscle_cols], torch
                )
        elif self.cfg.normalize_act:
            ctrl = self._ctrl_lo + (ctrl - a_lo) / (a_hi - a_lo) * (
                self._ctrl_hi - self._ctrl_lo
            )

        for stage in self._stages:
            ctrl = stage(ctrl)
        self._processed_actions[:] = ctrl

    def _build_stages(self, model: mujoco.MjModel) -> list[_TermStage]:
        """The stages that are configured (noise: always, it can be switched on later), by order."""
        cfg = self.cfg
        stages: list[_TermStage] = [_NoiseStage(self)]
        if cfg.muscle_fatigue:
            state = TorchFatigueState.from_mj_model(
                model, num_envs=self.num_envs, device=str(self.device)
            )
            stages.append(_FatigueStage(self, state))
        if self._reroute is not None:
            stages.append(_RerouteStage(*self._reroute))
        for make_stage in cfg.excitation_stages:
            custom = _CustomStage(self, make_stage())
            if custom.name in {st.name for st in stages} | set(STAGE_ORDER):
                raise ValueError(
                    f"The stage name {custom.name!r} is taken (built-in or twice)."
                )
            check_custom_order(custom.name, custom.explicit_order)
            if custom.explicit_order is not None:
                on = {st.name: st.order for st in stages if st.active()}
                warn_order_clash(custom.name, custom.order, on, 3)
            stages.append(custom)
        # Stable: stages of equal order keep the list order.
        return sorted(stages, key=lambda st: st.order)

    @property
    def _fatigue(self) -> TorchFatigueState | None:
        """The fatigue state of the fatigue stage (``None`` without fatigue)."""
        return next(
            (st.state for st in self._stages if isinstance(st, _FatigueStage)), None
        )

    @property
    def stage_names(self) -> tuple[str, ...]:
        """Names of the stages, in the order they run (noise is a no-op while off)."""
        return tuple(st.name for st in self._stages)

    def apply_actions(self) -> None:
        if len(self._joint_cols):
            self._entity.set_joint_effort_target(
                self._processed_actions[:, self._joint_cols],
                joint_ids=self._joint_targets,
            )
        if len(self._tendon_cols):
            self._entity.set_tendon_effort_target(
                self._processed_actions[:, self._tendon_cols],
                tendon_ids=self._tendon_targets,
            )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        for stage in self._stages:
            stage.reset(env_ids)
