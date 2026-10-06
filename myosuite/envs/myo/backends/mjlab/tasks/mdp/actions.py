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
3. the stages, run by priority (``muscle_stages.STAGE_ORDER``): ``motor_noise``
   (20): signal-dependent + constant noise on muscle excitations (``torch.randn``,
   independent per env and muscle; off by default); ``fatigue`` (30): muscle ctrl
   replaced by the 3CC-r active compartment, whose state is reset like the CPU env's
   (fresh, ``fatigue_reset_vec`` or ``fatigue_reset_random``); ``reafferentation``
   (40): one actuator's command is rerouted to another and the source is silenced.
   Portable custom stages (:class:`~myosuite.envs.muscle_stages.ExcitationStage`)
   run between them at their own order.

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
    STAGE_ORDER,
    ExcitationStage,
    warn_order_clash,
)
from myosuite.terms.base_action import (
    MotorNoiseCfg,
    sample_motor_noise,
    sigmoid_muscle_activation,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


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
            excitations; each env scene builds its own instances. They run at
            their ``order`` between the built-in stages.
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

        self._fatigue: TorchFatigueState | None = None
        if cfg.muscle_fatigue:
            self._fatigue = TorchFatigueState.from_mj_model(
                model, num_envs=self.num_envs, device=str(self.device)
            )
        # Global torch RNG on the sim device, seeded by mjlab (``seed_rng``).
        self._randn = functools.partial(torch.randn, device=self.device)

        self._stages = [cfg_stage() for cfg_stage in cfg.excitation_stages]
        self._pipeline = self._build_pipeline()

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

        for step in self._pipeline:
            ctrl = step(ctrl)
        self._processed_actions[:] = ctrl

    def _build_pipeline(self) -> list[Callable[[torch.Tensor], torch.Tensor]]:
        """The stages that are on, sorted by order (built-in and custom)."""
        cols = self._muscle_cols
        steps: dict[str, tuple[float, Callable[[torch.Tensor], torch.Tensor]]] = {}

        def noise(ctrl: torch.Tensor) -> torch.Tensor:
            ctrl[:, cols] = sample_motor_noise(
                ctrl[:, cols], self.cfg.motor_noise, self._randn, torch
            )
            return ctrl

        def fatigue(ctrl: torch.Tensor) -> torch.Tensor:
            ctrl[:, cols] = self._fatigue.step(ctrl[:, cols], self._env.step_dt)
            return ctrl

        def reroute(ctrl: torch.Tensor) -> torch.Tensor:
            src, dst = self._reroute
            ctrl[:, dst] = ctrl[:, src]
            ctrl[:, src] = 0.0
            return ctrl

        # The noise level can change at run time (``cfg.motor_noise``): the step is
        # always there, and does nothing and draws nothing while it is off.
        steps["noise"] = (STAGE_ORDER["noise"], noise)
        if self._fatigue is not None:
            steps["fatigue"] = (STAGE_ORDER["fatigue"], fatigue)
        if self._reroute is not None:
            steps["reroute"] = (STAGE_ORDER["reroute"], reroute)
        for stage in self._stages:
            if stage.name in steps or stage.name in STAGE_ORDER:
                raise ValueError(
                    f"The stage name {stage.name!r} is taken (built-in or twice)."
                )
            on = {
                n: o
                for n, (o, _) in steps.items()
                if n != "noise" or self.cfg.motor_noise.enabled
            }
            warn_order_clash(stage.name, stage.order, on, 3)

            def run(ctrl: torch.Tensor, stage: ExcitationStage = stage) -> torch.Tensor:
                ctrl[:, cols] = stage(ctrl[:, cols], torch)
                return ctrl

            steps[stage.name] = (stage.order, run)
        return [
            fn for _, (_, fn) in sorted(steps.items(), key=lambda i: (i[1][0], i[0]))
        ]

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
        if self._fatigue is not None:
            self._fatigue.reset(
                env_ids,
                fatigue_reset_vec=self.cfg.fatigue_reset_vec,
                fatigue_reset_random=self.cfg.fatigue_reset_random,
            )
