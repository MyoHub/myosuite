# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""mjlab ActionTerm wrapping myo_sim.warp's Bolt-parity DGF muscle + Hunt-Crossley contact physics.

Unlike MyoMuscleActivationActionCfg (myosuite's own muscle model, used by myoLegWalk-v0 and friends), this
drives the model with myo_sim.warp.MuscleSystem / myo_sim.warp.ContactSystem -- an independent
reimplementation of Bolt's own De Groote-Fregly muscle contraction dynamics and Hunt-Crossley contact,
validated to float32 round-off against Bolt (see myo_bolt_vv's V&V suite). Required for a fair, like-for-like
transfer test of a policy trained natively on Bolt.

Action convention: raw actions are muscle excitations / motor controls in MuJoCo ctrl order (nu-dim,
[0, 1] after the policy's own squashing), not Bolt's own action order -- callers replaying a Bolt-trained
policy remap Bolt's [muscles|motors] order into this order themselves (mirroring
myo_bolt_vv/scripts/msk_envs_repro/bolt_sprint_parity.py's bolt_action_to_mj_ctrl), rather than baking that
remap into this generic action term.

Ordering note: MuscleSystem.advance() only reads/writes muscle activation and fiber-length state -- it never
touches qpos/qvel/qfrc -- so it has no data dependency on mjw.step() having already run. apply_actions() can
therefore do compute -> apply -> advance entirely before the framework calls sim.step(), with no separate
post-step hook needed (verified bit-exact against the compute/apply/step/advance order myo_bolt_vv's own
WarpBackend uses, myo_sim/tests -- ad hoc verification script, not yet a committed test).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class BoltMuscleAction(ActionTerm):
    """Writes ctrl every env.step() call, then computes+applies+advances Bolt-parity muscle/contact physics
    every physics substep (mjlab calls apply_actions() once per substep, before each sim.step())."""

    cfg: "BoltMuscleActionCfg"

    def __init__(self, cfg: "BoltMuscleActionCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        from myo_sim.sprinter_assets import load_contact, load_muscles
        from myo_sim.warp.contact_system import ContactSystem
        from myo_sim.warp.muscle_system import MuscleSystem

        sim = env.sim
        nworld = env.num_envs
        muscles = load_muscles()
        cz = load_contact()
        self._muscles = MuscleSystem(muscles, nworld)
        self._contacts = ContactSystem(
            cz["geom_flat"], cz["pairs"], nworld, sim.wp_model.nbody, naconmax=cz["naconmax"]
        )
        self._nu = sim.mj_model.nu
        self._dt = float(sim.mj_model.opt.timestep)
        self._raw_action = torch.zeros(nworld, self._nu, device=env.device)
        self._equilibrated = False

    @property
    def action_dim(self) -> int:
        return self._nu

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_action

    def process_actions(self, actions: torch.Tensor) -> None:
        # actions: (num_envs, nu) in [0, 1], MuJoCo ctrl order (motors then muscles, per the compiled model).
        self._raw_action[:] = actions.clamp(0.0, 1.0)
        wd = self._env.sim.wp_data
        wd.ctrl.assign(self._raw_action.detach().cpu().numpy().astype("float32"))
        if not self._equilibrated:
            # First call after reset: fiber length starts at the degenerate value 0 for every muscle absent
            # this (see myo_bolt_vv CLAUDE.md's "round-off traps" -- the same equilibrate() step WarpBackend
            # and every other consumer of MuscleSystem in this codebase always calls before stepping).
            self._muscles.equilibrate(wd)
            self._equilibrated = True

    def apply_actions(self) -> None:
        sim = self._env.sim
        wd = sim.wp_data
        self._muscles.compute(wd)
        self._muscles.apply(wd, m=sim.wp_model, zero=True)
        wd.xfrc_applied.zero_()
        self._contacts.compute(sim.wp_model, wd)
        self._contacts.apply(wd)
        self._muscles.advance(self._dt)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        # A per-env equilibrate would be more precise on partial resets, but MuscleSystem.equilibrate()
        # currently equilibrates every world at once; re-equilibrating the whole batch on any reset is
        # correct, only slightly wasteful. Re-equilibrate lazily on the next process_actions() call instead
        # of eagerly here, since env may not have a valid post-reset kinematic state yet at reset() time.
        self._equilibrated = False


@dataclass(kw_only=True)
class BoltMuscleActionCfg(ActionTermCfg):
    """Config for BoltMuscleAction. `entity_name` is required by the base class but unused: this action
    term writes the full model's ctrl/qfrc_applied/xfrc_applied buffers directly, not through a per-entity
    id mapping (the sprinter model's every muscle and dof belongs to a single entity in practice)."""

    def build(self, env: "ManagerBasedRlEnv") -> BoltMuscleAction:
        return BoltMuscleAction(self, env)
