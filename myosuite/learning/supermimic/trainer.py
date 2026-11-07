# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""SuperMimic training loop (paper Algorithm A.1) on an mjlab Mimic env.

Adapted from SuperMimic-Reproduced (MIT, Copyright (c) 2026 Yuxi Liu); see
``LICENSE-SuperMimic-Reproduced.txt`` in this package.

Each iteration steps the simulator once with the current policy plus control
noise and stores the transition. Every ``1 / utd`` iterations the world model is
fitted on k-step windows, then the actor is improved by backpropagation through
``horizon`` imagined steps (analytic reward, soft survival, critic bootstrap) and
the critic is fitted to lambda-returns of the same imagined rollouts.

A Polyak-averaged copy of the actor (``ema_decay``) is evaluated next to it and
exported as ``ckpt_ema.pt``; ``ckpt_final.pt`` is the actor at the end of training.
``ckpt_best.pt`` is picked on the training evaluation, so its score is biased upward.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import _musclemimic_cpu_model
from myosuite.learning.supermimic.env_io import MimicEnvIO, make_mimic_env
from myosuite.learning.supermimic.grad_clip import clip_rows_to_median
from myosuite.learning.supermimic.heads import MimicHeads
from myosuite.learning.supermimic.instanttrack import (
    FailureResets,
    FootContact,
    foot_site_indices,
    make_optimizer,
    policy_lr_multiplier,
)
from myosuite.learning.supermimic.metrics import KinematicTracking
from myosuite.learning.supermimic.policy import (
    Actor,
    Critic,
    actor_checkpoint,
    ema_update,
)
from myosuite.learning.supermimic.replay import ReplayBuffer
from myosuite.learning.supermimic.state import StateSpec, forward_axis_from_quat
from myosuite.learning.supermimic.world_model import WorldModel, WorldModelCfg


@dataclass
class SuperMimicCfg:
    """Training settings (paper Table A.5 defaults where they carry over)."""

    clip: str = ""
    out: str = "runs/supermimic"
    fullbody_model: str = "myo_sim"
    """Body model: myo_sim (myo_sim's own myofullbody with the Mimic edits) or musclemimic (the
    musclemimic_models MJCF, needs myosuite[musclemimic]). Sets MYOSUITE_MIMIC_FULLBODY_MODEL."""
    num_envs: int = 1024
    buffer_steps: int = 400
    warmup_steps: int = 50
    horizon: int = 32
    gamma: float = 0.99
    lam: float = 0.95
    imag_batch: int = 1024
    utd: float = 0.4
    wm_batches: int = 2
    ctrl_noise: float = 0.1
    lr: float = 2e-4
    lr_decay_updates: int = 50000
    lr_final_frac: float = 0.03
    critic_epochs: int = 5
    critic_minibatches: int = 4
    elementwise_clip: float = 0.1
    act_grad_clip: float = 0.0
    """Clip each sample's dJ/da_t to this multiple of the batch median norm (0: off)."""
    ema_decay: float = 0.999
    """Polyak average of the actor, evaluated and exported next to it."""
    saturation_penalty: float = 0.0
    """Barrier on raw actions beyond |a - 0.5| > band, where the sigmoid control saturates
    and BPTT gradients vanish (0 disables it)."""
    saturation_band: float = 1.0
    actor_hidden: tuple[int, ...] = (256, 256, 128)
    max_hours: float = 12.0
    max_sim_steps: float = 2e9
    eval_every: int = 500
    log_every: int = 50
    """Updates between log.jsonl records."""
    eval_envs: int = 128
    eval_steps: int = 500
    best_metric: str = "reward"
    """Checkpoint kept as ckpt_best.pt: highest "reward" or lowest leg joint-angle RMSE
    ("leg_rmse") on the training evaluation (a selection, so its score is optimistic)."""
    save_every: int = 1000
    compile: bool = True
    tf32: bool = (
        True  # TF32 matmuls (process-wide torch setting) for the world model and actor
    )
    init_actor: str = ""
    # DynSyn groups file (scripts/dynsyn_groups.py): the actor outputs one action per group.
    action_groups: str = ""
    # Posture shaping inside imagination only (the env's reward and termination stay the metric):
    imag_site_err_threshold: float = (
        1.0  # mean site error (m) that terminates an imagined rollout
    )
    imag_trunk_err_threshold_deg: float = (
        0.0  # trunk orientation error that terminates (0: off)
    )
    posture_weight: float = (
        0.0  # weight of exp(-angle / 0.2 rad) for pelvis and trunk orientation
    )
    init_world_model: str = ""
    resume: str = ""
    seed: int = 0
    device: str = "cuda:0"
    wm: WorldModelCfg = field(default_factory=WorldModelCfg)
    # InstantTrack changes (instanttrack.py); all off by default, ``--instanttrack`` sets InstantTrack's values.
    instanttrack: bool = False
    """Preset: fail_reset 0.3, contact_reward 0.25, Muon for actor and world model, lr 4e-4, 5 world-model
    batches, world model 1024x3, EMA 0.995, imagined batch 2048 with a 4x policy learning rate decayed to 1x
    over 300 updates from update 1400, fused Adam and static-shape compilation."""
    fail_reset: float = 0.0
    """Share of resets started one bin before recent termination points (0: off)."""
    fail_bin_s: float = 1.0
    """Length (s) of the bins of the failure statistics."""
    contact_reward: float = 0.0
    """Weight of the foot-contact / anti-slip term in imagination (the env's reward stays the metric)."""
    pi_opt: str = "adam"
    """Actor optimizer: adam or muon (Muon on hidden matrices, Adam elsewhere)."""
    wm_opt: str = "adam"
    """World-model optimizer: adam or muon."""
    pi_lr_mult: float = 1.0
    """Policy learning-rate multiplier, constant until pi_lr_decay_at, then decayed to 1x."""
    pi_lr_decay_at: int = 1400
    pi_lr_decay_len: int = 300
    fused_adam: bool = False
    """Fused Adam kernels (CUDA only)."""
    static_compile: bool = False
    """torch.compile with static shapes (dynamic=False)."""


def use_fullbody_model(source: str) -> None:
    """Select the full-body model for every env and analytic model built in this process.

    Call before the first model is built: the compiled models are cached per process.
    """
    import os

    from myosuite.integrations.musclemimic.fullbody_model import (
        MIMIC_FULLBODY_MODEL_ENV,
        mimic_fullbody_model_source,
    )

    os.environ[MIMIC_FULLBODY_MODEL_ENV] = source
    mimic_fullbody_model_source()  # validates the name


def apply_instanttrack_preset(cfg: SuperMimicCfg) -> SuperMimicCfg:
    """InstantTrack's settings (``--instanttrack``) for every field still at its default.

    Values given explicitly (anything that differs from :class:`SuperMimicCfg`'s default) are kept,
    so ``--instanttrack --imag-batch 512`` runs the preset with a 512 imagined batch. The horizon is
    not part of the preset (64 beat 32 on MyoFullBody).
    """
    if not cfg.instanttrack:
        return cfg
    base = SuperMimicCfg()
    preset = {
        "fail_reset": 0.3,
        "contact_reward": 0.25,
        "pi_opt": "muon",
        "wm_opt": "muon",
        "lr": 4e-4,
        "wm_batches": 5,
        "ema_decay": 0.995,
        "imag_batch": 2048,
        "pi_lr_mult": 4.0,
        "fused_adam": True,
        "static_compile": True,
    }
    for k, v in preset.items():
        if getattr(cfg, k) == getattr(base, k):
            setattr(cfg, k, v)
    wm = {"lr": 4e-4, "hidden": (1024, 1024, 1024)}
    keep = {k: v for k, v in wm.items() if getattr(cfg.wm, k) == getattr(base.wm, k)}
    cfg.wm = dataclasses.replace(cfg.wm, **keep)
    return cfg
    cfg.fail_reset = cfg.fail_reset or 0.3
    cfg.contact_reward = cfg.contact_reward or 0.25
    cfg.pi_opt = cfg.wm_opt = "muon"
    cfg.lr = 4e-4
    cfg.wm_batches = 5
    cfg.wm = dataclasses.replace(cfg.wm, lr=4e-4, hidden=(1024, 1024, 1024))
    cfg.ema_decay = 0.995
    cfg.imag_batch = 2048
    cfg.pi_lr_mult = 4.0
    cfg.fused_adam = True
    cfg.static_compile = True
    return cfg


def _sigmoid_ctrl(a: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(5.0 * (a - 0.5))


class SuperMimicTrainer:
    """World model + BPTT actor-critic on ``myoMimicFullbody-v0``.

    Args:
        cfg: Training settings.
    """

    def __init__(self, cfg: SuperMimicCfg) -> None:
        self.cfg = cfg = apply_instanttrack_preset(cfg)
        use_fullbody_model(cfg.fullbody_model)
        dev = cfg.device
        fused = cfg.fused_adam and str(dev).startswith("cuda")
        torch.manual_seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
        self.out = Path(cfg.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.env = make_mimic_env(cfg.clip, cfg.num_envs, dev, seed=cfg.seed)
        self.env.reset()
        self.io = MimicEnvIO(self.env)
        self.heads = MimicHeads.from_env(
            self.env,
            "mimic_fullbody_robot",
            site_err_threshold=cfg.imag_site_err_threshold,
            trunk_err_threshold=math.radians(cfg.imag_trunk_err_threshold_deg),
        )
        model = _musclemimic_cpu_model("fullbody")
        axis = forward_axis_from_quat(self.heads.clip.qpos[0, 3:7].cpu())
        self.spec = StateSpec(
            model.nq, model.nv, model.na, self.heads.ctrl_dt, axis
        ).to(dev)
        self.wm = WorldModel(
            model, self.spec, self.env.cfg.decimation, self.env.physics_dt, cfg.wm
        ).to(dev)
        obs_dim = self.env.obs_buf["actor"].shape[-1]
        groups = None
        if cfg.action_groups:
            from myosuite.core.dynsyn import DynSynGroups

            g = DynSynGroups.load(cfg.action_groups)
            term = self.env.action_manager.get_term("muscles")
            names = [n.split("/")[-1] for n in term.cfg.actuator_names]
            if list(g.muscle_names) != names:
                raise ValueError(
                    f"{cfg.action_groups} was made for other muscles than this env's action term"
                )
            groups = g.groups
        self.actor = Actor(obs_dim, model.nu, cfg.actor_hidden, groups).to(dev)
        # Rewards are in [0, 1 + posture_weight + 2 contact_reward], so no return exceeds this bound.
        self.critic = Critic(
            obs_dim,
            (1.0 + cfg.posture_weight + 2.0 * cfg.contact_reward) / (1.0 - cfg.gamma),
        ).to(dev)
        if cfg.init_actor:
            self.actor = Actor.from_rsl_rl(cfg.init_actor, dev)
        self.actor_ema = copy.deepcopy(self.actor).requires_grad_(False)
        if cfg.init_world_model:
            self.wm.load_state_dict(
                torch.load(cfg.init_world_model, map_location=dev)["wm"]
            )
        self.opt_wm = make_optimizer(cfg.wm_opt, self.wm, cfg.lr, fused=fused)
        self.opt_actor = make_optimizer(
            cfg.pi_opt, self.actor.mlp, cfg.lr, betas=(0.9, 0.95), fused=fused
        )
        self.opt_critic = torch.optim.Adam(
            self.critic.mlp.parameters(), lr=cfg.lr, fused=fused or None
        )
        # InstantTrack: failure-focused resets (the env's reset event asks the clip source's hook)
        self.fail_resets: FailureResets | None = None
        if cfg.fail_reset > 0:
            src = self.io.source
            self.fail_resets = FailureResets(
                [self.heads.clip.n_frames],
                round(cfg.fail_bin_s / self.heads.ctrl_dt),
                cfg.fail_reset,
                dev,
            )
            src.start_sampler = self.fail_resets.sample
        # InstantTrack: contact reward in imagination
        self.contact: FootContact | None = None
        if cfg.contact_reward > 0:
            from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
                _resolve_mimic_mjlab_ids,
            )

            ids = _resolve_mimic_mjlab_ids(
                self.env, "mimic_fullbody_robot", "fullbody"
            )["site_ids"]
            names = [model.site(int(i)).name for i in ids]
            self.contact = FootContact(
                self.heads.clip.sites, foot_site_indices(names), self.heads.ctrl_dt
            )
        self.buffer = ReplayBuffer(
            cfg.num_envs, cfg.buffer_steps, self.spec.dim, model.nu, dev
        )
        self.updates = 0
        self.sim_steps = 0
        self.elapsed = 0.0
        self.best_eval = -math.inf
        self.ep_reward = torch.zeros(cfg.num_envs, device=dev)
        self.ep_len = torch.zeros(cfg.num_envs, device=dev)
        self.done_stats: list[tuple[float, float]] = []
        if cfg.resume:
            self.load(cfg.resume)
        self._eval_env: Any = None
        comp = (
            (lambda fn: torch.compile(fn, dynamic=False))
            if cfg.static_compile
            else torch.compile
        )
        self._step = comp(self._imag_step) if cfg.compile else self._imag_step
        self._obs = comp(self._observe) if cfg.compile else self._observe

    # ------------------------------------------------------------------ simulator
    @torch.no_grad()
    def collect(self, noise: float) -> None:
        """One vectorised simulator step with the current actor plus control noise."""
        obs = self.env.obs_buf["actor"]
        self.actor.obs_normalizer.update(obs)
        u = _sigmoid_ctrl(self.actor(obs))
        u = (u + torch.randn_like(u) * noise).clamp(0.0, 1.0)
        a = self.io.action(u)
        s, f = self.io.state(), self.io.frame()
        _, rew, term, trunc, _ = self.env.step(a)
        done = term | trunc
        self.buffer.add(s, f, self.io.ctrl(a), done)
        self.sim_steps += self.cfg.num_envs
        self.ep_reward += rew
        self.ep_len += 1
        if self.fail_resets is not None:
            # the frame each ending episode ended at (before the env's reset rewrote it)
            idx = torch.nonzero(done).flatten()
            self.fail_resets.update(torch.zeros_like(idx), f[idx] + 1, term[idx])
        if bool(done.any()):
            idx = torch.nonzero(done).flatten()
            self.done_stats += list(
                zip(self.ep_reward[idx].tolist(), self.ep_len[idx].tolist())
            )
            self.ep_reward[idx] = 0
            self.ep_len[idx] = 0

    # ------------------------------------------------------------------ world model
    def update_world_model(self) -> float:
        """``wm_batches`` gradient steps of the k-step window loss."""
        losses = []
        for _ in range(self.cfg.wm_batches):
            batch = self.buffer.sample_windows(self.cfg.wm.batch, self.cfg.wm.window)
            if batch is None:
                break
            loss, _ = self.wm.window_loss(*batch)
            self.opt_wm.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.cfg.wm.grad_clip)
            self.opt_wm.step()
            losses.append(float(loss.detach()))
        return sum(losses) / max(len(losses), 1)

    # ------------------------------------------------------------------ imagination
    def _observe(self, s: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        qpos, qvel, act = self.spec.split(s)
        return self.heads.observation(qpos, qvel, act, f)

    def _imag_step(
        self,
        s: torch.Tensor,
        f: torch.Tensor,
        obs: torch.Tensor,
        probe: torch.Tensor | None = None,
        sites_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Actor + world model + analytic heads for one imagined step (compiled as one graph).

        *sites_prev* (tracked sites before the step, for the contact reward) is the previous step's
        returned sites, so the forward kinematics runs once per step.
        """
        h = self.heads
        a = self.actor(obs)
        if self.cfg.act_grad_clip > 0:
            a = clip_rows_to_median(a, self.cfg.act_grad_clip, probe)
        sat = torch.relu((a - 0.5).abs() - self.cfg.saturation_band).square().mean(-1)
        s = self.wm.step(s, _sigmoid_ctrl(a))
        f = f + 1
        qpos, qvel, act = self.spec.split(s)
        sites = h.kin.site_pos(qpos)
        r = h.reward(qpos, qvel, f, sites)
        if self.cfg.posture_weight > 0:
            pelvis, trunk = h.posture_errors(qpos, f)
            posture = 0.5 * (torch.exp(-pelvis / 0.2) + torch.exp(-trunk / 0.2))
            r = r + self.cfg.posture_weight * posture
        if self.contact is not None:
            r = r + self.cfg.contact_reward * self.contact.reward(
                sites, sites_prev, h.frame_index(f)
            )
        d = h.termination_prob(qpos, f, sites)
        return s, f, r, d, sat, h.observation(qpos, qvel, act, f, sites), sites

    def imagine(self) -> dict[str, Any]:
        """Differentiable H-step rollout of the actor through the frozen world model."""
        cfg, h = self.cfg, self.heads
        s, f = self.buffer.sample_starts(cfg.imag_batch)
        ended = h.clip_end(f)
        obs_l, rew_l, dhat_l, live_l, end_l, sat_l = [], [], [], [], [], []
        obs = self._obs(s, f)
        # Its gradient sums the per-step fractions of clipped action gradients.
        probe = (
            torch.zeros((), device=s.device, requires_grad=True)
            if cfg.act_grad_clip > 0
            else None
        )
        sites = (
            h.kin.site_pos(self.spec.split(s)[0]) if self.contact is not None else None
        )
        for _ in range(cfg.horizon):
            obs_l.append(obs)
            live_l.append((~ended).float())
            s, f, r, d, sat, obs, sites = self._step(s, f, obs, probe, sites)
            rew_l.append(r)
            sat_l.append(sat)
            dhat_l.append(d)
            just = h.clip_end(f) & ~ended
            end_l.append(just.float())
            ended = ended | just
        obs_l.append(obs)
        live_l.append((~ended).float())
        st = lambda x: torch.stack(x, 0)  # noqa: E731
        return {
            "obs": st(obs_l),
            "reward": st(rew_l),
            "dhat": st(dhat_l),
            "live": st(live_l),
            "just_ended": st(end_l),
            "saturation": st(sat_l),
            "clip_probe": probe,
        }

    def update_actor_critic(self) -> dict[str, float]:
        """One actor step on the imagined objective, then the critic on lambda-returns."""
        cfg = self.cfg
        H, g = cfg.horizon, cfg.gamma
        # The world model is frozen here (paper: stop-gradient on omega during the policy update).
        self.wm.requires_grad_(False)
        try:
            roll = self.imagine()
        finally:
            self.wm.requires_grad_(True)
        r, d, live, just = (
            roll["reward"],
            roll["dhat"],
            roll["live"],
            roll["just_ended"],
        )
        v = self.critic(roll["obs"])  # (H + 1, B)
        # Objective (paper Eq. 2): survival-weighted discounted reward; a truncation at the
        # clip end and the horizon bootstrap from the critic.
        c = torch.ones_like(r[0])
        disc = 1.0
        J = torch.zeros_like(c)
        for t in range(H):
            J = J + disc * c * r[t] * live[t]
            c = c * (1 - d[t])
            disc *= g
            J = J + disc * c * v[t + 1] * just[t]
        J = J + disc * c * v[H] * live[H]
        sat = roll["saturation"].mean()
        loss = -J.mean() / H + cfg.saturation_penalty * sat
        self.opt_actor.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.mlp.parameters(), 1.0)
        for p in self.actor.mlp.parameters():
            if p.grad is not None:
                p.grad.clamp_(-cfg.elementwise_clip, cfg.elementwise_clip)
        self.opt_actor.step()
        ema_update(self.actor_ema, self.actor, cfg.ema_decay)
        probe = roll["clip_probe"]
        clip_stats = (
            {"act_grad_clip_frac": float(probe.grad) / H}
            if probe is not None and probe.grad is not None
            else {}
        )

        # Critic (paper Eq. 3): lambda-returns of the detached rollout, fitted with fixed targets.
        with torch.no_grad():
            obs = roll["obs"].detach()
            self.critic.normalizer.update(obs[:H].reshape(-1, obs.shape[-1]))
            v = self.critic(obs)
            R = torch.zeros_like(v)
            R[H] = v[H]
            for t in reversed(range(H)):
                lam = cfg.lam * (1 - just[t])
                R[t] = r[t] + g * (1 - d[t]) * ((1 - lam) * v[t + 1] + lam * R[t + 1])
            mask = live[:H].reshape(-1)
            x, y = obs[:H].reshape(-1, obs.shape[-1]), R[:H].reshape(-1)
        n = x.shape[0]
        closs = 0.0
        for _ in range(cfg.critic_epochs):
            perm = torch.randperm(n, device=x.device).view(cfg.critic_minibatches, -1)
            for idx in perm:
                mb_loss = (
                    (self.critic(x[idx]) - y[idx]) ** 2 * mask[idx]
                ).sum() / mask[idx].sum().clamp_min(1)
                self.opt_critic.zero_grad(set_to_none=True)
                mb_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.mlp.parameters(), 1.0)
                self.opt_critic.step()
                closs = float(mb_loss)
        surv = torch.cumprod(1 - d, 0)[-1]
        return {
            "imag_return": float(J.mean()),
            "action_saturation": float(sat),
            "imag_reward": float((r * live[:H]).sum() / live[:H].sum()),
            "imag_survival": float(surv.mean()),
            "critic_loss": closs,
            "value": float(v[0].mean()),
            **clip_stats,
        }

    def _set_lr(self) -> None:
        cfg = self.cfg
        frac = min(self.updates / max(cfg.lr_decay_updates, 1), 1.0)
        scale = cfg.lr_final_frac + (1 - cfg.lr_final_frac) * 0.5 * (
            1 + math.cos(math.pi * frac)
        )
        boost = policy_lr_multiplier(
            self.updates, cfg.pi_lr_mult, cfg.pi_lr_decay_at, cfg.pi_lr_decay_len
        )
        for opt, k in (
            (self.opt_wm, 1.0),
            (self.opt_actor, boost),
            (self.opt_critic, 1.0),
        ):
            for group in opt.param_groups:
                group["lr"] = cfg.lr * scale * k

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, actor: Actor | None = None) -> dict[str, float]:
        """Deterministic *actor* (default: the trained one) from RSI starts, see :func:`evaluate_policy`."""
        cfg = self.cfg
        if self._eval_env is None:
            self._eval_env = make_mimic_env(
                cfg.clip, cfg.eval_envs, cfg.device, seed=cfg.seed + 1000
            )
            self._eval_env.reset()
            model = _musclemimic_cpu_model("fullbody")
            self._eval_io = MimicEnvIO(self._eval_env)
            self._eval_tracking = KinematicTracking(model, self.heads.clip.qpos)
        return evaluate_policy(
            self._eval_env,
            self.actor if actor is None else actor,
            cfg.eval_steps,
            self._eval_tracking,
            self._eval_io,
        )

    def _eval_score(self, rec: dict[str, Any]) -> float:
        """Higher is better; the metric that picks ``ckpt_best.pt``."""
        if self.cfg.best_metric == "leg_rmse":
            return -rec.get("eval_leg_rmse_deg", math.inf)
        return rec["eval_reward_per_step"]

    # ------------------------------------------------------------------ checkpoints
    def save(self, name: str = "latest") -> None:
        """Actor (rsl_rl layout), its average, critic, world model, optimisers and counters."""
        torch.save(
            {
                **actor_checkpoint(self.actor),
                "actor_ema_native": self.actor_ema.state_dict(),
                "critic": self.critic.state_dict(),
                "wm": self.wm.state_dict(),
                "opt_wm": self.opt_wm.state_dict(),
                "opt_actor": self.opt_actor.state_dict(),
                "opt_critic": self.opt_critic.state_dict(),
                "updates": self.updates,
                "sim_steps": self.sim_steps,
                "elapsed": self.elapsed,
                "best_eval": self.best_eval,
                "fail_resets": (
                    fr.state_dict()
                    if (fr := getattr(self, "fail_resets", None)) is not None
                    else None
                ),
                "cfg": asdict(self.cfg),
            },
            self.out / f"ckpt_{name}.pt",
        )

    def save_ema(self) -> None:
        """The averaged actor alone as ``ckpt_ema.pt``, loadable like any actor checkpoint."""
        torch.save(
            {
                **actor_checkpoint(self.actor_ema),
                "updates": self.updates,
                "sim_steps": self.sim_steps,
            },
            self.out / "ckpt_ema.pt",
        )

    def load(self, path: str) -> None:
        """Resume from :meth:`save` output (the replay buffer refills during warm-up)."""
        ck = torch.load(path, map_location=self.cfg.device, weights_only=False)
        if "actor_native" in ck:
            self.actor.load_state_dict(ck["actor_native"])
        else:
            self.actor.load_state_dict(
                {
                    ("distribution_std" if k == "distribution.std_param" else k): v
                    for k, v in ck["actor_state_dict"].items()
                }
            )
        if "actor_ema_native" in ck:
            self.actor_ema.load_state_dict(ck["actor_ema_native"])
        else:
            self.actor_ema.load_state_dict(self.actor.state_dict())
        self.critic.load_state_dict(ck["critic"])
        self.wm.load_state_dict(ck["wm"])
        self.opt_wm.load_state_dict(ck["opt_wm"])
        self.opt_actor.load_state_dict(ck["opt_actor"])
        self.opt_critic.load_state_dict(ck["opt_critic"])
        self.updates, self.sim_steps, self.elapsed = (
            ck["updates"],
            ck["sim_steps"],
            ck["elapsed"],
        )
        self.best_eval = ck.get("best_eval", -math.inf)
        fr = getattr(self, "fail_resets", None)
        if fr is not None and ck.get("fail_resets") is not None:
            fr.load_state_dict(ck["fail_resets"])

    # ------------------------------------------------------------------ main loop
    def train(self) -> None:
        """Run until ``max_hours`` of training time or ``max_sim_steps``."""
        cfg = self.cfg
        log = (self.out / "log.jsonl").open("a")
        t_w = time.time()
        for _ in range(cfg.warmup_steps):
            self.collect(max(cfg.ctrl_noise, 0.3))
        print(
            f"warmup: {cfg.warmup_steps} collects of {cfg.num_envs} envs in {time.time() - t_w:.1f} s",
            flush=True,
        )
        start = time.time() - self.elapsed
        credit = 0.0
        last: dict[str, float] = {}
        t_sim = t_upd = 0.0
        while True:
            t0 = time.time()
            self.collect(cfg.ctrl_noise)
            t_sim += time.time() - t0
            credit += cfg.utd
            while credit >= 1.0:
                credit -= 1.0
                t0 = time.time()
                self._set_lr()
                wm_loss = self.update_world_model()
                last = {"wm_loss": wm_loss, **self.update_actor_critic()}
                self.updates += 1
                t_upd += time.time() - t0
                if self.updates <= 3:  # the first updates include torch.compile
                    print(
                        f"update {self.updates}: {time.time() - t0:.1f} s, sim steps {self.sim_steps}",
                        flush=True,
                    )
                if self.updates % cfg.log_every == 0:
                    self.elapsed = time.time() - start
                    eps = self.done_stats[-500:]
                    rec = {
                        "updates": self.updates,
                        "sim_steps": self.sim_steps,
                        "hours": self.elapsed / 3600,
                        "train_ep_reward": sum(e[0] for e in eps) / max(len(eps), 1),
                        "train_ep_len": sum(e[1] for e in eps) / max(len(eps), 1),
                        "t_sim": t_sim,
                        "t_update": t_upd,
                        **last,
                    }
                    if self.updates % cfg.eval_every == 0:
                        rec.update({f"eval_{k}": v for k, v in self.evaluate().items()})
                        rec.update(
                            {
                                f"eval_ema_{k}": v
                                for k, v in self.evaluate(self.actor_ema).items()
                            }
                        )
                        # Keep the best-eval policy: runs can decay after their peak.
                        if self._eval_score(rec) > self.best_eval:
                            self.best_eval = self._eval_score(rec)
                            self.save("best")
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    self.done_stats = self.done_stats[-500:]
                    t_sim = t_upd = 0.0
                if self.updates % cfg.save_every == 0:
                    self.save()
                    self.save_ema()
            self.elapsed = time.time() - start
            if (
                self.elapsed > cfg.max_hours * 3600
                or self.sim_steps >= cfg.max_sim_steps
            ):
                break
        self.save()
        shutil.copyfile(self.out / "ckpt_latest.pt", self.out / "ckpt_final.pt")
        self.save_ema()
        rec = {
            "updates": self.updates,
            "sim_steps": self.sim_steps,
            "hours": self.elapsed / 3600,
            "final": True,
            "best_eval_score": self.best_eval,
            **{f"eval_{k}": v for k, v in self.evaluate().items()},
            **{f"eval_ema_{k}": v for k, v in self.evaluate(self.actor_ema).items()},
        }
        log.write(json.dumps(rec) + "\n")
        log.close()


@torch.no_grad()
def evaluate_policy(
    env: Any,
    actor: Any,
    steps: int,
    tracking: KinematicTracking | None = None,
    io: MimicEnvIO | None = None,
) -> dict[str, float]:
    """Mean env reward per step, early terminations per 1000 steps, episode length and muscle control of a deterministic actor.

    Starts from the env's RSI reset; episodes that end are reset by the env and continue.
    With ``tracking`` (and the env's ``io``), joint-angle and tilt errors against the
    clip are added for every step that did not end an episode.
    """
    env.reset()
    if tracking is not None:
        tracking.reset()
    obs = env.obs_buf["actor"]
    rew_sum = ctrl_sum = 0.0
    terms = 0
    ends = 0
    for _ in range(steps):
        a = actor(obs)
        ctrl_sum += float(_sigmoid_ctrl(a).mean())
        obs, rew, term, trunc, _ = env.step(a)
        obs = obs["actor"]
        rew_sum += float(rew.mean())
        terms += int(term.sum())
        ends += int((term | trunc).sum())
        if tracking is not None and io is not None:
            frame = io.frame()
            valid = ~(term | trunc) & ~io.source.clip_end(env.episode_length_buf)
            tracking.update(io.state()[:, : tracking.clip_qpos.shape[1]], frame, valid)
    n = env.num_envs
    extra = tracking.summary() if tracking is not None else {}
    return {
        "reward_per_step": rew_sum / steps,
        "terminations_per_1k_steps": 1000.0 * terms / (n * steps),
        "mean_episode_len": n * steps / max(ends, 1),
        "mean_ctrl": ctrl_sum / steps,
        **extra,
    }
