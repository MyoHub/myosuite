# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""SuperMimic training loop (paper Algorithm A.1) on an mjlab Mimic env.

Each iteration steps the simulator once with the current policy plus control
noise and stores the transition. Every ``1 / utd`` iterations the world model is
fitted on k-step windows, then the actor is improved by backpropagation through
``horizon`` imagined steps (analytic reward, soft survival, critic bootstrap) and
the critic is fitted to lambda-returns of the same imagined rollouts.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import _musclemimic_cpu_model
from myosuite.learning.supermimic.env_io import MimicEnvIO, make_mimic_env
from myosuite.learning.supermimic.heads import MimicHeads
from myosuite.learning.supermimic.policy import Actor, Critic
from myosuite.learning.supermimic.replay import ReplayBuffer
from myosuite.learning.supermimic.state import StateSpec, forward_axis_from_quat
from myosuite.learning.supermimic.world_model import WorldModel, WorldModelCfg


@dataclass
class SuperMimicCfg:
    """Training settings (paper Table A.5 defaults where they carry over)."""

    clip: str = ""
    out: str = "runs/supermimic"
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
    # Barrier on raw actions beyond |a - 0.5| > band, where the sigmoid control saturates
    # and BPTT gradients vanish (0 disables it).
    saturation_penalty: float = 0.0
    saturation_band: float = 1.0
    actor_hidden: tuple[int, ...] = (256, 256, 128)
    max_hours: float = 12.0
    max_sim_steps: float = 2e9
    eval_every: int = 500
    eval_envs: int = 128
    eval_steps: int = 500
    save_every: int = 1000
    compile: bool = True
    init_actor: str = ""
    init_world_model: str = ""
    resume: str = ""
    seed: int = 0
    device: str = "cuda:0"
    wm: WorldModelCfg = field(default_factory=WorldModelCfg)


def _sigmoid_ctrl(a: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(5.0 * (a - 0.5))


class SuperMimicTrainer:
    """World model + BPTT actor-critic on ``myoMimicFullbody-v0``.

    Args:
        cfg: Training settings.
    """

    def __init__(self, cfg: SuperMimicCfg) -> None:
        self.cfg = cfg
        dev = cfg.device
        torch.manual_seed(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        self.out = Path(cfg.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.env = make_mimic_env(cfg.clip, cfg.num_envs, dev, seed=cfg.seed)
        self.env.reset()
        self.io = MimicEnvIO(self.env)
        self.heads = MimicHeads.from_env(self.env, "mimic_fullbody_robot")
        model = _musclemimic_cpu_model("fullbody")
        axis = forward_axis_from_quat(self.heads.clip.qpos[0, 3:7].cpu())
        self.spec = StateSpec(
            model.nq, model.nv, model.na, self.heads.ctrl_dt, axis
        ).to(dev)
        self.wm = WorldModel(
            model, self.spec, self.env.cfg.decimation, self.env.physics_dt, cfg.wm
        ).to(dev)
        obs_dim = self.env.obs_buf["actor"].shape[-1]
        self.actor = Actor(obs_dim, model.nu, cfg.actor_hidden).to(dev)
        self.critic = Critic(obs_dim, 1.0 / (1.0 - cfg.gamma)).to(dev)
        if cfg.init_actor:
            self.actor = Actor.from_rsl_rl(cfg.init_actor, dev)
        if cfg.init_world_model:
            self.wm.load_state_dict(
                torch.load(cfg.init_world_model, map_location=dev)["wm"]
            )
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.lr)
        self.opt_actor = torch.optim.Adam(
            self.actor.mlp.parameters(), lr=cfg.lr, betas=(0.9, 0.95)
        )
        self.opt_critic = torch.optim.Adam(self.critic.mlp.parameters(), lr=cfg.lr)
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
        self._step = torch.compile(self._imag_step) if cfg.compile else self._imag_step
        self._obs = torch.compile(self._observe) if cfg.compile else self._observe

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
        self, s: torch.Tensor, f: torch.Tensor, obs: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Actor + world model + analytic heads for one imagined step (compiled as one graph)."""
        h = self.heads
        a = self.actor(obs)
        sat = torch.relu((a - 0.5).abs() - self.cfg.saturation_band).square().mean(-1)
        s = self.wm.step(s, _sigmoid_ctrl(a))
        f = f + 1
        qpos, qvel, act = self.spec.split(s)
        sites = h.kin.site_pos(qpos)
        r = h.reward(qpos, qvel, f, sites)
        d = h.termination_prob(qpos, f, sites)
        return s, f, r, d, sat, h.observation(qpos, qvel, act, f, sites)

    def imagine(self) -> dict[str, torch.Tensor]:
        """Differentiable H-step rollout of the actor through the frozen world model."""
        cfg, h = self.cfg, self.heads
        s, f = self.buffer.sample_starts(cfg.imag_batch)
        ended = h.clip_end(f)
        obs_l, rew_l, dhat_l, live_l, end_l, sat_l = [], [], [], [], [], []
        obs = self._obs(s, f)
        for _ in range(cfg.horizon):
            obs_l.append(obs)
            live_l.append((~ended).float())
            s, f, r, d, sat, obs = self._step(s, f, obs)
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
        }

    def _set_lr(self) -> None:
        cfg = self.cfg
        frac = min(self.updates / max(cfg.lr_decay_updates, 1), 1.0)
        scale = cfg.lr_final_frac + (1 - cfg.lr_final_frac) * 0.5 * (
            1 + math.cos(math.pi * frac)
        )
        for opt in (self.opt_wm, self.opt_actor, self.opt_critic):
            for group in opt.param_groups:
                group["lr"] = cfg.lr * scale

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Deterministic actor from RSI starts: mean reward per step, terminations, sim/WM gap."""
        cfg = self.cfg
        if self._eval_env is None:
            self._eval_env = make_mimic_env(
                cfg.clip, cfg.eval_envs, cfg.device, seed=cfg.seed + 1000
            )
            self._eval_env.reset()
        return evaluate_policy(self._eval_env, self.actor, cfg.eval_steps)

    # ------------------------------------------------------------------ checkpoints
    def save(self, name: str = "latest") -> None:
        """Actor (rsl_rl layout), critic, world model, optimisers and counters."""
        torch.save(
            {
                "actor_state_dict": self.actor.rsl_rl_state_dict(),
                "critic": self.critic.state_dict(),
                "wm": self.wm.state_dict(),
                "opt_wm": self.opt_wm.state_dict(),
                "opt_actor": self.opt_actor.state_dict(),
                "opt_critic": self.opt_critic.state_dict(),
                "updates": self.updates,
                "sim_steps": self.sim_steps,
                "elapsed": self.elapsed,
                "best_eval": self.best_eval,
                "cfg": asdict(self.cfg),
            },
            self.out / f"ckpt_{name}.pt",
        )

    def load(self, path: str) -> None:
        """Resume from :meth:`save` output (the replay buffer refills during warm-up)."""
        ck = torch.load(path, map_location=self.cfg.device, weights_only=False)
        self.actor.load_state_dict(
            {
                ("distribution_std" if k == "distribution.std_param" else k): v
                for k, v in ck["actor_state_dict"].items()
            }
        )
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

    # ------------------------------------------------------------------ main loop
    def train(self) -> None:
        """Run until ``max_hours`` of training time or ``max_sim_steps``."""
        cfg = self.cfg
        log = (self.out / "log.jsonl").open("a")
        for _ in range(cfg.warmup_steps):
            self.collect(max(cfg.ctrl_noise, 0.3))
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
                if self.updates % 50 == 0:
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
                        # Keep the best-eval policy: runs can decay after their peak.
                        if rec["eval_reward_per_step"] > self.best_eval:
                            self.best_eval = rec["eval_reward_per_step"]
                            self.save("best")
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    self.done_stats = self.done_stats[-500:]
                    t_sim = t_upd = 0.0
                if self.updates % cfg.save_every == 0:
                    self.save()
            self.elapsed = time.time() - start
            if (
                self.elapsed > cfg.max_hours * 3600
                or self.sim_steps >= cfg.max_sim_steps
            ):
                break
        self.save()
        rec = {
            "updates": self.updates,
            "sim_steps": self.sim_steps,
            "hours": self.elapsed / 3600,
            "final": True,
            **{f"eval_{k}": v for k, v in self.evaluate().items()},
        }
        log.write(json.dumps(rec) + "\n")
        log.close()


@torch.no_grad()
def evaluate_policy(env: Any, actor: Any, steps: int) -> dict[str, float]:
    """Mean env reward per step, early terminations per 1000 steps, episode length and muscle control of a deterministic actor.

    Starts from the env's RSI reset; episodes that end are reset by the env and continue.
    """
    env.reset()
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
    n = env.num_envs
    return {
        "reward_per_step": rew_sum / steps,
        "terminations_per_1k_steps": 1000.0 * terms / (n * steps),
        "mean_episode_len": n * steps / max(ends, 1),
        "mean_ctrl": ctrl_sum / steps,
    }
