# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# training/supermimic/ for the notices of the code it includes.
"""SuperMimic runner (paper Algorithm A.1): collect -> fit world model -> BPTT policy + critic update.

Port of supermimic_vibed's ``train.py`` onto an mjlab ``ManagerBasedRlEnv``, with mjlab's runner
interface (``learn`` / ``save`` / ``load`` / ``get_inference_policy``) so mjlab's ``train`` and ``play``
CLIs drive it. The env's reward, termination and observation terms are reproduced on imagined
states through their analytic implementations (``term.func.analytic``), with the term weights and
params of the env cfg, so simulator and world model share one MDP definition.

Imagined rollouts evolve the muscle activations with the exact activation dynamics and the body
state with the learned world model, whose action input is the substep-mean activation.

Changed for MyoSuite: no DEP exploration and no evaluation-guided resets (both off in the fast recipe), the CUDA-graph
and stream options turn off on a CPU device, ``finetune`` restarts a trained baseline's clock and schedule, and
checkpoints store the env settings the policy depends on (control rate, goal stride, observation sizes).

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at commit
8a6244b, ``src/muscle_supermimic/rl/supermimic/runner.py``.
"""

from __future__ import annotations

import copy
import csv
import gc
import math
import os
import time
from dataclasses import asdict, fields

import torch
import torch.nn as nn
import torch.utils.checkpoint
from tensordict import TensorDict

from supermimic import networks
from supermimic.buffer import ReplayBuffer
from supermimic.collector import STATS, Collector
from supermimic.config import SuperMimicRunnerCfg
from supermimic.networks import (
    ActionMap,
    Critic,
    Policy,
    RunningNorm,
    WorldModel,
)
from supermimic.optim import MuonAdam, gated_step
from supermimic.heads import (
    NUM_TOUCH,
    _goal,
    StepCtx,
    analytic,
    flatten_body,
    unflatten_body,
)
from supermimic.mirror import Mirror
from supermimic.task import TaskView
from myosuite.envs.myo.backends.mjlab.tasks.mdp import SYNC_TERM


class AnalyticObsGroup:
    """An observation group of the env, evaluated from states: same terms, order, history and scale."""

    def __init__(self, env, group: str, heads):
        om = env.observation_manager
        self.heads, self.terms = heads, []
        for name, cfg in zip(
            om._group_obs_term_names[group], om._group_obs_term_cfgs[group], strict=True
        ):
            fn = analytic(cfg.func)
            if (
                (cfg.noise is not None and om.cfg[group].enable_corruption)
                or cfg.clip
                or cfg.delay_max_lag > 0
            ):
                raise NotImplementedError(
                    f"observation term {group}/{name}: noise/clip/delay are not reproduced in imagination"
                )
            params = {k: v for k, v in cfg.params.items() if k != "command_name"}
            self.terms.append((fn, params, cfg.history_length, cfg.scale))
        self.dim = om.group_obs_dim[group][0]

    def _value(self, fn, params, scale, ctx):
        x = fn(self.heads, ctx, **params)
        return x * scale if scale is not None else x

    def init_history(self, ctx_hist: StepCtx) -> list:
        """Histories from states with leading dims (B, H), oldest first."""
        return [
            self._value(fn, p, s, ctx_hist)[:, -h:] if h > 0 else None
            for fn, p, h, s in self.terms
        ]

    def push(self, hist: list, ctx: StepCtx) -> list:
        return [None if x is None else torch.cat((x[:, 1:], self._value(fn, p, s, ctx).unsqueeze(1)), 1)
                for x, (fn, p, h, s) in zip(hist, self.terms)]  # fmt: skip

    def assemble(self, hist: list | None, ctx: StepCtx) -> torch.Tensor:
        out = []
        for i, (fn, p, h, s) in enumerate(self.terms):
            out.append(hist[i].flatten(1) if h > 0 else self._value(fn, p, s, ctx))
        return torch.cat(out, -1)


class ClipRowGrad(torch.autograd.Function):
    """Identity on a group of (B, ...) tensors. Backward rescales each row's gradient, taken over the whole group and
    multiplied by ``scale``, to norm <= ``max_norm``, and appends the fraction of clipped rows to ``stats``."""

    @staticmethod
    def forward(ctx, max_norm: float, scale: float, stats: list, *xs):
        ctx.max_norm, ctx.scale, ctx.stats = max_norm, scale, stats
        return tuple(x.view_as(x) for x in xs)

    @staticmethod
    def backward(ctx, *gs):
        norm = ctx.scale * sum(g.flatten(1).square().sum(-1) for g in gs).sqrt()
        k = (ctx.max_norm / norm.clamp(min=1e-30)).clamp(max=1.0)
        ctx.stats.append((k < 1).float().mean())
        return (None, None, None) + tuple(
            g * k.view(-1, *(1,) * (g.dim() - 1)) for g in gs
        )


class Graphed:
    """Run ``fn`` eagerly until ``ready()``; then eagerly on a side stream for ``warmup`` calls, capture it as a CUDA
    graph on the next call and replay it from then on (BlinkMimic). Each call is one update; the outputs are cloned
    out of the graph's pool. ``own``: captured on its own stream (its own cuBLAS workspace) with its own random state,
    so that it can replay concurrently with other graphs (overlap_wm)."""

    def __init__(self, fn, ready, warmup: int = 3, own: bool = False):
        self.fn, self.ready, self.warmup, self.calls, self.graph, self.out = (
            fn,
            ready,
            warmup,
            0,
            None,
            None,
        )
        # one warm-up stream for all warm-up calls: the allocator caches freed blocks per stream, so a new stream per
        # call would keep a copy of every call's activations (without remat_imagination, GBs each) until the capture
        self.side = None
        self.own, self.capture_stream, self.rng = own, None, None
        if own:
            self.capture_stream = torch.cuda.Stream()
            self.rng = torch.cuda.default_generators[
                torch.cuda.current_device()
            ].clone_state()
            self.rng.manual_seed(int(torch.randint(0, 2**62, ())))

    def __call__(self, record=None):
        """``record``: a stream the outputs are also used on (when called on another one)."""
        if self.graph is None and not self.ready():
            return self.fn()
        self.calls += 1
        if self.calls <= self.warmup:
            if self.side is None:
                self.side = torch.cuda.Stream()
            self.side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.side):
                out = self.fn()
            torch.cuda.current_stream().wait_stream(self.side)
            return out
        if self.graph is None:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # return the eager phase's cached blocks before the graph takes its own pool
            t0 = time.time()
            self.graph = torch.cuda.CUDAGraph()
            # autograd nodes of the eager calls (the first ones run on the default stream) keep a reference to the
            # stream they ran on, which the capture refuses for the default stream (seen with remat_imagination=False
            # and compile): the device is synchronized here, so redirecting them to the capturing stream is safe
            override = getattr(
                torch.autograd.graph, "set_override_stale_capture_stream", None
            )
            if override is not None:
                override(True)
            gen = orig = None
            if self.rng is not None:
                gen = torch.cuda.default_generators[torch.cuda.current_device()]
                orig = gen.graphsafe_get_state()
                gen.graphsafe_set_state(self.rng)
            try:
                with torch.cuda.graph(self.graph, stream=self.capture_stream):
                    self.out = self.fn()
            finally:
                if override is not None:
                    override(False)
                if gen is not None:
                    gen.graphsafe_set_state(orig)
            torch.cuda.synchronize()
            print(f"cuda_graph: captured {self.fn.__name__} in {time.time() - t0:.1f} s, "
                  f"{torch.cuda.memory_reserved() / 2**30:.1f} GB reserved", flush=True)  # fmt: skip
        self.graph.replay()
        out = tuple(x.clone() for x in self.out)
        if record is not None:
            for x in out:
                x.record_stream(record)
        return out


def set_lr(opt, value: float):
    """The learning rate of every parameter group: filled in place where it is a device tensor (cuda_graph)."""
    for g in opt.param_groups:
        if torch.is_tensor(g["lr"]):
            g["lr"].fill_(value)
        else:
            g["lr"] = value


class SuperMimicRunner:
    """SuperMimic training on an mjlab tracking env, with mjlab's runner interface.

    Args:
        env: The ``RslRlVecEnvWrapper`` of a ``ManagerBasedRlEnv`` of this package's tracking task.
        train_cfg: ``SuperMimicRunnerCfg`` fields as a dict (mjlab's train script passes ``asdict`` of it).
        log_dir: Where checkpoints and ``train_log.csv`` go (None: nothing is written).
        device: The torch device of the env.
        **kwargs: Ignored (mjlab passes tracking-task extras).
    """

    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        names = {f.name for f in fields(SuperMimicRunnerCfg)}
        self.cfg = cfg = SuperMimicRunnerCfg(
            **{k: v for k, v in train_cfg.items() if k in names}
        )
        if not str(device).startswith("cuda") and (
            cfg.cuda_graph or cfg.overlap_collect or cfg.overlap_wm
        ):
            # CUDA graphs and streams need a GPU; the collector's wrapper runs eagerly on a CPU
            cfg.cuda_graph = cfg.overlap_collect = cfg.overlap_wm = False
        self.env, self.device, self.log_dir = env, device, log_dir
        self.uenv = u = env.unwrapped
        self.cmd = TaskView(u, cfg.command_name)
        self.heads, self.muscle, self.motion = (
            self.cmd.heads,
            self.cmd.muscle,
            self.cmd.motion,
        )
        self.nq, self.nm = self.heads.nq, self.cmd.info.num_muscles
        self.cmd.kin.unbind = cfg.kin_unbind
        self.obs_policy = AnalyticObsGroup(u, cfg.obs_groups["policy"][0], self.heads)
        self.obs_goal = AnalyticObsGroup(u, cfg.obs_groups["goal"][0], self.heads)
        assert not cfg.hoist_goal or all(
            fn is _goal and h == 0 for fn, _, h, _ in self.obs_goal.terms
        ), "hoist_goal: the goal group must depend on the reference frame only"
        self.hist = max(h for _, _, h, _ in self.obs_policy.terms)
        assert all(
            h in (0, self.hist) for _, _, h, _ in self.obs_policy.terms
        ), "mixed history lengths"
        rm, scale = (
            u.reward_manager,
            (u.step_dt if u.reward_manager._scale_by_dt else 1.0),
        )
        self.reward_terms = []
        for name, rc in zip(rm._term_names, rm._term_cfgs, strict=True):
            fn = analytic(rc.func)
            self.reward_terms.append(
                (
                    fn,
                    rc.weight * scale,
                    {k: v for k, v in rc.params.items() if k != "command_name"},
                )
            )
        tm = u.termination_manager
        self.fail_terms = []
        for name, tc in zip(tm._term_names, tm._term_cfgs, strict=True):
            if tc.time_out or name == SYNC_TERM:
                continue
            params = {k: v for k, v in tc.params.items() if k != "command_name"}
            self.fail_terms.append(
                (analytic(tc.func), {**params, "soft_width": cfg.soft_width})
            )
        self.max_step_reward = self.cmd.max_step_reward
        self.reward_scale, self.reward_names = scale, list(rm._term_names)

        networks.AMP_DTYPE = (
            torch.bfloat16
            if cfg.amp == "bf16" and str(device).startswith("cuda")
            else None
        )
        self._build_networks()
        self.it, self.wall_offset, self.sim_steps = 0, 0.0, 0
        self._decay_at = None  # lr_decay_ret: the iteration the policy learning-rate decay was triggered at
        self.mirror = (
            None  # wm_mirror: the model's left-right maps (validated on the model here)
        )
        if cfg.wm_mirror > 0:
            self.mirror = Mirror(
                self.cmd.kin,
                self.cmd.geo,
                tuple(self.cmd.touch_sensor.cfg.primary.pattern),
                device,
            )
            print(
                "wm_mirror: "
                + ", ".join(f"{k} {v:.1e}" for k, v in self.mirror.residuals.items()),
                flush=True,
            )
        self.collector = None  # collect_graph: the thin simulator wrapper (collector.py), built by learn()
        # per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
        f = lambda n, v: torch.full((n,), v)  # noqa: E731
        self.err_floor = torch.cat(
            (
                f(3, 3e-3),
                f(3, 5e-3),
                f(3, 2e-2),
                f(3, 5e-2),
                f(self.nq, 3e-3),
                f(self.nq, 5e-2),
            )
        ).to(device)
        comp = cfg.compile and str(device).startswith("cuda")
        c = torch.compile if comp else (lambda fn: fn)
        self.wm_rollout, self.wm_err = c(self._wm_rollout), c(self._wm_err)
        self.imag_step_z, self.encode_goals = (
            c(self._imag_step_z),
            c(self._encode_goals),
        )
        self.imag_step, self.reward_term, self.critic_input = (
            c(self._imag_step),
            c(self._reward_term),
            c(self._critic_input),
        )

    NET_FIELDS = (
        "wm_hidden",
        "enc_hidden",
        "dec_hidden",
        "latent_dim",
        "critic_hidden",
        "wm_structured",
    )

    def _build_networks(self):
        c, dev = self.cfg, self.device
        self.wm = WorldModel(self.nq, self.nm, NUM_TOUCH, c.wm_hidden, structured=c.wm_structured, dt=self.uenv.step_dt,
                             substeps=self.uenv.cfg.decimation).to(dev)  # fmt: skip
        self.pi = Policy(self.obs_policy.dim, self.obs_goal.dim, self.nm, c.enc_hidden, c.dec_hidden, c.latent_dim,
                         c.init_excitation, action_map=ActionMap(self.nm)).to(dev)  # fmt: skip
        # overlap_wm: imagination steps a copy of the world model taken at the start of each iteration (sync_wm_imag),
        # so that the world-model updates can run concurrently with the policy update
        self.wm_imag = (
            copy.deepcopy(self.wm).requires_grad_(False) if c.overlap_wm else self.wm
        )
        self._wm_sync = None
        self.critic = Critic(
            self.obs_policy.dim + self.obs_goal.dim + self.heads.critic_extras_dim(),
            c.critic_hidden,
        ).to(dev)
        self.critic_targ = self.critic
        if c.critic_target_alpha > 0:
            self.critic_targ = copy.deepcopy(self.critic).requires_grad_(False)
            self.critic_targ.norm = (
                self.critic.norm
            )  # one input normalizer, updated with the online critic
        assert (
            not c.no_sync or c.fused_adam
        ), "no_sync gates the policy step on the device: needs fused_adam"
        graph = c.cuda_graph
        assert not graph or (
            c.no_sync
            and c.fused_adam
            and not c.empty_cache
            and str(dev).startswith("cuda")
        ), "cuda_graph needs no_sync, fused_adam, empty_cache off and a CUDA device"
        for net in (self.wm, self.pi, self.critic):
            for m in net.modules():
                if isinstance(m, RunningNorm):
                    m.no_sync = c.no_sync

        def adam(params, **kw):
            if graph:  # the learning rate a device tensor (filled every iteration), the step count on the GPU
                return torch.optim.Adam(
                    params,
                    lr=torch.tensor(c.lr, device=dev),
                    fused=True,
                    capturable=True,
                    **kw,
                )
            return torch.optim.Adam(
                params, lr=c.lr, **({"fused": True} if c.fused_adam else {}), **kw
            )

        if c.wm_opt == "muon":
            self.opt_wm = MuonAdam(self.wm, [self.wm.net], lr=c.lr, betas=(0.9, 0.999), fused=c.fused_adam, capturable=graph,
                                   momentum=c.muon_momentum, batched=c.muon_batched)  # fmt: skip
        else:
            self.opt_wm = adam(self.wm.parameters())
        if c.pi_opt == "muon":
            self.opt_pi = MuonAdam(self.pi, [self.pi.enc, self.pi.dec], lr=c.lr, betas=(c.pi_beta1, 0.95), fused=c.fused_adam,
                                   capturable=graph, momentum=c.muon_momentum, batched=c.muon_batched)  # fmt: skip
        else:
            self.opt_pi = adam(self.pi.parameters(), betas=(c.pi_beta1, 0.95))
        self.opt_v = adam(self.critic.parameters())
        # pi_ema: Polyak average of the policy (parameters and normalizer buffers), the deployed policy
        self.pi_avg, self._ema_n = None, 0
        if (
            c.pi_ema > 0
        ):  # device=dev: the update counter on the GPU too, so a captured EMA step advances it on replay
            self.pi_avg = torch.optim.swa_utils.AveragedModel(
                self.pi,
                device=dev,
                multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(c.pi_ema),
                use_buffers=True,
            )
        self.nonfinite = torch.zeros(
            (), dtype=torch.long, device=dev
        )  # no_sync: skipped non-finite policy updates

    @property
    def deployed(self) -> Policy:
        """The policy that is evaluated, played and checkpointed as "pi": the EMA average with pi_ema, else the live one."""
        return self.pi_avg.module if self.pi_avg is not None else self.pi

    @torch.no_grad()
    def ema_update(self):
        """One EMA step of the deployed policy after a policy update: the first call copies the live policy (as
        AveragedModel does), later ones take AveragedModel's lerp without its host-side first-call test."""
        if self.pi_avg is None:
            return
        if self._ema_n == 0:
            self.pi_avg.update_parameters(self.pi)
        else:
            avg = list(self.pi_avg.module.parameters()) + list(
                self.pi_avg.module.buffers()
            )
            live = [
                t.detach() for t in list(self.pi.parameters()) + list(self.pi.buffers())
            ]
            torch._foreach_lerp_(avg, live, 1 - self.cfg.pi_ema)
            self.pi_avg.n_averaged += 1
        self._ema_n += 1

    # ------------------------------------------------------------------ analytic pieces
    def _wm_rollout(self, s, act, a):
        S, act = unflatten_body(s[:, 0], self.nq), act[:, 0]
        preds, touch = [], []
        for j in range(self.cfg.wm_window):
            act1, _ = self.muscle.step(act, a[:, j])
            S, touch_n = self.wm.step(S, act, act1)
            act = act1
            preds.append(S), touch.append(touch_n)
        return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6)) + (
            torch.stack(touch, 1),
        )  # (B, k, ...)

    def _wm_err(self, S_hat, s1, touch1, mask):
        std = torch.maximum(self.wm.inc_norm.std(), self.err_floor)
        e = (
            self.wm.state_error(S_hat[:6], unflatten_body(s1, self.nq)) / std
        ) ** 2  # (B, k, D)
        e = e.mean(-1)
        e_touch = ((S_hat[6] - self.wm.touch_norm(touch1)) ** 2).mean(-1)
        loss = (e * mask).sum() / s1.shape[0] + self.cfg.touch_loss_weight * (
            e_touch * mask
        ).sum() / s1.shape[0]
        return loss, e[:, -1], (e_touch * mask).sum() / mask.sum().clamp(min=1)

    def _imag_step(self, S, hist, f, a_last):
        """One imagined step from S. Also returns the critic input of S (detached), for the critic's targets: it is
        made of the observation the policy reads here (the observation has no history), so every imagined state's
        muscle geometry is evaluated once."""
        ctx = StepCtx(S, f, a_last)
        obs, goal = (
            self.obs_policy.assemble(hist, ctx),
            self.obs_goal.assemble(None, ctx),
        )
        a = self.pi(obs, goal)
        x = torch.cat((obs, goal, self.heads.critic_extras(ctx)), -1).detach()
        act1, _ = self.muscle.step(S[6], a)
        body1, touch_n = self.wm_imag.step(S, S[6], act1)
        S1 = body1 + (act1, self.wm_imag.touch(touch_n))
        f1 = f + 1
        return S1, a, self.obs_policy.push(hist, StepCtx(S1, f1, a)), f1, x

    def _encode_goals(self, f, H: int):
        """hoist_goal: the goals of frames f .. f + H - 1 and their latents, (H, B, ...) each."""
        F = (
            f.unsqueeze(0)
            + torch.arange(H, device=f.device, dtype=f.dtype).unsqueeze(1)
        ).flatten()
        G = self.obs_goal.assemble(None, StepCtx(None, F))
        return G.view(H, f.shape[0], -1), self.pi.encode(G).view(H, f.shape[0], -1)

    def _imag_step_z(self, S, hist, f, a_last, z, goal):
        """_imag_step with the step's goal and latent given (hoist_goal)."""
        ctx = StepCtx(S, f, a_last)
        obs = self.obs_policy.assemble(hist, ctx)
        a = self.pi.decode(z, obs)
        x = torch.cat((obs, goal, self.heads.critic_extras(ctx)), -1).detach()
        act1, _ = self.muscle.step(S[6], a)
        body1, touch_n = self.wm_imag.step(S, S[6], act1)
        S1 = body1 + (act1, self.wm_imag.touch(touch_n))
        f1 = f + 1
        return S1, a, self.obs_policy.push(hist, StepCtx(S1, f1, a)), f1, x

    def _reward_term(self, S, f, a, a_prev):
        ctx = StepCtx(S, f, a, a_prev)
        r = sum(w * fn(self.heads, ctx, **p) for fn, w, p in self.reward_terms)
        alive = torch.ones_like(r)
        for fn, p in self.fail_terms:
            alive = alive * (1 - fn(self.heads, ctx, **p))
        return r, 1 - alive

    def _critic_input(self, S, f, a):
        """The critic's input: the decoder observation, the goal and privileged reference errors."""
        ctx = StepCtx(S, f, a)
        return torch.cat(
            (
                self.obs_policy.assemble(None, ctx),
                self.obs_goal.assemble(None, ctx),
                self.heads.critic_extras(ctx),
            ),
            -1,
        )

    def value(self, x):
        """Bootstrap value: the target critic, optionally clamped to the attainable maximum."""
        v = self.critic_targ(x)
        return (
            v.clamp(max=self.max_step_reward / (1 - self.cfg.gamma))
            if self.cfg.value_clip
            else v
        )

    @torch.no_grad()
    def update_critic_targ(self):
        if self.critic_targ is not self.critic:
            for pt, p in zip(
                self.critic_targ.parameters(), self.critic.parameters(), strict=True
            ):
                pt.lerp_(p, 1 - self.cfg.critic_target_alpha)

    def lr_at(self, it: int) -> float:
        c = self.cfg
        if it >= c.lr_decay_iters:
            return c.lr * 0.03
        return c.lr * (
            0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / c.lr_decay_iters))
        )

    # ------------------------------------------------------------------ updates
    def _mirror_windows(self, w: dict, n: int) -> dict:
        """The first n windows replaced by their left-right mirror images (body states, activations, excitations,
        touch); the episode-end mask is the same for a window and its image."""
        out = dict(w)
        mb = lambda x: flatten_body(self.mirror.body(unflatten_body(x, self.nq)))  # noqa: E731
        for k, fn in (
            ("s", mb),
            ("s1", mb),
            ("act", self.mirror.muscles),
            ("a", self.mirror.muscles),
            ("touch1", self.mirror.touch),
        ):
            out[k] = torch.cat((fn(w[k][:n]), w[k][n:]))
        return out

    def wm_update(self):
        c = self.cfg
        w = self.buffer.sample_windows(c.wm_batch, c.wm_window)
        if self.mirror is not None:
            w = self._mirror_windows(w, int(c.wm_batch * c.wm_mirror))
        S_hat = self.wm_rollout(w["s"], w["act"], w["a"])
        loss, e_last, e_touch = self.wm_err(S_hat, w["s1"], w["touch1"], w["mask"])
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), c.grad_clip)
        self.opt_wm.step()
        m = w["mask"][:, -1]
        return (
            loss.detach() / c.wm_window,
            (e_last.detach() * m).sum() / m.sum().clamp(min=1),
            e_touch.detach(),
        )

    def sample_valid_starts(self, B: int):
        starts, hists, n = [], [], 0
        m, H = self.motion, self.cfg.horizon
        if self.cfg.no_sync:
            # fixed-size selection without reading the valid count on the host: stable-sort the valid candidates first
            # (only starts within H frames of a clip end are invalid: a few % of the KIT clips' frames)
            s, h = self.buffer.sample_starts(2 * B, self.hist)
            f = s["f"]
            ok = f + H <= m.clip_end[m.clip_id[f]]
            idx = torch.sort((~ok).to(torch.uint8), stable=True).indices[:B]
            return {k: v[idx] for k, v in s.items()}, {k: v[idx] for k, v in h.items()}
        while n < B:
            s, h = self.buffer.sample_starts(2 * B, self.hist)
            f = s["f"]
            idx = (
                (f + H <= m.clip_end[m.clip_id[f]])
                .nonzero(as_tuple=False)
                .squeeze(-1)[: B - n]
            )
            (
                starts.append({k: v[idx] for k, v in s.items()}),
                hists.append({k: v[idx] for k, v in h.items()}),
            )
            n += len(idx)
        cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
        return cat(starts), cat(hists)

    def policy_update(self):
        c, dev, HZ, B = self.cfg, self.device, self.cfg.horizon, self.cfg.pi_batch
        s, h = self.sample_valid_starts(B)
        S = unflatten_body(s["s"], self.nq) + (s["act"], s["touch"])
        f, a_last = s["f"], s["last_a"]
        with torch.no_grad():
            S_h = unflatten_body(h["s"], self.nq) + (h["act"], h["touch"])
            hist = self.obs_policy.init_history(StepCtx(S_h, h["f"], h["last_a"]))
        if (
            self.wm_imag is self.wm
        ):  # (overlap_wm: the copy never needs gradients, and the original trains concurrently)
            set_requires_grad(self.wm, False)
        set_requires_grad(self.critic, False)
        S0 = S
        Ss, As, Aps, Fs = [], [], [], []
        if c.use_critic:  # critic inputs of S_0 .. S_H, written in place (MS-Human-700: ~2 GB at 4096 x 33 states)
            X = torch.empty(HZ + 1, B, self.critic.norm.mean.shape[0], device=dev)
        step = self.imag_step
        if c.remat_imagination:  # recompute each step in the backward pass instead of keeping its activations

            def step(*x):
                return torch.utils.checkpoint.checkpoint(
                    self.imag_step, *x, use_reentrant=False
                )  # noqa: E731

        if c.hoist_goal:
            assert not c.remat_imagination, "hoist_goal: no rematerialization"
            G, Z = self.encode_goals(f, HZ)
            G, Z = G.unbind(0), Z.unbind(0)
        clipped = []  # per imagined step (filled by the backward pass): fraction of trajectories clipped
        for t in range(HZ):
            Aps.append(a_last)
            if c.hoist_goal:
                S, a_last, hist, f, x = self.imag_step_z(S, hist, f, a_last, Z[t], G[t])
            else:
                S, a_last, hist, f, x = step(S, hist, f, a_last)
            if c.state_grad_clip > 0:
                live = [i for i, xh in enumerate(hist) if xh is not None]
                out = ClipRowGrad.apply(
                    c.state_grad_clip, B, clipped, *S, a_last, *(hist[i] for i in live)
                )
                S, a_last, hist = out[: len(S)], out[len(S)], list(hist)
                for i, xh in zip(live, out[len(S) + 1 :], strict=True):
                    hist[i] = xh
            Ss.append(S), As.append(a_last), Fs.append(f)
            if c.use_critic:
                X[t] = x
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
        S_all = tuple(flat([St[i] for St in Ss]) for i in range(len(S0)))
        r, d = self.reward_term(S_all, flat(Fs), flat(As), flat(Aps))
        r, d = r.view(HZ, B), d.view(HZ, B)
        if c.cuda_graph:  # the survival products as a loop: cumprod's backward tests for zeros on the host
            acc, cs = torch.ones_like(d[0]), []
            for t in range(HZ):
                acc = acc * (1 - d[t])
                cs.append(acc)
            alive = torch.stack(cs)
        else:
            alive = torch.cumprod(1 - d, 0)  # c_{t+1}
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        disc = c.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(
            -1
        )
        J = (disc * c_t * r).sum(0)
        surv = alive[-1]
        r_sum = (c_t * r).detach().sum(0)
        rews, dones = r.detach(), d.detach()
        if c.use_critic:
            x_last = self.critic_input(S, f, a_last)
            X[HZ] = x_last.detach()
            J = J + (c.gamma**HZ) * surv * self.value(x_last)
        loss = -J.mean()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(self.pi.parameters(), c.grad_clip)
        nn.utils.clip_grad_value_(self.pi.parameters(), c.policy_grad_clip_value)
        if c.no_sync:
            # skip a non-finite update without a host read: zero the gradients on the device and gate the optimizer step
            # (parameters and optimizer state unchanged); counted in the log column nonfinite
            finite = torch.isfinite(gn)
            self.nonfinite.add_((~finite).long())
            for p in self.pi.parameters():
                if p.grad is not None:
                    p.grad.copy_(torch.where(finite, p.grad, 0.0))
            gated_step(self.opt_pi, finite)
        elif torch.isfinite(gn):
            self.opt_pi.step()
        self.ema_update()
        clip_frac = (
            torch.stack(clipped).mean() if clipped else torch.zeros((), device=dev)
        )
        if self.wm_imag is self.wm:
            set_requires_grad(self.wm, True)
        set_requires_grad(self.critic, True)
        # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
        v_loss = torch.zeros((), device=dev)
        if c.use_critic:
            # one imagined step at a time: a normalized copy of all (H+1) x B inputs is as large as X itself
            for x in X:
                self.critic.norm.update(x)
            with torch.no_grad():
                V = torch.stack([self.value(x) for x in X])
                R = V[-1]
                targets = []
                for t in range(HZ - 1, -1, -1):
                    R = rews[t] + c.gamma * (1 - dones[t]) * (
                        (1 - c.lam) * V[t + 1] + c.lam * R
                    )
                    targets.append(R)
                targets = torch.stack(targets[::-1])  # (H, B)
            Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
            n = Xf.shape[0]
            mb = n // c.critic_minibatches
            for _ in range(c.critic_epochs):
                perm = torch.randperm(n, device=dev)
                for i in range(c.critic_minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    v_loss = ((self.critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                    self.opt_v.zero_grad(set_to_none=True)
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), c.grad_clip)
                    self.opt_v.step()
            self.update_critic_targ()
        return (loss.detach(), v_loss.detach(), (r_sum / HZ / self.max_step_reward).mean(), gn.detach(), surv.mean().detach(),
                clip_frac)  # fmt: skip

    # ------------------------------------------------------------------ simulator
    @torch.no_grad()
    def init_collection(self):
        """A new replay buffer and episode returns, the env's observation, and with collect_graph the collector (it
        writes at the buffer's device-side write head)."""
        c, u = self.cfg, self.uenv
        self.buffer = ReplayBuffer(
            c.buffer_steps,
            u.num_envs,
            self.heads.body_state_dim,
            self.nm,
            NUM_TOUCH,
            self.device,
        )
        self.ep_ret = torch.zeros(u.num_envs, device=self.device)
        self.obs = self.env.get_observations()
        self.collector = None
        if c.collect_graph:
            self.buffer.use_device_ptr()
            self.collector = Collector(self, overlap=c.overlap_collect)
        if c.overlap_collect:
            assert (
                c.collect_graph and c.cuda_graph
            ), "overlap_collect overlaps the collection graph with the update graphs"
            self.buffer.use_sample_snapshot(
                reserve=math.ceil(1.0 / c.utd)
            )  # the most steps one iteration collects
            self.collect_stream = torch.cuda.Stream(device=self.device)

    @torch.no_grad()
    def collect(self, n_steps: int, stats: dict, record=None):
        """n_steps simulator steps into the buffer; their statistics appended to ``stats`` (``record``: a stream the
        statistics are also used on, when collecting on another one)."""
        u, cmd, N = self.uenv, self.cmd, self.uenv.num_envs
        if (
            self.collector is not None
        ):  # collect_graph: the thin simulator wrapper, one CUDA graph per step
            for _ in range(n_steps):
                out = self.collector.step()
                self.sim_steps += N
                for k in STATS:
                    v = out[k].clone()
                    if record is not None:
                        v.record_stream(record)
                    stats[k].append(v)
            return
        for _ in range(n_steps):
            S = cmd.read_state()
            f = cmd.f.clone()
            ep_step = u.episode_length_buf.clone()
            obs = self.obs
            self.pi.obs_norm.update(obs["policy"])
            self.pi.goal_norm.update(obs["goal"])
            a = self.pi(obs["policy"], obs["goal"])
            a = networks.explore(a, self.cfg.action_noise, self.cfg.action_noise_dist)
            obs, rew, dones, _ = self.env.step(a)
            S1 = cmd.read_state()
            done = dones.bool()
            self.buffer.add({"s": flatten_body(S), "act": S[6], "touch": S[7], "a": a, "s1": flatten_body(S1), "touch1": S1[7],
                             "r": rew, "f": f, "ep_step": ep_step, "done": done})  # fmt: skip
            self.wm.in_norm.update(self.wm.inputs(S, S[6], S1[6]))
            self.wm.out_norm.update(self.wm.targets(S, S1))
            self.wm.inc_norm.update(WorldModel.increments(S, S1))
            self.wm.touch_norm.update(S1[7])
            self.sim_steps += N
            self.ep_ret += rew
            stats["r"].append(rew.mean() / self.max_step_reward)
            # per-term share of the normalized step reward (reward manager stores weight * value per term)
            stats["r_terms"].append(
                u.reward_manager._step_reward.mean(0)
                * self.reward_scale
                / self.max_step_reward
            )
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            if len(ids) > 0:
                T = (ep_step[ids] + 1).float()
                stats["ep_len"].append(T.mean())
                stats["ep_ret"].append(
                    (self.ep_ret[ids] / (T * self.max_step_reward)).mean()
                )
                stats["term"].append(u.reset_terminated[ids].float().sum())
                stats["done"].append(float(len(ids)))
                self.ep_ret[ids] = 0.0
                obs_dict, _ = u.reset(env_ids=ids)
                obs = TensorDict(obs_dict, batch_size=[N])
            self.obs = obs

    # ------------------------------------------------------------------ one training iteration
    def make_graphs(self):
        """cuda_graph: the world-model and policy updates as Graphed (captured after their warm-up calls); with
        overlap_wm the world model's on its own stream and random state. Returns them."""
        c = self.cfg
        self.graphs = {"wm": Graphed(self.wm_update, lambda: True, own=c.overlap_wm),
                       "policy": Graphed(self.policy_update, lambda: self.pi_avg is None or self._ema_n > 0)}  # fmt: skip
        if c.overlap_wm:
            self.wm_stream = torch.cuda.Stream(device=self.device)
        return self.graphs["wm"], self.graphs["policy"]

    @torch.no_grad()
    def sync_wm_imag(self):
        """overlap_wm: the world model's parameters and normalizers into imagination's copy (on the current stream)."""
        if self.wm_imag is self.wm:
            return
        if self._wm_sync is None:
            self._wm_sync = (list(self.wm_imag.parameters()) + list(self.wm_imag.buffers()),
                             [t.detach() for t in list(self.wm.parameters()) + list(self.wm.buffers())])  # fmt: skip
        torch._foreach_copy_(*self._wm_sync)

    def can_overlap_wm(self, it: int) -> bool:
        """overlap_wm: both update graphs are captured and the policy updates this iteration."""
        g = self.graphs
        return (self.cfg.overlap_wm and it >= self.cfg.wm_only_iters and g is not None and g["wm"].graph is not None
                and g["policy"].graph is not None)  # fmt: skip

    def _updates(self, it: int, agg: dict, wm_update, policy_update):
        """The world-model updates, then the policy update; with overlap_wm (once both are captured) the world-model
        updates run on their own stream concurrently with the policy update, whose imagination steps the copy of the
        world model taken here (one update round older)."""
        c = self.cfg
        if c.overlap_wm:
            self.sync_wm_imag()
        overlap = self.can_overlap_wm(it)
        if overlap:
            main, side = torch.cuda.current_stream(), self.wm_stream
            side.wait_stream(main)
            with torch.cuda.stream(side):
                for _ in range(c.wm_batches):
                    wl, e8, et = wm_update(record=main)
        else:
            for _ in range(c.wm_batches):
                wl, e8, et = wm_update()
        agg["wm"].append(wl), agg["wm8"].append(e8), agg["wmt"].append(et)
        if it >= c.wm_only_iters:
            pl, vl, ir, gn, surv, clip = policy_update()
            if c.empty_cache and str(self.device).startswith("cuda"):
                # hand the update's cached blocks back: MuJoCo-Warp allocates outside torch's pool when it steps
                torch.cuda.empty_cache()
            (
                agg["pi"].append(pl),
                agg["v"].append(vl),
                agg["ir"].append(ir),
                agg["gn"].append(gn),
                agg["surv"].append(surv),
            )
            agg["clip"].append(clip)
        if overlap:
            main.wait_stream(side)

    def can_overlap(self, it: int) -> bool:
        """overlap_collect: every graph this iteration replays is captured (warm-ups and captures run serially)."""
        col, g = self.collector, self.graphs
        if (
            col is None
            or not col.overlap
            or col.graph is None
            or g is None
            or g["wm"].graph is None
        ):
            return False
        return it < self.cfg.wm_only_iters or g["policy"].graph is not None

    def iterate(
        self, it: int, n: int, stats: dict, agg: dict, wm_update, policy_update
    ):
        """Collect n simulator steps, then the world-model and policy updates; with overlap_collect (once every graph
        is captured) the collection runs on its own stream concurrently with the updates, which sample the buffer as it
        stood before it (see SuperMimicRunnerCfg.overlap_collect)."""
        col = self.collector
        overlap = col is not None and col.overlap
        if overlap and self.can_overlap(it):
            main, side = torch.cuda.current_stream(), self.collect_stream
            col.pull_policy()
            col.push_norms()
            self.buffer.snapshot()
            side.wait_stream(main)
            with torch.cuda.stream(side):
                self.collect(n, stats, record=main)
            self._updates(it, agg, wm_update, policy_update)
            main.wait_stream(side)
            return
        if overlap:
            col.pull_policy()
        self.collect(n, stats)
        if overlap:
            col.push_norms()
            self.buffer.snapshot()
        self._updates(it, agg, wm_update, policy_update)

    # ------------------------------------------------------------------ mjlab runner interface
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        del init_at_random_ep_len  # episodes already start at sampled reference frames
        c, u = self.cfg, self.uenv
        assert (
            not u.cfg.auto_reset
        ), "SuperMimic records terminal states: the training env needs auto_reset=False"
        self.init_collection()
        logger = self._make_logger()
        stats = {k: [] for k in STATS}
        self.collect(max(c.warmup_steps, c.wm_window + self.hist + 1), stats)
        if c.cuda_graph:
            # captured sampling reads the buffer's write head and fill level on the device, so the updates are captured
            # at once (after three eager warm-up calls each) instead of once the buffer is full; the policy update's
            # capture waits for the first (eager) EMA copy of the policy
            if self.buffer.ptr_t is None:
                self.buffer.use_device_ptr()
            wm_update, policy_update = self.make_graphs()
        else:
            assert not c.overlap_wm, "overlap_wm overlaps the world-model and policy update graphs: needs cuda_graph"
            self.graphs = None
            wm_update, policy_update = self.wm_update, self.policy_update
        if c.gc_freeze:
            gc.collect()
            gc.freeze()
        t_start, t_log, step_acc = time.time() - self.wall_offset, time.time(), 0.0
        agg = {
            "wm": [],
            "wm8": [],
            "wmt": [],
            "pi": [],
            "v": [],
            "ir": [],
            "gn": [],
            "surv": [],
            "clip": [],
        }
        start_it = self.it
        for it in range(start_it, start_it + num_learning_iterations):
            lr = self.lr_at(it)
            pi_mult = c.pi_lr_mult  # lr_decay_ret: ramps to 1 after the trigger
            if self._decay_at is not None:
                pi_mult = c.pi_lr_mult + (1.0 - c.pi_lr_mult) * min(
                    1.0, (it - self._decay_at) / max(c.lr_decay_len, 1)
                )
            for o, k in ((self.opt_wm, 1.0), (self.opt_pi, pi_mult), (self.opt_v, 1.0)):
                set_lr(o, lr * k)
            step_acc += 1.0 / c.utd
            n = int(step_acc)
            step_acc -= n
            self.iterate(it, n, stats, agg, wm_update, policy_update)
            self.it = it + 1
            if self.it % c.log_interval == 0:
                mean = lambda x: _nanmean([float(v) for v in x])  # noqa: E731  (collect_graph: NaN = no episode ended)
                ips = c.log_interval / (time.time() - t_log)
                t_log = time.time()
                wall_h = (time.time() - t_start) / 3600
                term_frac = float(
                    sum(float(v) for v in stats["term"])
                    / max(sum(float(v) for v in stats["done"]), 1)
                )
                row = {"iteration": self.it, "wall_h": wall_h, "sim_steps": self.sim_steps, "wm_loss": mean(agg["wm"]),
                       "wm_err8": mean(agg["wm8"]), "wm_touch": mean(agg["wmt"]), "pi_loss": mean(agg["pi"]), "v_loss": mean(agg["v"]), "imag_rew": mean(agg["ir"]),
                       "pi_gradnorm": mean(agg["gn"]), "imag_survival": mean(agg["surv"]), "sim_rew": mean(stats["r"]),
                       "ep_len": mean(stats["ep_len"]), "ep_ret_norm": mean(stats["ep_ret"]), "term_frac": term_frac,
                       "it_per_s": ips, "lr": lr,
                       "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0}  # fmt: skip
                row.update(
                    {
                        "pi_lr": lr * pi_mult,
                        "pi_mult": float(pi_mult),
                        "nonfinite": int(self.nonfinite),
                    }
                )
                if c.lr_decay_ret > 0 and self._decay_at is None:
                    deadline = c.lr_decay_by > 0 and self.it >= c.lr_decay_by
                    if row["ep_ret_norm"] >= c.lr_decay_ret or deadline:
                        self._decay_at = self.it
                        why = (f"training return {row['ep_ret_norm']:.3f} >= {c.lr_decay_ret}" if row["ep_ret_norm"] >= c.lr_decay_ret
                               else f"deadline lr_decay_by {c.lr_decay_by}")  # fmt: skip
                        print(
                            f"lr_decay_ret: {why} at it {self.it}: policy LR x{c.pi_lr_mult} -> x1 over {c.lr_decay_len} its",
                            flush=True,
                        )
                if c.state_grad_clip > 0:
                    row["pi_state_clip"] = mean(agg["clip"])
                if stats["r_terms"]:
                    terms = torch.stack(stats["r_terms"]).mean(0).tolist()
                    row.update(
                        {
                            f"rew_{n}": float(v)
                            for n, v in zip(self.reward_names, terms, strict=True)
                        }
                    )
                logger(row)
                clip_s = (
                    f" clip {row['pi_state_clip']:.3f}"
                    if "pi_state_clip" in row
                    else ""
                )
                print(f"it {self.it} | {wall_h:.3f} h | wm {row['wm_loss']:.4f} e8 {row['wm_err8']:.4f} | pi {row['pi_loss']:.3f} "
                      f"v {row['v_loss']:.4f} imag_r {row['imag_rew']:.3f} surv {row['imag_survival']:.3f}{clip_s} | sim_r {row['sim_rew']:.3f} "
                      f"ep_len {row['ep_len']:.1f} ep_ret {row['ep_ret_norm']:.3f} term {term_frac:.2f} | {ips:.2f} it/s {row['gpu_mem_gb']:.1f} GB", flush=True)  # fmt: skip
                stats = {k: [] for k in stats}
                agg = {k: [] for k in agg}
            stop = c.max_hours > 0 and time.time() - t_start >= c.max_hours * 3600
            if self.log_dir and (
                self.it % c.save_interval == 0
                or it + 1 == start_it + num_learning_iterations
                or stop
            ):
                if self.collector is not None:
                    self.collector.push_norms()  # overlap: the latest statistics into the saved networks
                self.wall_offset = time.time() - t_start
                self.save(os.path.join(self.log_dir, f"model_{self.it:06d}.pt"))
            if stop:
                print(f"max_hours: {(time.time() - t_start) / 3600:.3f} h of training reached at it {self.it} "
                      f"(budget {c.max_hours} h): stopping", flush=True)  # fmt: skip
                break

    def save(self, path: str, infos=None):
        # "pi" is the deployed policy (the EMA average with pi_ema: what evaluation and play read), "pi_live" the live one
        ck = {"wm": self.wm.state_dict(), "pi": self.deployed.state_dict(), "pi_live": self.pi.state_dict(),
              "pi_avg": self.pi_avg.state_dict() if self.pi_avg is not None else None, "critic": self.critic.state_dict(),
              "opt_wm": self.opt_wm.state_dict(), "opt_pi": self.opt_pi.state_dict(), "opt_v": self.opt_v.state_dict(),
              "iteration": self.it, "wall_time": self.wall_offset, "sim_steps": self.sim_steps, "cfg": asdict(self.cfg),
              "ctrl": {"decay_at": self._decay_at, "nonfinite": int(self.nonfinite)},
              "env": self.env_settings(), "action_map": self.pi.action_map.spec(),
              "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        if self.critic_targ is not self.critic:
            ck["critic_targ"] = self.critic_targ.state_dict()
        torch.save(ck, path + ".tmp")
        os.replace(path + ".tmp", path)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location=None,
    ):
        ck = torch.load(
            path, map_location=map_location or self.device, weights_only=False
        )
        # the policy's action map (identity; checkpoints from before action maps have none): another map would need
        # another policy head, so a mismatch is an error rather than a silent reinterpretation of the outputs
        saved_map = ActionMap.from_spec(ck.get("action_map"), self.nm).spec()
        if saved_map != self.pi.action_map.spec():
            raise ValueError(
                f"checkpoint action map {saved_map} != {self.pi.action_map.spec()}"
            )
        saved = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in ck["cfg"].items()
            if k in self.NET_FIELDS
        }
        if any(
            saved[k] != getattr(self.cfg, k) for k in saved
        ):  # e.g. play/eval with the registered default cfg
            for k, v in saved.items():
                setattr(self.cfg, k, v)
            self._build_networks()
        if load_cfg is not None and set(load_cfg) <= {
            "actor"
        }:  # evaluation, play: the deployed policy only
            self.deployed.load_state_dict(ck["pi"], strict=strict)
            return ck
        self.pi.load_state_dict(ck.get("pi_live", ck["pi"]), strict=strict)
        if self.pi_avg is not None:
            if ck.get("pi_avg") is not None:
                self.pi_avg.load_state_dict(ck["pi_avg"])
            else:  # a checkpoint without an average: continue the EMA from its deployed policy
                self.pi_avg.module.load_state_dict(ck["pi"], strict=strict)
                self.pi_avg.n_averaged.fill_(1)
            self._ema_n = int(self.pi_avg.n_averaged)
        wm_sd = ck["wm"]
        if (
            "inc_norm.mean" not in wm_sd
        ):  # a checkpoint of the code before inc_norm: its out_norm is the increments' normalizer
            wm_sd = {
                **wm_sd,
                **{
                    f"inc_norm.{k}": wm_sd[f"out_norm.{k}"]
                    for k in ("mean", "var", "count")
                },
            }
        self.wm.load_state_dict(wm_sd, strict=strict)
        self.critic.load_state_dict(ck["critic"], strict=strict)
        if self.critic_targ is not self.critic:
            self.critic_targ.load_state_dict(
                ck.get("critic_targ", ck["critic"]), strict=strict
            )
        self.graphs = None  # cuda_graph: the optimizer states below are new tensors, so a captured update would be stale
        for k in ("opt_wm", "opt_pi", "opt_v"):
            o = getattr(self, k)
            lr_t = [g["lr"] for g in o.param_groups]
            o.load_state_dict(ck[k])
            for g, lr0 in zip(
                o.param_groups, lr_t
            ):  # cuda_graph: keep the device learning-rate tensors (filled)
                if torch.is_tensor(lr0) and g["lr"] is not lr0:
                    lr0.fill_(float(g["lr"]))
                    g["lr"] = lr0
        self.it, self.wall_offset, self.sim_steps = (
            ck["iteration"],
            ck["wall_time"],
            ck["sim_steps"],
        )
        ctrl = ck.get("ctrl") or {}
        self._decay_at = ctrl.get("decay_at")
        self.nonfinite.fill_(int(ctrl.get("nonfinite", 0)))
        if self.cfg.finetune:  # a new run from a trained baseline: its clock, schedule and warm-up start again
            self.it, self.wall_offset, self.sim_steps, self._decay_at = 0, 0.0, 0, None
            self.cfg.wm_only_iters = 0
        return ck

    def env_settings(self) -> dict:
        """The env settings a policy depends on, stored in each checkpoint so evaluation can match them."""
        return {"decimation": self.uenv.cfg.decimation, "stride": self.cmd.stride, "lookahead": self.cmd.lookahead,
                "muscle_obs": self.cmd.observes_muscles, "obs_dim": self.obs_policy.dim, "goal_dim": self.obs_goal.dim}  # fmt: skip

    def get_inference_policy(self, device=None):
        pi = self.deployed
        pi.eval()
        if device is not None:
            pi.to(device)

        @torch.no_grad()
        def policy(obs) -> torch.Tensor:
            return pi(obs["policy"], obs["goal"])

        return policy

    def add_git_repo_to_log(self, repo_file_path: str):
        pass

    def _make_logger(self):
        if not self.log_dir:
            return lambda row: None
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "train_log.csv")
        new = not os.path.exists(path)
        fh = open(path, "a", newline="")
        writer, tb, wb = None, None, None
        if self.cfg.logger == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb = SummaryWriter(self.log_dir)
            except ImportError:
                print(
                    "[WARN] tensorboard is not installed; logging to train_log.csv only"
                )
        elif self.cfg.logger == "wandb":
            import wandb

            wb = wandb.init(project=self.cfg.wandb_project, name=self.cfg.run_name or os.path.basename(self.log_dir),
                            dir=self.log_dir, config=asdict(self.cfg))  # fmt: skip

        def log(row: dict):
            nonlocal writer, new
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    writer.writeheader()
            writer.writerow(
                {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()}
            )
            fh.flush()
            if tb is not None:
                for k, v in row.items():
                    if k != "iteration":
                        tb.add_scalar(f"train/{k}", v, row["iteration"])
            if wb is not None:
                wb.log({f"train/{k}": v for k, v in row.items()}, step=row["iteration"])

        return log


def _nanmean(x: list[float]) -> float:
    x = [v for v in x if v == v]
    return float(sum(x) / max(len(x), 1))


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)
