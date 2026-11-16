# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# training/supermimic/ for the notices of the code it includes.
"""Checkpoint evaluation with test-time action refinement through the learned world model.

As :func:`.evaluate.evaluate` (every clip once from its first frame, one env per clip, deterministic policy, the same metrics), but at
every control step the deployed policy's action is first refined from the simulator's state by K normalized
gradient-ascent steps on

    J(a0) = sum_{t<H} gamma^t c_t r_t + gamma^H c_H V(x_H),

the return of an H-step imagined rollout with a0 as its first action (the world model, the analytic rewards and soft
terminations, the deployed policy from the second step on, the bootstrap critic at the end), as the policy update
computes it. Each step moves every env's action by an RMS of ``eta`` per excitation; each env executes the iterate of
highest J, the policy's own action included (``keep_best=False``: the last one); K = 0 is the plain evaluation. It is Qflex's
Q-guided action refinement (Wei et al., ICLR 2026) with the world model's imagined return in place of a learned Q.

``search`` (step sizes e1, e2, ...): each of the K rounds takes one gradient and tries all the step sizes at once, the candidates of
every env batched into one imagined rollout (a parallel line search), and the best iterate seen is executed. A round
then costs about two rollouts instead of one per gradient step, which matters when few envs run and the rollouts are
latency-bound; on long clips it keeps less of the gain than K sequential gradient steps.

The refinement of one control step (its imagined rollouts and gradients) is captured as one CUDA graph after three
warm-up steps and replayed from then on (the runner's ``Graphed``, as its updates are); ``graph=False`` runs it
eagerly, ``check`` compares the two on the first N graphed steps. ``training/supermimic/scripts/evaluate.py --refine K`` is the
command line.

In muscle-supermimic, with the defaults (K 5, eta 0.01, the training horizon), MS-Human-700's 20-minute checkpoints go from a normalized
return of 0.940 to 0.965 on the 108 held-out KIT clips (two seeds, no failed clip) and from an MPJPE of 18.3 to 13.6 mm;
a control step of the 108 clips takes 0.17 s instead of the policy's 0.04 s on an RTX 5070 Ti.

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at commit
8a6244b, ``src/muscle_supermimic/tasks/tracking/scripts/refine.py``.
"""

import time

import numpy as np
import torch
from tensordict import TensorDict

from supermimic.runner import Graphed, set_requires_grad
from supermimic.heads import StepCtx
from supermimic.evaluate import _metrics, _track_errors
from supermimic.rotations import mat_to_quat


class Refiner:
    def __init__(
        self,
        runner,
        ck,
        K,
        eta,
        horizon,
        keep_best,
        compile_=True,
        graph=True,
        check=0,
        etas=None,
    ):
        c = ck["cfg"]
        self.r, self.K, self.eta, self.keep_best = runner, K, eta, keep_best
        self.etas = (
            None
            if etas is None
            else torch.tensor(etas, dtype=torch.float32, device=runner.device)
        )
        self.H, self.gamma = horizon or c["horizon"], c["gamma"]
        self.vmax = (
            runner.max_step_reward / (1 - self.gamma) if c.get("value_clip") else None
        )
        wm_sd = ck["wm"]
        if "inc_norm.mean" not in wm_sd:
            wm_sd = {
                **wm_sd,
                **{
                    f"inc_norm.{k}": wm_sd[f"out_norm.{k}"]
                    for k in ("mean", "var", "count")
                },
            }
        runner.wm.load_state_dict(wm_sd)
        runner.critic.load_state_dict(
            ck.get("critic_targ") or ck["critic"]
        )  # the bootstrap value training used
        self.wm, self.critic, self.pi = (
            runner.wm.eval(),
            runner.critic.eval(),
            runner.deployed.eval(),
        )
        for net in (self.wm, self.critic, self.pi):
            set_requires_grad(net, False)
        print(
            f"refine: K {K}, eta {eta}, horizon {self.H}, gamma {self.gamma}, value clip {self.vmax}, keep best {keep_best}, "
            f"search {etas}"
        )
        # per imagined step, as the runner compiles its imag_step; static shapes: the search runs two batch sizes
        comp = (
            (lambda fn: torch.compile(fn, dynamic=False))
            if compile_
            else (lambda fn: fn)
        )
        self.step_given, self.step_pi, self.tail = (
            comp(self._step_given),
            comp(self._step_pi),
            comp(self._tail),
        )
        self.graph, self.check, self.buf, self.graphed, self.max_diff = (
            graph,
            check,
            None,
            None,
            0.0,
        )

    def _transition(self, S, f, a_last, a, end):
        r = self.r
        act1, _ = r.muscle.step(S[6], a)
        body1, touch_n = self.wm.step(S, S[6], act1)
        S1, f1 = body1 + (act1, self.wm.touch(touch_n)), torch.minimum(f + 1, end)
        rew, d = r._reward_term(S1, f1, a, a_last)
        return S1, f1, rew, d

    def _step_given(self, S, f, a_last, a, end):
        return (a,) + self._transition(S, f, a_last, a, end)

    def _step_pi(self, S, f, a_last, end):
        ctx = StepCtx(S, f, a_last)
        a = self.pi(
            self.r.obs_policy.assemble(None, ctx), self.r.obs_goal.assemble(None, ctx)
        )
        return (a,) + self._transition(S, f, a_last, a, end)

    def _tail(self, S, f, a_last):
        v = self.critic(self.r._critic_input(S, f, a_last))
        return v.clamp(max=self.vmax) if self.vmax is not None else v

    def imagine(self, S, f, a_last, a0):
        m = self.r.motion
        end = m.clip_end[m.clip_id[f]]
        J, c = 0.0, torch.ones_like(f, dtype=torch.float32)
        for t in range(self.H):
            a, S1, f1, rew, d = (
                self.step_given(S, f, a_last, a0, end)
                if t == 0
                else self.step_pi(S, f, a_last, end)
            )
            J = J + self.gamma**t * c * rew
            c = c * (1 - d)
            S, f, a_last = S1, f1, a
        return J + self.gamma**self.H * c * self.tail(S, f, a_last)

    def __call__(self, ctx, a_pi):
        S, f, a_last = tuple(x.detach() for x in ctx.S), ctx.f, ctx.a.detach()
        if not self.graph:
            return self._refine(S, f, a_last, a_pi)
        inputs = (*S, f, a_last, a_pi)
        if self.buf is None:  # the captured refinement's static inputs
            self.buf = [x.clone() for x in inputs]
            self.graphed = Graphed(self._refine_buf, lambda: True, warmup=3)
        for b, x in zip(self.buf, inputs, strict=True):
            b.copy_(x)
        out = self.graphed()
        if (
            self.check > 0 and self.graphed.graph is not None
        ):  # the eager refinement of the same inputs
            self.check -= 1
            ref = self._refine(S, f, a_last, a_pi)
            d = (out[0] - ref[0]).abs().max().item()
            self.max_diff = max(self.max_diff, d)
            print(
                f"check-graph: max |a_graph - a_eager| {d:.2e}, "
                f"gain {out[1].mean().item():.2e} vs {ref[1].mean().item():.2e}",
                flush=True,
            )
        return out

    def _refine_buf(self):
        b, n = self.buf, len(self.buf) - 3
        return self._refine(tuple(b[:n]), b[n], b[n + 1], b[n + 2])

    def _refine_search(self, S, f, a_last, a_pi):
        """K rounds of one gradient at the best iterate so far and all of ``self.etas`` as step sizes along it, the
        candidates of every env batched into one imagined rollout; returns the best iterate seen and its gain."""
        C, M = a_pi.shape[0], len(self.etas)

        def rep(x):
            return x.repeat_interleave(
                M, 0
            )  # env-major, as the candidates are flattened  # noqa: E731

        S_m, f_m, a_last_m, rows = (
            tuple(rep(x) for x in S),
            rep(f),
            rep(a_last),
            torch.arange(C, device=a_pi.device),
        )
        best_a, best_J, J0 = a_pi.clone(), None, None
        with torch.enable_grad():
            for _ in range(self.K):
                a_in = best_a.detach().requires_grad_(True)
                J = self.imagine(S, f, a_last, a_in)
                (g,) = torch.autograd.grad(J.sum(), a_in)
                with torch.no_grad():
                    if J0 is None:
                        J0 = J.detach().clone()
                        best_J = J0.clone()
                    d = g / (g.pow(2).mean(-1, keepdim=True).sqrt() + 1e-12)
                    cand = (
                        best_a.unsqueeze(1) + self.etas.view(1, M, 1) * d.unsqueeze(1)
                    ).clamp(0.0, 1.0)  # (C, M, A)
                    Jc = self.imagine(S_m, f_m, a_last_m, cand.reshape(C * M, -1)).view(
                        C, M
                    )
                    Jm, i = Jc.max(1)
                    better = Jm > best_J
                    best_J = torch.where(better, Jm, best_J)
                    best_a = torch.where(better.unsqueeze(-1), cand[rows, i], best_a)
        return best_a, (best_J - J0)

    def _refine(self, S, f, a_last, a_pi):
        if self.etas is not None:
            return self._refine_search(S, f, a_last, a_pi)
        delta = torch.zeros_like(a_pi, requires_grad=True)
        best_a, best_J, J0 = a_pi.clone(), None, None
        with torch.enable_grad():
            for k in range(self.K + 1):
                a0 = (a_pi + delta).clamp(0.0, 1.0)
                J = self.imagine(S, f, a_last, a0)
                with torch.no_grad():
                    if best_J is None:
                        best_J, J0 = J.detach().clone(), J.detach().clone()
                    elif self.keep_best:
                        better = J > best_J
                        best_J = torch.where(better, J, best_J)
                        best_a = torch.where(better.unsqueeze(-1), a0.detach(), best_a)
                    else:
                        best_J, best_a = J.detach(), a0.detach()
                if k == self.K:
                    break
                (g,) = torch.autograd.grad(J.sum(), delta)
                with torch.no_grad():
                    delta += (
                        self.eta * g / (g.pow(2).mean(-1, keepdim=True).sqrt() + 1e-12)
                    )
        return best_a, (best_J - J0)


@torch.no_grad()
def run(env, runner, refiner, max_steps=0, record=None):
    u, cmd = env.unwrapped, runner.cmd
    m, C = cmd.motion, len(cmd.motion.names)
    ids = torch.arange(C, device=u.device)
    obs = TensorDict(u.reset(env_ids=ids)[0], batch_size=[C])
    policy = runner.get_inference_policy()
    ret = torch.zeros(C, device=u.device)
    done_eval = torch.zeros(C, dtype=torch.bool, device=u.device)
    failed = torch.zeros_like(done_eval)
    acc, n_ok = None, torch.zeros(C, device=u.device)
    gain, n_steps, t0, stamps = [], 0, time.time(), []
    q_adr = torch.cat((cmd._free_q, cmd.robot.indexing.joint_q_adr.long()))
    traj = {"qpos": [], "f": [], "alive": [], "act": []}  # as evaluate(record=...)
    while not bool(done_eval.all()):
        if max_steps and n_steps >= max_steps:
            last = sorted(b - a for a, b in zip(stamps[-21:-1], stamps[-20:]))
            print(
                f"PROFILE {n_steps} steps, {(time.time() - t0) / n_steps:.3f} s per step overall, "
                f"{last[len(last) // 2]:.3f} s median of the last 20",
                flush=True,
            )
            raise SystemExit
        alive = ~done_eval
        if record:
            qpos = u.sim.data.qpos[:, q_adr].clone()
            if not cmd.fixed_root:
                qpos[:, :3] -= u.scene.env_origins
            (
                traj["qpos"].append(qpos.cpu()),
                traj["f"].append(cmd.f.clone().cpu()),
                traj["alive"].append(alive.cpu()),
            )
            traj["act"].append(u.sim.data.act[:, cmd._act_adr].cpu())
        a = policy(obs)
        if refiner is not None and refiner.K > 0:
            a, dJ = refiner(cmd.obs_ctx(), a)
            gain.append(dJ[alive].mean().item())
        obs, rew, dones, _ = env.step(a)
        n_steps += 1
        if torch.cuda.is_available() and str(u.device).startswith("cuda"):
            torch.cuda.synchronize()
        stamps.append(time.time())
        ret += torch.where(alive, rew, 0.0)
        ok = (alive & ~dones.bool()).float()
        errs = _track_errors(cmd)
        acc = {k: e * ok if acc is None else acc[k] + e * ok for k, e in errs.items()}
        n_ok += ok
        failed |= u.reset_terminated & alive
        done_eval |= dones.bool()
        reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            obs = TensorDict(u.reset(env_ids=reset_ids)[0], batch_size=[C])
    norm = ret / ((m.clip_len - 1).float() * cmd.max_step_reward)
    alive_s = {
        m.names[i]: round(n_ok[i].item() * u.step_dt, 2) for i in range(C)
    }  # control steps counted, in seconds
    res = {"norm_return": norm.mean().item(), "n_failed": int(failed.sum().item()),
           "failed": [m.names[i] for i in failed.nonzero().flatten().tolist()], "alive_s": alive_s,
           "clip_s": {m.names[i]: round((m.clip_len[i].item() - 1) * u.step_dt, 2) for i in range(C)},
           **_metrics(acc, n_ok, m.names),
           "imagined_gain_mean": float(sum(gain) / len(gain)) if gain else 0.0, "s_per_step": (time.time() - t0) / n_steps,
           "s_per_step_median": sorted(b - a for a, b in zip(stamps[:-1], stamps[1:]))[(len(stamps) - 1) // 2]}  # fmt: skip
    if record:
        h = cmd.heads
        ref_q, ref_quat = h.ref_q, mat_to_quat(h.root_R @ cmd.kin.R_off.T)
        np.savez_compressed(record, names=np.array(m.names), **{k: torch.stack(v).numpy() for k, v in traj.items()},
                            ref_qpos=(ref_q if cmd.fixed_root else torch.cat((h.root_p, ref_quat, ref_q), -1)).cpu().numpy(),
                            clip_start=m.clip_start.cpu().numpy(), clip_len=m.clip_len.cpu().numpy(), fps=1.0 / u.step_dt)  # fmt: skip
    return res
