# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# training/supermimic/ for the notices of the code it includes.
"""Graphed collection (``collect_graph``; BlinkMimic's ``--collect_graph --reset_masked``).

A thin simulator wrapper replaces mjlab's manager env during collection. One step does what the runner's eager
``collect`` does through ``env.step`` and ``env.reset(ids)``, on the same tensors and in the same order:

1. the policy observation of the current state (the analytic observation terms; no history),
2. the live policy plus Gaussian noise, clipped to [0, 1], written as the muscles' ``ctrl``,
3. the decimation substeps (MuJoCo-Warp's ``step``, solver loop included), the episode-length update,
4. rewards and terminations of the post-step state against frame f + 1 before ``forward``, as the env computes them
   (the analytic heads, hard thresholds, NaN rewards zeroed), and the time-out terms of the env itself,
5. ``forward``, the post-step state, the motion command's update, the replay-buffer write at the device-side write
   head, the normalizer updates and the step statistics,
6. the episode ends as fixed-size masked updates: the bin statistics, MuJoCo-Warp's ``reset_data`` of the ended
   worlds, reset frames drawn for every env and written where an episode ended, the action history zeroed there,
   and a second ``forward`` (the eager reset's).

With no host synchronization in a step, the step is captured after three eager warm-up calls as one CUDA graph (the
torch work and the Warp kernels on one stream, Warp's capture nested in torch's) and replayed from then on. On a CPU
device the step runs eagerly. The same transitions as the eager path up to floating-point rounding, on another random
stream (the reset draw is an inverse-CDF draw instead of ``multinomial``).

The collector owns the training env's simulator once built: mjlab's action manager, actuator targets, observation
buffers and step counter are not advanced (nothing reads them during training).

With ``overlap`` (overlap_collect) the step runs on a second stream concurrently with the updates, so it shares no
written state with them: it acts with its own copy of the policy (``pull_policy`` refreshes its parameters from the
live policy), updates its own copies of the observation and world-model normalizers (``push_norms`` copies their
statistics to the trainer's), draws from its own random state and is captured on its own stream (its own cuBLAS
workspace).

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at commit
8a6244b, ``src/muscle_supermimic/rl/supermimic/collector.py``.
"""

from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING

import torch

from supermimic.networks import WorldModel
from supermimic.heads import (
    StepCtx,
    flatten_body,
)

if TYPE_CHECKING:
    from supermimic.runner import SuperMimicRunner

STATS = ("r", "r_terms", "ep_len", "ep_ret", "term", "done")


class Collector:
    def __init__(
        self, runner: SuperMimicRunner, warmup: int = 3, overlap: bool = False
    ):
        import mujoco_warp as mjwarp

        r, u, cmd = runner, runner.uenv, runner.cmd
        c = r.cfg
        assert (
            c.no_sync
        ), "collect_graph: the normalizer updates run on the device (no_sync)"
        assert (
            cmd.bin_fix
        ), "collect_graph: the masked bin statistics are bin_fix's (reset.bin_fix)"
        assert r.hist == 0, "collect_graph: observation histories are not reproduced"
        assert not u.cfg.auto_reset
        assert (
            not u.event_manager.available_modes
        ), "collect_graph: events are not reproduced"
        self.r, self.u, self.cmd, self.mjwarp = r, u, cmd, mjwarp
        self.sim = sim = u.sim
        d = sim.data
        # the simulator's arrays as torch tensors (zero-copy views of the Warp arrays), written directly
        self.qpos, self.qvel, self.act, self.ctrl = (
            getattr(d, k)._tensor for k in ("qpos", "qvel", "act", "ctrl")
        )
        self.reset_mask = sim._reset_mask._tensor
        self.ctrl_ids = cmd.robot.indexing.ctrl_ids.long()
        am = u.action_manager
        self.a_last, self.a_prev = (
            am.action.clone(),
            am.prev_action.clone(),
        )  # the action history the observation reads
        self.fail_terms = [
            (fn, {k: v for k, v in p.items() if k != "soft_width"})
            for fn, p in r.fail_terms
        ]  # hard
        tm = u.termination_manager
        self.time_outs = [(tc.func, tc.params) for tc in tm._term_cfgs if tc.time_out]
        # the foot touch the next observation reads, with the eager path's sensor caching: mjlab's contact-sensor cache
        # is filled by the reward terms before forward() and only dropped by a reset, so after a step the observation
        # reads the touch of the last substep's start, unless some env reset (then every env reads the post-forward
        # touch; at thousands of envs nearly every step)
        self.touch = cmd.read_state()[7].clone()
        self.cuda = str(r.device).startswith("cuda")
        self.warmup, self.calls, self.graph, self.out = warmup, 0, None, None
        # the policy it acts with and the normalizers it updates: the trainer's own, or (overlap) copies of them
        self.overlap = overlap
        assert not overlap or self.cuda, "overlap_collect needs a GPU"
        names = ("in_norm", "out_norm", "inc_norm", "touch_norm")
        self.pi = copy.deepcopy(r.pi) if overlap else r.pi
        self.norms = {
            k: copy.deepcopy(getattr(r.wm, k)) if overlap else getattr(r.wm, k)
            for k in names
        }
        if overlap:
            self.pi.requires_grad_(False)  # it only acts
            self._params = (list(self.pi.parameters()), list(r.pi.parameters()))
            mine = [self.pi.obs_norm, self.pi.goal_norm] + [
                self.norms[k] for k in names
            ]
            theirs = [r.pi.obs_norm, r.pi.goal_norm] + [getattr(r.wm, k) for k in names]
            self._norm_bufs = (
                [b for n in mine for b in n.buffers()],
                [b for n in theirs for b in n.buffers()],
            )
            gen = torch.cuda.default_generators[torch.device(r.device).index or 0]
            self.rng = (
                gen.clone_state()
            )  # the captured step's own random state (seeded from the run's seed)
            self.rng.manual_seed(int(torch.randint(0, 2**62, ())))
            self.stream = torch.cuda.Stream(
                device=r.device
            )  # the capture stream (and its cuBLAS workspace)

    @torch.no_grad()
    def pull_policy(self):
        """overlap: the live policy's parameters into the collector's copy (on the current stream)."""
        if self.overlap:
            torch._foreach_copy_(*self._params)

    @torch.no_grad()
    def push_norms(self):
        """overlap: the collector's normalizer statistics into the trainer's (on the current stream)."""
        if self.overlap:
            torch._foreach_copy_(self._norm_bufs[1], self._norm_bufs[0])

    # ------------------------------------------------------------------ one step (the captured body)
    def _step(self) -> dict:
        r, u, cmd, mjwarp = self.r, self.u, self.cmd, self.mjwarp
        _c, heads, m, sim = r.cfg, r.heads, cmd.motion, self.sim
        S = cmd.read_state(fresh_touch=True)[:7] + (self.touch.clone(),)
        f = cmd.f.clone()
        ep_step = u.episode_length_buf.clone()
        ctx = StepCtx(S, f, self.a_last, self.a_prev)
        ctx.cache.update(cmd._shared_obs(S))
        obs, goal = r.obs_policy.assemble(None, ctx), r.obs_goal.assemble(None, ctx)
        self.pi.obs_norm.update(obs)
        self.pi.goal_norm.update(goal)
        a = self.pi(obs, goal)
        a = r.explore_actions(a).to(a.dtype)
        # action j drives actuator j: a muscle's excitation, a motor's command mapped onto its range (as the env)
        self.ctrl[:, self.ctrl_ids] = cmd.to_ctrl(a)
        for _ in range(u.cfg.decimation):
            mjwarp.step(sim.wp_model, sim.wp_data)
        u.episode_length_buf.add_(1)

        # rewards and terminations: the post-step state against frame f + 1, before forward() as in the env
        S1 = cmd.read_state(fresh_touch=True)
        tctx = StepCtx(
            S1, torch.minimum(f + 1, m.clip_end[m.clip_id[f]]), a, self.a_last
        )
        tctx.cache.update(cmd._shared_transition(S1))
        terms = torch.stack([torch.nan_to_num(w * fn(heads, tctx, **p), nan=0.0, posinf=0.0, neginf=0.0)
                             for fn, w, p in r.reward_terms], -1)  # fmt: skip
        rew = terms.sum(-1)
        fail = torch.zeros_like(f, dtype=torch.bool)
        for fn, p in self.fail_terms:
            fail = fail | fn(heads, tctx, **p)
        trunc = torch.zeros_like(fail)
        for (
            fn,
            p,
        ) in self.time_outs:  # episode length and the clip's last frame (they read episode_length_buf and f)
            trunc = trunc | fn(u, **p)
        done = fail | trunc

        # forward() (derived quantities of the post-step state; S1, touch included, is what the eager path stores: the
        # cached pre-forward read), then the motion command's update (every env stepped)
        mjwarp.forward(sim.wp_model, sim.wp_data)
        T = u.episode_length_buf.float()
        cmd.ep_disc_ret.add_((cmd.gamma ** (T - 1).clamp(min=0)) * rew)
        cmd.ep_ret.add_(rew)
        cmd.f.add_(1)

        r.buffer.add_device({"s": flatten_body(S), "act": S[6], "touch": S[7], "a": a, "s1": flatten_body(S1), "touch1": S1[7],
                             "r": rew, "f": f, "ep_step": ep_step, "done": done})  # fmt: skip
        wm, nm = r.wm, self.norms
        nm["in_norm"].update(wm.inputs(S, S[6], S1[6]))
        nm["out_norm"].update(wm.targets(S, S1))
        nm["inc_norm"].update(WorldModel.increments(S, S1))
        nm["touch_norm"].update(S1[7])
        r.ep_ret.add_(rew)

        # step statistics (the means over this step's ended episodes are NaN if none ended; the log skips them)
        mx, nd = r.max_step_reward, done.sum()
        Tn = (ep_step + 1).float()
        nan = torch.full((), float("nan"), device=f.device)
        stats = {"r": rew.mean() / mx, "r_terms": terms.mean(0) / mx,
                 "ep_len": torch.where(nd > 0, (Tn * done).sum() / nd.clamp(min=1), nan),
                 "ep_ret": torch.where(nd > 0, (cmd.ep_ret / (Tn * mx) * done).sum() / nd.clamp(min=1), nan),
                 "term": (fail & done).float().sum(), "done": nd.float()}  # fmt: skip
        if r.det_mask is not None:  # the noise-free envs' episodes (noise_free_frac)
            dd = done & r.det_mask
            nn_ = dd.sum()
            stats["det_ret"] = torch.where(
                nn_ > 0, (cmd.ep_ret / (Tn * mx) * dd).sum() / nn_.clamp(min=1), nan
            )
            stats["det_len"] = torch.where(
                nn_ > 0, (Tn * dd).sum() / nn_.clamp(min=1), nan
            )

        # episode ends: the eager reset's updates for the ended envs, as masked updates of every env
        cmd.finish_episodes_masked(done, fail, u.episode_length_buf)
        r.ep_ret.masked_fill_(done, 0.0)
        self.reset_mask.copy_(done)
        mjwarp.reset_data(
            sim.wp_model, sim.wp_data, reset=sim._reset_mask_wp
        )  # zeroes their ctrl too
        frames, bins = cmd.sample_frames_masked(done.any())
        cmd.write_reference_state_masked(
            done, frames, bins, self.qpos, self.qvel, self.act
        )
        u.episode_length_buf.masked_fill_(done, 0)
        dm = done.unsqueeze(-1)
        self.a_prev.copy_(torch.where(dm, 0.0, self.a_last))
        self.a_last.copy_(torch.where(dm, 0.0, a))
        mjwarp.forward(sim.wp_model, sim.wp_data)
        fresh = cmd.read_state(fresh_touch=True)[7]
        self.touch.copy_(torch.where(done.any(), fresh, S1[7]))
        return stats

    # ------------------------------------------------------------------ eager warm-up, capture, replay
    @torch.no_grad()
    def step(self) -> dict:
        """One collection step; returns its statistics (static tensors when replayed: clone to keep them)."""
        import warp as wp

        self.calls += 1
        if not self.cuda:
            with wp.ScopedDevice(
                self.sim.wp_device
            ):  # MuJoCo-Warp launches on the default device otherwise
                out = self._step()
        elif self.calls <= self.warmup:
            # Warp's launches on torch's stream, as in the capture
            with (
                wp.ScopedDevice(self.sim.wp_device),
                wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())),
            ):
                out = self._step()
        else:
            if self.graph is None:
                self._capture()
            self.graph.replay()
            out = self.out
        self.r.buffer.advance_host()
        # mjlab's per-step caches would hold states of before the step
        if self.cmd.touch_sensor is not None:
            self.cmd.touch_sensor._invalidate_cache()
        return out

    def _capture(self):
        import warp as wp

        torch.cuda.synchronize()
        t0 = time.time()
        self.graph = torch.cuda.CUDAGraph()
        gen = orig = None
        if (
            self.overlap
        ):  # captured with its own random state on its own stream (see the class notes)
            gen = torch.cuda.default_generators[torch.device(self.r.device).index or 0]
            orig = gen.graphsafe_get_state()
            gen.graphsafe_set_state(self.rng)
        try:
            with torch.cuda.graph(self.graph, stream=self.stream if self.overlap else None, capture_error_mode="thread_local"), \
                    wp.ScopedDevice(self.sim.wp_device):  # fmt: skip
                ws = wp.stream_from_torch(torch.cuda.current_stream())
                with wp.ScopedStream(ws, sync_enter=False, sync_exit=False):
                    wp.capture_begin(stream=ws, external=True)
                    try:
                        self.out = self._step()
                    finally:
                        self.wp_graph = wp.capture_end(
                            stream=ws
                        )  # keep: it holds the captured kernels' modules
        finally:
            if gen is not None:
                gen.graphsafe_set_state(orig)
        torch.cuda.synchronize()
        print(f"collect_graph: captured one collection step in {time.time() - t0:.1f} s, "
              f"{torch.cuda.memory_reserved() / 1e9:.1f} GB reserved", flush=True)  # fmt: skip
