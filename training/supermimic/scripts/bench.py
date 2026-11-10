# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# training/supermimic/ for the notices of the code it includes.
"""Wall-clock of one training iteration, split into collection, world-model updates and the policy update.

Builds ``myoMimic-v0`` through ``make_env`` with the fast recipe's options and the SuperMimic runner with the
fast recipe, plus ``--set`` (runner cfg) and ``--env-set`` (``TaskEnvCfg`` field) overrides, trains ``--warmup`` iterations untimed (compilation, buffer fill;
policy updates from the first iteration) and then times ``--iters`` iterations piece by piece, the device
synchronized around each piece. Reports seconds per iteration and torch's peak memory.

  python training/supermimic/scripts/bench.py --motion-file <kit train list> --motion-root <dataset root>
  python training/supermimic/scripts/bench.py ... --set overlap_collect=false collect_graph=false cuda_graph=false

With the recipe's ``overlap_collect`` the runner's own ``iterate`` (the collection on a second stream concurrently
with the updates) is timed as one piece; without it, the collection, world-model and policy updates one by one.

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at commit 8a6244b,
``scripts/bench_iteration.py``: the task is this package's, built directly instead of through the registry.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict

import torch


def parse_override(default, v: str):
    """A ``KEY=VALUE`` value in the type of the cfg field's default (tuples comma-separated)."""
    if isinstance(default, bool):
        return v.lower() in ("1", "true", "yes")
    if isinstance(default, tuple):
        return tuple(type(default[0])(x) for x in v.split(","))
    return type(default)(v)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--motion-file", required=True)
    ap.add_argument("--motion-root", default="")
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="untimed iterations first (compilation, buffer fill)",
    )
    ap.add_argument("--iters", type=int, default=20, help="timed iterations")
    ap.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="runner cfg overrides, e.g. pi_batch=2048",
    )
    ap.add_argument(
        "--env-set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="motion command cfg overrides, e.g. bin_fix=true",
    )
    ap.add_argument(
        "--decimation",
        type=int,
        default=None,
        help="physics substeps per control step (task: 10, i.e. 50 Hz)",
    )
    ap.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument(
        "--profile",
        type=int,
        default=0,
        help="after the timing, profile this many more iterations (best without CUDA graphs) and print the top CUDA kernels",
    )
    args = ap.parse_args()

    from dataclasses import replace

    from myosuite.envs.myo.tasks.mimic.tracking import MIMIC_ENV_ID as ENV_ID
    from supermimic.env import TIMESTEP
    from supermimic.collector import STATS
    from supermimic.config import fast_runner_cfg
    from supermimic.env import TaskEnvCfg, make_task_env
    from supermimic.runner import SuperMimicRunner, set_lr

    from mjlab.utils.torch import configure_torch_backends

    # the training's backend flags (TF32): they are part of the compile caches' keys, so a benchmark with other flags
    # neither measures the training's kernels nor warms its caches
    configure_torch_backends()
    torch.manual_seed(0)
    task = TaskEnvCfg(
        motion_file=args.motion_file,
        motion_root=args.motion_root,
        num_envs=args.num_envs,
    )
    for kv in args.env_set:
        k, v = kv.split("=", 1)
        task = replace(task, **{k: parse_override(getattr(task, k), v)})
    if args.decimation:
        task.ctrl_dt = args.decimation * TIMESTEP
    N = args.num_envs
    env = make_task_env(task, args.device)
    rl_default = fast_runner_cfg()
    overrides = {
        k: parse_override(getattr(rl_default, k), v)
        for k, v in (kv.split("=", 1) for kv in args.set)
    }
    rl = asdict(rl_default)
    rl.update(wm_only_iters=0, **overrides)
    runner = SuperMimicRunner(env, rl, device=args.device)
    c = runner.cfg
    cuda = str(args.device).startswith("cuda")
    sync = torch.cuda.synchronize if cuda else (lambda: None)

    # learn() without its logging: the pieces of an iteration, timed one by one
    runner.init_collection()  # with collect_graph also the collector (captured during the warm-up, as in training)

    stats = {k: [] for k in STATS}
    runner.collect(max(c.warmup_steps, c.wm_window + runner.hist + 1), stats)

    for o in (runner.opt_wm, runner.opt_pi, runner.opt_v):
        set_lr(o, c.lr)
    wm_update, policy_update = runner.wm_update, runner.policy_update
    if c.cuda_graph:  # as learn(): the updates captured after three warm-up calls each (the policy's after the EMA copy)
        if runner.buffer.ptr_t is None:
            runner.buffer.use_device_ptr()
        wm_update, policy_update = runner.make_graphs()
    step_acc, t = 0.0, {"collect": 0.0, "wm": 0.0, "policy": 0.0, "overlapped": 0.0}
    n_overlapped = 0
    t0 = time.time()
    for it in range(args.warmup + args.iters):
        timed = it >= args.warmup
        if it == args.warmup:
            sync()
            if cuda:
                print(f"warm-up peak: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB allocated, "
                      f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GB reserved by torch", flush=True)  # fmt: skip
                torch.cuda.reset_peak_memory_stats()
            print(
                f"warm-up done in {time.time() - t0:.0f} s; timing {args.iters} iterations",
                flush=True,
            )
            t0 = time.time()
        step_acc += 1.0 / c.utd
        n = int(step_acc)
        step_acc -= n
        if (
            c.overlap_collect or c.overlap_wm
        ):  # the runner's iteration, as learn() runs it
            agg = {
                k: []
                for k in ("wm", "wm8", "wmt", "pi", "v", "ir", "gn", "surv", "clip")
            }
            sync()
            t1 = time.time()
            overlapped = runner.can_overlap(it) or runner.can_overlap_wm(it)
            runner.iterate(it, n, stats, agg, wm_update, policy_update)
            sync()
            if timed:
                t["overlapped"] += time.time() - t1
                n_overlapped += overlapped
            stats = {k: [] for k in stats}
            continue
        sync()
        t1 = time.time()
        runner.collect(n, stats)
        sync()
        t2 = time.time()
        for _ in range(c.wm_batches):
            wm_update()
        sync()
        t3 = time.time()
        policy_update()
        if c.empty_cache and cuda:
            torch.cuda.empty_cache()
        sync()
        t4 = time.time()
        if timed:
            t["collect"] += t2 - t1
            t["wm"] += t3 - t2
            t["policy"] += t4 - t3
        stats = {k: [] for k in stats}
    total = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
    sim_steps = sum(int(k) for k in [args.iters / c.utd])
    print(f"\n{ENV_ID}, {N} envs, {args.iters} timed iterations, overrides {overrides or 'none'}, "
          f"env overrides {args.env_set or 'none'}, device {args.device}")  # fmt: skip
    qpos = runner.uenv.sim.data.qpos._tensor
    print(
        f"simulator state finite in {int(torch.isfinite(qpos).all(-1).sum())} of {N} envs"
    )
    print(f"{'piece':10s} {'s / iteration':>14s} {'share':>7s}")
    for k, v in t.items():
        if v > 0:
            print(f"{k:10s} {v / args.iters:14.3f} {v / total:7.1%}")
    if c.overlap_collect or c.overlap_wm:
        print(
            f"{n_overlapped} of {args.iters} timed iterations ran overlapped (collection and/or world-model updates)"
        )
    if args.profile > 0 and cuda:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(args.profile):
                runner.collect(int(round(1.0 / c.utd)), stats)
                for _ in range(c.wm_batches):
                    wm_update()
                policy_update()
            sync()
        print(
            prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=30, max_name_column_width=70
            )
        )
    print(f"{'total':10s} {total / args.iters:14.3f} {'':7s}   ({args.iters / total:.3f} it/s; collection = {sim_steps / args.iters:.1f} "
          f"sim steps of {N} envs per iteration; torch peak {mem:.1f} GB)")  # fmt: skip


if __name__ == "__main__":
    main()
