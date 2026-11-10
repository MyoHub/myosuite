# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Port checks of the tracker on a myoSuperMimic checkpoint (training/supermimic).

    python scripts/check_port.py <checkpoint> [--clips 6]

1. Body: the FBVs players' joints and muscles (prefix stripped) are the tracker env's, in the same order and with
   the same parameters.
2. Observation and goal: the CPU tracker's decoder observation and the steering library's goal (no warp) match the
   CPU myoMimic-v0 env's at the start of held-out clips.
3. Closed loop: the tracker follows held-out clips on the CPU env (failures, mean tracking error).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from myosuite import make_env
from myosuite.core.config import EnvConfig
from myosuite.envs.myo.tasks.mimic.tracking import MIMIC_ENV_ID
from myosuite.envs.myo.tasks.mimic.tracking.models import root_offset
from supermimic.env import settings_cfg, task_overrides, held_out_clips

from chasetag_teacher.reference import ClipLibrary, SteeredReference
from chasetag_teacher.tracker import CpuTracker, check_same_body, load_policy


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoint")
    ap.add_argument("--clips", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(4)
    _, settings = load_policy(a.checkpoint)
    held_out, root = (
        held_out_clips()
    )  # the benchmark's held-out clips (training/supermimic)
    cfg = settings_cfg(
        settings, motion_file=held_out, motion_root=root, sampling_mode="start"
    )
    env = make_env(EnvConfig(MIMIC_ENV_ID, ctrl_dt=cfg.ctrl_dt,
                             task_kwargs={"preset": "supermimic", "overrides": task_overrides(cfg)}))  # fmt: skip
    u = env.unwrapped
    tj = check_same_body(
        make_env("myoChallengeChaseTagFBVs-v0").unwrapped.model, u.model
    )
    print(
        f"body: {len(tj)} joints, {u.model.nu} muscles, same order and parameters for a0_/a1_"
    )
    tr = CpuTracker(a.checkpoint, u.model, "")
    lib = ClipLibrary(u.motion, root_offset(u.model))
    ref = SteeredReference(lib)
    n_obs = settings["obs_dim"]
    picks = np.linspace(0, len(u.motion.names) - 1, a.clips).astype(int)
    fails, errs = 0, []
    for c in picks:
        obs, _ = env.reset(seed=int(c), options={"clip": int(c)})
        tr.reset()
        mine = tr.observation(tr.state(u.data))[0].numpy()
        ref.f, ref.off, ref.omega = u.frame, 0.0, 0.0
        g = ref.goal()[0].numpy()
        do, dg = np.abs(mine - obs[:n_obs]), np.abs(g - obs[n_obs:])
        print(f"clip {c} {u.motion.names[c].split('/')[-1]}: obs max|d| {do.max():.2e} (at {int(do.argmax())}), "
              f"goal max|d| {dg.max():.2e}")  # fmt: skip
        steps, failed, e = 0, False, []
        while True:
            with torch.no_grad():
                goal = torch.as_tensor(obs[n_obs:], dtype=torch.float32)[None]
                act = tr.act(u.data, goal)
            obs, _, term, trunc, info = env.step(act)
            steps += 1
            e.append(info["dev_rel_site"])
            if term or trunc:
                failed = bool(term)
                break
        fails += failed
        errs.append(np.nanmean(e))
        print(
            f"  closed loop {steps} steps, failed={failed}, mean dev {np.nanmean(e):.4f}"
        )
    print(f"closed loop: {fails}/{len(picks)} failed, mean dev {np.mean(errs):.4f}; dev keys "
          f"{[k for k in info if k.startswith('dev_')]}")  # fmt: skip


if __name__ == "__main__":
    main()
