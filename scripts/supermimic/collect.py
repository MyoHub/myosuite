# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Record Mimic rollouts of an rsl_rl checkpoint with per-env action noise for world-model fitting."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from myosuite.learning.supermimic.env_io import MimicEnvIO, make_mimic_env, record
from myosuite.learning.supermimic.policy import Actor


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument(
        "--noise",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5],
        help="control noise stds, assigned round-robin to envs",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    env = make_mimic_env(a.clip, a.num_envs, a.device, seed=a.seed)
    env.reset()
    io = MimicEnvIO(env)
    actor = Actor.from_rsl_rl(a.checkpoint, a.device)
    levels = torch.tensor(a.noise, device=a.device)
    noise = levels[torch.arange(a.num_envs, device=a.device) % len(levels)][:, None]
    t0 = time.time()
    with torch.no_grad():
        ro = record(io, actor, a.steps, noise)
    dt = time.time() - t0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ro.save(a.out)
    torch.save(
        {"noise": noise.squeeze(-1).cpu(), "levels": a.noise},
        a.out.with_suffix(".meta.pt"),
    )
    r = (
        ro.reward.view(a.steps, -1, len(levels)).mean((0, 1))
        if a.num_envs % len(levels) == 0
        else ro.reward.mean()
    )
    print(
        f"recorded {a.steps} x {a.num_envs} in {dt:.0f} s ({a.steps * a.num_envs / dt:.0f} steps/s); "
        f"mean reward per noise level {r.tolist()}; terminated {int(ro.terminated.sum())}"
    )


if __name__ == "__main__":
    main()
