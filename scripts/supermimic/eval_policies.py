# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate rsl_rl-layout actor checkpoints (PPO or SuperMimic) with one deterministic protocol.

Reports the task reward and the joint-angle / tilt errors against the clip. With
``--run-dir`` a SuperMimic run's best, final and averaged (EMA) actors are evaluated
side by side. The best one was picked on the training evaluation, so its training
score is optimistic; the default seed differs from the training evaluation's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import _musclemimic_cpu_model
from myosuite.learning.supermimic.env_io import MimicEnvIO, make_mimic_env
from myosuite.learning.supermimic.heads import MimicHeads
from myosuite.learning.supermimic.metrics import KinematicTracking
from myosuite.learning.supermimic.policy import Actor
from myosuite.learning.supermimic.trainer import evaluate_policy

# SuperMimic run outputs, in report order.
RUN_CHECKPOINTS = {
    "best (picked on the training evaluation)": "ckpt_best.pt",
    "final": "ckpt_final.pt",
    "ema": "ckpt_ema.pt",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", required=True)
    p.add_argument(
        "--fullbody-model",
        default="myo_sim",
        choices=["myo_sim", "musclemimic"],
        help="body model (must match the one the checkpoint was trained on)",
    )
    p.add_argument("checkpoints", nargs="*")
    p.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="SuperMimic run directory: adds its best, final and EMA checkpoints",
    )
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument(
        "--seed",
        type=int,
        default=2000,
        help="env seed (training evaluates with its seed + 1000)",
    )
    p.add_argument("--out", type=Path)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    from myosuite.learning.supermimic.trainer import use_fullbody_model

    use_fullbody_model(a.fullbody_model)
    runs = [(ck, "") for ck in a.checkpoints]
    for d in a.run_dir:
        runs += [
            (str(d / f), label)
            for label, f in RUN_CHECKPOINTS.items()
            if (d / f).exists()
        ]
    if not runs:
        p.error("no checkpoints given")
    env = make_mimic_env(a.clip, a.envs, a.device, seed=a.seed)
    io = MimicEnvIO(env)
    clip_qpos = MimicHeads.from_env(env, "mimic_fullbody_robot").clip.qpos
    tracking = KinematicTracking(_musclemimic_cpu_model("fullbody"), clip_qpos)
    rows = []
    for ck, label in runs:
        torch.manual_seed(a.seed)
        res = {
            "checkpoint": ck,
            "label": label,
            **evaluate_policy(
                env, Actor.from_rsl_rl(ck, a.device), a.steps, tracking, io
            ),
        }
        print(json.dumps(res), flush=True)
        rows.append(res)
    if a.out:
        a.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
