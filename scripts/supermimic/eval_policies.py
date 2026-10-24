# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate rsl_rl-layout actor checkpoints (PPO or SuperMimic) with one deterministic protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from myosuite.learning.supermimic.env_io import make_mimic_env
from myosuite.learning.supermimic.policy import Actor
from myosuite.learning.supermimic.trainer import evaluate_policy


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", required=True)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", type=Path)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    env = make_mimic_env(a.clip, a.envs, a.device, seed=a.seed)
    rows = []
    for ck in a.checkpoints:
        torch.manual_seed(a.seed)
        res = {
            "checkpoint": ck,
            **evaluate_policy(env, Actor.from_rsl_rl(ck, a.device), a.steps),
        }
        print(json.dumps(res), flush=True)
        rows.append(res)
    if a.out:
        a.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
