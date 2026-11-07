# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""World-model drift against the simulator under a SuperMimic checkpoint's own policy.

Records the deterministic policy in the mjlab env from clip starts (RSI), then drives
the checkpoint's world model from the same states: in closed loop (the policy acts on
the world model's states, as in imagination) and in replay (recorded controls). Reports
the mean tracked-site distance and the imagined vs simulated reward at chosen steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import _musclemimic_cpu_model
from myosuite.learning.supermimic.env_io import MimicEnvIO, make_mimic_env, record
from myosuite.learning.supermimic.heads import MimicHeads
from myosuite.learning.supermimic.metrics import world_model_drift
from myosuite.learning.supermimic.policy import Actor
from myosuite.learning.supermimic.state import StateSpec, forward_axis_from_quat
from myosuite.learning.supermimic.world_model import WorldModel, WorldModelCfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", required=True)
    p.add_argument(
        "--fullbody-model",
        default="myo_sim",
        choices=["myo_sim", "musclemimic"],
        help="body model (must match the one the checkpoint was trained on)",
    )
    p.add_argument(
        "--checkpoint", required=True, help="trainer checkpoint (world model + actor)"
    )
    p.add_argument(
        "--actor",
        help="other actor checkpoint, e.g. ckpt_ema.pt (default: --checkpoint)",
    )
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument(
        "--windows", type=int, default=3, help="windows per env, back to back"
    )
    p.add_argument("--report-steps", type=int, nargs="+", default=[8, 32, 64])
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    from myosuite.learning.supermimic.trainer import use_fullbody_model

    use_fullbody_model(a.fullbody_model)
    dev = a.device
    torch.manual_seed(a.seed)
    ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    env = make_mimic_env(a.clip, a.envs, dev, seed=a.seed)
    env.reset()
    io = MimicEnvIO(env)
    heads = MimicHeads.from_env(env, "mimic_fullbody_robot")
    model = _musclemimic_cpu_model("fullbody")
    spec = StateSpec(
        model.nq,
        model.nv,
        model.na,
        heads.ctrl_dt,
        forward_axis_from_quat(heads.clip.qpos[0, 3:7].cpu()),
    ).to(dev)
    wm_cfg = dict(ck["cfg"]["wm"])
    wm_cfg["hidden"] = tuple(wm_cfg["hidden"])
    wm = WorldModel(
        model, spec, env.cfg.decimation, env.physics_dt, WorldModelCfg(**wm_cfg)
    ).to(dev)
    wm.load_state_dict(ck["wm"])
    actor = Actor.from_rsl_rl(a.actor or a.checkpoint, dev)
    ro = record(io, actor, a.horizon * a.windows, store_device=dev)
    res = {
        "checkpoint": a.checkpoint,
        "actor": a.actor or a.checkpoint,
        "envs": a.envs,
        "horizon": a.horizon,
        **world_model_drift(
            wm, heads, actor, ro, a.horizon, report_steps=tuple(a.report_steps)
        ),
    }
    print(json.dumps(res), flush=True)
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
