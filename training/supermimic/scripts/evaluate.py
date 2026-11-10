# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate fast-runner checkpoints of ``myoMimic-v0`` on held-out clips.

Every clip plays once from its first frame with the deployed (averaged) policy, at the control rate and goal stride
the checkpoint was trained with. Reports the normalized return, the failed clips (those not tracked to the end), the
joint-angle errors and the MPJPE (see ``supermimic.evaluate``).

    # one checkpoint, optionally with test-time action refinement through the world model
    python training/supermimic/scripts/evaluate.py --checkpoint <run>/model_004000.pt --motion-file <kit test list> \
        --motion-root <dataset root> [--refine 5]
    # every checkpoint of a run, appended to <run>/eval.csv (with the training clock of each)
    python training/supermimic/scripts/evaluate.py --run-dir <run> --motion-file <kit test list> --motion-root <root> --batch 8
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import tyro


@dataclass
class EvalArgs:
    motion_file: str = ""
    """Evaluation clips (default: the benchmark's 108 held-out KIT clips, ``supermimic/data/kit_test.txt``)."""
    motion_root: str = ""
    checkpoint: str = ""
    run_dir: str = ""
    batch: int = 1
    """--run-dir: checkpoints evaluated at once."""
    watch: bool = False
    """--run-dir: keep polling for new checkpoints."""
    skip_before_min: float = 0.0
    csv: str = "eval.csv"
    refine: int = 0
    """Test-time action refinement: gradient steps per control step (0: off)."""
    eta: float = 0.01
    horizon: int = 0
    """Refinement horizon (0: the training horizon)."""
    out: str = ""
    """Write the results of --checkpoint as JSON."""
    record: str = ""
    """Save the rollouts of --checkpoint (npz)."""
    device: str = "cuda:0"


def main(a: EvalArgs) -> None:
    from supermimic.evaluate import (
        build,
        checkpoint_settings,
        evaluate,
        evaluate_run,
    )
    from supermimic.refine import Refiner, run

    from supermimic.env import held_out_clips

    motion, root = (a.motion_file, a.motion_root) if a.motion_file else held_out_clips()
    a.motion_root = a.motion_root or root
    if a.run_dir:
        evaluate_run(
            a.run_dir,
            motion,
            a.device,
            a.motion_root,
            a.batch,
            a.csv,
            a.skip_before_min,
            a.watch,
        )
        return
    env, runner = build(
        motion,
        a.device,
        motion_root=a.motion_root,
        settings=checkpoint_settings(a.checkpoint),
    )
    ck = runner.load(a.checkpoint, load_cfg={"actor": True})
    if a.refine > 0:
        cuda = a.device.startswith("cuda")
        refiner = Refiner(
            runner, ck, a.refine, a.eta, a.horizon, True, compile_=cuda, graph=cuda
        )
        res = run(env, runner, refiner, record=a.record or None)
    else:
        res = evaluate(env, runner, a.record or None)
    summary = {
        k: v
        for k, v in res.items()
        if not k.startswith("per_clip") and k not in ("alive_s", "clip_s")
    }
    print(
        json.dumps(
            {"checkpoint": a.checkpoint, "refine": a.refine, **summary}, indent=1
        )
    )
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(
                {"checkpoint": a.checkpoint, "refine": a.refine, **res}, fh, indent=1
            )


if __name__ == "__main__":
    main(tyro.cli(EvalArgs))
