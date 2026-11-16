# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""DAgger distillation of the steered SuperMimic teacher into a directional walker (528 -> 354, tutorial 5.4).

    python scripts/dagger_directional.py --out <dir> --checkpoint <tracker model_*.pt> [--rounds 8] [--episodes 480]

Round r plays ``--episodes`` episodes of 10 s on one ``myofullbody`` (random start facing, a random heading and one
or two heading changes; :func:`chasetag_teacher.directional.collect_episode`), executing the student of round r - 1
with probability 1 - beta_r (round 0: the teacher only), then fits the student on the data of all rounds (warm
start). Writes ``walker_r<r>.pt`` (ActorCritic state dict, normalisation folded in), ``walker_r<r>_full.pt`` (for warm
starts), ``walker_r<r>_chasetag.onnx`` (the walker as a 546 -> 354 ChaseTag player) and ``rounds.jsonl``, and prints
tutorial 5.4's evaluation of each student (8 headings x ``--eval-seeds`` x 200 steps).
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_c: dict = {}


def job(args: tuple) -> dict:
    """One episode in a worker process (scene, library and teacher are built once per worker)."""
    i, out, ck, root, student, beta, noise, takeover_z = args
    import torch  # noqa: PLC0415

    torch.set_num_threads(1)
    from chasetag_teacher.directional import (  # noqa: PLC0415
        DirectionalScene,
        EpisodeCfg,
        HeadingTeacher,
        collect_episode,
        load_walker,
        torch_policy,
    )
    from chasetag_teacher.library import load_library  # noqa: PLC0415

    if "scene" not in _c:
        _c["scene"] = DirectionalScene()
        _c["teacher"] = HeadingTeacher(ck, load_library(root), _c["scene"])
    if _c.get("student_path") != student:
        _c["student"] = torch_policy(load_walker(student)) if student else None
        _c["student_path"] = student
    tz = (
        takeover_z if (i // 2) % 2 == 0 else 0.0
    )  # half the episodes without takeover: the student's own falls
    r = collect_episode(_c["scene"], _c["teacher"], _c["student"], beta, i,
                        EpisodeCfg(noise=noise, takeover_z=tz))  # fmt: skip
    np.savez_compressed(
        Path(out) / f"b{i:07d}.npz",
        obs=r["obs"].astype(np.float16),
        act=r["act"].astype(np.float16),
    )
    return {
        "i": i,
        "n": len(r["obs"]),
        "fell": r["fell"],
        "took": r["took"],
        "student": r["student"],
        "takeover": tz > 0,
    }


def eval_job(args: tuple) -> dict:
    """Tutorial 5.4's evaluation of one heading in a worker process."""
    path, label, seeds, steps = args
    import torch  # noqa: PLC0415

    torch.set_num_threads(1)
    from chasetag_teacher.directional import evaluate, load_walker, torch_policy  # noqa: PLC0415

    res = evaluate(torch_policy(load_walker(path)), [label], seeds, steps)[label]
    return {"label": label, "steps": steps, "survival": res.survival, "steps_alive": res.steps_alive,
            "alignment": res.alignment}  # fmt: skip


def stats(rows: list[dict]) -> dict:
    """Falls of the student episodes without takeover, takeovers."""
    st = [r["fell"] for r in rows if r["student"] and not r["takeover"]]
    tk = [r["took"] for r in rows if r["student"] and r["takeover"]]
    falls = [t for t in st if t is not None]
    return {"episodes": len(rows), "samples": int(sum(r["n"] for r in rows)), "student_episodes": len(st),
            "student_falls": len(falls), "student_fall_median_s": float(np.median(falls)) if falls else None,
            "takeovers": f"{sum(t is not None for t in tk)}/{len(tk)}"}  # fmt: skip


def main() -> None:
    import torch  # noqa: PLC0415

    from chasetag_teacher.directional import DIR_OBS_DIM, export_chasetag_onnx  # noqa: PLC0415
    from chasetag_teacher.distill import load_data, train  # noqa: PLC0415

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--checkpoint", required=True, help="the myoSuperMimic tracker (model_*.pt)"
    )
    ap.add_argument(
        "--root",
        default=None,
        help="folder of retargeted KIT clips (default: $MYOSUITE_KIT_ROOT)",
    )
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=480)
    ap.add_argument("--betas", default="1,0.3,0.1,0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.03)
    ap.add_argument("--takeover-z", type=float, default=0.8, help="root height (m) below which the teacher takes "
                    "over a student (in half the episodes; 0: never)")  # fmt: skip
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--start-round", type=int, default=0)
    ap.add_argument(
        "--eval-seeds",
        type=int,
        default=10,
        help="tutorial 5.4 evaluation per round (0: none)",
    )
    a = ap.parse_args()
    torch.set_num_threads(8)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    betas = [float(b) for b in a.betas.split(",")]
    init = (
        torch.load(out / f"walker_r{a.start_round - 1}_full.pt", weights_only=True)
        if a.start_round
        else None
    )
    for r in range(a.start_round, a.rounds):
        d = out / "data" / f"r{r}"
        d.mkdir(parents=True, exist_ok=True)
        beta = betas[min(r, len(betas) - 1)] if r > 0 else 1.0
        student = str(out / f"walker_r{r - 1}.pt") if r > 0 else ""
        jobs = [(r * 100000 + i, str(d), a.checkpoint, a.root, student, beta, a.noise, a.takeover_z)
                for i in range(a.episodes)]  # fmt: skip
        with ProcessPoolExecutor(a.workers) as pool:
            rows = list(pool.map(job, jobs, chunksize=1))
        X, Y, held = load_data(
            [out / "data" / f"r{k}" for k in range(r + 1)], obs_dim=DIR_OBS_DIM
        )
        print(
            f"round {r}: beta {beta}, {len(X)} samples, training on {a.device}",
            flush=True,
        )
        res = train(X, Y, epochs=a.epochs, batch=a.batch, lr=a.lr, device=a.device, init=init,
                    log=lambda s: print(s, flush=True), obs_dim=DIR_OBS_DIM, held_out=held)  # fmt: skip
        del X, Y
        torch.save(res["state"], out / f"walker_r{r}.pt")
        torch.save(res, out / f"walker_r{r}_full.pt")
        export_chasetag_onnx(
            out / f"walker_r{r}.pt", out / f"walker_r{r}_chasetag.onnx"
        )
        init = res
        rec = {"round": r, "beta": beta, "val_mse": res["val_mse"], **stats(rows)}
        if a.eval_seeds:
            from chasetag_teacher.directional import COMPASS  # noqa: PLC0415

            with ProcessPoolExecutor(min(a.workers, 8)) as pool:
                ev = list(pool.map(eval_job, [(str(out / f"walker_r{r}.pt"), lab, a.eval_seeds, 200)
                                              for lab in COMPASS]))  # fmt: skip
            rec["eval200"] = {
                e["label"]: [e["survival"], e["steps_alive"], round(e["alignment"], 3)]
                for e in ev
            }
        with open(out / "rounds.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
