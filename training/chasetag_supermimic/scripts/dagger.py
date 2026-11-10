# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""DAgger distillation of the SuperMimic player into a 546 -> 354 FBVs student (see :mod:`chasetag_supermimic.distill`).

    python scripts/dagger.py --out <dir> --checkpoint <tracker model_*.pt> [--rounds 8] [--bouts 480] [--canon]

Round r collects ``--bouts`` bouts in which the labelled players execute the student of round r - 1 with
probability 1 - beta_r (round 0: the teacher only), then fits the student on the data of all rounds (warm start).
Opponents per bout cycle through: a second teacher (labelled too), the shipped baseline, idle (as runner); the
labelled teacher plays either side. Writes ``student_r<r>.pt`` (ActorCritic state dict), ``student_r<r>_full.pt``
(with the normalisation and the raw weights, for warm starts), ``student_r<r>.onnx`` and ``rounds.jsonl``; the bout
data goes to ``<out>/data/r<r>`` (float16, about 1 GB per round of 480 bouts).

With ``--canon`` the student reads yaw-canonicalised observations and its ONNX canonicalises the raw observation;
``--canon-from k`` starts the canonicalised student fresh at round k (earlier rounds trained a plain one).
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
    """One bout in a worker process (the env, the library and the players are built once per worker)."""
    i, out, ck, root, student, canon, beta, noise, every, takeover_z = args
    import torch  # noqa: PLC0415

    torch.set_num_threads(1)
    from chasetag_supermimic.distill import StudentPlayer, collect_bout  # noqa: PLC0415
    from chasetag_supermimic.library import load_library  # noqa: PLC0415
    from chasetag_supermimic.match import IdlePlayer, OnnxPlayer, SmPlayer, make  # noqa: PLC0415

    if "env" not in _c:
        _c["env"], lib = make(), load_library(root)
        _c["ta"], _c["tb"] = SmPlayer(ck, lib), SmPlayer(ck, lib)
        _c["base"], _c["idle"] = OnnxPlayer(), IdlePlayer()
        _c["student"] = StudentPlayer(student, canon=canon) if student else None
    kind = ("teacher", "teacher", "base", "idle")[i % 4]
    task = "CHASE" if (i // 4) % 2 == 0 else "EVADE"
    swap = (
        kind != "teacher" and (i // 16) % 2 == 1
    )  # the labelled teacher plays either side
    if kind == "idle":  # an idle chaser falls at once: the teacher chases
        task = "EVADE" if swap else "CHASE"
    tz = (
        takeover_z if (i // 8) % 2 == 0 else 0.0
    )  # half the bouts without takeover: the student's own falls
    r = collect_bout(_c["env"], _c["ta"], _c["tb"] if kind == "teacher" else None,
                     _c[kind] if kind != "teacher" else None, _c["student"], beta, 5000 + i, task, noise,
                     every=every, takeover_z=tz, swap=swap)  # fmt: skip
    np.savez_compressed(
        Path(out) / f"b{i:05d}.npz",
        obs=r["obs"].astype(np.float16),
        act=r["act"].astype(np.float16),
    )
    return {"i": i, "kind": kind, "task": task, "end": r["end"], "t": r["t"], "n": len(r["obs"]),
            "student": r["student"], "fell": r["fell"], "took": r["took"], "takeover": tz > 0}  # fmt: skip


def stats(rows: list[dict]) -> dict:
    """Fall times of the agents that executed the student (bouts without takeover), takeovers, and the bout ends."""
    st = [r["fell"][a] for r in rows if not r["takeover"] for a in r["student"]]
    tk = [r["took"][a] for r in rows if r["takeover"] for a in r["student"]]
    falls = [t for t in st if t is not None]
    return {"bouts": len(rows), "samples": int(sum(r["n"] for r in rows)), "student_agents": len(st),
            "student_falls": len(falls), "student_fall_median_s": float(np.median(falls)) if falls else None,
            "student_up_mean_s": float(np.mean([t if t is not None else 10.0 for t in st])) if st else None,
            "takeovers": f"{sum(t is not None for t in tk)}/{len(tk)}",
            "ends": {e: sum(r["end"] == e for r in rows) for e in ("tag", "time_up", "chaser_fall", "stopped")}}  # fmt: skip


def main() -> None:
    import torch  # noqa: PLC0415

    from chasetag_supermimic.augment import half_turn  # noqa: PLC0415
    from chasetag_supermimic.distill import (
        canonicalise_inplace,
        export_canon_onnx,
        load_data,
        train,
    )  # noqa: PLC0415
    from myosuite.integrations.musclemimic.fbvs_player import export_fbvs_onnx  # noqa: PLC0415

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
    ap.add_argument("--bouts", type=int, default=480)
    ap.add_argument("--betas", default="1,0.5,0.3,0.2,0.1,0.1,0,0")
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="bout processes (each builds the env, the library and two trackers)",
    )
    ap.add_argument("--noise", type=float, default=0.03)
    ap.add_argument("--every", type=int, default=2, help="store every n-th step")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device of the fits",
    )
    ap.add_argument("--start-round", type=int, default=0)
    ap.add_argument(
        "--augment",
        action="store_true",
        help="add every sample's half-turn copy (chasetag_supermimic.augment)",
    )
    ap.add_argument(
        "--canon",
        action="store_true",
        help="train on yaw-canonicalised observations; the ONNX canonicalises",
    )
    ap.add_argument(
        "--canon-from",
        type=int,
        default=-1,
        help="first round trained canonicalised (it starts fresh)",
    )
    ap.add_argument("--takeover-z", type=float, default=0.0, help="pelvis height (m) below which the teacher takes "
                    "over a student (in half the bouts; 0: never)")  # fmt: skip
    a = ap.parse_args()
    torch.set_num_threads(8)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    betas = [float(b) for b in a.betas.split(",")]
    canon_at = lambda r: a.canon and r >= a.canon_from  # noqa: E731
    init = None
    if a.start_round > 0 and not (
        canon_at(a.start_round) and a.start_round == a.canon_from
    ):
        init = torch.load(
            out / f"student_r{a.start_round - 1}_full.pt", weights_only=False
        )
    for r in range(a.start_round, a.rounds):
        d = out / "data" / f"r{r}"
        d.mkdir(parents=True, exist_ok=True)
        beta = betas[min(r, len(betas) - 1)] if r > 0 else 1.0
        student = str(out / f"student_r{r - 1}.pt") if r > 0 else ""
        jobs = [(r * 100000 + i, str(d), a.checkpoint, a.root, student, canon_at(r - 1), beta, a.noise, a.every,
                 a.takeover_z) for i in range(a.bouts)]  # fmt: skip
        with ProcessPoolExecutor(a.workers) as pool:
            rows = list(pool.map(job, jobs, chunksize=1))
        X, Y = load_data([out / "data" / f"r{k}" for k in range(r + 1)])
        if a.augment:
            X, Y = np.concatenate((X, half_turn(X))), np.concatenate((Y, Y))
        if canon_at(r):
            canonicalise_inplace(X)
            if r == a.canon_from:
                init = None  # the earlier students saw raw observations
        print(
            f"round {r}: beta {beta}, {len(X)} samples, training on {a.device}",
            flush=True,
        )
        res = train(X, Y, epochs=a.epochs, batch=a.batch, lr=a.lr, device=a.device, init=init,
                    log=lambda s: print(s, flush=True))  # fmt: skip
        del X, Y
        res["canon"] = canon_at(r)
        torch.save(res["state"], out / f"student_r{r}.pt")
        torch.save(res, out / f"student_r{r}_full.pt")
        if canon_at(r):
            export_canon_onnx(res["state"], out / f"student_r{r}.onnx")
        else:
            export_fbvs_onnx(out / f"student_r{r}.pt", out / f"student_r{r}.onnx")
        init = res
        rec = {
            "round": r,
            "beta": beta,
            "val_mse": res["val_mse"],
            "canon": res["canon"],
            **stats(rows),
        }
        with open(out / "rounds.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
