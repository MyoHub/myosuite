# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate ChaseTag FBVs players: chasing an idle player, head-to-head in both roles, mirror matches.

    python scripts/evaluate.py --out results.json --sm teacher=<tracker model_*.pt> --onnx s21=student_r21.onnx \
        --matches teacher:base,base:teacher,teacher:idle,s21:base,base:s21,s21:idle [--episodes 48] [--workers 8]

A player spec is the name of a ``--sm`` player (SuperMimic tracker + steering on the KIT clip library of
``--root``, see :func:`chasetag_supermimic.library.load_library`), the name of an ``--onnx`` player, ``onnx:<path>``,
``base`` (the shipped baseline ONNX) or ``idle``. A match is ``chaser:runner``; it runs with the chaser as agent_0
(side A) and as agent_1 (side B), and episode i uses seed 100 + i. Prints per match and per player (tags of an idle
player, head-to-head against base, mean time up; :func:`chasetag_supermimic.match.player_scores`).
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_cache: dict = {}


def _player(spec: str, sms: dict[str, tuple[str, str]], root: str | None):
    """The player of a spec (cached libraries)."""
    from chasetag_supermimic.match import IdlePlayer, OnnxPlayer, SmPlayer  # noqa: PLC0415

    if spec == "idle":
        return IdlePlayer()
    if spec == "base":
        return OnnxPlayer()
    if spec in sms and sms[spec][1] == "onnx":
        return OnnxPlayer(sms[spec][0])
    if spec.startswith("onnx:"):
        return OnnxPlayer(spec[5:])
    if spec in sms:
        from chasetag_supermimic.library import load_library  # noqa: PLC0415

        if "lib" not in _cache:
            _cache["lib"] = load_library(root)
        return SmPlayer(sms[spec][0], _cache["lib"])
    raise ValueError(spec)


def bout(job: tuple) -> dict:
    """One bout in a worker process."""
    import torch  # noqa: PLC0415

    torch.set_num_threads(1)
    from chasetag_supermimic.match import make, play  # noqa: PLC0415

    chaser, runner, chaser_is_a, seed, sms, root = job
    if "env" not in _cache:
        _cache["env"] = make()
    a, b = (chaser, runner) if chaser_is_a else (runner, chaser)
    players = {"agent_0": _player(a, sms, root), "agent_1": _player(b, sms, root)}
    r = play(_cache["env"], players, seed, "CHASE" if chaser_is_a else "EVADE")
    chaser_agent = "agent_0" if chaser_is_a else "agent_1"
    runner_agent = "agent_1" if chaser_is_a else "agent_0"
    return {"chaser": chaser, "runner": runner, "chaser_is_a": chaser_is_a, "seed": seed, "end": r.end, "t": r.t,
            "chaser_fell": r.fell[chaser_agent], "runner_fell": r.fell[runner_agent]}  # fmt: skip


def summarize(rows: list[dict]) -> list[dict]:
    """Per match and side: tags, chaser falls, time-ups, bout and tag times, fall times."""
    out = []
    keys = sorted({(r["chaser"], r["runner"], r["chaser_is_a"]) for r in rows}, key=str)
    for c, rn, a in keys:
        rs = [
            r
            for r in rows
            if (r["chaser"], r["runner"], r["chaser_is_a"]) == (c, rn, a)
        ]
        tags = [r for r in rs if r["end"] == "tag"]
        cf = [r["chaser_fell"] for r in rs if r["chaser_fell"] is not None]
        rf = [r["runner_fell"] for r in rs if r["runner_fell"] is not None]
        out.append({"chaser": c, "runner": rn, "chaser_side": "A" if a else "B", "n": len(rs), "tags": len(tags),
                    "chaser_falls": sum(r["end"] == "chaser_fall" for r in rs),
                    "time_up": sum(r["end"] == "time_up" for r in rs),
                    "mean_len_s": round(float(np.mean([r["t"] for r in rs])), 2),
                    "tag_time_s": round(float(np.mean([r["t"] for r in tags])), 2) if tags else None,
                    "chaser_fell_median_s": round(float(np.median(cf)), 2) if cf else None,
                    "runner_fell_n": len(rf), "runner_fell_median_s": round(float(np.median(rf)), 2) if rf else None})  # fmt: skip
    return out


def main() -> None:
    from chasetag_supermimic.match import player_scores  # noqa: PLC0415

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--sm", action="append", default=[], help="name=<tracker checkpoint>"
    )
    ap.add_argument(
        "--root",
        default=None,
        help="folder of retargeted KIT clips (default: $MYOSUITE_KIT_ROOT)",
    )
    ap.add_argument("--onnx", action="append", default=[], help="name=<file.onnx>")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--matches", required=True, help="chaser:runner pairs")
    args = ap.parse_args()
    sms = {n: (v, "sm") for n, v in (x.split("=", 1) for x in args.sm)}
    sms |= {n: (v, "onnx") for n, v in (x.split("=", 1) for x in args.onnx)}
    jobs = [(c, r, side, 100 + i, sms, args.root) for m in args.matches.split(",") for c, r in [m.split(":")]
            for side in (True, False) for i in range(args.episodes)]  # fmt: skip
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(bout, jobs, chunksize=1))
    summary = summarize(rows)
    names = sorted(
        ({r["chaser"] for r in rows} | {r["runner"] for r in rows}) - {"idle", "base"}
    )
    scores = {n: player_scores(rows, n) for n in names}
    with open(args.out, "w") as f:
        json.dump(
            {"players": sms, "summary": summary, "scores": scores, "rows": rows},
            f,
            indent=1,
        )
    for s in summary:
        print(json.dumps(s))
    for n, s in scores.items():
        print(
            f"{n}: tags idle {s['tag_idle']:.0f}%, head-to-head vs base {s['head_to_head']:.0f}%, "
            f"time up {s['time_up_s']:.2f} s"
        )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
