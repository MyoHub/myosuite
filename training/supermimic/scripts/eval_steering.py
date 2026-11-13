# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Score a myoSuperMimic tracker on directional walking and ChaseTag (CPU), one checkpoint or a whole run.

    # one checkpoint: tutorial 5.4's directional table, turns, and the FBVs matches against the published baseline
    python training/supermimic/scripts/eval_steering.py --checkpoint logs/.../model_013993.pt --turns 90 180 \
        --baseline-onnx hf --out results.json
    # training curves: every 4th checkpoint of a run, one CSV row each
    python training/supermimic/scripts/eval_steering.py --run logs/.../2026-11-08_16-00-00 --every 4 --out curves.csv

See :mod:`supermimic.steer_eval` for the protocols.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from supermimic.env import held_out_clips
from supermimic.steer_eval import (
    BASELINE_ONNX,
    BASELINE_REVISION,
    COMPASS,
    ChaseTagCfg,
    DirectionalCfg,
    default_walk_clip,
    directional_host,
    evaluate_chasetag,
    evaluate_directional,
    evaluate_run,
    evaluate_tracking,
    prepare_walk,
)
from supermimic.steering import TrackerPolicy


def _print_tables(directional: dict | None, chasetag: dict | None) -> None:
    if directional:
        print(f"{'Dir':>4}  {'Surv':>7}  {'MeanSteps':>9}  {'Alignment':>9}")
        for label, v in directional["per_heading"].items():
            print(
                f"{label:>4}  {v['surv']:>7.0%}  {v['mean_steps_alive']:>9.1f}  {v['align']:>+9.3f}"
            )
        for turn, t in directional["summary"].get("turns", {}).items():
            print(
                f"turn {turn:+.0f} deg: survived {t['surv']:.0%}, aligned within 15 deg {t['aligned']:.0%} "
                f"(median {t['align_time_s']:.2f} s), final heading error {t['final_err_deg']:.1f} deg"
            )
    if chasetag:
        print(
            f"{'Pairing / tracker side / task':<30} {'Tags':>6} {'Chaser falls':>13} {'Mean length':>12} "
            f"{'Chaser fell at':>15} {'Runner fell at':>15}"
        )
        for key, s in chasetag["summary"].items():
            print(
                f"{key:<30} {s['tags']:>3}/{s['n']:<2} {s['chaser_falls']:>10}/{s['n']:<2} {s['mean_length_s']:>10.1f} s "
                f"{s['chaser_fell_at_s']:>13.1f} s {s['runner_fell_at_s']:>13.1f} s"
            )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="one model_*.pt")
    src.add_argument(
        "--run", help="a run directory: every --every-th checkpoint, one CSV row each"
    )
    p.add_argument("--every", type=int, default=1)
    p.add_argument(
        "--walk",
        help="walking clip (default: KIT 167 walking_medium06 from kit_root or amathislab/musclemimic-retargeted)",
    )
    p.add_argument(
        "--loop-seconds", type=float, default=15.0, help="length of the looped walk"
    )
    p.add_argument("--no-directional", action="store_true")
    p.add_argument(
        "--headings", nargs="+", default=list(COMPASS), choices=list(COMPASS)
    )
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument(
        "--steps",
        type=int,
        default=200,
        help="env steps per directional episode (100 Hz)",
    )
    p.add_argument(
        "--turns",
        type=float,
        nargs="*",
        default=[],
        help="heading changes (deg) after --turn-at",
    )
    p.add_argument("--turn-at", type=float, default=2.0)
    p.add_argument("--turn-steps", type=int, default=600)
    p.add_argument("--no-chasetag", action="store_true")
    p.add_argument(
        "--episodes",
        type=int,
        default=12,
        help="ChaseTag matches per pairing, side and role",
    )
    p.add_argument(
        "--pairings",
        nargs="+",
        default=["still", "self", "onnx"],
        choices=["still", "self", "onnx"],
    )
    p.add_argument(
        "--baseline-onnx",
        help="ONNX opponent; 'hf': the published onnx/chasetag_fbvs_baseline.onnx",
    )
    p.add_argument(
        "--track",
        help="sim-to-sim check: a clip list (or 'held_out' for supermimic/data/kit_test.txt) tracked without "
        "steering in the directional env's body",
    )
    p.add_argument(
        "--motion-root", default="", help="the directory --track's paths start from"
    )
    p.add_argument(
        "--max-rate", type=float, default=1.0, help="largest steering turn rate (rad/s)"
    )
    p.add_argument(
        "--out", required=True, help=".json (one checkpoint) or .csv (a run)"
    )
    args = p.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    walk = out.with_name(out.stem + "_walk.npz")
    cycle = prepare_walk(
        args.walk or default_walk_clip(), walk, seconds=args.loop_seconds
    )
    print(
        f"looped walk: cycle {cycle['period_s']:.2f} s from {cycle['i'] / cycle['fps']:.2f} s, match cost "
        f"{cycle['cost']:.2e}, {cycle['turn_deg']:+.1f} deg and {cycle['speed']:.2f} m/s per cycle",
        flush=True,
    )
    onnx = args.baseline_onnx
    if onnx == "hf":
        from myosuite.core.hf_io import download_baseline_file  # noqa: PLC0415

        path = download_baseline_file(BASELINE_ONNX, revision=BASELINE_REVISION)
        if path is None:
            raise SystemExit(
                f"could not download {BASELINE_ONNX} at {BASELINE_REVISION}"
            )
        onnx = str(path)
    directional = (
        None
        if args.no_directional
        else DirectionalCfg(
            headings=tuple(args.headings),
            seeds=args.seeds,
            steps=args.steps,
            turns_deg=tuple(args.turns),
            turn_at_s=args.turn_at,
            turn_steps=args.turn_steps,
            max_rate=args.max_rate,
            first_s=cycle["i"] / cycle["fps"],
        )
    )
    chasetag = (
        None
        if args.no_chasetag
        else ChaseTagCfg(
            episodes=args.episodes,
            max_rate=args.max_rate,
            baseline_onnx=onnx,
            pairings=tuple(args.pairings),
        )
    )
    track: tuple[str, str] | None = None
    if args.track:
        if args.track == "held_out":
            t_file, t_root = held_out_clips()
            track = (t_file, args.motion_root or t_root)
        else:
            track = (args.track, args.motion_root)
    if args.run:
        evaluate_run(args.run, walk, out, args.every, directional, chasetag, track)
        return
    tracker = TrackerPolicy(args.checkpoint, walk)
    t = None
    if track:
        t = evaluate_tracking(
            TrackerPolicy(args.checkpoint, track[0], motion_root=track[1]),
            directional_host(),
        )
        print(
            f"CPU tracking of {len(t['per_clip'])} clips: normalized return {t['summary']['return']:.3f}, "
            f"{t['summary']['completed']:.0%} completed",
            flush=True,
        )
    d = evaluate_directional(tracker, directional) if directional else None
    c = evaluate_chasetag(tracker, chasetag) if chasetag else None
    _print_tables(d, c)
    result = {
        "checkpoint": args.checkpoint,
        "iteration": tracker.iteration,
        "wall_time_min": tracker.wall_time / 60,
        "walk": str(args.walk or "KIT/167/walking_medium06"),
        "cycle": cycle,
        "tracking": t,
        "directional": d,
        "chasetag": c,
    }
    out.write_text(json.dumps(result, indent=1, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
