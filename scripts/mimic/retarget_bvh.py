# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Retarget Biovision Hierarchy (``.bvh`` / LAFAN1) clips onto a task model.

    python scripts/mimic/retarget_bvh.py /path/to/lafan1 --scale 0.01 \
        --model myofullbody --workers 12

Every ``.bvh`` clip under the input (a single ``.bvh``, a directory, or a ``.txt`` clip
list) is fitted on the model's joint couplings and written as a ``.npz`` clip in the
layout ``myoMimic-v0``'s motion library reads:
``<out-root>/<dataset>/<relative path>.npz`` (default out-root:
``~/.cache/myosuite/mimic/<model>-<myo_sim version>``). Existing outputs matching the
source SHA-256, ``model_fingerprint``, map versions, scale, trim and solver config are
reused without refitting. Writes ``retarget_summary.json``, ``retarget_summary.md`` and
``retarget_clips.jsonl`` into ``--report-dir`` (default: ``<out-root>/<dataset>``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

MAX_WORKERS = 12
MIN_FREE_GB = 12.0
WORKER_GB = 0.6
STRAIGHT_WALK = re.compile(
    r"(?i)(^walk\d*_|walkingstraightforward|walking_(slow|medium|fast)|walk_6m|walking_forward_4steps)"
)

_STATE: dict[str, Any] = {}


def choose_workers(requested: int) -> int:
    """Choose worker count within cap, idle CPU cores and RAM headroom."""
    from myosuite.biomechanics import retarget as rt  # noqa: PLC0415

    return rt.choose_workers(
        requested,
        max_workers=MAX_WORKERS,
        min_free_gb=MIN_FREE_GB,
        worker_gb=WORKER_GB,
    )


def _init(model_name: str, version: str, scale: float, trim: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    import warnings  # noqa: PLC0415

    warnings.filterwarnings("ignore")
    from myosuite.biomechanics import retarget as rt  # noqa: PLC0415
    from myosuite.integrations.capture import bvh as rb  # noqa: PLC0415
    from myosuite.envs.myo.tasks.mimic.tracking.models import (  # noqa: PLC0415
        compiled_body,
        kinematic_fingerprint,
    )

    model = compiled_body(model_name)
    src, lmap = rb.bvh_source(
        model, rb.BVH_MAPS[model_name], scale=scale, trim=(trim, trim)
    )
    _STATE.update(
        rt=rt,
        model=model,
        lmap=lmap,
        name=model_name,
        version=version,
        fingerprint=kinematic_fingerprint(model_name),
        ik=rt.LandmarkIK(model, lmap),
        source=src,
    )


def _twist(
    rt: Any, model: Any, lmap: Any, poses: dict[str, np.ndarray], dt: float
) -> dict[str, Any]:
    out = {}
    for key, q in poses.items():
        tw = rt.twist_asymmetry(model, lmap, q, dt)
        if tw is not None:
            out[key] = {
                "period_s": tw["period_s"],
                "slope": tw["slope"],
                "x": tw["x"].astype(np.float32),
                "y": {k: v.astype(np.float32) for k, v in tw["y"].items()},
            }
    return out


def _work(job: tuple[str, str, str, dict[str, Any]]) -> dict[str, Any]:
    src, dst, label, curation = job
    s = _STATE
    rt, model, lmap = s["rt"], s["model"], s["lmap"]
    curation = dict(curation)
    cur_flags = curation.pop("quality_flags", None)
    try:
        meta = rt.is_cached_refit(
            dst,
            src,
            fingerprint=s["fingerprint"],
            model_version=s["version"],
            landmark_map_version=lmap.version,
            solver=dataclasses.asdict(s["ik"].config),
            extra_checks={
                "bvh_map_version": s["source"].bmap.version,
                "scale": float(s["source"].scale),
                "trim": list(s["source"].trim),
            },
        )
        if meta is not None:
            updates = {
                "model_fingerprint": s["fingerprint"],
                "quality_flags": rt.merge_quality_flags(
                    meta.get("quality_flags"), cur_flags
                ),
                **curation,
            }
            if any(meta.get(k) != v for k, v in updates.items()):
                meta.update(updates)
                with np.load(dst, allow_pickle=False) as d:
                    qpos, freq = d["qpos"], float(d["frequency"])
                rt.save_refit(dst, model, qpos, freq, meta)
            return {"meta": meta, "skipped": True}
        qpos, freq, meta = rt.refit_clip(
            src,
            model,
            s["name"],
            s["version"],
            lmap,
            s["source"],
            ik=s["ik"],
            model_fingerprint=s["fingerprint"],
        )
        meta["source"] = label
        meta.update(curation)
        meta["quality_flags"] = rt.merge_quality_flags(
            meta.get("quality_flags"), cur_flags
        )
        rt.save_refit(dst, model, qpos, freq, meta)
        res: dict[str, Any] = {"meta": meta, "skipped": False}
        if STRAIGHT_WALK.search(Path(src).name):
            res["twist"] = _twist(rt, model, lmap, {"fit": qpos}, 1.0 / freq)
        return res
    except Exception as err:
        return {"error": f"{type(err).__name__}: {err}", "source": src}


def stamp(root: Path, model_name: str) -> int:
    """Record the installed body's fingerprint in every refitted clip under *root*."""
    import myo_sim  # noqa: PLC0415

    from myosuite.biomechanics import retarget as rt  # noqa: PLC0415
    from myosuite.envs.myo.tasks.mimic.tracking.models import (  # noqa: PLC0415
        kinematic_fingerprint,
    )

    return rt.stamp_tree(
        root, model_name, str(myo_sim.__version__), kinematic_fingerprint(model_name)
    )


def main(argv: list[str] | None = None) -> int:
    """Retarget a ``.bvh`` clip, directory or clip list; returns failed clip count."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "input", nargs="?", help=".bvh file, directory, or .txt list of .bvh clips"
    )
    p.add_argument(
        "--stamp-fingerprint",
        default="",
        metavar="DIR",
        help="only record the body's kinematic fingerprint in the clips under DIR",
    )
    p.add_argument("--model", default="myofullbody")
    p.add_argument(
        "--out-root",
        default="",
        help="default: ~/.cache/myosuite/mimic/<model>-<version>",
    )
    p.add_argument(
        "--dataset",
        default="",
        help="sub-folder of the outputs (default: input dir name)",
    )
    p.add_argument("--workers", type=int, default=MAX_WORKERS)
    p.add_argument("--report-dir", default="")
    p.add_argument(
        "--limit", type=int, default=0, help="only the first N clips (testing)"
    )
    p.add_argument(
        "--scale",
        type=float,
        default=0.01,
        help="linear unit scale to metres (default: 0.01 for cm BVH files)",
    )
    p.add_argument(
        "--trim",
        type=int,
        default=0,
        help="source frames dropped at each end",
    )
    p.add_argument(
        "--quality-flags",
        default="",
        help="JSON {clip label: metadata fields} merged into the clips (curation)",
    )
    args = p.parse_args(argv)
    if args.stamp_fingerprint:
        return stamp(Path(args.stamp_fingerprint).expanduser(), args.model)
    if not args.input:
        p.error("input is required unless --stamp-fingerprint is set")

    import myo_sim  # noqa: PLC0415

    from myosuite.biomechanics import retarget as rt  # noqa: PLC0415
    from myosuite.integrations.capture import bvh as rb  # noqa: PLC0415

    version = str(myo_sim.__version__)
    files, base, dataset = rt.collect_clip_files(
        args.input, default_dataset=args.dataset, limit=args.limit, suffix=".bvh"
    )
    out_root = (
        Path(args.out_root).expanduser()
        if args.out_root
        else rt.default_cache_root(args.model, version)
    )
    curation = (
        json.loads(Path(args.quality_flags).read_text(encoding="utf-8"))
        if args.quality_flags
        else {}
    )
    jobs = []
    for f in files:
        rel = f.relative_to(base)
        rel_npz = rel.with_suffix(".npz")
        label = f"BVH/{dataset}/{rel.as_posix()}"
        label_npz = f"BVH/{dataset}/{rel_npz.as_posix()}"
        cur = curation.get(label) or curation.get(label_npz, {})
        dst = str(out_root / dataset / rel_npz)
        jobs.append((str(f), dst, label, cur))

    workers = (
        1
        if args.workers <= 1 or len(jobs) <= 1
        else choose_workers(min(args.workers, len(jobs)))
    )
    t0 = time.time()
    init_args = (args.model, version, args.scale, args.trim)
    results: list[dict[str, Any]] = []
    if workers == 1:
        _init(*init_args)
        results = [_work(j) for j in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            workers,
            initializer=_init,
            initargs=init_args,
            maxtasksperchild=50,
        ) as pool:
            for i, r in enumerate(pool.imap_unordered(_work, jobs, chunksize=1), 1):
                results.append(r)
                if i % 25 == 0 or i == len(jobs):
                    print(
                        f"{i}/{len(jobs)} clips, {time.time() - t0:.0f} s, "
                        f"MemAvailable {rt._mem_available_gb():.1f} GB",
                        flush=True,
                    )

    metas = [r["meta"] for r in results if "meta" in r]
    errors = [
        {"source": r["source"], "error": r["error"]} for r in results if "error" in r
    ]
    summary = rt.summarize(metas) if metas else rt.empty_summary()
    bmap = rb.BVH_MAPS[args.model]
    summary.update(
        model=args.model,
        model_version=version,
        landmark_map=bmap.name,
        landmark_map_version=bmap.version,
        scale=args.scale,
        trim=args.trim,
        skipped=sum(1 for r in results if r.get("skipped")),
        out_root=str(out_root / dataset),
        wall_s=time.time() - t0,
        workers=workers,
        errors=errors,
        thresholds=dataclasses.asdict(rt.RefitThresholds()),
        twist=rt.pooled_twist(results),
    )
    report = Path(args.report_dir) if args.report_dir else (out_root / dataset)
    print(rt.write_refit_report(report, summary, metas, title="BVH retarget"))
    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
