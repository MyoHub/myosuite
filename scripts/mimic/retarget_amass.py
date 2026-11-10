# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Retarget AMASS SMPL+H motions onto a task model (``myosuite.biomechanics.retarget_smplh``).

    python scripts/mimic/retarget_amass.py /path/to/amass/KIT --trim 3 \
        --body-models /path/to/amass/body_models/smplh --model myofullbody --workers 12

Every ``*_poses.npz`` under the input (a directory or a ``.txt`` list) is fitted in
parallel and written, in the layout ``myoMimic-v0``'s motion library reads, to
``<out-root>/<dataset>/<path relative to the input>`` (default out-root:
``~/.cache/myosuite/mimic/<model>-<myo_sim version>``, dataset: the input folder's
name). Clips whose output already holds a fit of the same source hash, model version,
maps, trim and solver options are skipped. ``--trim 3`` drops three frames at each end,
as MuscleMimic's KIT clips do. ``--quality-flags`` merges a curation file
(``{"AMASS/<dataset>/<clip>": {"quality_flags": [...], ...}}``) into the metadata; every
clip gets ``quality_flags`` (empty when unflagged). The report directory gets ``retarget_summary.json``,
``retarget_summary.md`` and one JSON line per clip; flagged clips are listed there, not
dropped silently. The AMASS data and SMPL+H models are not part of MyoSuite.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
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
    r"(?i)(walkingstraightforward|walking_(slow|medium|fast)|walk_6m|walking_forward_4steps)"
)
"""Clips of the left/right twist-symmetry measure (straight forward walking)."""

_STATE: dict[str, Any] = {}


def _mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return float("inf")


def choose_workers(requested: int) -> int:
    """Workers within the cap, the idle cores (load) and the RAM headroom."""
    cores = os.cpu_count() or 1
    idle = cores - math.ceil(os.getloadavg()[0]) - 4  # keep cores for other users
    ram = int((_mem_available_gb() - MIN_FREE_GB) / WORKER_GB)
    n = max(1, min(requested, MAX_WORKERS, idle, ram))
    print(f"cores {cores}, load {os.getloadavg()[0]:.1f}, "
          f"MemAvailable {_mem_available_gb():.1f} GB -> {n} workers", flush=True)  # fmt: skip
    return n


def _init(model_name: str, version: str, body_models: str, trim: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    import warnings  # noqa: PLC0415

    warnings.filterwarnings("ignore")
    from myosuite.envs.myo.tasks.mimic.tracking.models import compiled_body  # noqa: PLC0415
    from myosuite.biomechanics import retarget as rt  # noqa: PLC0415
    from myosuite.biomechanics import retarget_smplh as rs  # noqa: PLC0415

    model = compiled_body(model_name)
    src, lmap = rs.smplh_source(model, body_models, rs.SMPLH_MAPS[model_name],
                                trim=(trim, trim))  # fmt: skip
    _STATE.update(rt=rt, model=model, lmap=lmap, name=model_name, version=version,
                  ik=rt.LandmarkIK(model, lmap), source=src)  # fmt: skip


def _twist(
    rt: Any, model: Any, lmap: Any, poses: dict[str, np.ndarray], dt: float
) -> dict:
    out = {}
    for key, q in poses.items():
        tw = rt.twist_asymmetry(model, lmap, q, dt)
        if tw is not None:
            out[key] = {"period_s": tw["period_s"], "slope": tw["slope"],
                        "x": tw["x"].astype(np.float32),
                        "y": {k: v.astype(np.float32) for k, v in tw["y"].items()}}  # fmt: skip
    return out


def _work(job: tuple[str, str, str, dict[str, Any]]) -> dict[str, Any]:
    src, dst, label, curation = job
    s = _STATE
    rt, model, lmap = s["rt"], s["model"], s["lmap"]
    curation = {"quality_flags": [], **curation}
    try:
        if Path(dst).exists():
            meta = rt.read_refit_meta(dst)
            if (
                meta
                and meta.get("source_sha256") == rt.file_sha256(src)
                and (meta.get("model_version"), meta.get("landmark_map_version"))
                == (s["version"], lmap.version)
                and meta.get("solver") == dataclasses.asdict(s["ik"].config)
                and (meta.get("smplh_map_version"), meta.get("trim"))
                == (s["source"].smap.version, list(s["source"].trim))
            ):
                if any(meta.get(k) != v for k, v in curation.items()):
                    meta.update(curation)  # curation changed: rewrite the metadata only
                    with np.load(dst) as d:
                        qpos, freq = d["qpos"], float(d["frequency"])
                    rt.save_refit(dst, model, qpos, freq, meta)
                return {"meta": meta, "skipped": True}
        qpos, freq, meta = rt.refit_clip(
            src, model, s["name"], s["version"], lmap, s["source"], ik=s["ik"]
        )
        meta["source"] = label  # dataset-relative, no local paths in shared files
        meta.update(curation)
        rt.save_refit(dst, model, qpos, freq, meta)
        res: dict[str, Any] = {"meta": meta, "skipped": False}
        if STRAIGHT_WALK.search(Path(src).name):
            res["twist"] = _twist(rt, model, lmap, {"fit": qpos}, 1.0 / freq)
        return res
    except Exception as err:  # reported per clip, the run goes on
        return {"error": f"{type(err).__name__}: {err}", "source": src}


def _pooled_twist(results: list[dict]) -> dict[str, Any]:
    """#651's pooled regression of the left-minus-right twist on the coupling term."""
    out: dict[str, Any] = {}
    tw = [r["twist"] for r in results if r.get("twist")]
    for key in ("fit",):
        xs = [t[key]["x"] for t in tw if key in t]
        if not xs:
            continue
        x = np.concatenate(xs)
        row = {}
        for body in tw[0][key]["y"]:
            y = np.concatenate([t[key]["y"][body] for t in tw if key in t])
            A = np.stack([x, np.ones_like(x)], 1)
            slope = float(np.linalg.lstsq(A, y, rcond=None)[0][0])
            per = np.array([t[key]["slope"][body] for t in tw if key in t])
            row[body] = {"pooled_slope": slope, "clip_median": float(np.median(per)),
                         "clips_positive": int((per > 0).sum()), "clips": len(per)}  # fmt: skip
        out[key] = row
    return out


def _markdown(summary: dict[str, Any]) -> str:
    s = summary
    lines = [
        f"# AMASS retarget: {s['model']} {s['model_version']} ({s['landmark_map']} v{s['landmark_map_version']})",
        "",
        f"{s['clips']} clips, {s['frames']} frames, {s['fit_ms_per_frame']:.2f} ms/frame per worker, "
        f"{s['wall_s']:.0f} s wall on {s['workers']} workers. Errors: {len(s['errors'])}. "
        f"Flagged: {len(s['flagged'])}.",
        "",
        "| metric | median | p95 | max |",
        "|---|---|---|---|",
    ]
    for k, v in s["stats"].items():
        lines.append(f"| {k} | {v['median']:.4g} | {v['p95']:.4g} | {v['max']:.4g} |")
    lines += [
        "",
        "Thresholds: " + ", ".join(f"{k}={v}" for k, v in s["thresholds"].items()),
        "",
    ]
    if s.get("twist"):
        lines += ["## Left/right axial-twist asymmetry (straight walks)",
                  "Slope of left-minus-right internal twist (half a stride apart) on the left "
                  "knee coupling term; 0 is symmetric.", "",
                  "| pose | " + " | ".join(next(iter(s["twist"].values()))) + " |",
                  "|---|" + "---|" * len(next(iter(s["twist"].values())))]  # fmt: skip
        for key, row in s["twist"].items():
            cells = [f"{v['pooled_slope']:+.2f} ({v['clips_positive']}/{v['clips']} > 0)"
                     for v in row.values()]  # fmt: skip
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Flagged clips", ""]
    lines += [f"- {f['source']}: {', '.join(f['flags'])}" for f in s["flagged"]] or [
        "none"
    ]
    if s["errors"]:
        lines += ["", "## Errors", ""] + [
            f"- {e['source']}: {e['error']}" for e in s["errors"]
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Retarget a directory or list of AMASS clips; returns the number of failed clips."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("input", help="AMASS directory or .txt list of *_poses.npz")
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
    p.add_argument("--report-dir", default=".")
    p.add_argument(
        "--limit", type=int, default=0, help="only the first N clips (testing)"
    )
    p.add_argument(
        "--body-models",
        required=True,
        help="folder with {male,female,neutral}/model.npz (SMPL+H)",
    )
    p.add_argument(
        "--trim",
        type=int,
        default=0,
        help="source frames dropped at each end (3 for MuscleMimic's KIT clips)",
    )
    p.add_argument(
        "--quality-flags",
        default="",
        help="JSON {clip label: metadata fields} merged into the clips (curation)",
    )
    args = p.parse_args(argv)

    import myo_sim  # noqa: PLC0415

    from myosuite.biomechanics import retarget as rt  # noqa: PLC0415

    version = str(myo_sim.__version__)
    src = Path(args.input)
    if src.suffix == ".txt":
        files = [
            Path(line.strip()) for line in src.read_text().splitlines() if line.strip()
        ]
        base = Path(os.path.commonpath([str(f.parent) for f in files]))
    else:
        files, base = sorted(src.rglob("*.npz")), src
    if args.limit:
        files = files[: args.limit]
    out_root = (
        Path(args.out_root)
        if args.out_root
        else rt.default_cache_root(args.model, version)
    )
    dataset = args.dataset or base.name
    curation = (
        json.loads(Path(args.quality_flags).read_text()) if args.quality_flags else {}
    )
    jobs = []
    for f in files:
        label = f"AMASS/{dataset}/{f.relative_to(base)}"
        dst = str(out_root / dataset / f.relative_to(base))
        jobs.append((str(f), dst, label, curation.get(label, {})))
    workers = choose_workers(args.workers)
    t0 = time.time()
    results = []
    ctx = mp.get_context("spawn")
    init = (args.model, version, args.body_models, args.trim)
    with ctx.Pool(workers, initializer=_init, initargs=init,
                  maxtasksperchild=50) as pool:  # fmt: skip
        for i, r in enumerate(pool.imap_unordered(_work, jobs, chunksize=1), 1):
            results.append(r)
            if i % 25 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} clips, {time.time() - t0:.0f} s, "
                      f"MemAvailable {_mem_available_gb():.1f} GB", flush=True)  # fmt: skip
    metas = [r["meta"] for r in results if "meta" in r]
    errors = [
        {"source": r["source"], "error": r["error"]} for r in results if "error" in r
    ]
    summary = rt.summarize(metas) if metas else {"clips": 0, "frames": 0, "fit_ms_per_frame": 0,
                                                  "stats": {}, "flagged": []}  # fmt: skip
    from myosuite.biomechanics import retarget_smplh as rs  # noqa: PLC0415

    lmap = rs.SMPLH_MAPS[args.model]
    summary.update(model=args.model, model_version=version, landmark_map=lmap.name,
                   landmark_map_version=lmap.version, trim=args.trim, out_root=str(out_root / dataset),
                   wall_s=time.time() - t0, workers=workers, errors=errors,
                   thresholds=dataclasses.asdict(rt.RefitThresholds()),
                   twist=_pooled_twist(results))  # fmt: skip
    report = Path(args.report_dir)
    report.mkdir(parents=True, exist_ok=True)
    (report / "retarget_summary.json").write_text(json.dumps(summary, indent=1))
    (report / "retarget_summary.md").write_text(_markdown(summary))
    with (report / "retarget_clips.jsonl").open("w") as f:
        for m in sorted(metas, key=lambda m: m["source"]):
            f.write(json.dumps(m) + "\n")
    print(_markdown(summary))
    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
