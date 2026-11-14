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

Every clip records the body's kinematic fingerprint (``model_fingerprint``), so it keeps
loading under myo_sim releases that leave the body's kinematics alone. Clips fitted
before the fingerprint was recorded get it without a refit (qpos untouched), with the
myo_sim version they were fitted on installed:

    python scripts/mimic/retarget_amass.py --stamp-fingerprint ~/.cache/myosuite/mimic/myofullbody-<version>
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
    r"(?i)(walkingstraightforward|walking_(slow|medium|fast)|walk_6m|walking_forward_4steps)"
)
"""Clips of the left/right twist-symmetry measure (straight forward walking)."""

_STATE: dict[str, Any] = {}


def choose_workers(requested: int) -> int:
    """Workers within the cap, the idle cores (load) and the RAM headroom."""
    from myosuite.biomechanics import retarget as rt  # noqa: PLC0415

    return rt.choose_workers(
        requested,
        max_workers=MAX_WORKERS,
        min_free_gb=MIN_FREE_GB,
        worker_gb=WORKER_GB,
    )


def _init(model_name: str, version: str, body_models: str, trim: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    import warnings  # noqa: PLC0415

    warnings.filterwarnings("ignore")
    from myosuite.biomechanics import retarget as rt  # noqa: PLC0415
    from myosuite.biomechanics import retarget_smplh as rs  # noqa: PLC0415
    from myosuite.envs.myo.tasks.mimic.tracking.models import (  # noqa: PLC0415
        compiled_body,
        kinematic_fingerprint,
    )

    model = compiled_body(model_name)
    src, lmap = rs.smplh_source(model, body_models, rs.SMPLH_MAPS[model_name],
                                trim=(trim, trim))  # fmt: skip
    _STATE.update(rt=rt, model=model, lmap=lmap, name=model_name, version=version,
                  fingerprint=kinematic_fingerprint(model_name),
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
                "smplh_map_version": s["source"].smap.version,
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
                with np.load(dst) as d:
                    qpos, freq = d["qpos"], float(d["frequency"])
                rt.save_refit(dst, model, qpos, freq, meta)
            return {"meta": meta, "skipped": True}
        qpos, freq, meta = rt.refit_clip(
            src, model, s["name"], s["version"], lmap, s["source"], ik=s["ik"],
            model_fingerprint=s["fingerprint"],
        )  # fmt: skip
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
    except Exception as err:  # reported per clip, the run goes on
        return {"error": f"{type(err).__name__}: {err}", "source": src}


def stamp(root: Path, model_name: str) -> int:
    """Record the installed body's fingerprint in every refitted clip under *root*
    (qpos untouched); returns the number of clips refused."""
    import myo_sim  # noqa: PLC0415

    from myosuite.biomechanics import retarget as rt  # noqa: PLC0415
    from myosuite.envs.myo.tasks.mimic.tracking.models import (  # noqa: PLC0415
        kinematic_fingerprint,
    )

    return rt.stamp_tree(
        root, model_name, str(myo_sim.__version__), kinematic_fingerprint(model_name)
    )


def main(argv: list[str] | None = None) -> int:
    """Retarget a directory or list of AMASS clips; returns the number of failed clips."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "input", nargs="?", help="AMASS directory or .txt list of *_poses.npz"
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
    p.add_argument("--report-dir", default=".")
    p.add_argument(
        "--limit", type=int, default=0, help="only the first N clips (testing)"
    )
    p.add_argument(
        "--body-models",
        default="",
        help="folder with {male,female,neutral}/model.npz (SMPL+H); needed to retarget",
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
    if args.stamp_fingerprint:
        return stamp(Path(args.stamp_fingerprint).expanduser(), args.model)
    if not (args.input and args.body_models):
        p.error("input and --body-models are needed to retarget")

    import myo_sim  # noqa: PLC0415

    from myosuite.biomechanics import retarget as rt  # noqa: PLC0415

    version = str(myo_sim.__version__)
    files, base, dataset = rt.collect_clip_files(
        args.input, default_dataset=args.dataset, limit=args.limit
    )
    out_root = (
        Path(args.out_root)
        if args.out_root
        else rt.default_cache_root(args.model, version)
    )
    curation = (
        json.loads(Path(args.quality_flags).read_text()) if args.quality_flags else {}
    )
    jobs = []
    for f in files:
        label = f"AMASS/{dataset}/{f.relative_to(base).as_posix()}"
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
                      f"MemAvailable {rt._mem_available_gb():.1f} GB", flush=True)  # fmt: skip
    metas = [r["meta"] for r in results if "meta" in r]
    errors = [
        {"source": r["source"], "error": r["error"]} for r in results if "error" in r
    ]
    summary = rt.summarize(metas) if metas else rt.empty_summary()
    from myosuite.biomechanics import retarget_smplh as rs  # noqa: PLC0415

    lmap = rs.SMPLH_MAPS[args.model]
    summary.update(model=args.model, model_version=version, landmark_map=lmap.name,
                   landmark_map_version=lmap.version, trim=args.trim, out_root=str(out_root / dataset),
                   wall_s=time.time() - t0, workers=workers, errors=errors,
                   thresholds=dataclasses.asdict(rt.RefitThresholds()),
                   twist=rt.pooled_twist(results))  # fmt: skip
    print(
        rt.write_refit_report(args.report_dir, summary, metas, title="AMASS retarget")
    )
    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
