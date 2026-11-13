# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Retarget InstantHMR MHR70 clips (or videos) onto a MyoSuite body model.

    python scripts/mimic/retarget_instanthmr.py /path/to/instanthmr_clips \
        --model myofullbody --workers 4

Accepts a single ``.npz`` clip, a directory of ``.npz`` clips, a ``.txt`` clip list, or
a video file (``.mp4``, ``.mov``, ``.avi``, ``.mkv``, ``.webm``; requires optional
``instanthmr`` and ``opencv-python``). Outputs are written in the layout read by
``myoMimic-v0``'s motion library, stamped with ``model_fingerprint`` and accompanied by
``retarget_summary.json``, ``retarget_summary.md`` and ``retarget_clips.jsonl``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

MAX_WORKERS = 8
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_STATE: dict[str, Any] = {}


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def extract_video_clip(
    video_path: str | Path,
    out_npz: str | Path,
    *,
    hmr_model: str = "medium-commercial",
    fit: int = 2,
    device: str = "cuda",
    focal: float | None = None,
    pose_filter: str | None = "one-euro",
) -> Path:
    """Run ``instanthmr.PosePipeline`` on a video and write an MHR70 ``.npz`` clip.

    Tracks the primary person across frames (largest initial bounding box, then highest
    IoU continuity) and records missed detections in ``valid`` so
    :func:`~myosuite.biomechanics.retarget_instanthmr.load_instanthmr` interpolates
    short gaps smoothly.
    """
    try:
        import cv2  # noqa: PLC0415
        from instanthmr import PosePipeline  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Video extraction requires optional 'instanthmr' and 'opencv-python' "
            "packages; pass pre-extracted InstantHMR .npz clips otherwise."
        ) from exc

    from myosuite.biomechanics.retarget_instanthmr import MHR70_JOINTS  # noqa: PLC0415

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        pipeline = PosePipeline(
            hmr_model, fit=fit, device=device, focal=focal, pose_filter=pose_filter
        )
        joints_list: list[np.ndarray] = []
        valid_list: list[bool] = []
        prev_box: np.ndarray | None = None
        frame_idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = pipeline(rgb, timestamp=frame_idx / fps)
            persons = res.persons
            chosen = None
            if persons:
                if prev_box is None:
                    chosen = max(
                        persons,
                        key=lambda p: float(
                            (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1])
                        ),
                    )
                else:
                    best = max(persons, key=lambda p: _box_iou(prev_box, p.bbox))
                    if _box_iou(prev_box, best.bbox) > 0.05 or len(persons) == 1:
                        chosen = best
            if chosen is not None:
                prev_box = np.asarray(chosen.bbox, np.float32)
                joints_list.append(np.asarray(chosen.joints_3d_cam, np.float32))
                valid_list.append(True)
            else:
                joints_list.append(
                    np.full((len(MHR70_JOINTS), 3), np.nan, dtype=np.float32)
                )
                valid_list.append(False)
            frame_idx += 1
    finally:
        cap.release()

    if sum(valid_list) < 2:
        raise ValueError(
            f"{video_path}: fewer than 2 frames with a detected person ({sum(valid_list)}/{len(valid_list)})"
        )
    out = Path(out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        joints_3d_cam=np.stack(joints_list, axis=0),
        fps=np.float64(fps),
        valid=np.asarray(valid_list, dtype=bool),
    )
    return out


def _compile_target(model_name: str) -> tuple[Any, str]:
    from myosuite.biomechanics import retarget as rt  # noqa: PLC0415
    from myosuite.biomechanics import retarget_instanthmr as ri  # noqa: PLC0415

    if model_name == "myolegs":
        from myosuite import make_env  # noqa: PLC0415

        env = make_env("myoLegWalk-v0")
        try:
            model = env.unwrapped.model.__copy__()
        finally:
            env.close()
        imap = ri.INSTANTHMR_MAPS[model_name]
        fp = rt.kinematic_fingerprint(model, ri.landmark_bodies(imap), imap.feet)
        return model, fp

    from myosuite.envs.myo.tasks.mimic.tracking.models import (  # noqa: PLC0415
        compiled_body,
        kinematic_fingerprint,
    )

    return compiled_body(model_name), kinematic_fingerprint(model_name)


def _init(model_name: str, version: str, trim: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    import warnings  # noqa: PLC0415

    warnings.filterwarnings("ignore")
    from myosuite.biomechanics import retarget as rt  # noqa: PLC0415
    from myosuite.biomechanics import retarget_instanthmr as ri  # noqa: PLC0415

    model, fp = _compile_target(model_name)
    src, lmap = ri.instanthmr_source(
        model, ri.INSTANTHMR_MAPS[model_name], trim=(trim, trim)
    )
    _STATE.update(
        rt=rt,
        model=model,
        lmap=lmap,
        name=model_name,
        version=version,
        fingerprint=fp,
        ik=rt.LandmarkIK(model, lmap),
        source=src,
    )


def _work(job: tuple[str, str, str, dict[str, Any]]) -> dict[str, Any]:
    src, dst, label, curation = job
    s = _STATE
    rt, model, lmap = s["rt"], s["model"], s["lmap"]
    curation = {"quality_flags": [], **curation}
    try:
        if Path(dst).exists():
            meta = rt.read_refit_meta(dst)
            if meta and "model_fingerprint" in meta:
                same_body = meta["model_fingerprint"] == s["fingerprint"]
            else:
                same_body = bool(meta) and meta.get("model_version") == s["version"]
            if (
                same_body
                and meta.get("source_sha256") == rt.file_sha256(src)
                and meta.get("landmark_map_version") == lmap.version
                and meta.get("solver") == dataclasses.asdict(s["ik"].config)
                and (meta.get("instanthmr_map_version"), meta.get("trim"))
                == (s["source"].imap.version, list(s["source"].trim))
            ):
                curation = {"model_fingerprint": s["fingerprint"], **curation}
                if any(meta.get(k) != v for k, v in curation.items()):
                    meta.update(curation)
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
        rt.save_refit(dst, model, qpos, freq, meta)
        return {"meta": meta, "skipped": False}
    except Exception as err:
        return {"error": f"{type(err).__name__}: {err}", "source": label}


def _markdown(summary: dict[str, Any]) -> str:
    s = summary
    lines = [
        f"# InstantHMR retarget: {s['model']} {s['model_version']} ({s['landmark_map']} v{s['landmark_map_version']})",
        "",
        f"{s['clips']} clips, {s['frames']} frames, {s['fit_ms_per_frame']:.2f} ms/frame per worker, "
        f"{s['wall_s']:.1f} s wall on {s['workers']} workers. Errors: {len(s['errors'])}. "
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
        "## Flagged clips",
        "",
    ]
    lines += [f"- {f['source']}: {', '.join(f['flags'])}" for f in s["flagged"]] or [
        "none"
    ]
    if s["errors"]:
        lines += ["", "## Errors", ""] + [
            f"- {e['source']}: {e['error']}" for e in s["errors"]
        ]
    return "\n".join(lines) + "\n"


def stamp(root: Path, model_name: str) -> int:
    """Record the installed body's fingerprint in every refitted clip under *root*."""
    import myo_sim  # noqa: PLC0415

    from myosuite.biomechanics import retarget as rt  # noqa: PLC0415

    _, fp = _compile_target(model_name)
    version = str(myo_sim.__version__)
    counts = {"stamped": 0, "already": 0}
    refused = []
    for f in sorted(root.rglob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        try:
            key = (
                "stamped"
                if rt.stamp_fingerprint(f, model_name, version, fp)
                else "already"
            )
            counts[key] += 1
        except ValueError as err:
            refused.append(str(err))
    print(
        f"{model_name} myo_sim {version}, fingerprint {fp}: {counts['stamped']} "
        f"stamped, {counts['already']} already stamped, {len(refused)} refused"
    )
    for msg in refused:
        print("  " + msg)
    return len(refused)


def main(argv: list[str] | None = None) -> int:
    """Retarget InstantHMR clips or videos; returns the number of failed clips."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "input",
        nargs="?",
        help="InstantHMR .npz clip, directory, .txt clip list, or video file",
    )
    p.add_argument(
        "--stamp-fingerprint",
        default="",
        metavar="DIR",
        help="only record the body's kinematic fingerprint in the clips under DIR",
    )
    p.add_argument("--model", default="myofullbody", choices=("myofullbody", "myolegs"))
    p.add_argument(
        "--out-root",
        default="",
        help="default: ~/.cache/myosuite/mimic/<model>-<version>",
    )
    p.add_argument(
        "--dataset",
        default="",
        help="sub-folder of the outputs (default: input dir or stem name)",
    )
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--report-dir", default="")
    p.add_argument("--limit", type=int, default=0, help="only the first N clips")
    p.add_argument("--trim", type=int, default=0, help="frames dropped at each end")
    p.add_argument(
        "--quality-flags",
        default="",
        help="JSON {clip label: metadata fields} merged into the clips",
    )
    p.add_argument("--hmr-model", default="medium-commercial")
    p.add_argument("--hmr-fit", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--focal", type=float, default=None)
    args = p.parse_args(argv)

    if args.stamp_fingerprint:
        return stamp(Path(args.stamp_fingerprint).expanduser(), args.model)
    if not args.input:
        p.error("input is required unless --stamp-fingerprint is set")

    import myo_sim  # noqa: PLC0415

    from myosuite.biomechanics import retarget as rt  # noqa: PLC0415
    from myosuite.biomechanics import retarget_instanthmr as ri  # noqa: PLC0415

    version = str(myo_sim.__version__)
    out_root = (
        Path(args.out_root).expanduser()
        if args.out_root
        else rt.default_cache_root(args.model, version)
    )
    src = Path(args.input).expanduser()
    if src.is_file() and src.suffix.lower() in VIDEO_SUFFIXES:
        dataset = args.dataset or "instanthmr"
        extracted = out_root / "_raw_instanthmr" / dataset / f"{src.stem}.npz"
        extract_video_clip(
            src,
            extracted,
            hmr_model=args.hmr_model,
            fit=args.hmr_fit,
            device=args.device,
            focal=args.focal,
        )
        files, base = [extracted], extracted.parent
    elif src.is_file() and src.suffix == ".txt":
        files = [
            Path(line.split("#", 1)[0].strip())
            for line in src.read_text().splitlines()
            if line.split("#", 1)[0].strip()
        ]
        base = Path(os.path.commonpath([str(f.parent) for f in files]))
        dataset = args.dataset or src.stem
    elif src.is_file():
        files, base = [src], src.parent
        dataset = args.dataset or "instanthmr"
    elif src.is_dir():
        files, base = sorted(src.rglob("*.npz")), src
        dataset = args.dataset or src.name
    else:
        raise FileNotFoundError(f"input not found: {src}")

    if args.limit:
        files = files[: args.limit]
    curation = (
        json.loads(Path(args.quality_flags).read_text()) if args.quality_flags else {}
    )
    jobs = []
    for f in files:
        rel_posix = f.relative_to(base).as_posix()
        label = f"InstantHMR/{dataset}/{rel_posix}"
        dst = str(out_root / dataset / f.relative_to(base))
        jobs.append((str(f), dst, label, curation.get(label, {})))

    workers = max(
        1, min(args.workers, MAX_WORKERS, os.cpu_count() or 1, max(len(jobs), 1))
    )
    t0 = time.time()
    results: list[dict[str, Any]] = []
    if workers == 1:
        _init(args.model, version, args.trim)
        results = [_work(j) for j in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            workers,
            initializer=_init,
            initargs=(args.model, version, args.trim),
            maxtasksperchild=50,
        ) as pool:
            results = list(pool.imap_unordered(_work, jobs, chunksize=1))

    metas = [r["meta"] for r in results if "meta" in r]
    errors = [
        {"source": r["source"], "error": r["error"]} for r in results if "error" in r
    ]
    summary = (
        rt.summarize(metas)
        if metas
        else {
            "clips": 0,
            "frames": 0,
            "fit_ms_per_frame": 0.0,
            "stats": {},
            "flagged": [],
        }
    )
    imap = ri.INSTANTHMR_MAPS[args.model]
    summary.update(
        model=args.model,
        model_version=version,
        landmark_map=imap.name,
        landmark_map_version=imap.version,
        trim=args.trim,
        out_root=str(out_root / dataset),
        wall_s=time.time() - t0,
        workers=workers,
        errors=errors,
        thresholds=dataclasses.asdict(rt.RefitThresholds()),
    )
    report = Path(args.report_dir) if args.report_dir else (out_root / dataset)
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
