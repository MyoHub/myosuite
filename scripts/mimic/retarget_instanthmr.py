# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Retarget, scale and reconstruct 3D muscles from InstantHMR MHR70 ``.npz`` clips.

    python scripts/mimic/retarget_instanthmr.py /path/to/instanthmr_clips \
        --model myofullbody --scale --reconstruct-muscles volumetric --export-glb

Accepts a single ``.npz`` clip, a directory of ``.npz`` clips, or a ``.txt`` clip list
(for monocular video extraction, run ``myosuite-instanthmr-extract`` from
``tools/instanthmr`` first). Outputs are written in the layout read by ``myoMimic-v0``'s
motion library, stamped with ``model_fingerprint`` and accompanied by
``retarget_summary.json``, ``retarget_summary.md`` and ``retarget_clips.jsonl``.
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

MAX_WORKERS = 8
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
STRAIGHT_WALK = re.compile(
    r"(?i)(walkingstraightforward|walking_(slow|medium|fast)|walk_6m|walking_forward_4steps)"
)
_STATE: dict[str, Any] = {}


def _compile_target(
    model_name: str,
    *,
    scales: dict[str, float] | None = None,
    body_mass: float | None = None,
) -> tuple[Any, str]:
    from myosuite.biomechanics import retarget as rt  # noqa: PLC0415
    from myosuite.biomechanics import retarget_instanthmr as ri  # noqa: PLC0415

    imap = ri.INSTANTHMR_MAPS[model_name]
    if model_name == "myolegs":
        from myosuite import make_env  # noqa: PLC0415

        env = make_env("myoLegWalk-v0")
        try:
            if scales is not None:
                spec = ri.scale_spec_from_instanthmr(
                    env.unwrapped._mj_spec, scales, body_mass=body_mass
                )
                model = spec.compile()
            else:
                model = env.unwrapped.model.__copy__()
        finally:
            env.close()
        fp = rt.kinematic_fingerprint(model, ri.landmark_bodies(imap), imap.feet)
        return model, fp

    from myosuite.envs.myo.tasks.mimic.tracking.models import (  # noqa: PLC0415
        compiled_body,
        kinematic_fingerprint,
        model_cfg,
    )

    if scales is not None:
        spec = ri.scale_spec_from_instanthmr(
            model_cfg(model_name).build_spec(True), scales, body_mass=body_mass
        )
        model = spec.compile()
        fp = rt.kinematic_fingerprint(model, ri.landmark_bodies(imap), imap.feet)
        return model, fp
    return compiled_body(model_name), kinematic_fingerprint(model_name)


def _init(
    model_name: str,
    version: str,
    trim: int,
    max_gap_s: float = 0.25,
    on_long_gap: str = "reject",
    scales: dict[str, float] | None = None,
    body_mass: float | None = None,
    reconstruct_muscles: str = "none",
    activation: str = "shortening",
    export_glb: bool = False,
    glb_frame: int = 0,
) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    import warnings  # noqa: PLC0415

    warnings.filterwarnings("ignore")
    from myosuite.biomechanics import retarget as rt  # noqa: PLC0415
    from myosuite.biomechanics import retarget_instanthmr as ri  # noqa: PLC0415

    model, fp = _compile_target(model_name, scales=scales, body_mass=body_mass)
    src, lmap = ri.instanthmr_source(
        model,
        ri.INSTANTHMR_MAPS[model_name],
        trim=(trim, trim),
        max_gap_s=max_gap_s,
        metric_targets=scales is not None,
    )
    _STATE.update(
        rt=rt,
        ri=ri,
        model=model,
        lmap=lmap,
        name=model_name,
        version=version,
        fingerprint=fp,
        ik=rt.LandmarkIK(model, lmap),
        source=src,
        on_long_gap=on_long_gap,
        scales=scales,
        body_mass=body_mass,
        reconstruct_muscles=reconstruct_muscles,
        activation=activation,
        export_glb=export_glb,
        glb_frame=glb_frame,
    )


def _export_muscles(
    dst_path: Path, model: Any, qpos: np.ndarray, freq: float
) -> dict[str, Any]:
    from myosuite.viz.muscle_meshes import reconstruct_muscle_meshes  # noqa: PLC0415

    s = _STATE
    mode = s["reconstruct_muscles"]
    if mode == "none" and not s["export_glb"]:
        return {}
    mesh_mode = "volumetric" if mode == "none" else mode
    recon = reconstruct_muscle_meshes(
        model,
        qpos,
        freq,
        activation_mode=s["activation"],
        mesh_mode=mesh_mode,
    )
    npz_out = dst_path.with_name(f"{dst_path.stem}_muscles.npz")
    recon.save_npz(npz_out)
    info: dict[str, Any] = {
        "muscle_mesh_mode": recon.mode,
        "muscle_activation_mode": s["activation"],
        "muscle_count": recon.num_muscles,
        "muscle_npz": npz_out.name,
    }
    if s["export_glb"]:
        frame = min(max(int(s["glb_frame"]), 0), recon.num_frames - 1)
        glb_out = dst_path.with_name(f"{dst_path.stem}_muscles.glb")
        recon.write_glb(
            glb_out, frame=frame, model=model, qpos=qpos, include_bones=True
        )
        info["muscle_glb"] = glb_out.name
    return info


def _twist(
    rt: Any, model: Any, lmap: Any, qpos: np.ndarray, dt: float
) -> dict[str, Any]:
    tw = rt.twist_asymmetry(model, lmap, qpos, dt)
    if tw is None:
        return {}
    return {
        "fit": {
            "period_s": tw["period_s"],
            "slope": tw["slope"],
            "x": tw["x"].astype(np.float32),
            "y": {k: v.astype(np.float32) for k, v in tw["y"].items()},
        }
    }


def _fit_and_write(
    clip_input: Any,
    src_file: str,
    dst_path: Path,
    label: str,
    curation: dict[str, Any],
    cur_flags: Any,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    s = _STATE
    rt, model, lmap = s["rt"], s["model"], s["lmap"]
    qpos, freq, meta = rt.refit_clip(
        clip_input,
        model,
        s["name"],
        s["version"],
        lmap,
        s["source"],
        ik=s["ik"],
        model_fingerprint=s["fingerprint"],
        source_path=src_file,
    )
    meta["source"] = label
    meta["scaled"] = s["scales"] is not None
    if s["scales"] is not None:
        meta["subject_scales"] = s["scales"]
        if s["body_mass"] is not None:
            meta["subject_body_mass_kg"] = float(s["body_mass"])
    meta.update(_export_muscles(dst_path, model, qpos, freq))
    meta.update(curation)
    meta["quality_flags"] = rt.merge_quality_flags(meta.get("quality_flags"), cur_flags)
    rt.save_refit(dst_path, model, qpos, freq, meta)
    return qpos, freq, meta


def _work(
    job: tuple[str, str, str, dict[str, Any], dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    src, dst, label, curation, seg_curations = job
    s = _STATE
    rt, ri, model, lmap = s["rt"], s["ri"], s["model"], s["lmap"]
    curation = dict(curation)
    cur_flags = curation.pop("quality_flags", None)
    dst_path = Path(dst)
    try:
        if s["on_long_gap"] == "split":
            segments = ri.split_instanthmr_clips(src, max_gap_s=s["source"].max_gap_s)
            if len(segments) > 1:
                metas = []
                for idx, seg in enumerate(segments):
                    seg_dst = dst_path.with_name(f"{dst_path.stem}_seg{idx:02d}.npz")
                    seg_label = (
                        f"{label[:-4]}_seg{idx:02d}.npz"
                        if label.endswith(".npz")
                        else f"{label}_seg{idx:02d}"
                    )
                    seg_cur = {**curation, **seg_curations.get(seg_label, {})}
                    seg_flags = rt.merge_quality_flags(
                        cur_flags, seg_cur.pop("quality_flags", None)
                    )
                    _, _, seg_meta = _fit_and_write(
                        seg, src, seg_dst, seg_label, seg_cur, seg_flags
                    )
                    metas.append(seg_meta)
                return {"metas": metas, "skipped": False}

        meta = rt.is_cached_refit(
            dst_path,
            src,
            fingerprint=s["fingerprint"],
            model_version=s["version"],
            landmark_map_version=lmap.version,
            solver=dataclasses.asdict(s["ik"].config),
            extra_checks={
                "instanthmr_map_version": s["source"].imap.version,
                "trim": list(s["source"].trim),
                "scaled": s["scales"] is not None,
            },
        )
        if meta is not None:
            with np.load(dst_path, allow_pickle=False) as d:
                qpos, freq = d["qpos"], float(d["frequency"])
            m_info = _export_muscles(dst_path, model, qpos, freq)
            updates = {
                "model_fingerprint": s["fingerprint"],
                **m_info,
                **curation,
                "quality_flags": rt.merge_quality_flags(
                    meta.get("quality_flags"), cur_flags
                ),
            }
            if any(meta.get(k) != v for k, v in updates.items()):
                meta.update(updates)
                rt.save_refit(dst_path, model, qpos, freq, meta)
            return {"metas": [meta], "skipped": True}

        clip_in = segments[0] if s["on_long_gap"] == "split" else src
        qpos, freq, meta = _fit_and_write(
            clip_in, src, dst_path, label, curation, cur_flags
        )
        res: dict[str, Any] = {"metas": [meta], "skipped": False}
        if STRAIGHT_WALK.search(Path(src).name):
            res["twist"] = _twist(rt, model, lmap, qpos, 1.0 / freq)
        return res
    except Exception as err:
        return {"error": f"{type(err).__name__}: {err}", "source": label}


def stamp(root: Path, model_name: str) -> int:
    """Record the installed body's fingerprint in every refitted clip under *root*."""
    import myo_sim  # noqa: PLC0415

    from myosuite.biomechanics import retarget as rt  # noqa: PLC0415

    _, fp = _compile_target(model_name)
    return rt.stamp_tree(root, model_name, str(myo_sim.__version__), fp)


def main(argv: list[str] | None = None) -> int:
    """Retarget InstantHMR ``.npz`` clips; returns the number of failed clips."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "input",
        nargs="?",
        help="InstantHMR .npz clip, directory, or .txt clip list",
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
        help="sub-folder of the outputs (default: input dir or 'instanthmr')",
    )
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--report-dir", default="")
    p.add_argument("--limit", type=int, default=0, help="only the first N clips")
    p.add_argument("--trim", type=int, default=0, help="frames dropped at each end")
    p.add_argument(
        "--max-gap-s",
        type=float,
        default=0.25,
        help="maximum missing-frame gap duration (s) interpolated across",
    )
    p.add_argument(
        "--on-long-gap",
        default="reject",
        choices=("reject", "split"),
        help="reject clips with missing gaps > --max-gap-s or split into sub-clips",
    )
    p.add_argument(
        "--quality-flags",
        default="",
        help="JSON {clip label: metadata fields} merged into the clips",
    )
    p.add_argument(
        "--scale",
        action="store_true",
        help="scale the musculoskeletal model to the subject's metric MHR70 skeleton",
    )
    p.add_argument(
        "--body-mass",
        type=float,
        default=None,
        help="subject total body mass in kg (used with --scale)",
    )
    p.add_argument(
        "--reconstruct-muscles",
        default="none",
        help="reconstruct 3D muscle meshes: none, volumetric, atlas, atlas-hd, or .glb path",
    )
    p.add_argument(
        "--activation",
        default="shortening",
        choices=("shortening", "static_optimization", "none"),
        help="muscle activation estimator for reconstructed muscle meshes",
    )
    p.add_argument(
        "--export-glb",
        action="store_true",
        help="write a posed <clip>_muscles.glb scene alongside retargeted clips",
    )
    p.add_argument(
        "--glb-frame",
        type=int,
        default=0,
        help="frame index exported when --export-glb is enabled",
    )
    args = p.parse_args(argv)

    if args.stamp_fingerprint:
        return stamp(Path(args.stamp_fingerprint).expanduser(), args.model)
    if not args.input:
        p.error("input is required unless --stamp-fingerprint is set")

    src = Path(args.input).expanduser()
    if src.suffix.lower() in VIDEO_SUFFIXES:
        raise ValueError(
            f"{src}: monocular video extraction lives in the standalone "
            f"'tools/instanthmr' package; run 'myosuite-instanthmr-extract {src} "
            f"-o <clip.npz>' first and pass the extracted .npz here."
        )

    import myo_sim  # noqa: PLC0415

    from myosuite.biomechanics import retarget as rt  # noqa: PLC0415
    from myosuite.biomechanics import retarget_instanthmr as ri  # noqa: PLC0415

    version = str(myo_sim.__version__)
    out_root = (
        Path(args.out_root).expanduser()
        if args.out_root
        else rt.default_cache_root(args.model, version)
    )
    default_ds = args.dataset or (
        "instanthmr"
        if src.is_file() and src.suffix == ".npz"
        else src.stem
        if src.is_file()
        else ""
    )
    files, base, dataset = rt.collect_clip_files(
        src, default_dataset=default_ds, limit=args.limit
    )

    imap = ri.INSTANTHMR_MAPS[args.model]
    scales: dict[str, float] | None = None
    if args.scale and files:
        nominal, _ = _compile_target(args.model)
        scale_clip: Any = (
            ri.split_instanthmr_clips(files[0], max_gap_s=args.max_gap_s)[0]
            if args.on_long_gap == "split"
            else files[0]
        )
        scales = ri.estimate_instanthmr_scales(
            scale_clip,
            nominal,
            imap,
            trim=(args.trim, args.trim),
            max_gap_s=args.max_gap_s,
        )

    curation = (
        json.loads(Path(args.quality_flags).read_text()) if args.quality_flags else {}
    )
    jobs = []
    for f in files:
        rel_posix = f.relative_to(base).as_posix()
        label = f"InstantHMR/{dataset}/{rel_posix}"
        seg_prefix = f"{label[:-4]}_seg" if label.endswith(".npz") else f"{label}_seg"
        seg_cur = {k: v for k, v in curation.items() if k.startswith(seg_prefix)}
        dst = str(out_root / dataset / f.relative_to(base))
        jobs.append((str(f), dst, label, curation.get(label, {}), seg_cur))

    init_args = (
        args.model,
        version,
        args.trim,
        args.max_gap_s,
        args.on_long_gap,
        scales,
        args.body_mass,
        args.reconstruct_muscles,
        args.activation,
        args.export_glb,
        args.glb_frame,
    )
    workers = (
        1
        if args.workers <= 1 or len(jobs) <= 1
        else rt.choose_workers(
            min(args.workers, len(jobs)), max_workers=MAX_WORKERS, verbose=True
        )
    )
    t0 = time.time()
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
            results = list(pool.imap_unordered(_work, jobs, chunksize=1))

    metas = [m for r in results for m in r.get("metas", ())]
    errors = [
        {"source": r["source"], "error": r["error"]} for r in results if "error" in r
    ]
    summary = rt.summarize(metas) if metas else rt.empty_summary()
    summary.update(
        model=args.model,
        model_version=version,
        landmark_map=imap.name,
        landmark_map_version=imap.version,
        trim=args.trim,
        max_gap_s=args.max_gap_s,
        on_long_gap=args.on_long_gap,
        scaled=scales is not None,
        **({"subject_scales": scales} if scales is not None else {}),
        **(
            {"subject_body_mass_kg": args.body_mass}
            if args.body_mass is not None
            else {}
        ),
        reconstruct_muscles=args.reconstruct_muscles,
        out_root=str(out_root / dataset),
        wall_s=time.time() - t0,
        workers=workers,
        errors=errors,
        thresholds=dataclasses.asdict(rt.RefitThresholds()),
        twist=rt.pooled_twist(results),
    )
    report = Path(args.report_dir) if args.report_dir else (out_root / dataset)
    report.mkdir(parents=True, exist_ok=True)
    if scales is not None:
        (report / "subject_scales.json").write_text(
            json.dumps(
                {"model": args.model, "scales": scales, "body_mass_kg": args.body_mass},
                indent=2,
            )
            + "\n"
        )
    print(rt.write_refit_report(report, summary, metas, title="InstantHMR retarget"))
    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
