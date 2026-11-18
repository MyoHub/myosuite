# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""CLI for musculoskeletal model-data EMG validation and PR A/B comparisons."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mujoco

from myosuite.biomechanics.emg_validation import (
    EMGChannelMap,
    EMGSource,
    EMGTrial,
    compare_emg_validation_reports,
    export_mm_evalkit_rollout,
    musculoskeletal_fingerprint,
    save_emg_trial,
    summarize_emg_validation,
    validate_trial,
    write_mm_evalkit_policy_config,
)
from myosuite.biomechanics.static_optimization import (
    StaticOptimizationConfig,
)
from myosuite.biomechanics.uncertainty import UncertaintyConfig
from myosuite.integrations.mm_evalkit.plugin import _build_npz_directory_source

DATASET_CHOICES: tuple[str, ...] = (
    "rajagopal2016",
    "lencioni2019",
    "dimitrov2023",
    "camargo2021",
    "quesada2024",
    "emg2pose",
    "ninapro_db5",
    "grabmyo",
    "fritsche2026",
    "maurice2020",
    "diaz2026",
    "custom",
)

MANIFEST_TEMPLATE: dict[str, Any] = {
    "name": "custom_lab_emg",
    "signal_kind": "raw",
    "sampling_rate_hz": 2000.0,
    "kinematics_rate_hz": 100.0,
    "filter": {
        "bandpass_hz": [20.0, 450.0],
        "notch_hz": 50.0,
        "lowpass_hz": 6.0,
        "normalize": "mvc",
    },
    "channel_map": {
        "name": "forearm_surface_8ch",
        "version": 1,
        "bindings": [
            {"channel": "FCR", "actuators": ["FCR_r"]},
            {"channel": "FCU", "actuators": ["FCU_r"]},
            {
                "channel": "ECR",
                "actuators": ["ECRL_r", "ECRB_r"],
                "weights": [0.5, 0.5],
            },
            {"channel": "EDC", "actuators": ["EDC2_r", "EDC3_r", "EDC4_r", "EDC5_r"]},
            {"channel": "finger_flexors", "fmg": "Wrist-Flexor", "side": "r"},
        ],
    },
    "trials": [
        {
            "trial_id": "subj01_grasp_01",
            "emg_path": "subj01/grasp_01_emg.csv",
            "kinematics_path": "subj01/grasp_01_angles.mot",
            "mvc_peaks": [0.45, 0.40, 0.38, 0.42, 0.50],
            "window_s": [0.5, 4.5],
        }
    ],
}

_WORKTREE_COMPILE_SNIPPET = """
import sys
import mujoco
wt_dir, model_name, out_mjb = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, wt_dir)
import myo_sim
model = myo_sim.load_model(model_name)
mujoco.mj_saveModel(model, out_mjb, None)
"""


def _compile_worktree_model(worktree_dir: Path, model_name: str) -> mujoco.MjModel:
    with tempfile.TemporaryDirectory() as tmp:
        mjb_path = Path(tmp) / "compiled.mjb"
        subprocess.run(
            [
                sys.executable,
                "-c",
                _WORKTREE_COMPILE_SNIPPET,
                str(worktree_dir.resolve()),
                model_name,
                str(mjb_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return mujoco.MjModel.from_binary_path(str(mjb_path))


def load_target_model(spec: str) -> tuple[mujoco.MjModel, str]:
    """Resolve a worktree ``dir:model``, XML/MJB file, ``myo_sim`` model, mimic body or ``env_id``."""
    direct = Path(spec)
    if direct.is_file():
        if direct.suffix.lower() == ".mjb":
            return mujoco.MjModel.from_binary_path(str(direct)), direct.name
        return mujoco.MjModel.from_xml_path(str(direct)), direct.name

    sep = spec.rfind(":")
    if sep > 1:
        wt_part, model_name = spec[:sep], spec[sep + 1 :]
        if wt_part == "myo_sim":
            import myo_sim

            return myo_sim.load_model(model_name), f"myo_sim:{model_name}"
        wt_dir = Path(wt_part)
        if wt_dir.is_dir():
            return _compile_worktree_model(
                wt_dir, model_name
            ), f"{wt_dir.name}:{model_name}"

    from myosuite.envs.myo.tasks.mimic.tracking.models import (
        MODELS,
        compiled_scene,
    )

    if spec in MODELS:
        return copy.deepcopy(compiled_scene(spec, self_collisions=False)), spec

    import myo_sim

    if hasattr(myo_sim, "ALL_MODELS") and spec in myo_sim.ALL_MODELS:
        return myo_sim.load_model(spec), spec
    sim_candidate = Path(myo_sim.__file__).resolve().parent / spec
    if sim_candidate.is_file():
        return mujoco.MjModel.from_xml_path(str(sim_candidate)), spec

    from myosuite import make_env

    env = make_env(spec)
    try:
        unwrapped = env.unwrapped
        model = (
            unwrapped.model
            if hasattr(unwrapped, "model")
            else unwrapped.sim.model._model
        )
        return copy.deepcopy(model), spec
    finally:
        env.close()


def build_source(args: argparse.Namespace) -> EMGSource:
    """Instantiate the requested :class:`EMGSource`."""
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from scripts.validation.emg_sources import (
        Camargo2021Source,
        Dimitrov2023Source,
        Lencioni2019Source,
        ManifestEMGSource,
        RajagopalWalkingSource,
    )
    from scripts.validation.emg_upper_limb import (
        Diaz2026HandSource,
        Maurice2020Source,
        Emg2PoseHandSource,
        GrabMyoSource,
        NinaProDB5Source,
        Quesada2024UpperLimbSource,
        Fritsche2026Source,
    )

    subjects = [s.strip() for s in args.subjects.split(",")] if args.subjects else None
    if args.dataset == "rajagopal2016":
        root = args.data_root or "downloads/emg/rajagopal2016_opensim"
        return RajagopalWalkingSource(root, kinematics_rate_hz=args.kinematics_rate)
    if args.dataset == "lencioni2019":
        root = args.data_root or "downloads/emg/lencioni2019"
        return Lencioni2019Source(
            root,
            subjects=subjects,
            pattern=args.filter,
            kinematics_rate_hz=args.kinematics_rate,
        )
    if args.dataset == "dimitrov2023":
        root = args.data_root or "downloads/emg/dimitrov2023"
        return Dimitrov2023Source(
            root,
            subjects=subjects,
            pattern=args.filter,
            kinematics_rate_hz=args.kinematics_rate,
        )
    if args.dataset == "camargo2021":
        root = args.data_root or "downloads/emg/camargo2021"
        return Camargo2021Source(
            root,
            subjects=subjects,
            pattern=args.filter,
            kinematics_rate_hz=args.kinematics_rate,
        )
    if args.dataset == "quesada2024":
        root = args.data_root or "downloads/emg/quesada2024"
        win_s = args.window_s if args.window_s is not None else 8.0
        return Quesada2024UpperLimbSource(
            root,
            subjects=subjects,
            pattern=args.filter,
            window_duration_s=win_s,
            kinematics_rate_hz=args.kinematics_rate,
        )
    if args.dataset == "emg2pose":
        root = args.data_root or "downloads/emg/emg2pose"
        win_s = args.window_s if args.window_s is not None else 4.0
        return Emg2PoseHandSource(
            root,
            window_duration_s=win_s,
            kinematics_rate_hz=args.kinematics_rate,
        )
    if args.dataset == "ninapro_db5":
        root = args.data_root or "downloads/emg/ninapro_db5"
        return NinaProDB5Source(
            root,
            subjects=subjects,
            pattern=args.filter,
            kinematics_rate_hz=args.kinematics_rate,
        )
    if args.dataset == "grabmyo":
        root = args.data_root or "downloads/emg/grabmyo"
        return GrabMyoSource(
            root,
            subjects=subjects,
            pattern=args.filter,
            kinematics_rate_hz=args.kinematics_rate,
        )
    if args.dataset == "fritsche2026":
        root = args.data_root or "downloads/emg/fritsche2026"
        win_s = args.window_s if args.window_s is not None else 8.0
        return Fritsche2026Source(
            root,
            subjects=subjects,
            pattern=args.filter,
            window_duration_s=win_s,
            kinematics_rate_hz=args.kinematics_rate,
        )
    if args.dataset == "maurice2020":
        root = args.data_root or "downloads/emg/maurice2020"
        return Maurice2020Source(
            root,
            subjects=subjects,
            pattern=args.filter,
            kinematics_rate_hz=args.kinematics_rate,
        )
    if args.dataset == "diaz2026":
        root = args.data_root or "downloads/emg/diaz2026"
        return Diaz2026HandSource(
            root,
            subjects=subjects,
            pattern=args.filter,
            max_window_s=args.window_s if args.window_s is not None else 10.0,
            kinematics_rate_hz=args.kinematics_rate,
        )
    if args.dataset == "custom":
        if args.npz_dir:
            return _build_npz_directory_source(args.npz_dir)
        if not args.manifest:
            raise ValueError(
                "--manifest or --npz-dir is required when --dataset custom"
            )
        return ManifestEMGSource(args.manifest)
    raise ValueError(f"unsupported dataset {args.dataset!r}")


def _evaluate_model_on_source(
    model: mujoco.MjModel,
    model_label: str,
    source: EMGSource,
    trial_ids: Sequence[str],
    *,
    override_cmap: EMGChannelMap | None,
    rollout_audit: bool,
    tolerance_band: float,
    trim_s: float,
    contact_fallback: bool,
    uncertainty_samples: int,
    export_npz_dir: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[EMGTrial]]:
    cmap = (
        override_cmap
        if override_cmap is not None
        else source.default_channel_map(model)
    )
    unc_cfg = (
        UncertaintyConfig(
            samples=uncertainty_samples,
            joint_noise_deg=0.5,
            strength_cv=0.15,
            mass_cv=0.05,
        )
        if uncertainty_samples > 0
        else None
    )
    reports: list[dict[str, Any]] = []
    loaded_trials: list[EMGTrial] = []
    t_start = time.perf_counter()

    for tid in trial_ids:
        t0 = time.perf_counter()
        trial = source.get_trial(tid, model=model)
        loaded_trials.append(trial)
        if export_npz_dir is not None:
            safe_name = tid.replace("/", "__").replace("\\", "__") + ".npz"
            save_emg_trial(export_npz_dir / safe_name, trial)

        use_contacts = contact_fallback and bool(
            trial.meta.get("contact_forces_fallback")
        )
        supported = trial.supported_dofs or tuple(
            int(d) for d in trial.meta.get("supported_dofs", ())
        )
        opt_cfg = StaticOptimizationConfig(
            contact_forces=use_contacts,
            contact_distance=0.03 if use_contacts else 0.01,
            supported_dofs=supported,
        )
        ctrl_qadr = trial.meta.get("controlled_qpos_adr")
        rep = validate_trial(
            model,
            trial,
            channel_map=cmap,
            trim_s=trim_s,
            tolerance_band=tolerance_band,
            include_rollout_audit=rollout_audit,
            controlled_qpos_adr=ctrl_qadr if ctrl_qadr else None,
            optimization=opt_cfg,
            uncertainty=unc_cfg,
        )
        rep["elapsed_seconds"] = time.perf_counter() - t0
        rep["model"] = model_label
        reports.append(rep)

    agg = summarize_emg_validation(reports)
    agg["model"] = model_label
    agg["model_fingerprint"] = musculoskeletal_fingerprint(model)
    agg["dataset"] = source.name
    agg["elapsed_seconds"] = time.perf_counter() - t_start
    return reports, agg, loaded_trials


def _append_flagged_section(
    lines: list[str], label: str, summary: Mapping[str, Any]
) -> None:
    flagged = summary.get("flagged", [])
    lines.extend(
        [
            "",
            f"## Flagged Trials — `{label}` ({len(flagged)} / {summary['trials']})",
            "",
        ]
    )
    if flagged:
        for item in flagged[:25]:
            lines.append(f"- `{item['trial_id']}`: `{', '.join(item['flags'])}`")
    else:
        lines.append("- None")


def _render_markdown(
    summary_a: Mapping[str, Any],
    summary_b: Mapping[str, Any] | None = None,
    *,
    comparison: Mapping[str, Any] | None = None,
) -> str:
    lines = [
        f"# EMG Validation Report — `{summary_a['dataset']}`",
        "",
        f"- **Model A**: `{summary_a['model']}` (`fingerprint={summary_a['model_fingerprint']}`)",
    ]
    if summary_b is not None:
        lines.append(
            f"- **Model B**: `{summary_b['model']}` (`fingerprint={summary_b['model_fingerprint']}`)"
        )
    lines.extend(
        [
            f"- **Trials evaluated**: {summary_a['trials']}",
            f"- **Elapsed**: {summary_a['elapsed_seconds']:.2f} s",
            "",
            "## Aggregate Metrics",
            "",
        ]
    )

    stats_a = summary_a.get("stats", {})
    if summary_b is None:
        lines.extend(
            [
                "| Metric | Median | Mean | P95 | Min | Max |",
                "| :--- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for key, vals in stats_a.items():
            lines.append(
                f"| `{key}` | {vals['median']:.4f} | {vals['mean']:.4f} | "
                f"{vals['p95']:.4f} | {vals['min']:.4f} | {vals['max']:.4f} |"
            )
    else:
        stats_b = summary_b.get("stats", {})
        lines.extend(
            [
                f"| Metric | `{summary_a['model']}` (Median) | `{summary_b['model']}` (Median) | Delta (B - A) |",
                "| :--- | ---: | ---: | ---: |",
            ]
        )
        for key in sorted(set(stats_a) | set(stats_b)):
            va = stats_a.get(key, {}).get("median")
            vb = stats_b.get(key, {}).get("median")
            sa = f"{va:.4f}" if va is not None else "—"
            sb = f"{vb:.4f}" if vb is not None else "—"
            sd = f"{vb - va:+.4f}" if (va is not None and vb is not None) else "—"
            lines.append(f"| `{key}` | {sa} | {sb} | {sd} |")

    per_ch_a = summary_a.get("per_channel_r_opt_lag_median", {})
    per_ch_b = summary_b.get("per_channel_r_opt_lag_median", {}) if summary_b else {}
    if per_ch_a or per_ch_b:
        lines.extend(
            ["", "## Per-Channel Waveform Correlation (`r_opt_lag` Median)", ""]
        )
        if summary_b is None:
            lines.extend(["| Channel | `r_opt_lag` |", "| :--- | ---: |"])
            for ch, val in per_ch_a.items():
                lines.append(f"| `{ch}` | {val:.4f} |")
        else:
            lines.extend(
                [
                    f"| Channel | `{summary_a['model']}` | `{summary_b['model']}` | Delta |",
                    "| :--- | ---: | ---: | ---: |",
                ]
            )
            for ch in sorted(set(per_ch_a) | set(per_ch_b)):
                va, vb = per_ch_a.get(ch), per_ch_b.get(ch)
                sa = f"{va:.4f}" if va is not None else "—"
                sb = f"{vb:.4f}" if vb is not None else "—"
                sd = f"{vb - va:+.4f}" if (va is not None and vb is not None) else "—"
                lines.append(f"| `{ch}` | {sa} | {sb} | {sd} |")

    if comparison is not None and comparison.get("summary_rows"):
        lines.extend(
            [
                "",
                f"## Paired Cohort Comparison (`{comparison.get('backend', 'wilcoxon')}`, unit=`{comparison.get('unit_name', 'trial')}`)",
                "",
                "| Metric | `n` | Median Diff (A - B) | `p_holm` | Effect Size | Significant |",
                "| :--- | ---: | ---: | ---: | ---: | :---: |",
            ]
        )
        for row in comparison["summary_rows"]:
            mdiff = (
                f"{row['median_diff']:+.4f}"
                if row.get("median_diff") is not None
                else "—"
            )
            pholm = f"{row['p_holm']:.4g}" if row.get("p_holm") is not None else "—"
            eff = (
                f"{row['effect_size']:+.3f}"
                if row.get("effect_size") is not None
                else "—"
            )
            sig = "yes" if row.get("significant") else "no"
            lines.append(
                f"| `{row['metric']}` | {row['n']} | {mdiff} | {pholm} | {eff} | {sig} |"
            )

    _append_flagged_section(lines, str(summary_a["model"]), summary_a)
    if summary_b is not None:
        _append_flagged_section(lines, str(summary_b["model"]), summary_b)
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset", choices=DATASET_CHOICES, help="EMG dataset source to evaluate."
    )
    ap.add_argument(
        "--data-root", help="Dataset directory overriding default location."
    )
    ap.add_argument("--manifest", help="Path to custom JSON/YAML dataset manifest.")
    ap.add_argument(
        "--npz-dir",
        help="Path to a directory of standardized EMGTrial .npz clips (with --dataset custom).",
    )
    ap.add_argument(
        "--channel-map", help="Optional JSON file overriding the channel map."
    )
    ap.add_argument(
        "--model", help="Target worktree:model, XML/MJB path, myo_sim model or env_id."
    )
    ap.add_argument(
        "--model-b", help="Optional second model for side-by-side PR A/B comparison."
    )
    ap.add_argument(
        "--subjects", help="Comma-separated subject IDs (e.g. P01,P02 or S01,S02)."
    )
    ap.add_argument("--filter", help="Regex filter on trial IDs.")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of trials to run (0 = all).",
    )
    ap.add_argument(
        "--kinematics-rate",
        type=float,
        default=100.0,
        help="Kinematics resample rate [Hz].",
    )
    ap.add_argument(
        "--window-s", type=float, default=None, help="Active window duration [s]."
    )
    ap.add_argument(
        "--trim-s",
        type=float,
        default=0.05,
        help="Boundary seconds trimmed when scoring.",
    )
    ap.add_argument(
        "--tolerance-band", type=float, default=0.25, help="EMG feasibility half-band."
    )
    ap.add_argument(
        "--rollout-audit",
        action="store_true",
        help="Run Mode 3 sensitivity + open-loop drift audit.",
    )
    ap.add_argument(
        "--no-contact-fallback",
        action="store_true",
        help="Disable contact estimation when force plates are absent.",
    )
    ap.add_argument(
        "--uncertainty-samples",
        type=int,
        default=0,
        help="Monte Carlo samples for Mode 1 bands (0 = off).",
    )
    ap.add_argument(
        "--export-npz",
        help="Optional directory to export standardized EMGTrial .npz clips.",
    )
    ap.add_argument(
        "--export-mm-evalkit",
        help="Optional directory to export mm-evalkit rollout .npz archive(s) and policies.yaml.",
    )
    ap.add_argument(
        "--rollout-solver",
        choices=("static_optimization", "cmc"),
        default="static_optimization",
        help="Solver used for --export-mm-evalkit rollout generation.",
    )
    ap.add_argument("--out-dir", help="Report output directory.")
    ap.add_argument(
        "--init-manifest",
        help="Write a starter custom manifest JSON to this path and exit.",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.init_manifest:
        dst = Path(args.init_manifest)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(MANIFEST_TEMPLATE, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote custom EMG manifest template -> {dst}")
        return 0

    if not args.dataset or not args.model:
        raise SystemExit(
            "--dataset and --model are required unless --init-manifest is used"
        )

    source = build_source(args)
    all_ids = list(source.trial_ids())
    if args.filter and getattr(source, "pattern", None) is None:
        pat = re.compile(args.filter)
        all_ids = [t for t in all_ids if pat.search(t)]
    trial_ids = all_ids[: args.limit] if args.limit > 0 else all_ids
    if not trial_ids:
        raise SystemExit(f"no trials matched in dataset {source.name!r}")

    override_cmap = (
        EMGChannelMap.from_dict(
            json.loads(Path(args.channel_map).read_text(encoding="utf-8"))
        )
        if args.channel_map
        else None
    )
    export_dir = Path(args.export_npz) if args.export_npz else None

    model_a, label_a = load_target_model(args.model)
    rollout_audit = bool(
        args.rollout_audit or args.dataset in ("emg2pose", "ninapro_db5", "grabmyo")
    )
    reports_a, summary_a, trials_a = _evaluate_model_on_source(
        model_a,
        label_a,
        source,
        trial_ids,
        override_cmap=override_cmap,
        rollout_audit=rollout_audit,
        tolerance_band=args.tolerance_band,
        trim_s=args.trim_s,
        contact_fallback=not args.no_contact_fallback,
        uncertainty_samples=args.uncertainty_samples,
        export_npz_dir=export_dir,
    )

    reports_b: list[dict[str, Any]] | None = None
    summary_b: dict[str, Any] | None = None
    trials_b: list[EMGTrial] | None = None
    label_b: str | None = None
    model_b: mujoco.MjModel | None = None
    comparison: dict[str, Any] | None = None
    if args.model_b:
        model_b, label_b = load_target_model(args.model_b)
        reports_b, summary_b, trials_b = _evaluate_model_on_source(
            model_b,
            label_b,
            source,
            trial_ids,
            override_cmap=override_cmap,
            rollout_audit=rollout_audit,
            tolerance_band=args.tolerance_band,
            trim_s=args.trim_s,
            contact_fallback=not args.no_contact_fallback,
            uncertainty_samples=args.uncertainty_samples,
            export_npz_dir=None,
        )
        comparison = compare_emg_validation_reports(
            reports_a,
            reports_b,
            label_a=label_a,
            label_b=label_b,
        )

    if args.export_mm_evalkit:
        kin_a = [t for t in trials_a if t.qpos is not None]
        if kin_a:
            mm_dir = Path(args.export_mm_evalkit)
            mm_dir.mkdir(parents=True, exist_ok=True)
            slug_a = re.sub(r"[^a-zA-Z0-9_-]+", "_", label_a).strip("_") or "model_a"
            ep_a = (mm_dir / f"{slug_a}_episodes.npz").resolve()
            export_mm_evalkit_rollout(
                model_a,
                kin_a,
                ep_a,
                solver=args.rollout_solver,
                motion_group=source.name,
            )
            policy_entries: list[dict[str, str]] = [
                {
                    "id": slug_a,
                    "label": label_a,
                    "trajectory_data_path": str(ep_a),
                }
            ]
            if model_b is not None and label_b is not None and trials_b is not None:
                kin_b = [t for t in trials_b if t.qpos is not None]
                if kin_b:
                    slug_b = (
                        re.sub(r"[^a-zA-Z0-9_-]+", "_", label_b).strip("_") or "model_b"
                    )
                    if slug_b == slug_a:
                        slug_b = f"{slug_b}_b"
                    ep_b = (mm_dir / f"{slug_b}_episodes.npz").resolve()
                    export_mm_evalkit_rollout(
                        model_b,
                        kin_b,
                        ep_b,
                        solver=args.rollout_solver,
                        motion_group=source.name,
                    )
                    policy_entries.append(
                        {
                            "id": slug_b,
                            "label": label_b,
                            "trajectory_data_path": str(ep_b),
                        }
                    )
            write_mm_evalkit_policy_config(mm_dir / "policies.yaml", policy_entries)

    safe_lbl = label_a.replace("/", "_").replace(":", "_")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("downloads/emg/reports") / f"{source.name}_{safe_lbl}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = (
        summary_a
        if summary_b is None
        else {
            "dataset": source.name,
            "model_a": summary_a,
            "model_b": summary_b,
            "comparison": comparison,
        }
    )
    (out_dir / "emg_validation_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    md = _render_markdown(summary_a, summary_b, comparison=comparison)
    (out_dir / "emg_validation_summary.md").write_text(md, encoding="utf-8")

    with (out_dir / "emg_validation_trials.jsonl").open("w", encoding="utf-8") as fh:
        for rep in reports_a:
            fh.write(json.dumps(rep) + "\n")
        if reports_b is not None:
            for rep in reports_b:
                fh.write(json.dumps(rep) + "\n")

    print(md)
    print(f"Saved validation reports -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
