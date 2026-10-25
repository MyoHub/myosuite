# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Opt-in real anatomy landmark validation without redistributing source data."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy import stats
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from myosuite.integrations.capture.landmark_scaling import (  # noqa: E402
    LandmarkCorrespondences,
    fit_landmark_scales,
)
from myosuite.integrations.keast import (  # noqa: E402
    PALPABLE_LANDMARKS,
    KeastReference,
    resample_tibial_borders,
    tibial_landmark_frame,
)


def _rigid_pose(
    reference: np.ndarray, observed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rotation = Rotation.align_vectors(
        observed - observed.mean(axis=0), reference - reference.mean(axis=0)
    )[0].as_matrix()
    return rotation, observed.mean(axis=0) - reference.mean(axis=0) @ rotation.T


def validate(reference: KeastReference, subject_count: int = 35) -> dict[str, Any]:
    """Fit six named points per subject and evaluate separate border landmarks.

    The 35 MRI subjects are distinct from the 30 CT subjects of the source mean.
    Border evaluation points are never supplied to the scale solver. Their
    fractions use the palpable endpoints, so they are correlated measurements.
    This measures landmark geometry, not full surfaces or target joint accuracy.
    """
    if not 1 <= subject_count <= 35:
        raise ValueError("subject_count must be between 1 and 35.")
    mean = reference.read_landmarks("mean", units="mm", rotation_to_model=np.eye(3))
    origin, basis = tibial_landmark_frame(mean)
    reference_points = (mean.in_model_frame()[0] - origin) @ basis
    reference_by_label = dict(zip(mean.labels, reference_points))
    fit_reference = np.array(
        [reference_by_label[label] for label in PALPABLE_LANDMARKS]
    )
    border_reference = resample_tibial_borders(mean)
    rows = []
    for index in range(1, subject_count + 1):
        case = f"C{index:02d}RTF"
        observed = reference.read_landmarks(
            case, units="mm", rotation_to_model=np.eye(3)
        )
        observed_by_label = dict(zip(observed.labels, observed.in_model_frame()[0]))
        fit_observed = np.array(
            [observed_by_label[label] for label in PALPABLE_LANDMARKS]
        )
        border_observed = resample_tibial_borders(observed)
        heldout = list(border_reference)
        eval_reference = (
            np.array([border_reference[label] for label in heldout]) - origin
        ) @ basis
        eval_observed = np.array([border_observed[label] for label in heldout])
        problem = LandmarkCorrespondences(
            subject_id=case,
            segment="tibia_reference",
            labels=PALPABLE_LANDMARKS,
            reference_positions_m=fit_reference,
            observed_positions_m=fit_observed,
            reference_source=mean.source,
            observed_source=observed.source,
        )
        methods = {}
        length_factor = np.linalg.norm(
            observed_by_label["MC"] - observed_by_label["MM"]
        ) / np.linalg.norm(reference_by_label["MC"] - reference_by_label["MM"])
        for method in ("nominal", "single_length", "isotropic", "anisotropic"):
            if method in ("nominal", "single_length"):
                factors = np.array(
                    [1.0, length_factor if method == "single_length" else 1.0, 1.0]
                )
                rotation, translation = _rigid_pose(
                    fit_reference * factors, fit_observed
                )
            else:
                result = fit_landmark_scales([problem], mode=method, loss="linear")
                factors = np.array(result.estimate.factors["tibia_reference"])
                evidence = result.evidence["segments"][0]
                rotation = np.array(evidence["rotation_reference_to_observed"])
                translation = np.array(evidence["translation_m"])
            fit_error = np.linalg.norm(
                (fit_reference * factors) @ rotation.T + translation - fit_observed,
                axis=1,
            )
            eval_error = np.linalg.norm(
                (eval_reference * factors) @ rotation.T + translation - eval_observed,
                axis=1,
            )
            methods[method] = {
                "factors": factors.tolist(),
                "fit_rms_m": float(np.sqrt(np.mean(fit_error**2))),
                "heldout_rms_m": float(np.sqrt(np.mean(eval_error**2))),
                "heldout_max_m": float(np.max(eval_error)),
            }
        rows.append(
            {
                "case": case,
                "source_sha256": observed.metadata()["landmark_csv"]["sha256"],
                "fit_count": len(PALPABLE_LANDMARKS),
                "heldout_count": len(heldout),
                "methods": methods,
            }
        )
    summary = {}
    for method in rows[0]["methods"]:
        errors = np.array([row["methods"][method]["heldout_rms_m"] for row in rows])
        summary[method] = {
            "mean_heldout_rms_m": float(np.mean(errors)),
            "median_heldout_rms_m": float(np.median(errors)),
            "worst_heldout_rms_m": float(np.max(errors)),
        }
    comparisons = {}
    for baseline in ("nominal", "single_length", "isotropic"):
        differences = np.array(
            [
                row["methods"]["anisotropic"]["heldout_rms_m"]
                - row["methods"][baseline]["heldout_rms_m"]
                for row in rows
            ]
        )
        comparisons[baseline] = {
            "improved_subjects": int(np.count_nonzero(differences < 0)),
            "regressed_subjects": int(np.count_nonzero(differences > 0)),
            "mean_delta_m": float(np.mean(differences)),
            "worst_delta_m": float(np.max(differences)),
            "mean_delta_95pct_t_interval_m": None
            if len(differences) < 2
            else [
                float(
                    np.mean(differences)
                    + sign
                    * stats.t.ppf(0.975, len(differences) - 1)
                    * stats.sem(differences)
                )
                for sign in (-1, 1)
            ],
        }
    return {
        "schema": 2,
        "source": "https://doi.org/10.6084/m9.figshare.20454462.v4",
        "attribution": "Keast, Bonacci and Fox (2023), doi:10.7717/peerj.14708; Nolte et al. (2016), doi:10.1016/j.jbiomech.2016.09.005",
        "source_license": "CC-BY-4.0 source archive; source files remain local",
        "report_license": "CC-BY-4.0",
        "reference_sha256": mean.metadata()["landmark_csv"]["sha256"],
        "archive_sha256": mean.metadata()["landmark_csv"]["archive_sha256"],
        "fitter_source_sha256": hashlib.sha256(
            Path(inspect.getfile(fit_landmark_scales)).read_bytes()
        ).hexdigest(),
        "environment": {"numpy": np.__version__, "scipy": scipy.__version__},
        "reference_cohort": "30 CT training subjects; mean reference",
        "evaluation_cohort": "independent Nolte MRI subjects; no fitting of population hyperparameters",
        "frame_protocol": "origin condyle midpoint; y proximal; x anterior toward tuberosity; z right handed; not automatically a target body frame",
        "fit_landmarks": list(PALPABLE_LANDMARKS),
        "border_protocol": "Piecewise-linear T25/T50/T75 percent MC-to-MM and F25/F50/F75 percent FH-to-LM longitudinal fractions, projected onto each subject's anatomical proximal axis. Raw T/F sample indices are not correspondences. No surface snapping or extrapolation.",
        "fit_settings": {"loss": "linear", "scale_bounds": [0.5, 2.0]},
        "scope": "Withheld border geometry only; fractions share fitted endpoints and are correlated. Not full-surface, joint-centre, muscle or dynamics validation; this does not reproduce the source SSM reconstruction experiment.",
        "paired_interval_assumptions": "Descriptive Student-t intervals over per-subject paired RMS differences; assume independent subject deltas in this selected 35-person cohort, not a representative population sample. Source annotation measurement uncertainty is not supplied or inferred.",
        "summary": summary,
        "anisotropic_comparisons": comparisons,
        "subjects": rows,
    }


def plot_report(report: dict[str, Any], path: Path) -> None:
    """Plot derivative errors only; no source landmarks or surfaces are exported."""
    import matplotlib.pyplot as plt

    methods = ("nominal", "single_length", "isotropic", "anisotropic")
    names = (
        "Rigid\nnominal",
        "One length\naxis",
        "Uniform\nscale",
        "Three-axis\nscale",
    )
    errors = [
        report["summary"][method]["mean_heldout_rms_m"] * 1000 for method in methods
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), layout="constrained")
    bars = axes[0].bar(
        names, errors, color=("#8fa4ba", "#8fa4ba", "#8fa4ba", "#d8873b")
    )
    axes[0].bar_label(bars, fmt="%.2f", padding=3)
    axes[0].set(
        ylabel="Mean withheld-border RMS (mm)",
        ylim=(0, max(errors) * 1.2),
        title="Average geometry error",
    )
    isotropic = np.array(
        [
            row["methods"]["isotropic"]["heldout_rms_m"] * 1000
            for row in report["subjects"]
        ]
    )
    anisotropic = np.array(
        [
            row["methods"]["anisotropic"]["heldout_rms_m"] * 1000
            for row in report["subjects"]
        ]
    )
    improved = anisotropic < isotropic
    axes[1].scatter(
        isotropic[improved],
        anisotropic[improved],
        color="#27806e",
        label="Three-axis improves",
    )
    axes[1].scatter(
        isotropic[~improved],
        anisotropic[~improved],
        color="#c76a5e",
        label="Three-axis regresses",
    )
    maximum = max(float(np.max(isotropic)), float(np.max(anisotropic))) * 1.08
    axes[1].plot(
        [0, maximum], [0, maximum], color="#82909e", linestyle="--", linewidth=1
    )
    axes[1].set(
        xlabel="Uniform-scale RMS (mm)",
        ylabel="Three-axis RMS (mm)",
        xlim=(0, maximum),
        ylim=(0, maximum),
        title=f"{int(improved.sum())}/{len(improved)} subjects improve over uniform",
    )
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"Six palpable fit points; six withheld border fractions; {len(report['subjects'])} MRI subjects",
        fontsize=11,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    """Run the optional source download and write a derivative-only report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--subjects", type=int, default=35)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    reference = KeastReference(args.data_root)
    if args.download or args.archive is not None:
        reference.prepare(args.archive)
    report = validate(reference, args.subjects)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if args.plot is not None:
        plot_report(report, args.plot)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
