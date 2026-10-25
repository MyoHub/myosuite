# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Derive aggregate tissue-volume checks from user-provided Denver archives."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from myosuite.integrations.denver import (  # noqa: E402
    DenverReference,
    TissueSummary,
    VolumeScaleConstraint,
    volume_constrained_scale_estimate,
)


def _aggregate(
    reference: DenverReference,
) -> tuple[dict[str, Any], TissueSummary | None]:
    groups: dict[str, list[TissueSummary]] = defaultdict(list)
    tibia = None
    for geometry in reference.geometries():
        summary = reference.summarize_geometry(geometry.name)
        groups[geometry.tissue].append(summary)
        if (
            geometry.tissue == "Bone"
            and geometry.label == "Tibia"
            and geometry.side == "Right"
        ):
            tibia = summary
    totals = {}
    for tissue, summaries in sorted(groups.items()):
        errors = [
            summary.relative_volume_error
            for summary in summaries
            if summary.relative_volume_error is not None
        ]
        totals[tissue] = {
            "geometries": len(summaries),
            "watertight": sum(summary.watertight for summary in summaries),
            "winding_consistent": sum(
                summary.winding_consistent for summary in summaries
            ),
            "usable_enclosed_volumes": sum(
                summary.volume_m3 is not None for summary in summaries
            ),
            "total_surface_area_m2": sum(
                summary.surface_area_m2 for summary in summaries
            ),
            "total_usable_volume_m3": sum(
                summary.volume_m3 or 0 for summary in summaries
            ),
            "published_volume_relative_error": {
                "compared": len(errors),
                "median": float(np.median(errors)) if errors else None,
                "p95": float(np.percentile(errors, 95)) if errors else None,
                "maximum": float(np.max(errors)) if errors else None,
            },
        }
    return {
        "source": reference.metadata(),
        "geometry_count": sum(len(summaries) for summaries in groups.values()),
        "side_counts": dict(
            Counter(geometry.side for geometry in reference.geometries())
        ),
        "tissues": totals,
        "excluded_volume_geometries": [
            {
                "geometry": summary.geometry.name,
                "watertight": summary.watertight,
                "winding_consistent": summary.winding_consistent,
                "member_sha256": summary.metadata()["member_sha256"],
            }
            for summaries in groups.values()
            for summary in summaries
            if summary.volume_m3 is None
        ],
    }, tibia


def run_validation(
    female: DenverReference,
    male: DenverReference,
    *,
    longitudinal_factor: float | None = None,
    longitudinal_axis: str | None = None,
) -> dict[str, Any]:
    """Compare derived mesh volumes with independent source metadata tables.

    A volume demonstration requires an explicit independently supplied length
    factor and local axis. Neither archive establishes that length factor.
    Only aggregate geometry measures, checks, assumptions and hashes are emitted.
    """
    if female.subject != "VHF" or male.subject != "VHM":
        raise ValueError("Expected the VHF and VHM reference archive pairs.")
    if (longitudinal_factor is None) != (longitudinal_axis is None):
        raise ValueError(
            "Supply both the independent longitudinal factor and axis, or neither."
        )
    if longitudinal_factor is not None and (
        not np.isfinite(longitudinal_factor)
        or longitudinal_factor <= 0
        or longitudinal_axis not in ("x", "y", "z")
    ):
        raise ValueError("Independent length factor and target-local axis are invalid.")
    female_report, female_tibia = _aggregate(female)
    male_report, male_tibia = _aggregate(male)
    report = {
        "schema": 1,
        "protocol": "Independent mesh-volume versus published-volume cross-check",
        "report_license": "CC-BY-4.0",
        "report_license_url": "https://creativecommons.org/licenses/by/4.0/",
        "changes": "Computed aggregate geometry measures, volume discrepancies and quality exclusions from the cited local source archives; source surfaces and spreadsheet rows are not redistributed.",
        "environment": {
            "numpy": np.__version__,
            "trimesh": version("trimesh"),
            "openpyxl": version("openpyxl"),
        },
        "subjects": {"VHF": female_report, "VHM": male_report},
        "redistribution": "Derived aggregates and source hashes only; no STL or source spreadsheet data.",
        "limitations": [
            "The two donors are the same Visible Human subjects used by BoneHub.",
            "Archive coverage is counted from actual files; other release assets may differ.",
            "Volume consistency does not validate transverse anatomy, muscle forces or dynamics.",
            "Equal transverse factors are an explicit assumption, not two measured widths.",
        ],
    }
    if longitudinal_factor is not None:
        if (
            female_tibia is None
            or male_tibia is None
            or female_tibia.volume_m3 is None
            or male_tibia.volume_m3 is None
        ):
            raise ValueError(
                "Usable right-tibia volumes are required for the requested constraint."
            )
        constraint = VolumeScaleConstraint(
            "VHF",
            "tibia_r",
            male_tibia.volume_m3,
            female_tibia.volume_m3,
            longitudinal_factor,
            longitudinal_axis,
            f"Denver VHM right tibia@{male_tibia.metadata()['member_sha256']}",
            f"Denver VHF right tibia@{female_tibia.metadata()['member_sha256']}",
        )
        estimate = volume_constrained_scale_estimate(
            constraint, assume_equal_transverse=True
        )
        ratio = estimate.evidence["volume_ratio"]
        report["volume_constraint_demonstration"] = {
            "available": True,
            **estimate.evidence,
            "longitudinal_factor_origin": "Explicit caller-provided length factor, not inferred from these archives.",
            "constrained_relative_volume_error": abs(
                estimate.evidence["implied_volume_ratio"] - ratio
            )
            / ratio,
            "anatomical_accuracy_validated": False,
        }
    return report


def main() -> None:
    """Run local archive checks and write a derivative-only JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--female-mesh", type=Path, required=True)
    parser.add_argument("--female-metadata", type=Path, required=True)
    parser.add_argument("--male-mesh", type=Path, required=True)
    parser.add_argument("--male-metadata", type=Path, required=True)
    parser.add_argument("--longitudinal-factor", type=float)
    parser.add_argument("--axis", dest="longitudinal_axis", choices=("x", "y", "z"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.longitudinal_factor is None) != (args.longitudinal_axis is None):
        parser.error("--longitudinal-factor and --axis must be supplied together.")
    report = run_validation(
        DenverReference(args.female_mesh, args.female_metadata, "VHF"),
        DenverReference(args.male_mesh, args.male_metadata, "VHM"),
        longitudinal_factor=args.longitudinal_factor,
        longitudinal_axis=args.longitudinal_axis,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote derivative-only Denver validation: {args.output}")


if __name__ == "__main__":
    main()
