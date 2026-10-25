# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Derive population geometry summaries from local or opt-in VSD tibia meshes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from myosuite.integrations.bonehub import (  # noqa: E402
    VSD_DATASET_ID,
    VSD_PINNED_REVISION,
    BoneHubVsdReference,
)

COMPLETE_CASES = tuple(
    f"{index:03d}"
    for index in range(1, 31)
    if index not in (13, 18, 21, 22, 23, 25, 26, 28, 29, 30)
)
_UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001}


def mesh_descriptors(path: Path, units: Literal["m", "cm", "mm"]) -> dict[str, Any]:
    """Measure solid volume and sorted inertia-axis geometric spans locally.

    A span is a mesh descriptor, not an anatomical segment length. Volume is
    admitted only for closed, consistently wound, outward-facing meshes.
    """
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "VSD metrics require `pip install 'myosuite[anatomy]'`."
        ) from exc
    if units not in _UNITS:
        raise ValueError("Confirm mesh units explicitly as m, cm or mm.")
    mesh = trimesh.load(path, file_type="stl", force="mesh", process=True)
    if (
        not isinstance(mesh, trimesh.Trimesh)
        or not len(mesh.faces)
        or not np.all(np.isfinite(mesh.vertices))
    ):
        raise ValueError("A finite triangle mesh is required.")
    mesh.apply_scale(_UNITS[units])
    valid = bool(mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0)
    result = {
        "trimesh_version": trimesh.__version__,
        "volume_usable": valid,
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
    }
    if valid:
        spans = np.sort(
            np.ptp(mesh.vertices @ mesh.principal_inertia_vectors.T, axis=0)
        )[::-1]
        if not np.all(np.isfinite(spans)) or np.any(spans <= 0):
            raise ValueError("Principal-axis spans must be positive and finite.")
        volume = float(mesh.volume)
        result.update(
            volume_m3=volume,
            principal_axis_spans_m=spans.tolist(),
            volume_over_long_span_cubed=volume / float(spans[0]) ** 3,
        )
    return result


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    probabilities = (0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)
    return dict(
        zip(
            ("minimum", "q05", "q25", "median", "q75", "q95", "maximum"),
            (float(value) for value in np.quantile(values, probabilities)),
        )
    )


def validate(
    root: Path,
    *,
    units: Literal["m", "cm", "mm"],
    cases: tuple[str, ...] = COMPLETE_CASES,
    download: bool = False,
) -> dict[str, Any]:
    """Summarize eligible cases while retaining only aggregate derived measures.

    Source clinical quality and mesh volume quality are separate gates. Case
    identities and hashes are provenance; no vertices, centroids or source
    anthropometry are included in the public report.
    """
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("Select unique VSD case identifiers.")
    sources, metrics, excluded = [], [], Counter()
    for case in cases:
        reference = BoneHubVsdReference(root, case)
        try:
            path = (
                reference.download_mesh("Tibia_R")
                if download
                else reference.mesh_path("Tibia_R")
            )
            source = reference.metadata()
        except FileNotFoundError:
            excluded["not_cached"] += 1
            continue
        provenance = {
            "case": case,
            "bone": "Tibia_R",
            "mesh_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "metadata_sha256": source["source_metadata_sha256"],
        }
        if not source["source_in_complete_cohort"]:
            excluded["source_not_in_complete_cohort"] += 1
            sources.append({**provenance, "admitted": False})
            continue
        descriptor = mesh_descriptors(path, units)
        sources.append({**provenance, "admitted": descriptor["volume_usable"]})
        if not descriptor["volume_usable"]:
            excluded["mesh_volume_unusable"] += 1
            continue
        metrics.append(descriptor)
    return {
        "schema": 1,
        "dataset": VSD_DATASET_ID,
        "revision": VSD_PINNED_REVISION,
        "source_url": f"https://huggingface.co/datasets/{VSD_DATASET_ID}/tree/{VSD_PINNED_REVISION}",
        "source_license": "CC-BY-NC-SA-4.0; original assets remain local",
        "report_license": "CC-BY-NC-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "license_legalcode_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en",
        "derived_changes": "Computed aggregate mesh-volume and principal-span quantiles, quality coverage and source hashes; no source meshes or per-case geometric measurements are distributed.",
        "disclaimer": "Sources and report are provided without warranties; no source-author endorsement is implied. See the linked legal terms.",
        "attribution": "Alavi and Asseln (2026), doi:10.57967/hf/10471; Fischer (2023), doi:10.1038/s41597-023-02669-z",
        "source_units_confirmed": units,
        "environment": {
            "numpy": np.__version__,
            "trimesh": metrics[0]["trimesh_version"] if metrics else None,
        },
        "bone": "Tibia_R",
        "coverage": {
            "requested": len(cases),
            "source_complete_and_volume_usable": len(metrics),
            "excluded_counts": dict(excluded),
        },
        "geometry_protocol": "Solid-volume inertia principal axes; sorted spans of projected mesh vertices. Span is not joint-centre or segment length.",
        "volume_m3": _quantiles([row["volume_m3"] for row in metrics]),
        "principal_axis_spans_m": {
            name: _quantiles([row["principal_axis_spans_m"][axis] for row in metrics])
            for axis, name in enumerate(("largest", "middle", "smallest"))
        },
        "volume_over_long_span_cubed": _quantiles(
            [row["volume_over_long_span_cubed"] for row in metrics]
        ),
        "sources": sources,
        "limitations": [
            "The 20 source-complete cadaveric cases are not a representative population distribution.",
            "These are outer mesh/solid-volume descriptors, not cortical density, anatomical segment lengths or individual measurements.",
            "No factors, strength parameters, patient-specific estimates or automatically applied priors are derived from these quantiles.",
        ],
    }


def main() -> None:
    """Run the optional download and export a derivative-only cohort report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--units", choices=tuple(_UNITS), required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cases", nargs="+", default=COMPLETE_CASES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(
        args.data_root,
        units=args.units,
        cases=tuple(args.cases),
        download=args.download,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["coverage"], indent=2))


if __name__ == "__main__":
    main()
