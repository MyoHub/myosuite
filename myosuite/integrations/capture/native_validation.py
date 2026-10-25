# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Aggregate-only MyoLeg validation from locally evaluated native model origins."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite import make_env
from myosuite.core.subject_model import (
    GroundAlignment,
    ScaleEstimate,
    SubjectModelConfig,
    align_root_to_ground,
    build_subject_model,
)
from myosuite.integrations.capture.leg_workflow import SOLES
from myosuite.integrations.capture.markers import (
    MarkerTrial,
    SegmentMeasurement,
    estimate_marker_scales,
)
from myosuite.integrations.capture.retargeting import (
    motion_diagnostics,
    retarget_markers,
)

_BODIES = ("pelvis",) + tuple(
    f"{body}_{side}"
    for side in "rl"
    for body in ("femur", "tibia", "talus", "calcn", "toes")
)


def _scalars(values: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in values.items() if np.isscalar(value)}


def validate_native_origins(
    path: str | Path,
    *,
    source_label: str,
    definition_license: str,
    calibration_frames: tuple[int, ...] = (0, 1, 2),
    heldout_frames: tuple[int, ...] = (10, 20, 30),
    marker_tolerance_m: float = 0.02,
    max_evaluations: int = 200,
    evaluate_ground_projection: bool = False,
) -> dict[str, Any]:
    """Measure scaled-target fitting against an explicit nominal-fit control.

    Args:
        path: Local, pickle-free NPZ of evaluated ``positions`` in calibrated
            model-world metres, ``labels`` and ``frequency_hz``. Points must be
            native skeletal/body origins corresponding to the named MyoLeg
            body frames, not skin landmarks labelled as internal centres.
        source_label: Public source/version label for the aggregate report.
        definition_license: Terms of the caller's local derived coordinates.
        calibration_frames: Source frames used only for length estimation.
        heldout_frames: Disjoint, uniformly spaced frames used for fitting.
        marker_tolerance_m: Explicit geometric marker convergence threshold.
        max_evaluations: Per-frame nonlinear least-squares evaluation budget.
        evaluate_ground_projection: Also report a geometric root projection;
            the raw fitted motion remains the primary result.

    Returns:
        Scalar validation metrics and source hashes, excluding coordinates,
        qpos, shape parameters, individual scale factors and demographics.

    This is a comparison of kinematic model representations. Geometric contacts
    are not measured forces; a small marker residual does not validate anatomy
    or dynamics. Temporal held-out frames are not a subject-disjoint benchmark.
    """
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as archive:
        positions = np.asarray(archive["positions"], dtype=float)
        labels = tuple(archive["labels"].tolist())
        frequency = float(archive["frequency_hz"])
    if not source_label.strip() or not definition_license.strip():
        raise ValueError("A source label and definition licence are required.")
    missing = set(_BODIES) - set(labels)
    if missing:
        raise ValueError(
            f"Native model-origin protocol is missing labels: {sorted(missing)}"
        )
    frame_sets = (calibration_frames, heldout_frames)
    for indices in frame_sets:
        if (
            not indices
            or any(
                isinstance(i, bool) or not isinstance(i, (int, np.integer))
                for i in indices
            )
            or len(set(indices)) != len(indices)
            or min(indices) < 0
            or max(indices) >= len(positions)
        ):
            raise ValueError(
                "Frame selections must be unique in-range integer indices."
            )
    if set(calibration_frames) & set(heldout_frames):
        raise ValueError("Calibration and held-out frames must be disjoint.")
    spacing = np.diff(heldout_frames)
    if not len(spacing) or spacing[0] <= 0 or not np.all(spacing == spacing[0]):
        raise ValueError("Held-out frames must have one positive uniform spacing.")
    base_trial = MarkerTrial(
        labels,
        positions,
        "m",
        np.eye(3),
        source_label,
        frequency_hz=frequency,
        definition_source="native model skeletal/body origins; explicit source-to-target correspondence",
        definition_license=definition_license,
    )
    with make_env("myoLegWalk-v0") as env:
        spec = env.unwrapped._mj_spec.copy()
    for body in _BODIES:
        spec.body(body).add_site(name=f"origin_{body}", pos=[0, 0, 0], size=[0.005] * 3)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mapping = {body: f"origin_{body}" for body in _BODIES}
    measurements = []
    for side in "rl":
        for segment, child, axis in (
            (f"femur_{side}", f"tibia_{side}", "y"),
            (f"tibia_{side}", f"talus_{side}", "y"),
            (f"calcn_{side}", f"toes_{side}", "x"),
        ):
            length = np.linalg.norm(
                data.xpos[model.body(child).id] - data.xpos[model.body(segment).id]
            )
            measurements.append(
                SegmentMeasurement(segment, (segment, child), float(length), axis=axis)
            )
    width = np.linalg.norm(
        data.xpos[model.body("femur_r").id] - data.xpos[model.body("femur_l").id]
    )
    measurements.append(
        SegmentMeasurement("pelvis", ("femur_r", "femur_l"), float(width), axis="z")
    )
    static = replace(base_trial, positions=positions[list(calibration_frames)])
    estimate = estimate_marker_scales(static, measurements)
    ground = GroundAlignment("root", SOLES)
    target = build_subject_model(
        spec, SubjectModelConfig(estimate.estimate, ground=ground)
    )
    nominal = build_subject_model(
        spec, SubjectModelConfig(ScaleEstimate({}), ground=ground)
    )
    motion = replace(
        base_trial,
        positions=positions[list(heldout_frames)],
        frequency_hz=frequency / int(spacing[0]),
    )
    kwargs = dict(
        free_joint="root",
        sole_geoms=SOLES,
        root_marker="pelvis",
        tolerance_m=marker_tolerance_m,
        max_evaluations=max_evaluations,
        strict=False,
    )
    fitted = retarget_markers(
        target, motion, mapping, scale_evidence=estimate.evidence, **kwargs
    )
    control = retarget_markers(nominal, motion, mapping, **kwargs)
    replay = motion_diagnostics(
        target.model,
        control.clip,
        motion,
        mapping,
        free_joint="root",
        sole_geoms=SOLES,
        root_marker="pelvis",
    )
    report: dict[str, Any] = {
        "schema": 1,
        "source": source_label,
        "local_coordinate_export_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "calibration_frames": len(calibration_frames),
        "heldout_frames": len(heldout_frames),
        "calibration_frame_indices": list(calibration_frames),
        "heldout_frame_indices": list(heldout_frames),
        "max_evaluations_per_frame": max_evaluations,
        "requested_scale_axes": len(measurements),
        "estimated_scale_axes": len(estimate.estimate.factors),
        "omitted_scale_measurements": len(measurements)
        - len(estimate.estimate.factors),
        "marker_tolerance_m": marker_tolerance_m,
        "point_semantics": "native model origins, not measured anatomical joint centres",
        "evaluation_scope": "one representation/subject; temporal held-out frames; no dynamic validation",
        "ground_plane_assumption_m": 0.0,
        "source_floor_measured": False,
        "runtime": {"mujoco": mujoco.__version__, "numpy": np.__version__},
        "scaled_target": _scalars(fitted.diagnostics),
        "nominal_target": _scalars(control.diagnostics),
        "nominal_fit_replayed_on_scaled": _scalars(replay),
        "geometric_contact_geom_frames": int(
            np.count_nonzero(fitted.diagnostics["contact_labels"])
        ),
        "sole_geom_count": len(SOLES),
        "scaled_marker_rms_improvement_over_negative": float(
            replay["marker_rms_m"]
            / max(fitted.diagnostics["marker_rms_m"], np.finfo(float).tiny)
        ),
    }
    if evaluate_ground_projection:
        aligned = [
            align_root_to_ground(target.model, ground, pose)
            for pose in fitted.clip.qpos
        ]
        for key, qpos in (
            (
                "geometric_ground_projection",
                np.array([pose for pose, offset in aligned]),
            ),
            (
                "raise_only_nonpenetration_projection",
                np.array(
                    [
                        pose if offset > 0 else original
                        for (pose, offset), original in zip(aligned, fitted.clip.qpos)
                    ]
                ),
            ),
        ):
            qvel = np.zeros((len(qpos), target.model.nv))
            for index in range(1, len(qpos)):
                mujoco.mj_differentiatePos(
                    target.model,
                    qvel[index],
                    1 / motion.frequency_hz,
                    qpos[index - 1],
                    qpos[index],
                )
            qvel[0] = qvel[1]
            projected = replace(fitted.clip, qpos=qpos, qvel=qvel)
            quality = motion_diagnostics(
                target.model,
                projected,
                motion,
                mapping,
                free_joint="root",
                sole_geoms=SOLES,
                root_marker="pelvis",
            )
            report[key] = {
                **_scalars(quality),
                "within_marker_tolerance": bool(
                    quality["marker_max_m"] <= marker_tolerance_m
                ),
                "method": "root-height geometry postprocess only; no refitting or dynamic/contact solve",
                "lowers_airborne_frames": key == "geometric_ground_projection",
            }
    return report


def validate_lateral_shank_spans(
    path: str | Path, *, source_label: str, definition_license: str
) -> dict[str, Any]:
    """Check measured lateral condyle/malleolus spans without imputing segments.

    The local NPZ uses ``positions`` in metres, ``labels`` and ``frequency_hz``.
    This fixed Lencioni/Tiziana protocol calibrates with frames 0--2 and evaluates
    frames 10 onward. Span lengths are rotation invariant. Optional integer
    ``missing_grf`` values retain Nimble's zero=valid source-status convention.
    Returns aggregate span errors, never source samples or individual factors.
    """
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as archive:
        positions = np.asarray(archive["positions"], dtype=float)
        labels = tuple(archive["labels"].tolist())
        frequency = float(archive["frequency_hz"])
        reasons = (
            np.asarray(archive["missing_grf"])
            if "missing_grf" in archive.files
            else None
        )
    if len(positions) <= 10:
        raise ValueError("The lateral-shank protocol requires more than ten frames.")
    if not definition_license.strip():
        raise ValueError("A definition licence is required.")
    trial = MarkerTrial(
        labels,
        positions,
        "m",
        np.eye(3),
        source_label,
        frequency_hz=frequency,
        definition_source="Lencioni/Tiziana lateral condyle and lateral malleolus markers",
        definition_license=definition_license,
    )
    with make_env("myoLegWalk-v0") as env:
        spec = env.unwrapped._mj_spec.copy()
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    measurements, references, observed = [], [], []
    for side in "rl":
        source_pair = (f"{side.upper()}xLatCon", f"{side.upper()}xLatMal")
        target_pair = (f"LFC_{side}", f"LMAL_{side}")
        length = float(
            np.linalg.norm(
                data.site_xpos[model.site(target_pair[0]).id]
                - data.site_xpos[model.site(target_pair[1]).id]
            )
        )
        measurements.append(
            SegmentMeasurement(f"tibia_{side}", source_pair, length, axis="y")
        )
        references.append(length)
        observed.append(
            np.linalg.norm(
                positions[10:, labels.index(source_pair[0])]
                - positions[10:, labels.index(source_pair[1])],
                axis=1,
            )
        )
    estimate = estimate_marker_scales(
        replace(trial, positions=positions[:3]), measurements
    )
    target = build_subject_model(spec, SubjectModelConfig(estimate.estimate))
    target_data = mujoco.MjData(target.model)
    mujoco.mj_forward(target.model, target_data)
    scaled = []
    for side in "rl":
        scaled.append(
            float(
                np.linalg.norm(
                    target_data.site_xpos[target.model.site(f"LFC_{side}").id]
                    - target_data.site_xpos[target.model.site(f"LMAL_{side}").id]
                )
            )
        )
    observed_spans = np.array(observed).T
    if not np.any(np.isfinite(observed_spans)):
        raise ValueError("No finite held-out marker spans are available.")
    nominal_error = observed_spans - np.asarray(references)
    scaled_error = observed_spans - np.asarray(scaled)
    report = {
        "schema": 1,
        "source": source_label,
        "local_coordinate_export_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "calibration_frame_indices": [0, 1, 2],
        "heldout_first_frame": 10,
        "heldout_frame_count": len(positions) - 10,
        "source_frame_count": len(positions),
        "source_marker_count": len(labels),
        "source_frequency_hz": frequency,
        "finite_marker_fraction": float(np.mean(np.isfinite(positions).all(axis=-1))),
        "finite_heldout_span_count": int(np.count_nonzero(np.isfinite(observed_spans))),
        "missing_heldout_span_count": int(
            np.count_nonzero(~np.isfinite(observed_spans))
        ),
        "estimated_scale_axes": len(estimate.estimate.factors),
        "unmeasured_segments": "remain nominal; no height-based inferred factors",
        "nominal_heldout_span_rms_m": float(np.sqrt(np.nanmean(nominal_error**2))),
        "scaled_heldout_span_rms_m": float(np.sqrt(np.nanmean(scaled_error**2))),
        "scaled_heldout_span_max_m": float(np.nanmax(np.abs(scaled_error))),
        "evaluation_scope": "measured surface-marker span consistency; not joint-centre length or full-body retargeting",
        "runtime": {"mujoco": mujoco.__version__, "numpy": np.__version__},
    }
    if reasons is not None:
        if reasons.shape != (len(positions),) or not np.issubdtype(
            reasons.dtype, np.integer
        ):
            raise ValueError(
                "missing_grf must retain one source integer status per frame."
            )
        report["source_grf_valid_frame_count"] = int(np.count_nonzero(reasons == 0))
        report["source_grf_other_frame_count"] = int(np.count_nonzero(reasons != 0))
    return report


def main() -> None:
    """Run a local comparison and write only aggregate metrics and provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--definition-license", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        choices=("native-origins", "lateral-shank"),
        default="native-origins",
    )
    parser.add_argument("--ground-projection", action="store_true")
    parser.add_argument("--calibration-frames", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--heldout-start", type=int, default=10)
    parser.add_argument("--heldout-stride", type=int, default=10)
    parser.add_argument("--heldout-count", type=int, default=3)
    args = parser.parse_args()
    report = (
        validate_native_origins(
            args.trial,
            source_label=args.source_label,
            definition_license=args.definition_license,
            evaluate_ground_projection=args.ground_projection,
            calibration_frames=tuple(args.calibration_frames),
            heldout_frames=tuple(
                args.heldout_start + args.heldout_stride * i
                for i in range(args.heldout_count)
            ),
        )
        if args.protocol == "native-origins"
        else validate_lateral_shank_spans(
            args.trial,
            source_label=args.source_label,
            definition_license=args.definition_license,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
