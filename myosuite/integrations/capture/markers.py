# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Labelled static-trial scaling without capture or body-shape dependencies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from myosuite.core.subject_model import ScaleEstimate

_METRES = {"m": 1.0, "cm": 0.01, "mm": 0.001}


@dataclass(frozen=True)
class MarkerTrial:
    """Labelled coordinates with explicit units and a calibrated model-world frame.

    Positions have shape ``(frames, markers, 3)``; NaN means missing observations.
    ``rotation_to_model`` must be a proper rotation, not a reflection. Coordinates
    become ``positions * unit_scale @ rotation.T + translation_to_model_m``.
    Confidence is supplied by the source, not inferred as a probability here.
    """

    labels: tuple[str, ...]
    positions: ArrayLike
    units: Literal["m", "cm", "mm"]
    rotation_to_model: ArrayLike
    source: str
    translation_to_model_m: ArrayLike = (0.0, 0.0, 0.0)
    confidence: float | None = None
    uncertainty_m: float | None = None
    frequency_hz: float | None = None
    definition_source: str | None = None
    definition_license: str | None = None
    point_semantics: str = "unspecified"
    subject_id: str | None = None

    def __post_init__(self) -> None:
        labels = tuple(self.labels)
        if (
            not labels
            or len(set(labels)) != len(labels)
            or any(not isinstance(label, str) or not label.strip() for label in labels)
        ):
            raise ValueError("Marker labels must be unique nonempty strings.")
        positions = np.array(self.positions, dtype=float, copy=True)
        if (
            positions.ndim != 3
            or positions.shape[1:] != (len(labels), 3)
            or not len(positions)
        ):
            raise ValueError("positions must have shape (frames, markers, 3).")
        if np.any(np.isinf(positions)):
            raise ValueError("Use NaN for missing coordinates; infinity is invalid.")
        if self.units not in _METRES:
            raise ValueError("Explicit units must be 'm', 'cm', or 'mm'.")
        rotation = np.array(self.rotation_to_model, dtype=float, copy=True)
        if (
            rotation.shape != (3, 3)
            or not np.all(np.isfinite(rotation))
            or not (
                np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
                and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8)
            )
        ):
            raise ValueError("rotation_to_model must be a proper orthonormal rotation.")
        translation = np.array(self.translation_to_model_m, dtype=float, copy=True)
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError("translation_to_model_m must be a finite 3-vector.")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("A source identifier is required.")
        if (
            not isinstance(self.point_semantics, str)
            or not self.point_semantics.strip()
        ):
            raise ValueError("point_semantics must be a nonempty string.")
        if self.subject_id is not None and (
            not isinstance(self.subject_id, str) or not self.subject_id.strip()
        ):
            raise ValueError("subject_id must be a nonempty string or None.")
        if self.confidence is not None and (
            not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence must be finite and between zero and one.")
        for name in ("uncertainty_m", "frequency_hz"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive.")
        for name, value in (
            ("labels", labels),
            ("positions", positions),
            ("rotation_to_model", rotation),
            ("translation_to_model_m", translation),
        ):
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
            object.__setattr__(self, name, value)

    def in_model_frame(self) -> np.ndarray:
        """Return positions in model-world metres; missing samples remain NaN."""
        return np.asarray(self.positions) * _METRES[self.units] @ np.asarray(
            self.rotation_to_model
        ).T + np.asarray(self.translation_to_model_m)

    def metadata(self) -> dict[str, Any]:
        """Return the input coordinate convention and source-reported uncertainty."""
        return {
            "source": self.source,
            "labels": list(self.labels),
            "units": self.units,
            "rotation_to_model": np.asarray(self.rotation_to_model).tolist(),
            "translation_to_model_m": np.asarray(self.translation_to_model_m).tolist(),
            "confidence": None if self.confidence is None else float(self.confidence),
            "uncertainty_m": None
            if self.uncertainty_m is None
            else float(self.uncertainty_m),
            "frequency_hz": None
            if self.frequency_hz is None
            else float(self.frequency_hz),
            "definition_source": self.definition_source,
            "definition_license": self.definition_license,
            "point_semantics": self.point_semantics,
            "subject_id": self.subject_id,
        }


@dataclass(frozen=True)
class SegmentMeasurement:
    """Map one labelled distance to a target body's or segment's local scale.

    ``axis=None`` explicitly assumes isotropic scaling from one length; an axis
    keeps the other local factors at one. ``expected_direction_model`` is an
    optional directed static-pose calibration check in model-world coordinates.
    The caller supplies lengths and axes from the unscaled target, not height.
    """

    segment: str
    marker_pair: tuple[str, str]
    reference_length_m: float
    axis: Literal["x", "y", "z"] | None = None
    expected_direction_model: ArrayLike | None = None
    max_direction_error_deg: float = 20.0

    def __post_init__(self) -> None:
        if (
            not self.segment
            or len(self.marker_pair) != 2
            or self.marker_pair[0] == self.marker_pair[1]
        ):
            raise ValueError("A segment and two distinct marker labels are required.")
        if not np.isfinite(self.reference_length_m) or self.reference_length_m <= 0:
            raise ValueError("reference_length_m must be finite and positive.")
        if self.axis not in (None, "x", "y", "z"):
            raise ValueError("axis must be x, y, z, or None for isotropic scaling.")
        if (
            not np.isfinite(self.max_direction_error_deg)
            or not 0 < self.max_direction_error_deg < 90
        ):
            raise ValueError("max_direction_error_deg must be between zero and 90.")
        object.__setattr__(self, "marker_pair", tuple(self.marker_pair))
        if self.expected_direction_model is not None:
            direction = np.asarray(self.expected_direction_model, dtype=float)
            if (
                direction.shape != (3,)
                or not np.all(np.isfinite(direction))
                or np.linalg.norm(direction) == 0
            ):
                raise ValueError(
                    "expected_direction_model must be a finite nonzero 3-vector."
                )
            object.__setattr__(
                self,
                "expected_direction_model",
                tuple(direction / np.linalg.norm(direction)),
            )


@dataclass(frozen=True)
class MarkerScaleResult:
    """Core estimate plus serializable measurement and coordinate evidence."""

    estimate: ScaleEstimate
    evidence: dict[str, Any]


def estimate_marker_scales(
    trial: MarkerTrial,
    measurements: Sequence[SegmentMeasurement],
    *,
    min_valid_fraction: float = 0.5,
    scale_bounds: tuple[float, float] = (0.5, 2.0),
    height_m: float | None = None,
    mass_kg: float | None = None,
) -> MarkerScaleResult:
    """Estimate median segment lengths and report missing data without imputation.

    Bounds catch likely units/reference errors, not anatomical validity. Missing
    measurements are omitted, so unmapped segments remain unscaled. Dispersion
    across frames is reported separately from source uncertainty/confidence.
    """
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0, 1].")
    low, high = scale_bounds
    if not np.all(np.isfinite(scale_bounds)) or not 0 < low < high:
        raise ValueError("scale_bounds must be finite, positive and increasing.")
    positions = trial.in_model_frame()
    labels = {label: i for i, label in enumerate(trial.labels)}
    factors: dict[str, np.ndarray] = {}
    occupied: dict[str, set[str | None]] = {}
    notes, evidence = [], []
    for measurement in measurements:
        segment, axis = measurement.segment, measurement.axis
        used = occupied.setdefault(segment, set())
        if axis in used or (used and (axis is None or None in used)):
            raise ValueError(f"Overlapping scale measurements for {segment!r}.")
        used.add(axis)
        row: dict[str, Any] = {
            "segment": segment,
            "marker_pair": list(measurement.marker_pair),
            "reference_length_m": float(measurement.reference_length_m),
            "axis": axis,
            "expected_direction_model": measurement.expected_direction_model,
        }
        if any(label not in labels for label in measurement.marker_pair):
            notes.append(
                f"{segment}: missing marker label; this measurement stays unscaled."
            )
            evidence.append({**row, "status": "missing_label", "valid_fraction": 0.0})
            continue
        a, b = (positions[:, labels[label]] for label in measurement.marker_pair)
        delta = b - a
        lengths = np.linalg.norm(delta, axis=1)
        valid = np.isfinite(lengths) & (lengths > 0)
        fraction = float(valid.mean())
        if fraction < min_valid_fraction or not np.any(valid):
            notes.append(
                f"{segment}: insufficient valid marker samples ({fraction:.1%}); stays unscaled."
            )
            evidence.append(
                {**row, "status": "missing_samples", "valid_fraction": fraction}
            )
            continue
        if fraction < 1:
            notes.append(
                f"{segment}: {1 - fraction:.1%} missing/zero-length samples excluded."
            )
        if measurement.expected_direction_model is not None:
            directions = delta[valid] / lengths[valid, None]
            cosine = directions @ np.asarray(measurement.expected_direction_model)
            angle = float(np.degrees(np.arccos(np.clip(np.median(cosine), -1, 1))))
            if angle > measurement.max_direction_error_deg:
                raise ValueError(
                    f"{segment}: frame/directed-marker calibration error ({angle:.1f} degrees)."
                )
        length = float(np.median(lengths[valid]))
        factor = length / measurement.reference_length_m
        if not low <= factor <= high:
            raise ValueError(
                f"{segment}: scale {factor:g} outside {scale_bounds}; check units and reference lengths."
            )
        scale = factors.setdefault(segment, np.ones(3))
        if axis is None:
            scale[:] = factor
            notes.append(
                f"{segment}: isotropic scale assumed from one length; cross sections were not measured."
            )
        else:
            scale["xyz".index(axis)] = factor
            notes.append(
                f"{segment}: only local {axis} measured; unmeasured axes remain one."
            )
        evidence.append(
            {
                **row,
                "status": "estimated",
                "valid_fraction": fraction,
                "length_m": length,
                "factor": factor,
                "length_mad_m": float(np.median(np.abs(lengths[valid] - length))),
            }
        )
    if not factors:
        raise ValueError("No segment had sufficient valid marker measurements.")
    estimate = ScaleEstimate(
        factors,
        height_m=height_m,
        mass_kg=mass_kg,
        source=trial.source,
        confidence=trial.confidence,
        warnings=tuple(notes),
    )
    return MarkerScaleResult(
        estimate,
        {
            "schema": 1,
            "adapter": "labelled-static-lengths",
            "trial": trial.metadata(),
            "scale_axes": "target body/segment local axes supplied by caller",
            "measurements": evidence,
            "scale_bounds": list(scale_bounds),
            "min_valid_fraction": min_valid_fraction,
        },
    )


def virtual_marker_trial(
    *,
    labels: tuple[str, ...],
    positions: ArrayLike,
    units: Literal["m", "cm", "mm"],
    rotation_to_model: ArrayLike,
    definition_source: str,
    definition_license: str,
    source: str = "user-provided virtual markers",
    **kwargs: Any,
) -> MarkerTrial:
    """Consume user-provided virtual markers without shipping mesh vertex indices.

    The definition's source and licence are recorded, not asserted to grant any
    redistribution right. SMPL/SMPL-X/AMASS loading stays in the user's workflow.
    """
    if not definition_source.strip() or not definition_license.strip():
        raise ValueError("Virtual-marker definition source and licence are required.")
    return MarkerTrial(
        labels,
        positions,
        units,
        rotation_to_model,
        source,
        definition_source=definition_source,
        definition_license=definition_license,
        **kwargs,
    )


def read_c3d(
    path: str | Path, *, rotation_to_model: ArrayLike, **kwargs: Any
) -> MarkerTrial:
    """Read labelled C3D points using optional ``myosuite[capture]``/ezc3d.

    POINT units/rate are read from the file. Negative C3D residuals mark missing
    observations. The lab-to-model rotation must be calibrated by the caller.
    """
    try:
        import ezc3d
    except ImportError as exc:
        raise ImportError(
            "Reading C3D requires `pip install 'myosuite[capture]'`."
        ) from exc
    capture = ezc3d.c3d(str(path))
    parameters = capture["parameters"]["POINT"]
    labels = tuple(str(label).strip() for label in parameters["LABELS"]["value"])
    points = np.array(capture["data"]["points"][:3].transpose(2, 1, 0), copy=True)
    residuals = np.asarray(capture["data"]["meta_points"]["residuals"])[0].T
    points[residuals < 0] = np.nan
    return MarkerTrial(
        labels,
        points,
        str(parameters["UNITS"]["value"][0]).strip().lower(),
        rotation_to_model,
        source=str(Path(path).resolve()),
        frequency_hz=float(parameters["RATE"]["value"][0]),
        **kwargs,
    )
