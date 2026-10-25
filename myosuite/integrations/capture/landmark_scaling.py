# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Fit segment-local scales from explicit three-dimensional correspondences."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from myosuite.core.subject_model import ScaleEstimate
from myosuite.integrations.capture.markers import MarkerScaleResult


@dataclass(frozen=True)
class LandmarkCorrespondences:
    """One segment's corresponding reference and subject landmarks, in metres.

    Args:
        subject_id: Identity of the observed subject, shared by all segments.
        segment: Target body or segment receiving the fitted scale factors.
        labels: Unique anatomical correspondence identifiers.
        reference_positions_m: ``(landmarks, 3)`` points in target-local axes.
        observed_positions_m: Corresponding points in one arbitrary rigid frame.
        reference_source: Reference geometry/definition origin and version.
        observed_source: Subject geometry/annotation origin and version.
        uncertainty_m: Optional source-reported isotropic coordinate standard
            deviation, scalar or one value per landmark. This is not inferred.

    A partially missing point is omitted. Supply one observation per landmark,
    rather than stacking correlated motion frames as independent anatomy.
    """

    subject_id: str
    segment: str
    labels: tuple[str, ...]
    reference_positions_m: ArrayLike
    observed_positions_m: ArrayLike
    reference_source: str
    observed_source: str
    uncertainty_m: ArrayLike | None = None

    def __post_init__(self) -> None:
        for name in ("subject_id", "segment", "reference_source", "observed_source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string.")
        labels = tuple(self.labels)
        if (
            not labels
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError("Landmark labels must be unique nonempty strings.")
        object.__setattr__(self, "labels", labels)
        for name in ("reference_positions_m", "observed_positions_m"):
            values = np.array(getattr(self, name), dtype=float, copy=True)
            if values.shape != (len(labels), 3):
                raise ValueError(
                    f"{name} must have shape (landmarks, 3), without frames."
                )
            if np.any(np.isinf(values)):
                raise ValueError("Use NaN for missing landmarks; infinity is invalid.")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if self.uncertainty_m is not None:
            values = np.asarray(self.uncertainty_m, dtype=float)
            if values.ndim == 0:
                values = np.full(len(labels), float(values))
            else:
                values = np.array(values, copy=True)
            if values.shape != (len(labels),) or not np.all(
                np.isfinite(values) & (values > 0)
            ):
                raise ValueError(
                    "uncertainty_m must be positive and finite per landmark."
                )
            values.setflags(write=False)
            object.__setattr__(self, "uncertainty_m", values)


def _array_digest(points: np.ndarray) -> str:
    values = np.array(points, dtype="<f8", copy=True)
    values[np.isnan(values)] = np.nan
    return hashlib.sha256(values.tobytes()).hexdigest()


def _geometry_information(points: np.ndarray, required_rank: int) -> tuple[int, float]:
    if not len(points):
        return 0, float("inf")
    centred = points - points.mean(axis=0)
    rank = int(np.linalg.matrix_rank(centred))
    if rank < required_rank:
        return rank, float("inf")
    values = np.linalg.svd(centred, compute_uv=False)
    return rank, float(values[0] / values[required_rank - 1])


def _jacobian_condition(
    points: np.ndarray,
    scales: np.ndarray,
    rotation: Rotation,
    sigma: np.ndarray,
    mode: str,
) -> tuple[int, float]:
    scaled = points * scales
    matrix = rotation.as_matrix()
    columns = (
        [rotation.apply(scaled)]
        if mode == "isotropic"
        else [scaled[:, axis, None] * matrix[:, axis] for axis in range(3)]
    )
    columns.extend(rotation.apply(np.cross(axis, scaled)) for axis in np.eye(3))
    columns.extend(np.broadcast_to(axis, scaled.shape) for axis in np.eye(3))
    jacobian = np.stack(columns, axis=-1)
    jacobian *= (sigma.min() / sigma)[:, None, None]
    jacobian = jacobian.reshape(-1, len(columns))
    norms = np.linalg.norm(jacobian, axis=0)
    if np.any(norms == 0):
        return 0, float("inf")
    values = np.linalg.svd(jacobian / norms, compute_uv=False)
    rank = int(np.sum(values > values[0] * max(jacobian.shape) * np.finfo(float).eps))
    return rank, float(values[0] / values[-1])


def _fit_segment(
    case: LandmarkCorrespondences,
    *,
    mode: str,
    loss: str,
    robust_scale_m: float,
    scale_bounds: tuple[float, float],
    max_rms_m: float | None,
    max_condition_number: float,
    max_evaluations: int,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    reference = np.asarray(case.reference_positions_m)
    observed = np.asarray(case.observed_positions_m)
    valid = np.all(np.isfinite(reference) & np.isfinite(observed), axis=1)
    reference, observed = reference[valid], observed[valid]
    minimum = 4 if mode == "anisotropic" else 3
    required_rank = 3 if mode == "anisotropic" else 2
    reference_rank, reference_condition = _geometry_information(
        reference, required_rank
    )
    observed_rank, observed_condition = _geometry_information(observed, required_rank)
    if (
        len(reference) < minimum
        or reference_rank < required_rank
        or observed_rank < required_rank
    ):
        raise ValueError(
            f"{case.segment}: {mode} fitting needs {minimum} valid landmarks "
            f"with geometric rank at least {required_rank}."
        )
    geometry_condition = max(reference_condition, observed_condition)
    if not np.isfinite(geometry_condition) or geometry_condition > max_condition_number:
        raise ValueError(f"{case.segment}: landmark geometry is ill-conditioned.")
    if any(
        len(np.unique(points, axis=0)) != len(points)
        for points in (reference, observed)
    ):
        raise ValueError(
            "Repeated landmark positions cannot be independent correspondences."
        )
    sigma = (
        np.full(len(reference), robust_scale_m)
        if case.uncertainty_m is None
        else np.asarray(case.uncertainty_m)[valid]
    )
    weights = (sigma.min() / sigma) ** 2
    centre_ref = np.average(reference, axis=0, weights=weights)
    centre_obs = np.average(observed, axis=0, weights=weights)
    local, target = reference - centre_ref, observed - centre_obs
    low, high = scale_bounds
    if mode == "anisotropic":
        affine = np.linalg.lstsq(
            local * np.sqrt(weights[:, None]),
            target * np.sqrt(weights[:, None]),
            rcond=None,
        )[0]
        initial_scales = np.clip(np.linalg.norm(affine, axis=1), low, high)
    else:
        initial_rotation, _ = Rotation.align_vectors(target, local, weights=weights)
        aligned = initial_rotation.apply(local)
        scale = np.sum(weights[:, None] * target * aligned) / np.sum(
            weights[:, None] * local**2
        )
        initial_scales = np.full(3, np.clip(scale, low, high))
    initial_rotation, _ = Rotation.align_vectors(
        target,
        local * initial_scales,
        weights=weights,
    )
    count = 3 if mode == "anisotropic" else 1

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, Rotation, np.ndarray]:
        scales = np.exp(parameters[:count])
        if count == 1:
            scales = np.repeat(scales, 3)
        return (
            scales,
            Rotation.from_rotvec(parameters[count : count + 3]),
            parameters[-3:],
        )

    def residual(parameters: np.ndarray) -> np.ndarray:
        scales, rotation, translation = unpack(parameters)
        return (
            (rotation.apply(local * scales) + translation - target) / sigma[:, None]
        ).ravel()

    initial = np.r_[
        np.log(initial_scales[:count]), initial_rotation.as_rotvec(), np.zeros(3)
    ]
    result = least_squares(
        residual,
        initial,
        bounds=(
            np.r_[np.full(count, np.log(low)), np.full(6, -np.inf)],
            np.r_[np.full(count, np.log(high)), np.full(6, np.inf)],
        ),
        loss=loss,
        f_scale=1.0,
        jac="3-point",
        x_scale="jac",
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
        max_nfev=max_evaluations,
    )
    if not result.success:
        raise ValueError(
            f"{case.segment}: scale fit did not converge: {result.message}"
        )
    scales, rotation, translation = unpack(result.x)
    translation = translation + centre_obs - rotation.apply(centre_ref * scales)
    errors = rotation.apply(reference * scales) + translation - observed
    distances = np.linalg.norm(errors, axis=1)
    rms = float(np.sqrt(np.mean(distances**2)))
    rank, condition = _jacobian_condition(local, scales, rotation, sigma, mode)
    if (
        rank != count + 6
        or not np.isfinite(condition)
        or condition > max_condition_number
    ):
        raise ValueError(
            f"{case.segment}: landmark fit is rank-deficient or ill-conditioned."
        )
    if max_rms_m is not None and rms > max_rms_m:
        raise ValueError(
            f"{case.segment}: marker RMS {rms:g} m exceeds max_rms_m={max_rms_m:g}; "
            "check correspondences, frames, units and anatomical compatibility."
        )
    notes = []
    if not np.all(valid):
        notes.append(
            f"{case.segment}: {int((~valid).sum())} missing landmarks omitted."
        )
    if mode == "isotropic":
        notes.append(f"{case.segment}: isotropic scaling was explicitly assumed.")
    if case.uncertainty_m is None:
        notes.append(
            f"{case.segment}: equal landmark weights; source uncertainty unavailable."
        )
    elif np.any(distances > 3 * sigma):
        notes.append(
            f"{case.segment}: fitted residuals exceed three source coordinate "
            "standard deviations; inspect inconsistent landmarks."
        )
    if max_rms_m is None:
        notes.append(
            f"{case.segment}: no physical RMS admission threshold was supplied."
        )
    if np.any(
        np.isclose(scales, low, atol=1e-6, rtol=0)
        | np.isclose(scales, high, atol=1e-6, rtol=0)
    ):
        notes.append(f"{case.segment}: fitted scale reaches a configured bound.")
    row = {
        "segment": case.segment,
        "subject_id": case.subject_id,
        "reference_source": case.reference_source,
        "observed_source": case.observed_source,
        "reference_positions_sha256": _array_digest(
            np.asarray(case.reference_positions_m)
        ),
        "observed_positions_sha256": _array_digest(
            np.asarray(case.observed_positions_m)
        ),
        "used_labels": [label for label, used in zip(case.labels, valid) if used],
        "missing_labels": [
            label for label, used in zip(case.labels, valid) if not used
        ],
        "factors": scales.tolist(),
        "rotation_reference_to_observed": rotation.as_matrix().tolist(),
        "translation_m": translation.tolist(),
        "residuals_m": errors.tolist(),
        "rms_m": rms,
        "max_error_m": float(distances.max()),
        "fit_jacobian_rank": rank,
        "fit_condition_number": condition,
        "reference_geometry_rank": reference_rank,
        "observed_geometry_rank": observed_rank,
        "geometry_condition_number": geometry_condition,
        "uncertainty_m": None if case.uncertainty_m is None else sigma.tolist(),
        "evaluations": int(result.nfev),
    }
    return scales, row, notes


def fit_landmark_scales(
    correspondences: Sequence[LandmarkCorrespondences],
    *,
    mode: Literal["anisotropic", "isotropic"] = "anisotropic",
    loss: Literal["linear", "soft_l1", "huber"] = "linear",
    robust_scale_m: float = 0.005,
    scale_bounds: tuple[float, float] = (0.5, 2.0),
    max_rms_m: float | None = None,
    max_condition_number: float = 1e8,
    max_evaluations: int = 500,
    height_m: float | None = None,
    mass_kg: float | None = None,
) -> MarkerScaleResult:
    """Fit local-axis scale factors and a proper rigid pose for each segment.

    Args:
        correspondences: One labelled geometry pair per segment of one subject.
        mode: Three independent factors, or an explicit isotropic assumption.
        loss: SciPy robust loss applied to normalized coordinate residuals.
        robust_scale_m: Algorithmic residual scale when uncertainty is unknown;
            it is not a source-reported uncertainty or confidence interval.
        scale_bounds: Positive lower and upper bounds for each scale factor.
        max_rms_m: Optional strict physical RMS threshold over all valid points.
            Reflected/mislabelled anatomy cannot silently pass a strict fit.
        max_condition_number: Maximum column-normalized geometric Jacobian condition.
        max_evaluations: Maximum least-squares evaluations per segment.
        height_m: Optional recorded subject height; does not infer factors.
        mass_kg: Optional target mass, consumed by the existing subject builder.

    Returns:
        A source-neutral estimate and JSON evidence. Predicted observed points
        are ``(reference * factors) @ rotation.T + translation_m``. The rotation
        and translation are nuisance alignment, not edits to target joint frames.

    Correspondences must describe the same anatomical locations. Surface points
    are not joint centres. Full 3D geometry is required for anisotropic fitting;
    three non-collinear points suffice only with explicit isotropic scaling.
    Geometry fit does not validate muscle paths, contacts or dynamics. The
    source-reported coordinate uncertainties are assumed independent; repeated
    correlated frames must be aggregated before calling this static fitter.
    """
    cases = tuple(correspondences)
    if not cases:
        raise ValueError("At least one segment correspondence set is required.")
    if len({case.subject_id for case in cases}) != 1:
        raise ValueError(
            "Cannot pool different observed subjects into one scale estimate."
        )
    if len({case.segment for case in cases}) != len(cases):
        raise ValueError(
            "Each segment requires one correspondence set; duplicate segments found."
        )
    if mode not in ("anisotropic", "isotropic") or loss not in (
        "linear",
        "soft_l1",
        "huber",
    ):
        raise ValueError("Unsupported fitting mode or loss.")
    if not np.isfinite(robust_scale_m) or robust_scale_m <= 0:
        raise ValueError("robust_scale_m must be finite and positive.")
    bounds = np.asarray(scale_bounds, dtype=float)
    if (
        bounds.shape != (2,)
        or not np.all(np.isfinite(bounds))
        or not 0 < bounds[0] < bounds[1]
    ):
        raise ValueError("scale_bounds must be positive, finite and increasing.")
    if max_rms_m is not None and (not np.isfinite(max_rms_m) or max_rms_m <= 0):
        raise ValueError("max_rms_m must be finite and positive.")
    if not np.isfinite(max_condition_number) or max_condition_number <= 1:
        raise ValueError("max_condition_number must be finite and greater than one.")
    if (
        isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, int)
        or max_evaluations <= 0
    ):
        raise ValueError("max_evaluations must be a positive integer.")
    factors, rows, notes = {}, [], []
    for case in cases:
        scales, row, warnings = _fit_segment(
            case,
            mode=mode,
            loss=loss,
            robust_scale_m=robust_scale_m,
            scale_bounds=(float(bounds[0]), float(bounds[1])),
            max_rms_m=max_rms_m,
            max_condition_number=max_condition_number,
            max_evaluations=max_evaluations,
        )
        factors[case.segment] = scales
        rows.append(row)
        notes.extend(warnings)
    evidence = {
        "schema": 1,
        "adapter": "corresponding-3d-landmarks",
        "subject_id": cases[0].subject_id,
        "scale_axes": "reference target body/segment local axes",
        "mode": mode,
        "loss": loss,
        "weighting": "coordinate residual / source sigma; equal weights if unavailable",
        "robust_scale_m": float(robust_scale_m),
        "scale_bounds": bounds.tolist(),
        "max_rms_m": None if max_rms_m is None else float(max_rms_m),
        "segments": rows,
    }
    identity = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    estimate = ScaleEstimate(
        factors,
        height_m=height_m,
        mass_kg=mass_kg,
        source=f"landmark-fit:{cases[0].subject_id}@{identity}",
        warnings=tuple(notes),
    )
    return MarkerScaleResult(estimate, evidence)
