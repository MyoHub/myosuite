# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""User-defined vertex markers for body-model geometry such as SMPL exports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from numbers import Integral
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from myosuite.integrations.capture.markers import MarkerTrial, virtual_marker_trial


@dataclass(frozen=True)
class VertexMarkerSet:
    """User-supplied indices tied to one explicitly identified vertex ordering.

    Args:
        topology: Exact model/version/vertex-order identifier, not a guessed family.
        vertex_count: Required number of vertices in each source frame.
        indices: Marker labels mapped to zero-based vertex indices.
        definition_source: Origin of the marker definitions.
        definition_license: Source-reported terms; not a redistribution grant.
    """

    topology: str
    vertex_count: int
    indices: Mapping[str, int]
    definition_source: str
    definition_license: str

    def __post_init__(self) -> None:
        for name in ("topology", "definition_source", "definition_license"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string.")
        if (
            isinstance(self.vertex_count, bool)
            or not isinstance(self.vertex_count, Integral)
            or self.vertex_count <= 0
        ):
            raise ValueError("vertex_count must be a positive integer.")
        if not self.indices or any(
            not isinstance(label, str) or not label.strip() for label in self.indices
        ):
            raise ValueError("Vertex markers require nonempty string labels.")
        indices = {}
        for label, index in self.indices.items():
            if (
                isinstance(index, bool)
                or not isinstance(index, Integral)
                or not 0 <= index < self.vertex_count
            ):
                raise ValueError(
                    f"Marker {label!r} requires an in-range integer index."
                )
            indices[label] = int(index)
        object.__setattr__(self, "vertex_count", int(self.vertex_count))
        object.__setattr__(self, "indices", MappingProxyType(indices))

    def metadata(self) -> dict[str, Any]:
        """Return serializable topology and marker-definition provenance."""
        return {
            "topology": self.topology,
            "vertex_count": self.vertex_count,
            "indices": dict(self.indices),
            "definition_source": self.definition_source,
            "definition_license": self.definition_license,
        }


@dataclass(frozen=True, kw_only=True)
class _VertexMarkerTrial(MarkerTrial):
    vertex_markers: VertexMarkerSet

    def metadata(self) -> dict[str, Any]:
        return {**super().metadata(), "vertex_markers": self.vertex_markers.metadata()}


def vertex_marker_trial(
    vertices: ArrayLike,
    definitions: VertexMarkerSet,
    *,
    topology: str,
    units: Literal["m", "cm", "mm"],
    rotation_to_model: ArrayLike,
    source: str,
    translation_to_model_m: ArrayLike = (0.0, 0.0, 0.0),
    frequency_hz: float | None = None,
    confidence: float | None = None,
    uncertainty_m: float | None = None,
    point_semantics: str = "surface_vertex",
    subject_id: str | None = None,
) -> MarkerTrial:
    """Extract markers from static or animated user-provided body-model vertices.

    Args:
        vertices: Source coordinates shaped ``(vertices, 3)`` or
            ``(frames, vertices, 3)``. NaN marks missing observations.
        definitions: User-owned marker indices and their topology/provenance.
        topology: Identifier of the supplied geometry's actual vertex ordering.
        units: Explicit source coordinate units.
        rotation_to_model: Calibrated proper rotation to model-world coordinates.
        source: Identifier of the supplied geometry or fit.
        translation_to_model_m: Model-world translation in metres.
        frequency_hz: Motion sampling frequency, required by retargeting.
        confidence: Optional source-reported confidence.
        uncertainty_m: Optional source-reported coordinate uncertainty.
        point_semantics: Caller-declared point kind; defaults to surface vertices.
        subject_id: Optional caller-supplied identity for local provenance.

    Returns:
        A marker trial accepted by scale estimation and target-specific IK.

    No model loader or built-in marker definitions are included. Identity and
    count checks cannot detect reordered vertices with an unchanged identifier.
    Surface markers do not automatically represent anatomical joint centres.
    """
    if not isinstance(topology, str) or topology != definitions.topology:
        raise ValueError("Geometry topology must match the vertex marker definitions.")
    points = np.asarray(vertices, dtype=float)
    if points.ndim == 2:
        points = points[None, ...]
    if (
        points.ndim != 3
        or not len(points)
        or points.shape[1:] != (definitions.vertex_count, 3)
    ):
        raise ValueError(
            "vertices must have shape (vertex_count, 3) or (frames, vertex_count, 3)."
        )
    if np.any(np.isinf(points)):
        raise ValueError("Use NaN for missing vertices; infinity is invalid.")
    trial = virtual_marker_trial(
        labels=tuple(definitions.indices),
        positions=points[:, list(definitions.indices.values())],
        units=units,
        rotation_to_model=rotation_to_model,
        source=source,
        translation_to_model_m=translation_to_model_m,
        frequency_hz=frequency_hz,
        confidence=confidence,
        uncertainty_m=uncertainty_m,
        definition_source=definitions.definition_source,
        definition_license=definitions.definition_license,
        point_semantics=point_semantics,
        subject_id=subject_id,
    )
    return _VertexMarkerTrial(
        **{field.name: getattr(trial, field.name) for field in fields(MarkerTrial)},
        vertex_markers=definitions,
    )
