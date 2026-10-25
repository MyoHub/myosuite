# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Pickle-free local marker-trial interchange with complete source metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from myosuite.integrations.capture.markers import MarkerTrial

_SCHEMA_VERSION = 1
_ARCHIVE_FIELDS = {"schema_version", "positions", "metadata_json"}
_TRIAL_FIELDS = {
    "labels",
    "units",
    "rotation_to_model",
    "source",
    "translation_to_model_m",
    "confidence",
    "uncertainty_m",
    "frequency_hz",
    "definition_source",
    "definition_license",
    "point_semantics",
    "subject_id",
}


@dataclass(frozen=True, kw_only=True)
class _StoredMarkerTrial(MarkerTrial):
    _stored_metadata_json: str

    def metadata(self) -> dict[str, Any]:
        """Return retained reader metadata and the current base trial fields."""
        return {**json.loads(self._stored_metadata_json), **super().metadata()}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate metadata field: {key}.")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Nonfinite metadata value: {value}.")


def _validate_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict) or not _TRIAL_FIELDS <= metadata.keys():
        raise ValueError("Marker-trial metadata is missing required fields.")
    labels = metadata["labels"]
    if not isinstance(labels, list) or any(not isinstance(x, str) for x in labels):
        raise ValueError("Marker-trial labels must be a list of strings.")
    for name in ("source", "units", "point_semantics"):
        if not isinstance(metadata[name], str):
            raise ValueError(f"Marker-trial {name} must be a string.")
    for name in ("definition_source", "definition_license", "subject_id"):
        value = metadata[name]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"Marker-trial {name} must be nonempty text or null.")
    for name in ("confidence", "uncertainty_m", "frequency_hz"):
        value = metadata[name]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Real)
        ):
            raise ValueError(f"Marker-trial {name} must be numeric or null.")
    for name in ("rotation_to_model", "translation_to_model_m"):
        if np.asarray(metadata[name]).dtype.kind not in "fiu":
            raise ValueError(f"Marker-trial {name} must contain numeric values.")
    return {name: metadata[name] for name in _TRIAL_FIELDS}


def save_marker_trial(trial: MarkerTrial, path: str | Path) -> Path:
    """Save a complete trial to a versioned NPZ without pickle or unit changes.

    Args:
        trial: Original coordinates, convention and reader-specific provenance.
        path: Exact destination filename; parent directories are created.

    Returns:
        The absolute path of the saved local workflow artifact.

    Coordinates, subject identifiers and reader metadata may be restricted or
    identifying. Saving them does not confer permission to redistribute them.
    """
    if not isinstance(trial, MarkerTrial):
        raise TypeError("trial must be a MarkerTrial.")
    metadata = trial.metadata()
    _validate_metadata(metadata)
    base = MarkerTrial.metadata(trial)
    if any(metadata[name] != value for name, value in base.items()):
        raise ValueError("Reader metadata conflicts with the base marker trial.")
    encoded = json.dumps(metadata, ensure_ascii=False, allow_nan=False, sort_keys=True)
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.array(_SCHEMA_VERSION, dtype=np.int64),
            positions=np.asarray(trial.positions),
            metadata_json=np.array(encoded),
        )
    return path


def load_marker_trial(path: str | Path) -> MarkerTrial:
    """Load a validated local trial, preserving missing points and extra metadata.

    Args:
        path: A file written by :func:`save_marker_trial` using a supported schema.

    Returns:
        A marker trial with the same coordinates, convention and full metadata.
        Reader-specific Python classes are not reconstructed or executed.

    Raises:
        ValueError: The schema, archive fields or trial metadata are malformed.
    """
    archive = np.load(Path(path), allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("Marker-trial input must be a versioned NPZ archive.")
    with archive:
        if (
            len(archive.files) != len(_ARCHIVE_FIELDS)
            or set(archive.files) != _ARCHIVE_FIELDS
        ):
            raise ValueError("Marker-trial archive fields do not match the schema.")
        version = archive["schema_version"]
        if (
            version.shape != ()
            or version.dtype.kind not in "iu"
            or int(version) != _SCHEMA_VERSION
        ):
            raise ValueError("Unsupported marker-trial schema version.")
        encoded = archive["metadata_json"]
        if encoded.shape != () or encoded.dtype.kind != "U":
            raise ValueError("Marker-trial metadata must be a Unicode JSON scalar.")
        metadata = json.loads(
            str(encoded),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
        fields = _validate_metadata(metadata)
        positions = archive["positions"]
        if positions.dtype.kind not in "fiu":
            raise ValueError("Marker-trial positions must be a numeric array.")
        try:
            return _StoredMarkerTrial(
                positions=positions,
                **fields,
                _stored_metadata_json=json.dumps(metadata, allow_nan=False),
            )
        except (TypeError, OverflowError) as exc:
            raise ValueError("Malformed marker-trial field values.") from exc
