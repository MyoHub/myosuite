# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Source-neutral named landmark CSV inputs with explicit coordinate conventions."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from myosuite.integrations.capture.markers import MarkerTrial


@dataclass(frozen=True)
class _LandmarkTrial(MarkerTrial):
    _evidence_json: str = "{}"

    def metadata(self) -> dict[str, Any]:
        """Retain the exact input file identity alongside coordinate provenance."""
        return {**super().metadata(), "landmark_csv": json.loads(self._evidence_json)}


def read_named_landmarks(
    path: str | Path,
    *,
    units: Literal["m", "cm", "mm"],
    rotation_to_model: ArrayLike,
    source: str,
    definition_source: str,
    definition_license: str,
    translation_to_model_m: ArrayLike = (0.0, 0.0, 0.0),
    uncertainty_m: float | None = None,
    point_semantics: str = "unspecified",
    subject_id: str | None = None,
) -> MarkerTrial:
    """Read headerless ``label,x,y,z`` coordinates without inferring anatomy.

    Args:
        path: Local CSV containing unique labels and numeric or empty coordinates.
        units: Explicit source coordinate units.
        rotation_to_model: Calibrated proper source-to-model rotation.
        source: Input source/version identifier.
        definition_source: Origin of the landmark definitions.
        definition_license: Source-reported terms, not a redistribution grant.
        translation_to_model_m: Translation in model-world metres.
        uncertainty_m: Optional source-reported coordinate standard uncertainty.
        point_semantics: Caller-described meaning, such as surface annotations.
        subject_id: Optional explicit subject identity for matching local inputs.

    Returns:
        One static marker frame; empty coordinates and NaN remain missing.

    This reader does not infer joint centres or label correspondences. The
    caller retains responsibility for anatomical definitions and source rights.
    """
    for name, value in (
        ("definition_source", definition_source),
        ("definition_license", definition_license),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be supplied.")
    path = Path(path).resolve()
    raw = path.read_bytes()
    labels, positions = [], []
    for row in csv.reader(io.StringIO(raw.decode("utf-8-sig"))):
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) != 4:
            raise ValueError("Landmark CSV rows must contain label,x,y,z.")
        labels.append(row[0].strip())
        try:
            positions.append(
                [float(value) if value.strip() else np.nan for value in row[1:]]
            )
        except ValueError as exc:
            raise ValueError("Landmark coordinates must be numeric or empty.") from exc
    return _LandmarkTrial(
        labels=tuple(labels),
        positions=np.asarray(positions)[None],
        units=units,
        rotation_to_model=rotation_to_model,
        source=source,
        translation_to_model_m=translation_to_model_m,
        uncertainty_m=uncertainty_m,
        definition_source=definition_source,
        definition_license=definition_license,
        point_semantics=point_semantics,
        subject_id=subject_id,
        _evidence_json=json.dumps(
            {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
        ),
    )
