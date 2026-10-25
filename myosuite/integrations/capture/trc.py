# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""OpenSim marker trajectories for capture and external body-model workflows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from math import floor, log10
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from myosuite.integrations.capture.markers import MarkerTrial


@dataclass(frozen=True, kw_only=True)
class _TrcMarkerTrial(MarkerTrial):
    _trc_json: str

    def metadata(self) -> dict[str, Any]:
        return {**super().metadata(), "trc": json.loads(self._trc_json)}


def _columns(line: str, width: int) -> list[str]:
    columns = [value.strip() for value in line.split("\t")]
    if len(columns) > width and not any(columns[width:]):
        columns = columns[:width]
    if len(columns) != width:
        raise ValueError(f"TRC row needs {width} tab-separated columns.")
    return columns


def _rounding_step(text: str, value: float) -> float:
    if value == 0:
        return 0.0
    return min(
        10.0 ** Decimal(text).as_tuple().exponent,
        10.0 ** (floor(log10(abs(value))) - 5),
    )


def read_trc(
    path: str | Path,
    *,
    rotation_to_model: ArrayLike,
    translation_to_model_m: ArrayLike = (0.0, 0.0, 0.0),
    confidence: float | None = None,
    uncertainty_m: float | None = None,
    definition_source: str | None = None,
    definition_license: str | None = None,
) -> MarkerTrial:
    """Read uniformly sampled, tab-delimited OpenSim TRC marker coordinates.

    Args:
        path: Local TRC file, with DataRate, NumFrames, NumMarkers and Units.
        rotation_to_model: Calibrated proper source-to-model world rotation.
        translation_to_model_m: Model-world translation in metres.
        confidence: Optional source-reported confidence.
        uncertainty_m: Optional source-reported coordinate uncertainty.
        definition_source: Optional origin of the marker definitions.
        definition_license: Optional source-reported terms for those definitions.

    Returns:
        A trial for scale estimation or target-specific retargeting. Empty cells
        and NaN remain missing; no relabelling or interpolation occurs.

    Frame numbers must be consecutive and times agree with DataRate, allowing
    bounded six-significant-digit export rounding. The original time origin,
    frame range, header, timing tolerance and file hash are retained in metadata.
    """
    path = Path(path).resolve()
    raw = path.read_bytes()
    lines = raw.decode("utf-8-sig").splitlines()
    if len(lines) < 6 or not lines[0].startswith("PathFileType"):
        raise ValueError("Expected an OpenSim TRC header and marker rows.")
    keys, values = lines[1].split(), lines[2].split()
    if len(keys) != len(values) or len(set(keys)) != len(keys):
        raise ValueError("TRC metadata keys and values must match uniquely.")
    header = dict(zip(keys, values))
    try:
        rate = float(header["DataRate"])
        frame_count = int(header["NumFrames"])
        marker_count = int(header["NumMarkers"])
        units = header["Units"].lower()
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "TRC requires DataRate, NumFrames, NumMarkers and Units."
        ) from exc
    if not np.isfinite(rate) or rate <= 0 or frame_count <= 0 or marker_count <= 0:
        raise ValueError("TRC sampling rate and frame/marker counts must be positive.")
    if units not in ("m", "cm", "mm"):
        raise ValueError("TRC Units must be m, cm, or mm.")
    width = 2 + 3 * marker_count
    marker_line = lines[3].split("\t")
    if len(marker_line) < width:
        marker_line.extend([""] * (width - len(marker_line)))
    marker_columns = _columns("\t".join(marker_line), width)
    coordinates = _columns(lines[4], width)
    if marker_columns[:2] != ["Frame#", "Time"]:
        raise ValueError("TRC column labels must start with Frame# and Time.")
    if any(marker_columns[3::3]) or any(marker_columns[4::3]):
        raise ValueError("TRC marker labels must occupy every third coordinate column.")
    expected = [
        f"{axis}{index}" for index in range(1, marker_count + 1) for axis in "XYZ"
    ]
    if coordinates[:2] != ["", ""] or coordinates[2:] != expected:
        raise ValueError("TRC coordinate labels must be X1, Y1, Z1, X2, Y2, Z2, ...")
    rows = [_columns(line, width) for line in lines[5:] if line.strip()]
    if len(rows) != frame_count:
        raise ValueError("TRC NumFrames does not match the data rows.")
    try:
        data = np.array(
            [[float(value) if value else np.nan for value in row] for row in rows]
        )
    except ValueError as exc:
        raise ValueError("TRC data cells must be numeric or empty.") from exc
    frames, times = data[:, 0], data[:, 1]
    if (
        not np.all(np.isfinite(data[:, :2]))
        or np.any(frames != np.floor(frames))
        or np.any(np.diff(frames) != 1)
    ):
        raise ValueError(
            "TRC needs finite times and consecutive integer frame numbers."
        )
    elapsed = np.arange(frame_count) / rate
    half_steps = np.array(
        [_rounding_step(row[1], time) / 2 for row, time in zip(rows, times)]
    )
    rate_fraction = _rounding_step(header["DataRate"], rate) / (2 * rate)
    tolerance = np.minimum(
        0.25 / rate,
        1e-6
        + half_steps
        + half_steps[0]
        + elapsed * rate_fraction / (1 - rate_fraction),
    )
    if np.any(np.diff(times) <= 0) or np.any(
        np.abs(times - times[0] - elapsed) > tolerance
    ):
        raise ValueError("TRC times must be uniform and agree with DataRate.")
    return _TrcMarkerTrial(
        labels=tuple(marker_columns[2::3]),
        positions=data[:, 2:].reshape(frame_count, marker_count, 3),
        units=units,
        rotation_to_model=rotation_to_model,
        source=str(path),
        translation_to_model_m=translation_to_model_m,
        confidence=confidence,
        uncertainty_m=uncertainty_m,
        frequency_hz=rate,
        definition_source=definition_source,
        definition_license=definition_license,
        _trc_json=json.dumps(
            {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "header": header,
                "first_frame": int(frames[0]),
                "last_frame": int(frames[-1]),
                "start_time_s": float(times[0]),
                "end_time_s": float(times[-1]),
                "timestamp_tolerance_s_max": float(np.max(tolerance)),
            }
        ),
    )
