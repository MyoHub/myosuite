# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Private readers for labelled marker files: OpenSim TRC, C3D and landmark CSV.

Each reader returns marker names, positions of shape ``(frames, markers, 3)``
in metres (NaN where a marker is missing) and the frame rate in Hz.
Coordinates stay in the file's lab frame.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import NamedTuple

import numpy as np

_METRES = {"m": 1.0, "cm": 0.01, "mm": 0.001}


class _MarkerData(NamedTuple):
    names: tuple[str, ...]
    positions: np.ndarray
    rate: float | None


def _read_markers(path: str | Path) -> _MarkerData:
    """Read a ``.trc``, ``.c3d`` or landmark ``.csv`` file by its suffix."""
    path = Path(path)
    readers = {".trc": _read_trc, ".c3d": _read_c3d, ".csv": _read_landmarks_csv}
    reader = readers.get(path.suffix.lower())
    if reader is None:
        raise ValueError(
            f"Unsupported marker file {path.name!r}; use .trc, .c3d or .csv."
        )
    data = reader(path)
    if not data.names or len(set(data.names)) != len(data.names) or not all(data.names):
        raise ValueError(f"{path.name}: marker names must be unique and nonempty.")
    return data


def _scale(units: str, path: Path) -> float:
    try:
        return _METRES[units.strip().lower()]
    except KeyError:
        raise ValueError(f"{path.name}: units {units!r} are not m, cm or mm.") from None


def _read_trc(path: Path) -> _MarkerData:
    """Tab-separated OpenSim TRC with DataRate, NumFrames, NumMarkers and Units."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 5 or not lines[0].startswith("PathFileType"):
        raise ValueError(f"{path.name} is not a TRC file (no PathFileType line).")
    header = dict(zip(lines[1].split(), lines[2].split()))
    try:
        rate, frames = float(header["DataRate"]), int(header["NumFrames"])
        count, units = int(header["NumMarkers"]), header["Units"]
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"{path.name}: the TRC header needs DataRate, NumFrames, NumMarkers and Units."
        ) from exc
    if not np.isfinite(rate) or rate <= 0 or count <= 0:
        raise ValueError(f"{path.name}: DataRate and NumMarkers must be positive.")
    labels = lines[3].split("\t")
    names = tuple(
        labels[i].strip() if i < len(labels) else "" for i in range(2, 2 + 3 * count, 3)
    )
    width = 2 + 3 * count
    rows = [
        line + "\t" * max(0, width - 1 - line.count("\t"))
        for line in lines[5:]
        if line.strip()
    ]
    if len(rows) != frames:
        raise ValueError(
            f"{path.name}: NumFrames is {frames} but there are {len(rows)} rows."
        )
    table = np.genfromtxt(
        rows, delimiter="\t", usecols=range(width), filling_values=np.nan, ndmin=2
    )
    times = table[:, 1]
    if np.any(~np.isfinite(times)) or np.any(
        np.abs(np.diff(times) - 1 / rate) > 0.25 / rate
    ):
        raise ValueError(
            f"{path.name}: the Time column does not match DataRate {rate:g} Hz."
        )
    positions = table[:, 2:].reshape(frames, count, 3) * _scale(units, path)
    return _MarkerData(names, positions, rate)


def _read_c3d(path: Path) -> _MarkerData:
    """C3D points through the optional ``ezc3d``; negative residuals mark gaps."""
    try:
        import ezc3d
    except ImportError as exc:
        raise ImportError(
            "Reading .c3d files needs ezc3d: pip install 'myosuite[capture]'."
        ) from exc
    capture = ezc3d.c3d(str(path))
    point = capture["parameters"]["POINT"]
    labels = list(point["LABELS"]["value"])
    if "LABELS2" in point:  # C3D stores labels beyond the 255th here
        labels += list(point["LABELS2"]["value"])
    xyz = np.asarray(capture["data"]["points"][:3], dtype=float)  # (3, markers, frames)
    positions = xyz.transpose(2, 1, 0).copy()
    residuals = np.asarray(capture["data"]["meta_points"]["residuals"])[0].T
    positions[residuals < 0] = np.nan
    names = tuple(str(label).strip() for label in labels[: positions.shape[1]])
    scale = _scale(str(point["UNITS"]["value"][0]), path)
    return _MarkerData(names, positions * scale, float(point["RATE"]["value"][0]))


def _read_landmarks_csv(path: Path) -> _MarkerData:
    """One static frame of ``label,x,y,z`` rows in metres; a header row is skipped."""
    names, coords, header = [], [], False
    for row in csv.reader(path.read_text(encoding="utf-8-sig").splitlines()):
        if not any(cell.strip() for cell in row):
            continue
        if len(row) != 4:
            raise ValueError(f"{path.name}: landmark rows must be label,x,y,z.")
        try:
            point = [float(cell) if cell.strip() else np.nan for cell in row[1:]]
        except ValueError:
            if names or header:
                raise ValueError(
                    f"{path.name}: coordinates must be numbers or empty."
                ) from None
            header = True  # e.g. "label,x,y,z"
            continue
        names.append(row[0].strip())
        coords.append(point)
    return _MarkerData(
        tuple(names), np.array(coords, dtype=float).reshape(1, -1, 3), None
    )
