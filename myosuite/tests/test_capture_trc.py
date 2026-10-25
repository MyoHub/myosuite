# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""TRC interchange preserves missing cells, calibrated frames and source timing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite.core.subject_model import (
    GroundAlignment,
    SubjectModelConfig,
    build_subject_model,
)
from myosuite.integrations.capture import (
    SegmentMeasurement,
    estimate_marker_scales,
    read_trc,
)
from myosuite.integrations.capture.example import FIXTURES, MARKER_SITES
from myosuite.integrations.capture.retargeting import retarget_markers

pytestmark = pytest.mark.tier1


def _write(
    path: Path,
    positions: np.ndarray | None = None,
    labels: tuple[str, ...] = ("HIP", "KNEE"),
    rate: float = 100,
    units: str = "mm",
) -> Path:
    if positions is None:
        positions = np.array([[[20, 30, 600], [20, 30, 0]]] * 3, dtype=float)
    count, markers = positions.shape[:2]
    rows = [
        "PathFileType\t4\t(X/Y/Z)\tsynthetic.trc",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{rate}\t{rate}\t{count}\t{markers}\t{units}\t{rate}\t201\t{count}",
        "Frame#\tTime\t"
        + "\t".join(value for label in labels for value in (label, "", "")),
        "\t\t"
        + "\t".join(f"{axis}{i}" for i in range(1, markers + 1) for axis in "XYZ"),
        "",
    ]
    for i, points in enumerate(positions):
        cells = [str(i + 201), f"{10 + i / rate:.8f}"]
        cells.extend("" if np.isnan(value) else str(value) for value in points.ravel())
        rows.append("\t".join(cells))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_real_format_metadata_calibration_and_scale_evidence(tmp_path: Path) -> None:
    path = _write(tmp_path / "static.trc")
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    trial = read_trc(
        path,
        rotation_to_model=rotation,
        translation_to_model_m=(1, 2, 3),
        definition_source="authored synthetic landmarks",
        definition_license="Apache-2.0",
    )
    np.testing.assert_allclose(
        trial.in_model_frame()[0], [[0.97, 2.02, 3.6], [0.97, 2.02, 3]]
    )
    assert trial.frequency_hz == 100
    result = estimate_marker_scales(
        trial, [SegmentMeasurement("thigh", ("HIP", "KNEE"), 0.5, axis="z")]
    )
    np.testing.assert_allclose(result.estimate.factors["thigh"], [1, 1, 1.2])
    evidence = result.evidence["trial"]["trc"]
    assert evidence["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert evidence["first_frame"] == 201
    assert evidence["start_time_s"] == 10
    assert evidence["end_time_s"] == 10.02
    assert result.evidence["trial"]["definition_license"] == "Apache-2.0"
    json.dumps(result.evidence, allow_nan=False)


def test_missing_cells_do_not_shift_marker_coordinates(tmp_path: Path) -> None:
    points = np.array(
        [[[20, np.nan, 600], [70, 80, 90]], [[20, 30, 600], [70, 80, np.nan]]]
    )
    trial = read_trc(
        _write(tmp_path / "missing.trc", points), rotation_to_model=np.eye(3)
    )
    np.testing.assert_allclose(trial.positions, points, equal_nan=True)
    assert np.isnan(trial.in_model_frame()[0, 0]).all()
    np.testing.assert_allclose(trial.in_model_frame()[0, 1], [0.07, 0.08, 0.09])


@pytest.mark.parametrize(
    "old,new,message",
    [
        ("3\t2\tmm", "4\t2\tmm", "NumFrames"),
        ("3\t2\tmm", "3\t2\tum", "Units"),
        ("KNEE\t\t", "HIP\t\t", "unique"),
        ("10.01000000", "10.01500000", "DataRate"),
        ("202\t", "203\t", "consecutive"),
        ("X1\tY1\tZ1", "Y1\tX1\tZ1", "coordinate labels"),
        ("20.0\t30.0\t600.0", "20.0\t600.0", "columns"),
        ("600.0", "inf", "infinity"),
    ],
)
def test_malformed_or_inconsistent_trials_fail(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = _write(tmp_path / "invalid.trc")
    original = path.read_text()
    assert old in original
    path.write_text(original.replace(old, new, 1))
    with pytest.raises(ValueError, match=message):
        read_trc(path, rotation_to_model=np.eye(3))


def test_trc_requires_calibrated_proper_rotation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="proper orthonormal"):
        read_trc(
            _write(tmp_path / "reflection.trc"), rotation_to_model=np.diag([-1, 1, 1])
        )


@pytest.mark.parametrize("rate,count", [(60, 3), (60000 / 1001, 12000)])
def test_nimble_six_significant_digit_time_and_rate_exports(
    tmp_path: Path, rate: float, count: int
) -> None:
    points = np.zeros((count, 2, 3))
    path = _write(tmp_path / "nimble_precision.trc", points, rate=rate)
    lines = path.read_text().splitlines()
    header = lines[2].split("\t")
    header[0] = header[1] = header[5] = format(rate, ".6g")
    lines[2] = "\t".join(header)
    for index in range(count):
        cells = lines[6 + index].split("\t")
        cells[1] = format(10 + index / rate, ".6g")
        lines[6 + index] = "\t".join(cells)
    path.write_text("\n".join(lines) + "\n")
    trial = read_trc(path, rotation_to_model=np.eye(3))
    assert trial.frequency_hz == float(format(rate, ".6g"))
    assert (
        trial.metadata()["trc"]["timestamp_tolerance_s_max"]
        <= 0.25 / trial.frequency_hz
    )


def test_trc_markers_retarget_against_constructed_subject_and_export(
    tmp_path: Path,
) -> None:
    payload = json.loads((FIXTURES / "static_trial.json").read_text())
    static = read_trc(
        _write(
            tmp_path / "static.trc",
            np.array(payload["trial"]["positions"]),
            tuple(payload["trial"]["labels"]),
        ),
        rotation_to_model=np.eye(3),
    )
    result = estimate_marker_scales(
        static, [SegmentMeasurement(**row) for row in payload["measurements"]]
    )
    subject = build_subject_model(
        mujoco.MjSpec.from_file(str(FIXTURES / "two_link.xml")),
        SubjectModelConfig(result.estimate, ground=GroundAlignment("root", ("sole",))),
    )
    payload = json.loads((FIXTURES / "motion_trial.json").read_text())["trial"]
    motion = read_trc(
        _write(
            tmp_path / "motion.trc",
            np.array(payload["positions"]),
            tuple(payload["labels"]),
            payload["frequency_hz"],
            payload["units"],
        ),
        rotation_to_model=np.eye(3),
    )
    fitted = retarget_markers(
        subject,
        motion,
        MARKER_SITES,
        free_joint="root",
        sole_geoms=("sole",),
        scale_evidence=result.evidence,
    )
    assert fitted.diagnostics["marker_max_m"] < 1e-5
    output = fitted.save(tmp_path / "retargeted.npz")
    provenance = json.loads(output.with_suffix(".json").read_text())["provenance"]
    assert provenance["trial"]["trc"]["sha256"] == motion.metadata()["trc"]["sha256"]
