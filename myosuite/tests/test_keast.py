# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Synthetic named anatomy coordinates and pinned-cache validation."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import mujoco

from myosuite.core.subject_model import SubjectModelConfig, build_subject_model
from myosuite.integrations import keast
from myosuite.integrations.capture.markers import MarkerTrial
from myosuite.integrations.keast import (
    KeastReference,
    read_named_landmarks,
    resample_tibial_borders,
    tibial_landmark_frame,
)

pytestmark = pytest.mark.tier1


def _trial() -> MarkerTrial:
    return MarkerTrial(
        ("LC", "MC", "LM", "MM", "TT", "FH"),
        [
            [
                [0, 1, 0.1],
                [0, 1, -0.1],
                [0, 0, 0.05],
                [0, 0, -0.05],
                [0.1, 0.8, 0],
                [0, 0.9, 0.15],
            ]
        ],
        "m",
        np.eye(3),
        "independently authored test anatomy",
    )


def _read(path: Path, **kwargs: object) -> MarkerTrial:
    options = {
        "units": "mm",
        "rotation_to_model": np.eye(3),
        "source": "authored synthetic coordinates",
        "definition_source": "independently authored protocol",
        "definition_license": "Apache-2.0",
    }
    return read_named_landmarks(path, **(options | kwargs))


def test_named_landmarks_preserve_missing_points_frame_and_file_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "markers.csv"
    raw = b"A,10,20,30\nB,40,,60\nC,70,80,90\n"
    path.write_bytes(raw)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    trial = _read(path, rotation_to_model=rotation, translation_to_model_m=(1, 2, 3))
    np.testing.assert_allclose(trial.in_model_frame()[0, 0], [0.98, 2.01, 3.03])
    np.testing.assert_allclose(trial.in_model_frame()[0, 2], [0.92, 2.07, 3.09])
    assert np.isnan(trial.in_model_frame()[0, 1]).all()
    assert trial.metadata()["landmark_csv"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert trial.uncertainty_m is None
    json.dumps(trial.metadata(), allow_nan=False)


@pytest.mark.parametrize(
    "text,match",
    [
        ("A,1,2\n", "label,x,y,z"),
        ("A,1,two,3\n", "numeric"),
        ("A,1,2,3\nA,4,5,6\n", "unique"),
        ("A,1,inf,3\n", "infinity"),
    ],
)
def test_bad_named_landmarks_fail(tmp_path: Path, text: str, match: str) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        _read(path)


def test_tibial_frame_is_proper_and_equivariant_to_calibration() -> None:
    trial = _trial()
    origin, basis = tibial_landmark_frame(trial)
    np.testing.assert_allclose(origin, [0, 1, 0])
    np.testing.assert_allclose(basis, np.eye(3))
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    transformed = replace(
        trial, rotation_to_model=rotation, translation_to_model_m=(2, 3, 4)
    )
    other_origin, other_basis = tibial_landmark_frame(transformed)
    np.testing.assert_allclose(
        (transformed.in_model_frame()[0] - other_origin) @ other_basis,
        (trial.in_model_frame()[0] - origin) @ basis,
    )
    assert np.linalg.det(other_basis) == pytest.approx(1)


def test_tibial_frame_refuses_missing_or_degenerate_anatomy() -> None:
    trial = _trial()
    points = np.array(trial.positions)
    points[0, trial.labels.index("TT")] = [0, 0.8, 0]
    with pytest.raises(ValueError, match="anterior"):
        tibial_landmark_frame(replace(trial, positions=points))
    points[0, trial.labels.index("TT")] = np.nan
    with pytest.raises(ValueError, match="finite"):
        tibial_landmark_frame(replace(trial, positions=points))
    with pytest.raises(ValueError, match="LC, MC"):
        tibial_landmark_frame(
            replace(trial, labels=trial.labels[:-1], positions=trial.positions[:, :-1])
        )


def _borders(t_heights: list[float], f_heights: list[float]) -> MarkerTrial:
    trial = _trial()
    names = tuple(f"T{i + 1}" for i in range(len(t_heights))) + tuple(
        f"F{i + 1}" for i in range(len(f_heights))
    )
    points = [[0.1 * y + 0.02, y, 0] for y in t_heights] + [
        [0, y, 0.2 + 0.02 * y] for y in f_heights
    ]
    return replace(
        trial,
        labels=trial.labels + names,
        positions=np.concatenate((trial.positions, np.array(points)[None]), axis=1),
    )


def test_border_fractions_match_when_digitization_counts_and_locations_differ() -> None:
    sparse = _borders([0.9, 0.6, 0.3, 0.05], [0.85, 0.6, 0.4, 0.1])
    dense = _borders([0.85, 0.8, 0.7, 0.5, 0.3, 0.1], [0.8, 0.75, 0.5, 0.3, 0.2, 0.1])
    a, b = resample_tibial_borders(sparse), resample_tibial_borders(dense)
    assert set(a) == set(b) == {"T0.25", "T0.5", "T0.75", "F0.25", "F0.5", "F0.75"}
    for label in a:
        np.testing.assert_allclose(a[label], b[label])
    np.testing.assert_allclose(a["T0.25"], [0.095, 0.75, 0])
    np.testing.assert_allclose(a["F0.25"], [0, 0.675, 0.2135])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    other = resample_tibial_borders(
        replace(sparse, rotation_to_model=rotation, translation_to_model_m=(1, 2, 3))
    )
    for label in a:
        np.testing.assert_allclose(other[label], a[label] @ rotation.T + [1, 2, 3])


def test_border_fraction_extrapolation_and_unordered_points_are_refused() -> None:
    trial = _borders([0.9, 0.6, 0.3, 0.05], [0.85, 0.6, 0.4, 0.1])
    with pytest.raises(ValueError, match="outside"):
        resample_tibial_borders(trial, (0.99,))
    with pytest.raises(ValueError, match="unique"):
        resample_tibial_borders(trial, (0.5, 0.5))
    points = np.array(trial.positions)
    points[0, trial.labels.index("T2"), 1] = 0.95
    with pytest.raises(ValueError, match="ordered"):
        resample_tibial_borders(replace(trial, positions=points))


def test_pinned_archive_extracts_only_named_files_and_rejects_modified_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = "Examples/case3/samples/C01RTF.csv"
    data = b"LC,0,1000,100\nMC,0,1000,-100\nLM,0,0,50\nMM,0,0,-50\nTT,100,800,0\nFH,0,900,150\n"
    archive = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr(relative, data)
        source.writestr("../never-extract.txt", "not an allowed member")
    manifest = {
        **keast._MANIFEST,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": {relative: hashlib.sha256(data).hexdigest()},
    }
    monkeypatch.setattr(keast, "_MANIFEST", manifest)
    reference = KeastReference(tmp_path / "cache")
    reference.prepare(archive)
    assert not (tmp_path / "never-extract.txt").exists()
    trial = reference.read_landmarks("C01RTF", units="mm", rotation_to_model=np.eye(3))
    assert trial.metadata()["landmark_csv"]["case"] == "C01RTF"
    assert "independent 35" in trial.metadata()["landmark_csv"]["cohort"]
    assert trial.metadata()["definition_license"].startswith("CC-BY-4.0")
    (reference.data_root / relative).write_bytes(data.replace(b"1000", b"1001"))
    with pytest.raises(ValueError, match="checksum"):
        reference.read_landmarks("C01RTF", units="mm", rotation_to_model=np.eye(3))
    with pytest.raises(ValueError, match="changed"):
        reference.prepare(archive)
    archive.write_bytes(b"bad archive")
    with pytest.raises(ValueError, match="checksum"):
        KeastReference(tmp_path / "other").prepare(archive)


@pytest.mark.parametrize("case", ["C00RTF", "C36RTF", "../C01RTF", "1"])
def test_pinned_cases_and_units_are_explicit(tmp_path: Path, case: str) -> None:
    reference = KeastReference(tmp_path)
    with pytest.raises(ValueError, match="case must"):
        reference.read_landmarks(case, units="mm", rotation_to_model=np.eye(3))
    with pytest.raises(ValueError, match="in mm"):
        reference.read_landmarks("mean", units="m", rotation_to_model=np.eye(3))


def test_case_adapter_fits_target_local_factors_and_builds_subject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = KeastReference(tmp_path)
    target = _trial()
    factors = np.array([1.1, 0.9, 1.05])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    points = target.in_model_frame()[0] * factors @ rotation.T + [2, 3, 4]
    relative = "Examples/case3/samples/C01RTF.csv"
    path = reference.data_root / relative
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            label + "," + ",".join(str(value) for value in point * 1000)
            for label, point in zip(target.labels, points)
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        keast,
        "_MANIFEST",
        {
            **keast._MANIFEST,
            "files": {relative: hashlib.sha256(path.read_bytes()).hexdigest()},
        },
    )
    result = reference.estimate_scales(
        "C01RTF",
        target,
        segment="tibia_r",
        units="mm",
        rotation_to_model=np.eye(3),
        mode="anisotropic",
        max_rms_m=1e-5,
    )
    np.testing.assert_allclose(result.estimate.factors["tibia_r"], factors, atol=1e-7)
    assert (
        result.evidence["observed_trial"]["landmark_csv"]["sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert result.evidence["reference_trial"]["source"] == target.source
    spec = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><body name="tibia_r"><geom name="bone" type="box" size=".02 .5 .03" mass="1"/></body></worldbody></mujoco>'
    )
    subject = build_subject_model(spec, SubjectModelConfig(result.estimate))
    np.testing.assert_allclose(subject.model.geom("bone").size, [0.022, 0.45, 0.0315])
