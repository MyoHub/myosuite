# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Synthetic BoneHub annotations verify units, frame and revision provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from myosuite.integrations.bonehub import (
    DATASET_ID,
    PINNED_REVISION,
    VSD_DATASET_ID,
    VSD_PINNED_REVISION,
    BoneHubReference,
    BoneHubVsdReference,
)
from myosuite.integrations.capture import SegmentMeasurement, estimate_marker_scales

pytestmark = pytest.mark.tier1


def _markups(tmp_path: Path, *, frame: str = "LPS", units: Any = "mm") -> Path:
    points = np.array([[20, 30, 600], [20, 30, 0]], dtype=float)
    if frame == "RAS":
        points *= (-1, -1, 1)
    payload = {
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": frame,
                "coordinateUnits": units,
                "controlPoints": [
                    {"label": label, "position": position.tolist()}
                    for label, position in zip(("HIP", "KNEE"), points)
                ],
            }
        ],
    }
    path = tmp_path / "synthetic.mrk.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _read(reference: BoneHubReference, path: Path, **kwargs: Any) -> Any:
    options = {
        "units": "mm",
        "rotation_to_model": np.eye(3),
        "annotation_source": "Independently authored synthetic test landmarks",
        "definition_license": "Apache-2.0",
    }
    return reference.read_markups(path, **(options | kwargs))


@pytest.mark.parametrize("frame", ["LPS", "RAS"])
def test_slicer_landmarks_feed_existing_scale_estimator(
    tmp_path: Path, frame: str
) -> None:
    reference = BoneHubReference(tmp_path / "reference", "01_Male")
    path = _markups(tmp_path, frame=frame)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    trial = _read(
        reference, path, rotation_to_model=rotation, translation_to_model_m=(1, 2, 3)
    )
    np.testing.assert_allclose(
        trial.in_model_frame(), [[[0.97, 2.02, 3.6], [0.97, 2.02, 3]]]
    )
    result = estimate_marker_scales(
        trial,
        [
            SegmentMeasurement(
                "femur_r",
                ("HIP", "KNEE"),
                0.5,
                axis="y",
                expected_direction_model=(0, 0, -1),
            )
        ],
    )
    np.testing.assert_allclose(result.estimate.factors["femur_r"], [1, 1.2, 1])
    evidence = result.evidence["trial"]["bonehub"]
    assert evidence["subject"] == "01_Male"
    assert evidence["revision"] == PINNED_REVISION
    assert evidence["geometry_license"] == "CC-BY-4.0"
    assert evidence["annotation_license"] == "Apache-2.0"
    assert (
        evidence["annotation_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert evidence["annotation_coordinate_systems"] == [frame]
    assert "Andreassen" in evidence["attribution"]
    assert result.estimate.mass_kg is None
    assert result.estimate.height_m is None
    assert result.evidence["trial"]["definition_license"] == "Apache-2.0"
    json.dumps(result.evidence, allow_nan=False)


def test_units_codes_and_missing_points_are_explicit(tmp_path: Path) -> None:
    ref = BoneHubReference(tmp_path, "02_Female")
    path = _markups(tmp_path, units=["mm", "UCUM", "millimeter"])
    payload = json.loads(path.read_text())
    payload["markups"][0]["controlPoints"][1] = {
        "label": "KNEE",
        "positionStatus": "undefined",
    }
    path.write_text(json.dumps(payload))
    trial = _read(ref, path)
    assert np.isnan(trial.in_model_frame()[0, 1]).all()
    with pytest.raises(ValueError, match="No segment"):
        estimate_marker_scales(
            trial,
            [
                SegmentMeasurement(
                    "femur_r",
                    ("HIP", "KNEE"),
                    0.5,
                    axis="y",
                )
            ],
        )
    with pytest.raises(ValueError, match="coordinateUnits"):
        _read(ref, path, units="m")


@pytest.mark.parametrize(
    "change,error",
    [
        ({"coordinateSystem": None}, "LPS or RAS"),
        ({"coordinateSystem": "XYZ"}, "LPS or RAS"),
        ({"coordinateUnits": "um"}, "coordinateUnits"),
        ({"type": "Curve"}, "Fiducial"),
        ({"controlPoints": [{"label": "HIP", "position": [1, 2]}]}, "finite"),
        (
            {"controlPoints": [{"label": "HIP", "position": [1, 2, float("inf")]}]},
            "finite",
        ),
        (
            {"controlPoints": [{"label": "HIP", "positionStatus": "invalid"}]},
            "positionStatus",
        ),
    ],
)
def test_invalid_annotation_frames_and_values_fail(
    tmp_path: Path, change: dict[str, Any], error: str
) -> None:
    path = _markups(tmp_path)
    payload = json.loads(path.read_text())
    payload["markups"][0].update(change)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=error):
        _read(BoneHubReference(tmp_path, "01_Male"), path)


def test_annotation_license_rotation_and_duplicate_labels_are_checked(
    tmp_path: Path,
) -> None:
    ref = BoneHubReference(tmp_path, "01_Male")
    path = _markups(tmp_path)
    for kwargs, message in [
        ({"definition_license": ""}, "definition_license"),
        ({"annotation_source": ""}, "annotation_source"),
        ({"rotation_to_model": np.diag([-1, 1, 1])}, "proper orthonormal"),
    ]:
        with pytest.raises(ValueError, match=message):
            _read(ref, path, **kwargs)
    payload = json.loads(path.read_text())
    payload["markups"][0]["controlPoints"][1]["label"] = "HIP"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unique"):
        _read(ref, path)


def test_selective_download_is_pinned_and_keeps_subjects_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def download(**kwargs: Any) -> str:
        calls.append(kwargs)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"01_Male": {}, "02_Female": {}})
            if path.suffix == ".json"
            else "synthetic mesh placeholder"
        )
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    ref = BoneHubReference(tmp_path, "01_Male")
    path = ref.download_mesh("LOWER_EXTREMITY_RIGHT", "FEMUR_RIGHT")
    assert path == ref.mesh_path("LOWER_EXTREMITY_RIGHT", "FEMUR_RIGHT")
    assert len(calls) == 2
    assert all(call["revision"] == PINNED_REVISION for call in calls)
    assert all(call["repo_id"] == DATASET_ID for call in calls)
    assert calls[0]["filename"] == "visible_human_3d_models/CT/metadata.json"
    assert calls[1]["filename"].endswith(
        "01_Male/LOWER_EXTREMITY_RIGHT/FEMUR_RIGHT.stl"
    )
    with pytest.raises(FileNotFoundError):
        BoneHubReference(tmp_path, "02_Female").mesh_path(
            "LOWER_EXTREMITY_RIGHT", "FEMUR_RIGHT"
        )
    with pytest.raises(FileNotFoundError):
        BoneHubReference(tmp_path, "01_Male", "a" * 40).mesh_path(
            "LOWER_EXTREMITY_RIGHT", "FEMUR_RIGHT"
        )


@pytest.mark.parametrize(
    "anatomy,bone",
    [
        ("../SKULL", "BONE"),
        ("SKULL", "../BONE"),
        ("SKULL", "BÖNE"),
        ("HAND_RIGHT", "METACARPAL_1_LEFT"),
    ],
)
def test_mesh_identifiers_cannot_escape_or_mix_sides(
    tmp_path: Path, anatomy: str, bone: str
) -> None:
    with pytest.raises(ValueError):
        BoneHubReference(tmp_path, "01_Male").mesh_path(anatomy, bone)


def test_reference_subject_and_revision_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="subject"):
        BoneHubReference(tmp_path, "other")
    with pytest.raises(ValueError, match="commit SHA"):
        BoneHubReference(tmp_path, "01_Male", "main")


def _vsd_metadata(reference: BoneHubVsdReference, source_subject: str) -> None:
    path = reference.snapshot_root / "metadata" / f"{reference.subject}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source_subject": source_subject,
                "subjectSnapshot": {"heightInMeters": 1.62, "weightInKilograms": 75.0},
            }
        ),
        encoding="utf-8",
    )


def test_vsd_quality_license_and_original_identity_survive_scaling(
    tmp_path: Path,
) -> None:
    reference = BoneHubVsdReference(tmp_path / "reference", "028")
    _vsd_metadata(reference, "z063")
    trial = _read(reference, _markups(tmp_path))
    result = estimate_marker_scales(
        trial, [SegmentMeasurement("femur_r", ("HIP", "KNEE"), 0.5, axis="y")]
    )
    evidence = result.evidence["trial"]["bonehub"]
    assert evidence["dataset"] == VSD_DATASET_ID
    assert evidence["subject"] == "028"
    assert evidence["source_subject"] == "z063"
    assert evidence["geometry_license"] == "CC-BY-NC-SA-4.0"
    assert evidence["annotation_license"] == "Apache-2.0"
    assert not evidence["source_in_complete_cohort"]
    assert "hip replacements" in evidence["quality_flags"][0]
    assert evidence["source_measurements"]["mass_kg"] == 75.0
    assert result.estimate.mass_kg is None
    assert result.estimate.height_m is None
    assert len(evidence["source_metadata_sha256"]) == 64
    json.dumps(result.evidence, allow_nan=False)


def test_vsd_requires_matching_source_metadata_before_annotations(
    tmp_path: Path,
) -> None:
    reference = BoneHubVsdReference(tmp_path, "001")
    annotation = _markups(tmp_path)
    with pytest.raises(FileNotFoundError, match="metadata first"):
        _read(reference, annotation)
    _vsd_metadata(reference, "z063")
    with pytest.raises(ValueError, match="source_subject"):
        _read(reference, annotation)
    _vsd_metadata(reference, "002")
    assert reference.metadata()["source_in_complete_cohort"]
    assert not reference.metadata()["quality_flags"]


def test_vsd_downloads_one_registered_bone_and_retains_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def download(**kwargs: Any) -> str:
        calls.append(kwargs)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.parent.name == "metadata":
            payload = {"source_subject": "002", "subjectSnapshot": {}}
        elif path.name == "dataset.json":
            payload = {"labels": {"background": 0, "Patella_R": 7}}
        else:
            path.write_bytes(b"synthetic mesh placeholder")
            return str(path)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    reference = BoneHubVsdReference(tmp_path, "001")
    mesh = reference.download_mesh("Patella_R")
    assert mesh == reference.mesh_path("Patella_R")
    assert [call["filename"] for call in calls] == [
        "metadata/001.json",
        "lower_extremity_ct/dataset.json",
        "meshes/001/Patella_R.stl",
    ]
    assert all(call["repo_id"] == VSD_DATASET_ID for call in calls)
    assert all(call["revision"] == VSD_PINNED_REVISION for call in calls)
    with pytest.raises(ValueError, match="Unknown VSD bone"):
        reference.download_mesh("Invented_R")
    assert not any(call["filename"].endswith("Invented_R.stl") for call in calls)
    with pytest.raises(FileNotFoundError):
        BoneHubVsdReference(tmp_path, "002").mesh_path("Patella_R")
    assert (
        reference.snapshot_root != BoneHubReference(tmp_path, "01_Male").snapshot_root
    )


@pytest.mark.parametrize("subject", ["000", "031", "1", "../001", 1])
def test_vsd_case_identifiers_are_explicit(tmp_path: Path, subject: Any) -> None:
    with pytest.raises(ValueError, match="VSD case"):
        BoneHubVsdReference(tmp_path, subject)


def test_vsd_mesh_paths_reject_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ASCII dataset label"):
        BoneHubVsdReference(tmp_path, "001").mesh_path("../Patella_R")
