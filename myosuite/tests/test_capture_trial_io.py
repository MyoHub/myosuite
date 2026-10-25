# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Synthetic local trial interchange and schema-admission checks."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from myosuite.integrations.capture.markers import MarkerTrial
from myosuite.integrations.capture.trial_io import load_marker_trial, save_marker_trial

pytestmark = pytest.mark.tier1


@dataclass(frozen=True)
class _SyntheticSourceTrial(MarkerTrial):
    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "native_source": {
                "adapter": "independent synthetic reader",
                "sha256": "1" * 64,
                "source_frames": [17, 18],
            },
        }


def _trial() -> MarkerTrial:
    return _SyntheticSourceTrial(
        labels=("surface_a", "surface_b"),
        positions=[
            [[0, 100, 200], [300, np.nan, 500]],
            [[50, 150, 250], [350, 450, 550]],
        ],
        units="mm",
        rotation_to_model=[[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        source="independently authored synthetic coordinates",
        translation_to_model_m=(0.1, -0.2, 0.3),
        confidence=0.8,
        uncertainty_m=0.002,
        frequency_hz=120,
        definition_source="synthetic surface-marker protocol",
        definition_license="Apache-2.0 synthetic example; no source model",
        point_semantics="surface annotations; not joint centres",
        subject_id="synthetic-subject-a",
    )


def test_roundtrip_preserves_raw_convention_missing_points_and_reader_metadata(
    tmp_path: Path,
) -> None:
    original = _trial()
    path = save_marker_trial(original, tmp_path / "nested" / "trial.npz")
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"schema_version", "positions", "metadata_json"}
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
    loaded = load_marker_trial(path)
    assert loaded.metadata() == original.metadata()
    np.testing.assert_equal(loaded.positions, original.positions)
    np.testing.assert_allclose(loaded.in_model_frame(), original.in_model_frame())
    assert not loaded.positions.flags.writeable
    assert not loaded.rotation_to_model.flags.writeable
    assert not loaded.translation_to_model_m.flags.writeable
    metadata = loaded.metadata()
    metadata["native_source"]["sha256"] = "changed"
    assert loaded.metadata() == original.metadata()
    replaced = replace(loaded, frequency_hz=60)
    assert replaced.metadata()["frequency_hz"] == 60
    assert replaced.metadata()["native_source"] == original.metadata()["native_source"]
    assert (
        load_marker_trial(save_marker_trial(loaded, tmp_path / "again.npz")).metadata()
        == original.metadata()
    )


@pytest.mark.parametrize(
    "malformation",
    ["schema", "units", "missing", "subject", "transform", "extra", "pickle"],
)
def test_rejects_unknown_schema_and_malformed_fields(
    tmp_path: Path, malformation: str
) -> None:
    path = save_marker_trial(_trial(), tmp_path / "trial.npz")
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(payload["metadata_json"]))
    if malformation == "schema":
        payload["schema_version"] = np.array(100)
    elif malformation == "units":
        metadata["units"] = "yards"
    elif malformation == "missing":
        del metadata["rotation_to_model"]
    elif malformation == "subject":
        metadata["subject_id"] = ["ambiguous-subjects"]
    elif malformation == "transform":
        metadata["translation_to_model_m"] = ["0", "0", "0"]
    elif malformation == "extra":
        payload["unsupported_field"] = np.array(0)
    else:
        payload["positions"] = np.array([{"untrusted": "object"}], dtype=object)
    payload["metadata_json"] = np.array(json.dumps(metadata))
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError):
        load_marker_trial(path)


def test_rejects_unversioned_numpy_array(tmp_path: Path) -> None:
    path = tmp_path / "coordinates.npy"
    np.save(path, np.zeros((1, 2, 3)))
    with pytest.raises(ValueError, match="versioned NPZ"):
        load_marker_trial(path)


def test_rejects_nonfinite_provenance_without_writing_a_file(tmp_path: Path) -> None:
    @dataclass(frozen=True)
    class InvalidMetadataTrial(MarkerTrial):
        def metadata(self) -> dict[str, Any]:
            return {**super().metadata(), "native_source": {"height_m": float("nan")}}

    source = _trial()
    trial = InvalidMetadataTrial(
        labels=source.labels,
        positions=source.positions,
        units=source.units,
        rotation_to_model=source.rotation_to_model,
        source=source.source,
    )
    path = tmp_path / "invalid.npz"
    with pytest.raises(ValueError):
        save_marker_trial(trial, path)
    assert not path.exists()


@pytest.mark.parametrize(
    "field,value", [("point_semantics", ""), ("subject_id", ""), ("subject_id", 3)]
)
def test_marker_trial_requires_explicit_text_for_new_identity_fields(
    field: str, value: Any
) -> None:
    with pytest.raises(ValueError):
        replace(_trial(), **{field: value})
