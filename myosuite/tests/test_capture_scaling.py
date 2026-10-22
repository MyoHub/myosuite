# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Scale estimates from known labelled dimensions and calibrated capture frames."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from myosuite.integrations.capture.example import FIXTURES
from myosuite.integrations.capture.markers import (
    MarkerTrial,
    SegmentMeasurement,
    estimate_marker_scales,
    read_c3d,
    virtual_marker_trial,
)

pytestmark = pytest.mark.tier1


def _fixture(
    name: str = "static_trial.json",
) -> tuple[MarkerTrial, list[SegmentMeasurement]]:
    payload = json.loads((FIXTURES / name).read_text())
    return MarkerTrial(**payload["trial"]), [
        SegmentMeasurement(**row) for row in payload["measurements"]
    ]


def test_labelled_static_trial_estimates_known_local_axis_factors() -> None:
    trial, mapping = _fixture()
    result = estimate_marker_scales(trial, mapping, mass_kg=72)
    np.testing.assert_allclose(result.estimate.factors["thigh"], (1, 1, 1.2))
    np.testing.assert_allclose(result.estimate.factors["shank"], (1, 1, 0.9))
    np.testing.assert_allclose(result.estimate.factors["foot"], (1, 1, 1))
    assert result.estimate.mass_kg == 72
    assert result.estimate.confidence == 0.98
    assert result.evidence["trial"]["units"] == "mm"
    assert result.evidence["trial"]["uncertainty_m"] == 0.002
    assert result.evidence["measurements"][0]["length_m"] == pytest.approx(0.6)
    assert result.evidence["scale_axes"].startswith("target body/segment local")
    assert any("unmeasured axes" in warning for warning in result.estimate.warnings)
    json.dumps(result.evidence, allow_nan=False)


def test_missing_marker_omits_only_affected_scales_with_evidence() -> None:
    trial, mapping = _fixture("missing_marker.json")
    result = estimate_marker_scales(trial, mapping)
    assert set(result.estimate.factors) == {"foot"}
    assert sum("missing marker" in warning for warning in result.estimate.warnings) == 2
    assert [row["status"] for row in result.evidence["measurements"]] == [
        "missing_label",
        "missing_label",
        "estimated",
    ]


@pytest.mark.parametrize(
    "fixture,error",
    [
        ("wrong_units.json", "check units"),
        ("wrong_frame.json", "frame/directed-marker"),
    ],
)
def test_wrong_unit_or_frame_fixture_is_rejected(fixture: str, error: str) -> None:
    trial, mapping = _fixture(fixture)
    with pytest.raises(ValueError, match=error):
        estimate_marker_scales(trial, mapping)


def test_calibrated_rotation_and_units_produce_identical_scales() -> None:
    trial, mapping = _fixture()
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    model_points = trial.in_model_frame()
    other = replace(
        trial,
        positions=(model_points - [0.3, 0.4, 0.5]) @ rotation * 100,
        units="cm",
        rotation_to_model=rotation,
        translation_to_model_m=(0.3, 0.4, 0.5),
    )
    a, b = (
        estimate_marker_scales(trial, mapping),
        estimate_marker_scales(other, mapping),
    )
    for segment in a.estimate.factors:
        np.testing.assert_allclose(
            a.estimate.factors[segment], b.estimate.factors[segment]
        )


def test_missing_samples_are_excluded_without_inventing_confidence() -> None:
    trial, mapping = _fixture()
    points = np.array(trial.positions)
    points[0, trial.labels.index("KNEE")] = np.nan
    result = estimate_marker_scales(replace(trial, positions=points), mapping)
    assert result.evidence["measurements"][0]["valid_fraction"] == pytest.approx(2 / 3)
    assert result.estimate.confidence == trial.confidence
    np.testing.assert_allclose(result.estimate.factors["thigh"], [1, 1, 1.2])
    points[1, trial.labels.index("KNEE")] = np.nan
    result = estimate_marker_scales(replace(trial, positions=points), mapping)
    assert set(result.estimate.factors) == {"foot"}
    with pytest.raises(ValueError, match="No segment"):
        estimate_marker_scales(replace(trial, positions=points), mapping[:2])


def test_isotropic_assumption_and_overlapping_measurements_are_explicit() -> None:
    trial, mapping = _fixture()
    result = estimate_marker_scales(trial, [replace(mapping[0], axis=None)])
    np.testing.assert_allclose(result.estimate.factors["thigh"], [1.2] * 3)
    assert "cross sections were not measured" in result.estimate.warnings[0]
    with pytest.raises(ValueError, match="Overlapping"):
        estimate_marker_scales(trial, [mapping[0], replace(mapping[0], axis=None)])


def test_virtual_marker_definitions_are_user_provided_and_provenance_is_retained() -> (
    None
):
    trial, mapping = _fixture()
    virtual = virtual_marker_trial(
        labels=trial.labels,
        positions=trial.positions,
        units="mm",
        rotation_to_model=np.eye(3),
        definition_source="independently authored fixture markers",
        definition_license="Apache-2.0",
    )
    result = estimate_marker_scales(virtual, mapping)
    assert result.evidence["trial"]["definition_license"] == "Apache-2.0"
    np.testing.assert_allclose(result.estimate.factors["thigh"], [1, 1, 1.2])
    with pytest.raises(ValueError, match="source and licence"):
        virtual_marker_trial(
            labels=trial.labels,
            positions=trial.positions,
            units="mm",
            rotation_to_model=np.eye(3),
            definition_source="",
            definition_license="",
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"rotation_to_model": np.diag([-1, 1, 1])},
        {"rotation_to_model": np.eye(3) * 1000},
        {"units": "unknown"},
        {"confidence": np.nan},
        {"uncertainty_m": -1},
        {"frequency_hz": 0},
    ],
)
def test_trial_refuses_invalid_conventions(updates: dict[str, object]) -> None:
    trial, _ = _fixture()
    with pytest.raises(ValueError):
        replace(trial, **updates)


def test_real_c3d_roundtrip_reads_units_labels_and_missing_residuals(
    tmp_path: Path,
) -> None:
    ezc3d = pytest.importorskip("ezc3d")
    trial, mapping = _fixture()
    capture = ezc3d.c3d()
    capture["parameters"]["POINT"]["RATE"]["value"] = [100]
    capture["parameters"]["POINT"]["LABELS"]["value"] = list(trial.labels)
    capture["parameters"]["POINT"]["UNITS"]["value"] = ["mm"]
    points = np.ones((4, len(trial.labels), 3))
    points[:3] = np.asarray(trial.positions).transpose(2, 1, 0)
    capture["data"]["points"] = points
    residuals = np.zeros((1, len(trial.labels), 3))
    residuals[0, trial.labels.index("KNEE"), 0] = -1
    capture["data"]["meta_points"]["residuals"] = residuals
    path = tmp_path / "authored.c3d"
    capture.write(str(path))
    loaded = read_c3d(path, rotation_to_model=np.eye(3))
    assert loaded.units == "mm"
    assert loaded.labels == trial.labels
    assert loaded.frequency_hz == 100
    assert np.all(np.isnan(np.asarray(loaded.positions)[0, trial.labels.index("KNEE")]))
    result = estimate_marker_scales(loaded, mapping)
    np.testing.assert_allclose(result.estimate.factors["thigh"], [1, 1, 1.2])
