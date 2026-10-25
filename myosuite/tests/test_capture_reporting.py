# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Synthetic quality decisions and aggregate-export disclosure boundaries."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from myosuite.integrations.capture.reporting import (
    QualityThresholds,
    SourceAttribution,
    aggregate_validation_report,
    evaluate_quality,
    write_aggregate_report,
)
from myosuite.integrations.capture.retargeting import MarkerRetargetResult

pytestmark = pytest.mark.tier1


def test_requested_limits_and_failed_frame_count_survive_aggregation(
    tmp_path: Path,
) -> None:
    diagnostics = {
        "marker_rms_m": np.float64(0.002),
        "marker_max_m": 0.005,
        "sole_penetration_max_m": 0.0,
        "root_tracking_rms_m": 0.003,
        "equality_error_max": 1e-8,
        "frames": 10,
        "converged_frames": 9,
    }
    thresholds = QualityThresholds(
        marker_rms_m=0.003,
        marker_max_m=0.01,
        sole_penetration_max_m=0.0,
        root_tracking_rms_m=0.004,
        equality_error_max=1e-7,
        min_converged_fraction=0.9,
    )
    report = aggregate_validation_report(diagnostics, thresholds=thresholds)
    assert report["quality"]["status"] == "pass"
    assert report["quality"]["accepted"] is True
    assert report["metrics"]["failed_frames"] == 1
    assert report["metrics"]["converged_fraction"] == 0.9
    path = write_aggregate_report(report, tmp_path / "nested" / "aggregate.json")
    assert json.loads(path.read_text()) == report
    failed = evaluate_quality(diagnostics, QualityThresholds(min_converged_fraction=1))
    assert failed["status"] == "fail"
    assert failed["checks"]["converged_fraction"]["observed"] == 0.9


def test_missing_metrics_are_unknown_including_fixed_base_root() -> None:
    result = evaluate_quality(
        {"marker_rms_m": 0.001},
        QualityThresholds(marker_rms_m=0.01, root_tracking_rms_m=0.01),
    )
    assert result["status"] == "unknown" and result["accepted"] is False
    assert result["checks"]["marker_rms_m"]["status"] == "pass"
    assert result["checks"]["root_tracking_rms_m"] == {
        "status": "unknown",
        "observed": None,
        "threshold": 0.01,
        "comparison": "at_most",
        "reason": "missing_metric",
    }
    assert evaluate_quality({}, QualityThresholds())["status"] == "unknown"
    assert not evaluate_quality({}, QualityThresholds())["accepted"]
    no_convergence = evaluate_quality({}, QualityThresholds(min_converged_fraction=0))
    assert no_convergence["checks"]["converged_fraction"]["status"] == "unknown"


@pytest.mark.parametrize(
    "bad", [-0.001, np.nan, np.inf, -np.inf, True, "private", [0.0], {"x": 0}]
)
def test_invalid_errors_fail_without_serializing_input(
    bad: object, tmp_path: Path
) -> None:
    report = aggregate_validation_report(
        {"sole_penetration_max_m": bad},
        thresholds=QualityThresholds(sole_penetration_max_m=0.001),
    )
    assert report["quality"]["status"] == "fail"
    assert (
        report["quality"]["checks"]["sole_penetration_max_m"]["reason"]
        == "invalid_metric"
    )
    assert report["metrics"] == {}
    assert "private" not in json.dumps(report)
    assert write_aggregate_report(report, tmp_path / "failed.json").exists()


@pytest.mark.parametrize(
    "values,invalid",
    [
        ({"frames": 0, "converged_frames": 0}, "frames"),
        ({"frames": 10, "converged_frames": 11}, "converged_frames"),
        ({"frames": 10, "converged_frames": -1}, "converged_frames"),
        ({"frames": 10, "converged_frames": 2.5}, "converged_frames"),
        ({"frames": 10, "converged_frames": 9, "failed_frames": 0}, "failed_frames"),
        (
            {"frames": 10, "converged_frames": 9, "converged_fraction": 1},
            "converged_fraction",
        ),
        ({"converged_fraction": 1.01}, "converged_fraction"),
    ],
)
def test_invalid_counts_fail_without_clipping(
    values: dict[str, object], invalid: str, tmp_path: Path
) -> None:
    report = aggregate_validation_report(
        values, thresholds=QualityThresholds(min_converged_fraction=0.5)
    )
    assert report["quality"]["status"] == "fail"
    assert invalid in report["quality"]["invalid_metrics"]
    assert invalid not in report["metrics"]
    write_aggregate_report(report, tmp_path / "invalid-count.json")


@pytest.mark.parametrize("value", [-1, np.nan, np.inf, True, "0.01"])
def test_thresholds_require_finite_nonnegative_numbers(value: object) -> None:
    with pytest.raises(ValueError):
        QualityThresholds(marker_rms_m=value)
    with pytest.raises(ValueError):
        QualityThresholds(min_converged_fraction=1.01)


def test_only_allowed_scalar_metrics_are_copied_from_result(tmp_path: Path) -> None:
    secret = "C:\\private\\subject-secret.csv"
    diagnostics = {
        "marker_rms_m": 0.001,
        "marker_errors_m": np.zeros((2, 4)),
        "qpos": np.ones((2, 7)),
        "rotation_reference_to_observed": np.eye(3),
        "translation_m": [1, 2, 3],
        "residuals_m": [[0, 0, 0]],
        "factors": [1.1, 0.9, 1.0],
        "height_m": 1.7,
        "mass_kg": 70.0,
        "subject_id": secret,
        "source_metadata": {"deep": {"positions": [1, 2, 3], "path": secret}},
        "site_xpos": secret,
        "contact_labels": np.ones((2, 2)),
    }
    result = MarkerRetargetResult(
        clip=None, diagnostics=diagnostics, provenance={"source": secret}
    )
    report = aggregate_validation_report(
        result, thresholds=QualityThresholds(marker_rms_m=0.01)
    )
    assert report["metrics"] == {"marker_rms_m": 0.001}
    assert report["sources"] == []
    path = write_aggregate_report(report, tmp_path / "summary.json")
    serialized = path.read_text()
    assert "subject-secret" not in serialized and "source_metadata" not in serialized
    assert "qpos" not in serialized and "site_xpos" not in serialized


def test_explicit_attribution_is_validated_and_copied(tmp_path: Path) -> None:
    hashes = {"model_sha256": "a" * 64}
    source = SourceAttribution(
        source_id="skel",
        license="LicenseRef-SKEL-Research",
        citations=("10.1038/s41597-022-01905-2",),
        source_urls=("https://example.org/dataset/v1",),
        source_version="v1.0",
        hashes=hashes,
    )
    hashes["model_sha256"] = "b" * 64
    report = aggregate_validation_report(
        {}, thresholds=QualityThresholds(), attributions=(source,)
    )
    assert report["sources"][0]["hashes"]["model_sha256"] == "a" * 64
    assert report["sources"][0]["license"] == "LicenseRef-SKEL-Research"
    assert report["quality"]["status"] == "unknown"
    assert "grant no redistribution rights" in report["scope"][-1]
    path = write_aggregate_report(report, tmp_path / "attributed.json")
    assert json.loads(path.read_text()) == report
    with pytest.raises(TypeError):
        aggregate_validation_report(
            {}, thresholds=QualityThresholds(), attributions=({"source": "secret"},)
        )


@pytest.mark.parametrize(
    "fields",
    [
        {"source_id": "C:\\private\\data"},
        {"source_version": "/home/private/data"},
        {"license": "private source definition at C:/private"},
        {"citations": ("C:/private/data.csv",)},
        {"citations": ("arbitrary secret text",)},
        {"source_urls": ("file:///C:/private/data",)},
        {"source_urls": ("https://localhost/data",)},
        {"source_urls": ("http://127.0.0.1/data",)},
        {"source_urls": ("https://10.1.2.3/data",)},
        {"source_urls": ("https://user:password@example.org/data",)},
        {"source_urls": ("https://example.org/data?token=secret",)},
        {"hashes": {"private_path": "C:/private/data"}},
    ],
)
def test_unapproved_or_local_attribution_values_are_rejected(
    fields: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        SourceAttribution(
            **{"source_id": "synthetic", "license": "CC-BY-4.0", **fields}
        )


@pytest.mark.parametrize(
    "location", ["top", "metrics", "quality", "source", "scope", "decision"]
)
def test_writer_rejects_modified_or_injected_report_before_creating_file(
    location: str, tmp_path: Path
) -> None:
    report = aggregate_validation_report(
        {"marker_rms_m": 0.002},
        thresholds=QualityThresholds(marker_rms_m=0.001),
        attributions=(SourceAttribution("synthetic", "CC-BY-4.0"),),
    )
    injected = copy.deepcopy(report)
    if location == "top":
        injected["coordinates"] = [1, 2, 3]
    elif location == "metrics":
        injected["metrics"]["marker_rms_m"] = "C:/private/subject.csv"
    elif location == "quality":
        injected["quality"]["residuals"] = [[1, 2, 3]]
    elif location == "source":
        injected["sources"][0]["coordinates"] = [1, 2, 3]
    elif location == "scope":
        injected["scope"].append("C:/private/subject.csv")
    else:
        injected["quality"]["accepted"] = True
    target = tmp_path / "not-created" / "aggregate.json"
    with pytest.raises(ValueError):
        write_aggregate_report(injected, target)
    assert not target.parent.exists()


def test_contact_counts_remain_explicit_geometry_heuristics() -> None:
    report = aggregate_validation_report(
        {
            "geometric_contact_pair_count": 20,
            "geometric_contact_frame_count": 8,
            "contact_pair_count": 99,
            "contact_labels": np.ones((10, 2)),
            "fit_jacobian_rank": 9,
            "fit_condition_number": 24.0,
        },
        thresholds=QualityThresholds(),
        protocol_id="synthetic-v1",
    )
    assert report["metrics"] == {
        "geometric_contact_pair_count": 20,
        "geometric_contact_frame_count": 8,
        "fit_jacobian_rank": 9,
        "fit_condition_number": 24.0,
    }
    assert "heuristics" in report["scope"][2]
    assert report["quality"]["accepted"] is False
