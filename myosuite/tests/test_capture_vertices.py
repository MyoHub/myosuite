# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Synthetic vertex geometry through existing scaling and target-specific IK."""

from __future__ import annotations

import json
from dataclasses import replace
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
    VertexMarkerSet,
    estimate_marker_scales,
    vertex_marker_trial,
)
from myosuite.integrations.capture.example import FIXTURES, MARKER_SITES
from myosuite.integrations.capture.retargeting import (
    load_retargeted_clip,
    retarget_markers,
)

pytestmark = pytest.mark.tier1
TOPOLOGY = "MyoSuite independently authored analytic vertices v1"


def _fixture_vertices(
    fixture: str = "static_trial.json",
) -> tuple[np.ndarray, VertexMarkerSet, dict]:
    payload = json.loads((FIXTURES / fixture).read_text())
    labels = payload["trial"]["labels"]
    indices = {label: i * 2 + 1 for i, label in enumerate(labels)}
    points = np.zeros((len(payload["trial"]["positions"]), 2 * len(labels) + 1, 3))
    points[:, list(indices.values())] = payload["trial"]["positions"]
    definitions = VertexMarkerSet(
        TOPOLOGY,
        points.shape[1],
        indices,
        "MyoSuite analytic two-link markers embedded in synthetic vertices",
        "Apache-2.0",
    )
    return points, definitions, payload


def test_vertex_geometry_scales_builds_retargets_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    vertices, definitions, payload = _fixture_vertices()
    trial = vertex_marker_trial(
        vertices[0],
        definitions,
        topology=TOPOLOGY,
        units="mm",
        rotation_to_model=np.eye(3),
        source="independently authored static geometry",
    )
    assert np.asarray(trial.positions).shape == (1, len(definitions.indices), 3)
    result = estimate_marker_scales(
        trial, [SegmentMeasurement(**row) for row in payload["measurements"]]
    )
    np.testing.assert_allclose(result.estimate.factors["thigh"], [1, 1, 1.2])
    np.testing.assert_allclose(result.estimate.factors["shank"], [1, 1, 0.9])
    subject = build_subject_model(
        mujoco.MjSpec.from_file(str(FIXTURES / "two_link.xml")),
        SubjectModelConfig(result.estimate, ground=GroundAlignment("root", ("sole",))),
    )
    motion_vertices, _, motion_payload = _fixture_vertices("motion_trial.json")
    motion = vertex_marker_trial(
        motion_vertices,
        definitions,
        topology=TOPOLOGY,
        units=motion_payload["trial"]["units"],
        rotation_to_model=np.eye(3),
        source="independently authored moving geometry",
        frequency_hz=motion_payload["trial"]["frequency_hz"],
    )
    fitted = retarget_markers(
        subject,
        motion,
        MARKER_SITES,
        free_joint="root",
        sole_geoms=("sole",),
        root_marker="ROOT",
        scale_evidence=result.evidence,
    )
    assert fitted.diagnostics["marker_max_m"] < 1e-5
    assert fitted.diagnostics["root_tracking_rms_m"] < 1e-5
    points = motion.in_model_frame()
    thigh = (
        points[:, motion.labels.index("KNEE")] - points[:, motion.labels.index("ROOT")]
    )
    expected_hip = np.arctan2(-thigh[:, 0], -thigh[:, 2])
    assert np.ptp(expected_hip) > 0.05
    np.testing.assert_allclose(
        fitted.clip.qpos[:, subject.model.joint("hip").qposadr[0]],
        expected_hip,
        atol=1e-8,
    )
    path = fitted.save(tmp_path / "vertex_retargeting.npz")
    replay = load_retargeted_clip(path, subject)
    np.testing.assert_allclose(replay.qpos, fitted.clip.qpos)
    provenance = json.loads(path.with_suffix(".json").read_text())["provenance"]
    assert provenance["trial"]["vertex_markers"] == definitions.metadata()
    assert result.evidence["trial"]["vertex_markers"]["indices"] == dict(
        definitions.indices
    )
    json.dumps(result.evidence, allow_nan=False)


def test_vertex_units_frame_and_source_uncertainty_reuse_trial_contract() -> None:
    vertices, definitions, _ = _fixture_vertices()
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    translation = np.array([0.3, 0.4, 0.5])
    vertices = (vertices * 0.001 - translation) @ rotation * 100
    trial = vertex_marker_trial(
        vertices,
        definitions,
        topology=TOPOLOGY,
        units="cm",
        rotation_to_model=rotation,
        translation_to_model_m=translation,
        source="local fit export",
        frequency_hz=120,
        confidence=0.9,
        uncertainty_m=0.003,
    )
    expected = json.loads((FIXTURES / "static_trial.json").read_text())["trial"]
    np.testing.assert_allclose(
        trial.in_model_frame(), np.array(expected["positions"]) / 1000
    )
    assert trial.metadata()["source"] == "local fit export"
    assert trial.metadata()["confidence"] == 0.9
    assert trial.metadata()["uncertainty_m"] == 0.003
    assert trial.frequency_hz == 120
    with pytest.raises(ValueError, match="proper orthonormal"):
        replace(trial, rotation_to_model=np.diag([-1, 1, 1]))


def test_missing_vertex_samples_omit_only_affected_measurements() -> None:
    vertices, definitions, payload = _fixture_vertices()
    vertices[:, definitions.indices["KNEE"]] = np.nan
    vertices[:, 0] = np.nan
    trial = vertex_marker_trial(
        vertices,
        definitions,
        topology=TOPOLOGY,
        units="mm",
        rotation_to_model=np.eye(3),
        source="incomplete synthetic fit",
    )
    result = estimate_marker_scales(
        trial, [SegmentMeasurement(**row) for row in payload["measurements"]]
    )
    assert set(result.estimate.factors) == {"foot"}
    assert [row["status"] for row in result.evidence["measurements"]] == [
        "missing_samples",
        "missing_samples",
        "estimated",
    ]


@pytest.mark.parametrize(
    "updates",
    [
        {"topology": ""},
        {"vertex_count": 0},
        {"vertex_count": True},
        {"vertex_count": 15.0},
        {"indices": {}},
        {"indices": {"": 1}},
        {"indices": {"KNEE": -1}},
        {"indices": {"KNEE": 15}},
        {"indices": {"KNEE": 1.5}},
        {"indices": {"KNEE": True}},
        {"definition_source": " "},
        {"definition_license": ""},
    ],
)
def test_vertex_definitions_reject_invalid_indices_or_provenance(updates: dict) -> None:
    _, definitions, _ = _fixture_vertices()
    with pytest.raises(ValueError):
        replace(definitions, **updates)


@pytest.mark.parametrize("shape", [(15,), (15, 2), (14, 3), (0, 15, 3), (2, 14, 3)])
def test_geometry_rejects_wrong_shape_or_vertex_count(shape: tuple[int, ...]) -> None:
    _, definitions, _ = _fixture_vertices()
    with pytest.raises(ValueError, match="shape"):
        vertex_marker_trial(
            np.zeros(shape),
            definitions,
            topology=TOPOLOGY,
            units="m",
            rotation_to_model=np.eye(3),
            source="invalid geometry",
        )


def test_geometry_rejects_mismatched_topology_and_infinite_coordinates() -> None:
    vertices, definitions, _ = _fixture_vertices()
    with pytest.raises(ValueError, match="topology"):
        vertex_marker_trial(
            vertices,
            definitions,
            topology="different topology with the same vertex count",
            units="m",
            rotation_to_model=np.eye(3),
            source="other fit",
        )
    vertices[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        vertex_marker_trial(
            vertices,
            definitions,
            topology=TOPOLOGY,
            units="m",
            rotation_to_model=np.eye(3),
            source="invalid unused vertex",
        )


def test_definitions_and_extracted_markers_are_copied_from_mutable_inputs() -> None:
    indices = {"A": np.int64(0), "B": 1}
    definitions = VertexMarkerSet(
        TOPOLOGY, np.int64(2), indices, "authored", "Apache-2.0"
    )
    vertices = np.array([[0, 0, 0], [0, 0, 0.6]])
    trial = vertex_marker_trial(
        vertices,
        definitions,
        topology=TOPOLOGY,
        units="m",
        rotation_to_model=np.eye(3),
        source="mutable input",
    )
    indices["A"] = 1
    vertices[:] = 99
    assert definitions.indices["A"] == 0
    assert trial.in_model_frame()[0, 1, 2] == 0.6
    with pytest.raises(TypeError):
        definitions.indices["A"] = 1
    json.dumps(trial.metadata(), allow_nan=False)
