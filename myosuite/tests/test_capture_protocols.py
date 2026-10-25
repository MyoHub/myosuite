# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Explicit protocol round trips, semantic preflight and scale selection."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite.integrations.capture.arm_example import write_example_inputs
from myosuite.integrations.capture.protocols import TargetProtocol
from myosuite.integrations.capture.trial_io import load_marker_trial

pytestmark = pytest.mark.tier1


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[TargetProtocol, object, mujoco.MjModel]:
    paths = write_example_inputs(tmp_path)
    return (
        TargetProtocol.load(paths["protocol"]),
        load_marker_trial(paths["trial"]),
        mujoco.MjModel.from_xml_path(paths["model"]),
    )


def test_protocol_json_round_trip_and_explicit_axes(
    inputs: tuple, tmp_path: Path
) -> None:
    protocol, trial, model = inputs
    protocol.validate_against_model(model, trial)
    restored = TargetProtocol.load(protocol.save(tmp_path / "nested/protocol.json"))
    assert restored.to_dict() == protocol.to_dict()
    estimate = restored.estimate_scales(trial, np.arange(3))
    np.testing.assert_allclose(estimate.estimate.factors["upper_segment"], [1.2, 1, 1])
    np.testing.assert_allclose(estimate.estimate.factors["lower_segment"], [0.8, 1, 1])
    assert set(estimate.estimate.factors) == {"upper_segment", "lower_segment"}
    assert estimate.evidence["target_protocol"] == protocol.to_dict()
    assert estimate.evidence["trial"]["units"] == "mm"
    json.dumps(estimate.evidence, allow_nan=False)


@pytest.mark.parametrize("semantics", ["unspecified", "surface_marker", "model_origin"])
def test_semantics_are_required_before_estimation_or_target_validation(
    inputs: tuple, semantics: str
) -> None:
    protocol, trial, model = inputs
    changed = replace(trial, point_semantics=semantics)
    with pytest.raises(ValueError, match="point_semantics"):
        protocol.estimate_scales(changed, (0,))
    with pytest.raises(ValueError, match="point_semantics"):
        protocol.validate_against_model(model, changed)


def test_missing_label_is_early_but_missing_observations_are_preserved(
    inputs: tuple,
) -> None:
    protocol, trial, model = inputs
    protocol.validate_against_model(model, trial)
    assert np.isnan(trial.positions[5, trial.labels.index("WRIST")]).all()
    missing = replace(
        trial, labels=trial.labels[:-1], positions=trial.positions[:, :-1]
    )
    with pytest.raises(ValueError, match="missing trial labels.*LOWER_AUX"):
        protocol.validate_against_model(model, missing)


@pytest.mark.parametrize("selection", [(), (0, 0), (-1,), (1000,), (True,), (0.5,)])
def test_calibration_frame_selection_is_explicit(
    inputs: tuple, selection: tuple
) -> None:
    protocol, trial, _ = inputs
    with pytest.raises(ValueError, match="Calibration frames"):
        protocol.estimate_scales(trial, selection)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("marker_sites", {"SHOULDER": "unknown_site"}, "target site"),
        ("joint_names", ("unknown_joint",), "target joint"),
        ("free_joint", "shoulder_yaw", "free root joint"),
        ("free_joint", "missing_root", "free joint or sole geometry"),
    ],
)
def test_target_names_and_root_type_are_preflighted(
    inputs: tuple, field: str, value: object, message: str
) -> None:
    protocol, trial, model = inputs
    with pytest.raises(ValueError, match=message):
        replace(protocol, **{field: value}).validate_against_model(model, trial)


def test_unknown_body_and_overlapping_axis_are_rejected(inputs: tuple) -> None:
    protocol, trial, model = inputs
    row = protocol.measurements[0]
    with pytest.raises(ValueError, match="target body"):
        replace(
            protocol, measurements=(replace(row, segment="unknown_body"),)
        ).validate_against_model(model, trial)
    with pytest.raises(ValueError, match="Overlapping"):
        replace(protocol, measurements=(row, row))


@pytest.mark.parametrize(
    "changes",
    [
        {"schema": 2},
        {"schema": True},
        {"unexpected": "field"},
        {"joint_names": []},
        {"sole_geoms": None},
        {"sole_geoms": ["sole"]},
        {"root_marker": "SHOULDER"},
        {"ground_height_m": 0.1},
    ],
)
def test_malformed_schema_and_fixed_base_ground_settings_fail(
    inputs: tuple, changes: dict
) -> None:
    protocol, _, _ = inputs
    with pytest.raises(ValueError):
        TargetProtocol.from_dict({**protocol.to_dict(), **changes})
