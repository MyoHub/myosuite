# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Independent named CSV inputs without dataset-specific assumptions."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from myosuite.integrations.capture.landmarks import read_named_landmarks
from myosuite.integrations.capture.markers import MarkerTrial
from myosuite.integrations.capture.trial_io import load_marker_trial, save_marker_trial

pytestmark = pytest.mark.tier1


def _read(path: Path, **kwargs: Any) -> MarkerTrial:
    return read_named_landmarks(
        path,
        units="cm",
        rotation_to_model=[[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        translation_to_model_m=[0.1, 0.2, 0.3],
        source="independent synthetic source v1",
        definition_source="independent annotation protocol",
        definition_license="Apache-2.0; synthetic test data",
        **kwargs,
    )


def test_generic_csv_preserves_missing_coordinates_frame_and_source_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "annotations.csv"
    path.write_text("\ufeffsurface_a,10,20,30\n\nsurface_β,40,,NaN\n", encoding="utf-8")
    trial = _read(
        path,
        point_semantics="manually annotated surface points",
        subject_id="synthetic-a",
        uncertainty_m=0.003,
    )
    assert trial.labels == ("surface_a", "surface_β")
    assert trial.positions.shape == (1, 2, 3)
    np.testing.assert_allclose(trial.in_model_frame()[0, 0], [-0.1, 0.3, 0.6])
    assert np.isnan(trial.positions[0, 1, 1:]).all()
    assert trial.metadata()["landmark_csv"] == {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert trial.point_semantics == "manually annotated surface points"
    assert trial.subject_id == "synthetic-a"
    restored = load_marker_trial(save_marker_trial(trial, tmp_path / "trial.npz"))
    assert restored.metadata() == trial.metadata()


@pytest.mark.parametrize(
    "content",
    [
        "point,1,2\n",
        "point,1,2,3\npoint,4,5,6\n",
        "point,one,2,3\n",
        "point,1,inf,3\n",
        "",
    ],
)
def test_rejects_ambiguous_or_malformed_csv(tmp_path: Path, content: str) -> None:
    path = tmp_path / "annotations.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        _read(path)
