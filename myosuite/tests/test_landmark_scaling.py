# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Independent geometric controls for correspondence-based segment scaling."""

from __future__ import annotations

import json
from dataclasses import replace

import mujoco
import numpy as np
import pytest
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation

from myosuite.core.subject_model import SubjectModelConfig, build_subject_model
from myosuite.integrations.capture.landmark_scaling import (
    LandmarkCorrespondences,
    fit_landmark_scales,
)

pytestmark = pytest.mark.tier1
FACTORS = np.array([1.17, 0.91, 1.08])
ROTATION = Rotation.from_rotvec([0.4, -0.2, 0.7])
TRANSLATION = np.array([0.2, -0.3, 1.1])


def _case(count: int = 32) -> LandmarkCorrespondences:
    reference = np.random.default_rng(37).uniform(-1, 1, (count, 3)) * [0.05, 0.3, 0.03]
    return LandmarkCorrespondences(
        "synthetic-person",
        "tibia_r",
        tuple(f"P{i}" for i in range(count)),
        reference,
        ROTATION.apply(reference * FACTORS) + TRANSLATION,
        "independently authored asymmetric reference v1",
        "analytic subject transform v1",
    )


def test_recovers_local_axis_factors_and_proper_pose_with_complete_evidence() -> None:
    case = _case()
    result = fit_landmark_scales([case], max_rms_m=1e-8, mass_kg=73.0)
    np.testing.assert_allclose(result.estimate.factors["tibia_r"], FACTORS, atol=1e-9)
    row = result.evidence["segments"][0]
    np.testing.assert_allclose(
        row["rotation_reference_to_observed"], ROTATION.as_matrix(), atol=1e-9
    )
    np.testing.assert_allclose(row["translation_m"], TRANSLATION, atol=1e-9)
    assert np.linalg.det(row["rotation_reference_to_observed"]) == pytest.approx(1)
    assert row["rms_m"] < 1e-10
    assert row["fit_jacobian_rank"] == 9
    assert row["reference_geometry_rank"] == row["observed_geometry_rank"] == 3
    assert (
        len(row["reference_positions_sha256"])
        == len(row["observed_positions_sha256"])
        == 64
    )
    assert row["used_labels"] == list(case.labels)
    assert row["reference_source"] == case.reference_source
    assert result.estimate.mass_kg == 73
    assert result.estimate.confidence is None
    assert "reference_positions_m" not in row
    assert "observed_positions_m" not in row
    json.dumps(result.evidence, allow_nan=False)


def test_pose_origin_and_rigid_observation_frame_do_not_change_factors() -> None:
    case = _case()
    shift = np.array([30.0, -20.0, 70.0])
    rotation = Rotation.from_rotvec([1.1, 0.4, -0.3])
    # Copy: the case freezes its arrays and SciPy's apply can reject read-only input.
    observed = rotation.apply(np.array(case.observed_positions_m)) + shift
    moved = replace(
        case,
        reference_positions_m=np.asarray(case.reference_positions_m) + shift,
        observed_positions_m=observed,
    )
    original, transformed = (fit_landmark_scales([value]) for value in (case, moved))
    np.testing.assert_allclose(
        transformed.estimate.factors["tibia_r"],
        original.estimate.factors["tibia_r"],
        atol=1e-8,
    )
    assert transformed.evidence["segments"][0]["reference_geometry_rank"] == 3


def test_robust_multiaxis_fit_improves_independent_heldout_geometry() -> None:
    case = _case(40)
    observed = np.array(case.observed_positions_m)
    observed += np.random.default_rng(73).normal(0, 0.0004, observed.shape)
    observed[0] += [0.04, -0.035, 0.025]
    noisy = replace(case, observed_positions_m=observed, uncertainty_m=0.0004)
    robust = fit_landmark_scales([noisy], loss="soft_l1")
    ordinary = fit_landmark_scales([noisy], loss="linear")
    isotropic = fit_landmark_scales([noisy], mode="isotropic", loss="soft_l1")
    reference = np.asarray(case.reference_positions_m)
    single = np.ones(3)
    single[1] = np.linalg.norm(observed[1] - observed[0]) / np.linalg.norm(
        reference[1] - reference[0]
    )
    heldout = np.random.default_rng(91).uniform(-1, 1, (80, 3)) * [0.05, 0.3, 0.03]
    actual_distances = pdist(heldout * FACTORS)

    def error(scales: np.ndarray) -> float:
        return float(
            np.sqrt(np.mean((pdist(heldout * scales) - actual_distances) ** 2))
        )

    fitted = np.asarray(robust.estimate.factors["tibia_r"])
    assert error(fitted) < 0.001
    assert error(fitted) < 0.25 * min(
        error(np.ones(3)),
        error(single),
        error(np.asarray(isotropic.estimate.factors["tibia_r"])),
    )
    assert error(fitted) < 0.25 * error(
        np.asarray(ordinary.estimate.factors["tibia_r"])
    )
    assert robust.evidence["segments"][0]["max_error_m"] > 0.04


def test_source_uncertainty_changes_weighting_without_inventing_confidence() -> None:
    case = _case()
    points = np.array(case.observed_positions_m)
    points[0] += [0.025, 0.035, -0.02]
    sigma = np.full(len(points), 0.001)
    sigma[0] = 0.1
    equal = fit_landmark_scales([replace(case, observed_positions_m=points)])
    weighted = fit_landmark_scales(
        [replace(case, observed_positions_m=points, uncertainty_m=sigma)]
    )
    a, b = (np.asarray(fit.estimate.factors["tibia_r"]) for fit in (equal, weighted))
    assert np.linalg.norm(b - FACTORS) < 0.01 * np.linalg.norm(a - FACTORS)
    assert weighted.evidence["segments"][0]["uncertainty_m"] == sigma.tolist()
    assert weighted.estimate.confidence is None


def test_missing_landmarks_are_omitted_and_input_arrays_are_immutable_copies() -> None:
    case = _case()
    points = np.array(case.observed_positions_m)
    points[0, 1] = np.nan
    missing = replace(case, observed_positions_m=points, uncertainty_m=0.002)
    points[:] = 99
    assert np.isnan(np.asarray(missing.observed_positions_m)[0, 1])
    with pytest.raises(ValueError, match="read-only"):
        missing.observed_positions_m[1, 0] = 99
    result = fit_landmark_scales([missing], max_rms_m=1e-8)
    np.testing.assert_allclose(result.estimate.factors["tibia_r"], FACTORS, atol=1e-9)
    assert result.evidence["segments"][0]["missing_labels"] == ["P0"]
    assert any("missing landmarks" in note for note in result.estimate.warnings)
    json.dumps(result.evidence, allow_nan=False)


def test_missing_geometry_cannot_invent_axes_or_scale_estimates() -> None:
    case = _case()
    with pytest.raises(ValueError, match="geometric rank"):
        fit_landmark_scales(
            [replace(case, observed_positions_m=np.full((len(case.labels), 3), np.nan))]
        )


def test_factors_feed_existing_subject_builder_and_keep_source_model_unchanged() -> (
    None
):
    result = fit_landmark_scales([_case()], max_rms_m=1e-8)
    source = mujoco.MjSpec.from_string("""<mujoco><worldbody>
        <body name="tibia_r"><freejoint/>
        <geom type="box" size="0.02 0.2 0.03" mass="1"/>
        <site name="probe" pos="0.03 0.1 0.02"/>
        </body></worldbody></mujoco>""")
    subject = build_subject_model(source, SubjectModelConfig(result.estimate))
    np.testing.assert_allclose(
        subject.model.site("probe").pos, [0.03, 0.1, 0.02] * FACTORS
    )
    np.testing.assert_allclose(source.compile().site("probe").pos, [0.03, 0.1, 0.02])
    assert subject.metadata["config"]["estimate"]["source"].startswith(
        "landmark-fit:synthetic-person@"
    )


def test_rank_deficient_geometry_requires_explicit_identifiable_isotropy() -> None:
    case = _case()
    reference = np.asarray(case.reference_positions_m)[:3].copy()
    reference[:, 2] = 0
    planar = replace(
        case,
        labels=case.labels[:3],
        reference_positions_m=reference,
        observed_positions_m=ROTATION.apply(reference * 1.12) + TRANSLATION,
    )
    with pytest.raises(ValueError, match="geometric rank"):
        fit_landmark_scales([planar])
    result = fit_landmark_scales([planar], mode="isotropic", max_rms_m=1e-8)
    np.testing.assert_allclose(
        result.estimate.factors["tibia_r"], [1.12] * 3, atol=1e-9
    )
    assert result.evidence["segments"][0]["fit_jacobian_rank"] == 7
    assert any("isotropic" in note for note in result.estimate.warnings)
    reference[:, 0] = 0
    collinear = replace(planar, reference_positions_m=reference)
    with pytest.raises(ValueError, match="geometric rank"):
        fit_landmark_scales([collinear], mode="isotropic")


def test_nearly_planar_geometry_is_rejected_independently_of_rigid_frame() -> None:
    case = _case()
    points = np.array(case.reference_positions_m)
    points[:, 2] *= 1e-10
    thin = replace(
        case,
        reference_positions_m=points,
        observed_positions_m=ROTATION.apply(points * FACTORS) + TRANSLATION,
    )
    with pytest.raises(ValueError, match="ill-conditioned"):
        fit_landmark_scales([thin])


def test_reflections_and_incompatible_scale_bounds_fail_strict_physical_quality() -> (
    None
):
    case = _case()
    reflected = replace(
        case, observed_positions_m=np.asarray(case.reference_positions_m) * [-1, 1, 1]
    )
    with pytest.raises(ValueError, match="marker RMS"):
        fit_landmark_scales([reflected], max_rms_m=1e-6)
    with pytest.raises(ValueError, match="marker RMS"):
        fit_landmark_scales([case], scale_bounds=(0.8, 1.1), max_rms_m=1e-6)
    bounded = fit_landmark_scales([case], scale_bounds=(0.8, 1.1))
    assert any("configured bound" in note for note in bounded.estimate.warnings)


def test_different_people_duplicate_segments_and_repeated_points_are_rejected() -> None:
    case = _case()
    with pytest.raises(ValueError, match="different observed subjects"):
        fit_landmark_scales(
            [case, replace(case, subject_id="another", segment="tibia_l")]
        )
    with pytest.raises(ValueError, match="duplicate segments"):
        fit_landmark_scales([case, case])
    points = np.array(case.reference_positions_m)
    points[0] = points[1]
    with pytest.raises(ValueError, match="Repeated landmark positions"):
        fit_landmark_scales([replace(case, reference_positions_m=points)])


@pytest.mark.parametrize(
    "change",
    [
        {"observed_positions_m": np.zeros((2, 32, 3))},
        {"reference_positions_m": np.full((32, 3), np.inf)},
        {"uncertainty_m": 0},
        {"uncertainty_m": [0.01] * 31},
        {"reference_source": ""},
        {"labels": tuple("same" for _ in range(32))},
    ],
)
def test_invalid_correspondence_inputs_fail(change: dict) -> None:
    with pytest.raises(ValueError):
        replace(_case(), **change)


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "invented"},
        {"loss": "invented"},
        {"robust_scale_m": 0},
        {"scale_bounds": (2, 1)},
        {"max_rms_m": -1},
        {"max_condition_number": 1},
        {"max_evaluations": True},
    ],
)
def test_invalid_fitting_options_fail(options: dict) -> None:
    with pytest.raises(ValueError):
        fit_landmark_scales([_case()], **options)
