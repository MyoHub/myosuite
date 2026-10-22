# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache 2 license in the LICENSE file.
"""External OpenSim reference bounds and deliberate scaling mutations."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from scripts import validate_opensim_scaling as validation

pytestmark = pytest.mark.tier1


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads((validation.DATA / "reference.json").read_text())


@pytest.fixture(scope="module")
def measured() -> dict:
    return validation.sample_myo()


def _bounds(case: str) -> dict[str, float]:
    limits = json.loads((validation.DATA / "tolerances.json").read_text())
    return limits["uniform" if case.startswith("uniform") else "anisotropic"]


def _increment(myo: dict, oracle: dict, case: str, metric: str) -> float:
    difference = np.array(myo["cases"][case][metric]) - np.array(
        oracle["cases"][case][metric]
    )
    nominal = np.array(myo["cases"]["nominal"][metric]) - np.array(
        oracle["cases"]["nominal"][metric]
    )
    return float(np.abs(difference - nominal).max())


def test_reference_provenance_is_complete(reference: dict) -> None:
    assert reference["model_sha256"] == validation.MODEL_SHA256
    assert reference["fixture"] == validation.fixture()
    assert reference["fixture"]["muscles"] == list(validation.MUSCLES)
    assert reference["fixture"]["body_metrics_order"] == list(validation.BODIES)
    assert "4.6" in reference["opensim"]["version"]
    assert reference["myosuite"]["myo_sim_version"] == "0.2.3"
    assert "MIT" in reference["license"]
    assert "simtk.org/projects/full_body" in reference["license_source"]
    assert set(reference["code_sha256"]) == {
        "scripts/validate_opensim_scaling.py",
        "myosuite/core/body_scaling.py",
        "myosuite/core/subject_model.py",
    }


@pytest.mark.parametrize("case", list(validation.fixture()["cases"])[1:])
def test_measured_scaling_increment_stays_within_external_bounds(
    case: str,
    reference: dict,
    measured: dict,
) -> None:
    for metric, tolerance in _bounds(case).items():
        assert _increment(measured, reference["opensim"], case, metric) < tolerance, (
            case,
            metric,
        )


@pytest.mark.parametrize("case", list(validation.fixture()["cases"])[1:])
def test_every_external_bound_rejects_a_known_bad_mutation(
    case: str,
    reference: dict,
    measured: dict,
) -> None:
    changed = {
        mutation: validation.sample_myo_case(
            validation.build_myo_case(validation.fixture()["cases"][case], mutation)
        )
        for mutation in ("wrong_factor", "wrong_lengthrange")
    }
    if case.startswith("anisotropic"):
        changed["wrong_axis"] = validation.sample_myo_case(
            validation.build_myo_case(validation.fixture()["cases"][case], "wrong_axis")
        )
    for metric, tolerance in _bounds(case).items():
        scores = []
        for values in changed.values():
            mutated = {"cases": dict(measured["cases"], **{case: values})}
            scores.append(_increment(mutated, reference["opensim"], case, metric))
        assert max(scores) > tolerance, (case, metric, scores)


def test_mass_modes_and_uniform_inertia_match_opensim_scaling_ratios(
    reference: dict,
    measured: dict,
) -> None:
    comparisons = validation.compare(measured, reference["opensim"])
    for case, values in comparisons.items():
        assert values["body_mass_kg"]["scaling_ratio_difference_max"] < 1e-12
        if not case.startswith("anisotropic"):
            assert values["body_inertia_kgm2"]["scaling_ratio_difference_max"] < 1e-10
    # The nonuniform inertia disagreement is measured and retained, not called parity.
    assert (
        comparisons["anisotropic_volume"]["body_inertia_kgm2"][
            "scaling_ratio_difference_max"
        ]
        > 0.1
    )


def test_controlled_inertia_probe_exposes_the_different_scaling_laws(
    reference: dict,
    measured: dict,
) -> None:
    for mode in ("volume", "keep"):
        ours = measured["inertia_probe"][mode]
        independent = reference["opensim"]["inertia_probe"][mode]
        assert ours["mass_kg"] == pytest.approx(independent["mass_kg"])
        difference = np.array(ours["inertia_kgm2"]) - np.array(
            independent["inertia_kgm2"]
        )
        assert np.abs(difference).max() > 0.03
        np.testing.assert_allclose(
            ours["inertia_kgm2"],
            reference["myosuite"]["inertia_probe"][mode]["inertia_kgm2"],
            atol=1e-12,
        )


def test_nominal_conversion_error_and_wrap_replacement_remain_visible(
    reference: dict,
) -> None:
    nominal = reference["comparison"]["nominal"]
    assert nominal["joint_centres_m"]["absolute_max"] > 0.005
    assert nominal["moment_arms_m"]["absolute_max"] > 0.005
    assert nominal["normalized_fibre_length"]["absolute_max"] > 0.3
    assert nominal["psoas_wrap_dimensions_m"]["absolute_max"] == pytest.approx(0.1)
    for runtime in ("myosuite", "opensim"):
        fibre = np.array(
            reference[runtime]["cases"]["anisotropic_volume"]["normalized_fibre_length"]
        )
        assert np.isfinite(fibre).all() and np.all(
            fibre.max(axis=0) > fibre.min(axis=0)
        )


def test_model_download_refuses_corrupted_cache(tmp_path: Path) -> None:
    path = tmp_path / "Rajagopal2016.osim"
    path.write_text("changed upstream data")
    with pytest.raises(ValueError, match="SHA-256"):
        validation.download_model(path)


def test_moment_arm_finite_difference_is_stable(measured: dict) -> None:
    subject = validation.build_myo_case(
        validation.fixture()["cases"]["anisotropic_volume"]
    )
    finer = validation.sample_myo_case(subject, moment_arm_step=1e-6)
    np.testing.assert_allclose(
        finer["moment_arms_m"],
        measured["cases"]["anisotropic_volume"]["moment_arms_m"],
        rtol=0,
        atol=1e-7,
    )


@pytest.mark.slow
def test_live_opensim_reproduces_saved_oracle(reference: dict, tmp_path: Path) -> None:
    python = os.environ.get("OPENSIM_PYTHON")
    if not python:
        pytest.skip("Set OPENSIM_PYTHON to run the independent OpenSim oracle.")
    path = validation.download_model(tmp_path / "Rajagopal2016.osim")
    output = tmp_path / "oracle.json"
    subprocess.run(
        [
            python,
            str(Path(validation.__file__).resolve()),
            "--opensim-worker",
            "--model",
            str(path),
            "--output",
            str(output),
        ],
        check=True,
    )
    actual = json.loads(output.read_text())
    for case in reference["fixture"]["cases"]:
        for metric in validation.METRICS:
            np.testing.assert_allclose(
                actual["cases"][case][metric],
                reference["opensim"]["cases"][case][metric],
                rtol=1e-6,
                atol=1e-8,
            )
