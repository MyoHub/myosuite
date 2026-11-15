# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache 2 license in the LICENSE file.
"""scale_model's OpenSim-file route against the stored OpenSim ModelScaler reference.

The fast tests rerun the MyoSuite half and compare it with the stored OpenSim
half (bounds and the mutations each bound must reject) and with the stored
MyoSuite half (a tight self-regression). Only the slow test runs OpenSim.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import mujoco
import numpy as np
import pytest

from myosuite.core.body_scaling import scale_model
from scripts import validate_opensim_scaling as validation

pytestmark = pytest.mark.tier1

TOLERANCES = json.loads((validation.DATA / "tolerances.json").read_text())
SCALED_CASES = validation.compared_cases()[1:]
# The preserved-mass case differs from uniform_volume by its masses only.
MUTATED_CASES = ("uniform_volume", "anisotropic_volume")


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(validation.REFERENCE.read_text())


@pytest.fixture(scope="module")
def measured() -> dict:
    return validation.sample_myo()


@pytest.fixture(scope="module")
def mutants() -> dict:
    """Sampled mutants by (case, mutation), filled on first use."""
    return {}


def _mutant(cache: dict, case: str, mutation: str) -> dict[str, Any]:
    if (case, mutation) not in cache:
        recipe = validation.fixture()["cases"][case]
        cache[case, mutation] = validation.sample_myo_case(
            validation.build_myo_case(recipe, mutation),
            recipe["factors"].get("calcn_r", [1.0, 1.0, 1.0]),
        )
    return cache[case, mutation]


def _mutations(case: str) -> list[str]:
    # Swapping two equal factors changes nothing.
    uniform = validation.category(case) == "uniform"
    return [m for m in validation.MUTATIONS if not (uniform and m == "femur_r_swap_xy")]


def _wrap_increment(
    sample: Mapping[str, Any], nominal: Mapping[str, Any], opensim: dict, case: str
) -> float:
    record = "pelvis_wrap_dimensions_m"
    scaled = np.array(sample[record]) - np.array(opensim["cases"][case][record])
    unscaled = np.array(nominal[record]) - np.array(opensim["cases"]["nominal"][record])
    return float(np.abs(scaled - unscaled).max())


def _scores(
    sample: Mapping[str, Any], measured: dict, reference: dict, case: str
) -> dict[str, float]:
    """Discrepancy over bound of every bounded quantity of a sampled *case*."""
    myo = {"cases": {"nominal": measured["cases"]["nominal"], case: sample}}
    bounds = TOLERANCES[validation.category(case)]
    scores = {
        metric: float(
            np.abs(
                validation.discrepancy(myo, reference["opensim"], case, metric)
            ).max()
        )
        / bound
        for metric, bound in bounds.items()
    }
    nominal = measured["cases"]["nominal"]
    scores["pelvis_wrap_dimensions_m"] = (
        _wrap_increment(sample, nominal, reference["opensim"], case)
        / TOLERANCES["pelvis_wrap_dimensions_m"]
    )
    for metric, value in scores.items():
        assert np.isfinite(value), f"{metric}: nonfinite score {value!r}"
    return scores


def _regression_score(sample: Mapping[str, Any], stored: Mapping[str, Any]) -> float:
    """Largest deviation from the stored MyoSuite half, in units of its tolerance."""
    limits = TOLERANCES["self_regression"]
    worst = 0.0
    for record, expected in stored.items():
        atol = limits["atol_by_record"].get(record, limits["atol"])
        expected = np.asarray(expected, dtype=float)
        actual = np.asarray(sample[record], dtype=float)
        assert (
            actual.shape == expected.shape
        ), f"{record}: shape {actual.shape} != {expected.shape}"
        assert np.all(np.isfinite(expected)) and np.all(
            np.isfinite(actual)
        ), f"{record}: nonfinite value"
        allowed = atol + limits["rtol"] * np.abs(expected)
        worst = max(worst, float(np.max(np.abs(actual - expected) / allowed)))
    return worst


def _affine_inertia(
    inertia: np.ndarray, factors: np.ndarray, mass_ratio: float
) -> np.ndarray:
    """Inertia after stretching the mass distribution by *factors* (the documented rule)."""
    second = 0.5 * np.trace(inertia) * np.eye(3) - inertia
    second = mass_ratio * np.diag(factors) @ second @ np.diag(factors)
    return np.trace(second) * np.eye(3) - second


def test_reference_inputs_match_fixture_and_assets(reference: dict) -> None:
    assert reference["schema"] == 2
    assert reference["fixture"] == validation.fixture()
    assert reference["fixture"]["muscles"] == list(validation.MUSCLES)
    assert reference["fixture"]["body_metrics_order"] == list(validation.BODIES)
    assert reference["opensim"]["provenance"]["model_sha256"] == validation.MODEL_SHA256
    # The stored MyoSuite half is a baseline for these model assets only.
    assert (
        importlib.metadata.version("myo-sim")
        == reference["myosuite"]["provenance"]["myo_sim_version"]
    )


@pytest.mark.parametrize("case", SCALED_CASES)
def test_scaling_discrepancy_stays_within_bounds(
    case: str, reference: dict, measured: dict
) -> None:
    scores = _scores(measured["cases"][case], measured, reference, case)
    assert max(scores.values()) < 1, scores


@pytest.mark.parametrize("case", MUTATED_CASES)
def test_each_bound_rejects_the_mutations_it_covers(
    case: str, reference: dict, measured: dict, mutants: dict
) -> None:
    missed = []
    for quantity, mutations in TOLERANCES["covers"][validation.category(case)].items():
        for mutation in mutations:
            sample = _mutant(mutants, case, mutation)
            if _scores(sample, measured, reference, case)[quantity] <= 1:
                missed.append((quantity, mutation))
    assert not missed


@pytest.mark.parametrize("case", MUTATED_CASES)
def test_self_regression_rejects_every_mutation(
    case: str, reference: dict, mutants: dict
) -> None:
    category = validation.category(case)
    covered = {m for ms in TOLERANCES["covers"][category].values() for m in ms}
    regression_only = set(TOLERANCES["regression_only"][category])
    # Every mutation is either seen by a cross-model bound or listed as unseen.
    assert covered | regression_only == set(_mutations(case))
    assert not covered & regression_only
    stored = reference["myosuite"]["cases"][case]
    missed = [
        mutation
        for mutation in _mutations(case)
        if _regression_score(_mutant(mutants, case, mutation), stored) <= 1
    ]
    assert not missed


def test_myosuite_half_reproduces_its_stored_baseline(
    reference: dict, measured: dict
) -> None:
    stored = reference["myosuite"]
    assert set(measured["cases"]) == set(stored["cases"])
    for case, values in stored["cases"].items():
        assert _regression_score(measured["cases"][case], values) <= 1, case
    for mode, values in stored["inertia_probe"].items():
        np.testing.assert_allclose(
            measured["inertia_probe"][mode]["inertia_kgm2"],
            values["inertia_kgm2"],
            rtol=1e-12,
            atol=1e-15,
        )


def test_regression_score_rejects_nonfinite_and_shape_mismatch(
    reference: dict,
) -> None:
    stored = reference["myosuite"]["cases"]["anisotropic_volume"]
    for record in ("head_site_m", "torso_geom_offsets_m", "torso_mass_kg"):
        mutated = np.array(stored[record], dtype=float, copy=True)
        mutated.flat[0] = np.nan
        with pytest.raises(AssertionError, match=record):
            _regression_score({**stored, record: mutated}, stored)

    inf_mass = np.array(stored["torso_mass_kg"], dtype=float, copy=True)
    inf_mass.flat[0] = np.inf
    with pytest.raises(AssertionError, match="torso_mass_kg"):
        _regression_score({**stored, "torso_mass_kg": inf_mass}, stored)

    truncated_head = np.asarray(stored["head_site_m"], dtype=float)[:-1]
    with pytest.raises(AssertionError, match="head_site_m"):
        _regression_score({**stored, "head_site_m": truncated_head}, stored)


def test_mass_scaling_matches_opensim(reference: dict, measured: dict) -> None:
    comparison = validation.compare(measured, reference["opensim"])
    for case in SCALED_CASES:
        assert comparison[case]["body_mass_kg"]["scaling_ratio_difference_max"] < 1e-12
        if validation.category(case) == "uniform":
            inertia = comparison[case]["body_inertia_kgm2"]
            assert inertia["scaling_ratio_difference_max"] < 1e-10


def test_inertia_follows_the_affine_rule(measured: dict) -> None:
    # The controlled probe: 10 kg, diagonal inertia (0.1, 0.2, 0.25), factors (1.2, 0.8, 1.1).
    factors = np.array([1.2, 0.8, 1.1])
    for mode, mass_ratio in (("volume", float(np.prod(factors))), ("keep", 1.0)):
        probe = measured["inertia_probe"][mode]
        assert probe["mass_kg"] == pytest.approx(10.0 * mass_ratio, rel=1e-12)
        np.testing.assert_allclose(
            probe["inertia_kgm2"],
            _affine_inertia(np.diag([0.1, 0.2, 0.25]), factors, mass_ratio),
            rtol=1e-12,
            atol=1e-15,
        )
    # MyoLeg's pelvis and right-leg bodies, whose frames are OpenSim's.
    recipe = validation.fixture()["cases"]["anisotropic_volume"]["factors"]
    scaled = measured["cases"]["anisotropic_volume"]
    nominal = measured["cases"]["nominal"]
    for index, body in enumerate(validation.BODIES):
        factors = np.array(recipe[body])
        expected = _affine_inertia(
            np.array(nominal["body_inertia_kgm2"][index]),
            factors,
            scaled["body_mass_kg"][index] / nominal["body_mass_kg"][index],
        )
        np.testing.assert_allclose(
            scaled["body_inertia_kgm2"][index], expected, rtol=1e-10, atol=1e-14
        )


def _segment_axes(model: mujoco.MjModel, body: str) -> np.ndarray:
    """For each axis of *body*, the pelvis (OpenSim) axis it lies along."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    pelvis = data.xmat[model.body("pelvis").id].reshape(3, 3)
    turn = pelvis.T @ data.xmat[model.body(body).id].reshape(3, 3)
    return np.argmax(np.abs(turn), axis=0)


def _torso_and_head(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Torso primitive geom positions, torso inertia and the root's head site, locally."""
    torso = model.body("torso").id
    primitives = (model.geom_bodyid == torso) & (
        model.geom_type != mujoco.mjtGeom.mjGEOM_MESH
    )
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.body_iquat[torso])
    rot = rot.reshape(3, 3)
    inertia = rot @ np.diag(model.body_inertia[torso]) @ rot.T
    return model.geom_pos[primitives], inertia, model.site_pos[model.site("head").id]


def test_torso_and_root_take_opensim_axes(tmp_path: Path) -> None:
    # MyoLeg's root and torso are z-up; OpenSim's y is up. Distinct factors per
    # OpenSim axis must land on the body axes that point the same way.
    factors = np.array([1.1, 1.2, 1.3])
    spec = validation.nominal_spec()
    nominal = spec.copy().compile()
    torso_factors = factors[_segment_axes(nominal, "torso")]
    root_factors = factors[_segment_axes(nominal, "root")]
    assert not np.array_equal(torso_factors, factors)
    geoms, inertia, head = _torso_and_head(nominal)

    def matches(model: mujoco.MjModel) -> bool:
        scaled_geoms, scaled_inertia, scaled_head = _torso_and_head(model)
        expected_inertia = _affine_inertia(
            inertia, torso_factors, float(np.prod(factors))
        )
        return (
            np.allclose(scaled_geoms, geoms * torso_factors, rtol=0, atol=1e-12)
            and np.allclose(scaled_inertia, expected_inertia, rtol=1e-10, atol=1e-14)
            and np.allclose(scaled_head, head * root_factors, rtol=0, atol=1e-12)
        )

    path = validation.write_scaleset(
        {"pelvis": factors, "torso": factors}, tmp_path / "scale.xml"
    )
    assert matches(scale_model(spec, path))
    # The known-bad mapping: OpenSim factors along each body's own axes.
    assert not matches(scale_model(spec, {"root": factors, "torso": factors}))


def test_root_height_regression_sees_unscaled_feet(
    reference: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Inject the bug through the private geom-scaling helper: the feet keep their
    # nominal collision geoms. The calcaneus landmarks cannot see this.
    from myo_sim import scaling as model_scaling

    scale_geom = model_scaling._scale_geom

    def skip_feet(spec: mujoco.MjSpec, geom: Any, factors: np.ndarray) -> None:
        if not (
            isinstance(geom, mujoco.MjsGeom)
            and geom.parent.name.startswith(("calcn_", "toes_"))
        ):
            scale_geom(spec, geom, factors)

    monkeypatch.setattr(model_scaling, "_scale_geom", skip_feet)
    case = "anisotropic_volume"
    recipe = validation.fixture()["cases"][case]
    model = validation.build_myo_case(recipe)
    record = "root_heights_m"
    stored = {record: reference["myosuite"]["cases"][case][record]}
    assert _regression_score(validation.subject_records(model), stored) > 1


def test_model_download_refuses_corrupted_cache(tmp_path: Path) -> None:
    path = tmp_path / "Rajagopal2016.osim"
    path.write_text("changed upstream data")
    with pytest.raises(ValueError, match="SHA-256"):
        validation.download_model(path)


def test_moment_arm_finite_difference_is_stable(measured: dict) -> None:
    recipe = validation.fixture()["cases"]["anisotropic_volume"]
    finer = validation.sample_myo_case(
        validation.build_myo_case(recipe),
        recipe["factors"]["calcn_r"],
        moment_arm_step=1e-6,
    )
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
    stored = reference["opensim"]
    for case, values in stored["cases"].items():
        for record, expected in values.items():
            np.testing.assert_allclose(
                actual["cases"][case][record], expected, rtol=1e-6, atol=1e-8
            )
    for mode, values in stored["inertia_probe"].items():
        np.testing.assert_allclose(
            actual["inertia_probe"][mode]["inertia_kgm2"],
            values["inertia_kgm2"],
            rtol=1e-12,
            atol=1e-15,
        )
