# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Independent fixed-base pipeline, persisted-provenance and CLI failures."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from myosuite.integrations.capture.arm_example import write_example_inputs
from myosuite.integrations.capture.reporting import QualityThresholds
from myosuite.integrations.capture.trial_io import load_marker_trial, save_marker_trial
from myosuite.integrations.capture.workflow import (
    ModelSource,
    build_target,
    dependency_status,
    estimate_trial,
    main,
    retarget_trial,
    validate_trial,
)

pytestmark = pytest.mark.tier1


def _workflow(tmp_path: Path) -> tuple[dict, Path, Path, Path]:
    inputs = write_example_inputs(tmp_path / "inputs")
    estimate = estimate_trial(
        inputs["trial"],
        inputs["protocol"],
        tmp_path / "estimate.json",
        calibration_frames=inputs["calibration_frames"],
    )
    payload = json.loads(estimate.read_text())
    for name, values in inputs["known_factors"].items():
        np.testing.assert_allclose(payload["estimate"]["factors"][name], values)
    build = build_target(
        ModelSource("xml", inputs["model"]),
        inputs["protocol"],
        estimate,
        tmp_path / "build.json",
    )
    trial = load_marker_trial(inputs["trial"])
    motion = save_marker_trial(
        replace(trial, positions=trial.positions[inputs["heldout_frames"]]),
        tmp_path / "motion_trial.npz",
    )
    clip = retarget_trial(motion, build, tmp_path / "clip.npz")
    return inputs, build, motion, clip


def test_fixed_base_workflow_replays_and_exports_only_aggregate_quality(
    tmp_path: Path,
) -> None:
    _, build, motion, clip = _workflow(tmp_path)
    report = validate_trial(
        motion,
        build,
        clip,
        tmp_path / "aggregate.json",
        thresholds=QualityThresholds(marker_max_m=1e-5, min_converged_fraction=1),
    )
    assert report["quality"]["accepted"]
    assert report["metrics"]["frames"] == 7
    assert report["metrics"]["marker_max_m"] < 1e-8
    assert "root_tracking_rms_m" not in report["metrics"]
    assert "sole_penetration_max_m" not in report["metrics"]
    serialized = (tmp_path / "aggregate.json").read_text()
    for forbidden in (
        str(tmp_path),
        "positions",
        "qpos",
        "rotation_to_model",
        "upper_segment",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "change", ["points", "subject", "qpos", "frequency", "profile"]
)
def test_validation_rejects_changed_inputs_or_stale_solver_evidence(
    tmp_path: Path, change: str
) -> None:
    _, build, motion, clip = _workflow(tmp_path)
    if change in {"points", "subject"}:
        trial = load_marker_trial(motion)
        updated = (
            replace(trial, positions=trial.positions + 0.001)
            if change == "points"
            else replace(trial, subject_id="another-subject")
        )
        save_marker_trial(updated, motion)
    elif change == "profile":
        payload = json.loads(build.read_text())
        payload["protocol"]["name"] = "changed-profile"
        build.write_text(json.dumps(payload))
    else:
        with np.load(clip, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        arrays["qpos" if change == "qpos" else "frequency"] += 0.01
        np.savez_compressed(clip, **arrays)
    with pytest.raises(ValueError, match="differ|changed"):
        validate_trial(
            motion,
            build,
            clip,
            tmp_path / "aggregate.json",
            thresholds=QualityThresholds(marker_max_m=1e-4),
        )
    assert not (tmp_path / "aggregate.json").exists()


def test_cli_run_supports_separate_calibration_and_motion_and_unknown_checks(
    tmp_path: Path,
) -> None:
    inputs = write_example_inputs(tmp_path / "inputs")
    trial = load_marker_trial(inputs["trial"])
    motion = save_marker_trial(
        replace(trial, positions=trial.positions[3:]), tmp_path / "motion.npz"
    )
    argv = [
        "run",
        "--trial",
        inputs["trial"],
        "--motion-trial",
        str(motion),
        "--model",
        inputs["model"],
        "--profile",
        inputs["protocol"],
        "--output",
        str(tmp_path / "run"),
        "--calibration-frames",
        "0",
        "1",
        "2",
        "--marker-max-m",
        "0.00001",
    ]
    main(argv)
    assert json.loads((tmp_path / "run/aggregate.json").read_text())["quality"][
        "accepted"
    ]
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "validate",
                "--trial",
                str(motion),
                "--build",
                str(tmp_path / "run/build.local.json"),
                "--clip",
                str(tmp_path / "run/motion.local.npz"),
                "--output",
                str(tmp_path / "unknown.json"),
                "--root-tracking-rms-m",
                "0.001",
            ]
        )
    assert exc.value.code == 2
    assert (
        json.loads((tmp_path / "unknown.json").read_text())["quality"]["status"]
        == "unknown"
    )


def test_preflight_and_cli_errors_do_not_load_native_assets(tmp_path: Path) -> None:
    assert dependency_status("trial")["available"]
    with pytest.raises(ValueError, match="Unknown source format"):
        dependency_status("made-up")
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "export",
                "--format",
                "csv",
                "--input",
                str(tmp_path / "not-read.csv"),
                "--output",
                str(tmp_path / "out.npz"),
            ]
        )
    assert exc.value.code == 2
    assert not (tmp_path / "out.npz").exists()


def test_unknown_environment_is_a_cli_error(tmp_path: Path) -> None:
    inputs = write_example_inputs(tmp_path)
    estimate = estimate_trial(
        inputs["trial"], inputs["protocol"], tmp_path / "estimate.json"
    )
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "build",
                "--env",
                "not_a_registered_myo_environment-v0",
                "--profile",
                inputs["protocol"],
                "--estimate",
                str(estimate),
                "--output",
                str(tmp_path / "build.json"),
            ]
        )
    assert exc.value.code == 2


def test_descriptive_protocol_name_gets_a_stable_public_identifier(
    tmp_path: Path,
) -> None:
    inputs = write_example_inputs(tmp_path / "inputs")
    profile = Path(inputs["protocol"])
    payload = json.loads(profile.read_text())
    payload["name"] = "left arm / reviewed protocol"
    profile.write_text(json.dumps(payload))
    main(
        [
            "run",
            "--trial",
            inputs["trial"],
            "--model",
            inputs["model"],
            "--profile",
            inputs["protocol"],
            "--output",
            str(tmp_path / "run"),
            "--marker-max-m",
            "0.00001",
        ]
    )
    report = json.loads((tmp_path / "run/aggregate.json").read_text())
    assert report["protocol_id"].startswith("protocol-")
    assert report["quality"]["accepted"]


def test_build_xml_reference_survives_working_directory_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = write_example_inputs(tmp_path)
    estimate = estimate_trial(
        inputs["trial"], inputs["protocol"], tmp_path / "estimate.json"
    )
    monkeypatch.chdir(tmp_path)
    build = build_target(
        ModelSource("xml", "arm_model.xml"),
        inputs["protocol"],
        estimate,
        tmp_path / "build.json",
    )
    monkeypatch.chdir(tmp_path.parent)
    retarget_trial(inputs["trial"], build, tmp_path / "clip.npz")


def test_motion_cannot_silently_use_another_subjects_calibration(
    tmp_path: Path,
) -> None:
    inputs, build, motion, _ = _workflow(tmp_path)
    trial = load_marker_trial(motion)
    save_marker_trial(replace(trial, subject_id="other"), motion)
    with pytest.raises(ValueError, match="different subjects"):
        retarget_trial(motion, build, tmp_path / "other.npz")


def test_workflow_outputs_cannot_overwrite_source_trials(tmp_path: Path) -> None:
    inputs, build, motion, _ = _workflow(tmp_path)
    original = motion.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        retarget_trial(motion, build, motion)
    assert motion.read_bytes() == original
    model = Path(inputs["model"])
    original_model = model.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        build_target(
            ModelSource("xml", str(model)),
            inputs["protocol"],
            tmp_path / "estimate.json",
            model,
        )
    assert model.read_bytes() == original_model
