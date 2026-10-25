# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Independent arm motion recovers morphology without inventing root diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite.integrations.capture.arm_example import run_example, write_example_inputs
from myosuite.core.trajectory_io import MotionClip
from myosuite.integrations.capture.markers import MarkerTrial
from myosuite.integrations.capture.protocols import TargetProtocol
from myosuite.integrations.capture.reporting import QualityThresholds, evaluate_quality
from myosuite.integrations.capture.retargeting import (
    load_retargeted_clip,
    motion_diagnostics,
    retarget_markers,
)
from myosuite.integrations.capture.trial_io import load_marker_trial

pytestmark = pytest.mark.tier1


def test_fixed_arm_scale_build_missing_marker_fit_export_load_replay_and_control(
    tmp_path: Path,
) -> None:
    report = run_example(tmp_path)
    assert report["root_mode"] == "fixed_base"
    assert report["scaled_target"]["converged_frames"] == 7
    assert report["scaled_target"]["marker_max_m"] < 1e-5
    assert report["replayed_target"]["marker_max_m"] < 1e-5
    assert report["nominal_fit_replayed_on_scaled_target"]["marker_rms_m"] > 0.01
    assert report["nominal_fit_replayed_on_scaled_target"]["marker_max_m"] > 0.02
    with np.load(report["clip"], allow_pickle=False) as archive:
        assert archive["qpos_names"].tolist() == ["shoulder_yaw", "elbow_bend"]
        np.testing.assert_allclose(
            archive["qpos"][:, 0], [-0.2, -0.1, 0, 0.1, 0.2, 0.15, 0.05], atol=1e-5
        )
        assert archive["root_trajectory_m"].shape == (0, 3)
        assert archive["sole_bottom_m"].shape == (7, 0)
        assert archive["contact_labels"].shape == (7, 0)
        provenance = json.loads(str(archive["provenance_json"]))
        assert provenance["free_joint"] is None
        assert provenance["target_protocol"]["name"] == "authored-fixed-arm"
        assert (
            provenance["fitted_qpos_sha256"]
            == hashlib.sha256(archive["qpos"].tobytes()).hexdigest()
        )
        assert provenance["fitted_frequency_hz"] == 100
        assert (
            provenance["solver_summary"]["converged_frames"]
            == provenance["solver_summary"]["frames"]
            == 7
        )
    sidecar = json.loads(Path(report["clip"]).with_suffix(".json").read_text())
    assert "root_tracking_rms_m" not in sidecar["quality"]
    assert "sole_penetration_max_m" not in sidecar["quality"]


def test_invalid_root_options_fail_before_any_inverse_kinematics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = write_example_inputs(tmp_path)
    trial = load_marker_trial(paths["trial"])
    protocol = TargetProtocol.load(paths["protocol"])
    spec = mujoco.MjSpec.from_file(paths["model"])
    subject = protocol.build_subject(spec, protocol.estimate_scales(trial, (0,)))

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("IK must not run for an invalid root configuration")

    monkeypatch.setattr(
        "myosuite.integrations.capture.retargeting.qpos_from_marker_targets", forbidden
    )
    with pytest.raises(ValueError, match="explicit free_joint"):
        retarget_markers(subject, trial, protocol.marker_sites, root_marker="SHOULDER")
    with pytest.raises(ValueError, match="free root joint"):
        retarget_markers(
            subject, trial, protocol.marker_sites, free_joint="shoulder_yaw"
        )
    with pytest.raises(ValueError, match="explicit free_joint"):
        retarget_markers(subject, trial, protocol.marker_sites, ground_height_m=0.1)


def test_fixed_base_complete_missing_frame_is_a_failure(tmp_path: Path) -> None:
    paths = write_example_inputs(tmp_path)
    trial = load_marker_trial(paths["trial"])
    protocol = TargetProtocol.load(paths["protocol"])
    spec = mujoco.MjSpec.from_file(paths["model"])
    subject = protocol.build_subject(spec, protocol.estimate_scales(trial, (0,)))
    points = trial.positions.copy()
    points[4] = np.nan
    with pytest.raises(ValueError, match="At least one marker"):
        protocol.retarget(subject, replace(trial, positions=points))


def test_fixed_base_reload_rejects_other_morphology(tmp_path: Path) -> None:
    paths = write_example_inputs(tmp_path)
    trial = load_marker_trial(paths["trial"])
    protocol = TargetProtocol.load(paths["protocol"])
    spec = mujoco.MjSpec.from_file(paths["model"])
    subject = protocol.build_subject(spec, protocol.estimate_scales(trial, (0,)))
    fitted = protocol.retarget(subject, trial)
    path = fitted.save(tmp_path / "motion.npz")
    altered = spec.copy()
    altered.body("lower_segment").pos = [0.4, 0, 0]
    other = protocol.build_subject(altered, protocol.estimate_scales(trial, (0,)))
    with pytest.raises(ValueError, match="different subject"):
        load_retargeted_clip(path, other)
    measured = motion_diagnostics(
        subject.model, fitted.clip, trial, protocol.marker_sites
    )
    assert "root_tracking_rms_m" not in measured
    assert "sole_penetration_max_m" not in measured


def test_equality_diagnostics_evaluate_actual_replayed_poses() -> None:
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body><joint name="a" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1"/><site name="ma"/></body>
      <body pos="0 1 0"><joint name="b" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1"/><site name="mb"/></body>
      </worldbody><equality><joint joint1="a" joint2="b"/></equality></mujoco>""")
    clip = MotionClip(
        np.array([[0, 0], [0.2, 0]]),
        np.zeros((2, 2)),
        np.zeros((2, 2, 3)),
        ["ma", "mb"],
        qpos_joint_names=["a", "b"],
        qvel_joint_names=["a", "b"],
        frequency_hz=100,
    )
    trial = MarkerTrial(("A", "B"), np.zeros((2, 2, 3)), "m", np.eye(3), "authored")
    quality = motion_diagnostics(model, clip, trial, {"A": "ma", "B": "mb"})
    assert quality["equality_error_max"] == pytest.approx(0.2)


def test_free_root_without_sole_geometries_leaves_penetration_unknown() -> None:
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body pos="0 0 .5"><freejoint name="root"/>
        <geom type="sphere" size=".1"/><site name="root_point"/>
      </body></worldbody></mujoco>""")
    poses = np.repeat(model.qpos0[None], 2, axis=0)
    points = np.array([[[0, 0, 0.5]], [[0, 0, 0.5]]])
    clip = MotionClip(
        poses,
        np.zeros((2, model.nv)),
        points,
        ["root_point"],
        qpos_joint_names=["root"],
        qvel_joint_names=["root"],
        frequency_hz=100,
    )
    trial = MarkerTrial(("ROOT",), points, "m", np.eye(3), "authored")
    quality = motion_diagnostics(
        model,
        clip,
        trial,
        {"ROOT": "root_point"},
        free_joint="root",
        root_marker="ROOT",
    )
    assert quality["root_tracking_rms_m"] == 0
    assert quality["root_trajectory_m"].shape == (2, 3)
    assert "sole_penetration_max_m" not in quality
    decision = evaluate_quality(
        quality,
        QualityThresholds(sole_penetration_max_m=0.01, root_tracking_rms_m=0.01),
    )
    assert decision["status"] == "unknown"
    assert not decision["accepted"]
