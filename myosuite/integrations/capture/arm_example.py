# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Independent analytic fixed-base arm: scale, construct, fit, export and replay."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite.core.subject_model import ScaleEstimate
from myosuite.integrations.capture.markers import MarkerTrial, SegmentMeasurement
from myosuite.integrations.capture.protocols import TargetProtocol
from myosuite.integrations.capture.retargeting import (
    load_retargeted_clip,
    motion_diagnostics,
)
from myosuite.integrations.capture.trial_io import load_marker_trial, save_marker_trial

ARM_XML = """<mujoco model="authored_fixed_arm">
  <compiler angle="radian"/>
  <worldbody><body name="upper_segment" pos=".1 -.2 .5">
    <joint name="shoulder_yaw" axis="0 0 1" range="-1.5 1.5"/>
    <geom type="capsule" fromto="0 0 0 .30 0 0" size=".02"/>
    <site name="shoulder_point" pos="0 0 0"/>
    <site name="upper_aux" pos=".15 .04 0"/>
    <body name="lower_segment" pos=".30 0 0">
      <joint name="elbow_bend" axis="0 0 1" range="-1.5 1.5"/>
      <geom type="capsule" fromto="0 0 0 .25 0 0" size=".015"/>
      <site name="elbow_point" pos="0 0 0"/>
      <site name="lower_aux" pos=".125 .03 0"/>
      <site name="wrist_point" pos=".25 0 0"/>
    </body>
  </body></worldbody>
</mujoco>"""


def _analytic_points(angles: np.ndarray) -> np.ndarray:
    first, second = angles[:, 0], angles.sum(axis=1)
    first_axis = np.column_stack((np.cos(first), np.sin(first), np.zeros(len(first))))
    second_axis = np.column_stack(
        (np.cos(second), np.sin(second), np.zeros(len(first)))
    )
    first_normal = np.column_stack(
        (-np.sin(first), np.cos(first), np.zeros(len(first)))
    )
    second_normal = np.column_stack(
        (-np.sin(second), np.cos(second), np.zeros(len(first)))
    )
    shoulder = np.broadcast_to([0.1, -0.2, 0.5], first_axis.shape)
    elbow = shoulder + 0.36 * first_axis
    wrist = elbow + 0.20 * second_axis
    return np.stack(
        (
            shoulder,
            elbow,
            wrist,
            shoulder + 0.18 * first_axis + 0.04 * first_normal,
            elbow + 0.10 * second_axis + 0.03 * second_normal,
        ),
        axis=1,
    )


def write_example_inputs(output_dir: str | Path) -> dict[str, Any]:
    """Write Apache-2.0 toy model/protocol/trial inputs for the generic workflow.

    Observations are evaluated from analytic link equations, independently of
    target forward kinematics. These fictional points represent no captured
    person, body-model topology or third-party anatomical definition.
    """
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "arm_model.xml"
    model_path.write_text(ARM_XML, encoding="utf-8")
    protocol = TargetProtocol(
        "authored-fixed-arm",
        {
            "SHOULDER": "shoulder_point",
            "ELBOW": "elbow_point",
            "WRIST": "wrist_point",
            "UPPER_AUX": "upper_aux",
            "LOWER_AUX": "lower_aux",
        },
        (
            SegmentMeasurement("upper_segment", ("SHOULDER", "ELBOW"), 0.30, axis="x"),
            SegmentMeasurement("lower_segment", ("ELBOW", "WRIST"), 0.25, axis="x"),
        ),
        joint_names=("shoulder_yaw", "elbow_bend"),
        point_semantics="anatomical_landmark",
    )
    angles = np.array(
        [[0, 0]] * 3
        + [
            [-0.2, 0.35],
            [-0.1, 0.45],
            [0, 0.55],
            [0.1, 0.5],
            [0.2, 0.4],
            [0.15, 0.3],
            [0.05, 0.2],
        ]
    )
    points = _analytic_points(angles)
    points[5, 2] = np.nan
    trial = MarkerTrial(
        tuple(protocol.marker_sites),
        points * 1000,
        "mm",
        np.eye(3),
        "independent analytic fixed-base arm equations",
        frequency_hz=100,
        definition_source="MyoSuite-authored fictional arm points",
        definition_license="Apache-2.0",
        point_semantics="anatomical_landmark",
        subject_id="synthetic-arm",
    )
    return {
        "model": str(model_path),
        "protocol": str(protocol.save(output / "protocol.json")),
        "trial": str(save_marker_trial(trial, output / "trial.npz")),
        "calibration_frames": [0, 1, 2],
        "heldout_frames": list(range(3, 10)),
        "known_factors": {"upper_segment": [1.2, 1, 1], "lower_segment": [0.8, 1, 1]},
    }


def run_example(output_dir: str | Path) -> dict[str, Any]:
    """Verify construction and retargeting with a nominal-fit negative control."""
    inputs = write_example_inputs(output_dir)
    protocol = TargetProtocol.load(inputs["protocol"])
    trial = load_marker_trial(inputs["trial"])
    spec = mujoco.MjSpec.from_file(inputs["model"])
    protocol.validate_against_model(spec.compile(), trial)
    estimate = protocol.estimate_scales(trial, inputs["calibration_frames"])
    subject = protocol.build_subject(spec, estimate)
    motion = replace(trial, positions=trial.positions[inputs["heldout_frames"]])
    fitted = protocol.retarget(subject, motion, scale_evidence=estimate.evidence)
    path = fitted.save(Path(output_dir) / "arm_motion.npz")
    replay = load_retargeted_clip(path, subject)
    replay_quality = motion_diagnostics(
        subject.model, replay, motion, protocol.marker_sites
    )
    nominal = protocol.build_subject(spec, ScaleEstimate({}))
    control = protocol.retarget(nominal, motion, strict=False)
    bad = motion_diagnostics(subject.model, control.clip, motion, protocol.marker_sites)
    for body, expected in inputs["known_factors"].items():
        if not np.allclose(estimate.estimate.factors[body], expected, atol=1e-10):
            raise RuntimeError("The analytic arm's known factors were not recovered.")
    if replay_quality["marker_max_m"] > 1e-4 or bad["marker_rms_m"] < 0.005:
        raise RuntimeError(
            "Arm replay accuracy or the nominal negative control failed."
        )
    report = {
        "schema": 1,
        "fixture_license": "Apache-2.0",
        "inputs": inputs,
        "root_mode": fitted.provenance["root_mode"],
        "scaled_target": {
            key: fitted.diagnostics[key]
            for key in ("marker_rms_m", "marker_max_m", "converged_frames", "frames")
        },
        "replayed_target": {
            key: replay_quality[key] for key in ("marker_rms_m", "marker_max_m")
        },
        "nominal_fit_replayed_on_scaled_target": {
            key: bad[key] for key in ("marker_rms_m", "marker_max_m")
        },
        "clip": str(path.resolve()),
    }
    (Path(output_dir) / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    """Run without a viewer, registered environment or externally licensed assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/arm_retargeting"))
    args = parser.parse_args()
    print(json.dumps(run_example(args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
