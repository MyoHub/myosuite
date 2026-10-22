# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Run the licensed scale -> align -> retarget -> export -> replay example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite.core.subject_model import (
    GroundAlignment,
    ScaleEstimate,
    SubjectModelConfig,
    build_subject_model,
)
from myosuite.integrations.capture.markers import (
    MarkerTrial,
    SegmentMeasurement,
    estimate_marker_scales,
)
from myosuite.integrations.capture.retargeting import (
    load_retargeted_clip,
    motion_diagnostics,
    retarget_markers,
)

FIXTURES = Path(__file__).with_name("fixtures")
MARKER_SITES = {
    "ROOT": "root_marker",
    "PELVIS_FRONT": "pelvis_front",
    "PELVIS_SIDE": "pelvis_side",
    "KNEE": "knee_marker",
    "ANKLE": "ankle_marker",
    "TOE": "toe_marker",
    "HEEL": "heel_marker",
}


def run_example(output_dir: str | Path) -> dict[str, Any]:
    """Run analytic input through target-specific IK and a deliberately bad control."""
    output = Path(output_dir)
    static = json.loads((FIXTURES / "static_trial.json").read_text())
    estimate = estimate_marker_scales(
        MarkerTrial(**static["trial"]),
        [SegmentMeasurement(**row) for row in static["measurements"]],
    )
    source = mujoco.MjSpec.from_file(str(FIXTURES / "two_link.xml"))
    ground = GroundAlignment("root", ("sole",))
    subject = build_subject_model(
        source, SubjectModelConfig(estimate.estimate, ground=ground)
    )
    motion = MarkerTrial(
        **json.loads((FIXTURES / "motion_trial.json").read_text())["trial"]
    )
    fitted = retarget_markers(
        subject,
        motion,
        MARKER_SITES,
        free_joint="root",
        sole_geoms=("sole",),
        root_marker="ROOT",
        scale_evidence=estimate.evidence,
    )
    path = fitted.save(output / "scaled_target_motion.npz")
    replay = load_retargeted_clip(path, subject)
    data = mujoco.MjData(subject.model)
    for frame in replay.qpos:
        data.qpos[:] = frame
        mujoco.mj_forward(subject.model, data)
        if not np.all(np.isfinite(data.site_xpos)):
            raise RuntimeError("Nonfinite replay state.")
    nominal = build_subject_model(
        source, SubjectModelConfig(ScaleEstimate({}), ground=ground)
    )
    bad = retarget_markers(
        nominal,
        motion,
        MARKER_SITES,
        free_joint="root",
        sole_geoms=("sole",),
        root_marker="ROOT",
        strict=False,
    )
    bad_replay = motion_diagnostics(
        subject.model,
        bad.clip,
        motion,
        MARKER_SITES,
        free_joint="root",
        sole_geoms=("sole",),
        root_marker="ROOT",
    )
    quality = fitted.diagnostics
    if quality["marker_max_m"] > 1e-4 or quality["sole_penetration_max_m"] > 1e-5:
        raise RuntimeError(
            "Scaled-target retargeting failed the fixture's geometric tolerances."
        )
    if bad_replay["marker_rms_m"] < 0.01:
        raise RuntimeError(
            "Nominal-target negative control did not detect the morphology error."
        )
    report = {
        "fixture_license": "Apache-2.0",
        "scale_estimate": estimate.estimate.to_dict(),
        "scaled_target": {
            key: value for key, value in quality.items() if np.isscalar(value)
        },
        "nominal_fit_replayed_on_scaled_target": {
            key: value for key, value in bad_replay.items() if np.isscalar(value)
        },
        "contact_frames": int(np.count_nonzero(quality["contact_labels"])),
        "clip": str(path.resolve()),
        "subject_model": subject.metadata,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    """Run the fixture without a viewer, network access or restricted assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/marker_retargeting")
    )
    args = parser.parse_args()
    print(json.dumps(run_example(args.output), indent=2))


if __name__ == "__main__":
    main()
