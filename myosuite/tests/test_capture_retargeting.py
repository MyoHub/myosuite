# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Real target-model fitting, named exports and nominal-target failure controls."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite.core.subject_model import (
    GroundAlignment,
    ScaleEstimate,
    SubjectModelConfig,
    SubjectModel,
    build_subject_model,
)
from myosuite.integrations.capture.example import FIXTURES, MARKER_SITES, run_example
from myosuite.integrations.capture.markers import (
    MarkerTrial,
    SegmentMeasurement,
    estimate_marker_scales,
)
from myosuite.integrations.capture.retargeting import (
    load_retargeted_clip,
    retarget_markers,
)
from myosuite.physics.marker_ik import qpos_from_marker_targets

pytestmark = pytest.mark.tier1


def _inputs() -> tuple[mujoco.MjSpec, SubjectModel, MarkerTrial]:
    payload = json.loads((FIXTURES / "static_trial.json").read_text())
    scales = estimate_marker_scales(
        MarkerTrial(**payload["trial"]),
        [SegmentMeasurement(**row) for row in payload["measurements"]],
    )
    source = mujoco.MjSpec.from_file(str(FIXTURES / "two_link.xml"))
    subject = build_subject_model(
        source,
        SubjectModelConfig(scales.estimate, ground=GroundAlignment("root", ("sole",))),
    )
    trial = MarkerTrial(
        **json.loads((FIXTURES / "motion_trial.json").read_text())["trial"]
    )
    return source, subject, trial


def test_licensed_scale_align_retarget_export_replay_pipeline_and_bad_control(
    tmp_path: Path,
) -> None:
    report = run_example(tmp_path)
    good = report["scaled_target"]
    bad = report["nominal_fit_replayed_on_scaled_target"]
    assert good["converged_frames"] == good["frames"] == 7
    assert good["marker_max_m"] < 1e-5
    assert good["root_tracking_rms_m"] < 1e-5
    assert good["sole_penetration_max_m"] < 1e-6
    assert report["contact_frames"] == 7
    assert bad["marker_rms_m"] > 0.03
    assert bad["root_tracking_rms_m"] > 0.03
    assert bad["sole_penetration_max_m"] > 0.03
    with np.load(tmp_path / "scaled_target_motion.npz", allow_pickle=False) as archive:
        assert archive["qpos_names"].tolist() == ["root", "hip", "knee"]
        assert np.max(np.abs(archive["qvel"])) > 0.01
        assert np.all(archive["contact_labels"])
        # IK recovered nonzero hip/knee angles from Cartesian marker input.
        np.testing.assert_allclose(
            archive["qpos"][:, 7],
            [-0.1, -0.12, -0.14, -0.16, -0.14, -0.12, -0.1],
            atol=1e-5,
        )
        provenance = json.loads(str(archive["provenance_json"]))
        assert provenance["subject_model"] == report["subject_model"]
        assert provenance["scale_evidence"]["trial"]["units"] == "mm"
    sidecar = json.loads((tmp_path / "scaled_target_motion.json").read_text())
    assert sidecar["quality"]["marker_max_m"] < 1e-5


def test_saved_motion_requires_the_actual_target_model(tmp_path: Path) -> None:
    source, subject, trial = _inputs()
    result = retarget_markers(subject, trial, MARKER_SITES, free_joint="root")
    path = result.save(tmp_path / "motion.npz")
    loaded = load_retargeted_clip(path, subject)
    np.testing.assert_allclose(loaded.qpos, result.clip.qpos)
    nominal = build_subject_model(source, SubjectModelConfig(ScaleEstimate({})))
    with pytest.raises(ValueError, match="different subject"):
        load_retargeted_clip(path, nominal)
    subject.model.geom_size[0, 0] *= 2
    with pytest.raises(ValueError, match="changed after construction"):
        retarget_markers(subject, trial, MARKER_SITES, free_joint="root")


def test_nominal_target_fails_strict_fitting_instead_of_claiming_retargeting() -> None:
    source, _, trial = _inputs()
    nominal = build_subject_model(source, SubjectModelConfig(ScaleEstimate({})))
    with pytest.raises(ValueError, match="IK did not converge"):
        retarget_markers(nominal, trial, MARKER_SITES, free_joint="root")


def test_partial_missing_marker_frame_is_fit_without_imputation() -> None:
    _, subject, trial = _inputs()
    points = np.array(trial.positions)
    points[2, trial.labels.index("PELVIS_SIDE")] = np.nan
    fitted = retarget_markers(
        subject, replace(trial, positions=points), MARKER_SITES, free_joint="root"
    )
    assert fitted.diagnostics["marker_max_m"] < 1e-4
    points[2] = np.nan
    with pytest.raises(ValueError, match="At least one marker"):
        retarget_markers(
            subject, replace(trial, positions=points), MARKER_SITES, free_joint="root"
        )


def test_marker_ik_enforces_scalar_joint_limits() -> None:
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body>
      <joint name="slide" type="slide" axis="1 0 0" range="0 .1"/>
      <geom type="sphere" size=".1"/><site name="marker"/>
    </body></worldbody></mujoco>""")
    result = qpos_from_marker_targets(model, {"marker": [0.2, 0, 0]})
    assert not result.success
    assert 0 <= result.qpos[0] <= 0.1
    assert result.max_marker_error_m >= 0.1 - 1e-9


def test_marker_ik_checks_active_joint_equalities() -> None:
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body><joint name="a" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1"/><site name="ma"/></body>
      <body pos="0 1 0"><joint name="b" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1"/><site name="mb"/></body>
      </worldbody><equality><joint joint1="a" joint2="b"/></equality></mujoco>""")
    good = qpos_from_marker_targets(model, {"ma": [0.1, 0, 0], "mb": [0.1, 1, 0]})
    assert good.success
    assert good.max_equality_error < 1e-6
    bad = qpos_from_marker_targets(model, {"ma": [0.1, 0, 0], "mb": [0.3, 1, 0]})
    assert not bad.success
