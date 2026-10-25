# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Independent native-source contract fixtures; no third-party data assets."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from myosuite.integrations.capture.native_sources import (
    read_addbiomechanics,
    read_bioamass,
    skel_joint_trial,
)
from myosuite.integrations.capture.markers import MarkerTrial
from myosuite.integrations.capture.native_validation import (
    _load_validation_trial,
    validate_native_origins,
)
from myosuite.integrations.capture import native_validation
from myosuite.integrations.capture.trial_io import load_marker_trial, save_marker_trial

pytestmark = pytest.mark.tier1


def test_skel_native_output_uses_bone_order_and_batches_without_asset_loading(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")

    class Model:
        num_q_params, num_betas = 2, 1
        bone_names = ("pelvis", "knee")
        joints_name = ("unrelated_SMPL_order",)

        def buffers(self) -> Any:
            return iter([torch.zeros(1)])

        def __call__(self, poses: Any, betas: Any, trans: Any, **kwargs: Any) -> Any:
            assert kwargs == {"poses_type": "skel", "skelmesh": False}
            assert not torch.is_grad_enabled()
            points = torch.stack(
                (
                    trans,
                    trans + torch.stack((poses[:, 0], betas[:, 0], poses[:, 1]), dim=1),
                ),
                dim=1,
            )
            return SimpleNamespace(joints=points)

    trial = skel_joint_trial(
        Model(),
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        [[1], [1], [1]],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        units="m",
        rotation_to_model=np.eye(3),
        frequency_hz=60,
        source="independently authored parameters",
        model_source="independently authored native model substitute",
        definition_license="Apache-2.0",
        batch_size=2,
        subject_id="authored-skeleton",
    )
    assert trial.labels == ("pelvis", "knee")
    np.testing.assert_allclose(trial.positions[2], [[2, 0, 0], [2.5, 1, 0.6]])
    assert trial.metadata()["native_source"]["joint_order"] == "model.bone_names"
    assert trial.metadata()["native_source"]["frames"] == 3
    json.dumps(trial.metadata(), allow_nan=False)
    assert trial.point_semantics == "model_origin"
    assert trial.subject_id == "authored-skeleton"
    assert (
        load_marker_trial(save_marker_trial(trial, tmp_path / "skel.npz")).metadata()
        == trial.metadata()
    )


def test_addbiomechanics_raw_marker_window_preserves_absence_and_quality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "authored.b3d"
    path.write_bytes(b"authored native-reader substitute")

    class Subject:
        def __init__(self, filename: str) -> None:
            assert filename == str(path)

        def getNumTrials(self) -> int:
            return 1

        def getTrialLength(self, trial: int) -> int:
            return 3

        def readFrames(self, **kwargs: Any) -> list[Any]:
            assert kwargs == {
                "trial": 0,
                "startFrame": 0,
                "numFramesToRead": 2,
                "includeSensorData": True,
                "includeProcessingPasses": False,
                "stride": 2,
            }
            return [
                SimpleNamespace(
                    markerObservations=[("A", [0, 0, 0]), ("B", [0, 0, 0.6])]
                ),
                SimpleNamespace(markerObservations=[("A", [1, 0, 0])]),
            ]

        def getTrialTimestep(self, trial: int) -> float:
            return 0.01

        def getMissingGRF(self, trial: int) -> list[str]:
            return ["notMissingGRF", "manualReview", "unmeasuredExternalForceDetected"]

        def getHeightM(self) -> float:
            return float("nan")

        def getMassKg(self) -> float:
            return 65

        def getHref(self) -> str:
            return "synthetic independent subject"

    monkeypatch.setitem(
        sys.modules,
        "nimblephysics",
        SimpleNamespace(biomechanics=SimpleNamespace(SubjectOnDisk=Subject)),
    )
    trial = read_addbiomechanics(
        path,
        0,
        rotation_to_model=np.eye(3),
        definition_source="authored markers",
        definition_license="Apache-2.0",
        stride=2,
        subject_id="authored-capture",
    )
    assert trial.labels == ("A", "B")
    assert trial.frequency_hz == 50
    assert np.isnan(trial.positions[1, 1]).all()
    assert trial.metadata()["native_source"]["frame_indices"] == [0, 2]
    assert (
        trial.metadata()["native_source"]["missing_grf_reasons"][-1]
        == "unmeasuredExternalForceDetected"
    )
    assert trial.metadata()["native_source"]["source_mass_kg"] == 65
    assert trial.metadata()["native_source"]["source_height_m"] is None
    json.dumps(trial.metadata(), allow_nan=False)
    assert trial.point_semantics == "surface_marker"
    assert trial.subject_id == "authored-capture"
    assert (
        load_marker_trial(save_marker_trial(trial, tmp_path / "markers.npz")).metadata()
        == trial.metadata()
    )


def test_b3d_without_local_source_is_rejected_before_native_import(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="local B3D"):
        read_addbiomechanics(
            tmp_path / "absent.b3d",
            0,
            rotation_to_model=np.eye(3),
            definition_source="authored",
            definition_license="Apache-2.0",
        )


def test_native_opensim_child_joint_origins_honor_degree_motion(tmp_path: Path) -> None:
    opensim = pytest.importorskip("opensim")
    model = opensim.Model()
    first = opensim.Body("upper", 1, opensim.Vec3(0), opensim.Inertia(1))
    second = opensim.Body("lower", 1, opensim.Vec3(0), opensim.Inertia(1))
    hip = opensim.PinJoint(
        "hip",
        model.getGround(),
        opensim.Vec3(0),
        opensim.Vec3(0),
        first,
        opensim.Vec3(0),
        opensim.Vec3(0),
    )
    hip.updCoordinate().setName("theta")
    knee = opensim.WeldJoint(
        "knee",
        first,
        opensim.Vec3(0, -0.5, 0),
        opensim.Vec3(0),
        second,
        opensim.Vec3(0),
        opensim.Vec3(0),
    )
    model.addBody(first)
    model.addJoint(hip)
    model.addBody(second)
    model.addJoint(knee)
    model.finalizeConnections()
    source = tmp_path / "authored.osim"
    model.printToXML(str(source))
    motion = tmp_path / "authored.mot"
    motion.write_text(
        "Coordinates\nnRows=2\nnColumns=2\ninDegrees=yes\nendheader\ntime\ttheta\n0\t0\n0.01\t90\n"
    )
    trial = read_bioamass(
        source,
        motion,
        joint_names=("hip", "knee"),
        rotation_to_model=np.eye(3),
        definition_source="independently authored two-link origins",
        definition_license="Apache-2.0",
        subject_id="authored-opensim",
    )
    np.testing.assert_allclose(
        trial.positions[:, 1], [[0, -0.5, 0], [0.5, 0, 0]], atol=1e-10
    )
    assert trial.frequency_hz == 100
    assert trial.metadata()["native_source"]["motion_in_degrees"]
    assert len(trial.metadata()["native_source"]["model_sha256"]) == 64
    assert trial.point_semantics == "model_origin"
    assert trial.subject_id == "authored-opensim"


def test_native_validation_rejects_an_unrelated_joint_label_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unrelated.npz"
    np.savez(
        path,
        positions=np.zeros((31, 1, 3)),
        labels=np.array(["pelvis"]),
        frequency_hz=120,
    )
    with pytest.raises(ValueError, match="missing labels"):
        validate_native_origins(
            path, source_label="authored test", definition_license="Apache-2.0"
        )


def test_native_validation_keeps_calibration_frames_out_of_evaluation(
    tmp_path: Path,
) -> None:
    labels = ("pelvis",) + tuple(
        f"{body}_{side}"
        for side in "rl"
        for body in ("femur", "tibia", "talus", "calcn", "toes")
    )
    path = tmp_path / "authored.npz"
    np.savez(
        path,
        positions=np.zeros((31, len(labels), 3)),
        labels=np.array(labels),
        frequency_hz=120,
    )
    with pytest.raises(ValueError, match="disjoint"):
        validate_native_origins(
            path,
            source_label="authored test",
            definition_license="Apache-2.0",
            heldout_frames=(0, 10, 20),
        )


def test_versioned_validation_bridge_preserves_units_transform_semantics_and_identity(
    tmp_path: Path,
) -> None:
    trial = MarkerTrial(
        ("A", "B"),
        np.array([[[0, 0, 0], [0, 40, 0]]]),
        "cm",
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        "authored source",
        translation_to_model_m=(0.1, 0.2, 0.3),
        frequency_hz=100,
        point_semantics="model_origin",
        subject_id="authored-subject",
        definition_source="authored definitions",
        definition_license="Apache-2.0",
    )
    path = save_marker_trial(trial, tmp_path / "trial.npz")
    restored, reasons = _load_validation_trial(
        path,
        source_label="public label",
        definition_license="public terms",
        point_semantics="model_origin",
    )
    np.testing.assert_allclose(restored.in_model_frame(), trial.in_model_frame())
    assert restored.metadata() == trial.metadata()
    assert reasons is None
    with pytest.raises(ValueError, match="point_semantics"):
        _load_validation_trial(
            path,
            source_label="public label",
            definition_license="public terms",
            point_semantics="surface_marker",
        )


def test_lateral_shank_cli_accepts_versioned_and_legacy_inputs_without_native_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mujoco

    spec = mujoco.MjSpec.from_string("""<mujoco><worldbody>
      <body name="tibia_r"><geom type="sphere" size=".03"/>
        <site name="LFC_r"/><site name="LMAL_r" pos="0 -.5 0"/></body>
      <body name="tibia_l" pos=".2 0 0"><geom type="sphere" size=".03"/>
        <site name="LFC_l"/><site name="LMAL_l" pos="0 -.5 0"/></body>
      </worldbody></mujoco>""")

    @contextmanager
    def environment(name: str) -> Any:
        assert name == "myoLegWalk-v0"
        yield SimpleNamespace(unwrapped=SimpleNamespace(_mj_spec=spec))

    monkeypatch.setattr(native_validation, "make_env", environment)
    points = np.array([[[0, 0, 0], [0, -600, 0], [200, 0, 0], [200, -400, 0]]] * 14)
    trial = MarkerTrial(
        ("RxLatCon", "RxLatMal", "LxLatCon", "LxLatMal"),
        points,
        "mm",
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        "authored markers",
        translation_to_model_m=(0.1, -0.2, 0.3),
        frequency_hz=100,
        point_semantics="surface_marker",
        definition_license="Apache-2.0",
    )
    versioned = save_marker_trial(trial, tmp_path / "versioned.npz")
    legacy = tmp_path / "legacy.npz"
    np.savez(
        legacy,
        positions=trial.in_model_frame(),
        labels=trial.labels,
        frequency_hz=100,
        missing_grf=np.zeros(len(points), dtype=int),
    )
    reports = []
    for source in (versioned, legacy):
        output = tmp_path / (source.stem + ".json")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "native_validation",
                "--trial",
                str(source),
                "--protocol",
                "lateral-shank",
                "--source-label",
                "authored-test",
                "--definition-license",
                "Apache-2.0",
                "--output",
                str(output),
            ],
        )
        native_validation.main()
        reports.append(json.loads(output.read_text()))
    for report in reports:
        assert report["heldout_frame_count"] == 4
        assert report["estimated_scale_axes"] == 2
        assert report["nominal_heldout_span_rms_m"] == pytest.approx(0.1)
        assert report["scaled_heldout_span_rms_m"] < 1e-12
    assert reports[0]["source_grf_status_available"] is False
    assert "source_grf_valid_frame_count" not in reports[0]
    assert reports[1]["source_grf_status_available"] is True
    assert reports[1]["source_grf_valid_frame_count"] == 14
