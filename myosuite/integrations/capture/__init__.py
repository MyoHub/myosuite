"""Optional capture adapters and marker retargeting for subject models."""

from myosuite.integrations.capture.markers import (
    MarkerScaleResult,
    MarkerTrial,
    SegmentMeasurement,
    estimate_marker_scales,
    read_c3d,
    virtual_marker_trial,
)
from myosuite.integrations.capture.vertices import VertexMarkerSet, vertex_marker_trial
from myosuite.integrations.capture.trc import read_trc
from myosuite.integrations.capture.native_sources import (
    read_addbiomechanics,
    read_bioamass,
    skel_joint_trial,
)
from myosuite.integrations.capture.landmarks import read_named_landmarks
from myosuite.integrations.capture.landmark_scaling import (
    LandmarkCorrespondences,
    fit_landmark_scales,
)
from myosuite.integrations.capture.protocols import TargetProtocol
from myosuite.integrations.capture.reporting import (
    QualityThresholds,
    SourceAttribution,
    aggregate_validation_report,
    evaluate_quality,
    write_aggregate_report,
)
from myosuite.integrations.capture.trial_io import load_marker_trial, save_marker_trial

__all__ = [
    "LandmarkCorrespondences",
    "TargetProtocol",
    "QualityThresholds",
    "SourceAttribution",
    "aggregate_validation_report",
    "evaluate_quality",
    "write_aggregate_report",
    "read_named_landmarks",
    "fit_landmark_scales",
    "load_marker_trial",
    "save_marker_trial",
    "MarkerScaleResult",
    "MarkerTrial",
    "SegmentMeasurement",
    "VertexMarkerSet",
    "estimate_marker_scales",
    "read_c3d",
    "read_trc",
    "read_addbiomechanics",
    "read_bioamass",
    "skel_joint_trial",
    "virtual_marker_trial",
    "vertex_marker_trial",
]
