# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Optional native readers for licensed skeletal fits and measured motion."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from myosuite.integrations.capture.markers import MarkerTrial


@dataclass(frozen=True, kw_only=True)
class _SourceTrial(MarkerTrial):
    _source_evidence: str

    def metadata(self) -> dict[str, Any]:
        """Retain native-reader and file provenance through scale estimation."""
        return {
            **super().metadata(),
            "native_source": json.loads(self._source_evidence),
        }


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must identify the source and its terms.")


def _known_positive(value: float) -> float | None:
    number = float(value)
    return number if np.isfinite(number) and number > 0 else None


def _window(length: int, start: int, count: int | None, stride: int) -> np.ndarray:
    for name, value in (("start_frame", start), ("stride", stride)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer.")
    if not 0 <= start < length or stride <= 0:
        raise ValueError(
            "Frame window must start inside the trial with positive stride."
        )
    available = np.arange(start, length, stride)
    if count is not None:
        if (
            isinstance(count, bool)
            or not isinstance(count, Integral)
            or not 0 < count <= len(available)
        ):
            raise ValueError("num_frames must be positive and fit inside the trial.")
        available = available[:count]
    return available


def skel_joint_trial(
    model: Any,
    poses: ArrayLike,
    betas: ArrayLike,
    translations: ArrayLike,
    *,
    units: Literal["m", "cm", "mm"],
    rotation_to_model: ArrayLike,
    frequency_hz: float,
    source: str,
    model_source: str,
    definition_license: str,
    translation_to_model_m: ArrayLike = (0.0, 0.0, 0.0),
    batch_size: int = 64,
    confidence: float | None = None,
    uncertainty_m: float | None = None,
    subject_id: str | None = None,
) -> MarkerTrial:
    """Evaluate an externally loaded SKEL model without distributing its assets.

    Args:
        model: User-loaded upstream SKEL model, on the desired Torch device.
        poses: Native SKEL pose parameters, shaped (frames, model.num_q_params).
        betas: Shape parameters, shaped (frames, model.num_betas).
        translations: Source-world translations, shaped (frames, 3).
        units: Explicit units of the upstream joint coordinates.
        rotation_to_model: Calibrated proper source-to-target world rotation.
        frequency_hz: Sampling frequency of the supplied parameter sequence.
        source: Parameter sequence identifier and preferably a file hash.
        model_source: Model/version/asset identifier and preferably a file hash.
        definition_license: Terms applying to the model-derived joint definitions.
        translation_to_model_m: Target-world translation in metres.
        batch_size: Positive maximum number of frames per native evaluation.
        confidence: Optional source-reported confidence.
        uncertainty_m: Optional source-reported coordinate uncertainty.
        subject_id: Optional caller-supplied identity retained only as provenance.

    Returns:
        A marker trial labelled in the upstream BSM bone order. SKEL's separate
        ``joints_name`` field can contain SMPL-order labels and is not used.

    Torch and SKEL model loading remain optional and outside the core. No
    pickle, model file, vertex index list or parameter sequence is loaded here.
    """
    _text(model_source, "model_source")
    _text(definition_license, "definition_license")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, Integral)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    values = [np.asarray(value, dtype=float) for value in (poses, betas, translations)]
    count = len(values[0]) if values[0].ndim else 0
    expected = ((count, model.num_q_params), (count, model.num_betas), (count, 3))
    if not count or any(
        value.shape != shape or not np.all(np.isfinite(value))
        for value, shape in zip(values, expected)
    ):
        raise ValueError(
            "SKEL parameters need finite, matching frame and parameter shapes."
        )
    labels = tuple(model.bone_names)
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "SKEL evaluation requires the user's optional Torch/SKEL setup."
        ) from exc
    device = next(model.buffers()).device
    positions = []
    with torch.no_grad():
        for start in range(0, count, batch_size):
            batch = [
                torch.as_tensor(
                    v[start : start + batch_size], dtype=torch.float32, device=device
                )
                for v in values
            ]
            output = model(*batch, poses_type="skel", skelmesh=False)
            points = output.joints.detach().cpu().numpy()
            if points.shape != (len(batch[0]), len(labels), 3):
                raise ValueError(
                    "SKEL output rows do not match the model's bone_names order."
                )
            positions.append(points)
    return _SourceTrial(
        labels=labels,
        positions=np.concatenate(positions),
        units=units,
        rotation_to_model=rotation_to_model,
        source=source,
        translation_to_model_m=translation_to_model_m,
        frequency_hz=frequency_hz,
        confidence=confidence,
        uncertainty_m=uncertainty_m,
        definition_source=model_source,
        definition_license=definition_license,
        point_semantics="model_origin",
        subject_id=subject_id,
        _source_evidence=json.dumps(
            {
                "adapter": "upstream-skel-joints",
                "model_source": model_source,
                "joint_order": "model.bone_names",
                "frames": count,
            }
        ),
    )


def read_addbiomechanics(
    path: str | Path,
    trial: int,
    *,
    rotation_to_model: ArrayLike,
    definition_source: str,
    definition_license: str,
    start_frame: int = 0,
    num_frames: int | None = None,
    stride: int = 1,
    translation_to_model_m: ArrayLike = (0.0, 0.0, 0.0),
    subject_id: str | None = None,
) -> MarkerTrial:
    """Read raw measured markers through the optional official Nimble B3D reader.

    The marker set is the union observed in the selected frames; a marker absent
    throughout that window has no inferred label. Missing observations stay NaN.
    Subject anthropometry and missing-GRF reasons are retained as source evidence,
    without becoming segment factors or a contact/dynamics validity guarantee.
    """
    _text(definition_source, "definition_source")
    _text(definition_license, "definition_license")
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size < 8:
        raise ValueError("A nonempty local B3D subject file is required.")
    try:
        import nimblephysics as nimble
    except ImportError as exc:
        raise ImportError(
            "B3D import requires the optional official nimblephysics reader on a supported platform."
        ) from exc
    subject = nimble.biomechanics.SubjectOnDisk(str(path))
    if (
        isinstance(trial, bool)
        or not isinstance(trial, Integral)
        or not 0 <= trial < subject.getNumTrials()
    ):
        raise ValueError("trial must name an existing B3D trial index.")
    indices = _window(subject.getTrialLength(trial), start_frame, num_frames, stride)
    frames = subject.readFrames(
        trial=trial,
        startFrame=int(indices[0]),
        numFramesToRead=len(indices),
        includeSensorData=True,
        includeProcessingPasses=False,
        stride=stride,
    )
    if len(frames) != len(indices):
        raise ValueError("The native B3D reader returned an incomplete frame window.")
    observed = [dict(frame.markerObservations) for frame in frames]
    labels = tuple(sorted({label for frame in observed for label in frame}))
    positions = np.full((len(frames), len(labels), 3), np.nan)
    for frame, markers in enumerate(observed):
        for column, label in enumerate(labels):
            if label in markers:
                positions[frame, column] = np.asarray(markers[label]).reshape(3)
    timestep = float(subject.getTrialTimestep(trial))
    if not np.isfinite(timestep) or timestep <= 0:
        raise ValueError("B3D trial timestep must be finite and positive.")
    reasons = list(subject.getMissingGRF(trial))
    evidence = {
        "adapter": "nimble-subject-on-disk",
        "sha256": _digest(path),
        "trial": int(trial),
        "frame_indices": indices.tolist(),
        "source_timestep_s": timestep,
        "source_height_m": _known_positive(subject.getHeightM()),
        "source_mass_kg": _known_positive(subject.getMassKg()),
        "anthropometry_semantics": "source-reported; nonfinite/nonpositive means unknown",
        "missing_grf_reasons": [
            str(reasons[index]) if index < len(reasons) else None for index in indices
        ],
        "missing_grf_semantics": "source-reported; absent reason is unknown",
        "source_url": str(subject.getHref()),
        "marker_labels": "union observed in selected frames",
    }
    return _SourceTrial(
        labels=labels,
        positions=positions,
        units="m",
        rotation_to_model=rotation_to_model,
        source=str(path),
        translation_to_model_m=translation_to_model_m,
        frequency_hz=1 / (timestep * stride),
        definition_source=definition_source,
        definition_license=definition_license,
        point_semantics="surface_marker",
        subject_id=subject_id,
        _source_evidence=json.dumps(evidence),
    )


def read_bioamass(
    model_path: str | Path,
    motion_path: str | Path,
    *,
    joint_names: Sequence[str] = (),
    body_names: Sequence[str] = (),
    rotation_to_model: ArrayLike,
    definition_source: str,
    definition_license: str,
    start_frame: int = 0,
    num_frames: int | None = None,
    stride: int = 1,
    translation_to_model_m: ArrayLike = (0.0, 0.0, 0.0),
    subject_id: str | None = None,
) -> MarkerTrial:
    """Evaluate local BioAmass/OpenSim motion using the native OpenSim library.

    Select either joint child-frame origins with ``joint_names`` or explicit
    body-frame origins with ``body_names``. Coordinate values
    are applied by the source model's names and motion types; they are never
    renamed into a MyoSuite qpos vector. Cartesian outputs can be mapped to a
    constructed target with explicit anatomical correspondences and calibration.
    Models, motion files and native dependencies stay in the user's workflow.
    """
    _text(definition_source, "definition_source")
    _text(definition_license, "definition_license")
    if bool(joint_names) == bool(body_names):
        raise ValueError("Select exactly one of joint_names and body_names.")
    model_path, motion_path = Path(model_path).resolve(), Path(motion_path).resolve()
    try:
        import opensim
    except ImportError as exc:
        raise ImportError(
            "BioAmass forward kinematics requires the optional native OpenSim Python bindings."
        ) from exc
    model = opensim.Model(str(model_path))
    storage = opensim.Storage(str(motion_path))
    state = model.initSystem()
    indices = _window(storage.getSize(), start_frame, num_frames, stride)
    columns = storage.getColumnLabels()
    names = [columns.get(i) for i in range(1, columns.getSize())]
    coordinates = model.getCoordinateSet()
    source_coordinates = [coordinates.get(name) for name in names]
    labels = tuple(joint_names or body_names)
    frames = (
        [model.getJointSet().get(name).getChildFrame() for name in labels]
        if joint_names
        else [model.getBodySet().get(name) for name in labels]
    )
    positions, times = [], []
    for index in indices:
        frame = storage.getStateVector(int(index))
        values = frame.getData()
        for column, coordinate in enumerate(source_coordinates):
            value = values.get(column)
            if (
                storage.isInDegrees()
                and coordinate.getMotionType() == opensim.Coordinate.Rotational
            ):
                value = np.deg2rad(value)
            coordinate.setValue(state, float(value), False)
        state.setTime(frame.getTime())
        model.assemble(state)
        model.realizePosition(state)
        positions.append(
            [
                [source_frame.getPositionInGround(state).get(axis) for axis in range(3)]
                for source_frame in frames
            ]
        )
        times.append(float(frame.getTime()))
    all_times = np.array(
        [storage.getStateVector(i).getTime() for i in range(storage.getSize())]
    )
    if (
        len(all_times) < 2
        or not np.all(np.isfinite(all_times))
        or np.any(np.diff(all_times) <= 0)
    ):
        raise ValueError("OpenSim motion requires finite, increasing sampled times.")
    period = float((all_times[-1] - all_times[0]) / (len(all_times) - 1))
    if not np.allclose(
        all_times - all_times[0],
        np.arange(len(all_times)) * period,
        rtol=0,
        atol=max(1e-6, 0.01 * period),
    ):
        raise ValueError("OpenSim motion times must be uniformly sampled.")
    evidence = {
        "adapter": "opensim-cartesian-frame-origins",
        "point_definition": "joint child frame" if joint_names else "body origin",
        "model_sha256": _digest(model_path),
        "motion_sha256": _digest(motion_path),
        "frame_indices": indices.tolist(),
        "source_times_s": times,
        "joint_names": list(labels),
        "motion_in_degrees": bool(storage.isInDegrees()),
        "frequency_method": "uniform full-file timestamp span; no resampling",
        "omitted_coordinates": [
            coordinates.get(i).getName()
            for i in range(coordinates.getSize())
            if coordinates.get(i).getName() not in names
        ],
        "omitted_coordinate_policy": "model defaults, then native assembly constraints",
    }
    return _SourceTrial(
        labels=labels,
        positions=np.asarray(positions),
        units="m",
        rotation_to_model=rotation_to_model,
        source=str(motion_path),
        translation_to_model_m=translation_to_model_m,
        frequency_hz=1 / (period * stride),
        definition_source=definition_source,
        definition_license=definition_license,
        point_semantics="model_origin",
        subject_id=subject_id,
        _source_evidence=json.dumps(evidence),
    )
