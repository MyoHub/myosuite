# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Named anatomy landmarks and opt-in Keast tibia-fibula reference data."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from myosuite.integrations.capture.markers import MarkerScaleResult, MarkerTrial

PALPABLE_LANDMARKS = ("LC", "MC", "LM", "MM", "TT", "FH")
_MANIFEST = json.loads(
    files("myosuite.integrations")
    .joinpath("keast_source_manifest.json")
    .read_text("utf-8")
)
_MEAN_PATH = "ShapeModels/tibia-fibula/tibiaFibulaShapeModel_mean.csv"


@dataclass(frozen=True)
class _LandmarkTrial(MarkerTrial):
    _evidence_json: str = "{}"

    def metadata(self) -> dict[str, Any]:
        return {**super().metadata(), "landmark_csv": json.loads(self._evidence_json)}


def read_named_landmarks(
    path: str | Path,
    *,
    units: Literal["m", "cm", "mm"],
    rotation_to_model: ArrayLike,
    source: str,
    definition_source: str,
    definition_license: str,
    translation_to_model_m: ArrayLike = (0.0, 0.0, 0.0),
    uncertainty_m: float | None = None,
) -> MarkerTrial:
    """Read headerless ``label,x,y,z`` anatomy coordinates with explicit units.

    Empty coordinates and NaN remain missing. The caller supplies anatomical
    definitions, their source-reported terms and a calibrated proper rotation.
    File identity is retained; no joint centres or landmark labels are inferred.
    """
    for name, value in (
        ("definition_source", definition_source),
        ("definition_license", definition_license),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be supplied.")
    path = Path(path).resolve()
    raw = path.read_bytes()
    labels, positions = [], []
    for row in csv.reader(io.StringIO(raw.decode("utf-8-sig"))):
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) != 4:
            raise ValueError("Landmark CSV rows must contain label,x,y,z.")
        labels.append(row[0].strip())
        try:
            positions.append(
                [float(value) if value.strip() else np.nan for value in row[1:]]
            )
        except ValueError as exc:
            raise ValueError("Landmark coordinates must be numeric or empty.") from exc
    return _LandmarkTrial(
        labels=tuple(labels),
        positions=np.asarray(positions)[None],
        units=units,
        rotation_to_model=rotation_to_model,
        source=source,
        translation_to_model_m=translation_to_model_m,
        uncertainty_m=uncertainty_m,
        definition_source=definition_source,
        definition_license=definition_license,
        _evidence_json=json.dumps(
            {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
        ),
    )


def tibial_landmark_frame(trial: MarkerTrial) -> tuple[np.ndarray, np.ndarray]:
    """Return reference origin and proper basis from explicit right-tibia landmarks.

    Origin is the condyle midpoint, +y points proximally from the malleolar
    midpoint, +x points toward the tibial tuberosity after removing its y
    component, and +z completes a right-handed frame. Returned columns are
    axes in the trial's calibrated metre frame; local points are
    ``(points - origin) @ basis``. This protocol is not automatically a target
    MuJoCo body's local frame; callers must calibrate that transfer separately.
    """
    if len(trial.positions) != 1 or not set(PALPABLE_LANDMARKS) <= set(trial.labels):
        raise ValueError("A single trial with LC, MC, LM, MM, TT and FH is required.")
    points = dict(zip(trial.labels, trial.in_model_frame()[0]))
    if not np.all(np.isfinite([points[label] for label in PALPABLE_LANDMARKS])):
        raise ValueError("The anatomical frame requires finite palpable landmarks.")
    origin = (points["LC"] + points["MC"]) / 2
    y = origin - (points["LM"] + points["MM"]) / 2
    if np.linalg.norm(y) <= 1e-8:
        raise ValueError("Condyle and malleolar midpoints must be distinct.")
    y /= np.linalg.norm(y)
    x = points["TT"] - origin
    x -= np.dot(x, y) * y
    if np.linalg.norm(x) <= 1e-8:
        raise ValueError("Tuberosity must define a distinct anterior direction.")
    x /= np.linalg.norm(x)
    basis = np.column_stack((x, y, np.cross(x, y)))
    if np.dot(points["LC"] - points["MC"], basis[:, 2]) <= 0:
        raise ValueError("Landmarks do not follow this right-tibia frame convention.")
    return origin, basis


def resample_tibial_borders(
    trial: MarkerTrial,
    fractions: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> dict[str, np.ndarray]:
    """Interpolate anatomically matched positions on digitized border polylines.

    Raw T1/T2/... and F1/F2/... are sampling labels, not corresponding locations
    across subjects. This protocol defines T fractions from MC to MM and F
    fractions from FH to LM along the subject's proximal tibial axis. It uses
    the observed piecewise-linear borders without extrapolation or surface
    snapping. Results are in the trial's calibrated world metre frame.
    """
    if (
        not fractions
        or len(set(fractions)) != len(fractions)
        or any(
            not np.isfinite(fraction) or not 0 < fraction < 1 for fraction in fractions
        )
    ):
        raise ValueError("fractions must be unique finite values between zero and one.")
    _, basis = tibial_landmark_frame(trial)
    axis = basis[:, 1]
    points = dict(zip(trial.labels, trial.in_model_frame()[0]))
    result = {}
    for prefix, proximal, distal in (("T", "MC", "MM"), ("F", "FH", "LM")):
        names = sorted(
            (
                label
                for label in trial.labels
                if re.fullmatch(prefix + r"[0-9]+", label)
            ),
            key=lambda label: int(label[1:]),
        )
        if len(names) < 2:
            raise ValueError(
                f"At least two digitized {prefix} border points are required."
            )
        border = np.array([points[label] for label in names])
        heights = border @ axis
        if not np.all(np.isfinite(border)) or not np.all(np.diff(heights) < 0):
            raise ValueError(
                "Border points must be finite and ordered proximal to distal."
            )
        start, end = float(points[proximal] @ axis), float(points[distal] @ axis)
        for fraction in fractions:
            height = start + fraction * (end - start)
            if not heights[-1] <= height <= heights[0]:
                raise ValueError(
                    "Requested fraction lies outside the digitized border."
                )
            result[f"{prefix}{fraction:g}"] = np.array(
                [
                    np.interp(height, heights[::-1], border[::-1, coordinate])
                    for coordinate in range(3)
                ]
            )
    return result


@dataclass(frozen=True)
class KeastReference:
    """Hash-pinned local landmarks from the Keast tibia-fibula archive v4.

    Args:
        root: Local cache parent. Source data are never part of the package.
    """

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    @property
    def data_root(self) -> Path:
        """Return the local directory for this exact source version."""
        return self.root / "keast-tibia-fibula-v4"

    def prepare(self, archive_path: str | Path | None = None) -> Path:
        """Download or verify an archive, then extract only the landmark CSVs.

        This explicit opt-in may download 550.6 MB; extracted landmarks need
        under 40 KB. A supplied archive is reused without copying. Checksums
        precede extraction and only fixed manifest members are written.
        """
        if not self.data_root.resolve().is_relative_to(self.root):
            raise ValueError("Source cache directory leaves the selected root.")
        archive = (
            Path(archive_path)
            if archive_path is not None
            else self.root / "SSM-tibia-V3.zip"
        )
        if not archive.is_file():
            if archive_path is not None:
                raise FileNotFoundError(archive)
            self.root.mkdir(parents=True, exist_ok=True)
            partial = archive.with_suffix(".zip.part")
            with (
                urllib.request.urlopen(_MANIFEST["archive_url"]) as response,
                partial.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            partial.replace(archive)
        with archive.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if (
            archive.stat().st_size != _MANIFEST["archive_bytes"]
            or digest.hexdigest() != _MANIFEST["archive_sha256"]
        ):
            raise ValueError("Keast archive checksum or size does not match version 4.")
        with zipfile.ZipFile(archive) as source:
            for relative, expected in _MANIFEST["files"].items():
                data = source.read(relative)
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError(
                        f"Keast archive member checksum differs: {relative}"
                    )
                target = (self.data_root / relative).resolve()
                if not target.is_relative_to(self.data_root.resolve()):
                    raise ValueError("Landmark member leaves the source cache.")
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.read_bytes() != data:
                    raise ValueError(f"Local source data were changed: {target}")
                target.write_bytes(data)
        return self.data_root

    def read_landmarks(
        self,
        case: str,
        *,
        units: Literal["mm"],
        rotation_to_model: ArrayLike,
        translation_to_model_m: ArrayLike = (0.0, 0.0, 0.0),
        uncertainty_m: float | None = None,
    ) -> MarkerTrial:
        """Read the source mean or one of 35 independent MRI subject landmarks.

        The source CSVs are in millimetres, explicitly confirmed by the caller.
        ``case='mean'`` is from 30 CT training subjects; ``C01RTF`` through
        ``C35RTF`` are the separate Nolte MRI evaluation subjects. Named points
        are anatomical correspondences, not a direct MuJoCo body-axis mapping.
        """
        if units != "mm":
            raise ValueError("Published Keast landmark CSV coordinates are in mm.")
        if case == "mean":
            relative = _MEAN_PATH
        elif isinstance(case, str) and re.fullmatch(
            r"C(?:0[1-9]|[12][0-9]|3[0-5])RTF", case
        ):
            relative = f"Examples/case3/samples/{case}.csv"
        else:
            raise ValueError("case must be 'mean' or C01RTF through C35RTF.")
        trial = read_named_landmarks(
            self.data_root / relative,
            units=units,
            rotation_to_model=rotation_to_model,
            translation_to_model_m=translation_to_model_m,
            uncertainty_m=uncertainty_m,
            source=f"https://doi.org/10.6084/m9.figshare.20454462.v4; {relative}",
            definition_source="Keast, Bonacci and Fox (2023), doi:10.7717/peerj.14708; Nolte et al. (2016), doi:10.1016/j.jbiomech.2016.09.005",
            definition_license="CC-BY-4.0 source dataset; retain Keast and Nolte attribution",
        )
        evidence = trial.metadata()["landmark_csv"]
        if evidence["sha256"] != _MANIFEST["files"][relative]:
            raise ValueError(
                "Local landmark checksum does not match the pinned source."
            )
        return replace(
            trial,
            _evidence_json=json.dumps(
                {
                    **evidence,
                    "case": case,
                    "dataset": "Keast tibia-fibula SSM v4",
                    "archive_sha256": _MANIFEST["archive_sha256"],
                    "cohort": "30 CT training subjects"
                    if case == "mean"
                    else "independent 35-subject Nolte MRI cohort",
                }
            ),
        )

    def estimate_scales(
        self,
        case: str,
        reference_trial: MarkerTrial,
        *,
        segment: str,
        units: Literal["mm"],
        rotation_to_model: ArrayLike,
        labels: tuple[str, ...] = PALPABLE_LANDMARKS,
        mode: Literal["anisotropic", "isotropic"] = "isotropic",
        max_rms_m: float | None = None,
    ) -> MarkerScaleResult:
        """Fit one real case to reviewed landmarks in the unscaled target frame.

        ``reference_trial`` must contain one frame of corresponding anatomical
        points expressed in the target body's local axes. A source mean-frame
        fit is not automatically transferable to a MuJoCo body. Missing labels
        become NaN rows and are handled explicitly by the shared estimator.
        Source annotation uncertainty is not reported and remains unspecified.
        Choose a residual admission threshold for the intended application.
        Isotropic fitting is conservative by default; three independent factors
        require explicit selection and application-specific validation.
        """
        from myosuite.integrations.capture.landmark_scaling import (
            LandmarkCorrespondences,
            fit_landmark_scales,
        )

        if len(reference_trial.positions) != 1:
            raise ValueError("Reference anatomy must contain exactly one static frame.")
        observed = self.read_landmarks(
            case, units=units, rotation_to_model=rotation_to_model
        )
        reference_points = dict(
            zip(reference_trial.labels, reference_trial.in_model_frame()[0])
        )
        observed_points = dict(zip(observed.labels, observed.in_model_frame()[0]))
        missing = np.full(3, np.nan)
        result = fit_landmark_scales(
            [
                LandmarkCorrespondences(
                    subject_id=f"Keast-v4:{case}",
                    segment=segment,
                    labels=labels,
                    reference_positions_m=np.array(
                        [reference_points.get(label, missing) for label in labels]
                    ),
                    observed_positions_m=np.array(
                        [observed_points.get(label, missing) for label in labels]
                    ),
                    reference_source=reference_trial.source,
                    observed_source=observed.source,
                )
            ],
            mode=mode,
            loss="linear",
            max_rms_m=max_rms_m,
        )
        result.evidence["reference_trial"] = reference_trial.metadata()
        result.evidence["observed_trial"] = observed.metadata()
        return result
