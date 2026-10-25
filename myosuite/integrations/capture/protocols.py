# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Explicit, serializable capture correspondences for a chosen target model."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import mujoco
import numpy as np

from myosuite.core.subject_model import (
    GroundAlignment,
    ScaleEstimate,
    SubjectModel,
    SubjectModelConfig,
    build_subject_model,
)
from myosuite.integrations.capture.markers import (
    MarkerScaleResult,
    MarkerTrial,
    SegmentMeasurement,
    estimate_marker_scales,
)
from myosuite.integrations.capture.retargeting import (
    MarkerRetargetResult,
    _validate_root_selection,
    retarget_markers,
)


def _name(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string.")


@dataclass(frozen=True)
class TargetProtocol:
    """Caller-defined point mapping, length measurements and root policy.

    Measurements name actual target bodies and declare reference lengths/local
    axes. No anatomy, cross-section or source-to-target correspondence is inferred.
    ``point_semantics`` requires an exact known trial declaration when explicit;
    ``unspecified`` retains compatibility without asserting anatomical validity.
    Root/ground settings are available only with an explicit free joint.
    """

    name: str
    marker_sites: Mapping[str, str]
    measurements: tuple[SegmentMeasurement, ...]
    free_joint: str | None = None
    sole_geoms: tuple[str, ...] = ()
    root_marker: str | None = None
    ground_height_m: float = 0.0
    joint_names: tuple[str, ...] | None = None
    point_semantics: str = "unspecified"

    def __post_init__(self) -> None:
        _name(self.name, "name")
        _name(self.point_semantics, "point_semantics")
        if not isinstance(self.marker_sites, Mapping) or not self.marker_sites:
            raise ValueError("marker_sites must be a nonempty label-to-site mapping.")
        for label, site in self.marker_sites.items():
            _name(label, "marker label")
            _name(site, "target site")
        if len(set(self.marker_sites.values())) != len(self.marker_sites):
            raise ValueError("marker_sites must map to distinct target sites.")
        object.__setattr__(
            self, "marker_sites", MappingProxyType(dict(self.marker_sites))
        )
        if (
            not isinstance(self.measurements, Sequence)
            or isinstance(self.measurements, (str, bytes))
            or any(not isinstance(row, SegmentMeasurement) for row in self.measurements)
        ):
            raise ValueError("measurements must contain SegmentMeasurement objects.")
        object.__setattr__(self, "measurements", tuple(self.measurements))
        occupied: dict[str, set[str | None]] = {}
        for row in self.measurements:
            used = occupied.setdefault(row.segment, set())
            if row.axis in used or (used and (row.axis is None or None in used)):
                raise ValueError(f"Overlapping scale measurements for {row.segment!r}.")
            used.add(row.axis)
        for key in ("free_joint", "root_marker"):
            if getattr(self, key) is not None:
                _name(getattr(self, key), key)
        for key in ("sole_geoms", "joint_names"):
            values = getattr(self, key)
            if values is None and key == "joint_names":
                continue
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValueError(f"{key} must be a sequence of distinct names.")
            for value in values:
                _name(value, key)
            if len(set(values)) != len(values) or (key == "joint_names" and not values):
                raise ValueError(
                    f"{key} must contain distinct names and select at least one joint."
                )
            object.__setattr__(self, key, tuple(values))
        if not np.isfinite(self.ground_height_m):
            raise ValueError("ground_height_m must be finite.")
        if self.free_joint is None and (
            self.sole_geoms or self.root_marker is not None or self.ground_height_m != 0
        ):
            raise ValueError("Root/ground settings require an explicit free_joint.")

    def _validate_trial(self, trial: MarkerTrial) -> None:
        required = set(self.marker_sites)
        for measurement in self.measurements:
            required.update(measurement.marker_pair)
        if self.root_marker is not None:
            required.add(self.root_marker)
        missing = required - set(trial.labels)
        if missing:
            raise ValueError(
                f"Protocol {self.name!r} is missing trial labels: {sorted(missing)}"
            )
        actual = getattr(trial, "point_semantics", "unspecified")
        if self.point_semantics != "unspecified" and actual != self.point_semantics:
            raise ValueError(
                f"Protocol point_semantics {self.point_semantics!r} requires a matching trial declaration; got {actual!r}."
            )

    def validate_against_model(self, model: mujoco.MjModel, trial: MarkerTrial) -> None:
        """Reject absent labels/model names, semantic mismatches and invalid roots.

        NaN samples remain missing observations. This validates declared point
        kinds and names, not whether a caller's anatomical mapping is accurate.
        """
        self._validate_trial(trial)
        for name in self.marker_sites.values():
            try:
                model.site(name)
            except KeyError as exc:
                raise ValueError(f"Protocol target site {name!r} is absent.") from exc
        for row in self.measurements:
            try:
                body = model.body(row.segment)
            except KeyError as exc:
                raise ValueError(
                    f"Measurement target body {row.segment!r} is absent."
                ) from exc
            if body.id == 0:
                raise ValueError("The world body cannot receive a segment scale.")
        for name in self.joint_names or ():
            try:
                model.joint(name)
            except KeyError as exc:
                raise ValueError(f"Selected target joint {name!r} is absent.") from exc
        try:
            _validate_root_selection(
                model, self.free_joint, self.sole_geoms, self.root_marker
            )
        except KeyError as exc:
            raise ValueError(
                "A protocol free joint or sole geometry is absent from the target."
            ) from exc

    def estimate_scales(
        self, trial: MarkerTrial, calibration_frames: Sequence[int]
    ) -> MarkerScaleResult:
        """Estimate only supplied lengths from explicit calibration frame indices."""
        self._validate_trial(trial)
        indices = tuple(calibration_frames)
        if (
            not indices
            or any(
                isinstance(i, bool) or not isinstance(i, (int, np.integer))
                for i in indices
            )
            or len(set(indices)) != len(indices)
            or min(indices) < 0
            or max(indices) >= len(trial.positions)
        ):
            raise ValueError(
                "Calibration frames must be unique in-range integer indices."
            )
        if not self.measurements:
            raise ValueError("Scale estimation requires explicit measurements.")
        indices = tuple(int(index) for index in indices)
        result = estimate_marker_scales(
            replace(trial, positions=np.asarray(trial.positions)[list(indices)]),
            self.measurements,
        )
        return MarkerScaleResult(
            result.estimate,
            {
                **result.evidence,
                "target_protocol": self.to_dict(),
                "calibration_frames": list(indices),
            },
        )

    def build_subject(
        self, spec: mujoco.MjSpec, estimate: ScaleEstimate | MarkerScaleResult
    ) -> SubjectModel:
        """Construct the subject with the shared scaling/optional alignment recipe."""
        estimate = (
            estimate.estimate if isinstance(estimate, MarkerScaleResult) else estimate
        )
        ground = None
        if self.sole_geoms:
            ground = GroundAlignment(
                self.free_joint, self.sole_geoms, self.ground_height_m
            )
        return build_subject_model(spec, SubjectModelConfig(estimate, ground=ground))

    def retarget(
        self, subject: SubjectModel, trial: MarkerTrial, **solver_kwargs: Any
    ) -> MarkerRetargetResult:
        """Preflight then fit motion against the constructed target model."""
        self.validate_against_model(subject.model, trial)
        result = retarget_markers(
            subject,
            trial,
            self.marker_sites,
            free_joint=self.free_joint,
            sole_geoms=self.sole_geoms,
            root_marker=self.root_marker,
            ground_height_m=self.ground_height_m,
            joint_names=self.joint_names,
            **solver_kwargs,
        )
        return replace(
            result, provenance={**result.provenance, "target_protocol": self.to_dict()}
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned JSON-compatible protocol without source coordinates."""
        return {
            "schema": 1,
            "name": self.name,
            "marker_sites": dict(self.marker_sites),
            "measurements": [
                {
                    "segment": row.segment,
                    "marker_pair": list(row.marker_pair),
                    "reference_length_m": float(row.reference_length_m),
                    "axis": row.axis,
                    "expected_direction_model": None
                    if row.expected_direction_model is None
                    else list(row.expected_direction_model),
                    "max_direction_error_deg": float(row.max_direction_error_deg),
                }
                for row in self.measurements
            ],
            "free_joint": self.free_joint,
            "sole_geoms": list(self.sole_geoms),
            "root_marker": self.root_marker,
            "ground_height_m": float(self.ground_height_m),
            "joint_names": None if self.joint_names is None else list(self.joint_names),
            "point_semantics": self.point_semantics,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TargetProtocol:
        """Read schema 1, rejecting unknown fields and incomplete measurements."""
        if (
            not isinstance(value, Mapping)
            or not isinstance(value.get("schema"), int)
            or isinstance(value.get("schema"), bool)
            or value["schema"] != 1
        ):
            raise ValueError("Target protocol requires integer schema version 1.")
        unknown = set(value) - {field.name for field in fields(cls)} - {"schema"}
        if unknown:
            raise ValueError(f"Unknown protocol fields: {sorted(unknown)}")
        kwargs = {key: item for key, item in value.items() if key != "schema"}
        try:
            kwargs["measurements"] = tuple(
                SegmentMeasurement(**row) for row in kwargs["measurements"]
            )
            return cls(**kwargs)
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Invalid or incomplete target protocol fields/measurements."
            ) from exc

    def save(self, path: str | Path) -> Path:
        """Write the protocol as strict JSON, creating its parent directory."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> TargetProtocol:
        """Load and validate a schema-versioned JSON protocol."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
